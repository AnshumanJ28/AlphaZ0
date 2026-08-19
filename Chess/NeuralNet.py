import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_SIZE = 4096   # 64 * 64 from→to pairs


def move_to_index(move) -> int:
    return move.start_row * 512 + move.start_col * 64 + move.end_row * 8 + move.end_col


def index_to_coords(idx: int):
    sr = idx // 512
    sc = (idx % 512) // 64
    er = (idx % 64) // 8
    ec = idx % 8
    return sr, sc, er, ec


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, in_channels: int = 18, num_blocks: int = 10,
                 channels: int = 128):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Trunk
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, 18, 8, 8)
        Returns
        -------
        log_policy : (batch, POLICY_SIZE)   log-softmax probabilities
        value      : (batch,)               tanh scalar in [-1, 1]
        """
        x = self.trunk(self.stem(x))

        # Policy — .float() casts back to FP32 before the FC layer.
        # Conv layers run in FP16 under AMP (fast), but cuBLAS matmul in
        # Linear layers raises CUBLAS_STATUS_NOT_SUPPORTED on FP16 inputs
        # for certain matrix dimensions on some GPUs.
        p = self.policy_conv(x).flatten(1).float()
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value — same FP16→FP32 cast before FC layers
        v = self.value_conv(x).flatten(1).float()
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(1)

        return log_policy, value


def get_policy_priors(
    log_policy: torch.Tensor,
    legal_moves: list,
) -> dict:
    """
    log_policy  : (POLICY_SIZE,)  raw log-softmax output for one position
    legal_moves : list of Move objects
    Returns     : dict {move: prior_probability}
    """
    policy  = torch.exp(log_policy).float()   # ensure FP32
    indices = [move_to_index(m) for m in legal_moves]
    masked  = policy[indices]

    total = masked.sum()
    if total < 1e-8:                 # all priors collapsed — uniform fallback
        masked = torch.ones_like(masked)
        total  = masked.sum()

    priors = (masked / total).tolist()
    return {m: p for m, p in zip(legal_moves, priors)}


# ─── C++ MCTS Evaluation Callback ────────────────────────────────────────────

import numpy as np
import threading

# PyTorch CUDA operations can cause silent segfaults on Windows if called
# concurrently from multiple threads. We serialize the GPU forward pass globally.
_GLOBAL_EVAL_LOCK = threading.Lock()

def make_eval_fn(model: ChessNet, device: torch.device):
    """
    Create an evaluation callback for the C++ MCTS engine.
    """
    import numpy as np

    def eval_fn(board_np):
        """
        board_np : numpy array of shape (18, 8, 8), float32
        Returns  : (log_policy, value) where log_policy is (4096,) numpy array
        """
        # Safe copy to prevent C++ read-only pointer segfaults
        board_tensor = torch.tensor(
            np.array(board_np), dtype=torch.float32
        ).unsqueeze(0).to(device)

        with _GLOBAL_EVAL_LOCK:
            with torch.no_grad():
                log_policy, value = model(board_tensor)

        log_policy_np = log_policy[0].cpu().numpy().astype(np.float32)
        value_float = value.item()
        return (log_policy_np, value_float)

    return eval_fn