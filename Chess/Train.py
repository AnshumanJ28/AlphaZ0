from __future__ import annotations

import os
import copy
import time
import random
import logging
import argparse
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import chesseng as ce
import BoardEncoder
from NeuralNet import ChessNet, POLICY_SIZE, move_to_index
from mcts import MCTS

# ─────────────────────────────── Config ───────────────────────────────────

@dataclass
class TrainConfig:
    # ── model ──────────────────────────────────────────────────────────
    in_channels:       int   = 18
    num_blocks:        int   = 10
    channels:          int   = 128

    # ── self-play ──────────────────────────────────────────────────────
    num_workers:       int   = 2
    games_per_iter:    int   = 2
    mcts_sims:         int   = 50
    max_game_moves:    int   = 80
    temperature_moves: int   = 30

    # ── replay buffer ──────────────────────────────────────────────────
    buffer_size:       int   = 10_000
    min_buffer_before_train: int = 200

    # ── training ───────────────────────────────────────────────────────
    batch_size:        int   = 128
    lr:                float = 1e-3
    weight_decay:      float = 1e-4
    num_epochs:        int   = 3
    grad_clip:         float = 1.0

    # ── A3C ────────────────────────────────────────────────────────────
    a3c_entropy_coef:  float = 0.01
    a3c_value_coef:    float = 0.5
    a3c_gamma:         float = 0.99
    a3c_gae_lambda:    float = 0.95
    a3c_rollout_len:   int   = 10

    # ── loss weights ───────────────────────────────────────────────────
    policy_loss_w:     float = 1.0
    value_loss_w:      float = 1.0
    a3c_actor_w:       float = 0.5
    a3c_critic_w:      float = 0.5

    # ── checkpointing ──────────────────────────────────────────────────
    checkpoint_dir:    str   = "checkpoints"
    save_every:        int   = 10
    log_every:         int   = 1

    # ── misc ───────────────────────────────────────────────────────────
    seed:              int   = 42
    device:            str   = "auto"
    num_iterations:    int   = 200

# ─────────────────────────── Replay Buffer ────────────────────────────────

@dataclass
class Experience:
    """One (state, pi, z) sample from self-play."""
    board_tensor: np.ndarray           # (18, 8, 8)
    policy_target: np.ndarray          # (4096,) MCTS visit distribution
    value_target: float                # game outcome from the perspective of current player


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self._buf: deque[Experience] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, exps: List[Experience]):
        with self._lock:
            self._buf.extend(exps)

    def sample(self, batch_size: int) -> List[Experience]:
        with self._lock:
            return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ─────────────────────────────── Self-Play ────────────────────────────────

def play_one_game(
    model: ChessNet,
    device: torch.device,
    cfg: TrainConfig,
) -> List[Experience]:
    """
    Play a complete game via MCTS and collect (state, π, z) tuples.
    Returns a list of Experience objects with outcomes filled in.
    """
    model.eval()   # disable BN updates during inference — faster + safer
    mcts = MCTS(model, device, num_simulations=cfg.mcts_sims)
    gs   = ce.GameState()

    # (board_tensor, policy_target, current_player_sign)
    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

    move_count = 0
    while move_count < cfg.max_game_moves:
        temperature = 1.0 if move_count < cfg.temperature_moves else 1e-3
        move_probs, best_move = mcts.get_move_probs(gs, temperature=temperature)

        if best_move is None:
            break  # terminal position

        # Build full policy vector
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        for m, p in move_probs.items():
            pi[move_to_index(m)] = p

        board_enc = BoardEncoder.encode(gs).copy()
        player_sign = 1 if gs.white_to_move else -1
        trajectory.append((board_enc, pi, player_sign))

        gs.make_move(best_move)
        move_count += 1

        valid_moves = gs.get_valid_moves()
        status = gs.get_status(valid_moves)

        if status == "checkmate":
            # The player who just moved won → previous player (who now must move) lost
            winner_sign = -player_sign   # who just moved
            break
        elif status == "stalemate":
            winner_sign = 0
            break
    else:
        winner_sign = 0  # draw by move limit

    # Assign outcomes relative to each position's current player
    experiences = []
    for board_enc, pi, player_sign in trajectory:
        z = float(winner_sign * player_sign)
        experiences.append(Experience(board_enc, pi, z))

    return experiences


# ─────────────────────────── A3C Worker ───────────────────────────────────

class A3CWorker(threading.Thread):
    """
    Each worker:
      1. Clones global model weights locally.
      2. Runs a short rollout (a3c_rollout_len steps) of self-play.
      3. Computes actor-critic gradients with GAE advantage.
      4. Pushes gradients to the shared optimizer.
      5. Pushes collected (s, π, z) samples to the replay buffer.
    """

    def __init__(
        self,
        worker_id: int,
        global_model: ChessNet,
        global_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        cfg: TrainConfig,
        device: torch.device,
        stop_event: threading.Event,
        stats: dict,
        stats_lock: threading.Lock,
        model_lock: threading.Lock,
    ):
        super().__init__(daemon=True)
        self.worker_id        = worker_id
        self.global_model     = global_model
        self.global_optimizer = global_optimizer
        self.replay_buffer    = replay_buffer
        self.cfg              = cfg
        self.device           = device
        self.stop_event       = stop_event
        self.stats            = stats
        self.stats_lock       = stats_lock
        self.model_lock       = model_lock

        # Local model runs on CPU — CUDA contexts are not thread-safe.
        # Workers compute gradients on CPU, then push them to the GPU global model.
        self.cpu_device = torch.device("cpu")
        self.local_model = ChessNet(cfg.in_channels, cfg.num_blocks, cfg.channels).to(self.cpu_device)

    def run(self):
        cfg = self.cfg
        while not self.stop_event.is_set():
            # ── sync local weights from global (map to CPU) ─────────
            # Hold model_lock so we don't read BN running stats while
            # the main thread's backward() is updating them inplace.
            with self.model_lock:
                cpu_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
            self.local_model.load_state_dict(cpu_state)
            self.local_model.train()

            # ── collect a rollout ────────────────────────────────────
            rollout_states, rollout_log_probs, rollout_values, rollout_rewards = \
                [], [], [], []

            gs = ce.GameState()
            mcts = MCTS(self.local_model, self.cpu_device,
                        num_simulations=max(10, cfg.mcts_sims // 4))  # lighter for A3C rollout

            game_exps: List[Experience] = []
            move_count = 0
            done = False

            for _ in range(cfg.a3c_rollout_len):
                if done:
                    break

                board_t = torch.tensor(
                    BoardEncoder.encode(gs), dtype=torch.float32
                ).unsqueeze(0).to(self.cpu_device)

                log_policy, value = self.local_model(board_t)
                value_scalar = value.item()

                # Get move via lightweight MCTS
                temperature = 1.0 if move_count < cfg.temperature_moves else 1e-3
                move_probs, best_move = mcts.get_move_probs(gs, temperature=temperature)

                if best_move is None:
                    done = True
                    rollout_rewards.append(0.0)
                    break

                # Log-prob of chosen move under current policy
                idx = move_to_index(best_move)
                log_prob = log_policy[0, idx]

                # Build full pi vector for replay
                pi = np.zeros(POLICY_SIZE, dtype=np.float32)
                for m, p in move_probs.items():
                    pi[move_to_index(m)] = p

                game_exps.append(Experience(
                    BoardEncoder.encode(gs).copy(), pi, 0.0  # z filled after
                ))

                rollout_states.append(board_t)
                rollout_log_probs.append(log_prob)
                rollout_values.append(value_scalar)

                gs.make_move(best_move)
                move_count += 1

                valid_moves = gs.get_valid_moves()
                status = gs.get_status(valid_moves)

                if status == "checkmate":
                    rollout_rewards.append(-1.0)  # current player lost (was just mated)
                    done = True
                elif status == "stalemate":
                    rollout_rewards.append(0.0)
                    done = True
                else:
                    rollout_rewards.append(0.0)  # intermediate — 0 until game ends

            if not rollout_states:
                continue

            # ── bootstrap value ──────────────────────────────────────
            if done:
                R = rollout_rewards[-1]
            else:
                with torch.no_grad():
                    board_t = torch.tensor(
                        BoardEncoder.encode(gs), dtype=torch.float32
                    ).unsqueeze(0).to(self.cpu_device)
                    _, boot_v = self.local_model(board_t)
                    R = boot_v.item()

            # ── compute GAE returns & advantages ─────────────────────
            returns, advantages = [], []
            gae = 0.0
            values_ext = rollout_values + [R]

            for t in reversed(range(len(rollout_rewards))):
                delta = rollout_rewards[t] + cfg.a3c_gamma * values_ext[t + 1] - values_ext[t]
                gae   = delta + cfg.a3c_gamma * cfg.a3c_gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values_ext[t])

            advantages_t = torch.tensor(advantages, dtype=torch.float32)  # stays on CPU
            returns_t    = torch.tensor(returns,    dtype=torch.float32)  # stays on CPU

            # Normalize advantages
            if len(advantages) > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # ── A3C losses ───────────────────────────────────────────
            log_probs_t = torch.stack(rollout_log_probs)

            # Re-run states through local model for entropy calculation
            boards_batch = torch.cat(rollout_states, dim=0)
            log_policies_batch, values_batch = self.local_model(boards_batch)

            # Actor loss: -log_prob * advantage (policy gradient)
            actor_loss = -(log_probs_t * advantages_t.detach()).mean()

            # Critic loss: MSE(V(s), R)
            critic_loss = F.mse_loss(values_batch, returns_t)

            # Entropy bonus: encourages exploration
            probs = torch.exp(log_policies_batch)
            entropy = -(probs * log_policies_batch).sum(dim=1).mean()

            a3c_loss = (
                cfg.a3c_actor_w  * actor_loss
              + cfg.a3c_critic_w * cfg.a3c_value_coef * critic_loss
              - cfg.a3c_entropy_coef * entropy
            )

            # ── push gradients to global model ───────────────────────
            # Hold model_lock so backward/optimizer step doesn't race
            # with the main thread's train_on_batch backward pass.
            self.global_optimizer.zero_grad(set_to_none=True)
            a3c_loss.backward()

            # Clip gradients on local model (only params that have grads)
            nn.utils.clip_grad_norm_(
                [p for p in self.local_model.parameters() if p.grad is not None],
                cfg.grad_clip,
            )

            # Copy local grads → global params (move to CPU if needed)
            for local_p, global_p in zip(self.local_model.parameters(),
                                         self.global_model.parameters()):
                if local_p.grad is None:
                    continue
                grad = local_p.grad.to(global_p.device)
                if global_p.grad is None:
                    global_p.grad = grad.clone()
                else:
                    global_p.grad.add_(grad)

            with self.model_lock:
                self.global_optimizer.step()

            # ── push experience to replay buffer ─────────────────────
            # Fill in outcome z for collected game experiences
            z_val = rollout_rewards[-1] if done else 0.0
            for exp in game_exps:
                exp.value_target = z_val
            self.replay_buffer.push(game_exps)

            # ── log stats ────────────────────────────────────────────
            with self.stats_lock:
                self.stats["a3c_actor_loss"]  = self.stats.get("a3c_actor_loss",  [])
                self.stats["a3c_critic_loss"] = self.stats.get("a3c_critic_loss", [])
                self.stats["a3c_entropy"]     = self.stats.get("a3c_entropy",     [])
                self.stats["a3c_actor_loss"].append(actor_loss.item())
                self.stats["a3c_critic_loss"].append(critic_loss.item())
                self.stats["a3c_entropy"].append(entropy.item())


# ────────────────────────── Supervised Train Step ─────────────────────────

def train_on_batch(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    batch: List[Experience],
    cfg: TrainConfig,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> dict:
    """
    AlphaZero-style supervised update on (s, π, z) triples collected from self-play.
    Uses automatic mixed precision (AMP) when a CUDA scaler is provided.
    Returns dict of loss scalars for logging.
    """
    model.train()
    use_amp = (scaler is not None) and device.type == "cuda"

    # Pin tensors to GPU in one shot — non_blocking=True overlaps H2D transfer
    boards = torch.tensor(
        np.stack([e.board_tensor  for e in batch]), dtype=torch.float32
    ).to(device, non_blocking=True)
    pi_tgt = torch.tensor(
        np.stack([e.policy_target for e in batch]), dtype=torch.float32
    ).to(device, non_blocking=True)
    z_tgt  = torch.tensor(
        [e.value_target for e in batch], dtype=torch.float32
    ).to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        log_policy, value = model(boards)

        # Policy loss: cross-entropy = -sum(pi * log_pi_net)
        policy_loss = -(pi_tgt * log_policy).sum(dim=1).mean()

        # Value loss: MSE between predicted value and game outcome z
        value_loss = F.mse_loss(value, z_tgt)

        # Entropy regularization
        probs   = torch.exp(log_policy)
        entropy = -(probs * log_policy).sum(dim=1).mean()

        total_loss = (
            cfg.policy_loss_w * policy_loss
          + cfg.value_loss_w  * value_loss
          - cfg.a3c_entropy_coef * entropy
        )

    if use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     entropy.item(),
        "total_loss":  total_loss.item(),
    }


# ────────────────────────────── Checkpointing ─────────────────────────────

def save_checkpoint(model: ChessNet, optimizer, scheduler, iteration: int, cfg: TrainConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"chess_net_iter_{iteration:04d}.pt")
    torch.save({
        "iteration":        iteration,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "config":           cfg.__dict__,
    }, path)
    logging.info(f"  ✓ Checkpoint saved → {path}")
    return path


def load_checkpoint(path: str, model: ChessNet, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("iteration", 0)


# ──────────────────────────── Main Training Loop ──────────────────────────

def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # fastest conv kernels
            return dev
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # Fall back to CPU with a loud warning — GPU strongly recommended
        import warnings
        warnings.warn(
            "No CUDA or MPS GPU found — training on CPU will be very slow. "
            "Pass --device cpu explicitly to silence this warning.",
            RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")
    dev = torch.device(cfg.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return dev


def train(cfg: TrainConfig, resume_path: Optional[str] = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = get_device(cfg)
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU : {torch.cuda.get_device_name(device)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory // 1024**2} MB")
        torch.cuda.empty_cache()

    # Mixed-precision scaler (no-op on CPU / MPS)
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── global model & optimizer ─────────────────────────────────────
    model = ChessNet(cfg.in_channels, cfg.num_blocks, cfg.channels).to(device)
    model.share_memory()   # needed for multiprocessing gradient sharing

    # Lock that workers must hold when reading global model weights.
    # Prevents race between state_dict() and backward() touching BN stats.
    model_lock = threading.Lock()

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_iterations, eta_min=cfg.lr * 0.01)

    start_iter = 0
    if resume_path:
        start_iter = load_checkpoint(resume_path, model, optimizer, scheduler)
        log.info(f"Resumed from iteration {start_iter}  ({resume_path})")

    replay_buffer = ReplayBuffer(cfg.buffer_size)
    stop_event    = threading.Event()
    stats: dict   = {}
    stats_lock    = threading.Lock()

    # ── spawn A3C worker threads ─────────────────────────────────────
    workers = [
        A3CWorker(
            worker_id=i,
            global_model=model,
            global_optimizer=optimizer,
            replay_buffer=replay_buffer,
            cfg=cfg,
            device=device,
            stop_event=stop_event,
            stats=stats,
            stats_lock=stats_lock,
            model_lock=model_lock,
        )
        for i in range(cfg.num_workers)
    ]
    for w in workers:
        w.start()
    log.info(f"Started {cfg.num_workers} A3C worker threads")

    # ── main training iterations ─────────────────────────────────────
    best_loss = float("inf")

    for iteration in range(start_iter + 1, cfg.num_iterations + 1):
        iter_start = time.time()
        log.info(f"\n{'═'*60}")
        log.info(f"  Iteration {iteration}/{cfg.num_iterations}")
        log.info(f"{'═'*60}")

        # ── self-play: collect full games (main thread) ──────────────
        log.info(f"  Self-play: collecting {cfg.games_per_iter} games …")
        model.eval()
        for g in range(cfg.games_per_iter):
            game_exps = play_one_game(model, device, cfg)
            replay_buffer.push(game_exps)
            if (g + 1) % 5 == 0:
                log.info(f"    Game {g+1}/{cfg.games_per_iter}  "
                         f"buf={len(replay_buffer)}  "
                         f"moves={len(game_exps)}")

        # ── wait until buffer is populated enough ────────────────────
        if len(replay_buffer) < cfg.min_buffer_before_train:
            log.info(f"  Buffer too small ({len(replay_buffer)} < "
                     f"{cfg.min_buffer_before_train}), skipping train step")
            continue

        # ── supervised training on replay samples ────────────────────
        log.info(f"  Training: {cfg.num_epochs} epochs, batch={cfg.batch_size}")
        epoch_losses = []

        for epoch in range(cfg.num_epochs):
            batch = replay_buffer.sample(cfg.batch_size)
            with model_lock:
                model.train()
                losses = train_on_batch(model, optimizer, batch, cfg, device, scaler)
                model.eval()
            epoch_losses.append(losses)

        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses])
            for k in epoch_losses[0]
        }

        # ── read A3C stats from workers ───────────────────────────────
        with stats_lock:
            a3c_actor  = np.mean(stats.get("a3c_actor_loss",  [0]))
            a3c_critic = np.mean(stats.get("a3c_critic_loss", [0]))
            a3c_ent    = np.mean(stats.get("a3c_entropy",     [0]))
            stats.clear()

        scheduler.step()
        elapsed = time.time() - iter_start
        if device.type == "cuda":
            vram_mb = torch.cuda.memory_reserved(device) // 1024 ** 2
            log.info(f"  GPU VRAM reserved: {vram_mb} MB")

        # ── logging ──────────────────────────────────────────────────
        if iteration % cfg.log_every == 0:
            log.info(
                f"  ┌─ Supervised Losses ─────────────────────────────\n"
                f"  │  policy_loss = {avg_losses['policy_loss']:.4f}\n"
                f"  │  value_loss  = {avg_losses['value_loss']:.4f}\n"
                f"  │  entropy     = {avg_losses['entropy']:.4f}\n"
                f"  │  total_loss  = {avg_losses['total_loss']:.4f}\n"
                f"  ├─ A3C Worker Losses ─────────────────────────────\n"
                f"  │  actor_loss  = {a3c_actor:.4f}\n"
                f"  │  critic_loss = {a3c_critic:.4f}\n"
                f"  │  entropy     = {a3c_ent:.4f}\n"
                f"  ├─ Misc ──────────────────────────────────────────\n"
                f"  │  buffer_size = {len(replay_buffer)}\n"
                f"  │  lr          = {scheduler.get_last_lr()[0]:.6f}\n"
                f"  │  elapsed     = {elapsed:.1f}s\n"
                f"  └─────────────────────────────────────────────────"
            )

        # ── checkpoint ───────────────────────────────────────────────
        if iteration % cfg.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, iteration, cfg)
            if avg_losses["total_loss"] < best_loss:
                best_loss = avg_losses["total_loss"]
                best_path = os.path.join(cfg.checkpoint_dir, "chess_net_best.pt")
                torch.save(model.state_dict(), best_path)
                log.info(f"  ★ New best model saved → {best_path}")

    # ── cleanup ──────────────────────────────────────────────────────
    log.info("\nTraining complete. Stopping workers …")
    stop_event.set()
    for w in workers:
        w.join(timeout=5)

    final_path = os.path.join(cfg.checkpoint_dir, "chess_net_final.pt")
    torch.save(model.state_dict(), final_path)
    log.info(f"Final model → {final_path}")
    return model


# ─────────────────────────── Quick Evaluation ─────────────────────────────

def evaluate(model_path: str, cfg: TrainConfig, num_games: int = 10):
    """Play num_games with the saved model and report win/draw/loss rates."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    device = get_device(cfg)
    model  = ChessNet(cfg.in_channels, cfg.num_blocks, cfg.channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {"white_win": 0, "black_win": 0, "draw": 0}

    for i in range(num_games):
        mcts = MCTS(model, device, num_simulations=cfg.mcts_sims)
        gs   = ce.GameState()
        move_count = 0

        while move_count < cfg.max_game_moves:
            temperature = 1e-3   # greedy during evaluation
            _, best_move = mcts.get_move_probs(gs, temperature=temperature)
            if best_move is None:
                break
            gs.make_move(best_move)
            move_count += 1
            valid_moves = gs.get_valid_moves()
            status = gs.get_status(valid_moves)
            if status == "checkmate":
                winner = "black_win" if gs.white_to_move else "white_win"
                results[winner] += 1
                break
            elif status == "stalemate":
                results["draw"] += 1
                break
        else:
            results["draw"] += 1

        log.info(f"  Game {i+1}/{num_games}: {results}")

    log.info(f"\nFinal: {results} over {num_games} self-play games")
    return results


# ──────────────────────────────── CLI ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chess AlphaZero + A3C Trainer")
    p.add_argument("--mode",       choices=["train", "eval"], default="train")
    p.add_argument("--resume",     type=str, default=None,  help="Checkpoint path to resume")
    p.add_argument("--eval-model", type=str, default=None,  help="Model path for eval mode")
    p.add_argument("--iters",      type=int, default=200)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--sims",       type=int, default=50,    help="MCTS simulations per move")
    p.add_argument("--games",      type=int, default=2,     help="Self-play games per iter")
    p.add_argument("--batch",      type=int, default=256)
    p.add_argument("--epochs",     type=int, default=3,     help="Train epochs per iter")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--blocks",     type=int, default=10,    help="Residual blocks in trunk")
    p.add_argument("--channels",   type=int, default=128)
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--ckpt-dir",   type=str, default="checkpoints")
    p.add_argument("--eval-games", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        num_iterations  = args.iters,
        num_workers     = args.workers,
        mcts_sims       = args.sims,
        games_per_iter  = args.games,
        batch_size      = args.batch,
        num_epochs      = args.epochs,
        lr              = args.lr,
        num_blocks      = args.blocks,
        channels        = args.channels,
        device          = args.device,
        checkpoint_dir  = args.ckpt_dir,
    )

    if args.mode == "train":
        train(cfg, resume_path=args.resume)
    else:
        model_path = args.eval_model or os.path.join(cfg.checkpoint_dir, "chess_net_best.pt")
        evaluate(model_path, cfg, num_games=args.eval_games)
