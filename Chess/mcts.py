import math
import numpy as np
import torch

import BoardEncoder
from NeuralNet import ChessNet, get_policy_priors

C_PUCT = 1.5          # exploration constant
DIRICHLET_ALPHA = 0.3 # noise at root to encourage exploration
DIRICHLET_EPS   = 0.25


class MCTSNode:
    __slots__ = ("move", "parent", "children", "N", "W", "Q", "P",
                 "is_expanded", "game_state_snapshot")

    def __init__(self, move=None, parent=None, prior: float = 0.0):
        self.move   = move          # Move that led to this node (None for root)
        self.parent = parent

        self.children: dict = {}    # move → MCTSNode

        self.N = 0                  # visit count
        self.W = 0.0                # total value (from current player's POV)
        self.Q = 0.0                # mean value
        self.P = prior              # prior from policy head

        self.is_expanded = False

    def ucb_score(self, parent_N: int) -> float:
        u = C_PUCT * self.P * math.sqrt(parent_N) / (1 + self.N)
        return self.Q + u

    def best_child(self) -> "MCTSNode":
        parent_N = self.N
        return max(self.children.values(), key=lambda c: c.ucb_score(parent_N))

    def expand(self, priors: dict):
        """Create child nodes for all legal moves with their prior probs."""
        for move, prob in priors.items():
            self.children[move] = MCTSNode(move=move, parent=self, prior=prob)
        self.is_expanded = True

    def backup(self, value: float):
        """Propagate value up the tree, flipping sign at each level."""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q  = node.W / node.N
            value   = -value          # flip: opponent's gain is our loss
            node    = node.parent


class MCTS:
    def __init__(self, model: ChessNet, device: torch.device,
                 num_simulations: int = 200):
        self.model          = model
        self.device         = device
        self.num_simulations = num_simulations

        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────

    def get_move_probs(self, gs, temperature: float = 1.0) -> tuple:
        """
        Run MCTS from the current game state.

        Returns
        -------
        move_probs : dict  {move: probability}   π_mcts (training target)
        best_move  : Move  sampled from π_mcts
        """
        root = MCTSNode()
        self._expand(root, gs)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node  = root
            state = self._copy_state(gs)

            # Selection — traverse to a leaf
            while node.is_expanded and node.children:
                node = node.best_child()
                state.make_move(node.move)

            # Check terminal
            valid = state.get_valid_moves()
            status = state.get_status(valid)

            if status == "checkmate":
                # The side that just moved delivered mate → value = +1 for mover
                value = 1.0
            elif status == "stalemate":
                value = 0.0
            else:
                # Expansion + evaluation
                self._expand(node, state)
                value = self._evaluate(state)

            # Backup
            node.backup(-value)   # negate because expand was from opponent's view

        return self._build_policy(root, temperature)

    # ── Private helpers ───────────────────────────────────────────────────

    def _evaluate(self, gs) -> float:
        """Run board through the value head. Returns float in [-1, 1]."""
        board_tensor = torch.tensor(BoardEncoder.encode(gs), dtype=torch.float32) \
                           .unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(board_tensor)
        return value.item()

    def _get_priors(self, gs, legal_moves: list) -> dict:
        """Run board through the policy head, mask & normalise."""
        board_tensor = torch.tensor(BoardEncoder.encode(gs), dtype=torch.float32) \
                           .unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_policy, _ = self.model(board_tensor)
        return get_policy_priors(log_policy[0], legal_moves)

    def _expand(self, node: MCTSNode, gs):
        legal = gs.get_valid_moves()
        if not legal:
            return
        priors = self._get_priors(gs, legal)
        node.expand(priors)

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        if not root.children:
            return
        moves   = list(root.children.keys())
        noise   = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for move, n in zip(moves, noise):
            child   = root.children[move]
            child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n

    def _build_policy(self, root: MCTSNode, temperature: float) -> tuple:
        moves  = list(root.children.keys())
        visits = np.array([root.children[m].N for m in moves], dtype=np.float64)

        if temperature < 1e-3:              # greedy
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Normalize visits BEFORE exponentiation to prevent overflow.
            # visits / max is in [0, 1], so (v/max)^(1/T) never overflows.
            visits = visits / (visits.max() + 1e-8)
            visits_t = visits ** (1.0 / temperature)
            total = visits_t.sum()
            if total < 1e-8:                # all zeros — fall back to uniform
                probs = np.ones(len(moves), dtype=np.float64) / len(moves)
            else:
                probs = visits_t / total

        # Final safety: clamp any residual floating-point dirt, renormalize
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        move_probs = {m: float(p) for m, p in zip(moves, probs)}

        # Sample move
        best_move = np.random.choice(moves, p=probs)
        return move_probs, best_move

    @staticmethod
    def _copy_state(gs):
        """Deep-copy a GameState. deepcopy is faster than replaying all moves."""
        import copy
        return copy.deepcopy(gs)