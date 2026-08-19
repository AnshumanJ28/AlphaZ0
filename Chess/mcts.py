"""
mcts.py — Monte Carlo Tree Search.

Uses the C++ MCTS implementation when available (much faster),
falls back to pure Python otherwise.
"""

import math
import numpy as np
import torch

import BoardEncoder
from NeuralNet import ChessNet, get_policy_priors, make_eval_fn

C_PUCT = 1.5          # exploration constant
DIRICHLET_ALPHA = 0.3 # noise at root to encourage exploration
DIRICHLET_EPS   = 0.25

# ── Try to load C++ MCTS ─────────────────────────────────────────────────────
_USE_CPP = False
try:
    import alphaz0_cpp as _cpp
    _USE_CPP = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  C++ MCTS Wrapper (preferred — much faster)
# ═══════════════════════════════════════════════════════════════════════════════

class CppMCTS:
    """
    Wrapper around the C++ MCTS implementation.
    API-compatible with the pure-Python MCTS class below.
    """

    def __init__(self, model: ChessNet, device: torch.device,
                 num_simulations: int = 200):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.model.eval()

        # Create the evaluation callback for C++
        self._eval_fn = make_eval_fn(model, device)
        self._cpp_mcts = _cpp.MCTS(self._eval_fn, num_simulations)

    def get_move_probs(self, gs, temperature: float = 1.0) -> tuple:
        """
        Run MCTS from the current game state.

        Parameters
        ----------
        gs : GameState (Python or C++)
        temperature : float

        Returns
        -------
        move_probs : dict {move: probability}
        best_move  : Move
        """
        # If gs is a Python GameState, we need to convert to C++ GameState
        if not isinstance(gs, _cpp.GameState):
            cpp_gs = _python_gs_to_cpp(gs)
        else:
            cpp_gs = gs

        index_probs, best_move_cpp = self._cpp_mcts.get_move_probs(
            cpp_gs, temperature
        )

        # Convert C++ Move to Python Move for compatibility
        if not isinstance(gs, _cpp.GameState):
            # Need to map back to Python Move objects
            valid_moves = gs.get_valid_moves()
            best_move = None
            move_probs = {}
            for m in valid_moves:
                from NeuralNet import move_to_index
                idx = move_to_index(m)
                if idx in index_probs:
                    move_probs[m] = index_probs[idx]
                if (m.start_row == best_move_cpp.start_row and
                    m.start_col == best_move_cpp.start_col and
                    m.end_row == best_move_cpp.end_row and
                    m.end_col == best_move_cpp.end_col):
                    best_move = m

            if best_move is None and valid_moves:
                best_move = valid_moves[0]

            return move_probs, best_move
        else:
            # Return C++ Move objects directly
            move_probs = {}
            valid = cpp_gs.get_valid_moves()
            for m in valid:
                idx = m.to_index()
                if idx in index_probs:
                    move_probs[m] = index_probs[idx]
            return move_probs, best_move_cpp


def _python_gs_to_cpp(gs):
    """Convert a Python GameState to a C++ GameState by replaying moves."""
    # The simplest approach: create a fresh C++ GameState and replay all moves
    import chesseng as ce
    cpp_gs = _cpp.GameState()

    # We can't easily replay since we don't have the full move history with
    # all flags. Instead, sync the board state directly.
    # For MCTS purposes, just create a new game and replay the move log.
    for py_move in gs.move_log:
        valid = cpp_gs.get_valid_moves()
        for m in valid:
            if (m.start_row == py_move.start_row and
                m.start_col == py_move.start_col and
                m.end_row == py_move.end_row and
                m.end_col == py_move.end_col):
                cpp_gs.make_move(m)
                break

    return cpp_gs


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure Python MCTS (fallback when C++ is not available)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory function — returns the best available MCTS implementation
# ═══════════════════════════════════════════════════════════════════════════════

def create_mcts(model: ChessNet, device: torch.device,
                num_simulations: int = 200):
    """
    Create the fastest available MCTS instance.
    Returns CppMCTS if the C++ extension is available, else pure-Python MCTS.
    """
    if _USE_CPP:
        return CppMCTS(model, device, num_simulations)
    return MCTS(model, device, num_simulations)