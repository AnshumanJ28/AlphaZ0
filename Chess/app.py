"""
app.py — FastAPI backend for the AlphaZ0 chess engine.

Serves the HTML frontend and provides the move API.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import chesseng as ce
from NeuralNet import ChessNet, make_eval_fn
from mcts import MCTS, create_mcts

# ── Try to load C++ core ─────────────────────────────────────────────────────
_USE_CPP = False
try:
    import alphaz0_cpp as _cpp
    _USE_CPP = True
except ImportError:
    pass

# ─────────────────────────── Model loading ────────────────────────────────

CKPT_PATHS = [
    "checkpoints/chess_net_best.pt",
    "checkpoints/chess_net_final.pt",
    "chess_net_best.pt",
]
MCTS_SIMS = int(os.environ.get("MCTS_SIMS", "80"))

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[ChessNet] = None
_model_lock = threading.Lock()
_mcts_instance = None


def load_model() -> ChessNet:
    loaded_from = None
    state_dict = None

    # Find the first available checkpoint
    for path in CKPT_PATHS:
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=_device)
            loaded_from = path
            break

    if state_dict is None:
        # No checkpoint — use new small model with random weights
        model = ChessNet(in_channels=18, num_blocks=6, channels=64).to(_device)
        model.eval()
        print("[app] No checkpoint found — playing with random weights.")
        return model

    # Auto-detect architecture from checkpoint weights
    # Check the stem conv weight shape: (channels, 18, 3, 3)
    stem_shape = state_dict.get("stem.0.weight", None)
    if stem_shape is not None:
        channels = stem_shape.shape[0]
        num_blocks = sum(1 for k in state_dict if k.startswith("trunk.") and k.endswith(".conv1.weight"))
    else:
        channels, num_blocks = 64, 6  # default to new model

    print(f"[app] Detected architecture: {num_blocks} blocks, {channels} channels")
    model = ChessNet(in_channels=18, num_blocks=num_blocks, channels=channels).to(_device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[app] Loaded checkpoint: {loaded_from}")
    return model


# ─────────────────────────── Square <-> (row, col) ─────────────────────────

FILES = "abcdefgh"


def sq_to_rc(sq: str) -> tuple[int, int]:
    """'e2' -> (row, col) matching chesseng's board indexing
    (row 0 = rank 8 / black back rank, row 7 = rank 1 / white back rank)."""
    col = FILES.index(sq[0])
    row = 8 - int(sq[1])
    return row, col


def rc_to_sq(row: int, col: int) -> str:
    return FILES[col] + str(8 - row)


def uci_to_move_python(uci: str, gs: ce.GameState) -> tuple[ce.Move, str]:
    """Find the matching legal Move object for a UCI string like 'e2e4' or
    'e7e8q'. Returns (move, promotion_letter_or_'Q')."""
    start_sq, end_sq = uci[0:2], uci[2:4]
    promo = uci[4].upper() if len(uci) > 4 else "Q"
    start_rc, end_rc = sq_to_rc(start_sq), sq_to_rc(end_sq)

    candidate = ce.Move(start_rc, end_rc, gs.board)
    valid_moves = gs.get_valid_moves()
    matched = next((m for m in valid_moves if m == candidate), None)
    if matched is None:
        raise ValueError(f"Illegal move in history: {uci}")
    return matched, promo


def uci_to_move_cpp(uci: str, gs) -> tuple:
    """Find the matching legal Move for a UCI string using C++ GameState."""
    start_sq, end_sq = uci[0:2], uci[2:4]
    promo = uci[4].upper() if len(uci) > 4 else "Q"
    start_rc, end_rc = sq_to_rc(start_sq), sq_to_rc(end_sq)

    valid_moves = gs.get_valid_moves()
    for m in valid_moves:
        if (m.start_row == start_rc[0] and m.start_col == start_rc[1] and
            m.end_row == end_rc[0] and m.end_col == end_rc[1]):
            return m, promo
    raise ValueError(f"Illegal move in history: {uci}")


def replay_moves(uci_moves: List[str]):
    """Replay moves and return the GameState (C++ or Python)."""
    if _USE_CPP:
        gs = _cpp.GameState()
        for uci in uci_moves:
            move, promo = uci_to_move_cpp(uci, gs)
            gs.make_move(move, ord(promo))
        return gs
    else:
        gs = ce.GameState()
        for uci in uci_moves:
            move, promo = uci_to_move_python(uci, gs)
            gs.make_move(move, promotion_choice=promo)
        return gs


def move_to_uci(move) -> str:
    return rc_to_sq(move.start_row, move.start_col) + rc_to_sq(move.end_row, move.end_col)


# ─────────────────────────────── API ───────────────────────────────────────

class MoveRequest(BaseModel):
    moves: List[str] = []   # e.g. ["e2e4", "e7e5", "g1f3"]
    sims: Optional[int] = None


class MoveResponse(BaseModel):
    from_sq: str
    to_sq: str
    promotion: Optional[str] = None
    status: str   # "ongoing" | "checkmate" | "stalemate"


app = FastAPI(title="AlphaZ0 Chess Engine")

# Allow the frontend to call this API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    global _model, _mcts_instance
    with _model_lock:
        _model = load_model()
        _mcts_instance = create_mcts(_model, _device, num_simulations=MCTS_SIMS)
    cpp_status = "ACTIVE" if _USE_CPP else "NOT AVAILABLE (using pure Python)"
    print(f"[app] C++ core: {cpp_status}")
    print(f"[app] Device: {_device}")
    print(f"[app] Default MCTS sims: {MCTS_SIMS}")


# ── Serve the HTML frontend ──────────────────────────────────────────────

CHESS_DIR = Path(__file__).parent.resolve()


@app.get("/")
def serve_frontend():
    """Serve the main chess UI."""
    index_path = CHESS_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "index.html not found")
    return FileResponse(index_path, media_type="text/html")


# Serve static assets (images, etc.)
if (CHESS_DIR / "Image").exists():
    app.mount("/Image", StaticFiles(directory=str(CHESS_DIR / "Image")), name="images")
if (CHESS_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(CHESS_DIR / "static")), name="static")


# ── API endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(_device),
        "sims_default": MCTS_SIMS,
        "cpp_core": _USE_CPP,
    }


@app.get("/api/status")
def api_status():
    """Extended status for monitoring."""
    return {
        "status": "ok",
        "device": str(_device),
        "sims_default": MCTS_SIMS,
        "cpp_core": _USE_CPP,
        "model_loaded": _model is not None,
        "gpu_name": torch.cuda.get_device_name(_device) if _device.type == "cuda" else None,
    }


@app.post("/get_move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    try:
        gs = replay_moves(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e))

    valid_moves = gs.get_valid_moves()
    status = gs.get_status(valid_moves)
    if status != "ongoing":
        raise HTTPException(400, f"Game already over: {status}")

    sims = req.sims or MCTS_SIMS
    with _model_lock:
        mcts_obj = create_mcts(_model, _device, num_simulations=sims)
        _, best_move = mcts_obj.get_move_probs(gs, temperature=1e-3)

    if best_move is None:
        raise HTTPException(500, "MCTS returned no move")

    # Detect promotion
    promo = None
    if _USE_CPP:
        # C++ Move has piece_moved as int8 (1=P, 2=R, 3=N, 4=B, 5=Q, 6=K)
        pt = abs(best_move.piece_moved)
        if pt == 1 and best_move.end_row in (0, 7):
            promo = "q"
    else:
        if best_move.piece_moved[1] == "P" and best_move.end_row in (0, 7):
            promo = "q"

    # Make the move to check resulting status
    if promo:
        gs.make_move(best_move, ord('Q') if _USE_CPP else 'Q')
    else:
        gs.make_move(best_move)

    new_valid = gs.get_valid_moves()
    new_status = gs.get_status(new_valid)

    return MoveResponse(
        from_sq=rc_to_sq(best_move.start_row, best_move.start_col),
        to_sq=rc_to_sq(best_move.end_row, best_move.end_col),
        promotion=promo,
        status=new_status,
    )
