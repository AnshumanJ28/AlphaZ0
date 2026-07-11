"""
app.py — FastAPI bridge between the web frontend (chess.js) and the
trained AlphaZ0 engine (chesseng.py + NeuralNet.py + mcts.py).

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

The frontend POSTs the full move history (UCI-style strings like "e2e4",
or "e7e8q" for promotion) built from chess.js. This file replays that
history through chesseng.GameState (which has no FEN loader — it only
builds state via make_move from the starting position), then runs MCTS
on the resulting position and returns the bot's move.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chesseng as ce
from NeuralNet import ChessNet
from mcts import MCTS

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


def load_model() -> ChessNet:
    model = ChessNet().to(_device)
    loaded_from = None
    for path in CKPT_PATHS:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=_device))
            loaded_from = path
            break
    model.eval()
    if loaded_from:
        print(f"[app] Loaded checkpoint: {loaded_from}")
    else:
        print("[app] No checkpoint found — playing with random weights.")
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


def uci_to_move(uci: str, gs: ce.GameState) -> tuple[ce.Move, str]:
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


def replay_moves(uci_moves: List[str]) -> ce.GameState:
    gs = ce.GameState()
    for uci in uci_moves:
        move, promo = uci_to_move(uci, gs)
        gs.make_move(move, promotion_choice=promo)
    return gs


def move_to_uci(move: ce.Move) -> str:
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


app = FastAPI(title="AlphaZ0 Move API")

# Allow the static frontend (served from a different host) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your actual frontend origin once deployed
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    global _model
    with _model_lock:
        _model = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device), "sims_default": MCTS_SIMS}


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
        mcts = MCTS(_model, _device, num_simulations=sims)
        _, best_move = mcts.get_move_probs(gs, temperature=1e-3)

    if best_move is None:
        raise HTTPException(500, "MCTS returned no move")

    promo = None
    if best_move.piece_moved[1] == "P" and best_move.end_row in (0, 7):
        promo = "q"

    gs.make_move(best_move, promotion_choice=(promo.upper() if promo else "Q"))
    new_status = gs.get_status(gs.get_valid_moves())

    return MoveResponse(
        from_sq=rc_to_sq(best_move.start_row, best_move.start_col),
        to_sq=rc_to_sq(best_move.end_row, best_move.end_col),
        promotion=promo,
        status=new_status,
    )
