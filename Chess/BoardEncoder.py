import numpy as np

PIECE_ORDER = "PRNBQK"
PIECE_TO_PLANE = {p: i for i, p in enumerate(PIECE_ORDER)}


def encode(gs) -> np.ndarray:
    """
    Parameters
    ----------
    gs : chess_engine.GameState

    Returns
    -------
    np.ndarray  shape (18, 8, 8)  dtype float32
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            cell = gs.board[r][c]
            if cell == "--":
                continue
            color, ptype = cell[0], cell[1]
            plane = PIECE_TO_PLANE[ptype] + (0 if color == "w" else 6)
            planes[plane, r, c] = 1.0

    cr = gs.current_castling_rights
    if cr.wks: planes[12] = 1.0
    if cr.wqs: planes[13] = 1.0
    if cr.bks: planes[14] = 1.0
    if cr.bqs: planes[15] = 1.0

    if gs.en_passant_possible:
        r, c = gs.en_passant_possible
        planes[16, r, c] = 1.0

    if gs.white_to_move:
        planes[17] = 1.0

    return planes