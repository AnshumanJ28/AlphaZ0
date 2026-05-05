"""
Chess engine — GameState, Move, CastleRights.
"""

from __future__ import annotations


class CastleRights:
    __slots__ = ("wks", "bks", "wqs", "bqs")

    def __init__(self, wks: bool, bks: bool, wqs: bool, bqs: bool) -> None:
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs

    def copy(self) -> "CastleRights":
        return CastleRights(self.wks, self.bks, self.wqs, self.bqs)


class GameState:
    def __init__(self) -> None:
        self.board: list[list[str]] = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]
        self.white_to_move = True
        self.move_log: list[Move] = []
        self.en_passant_possible: tuple = ()
        self.current_castling_rights = CastleRights(True, True, True, True)
        self.castle_rights_log: list[CastleRights] = [CastleRights(True, True, True, True)]

    # ------------------------------------------------------------------ #
    # Make / Undo
    # ------------------------------------------------------------------ #

    def make_move(self, move: "Move", promotion_choice: str = "Q") -> None:
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.move_log.append(move)
        self.white_to_move = not self.white_to_move

        # Pawn promotion
        if move.piece_moved[1] == "P" and (move.end_row == 0 or move.end_row == 7):
            choice = promotion_choice if promotion_choice in ("R", "B", "N", "Q") else "Q"
            self.board[move.end_row][move.end_col] = move.piece_moved[0] + choice

        # En passant capture — remove the captured pawn
        if move.is_en_passant:
            self.board[move.start_row][move.end_col] = "--"

        # Update en passant square
        if move.piece_moved[1] == "P" and abs(move.start_row - move.end_row) == 2:
            self.en_passant_possible = ((move.start_row + move.end_row) // 2, move.start_col)
        else:
            self.en_passant_possible = ()

        # Castle — move the rook
        if move.is_castle:
            row = move.end_row
            if move.end_col == 6:       # king-side
                self.board[row][5] = self.board[row][7]
                self.board[row][7] = "--"
            else:                       # queen-side
                self.board[row][3] = self.board[row][0]
                self.board[row][0] = "--"

        self._update_castle_rights(move)
        self.castle_rights_log.append(self.current_castling_rights.copy())

    def undo_move(self) -> None:
        if not self.move_log:
            return
        move = self.move_log.pop()
        self.board[move.start_row][move.start_col] = move.piece_moved
        self.board[move.end_row][move.end_col] = move.piece_captured
        self.white_to_move = not self.white_to_move

        # Restore en passant
        if move.is_en_passant:
            self.board[move.end_row][move.end_col] = "--"
            self.board[move.start_row][move.end_col] = move.piece_captured
            self.en_passant_possible = (move.end_row, move.end_col)
        else:
            self.en_passant_possible = ()

        # Undo castle — put rook back
        if move.is_castle:
            row = move.end_row
            if move.end_col == 6:
                self.board[row][7] = self.board[row][5]
                self.board[row][5] = "--"
            else:
                self.board[row][0] = self.board[row][3]
                self.board[row][3] = "--"

        self.castle_rights_log.pop()
        self.current_castling_rights = self.castle_rights_log[-1].copy()

    def _update_castle_rights(self, move: "Move") -> None:
        cr = self.current_castling_rights
        if move.piece_moved == "wK":
            cr.wks = cr.wqs = False
        elif move.piece_moved == "bK":
            cr.bks = cr.bqs = False
        elif move.piece_moved == "wR":
            if move.start_row == 7:
                if move.start_col == 0:   cr.wqs = False
                elif move.start_col == 7: cr.wks = False
        elif move.piece_moved == "bR":
            if move.start_row == 0:
                if move.start_col == 0:   cr.bqs = False
                elif move.start_col == 7: cr.bks = False
        # Also revoke if a rook is captured
        if move.piece_captured == "wR":
            if move.end_row == 7:
                if move.end_col == 0:   cr.wqs = False
                elif move.end_col == 7: cr.wks = False
        elif move.piece_captured == "bR":
            if move.end_row == 0:
                if move.end_col == 0:   cr.bqs = False
                elif move.end_col == 7: cr.bks = False

    # ------------------------------------------------------------------ #
    # Valid move filtering
    # ------------------------------------------------------------------ #

    def get_valid_moves(self) -> list["Move"]:
        """Return legal moves — does not leave own king in check."""
        saved_ep = self.en_passant_possible
        saved_cr = self.current_castling_rights.copy()

        moves = self.get_all_possible_moves()

        for i in range(len(moves) - 1, -1, -1):
            self.make_move(moves[i])
            self.white_to_move = not self.white_to_move
            in_check = self.in_check()
            self.white_to_move = not self.white_to_move
            self.undo_move()
            if in_check:
                moves.pop(i)

        # Restore state snapshots
        self.en_passant_possible = saved_ep
        self.current_castling_rights = saved_cr
        return moves

    def in_check(self) -> bool:
        color = "w" if self.white_to_move else "b"
        king_pos = self._find_king(color)
        return self._square_under_attack(king_pos[0], king_pos[1])

    def _square_under_attack(self, r: int, c: int) -> bool:
        self.white_to_move = not self.white_to_move
        opp_moves = self.get_all_possible_moves(ignore_castling=True)
        self.white_to_move = not self.white_to_move
        return any(m.end_row == r and m.end_col == c for m in opp_moves)

    def _find_king(self, color: str) -> tuple:
        target = color + "K"
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == target:
                    return (r, c)
        raise RuntimeError(f"King not found for '{color}'")

    # ------------------------------------------------------------------ #
    # Move generation
    # ------------------------------------------------------------------ #

    def get_all_possible_moves(self, ignore_castling: bool = False) -> list["Move"]:
        moves: list[Move] = []
        generators = {
            "P": self._get_pawn_moves,
            "R": self._get_rook_moves,
            "N": self._get_knight_moves,
            "B": self._get_bishop_moves,
            "Q": self._get_queen_moves,
            "K": self._get_king_moves,
        }
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece == "--":
                    continue
                if (piece[0] == "w") == self.white_to_move:
                    generators[piece[1]](r, c, moves, ignore_castling)
        return moves

    def _get_pawn_moves(self, r, c, moves, *_):
        direction = -1 if self.white_to_move else 1
        start_row = 6 if self.white_to_move else 1
        enemy = "b" if self.white_to_move else "w"

        nr = r + direction
        if 0 <= nr < 8 and self.board[nr][c] == "--":
            moves.append(Move((r, c), (nr, c), self.board))
            if r == start_row and self.board[r + 2 * direction][c] == "--":
                moves.append(Move((r, c), (r + 2 * direction, c), self.board))

        for dc in (-1, 1):
            nc = c + dc
            if not (0 <= nc < 8):
                continue
            nr = r + direction
            if not (0 <= nr < 8):
                continue
            if self.board[nr][nc][0] == enemy:
                moves.append(Move((r, c), (nr, nc), self.board))
            elif (nr, nc) == self.en_passant_possible:
                moves.append(Move((r, c), (nr, nc), self.board, is_en_passant=True))

    def _get_rook_moves(self, r, c, moves, *_):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for i in range(1, 8):
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    break
                target = self.board[nr][nc]
                if target == "--":
                    moves.append(Move((r, c), (nr, nc), self.board))
                else:
                    if target[0] != self.board[r][c][0]:
                        moves.append(Move((r, c), (nr, nc), self.board))
                    break

    def _get_knight_moves(self, r, c, moves, *_):
        for dr, dc in ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = self.board[nr][nc]
                if target == "--" or target[0] != self.board[r][c][0]:
                    moves.append(Move((r, c), (nr, nc), self.board))

    def _get_bishop_moves(self, r, c, moves, *_):
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            for i in range(1, 8):
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    break
                target = self.board[nr][nc]
                if target == "--":
                    moves.append(Move((r, c), (nr, nc), self.board))
                else:
                    if target[0] != self.board[r][c][0]:
                        moves.append(Move((r, c), (nr, nc), self.board))
                    break

    def _get_queen_moves(self, r, c, moves, *_):
        self._get_rook_moves(r, c, moves)
        self._get_bishop_moves(r, c, moves)

    def _get_king_moves(self, r, c, moves, ignore_castling=False):
        color = self.board[r][c][0]
        for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = self.board[nr][nc]
                if target == "--" or target[0] != color:
                    moves.append(Move((r, c), (nr, nc), self.board))

        if ignore_castling or self.in_check():
            return

        cr = self.current_castling_rights

        if self.white_to_move:
            if cr.wks:
                if (self.board[7][5] == "--" and self.board[7][6] == "--"
                        and not self._square_under_attack(7, 5)
                        and not self._square_under_attack(7, 6)):
                    moves.append(Move((7, 4), (7, 6), self.board, is_castle=True))
            if cr.wqs:
                if (self.board[7][1] == "--" and self.board[7][2] == "--"
                        and self.board[7][3] == "--"
                        and not self._square_under_attack(7, 2)
                        and not self._square_under_attack(7, 3)):
                    moves.append(Move((7, 4), (7, 2), self.board, is_castle=True))
        else:
            if cr.bks:
                if (self.board[0][5] == "--" and self.board[0][6] == "--"
                        and not self._square_under_attack(0, 5)
                        and not self._square_under_attack(0, 6)):
                    moves.append(Move((0, 4), (0, 6), self.board, is_castle=True))
            if cr.bqs:
                if (self.board[0][1] == "--" and self.board[0][2] == "--"
                        and self.board[0][3] == "--"
                        and not self._square_under_attack(0, 2)
                        and not self._square_under_attack(0, 3)):
                    moves.append(Move((0, 4), (0, 2), self.board, is_castle=True))

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #

    def get_status(self, valid_moves: list) -> str:
        """
        Pass in the already-computed valid_moves list to avoid recomputing.
        Returns 'checkmate', 'stalemate', or 'ongoing'.
        """
        if len(valid_moves) == 0:
            return "checkmate" if self.in_check() else "stalemate"
        return "ongoing"


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

class Move:
    _FILES = "abcdefgh"
    _RANKS = "87654321"

    def __init__(self, start_sq, end_sq, board,
                 is_en_passant=False, is_castle=False):
        self.start_row = start_sq[0]
        self.start_col = start_sq[1]
        self.end_row   = end_sq[0]
        self.end_col   = end_sq[1]

        self.piece_moved    = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]

        self.is_en_passant = is_en_passant
        self.is_castle     = is_castle

        if self.is_en_passant:
            self.piece_captured = "bP" if self.piece_moved[0] == "w" else "wP"

    def get_notation(self) -> str:
        return (self._FILES[self.start_col] + self._RANKS[self.start_row]
                + self._FILES[self.end_col] + self._RANKS[self.end_row])

    def __eq__(self, other):
        if not isinstance(other, Move):
            return NotImplemented
        return (self.start_row == other.start_row
                and self.start_col == other.start_col
                and self.end_row == other.end_row
                and self.end_col == other.end_col)

    def __hash__(self):
        return hash((self.start_row, self.start_col, self.end_row, self.end_col))

    def __repr__(self):
        return f"Move({self.get_notation()} {self.piece_moved})"