#pragma once
// chess_engine.hpp — C++ port of chesseng.py
// Identical semantics: same board representation, move-gen, make/undo.

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace alphaz0 {

// ─── Piece encoding ─────────────────────────────────────────────
// We store pieces as int8_t:
//   0 = empty ("--")
//   Positive = white, negative = black
//   |1|=P, |2|=R, |3|=N, |4|=B, |5|=Q, |6|=K
enum Piece : int8_t {
    EMPTY = 0,
    wP =  1, wR =  2, wN =  3, wB =  4, wQ =  5, wK =  6,
    bP = -1, bR = -2, bN = -3, bB = -4, bQ = -5, bK = -6
};

inline bool is_white(int8_t p) { return p > 0; }
inline bool is_black(int8_t p) { return p < 0; }
inline int8_t piece_type(int8_t p) { return p > 0 ? p : -p; }
inline int8_t piece_color_sign(int8_t p) { return p > 0 ? 1 : -1; }

// Convert between string representation ("wP","bK","--") and int8_t
inline int8_t str_to_piece(const char* s) {
    if (s[0] == '-') return EMPTY;
    int8_t base = 0;
    switch (s[1]) {
        case 'P': base = 1; break;
        case 'R': base = 2; break;
        case 'N': base = 3; break;
        case 'B': base = 4; break;
        case 'Q': base = 5; break;
        case 'K': base = 6; break;
        default: return EMPTY;
    }
    return s[0] == 'w' ? base : -base;
}

inline std::string piece_to_str(int8_t p) {
    if (p == EMPTY) return "--";
    const char* types = "XPRNBQK";
    char color = p > 0 ? 'w' : 'b';
    int8_t t = piece_type(p);
    return std::string(1, color) + types[t];
}

// ─── CastleRights ───────────────────────────────────────────────
struct CastleRights {
    bool wks = true, bks = true, wqs = true, bqs = true;

    CastleRights() = default;
    CastleRights(bool wks_, bool bks_, bool wqs_, bool bqs_)
        : wks(wks_), bks(bks_), wqs(wqs_), bqs(bqs_) {}

    CastleRights copy() const { return *this; }
};

// ─── Move ───────────────────────────────────────────────────────
struct Move {
    int start_row, start_col, end_row, end_col;
    int8_t piece_moved;
    int8_t piece_captured;
    bool is_en_passant;
    bool is_castle;

    Move() : start_row(0), start_col(0), end_row(0), end_col(0),
             piece_moved(EMPTY), piece_captured(EMPTY),
             is_en_passant(false), is_castle(false) {}

    Move(int sr, int sc, int er, int ec,
         const int8_t board[8][8],
         bool en_passant = false, bool castle = false)
        : start_row(sr), start_col(sc), end_row(er), end_col(ec),
          is_en_passant(en_passant), is_castle(castle)
    {
        piece_moved = board[sr][sc];
        piece_captured = board[er][ec];
        if (is_en_passant) {
            piece_captured = is_white(piece_moved) ? bP : wP;
        }
    }

    bool operator==(const Move& o) const {
        return start_row == o.start_row && start_col == o.start_col
            && end_row == o.end_row && end_col == o.end_col;
    }

    bool operator!=(const Move& o) const { return !(*this == o); }

    int to_index() const {
        return start_row * 512 + start_col * 64 + end_row * 8 + end_col;
    }

    std::string get_notation() const {
        static const char files[] = "abcdefgh";
        static const char ranks[] = "87654321";
        std::string s;
        s += files[start_col];
        s += ranks[start_row];
        s += files[end_col];
        s += ranks[end_row];
        return s;
    }

    // For use as map key
    struct Hash {
        size_t operator()(const Move& m) const {
            return std::hash<int>()(m.to_index());
        }
    };
};

// ─── GameState ──────────────────────────────────────────────────
class GameState {
public:
    int8_t board[8][8];
    bool white_to_move;
    int ep_row, ep_col;  // en passant square (-1 if none)
    bool has_ep;
    CastleRights current_castling_rights;
    std::vector<CastleRights> castle_rights_log;
    std::vector<Move> move_log;
    // Store en passant history for undo
    std::vector<std::pair<bool, std::pair<int,int>>> ep_log;

    GameState() : white_to_move(true), ep_row(-1), ep_col(-1), has_ep(false) {
        // Initial position
        const int8_t init[8][8] = {
            {bR, bN, bB, bQ, bK, bB, bN, bR},
            {bP, bP, bP, bP, bP, bP, bP, bP},
            {0,  0,  0,  0,  0,  0,  0,  0},
            {0,  0,  0,  0,  0,  0,  0,  0},
            {0,  0,  0,  0,  0,  0,  0,  0},
            {0,  0,  0,  0,  0,  0,  0,  0},
            {wP, wP, wP, wP, wP, wP, wP, wP},
            {wR, wN, wB, wQ, wK, wB, wN, wR}
        };
        std::memcpy(board, init, sizeof(board));
        castle_rights_log.push_back(CastleRights(true, true, true, true));
    }

    // ── Make / Undo ─────────────────────────────────────────────
    void make_move(const Move& move, char promotion_choice = 'Q') {
        // Save ep state for undo
        ep_log.push_back({has_ep, {ep_row, ep_col}});

        board[move.start_row][move.start_col] = EMPTY;
        board[move.end_row][move.end_col] = move.piece_moved;
        move_log.push_back(move);
        white_to_move = !white_to_move;

        // Pawn promotion
        int8_t pt = piece_type(move.piece_moved);
        if (pt == 1 && (move.end_row == 0 || move.end_row == 7)) {
            int8_t promo;
            switch (promotion_choice) {
                case 'R': case 'r': promo = 2; break;
                case 'B': case 'b': promo = 4; break;
                case 'N': case 'n': promo = 3; break;
                default:            promo = 5; break; // Q
            }
            board[move.end_row][move.end_col] =
                is_white(move.piece_moved) ? promo : -promo;
        }

        // En passant capture
        if (move.is_en_passant) {
            board[move.start_row][move.end_col] = EMPTY;
        }

        // Update en passant square
        if (pt == 1 && std::abs(move.start_row - move.end_row) == 2) {
            has_ep = true;
            ep_row = (move.start_row + move.end_row) / 2;
            ep_col = move.start_col;
        } else {
            has_ep = false;
            ep_row = -1;
            ep_col = -1;
        }

        // Castle — move the rook
        if (move.is_castle) {
            int row = move.end_row;
            if (move.end_col == 6) { // king-side
                board[row][5] = board[row][7];
                board[row][7] = EMPTY;
            } else { // queen-side
                board[row][3] = board[row][0];
                board[row][0] = EMPTY;
            }
        }

        update_castle_rights(move);
        castle_rights_log.push_back(current_castling_rights.copy());
    }

    void undo_move() {
        if (move_log.empty()) return;
        Move move = move_log.back();
        move_log.pop_back();

        board[move.start_row][move.start_col] = move.piece_moved;
        board[move.end_row][move.end_col] = move.piece_captured;
        white_to_move = !white_to_move;

        // Restore en passant
        if (move.is_en_passant) {
            board[move.end_row][move.end_col] = EMPTY;
            board[move.start_row][move.end_col] = move.piece_captured;
        }

        // Undo castle
        if (move.is_castle) {
            int row = move.end_row;
            if (move.end_col == 6) {
                board[row][7] = board[row][5];
                board[row][5] = EMPTY;
            } else {
                board[row][0] = board[row][3];
                board[row][3] = EMPTY;
            }
        }

        castle_rights_log.pop_back();
        current_castling_rights = castle_rights_log.back().copy();

        // Restore ep state
        if (!ep_log.empty()) {
            auto& prev = ep_log.back();
            has_ep = prev.first;
            ep_row = prev.second.first;
            ep_col = prev.second.second;
            ep_log.pop_back();
        }
    }

    // ── Valid move filtering ────────────────────────────────────
    std::vector<Move> get_valid_moves() {
        bool saved_ep = has_ep;
        int saved_ep_r = ep_row, saved_ep_c = ep_col;
        CastleRights saved_cr = current_castling_rights.copy();

        auto moves = get_all_possible_moves(false);

        for (int i = (int)moves.size() - 1; i >= 0; --i) {
            make_move(moves[i]);
            white_to_move = !white_to_move;
            bool check = in_check();
            white_to_move = !white_to_move;
            undo_move();
            if (check) {
                moves.erase(moves.begin() + i);
            }
        }

        has_ep = saved_ep;
        ep_row = saved_ep_r;
        ep_col = saved_ep_c;
        current_castling_rights = saved_cr;
        return moves;
    }

    bool in_check() const {
        int8_t color_sign = white_to_move ? 1 : -1;
        auto [kr, kc] = find_king(color_sign);
        return square_under_attack(kr, kc);
    }

    std::string get_status(const std::vector<Move>& valid_moves) const {
        if (valid_moves.empty()) {
            return in_check() ? "checkmate" : "stalemate";
        }
        return "ongoing";
    }

    // ── Move generation ─────────────────────────────────────────
    std::vector<Move> get_all_possible_moves(bool ignore_castling = false) const {
        std::vector<Move> moves;
        moves.reserve(64);
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                int8_t piece = board[r][c];
                if (piece == EMPTY) continue;
                if ((is_white(piece)) != white_to_move) continue;
                int8_t pt = piece_type(piece);
                switch (pt) {
                    case 1: get_pawn_moves(r, c, moves); break;
                    case 2: get_rook_moves(r, c, moves); break;
                    case 3: get_knight_moves(r, c, moves); break;
                    case 4: get_bishop_moves(r, c, moves); break;
                    case 5: get_queen_moves(r, c, moves); break;
                    case 6: get_king_moves(r, c, moves, ignore_castling); break;
                }
            }
        }
        return moves;
    }

    // Get board as a 2D string array (for Python interop)
    std::array<std::array<std::string, 8>, 8> get_board_strings() const {
        std::array<std::array<std::string, 8>, 8> result;
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
                result[r][c] = piece_to_str(board[r][c]);
        return result;
    }

private:
    void update_castle_rights(const Move& move) {
        auto& cr = current_castling_rights;
        if (move.piece_moved == wK) { cr.wks = cr.wqs = false; }
        else if (move.piece_moved == bK) { cr.bks = cr.bqs = false; }
        else if (move.piece_moved == wR) {
            if (move.start_row == 7) {
                if (move.start_col == 0) cr.wqs = false;
                else if (move.start_col == 7) cr.wks = false;
            }
        }
        else if (move.piece_moved == bR) {
            if (move.start_row == 0) {
                if (move.start_col == 0) cr.bqs = false;
                else if (move.start_col == 7) cr.bks = false;
            }
        }
        // Revoke if a rook is captured
        if (move.piece_captured == wR) {
            if (move.end_row == 7) {
                if (move.end_col == 0) cr.wqs = false;
                else if (move.end_col == 7) cr.wks = false;
            }
        }
        else if (move.piece_captured == bR) {
            if (move.end_row == 0) {
                if (move.end_col == 0) cr.bqs = false;
                else if (move.end_col == 7) cr.bks = false;
            }
        }
    }

    std::pair<int,int> find_king(int8_t color_sign) const {
        int8_t target = color_sign > 0 ? wK : bK;
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
                if (board[r][c] == target) return {r, c};
        throw std::runtime_error("King not found");
    }

    bool square_under_attack(int r, int c) const {
        // Temporarily flip turn to generate opponent moves
        // (const_cast needed because we restore state)
        GameState* self = const_cast<GameState*>(this);
        self->white_to_move = !self->white_to_move;
        auto opp_moves = self->get_all_possible_moves(true);
        self->white_to_move = !self->white_to_move;
        for (auto& m : opp_moves) {
            if (m.end_row == r && m.end_col == c) return true;
        }
        return false;
    }

    // ── Individual piece move generators ────────────────────────
    void get_pawn_moves(int r, int c, std::vector<Move>& moves) const {
        int direction = white_to_move ? -1 : 1;
        int start_row = white_to_move ? 6 : 1;
        int8_t enemy_sign = white_to_move ? -1 : 1;

        int nr = r + direction;
        if (nr >= 0 && nr < 8 && board[nr][c] == EMPTY) {
            moves.push_back(Move(r, c, nr, c, board));
            int nr2 = r + 2 * direction;
            if (r == start_row && board[nr2][c] == EMPTY) {
                moves.push_back(Move(r, c, nr2, c, board));
            }
        }

        for (int dc : {-1, 1}) {
            int nc = c + dc;
            if (nc < 0 || nc >= 8) continue;
            nr = r + direction;
            if (nr < 0 || nr >= 8) continue;
            if (board[nr][nc] != EMPTY && piece_color_sign(board[nr][nc]) == enemy_sign) {
                moves.push_back(Move(r, c, nr, nc, board));
            } else if (has_ep && nr == ep_row && nc == ep_col) {
                moves.push_back(Move(r, c, nr, nc, board, true, false));
            }
        }
    }

    void get_rook_moves(int r, int c, std::vector<Move>& moves) const {
        static const int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        int8_t my_color = piece_color_sign(board[r][c]);
        for (auto& d : dirs) {
            for (int i = 1; i < 8; ++i) {
                int nr = r + d[0]*i, nc = c + d[1]*i;
                if (nr < 0 || nr >= 8 || nc < 0 || nc >= 8) break;
                if (board[nr][nc] == EMPTY) {
                    moves.push_back(Move(r, c, nr, nc, board));
                } else {
                    if (piece_color_sign(board[nr][nc]) != my_color)
                        moves.push_back(Move(r, c, nr, nc, board));
                    break;
                }
            }
        }
    }

    void get_knight_moves(int r, int c, std::vector<Move>& moves) const {
        static const int jumps[8][2] = {
            {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
        };
        int8_t my_color = piece_color_sign(board[r][c]);
        for (auto& j : jumps) {
            int nr = r + j[0], nc = c + j[1];
            if (nr < 0 || nr >= 8 || nc < 0 || nc >= 8) continue;
            if (board[nr][nc] == EMPTY || piece_color_sign(board[nr][nc]) != my_color)
                moves.push_back(Move(r, c, nr, nc, board));
        }
    }

    void get_bishop_moves(int r, int c, std::vector<Move>& moves) const {
        static const int dirs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
        int8_t my_color = piece_color_sign(board[r][c]);
        for (auto& d : dirs) {
            for (int i = 1; i < 8; ++i) {
                int nr = r + d[0]*i, nc = c + d[1]*i;
                if (nr < 0 || nr >= 8 || nc < 0 || nc >= 8) break;
                if (board[nr][nc] == EMPTY) {
                    moves.push_back(Move(r, c, nr, nc, board));
                } else {
                    if (piece_color_sign(board[nr][nc]) != my_color)
                        moves.push_back(Move(r, c, nr, nc, board));
                    break;
                }
            }
        }
    }

    void get_queen_moves(int r, int c, std::vector<Move>& moves) const {
        get_rook_moves(r, c, moves);
        get_bishop_moves(r, c, moves);
    }

    void get_king_moves(int r, int c, std::vector<Move>& moves,
                        bool ignore_castling = false) const {
        static const int dirs[8][2] = {
            {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
        };
        int8_t my_color = piece_color_sign(board[r][c]);
        for (auto& d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr < 0 || nr >= 8 || nc < 0 || nc >= 8) continue;
            if (board[nr][nc] == EMPTY || piece_color_sign(board[nr][nc]) != my_color)
                moves.push_back(Move(r, c, nr, nc, board));
        }

        if (ignore_castling || in_check()) return;

        const auto& cr = current_castling_rights;
        if (white_to_move) {
            if (cr.wks && board[7][5] == EMPTY && board[7][6] == EMPTY
                && !square_under_attack(7, 5) && !square_under_attack(7, 6)) {
                moves.push_back(Move(7, 4, 7, 6, board, false, true));
            }
            if (cr.wqs && board[7][1] == EMPTY && board[7][2] == EMPTY
                && board[7][3] == EMPTY
                && !square_under_attack(7, 2) && !square_under_attack(7, 3)) {
                moves.push_back(Move(7, 4, 7, 2, board, false, true));
            }
        } else {
            if (cr.bks && board[0][5] == EMPTY && board[0][6] == EMPTY
                && !square_under_attack(0, 5) && !square_under_attack(0, 6)) {
                moves.push_back(Move(0, 4, 0, 6, board, false, true));
            }
            if (cr.bqs && board[0][1] == EMPTY && board[0][2] == EMPTY
                && board[0][3] == EMPTY
                && !square_under_attack(0, 2) && !square_under_attack(0, 3)) {
                moves.push_back(Move(0, 4, 0, 2, board, false, true));
            }
        }
    }
};

} // namespace alphaz0
