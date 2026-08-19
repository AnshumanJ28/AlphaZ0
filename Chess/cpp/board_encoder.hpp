#pragma once
// board_encoder.hpp — C++ port of BoardEncoder.py
// encode(GameState) → float[18][8][8]

#include "chess_engine.hpp"
#include <array>
#include <cstring>

namespace alphaz0 {

// Piece type → plane index:  P=0, R=1, N=2, B=3, Q=4, K=5
// White pieces on planes 0-5, black on 6-11
// Planes 12-15: castling rights (wks, wqs, bks, bqs)
// Plane 16: en passant square
// Plane 17: side to move (1.0 if white)

constexpr int NUM_PLANES = 18;
constexpr int BOARD_DIM = 8;

// Map piece_type (1-6) → plane offset (0-5)
// piece_type: 1=P, 2=R, 3=N, 4=B, 5=Q, 6=K
// plane:      0=P, 1=R, 2=N, 3=B, 4=Q, 5=K
inline int piece_type_to_plane(int8_t pt) {
    return pt - 1; // P=0, R=1, N=2, B=3, Q=4, K=5
}

struct BoardTensor {
    float data[NUM_PLANES][BOARD_DIM][BOARD_DIM];

    BoardTensor() { std::memset(data, 0, sizeof(data)); }

    float* flat() { return &data[0][0][0]; }
    const float* flat() const { return &data[0][0][0]; }
    static constexpr size_t size() { return NUM_PLANES * BOARD_DIM * BOARD_DIM; }
};

inline BoardTensor encode(const GameState& gs) {
    BoardTensor t;

    // Piece planes
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int8_t piece = gs.board[r][c];
            if (piece == EMPTY) continue;
            int8_t pt = piece_type(piece);
            int plane = piece_type_to_plane(pt);
            if (is_black(piece)) plane += 6;
            t.data[plane][r][c] = 1.0f;
        }
    }

    // Castling rights
    const auto& cr = gs.current_castling_rights;
    if (cr.wks) for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) t.data[12][r][c] = 1.0f;
    if (cr.wqs) for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) t.data[13][r][c] = 1.0f;
    if (cr.bks) for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) t.data[14][r][c] = 1.0f;
    if (cr.bqs) for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) t.data[15][r][c] = 1.0f;

    // En passant
    if (gs.has_ep) {
        t.data[16][gs.ep_row][gs.ep_col] = 1.0f;
    }

    // Side to move
    if (gs.white_to_move) {
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
                t.data[17][r][c] = 1.0f;
    }

    return t;
}

} // namespace alphaz0
