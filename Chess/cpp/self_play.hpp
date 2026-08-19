#pragma once
// self_play.hpp — C++ self-play loop
// Runs full games in C++, calling back to Python only for NN evaluation.

#include "chess_engine.hpp"
#include "board_encoder.hpp"
#include "mcts.hpp"

#include <vector>
#include <cstring>

namespace alphaz0 {

// One training sample from self-play
struct Experience {
    std::vector<float> board_tensor;   // 18*8*8 = 1152 floats
    std::vector<float> policy_target;  // POLICY_SIZE floats
    float value_target;                // game outcome z

    Experience() : board_tensor(BoardTensor::size(), 0.0f),
                   policy_target(POLICY_SIZE, 0.0f),
                   value_target(0.0f) {}
};

// Self-play configuration
struct SelfPlayConfig {
    int mcts_sims = 50;
    int max_game_moves = 80;
    int temperature_moves = 30;
};

// Play one complete game via MCTS and collect (state, π, z) tuples.
inline std::vector<Experience> play_one_game(
    EvalCallback eval_fn,
    const SelfPlayConfig& cfg)
{
    MCTS mcts(eval_fn, cfg.mcts_sims);
    GameState gs;

    // Trajectory: (board_tensor, policy, player_sign)
    struct TrajectoryEntry {
        std::vector<float> board_tensor;
        std::vector<float> policy;
        int player_sign;
    };
    std::vector<TrajectoryEntry> trajectory;

    int move_count = 0;
    int winner_sign = 0;

    while (move_count < cfg.max_game_moves) {
        float temperature = (move_count < cfg.temperature_moves) ? 1.0f : 1e-3f;

        auto [move_probs, best_move] = mcts.get_move_probs(gs, temperature);

        // Check if we got a valid move
        if (best_move.piece_moved == EMPTY) break;

        // Build full policy vector
        std::vector<float> pi(POLICY_SIZE, 0.0f);
        for (auto& [idx, prob] : move_probs) {
            pi[idx] = prob;
        }

        // Encode board
        BoardTensor bt = encode(gs);
        std::vector<float> board_vec(bt.flat(), bt.flat() + BoardTensor::size());

        int player_sign = gs.white_to_move ? 1 : -1;
        trajectory.push_back({std::move(board_vec), std::move(pi), player_sign});

        gs.make_move(best_move);
        move_count++;

        auto valid_moves = gs.get_valid_moves();
        auto status = gs.get_status(valid_moves);

        if (status == "checkmate") {
            winner_sign = -player_sign; // the side that just moved won
            break;
        } else if (status == "stalemate") {
            winner_sign = 0;
            break;
        }
    }
    // If loop ended without break, winner_sign stays 0 (draw by move limit)

    // Assign outcomes
    std::vector<Experience> experiences;
    experiences.reserve(trajectory.size());
    for (auto& entry : trajectory) {
        Experience exp;
        exp.board_tensor = std::move(entry.board_tensor);
        exp.policy_target = std::move(entry.policy);
        exp.value_target = (float)(winner_sign * entry.player_sign);
        experiences.push_back(std::move(exp));
    }

    return experiences;
}

} // namespace alphaz0
