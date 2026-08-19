#pragma once
// mcts.hpp — C++ port of mcts.py
// MCTSNode + MCTS with a Python neural-net evaluation callback.

#include "chess_engine.hpp"
#include "board_encoder.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>

namespace alphaz0 {

constexpr float C_PUCT = 1.5f;
constexpr float DIRICHLET_ALPHA = 0.3f;
constexpr float DIRICHLET_EPS = 0.25f;
constexpr int POLICY_SIZE = 4096;

inline int move_to_index(const Move& m) {
    return m.start_row * 512 + m.start_col * 64 + m.end_row * 8 + m.end_col;
}

// Result from the neural network evaluation callback
struct EvalResult {
    std::vector<float> policy;  // log_softmax output, size POLICY_SIZE
    float value;                // tanh output in [-1, 1]
};

// Callback type: takes a BoardTensor (18x8x8 floats) and returns EvalResult
using EvalCallback = std::function<EvalResult(const float*, size_t)>;

// ─── MCTSNode ───────────────────────────────────────────────────
struct MCTSNode {
    Move move;
    MCTSNode* parent;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children; // key = move_to_index
    std::vector<Move> child_moves; // ordered list of child moves

    int N = 0;
    float W = 0.0f;
    float Q = 0.0f;
    float P = 0.0f;  // prior
    bool is_expanded = false;

    MCTSNode(Move m = Move(), MCTSNode* p = nullptr, float prior = 0.0f)
        : move(m), parent(p), P(prior) {}

    float ucb_score(int parent_N) const {
        float u = C_PUCT * P * std::sqrt((float)parent_N) / (1.0f + N);
        return Q + u;
    }

    MCTSNode* best_child() {
        MCTSNode* best = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();
        for (auto& [idx, child] : children) {
            float score = child->ucb_score(N);
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }

    void expand(const std::vector<Move>& moves, const std::vector<float>& priors) {
        child_moves = moves;
        for (size_t i = 0; i < moves.size(); ++i) {
            int idx = move_to_index(moves[i]);
            children[idx] = std::make_unique<MCTSNode>(moves[i], this, priors[i]);
        }
        is_expanded = true;
    }

    void backup(float value) {
        MCTSNode* node = this;
        while (node) {
            node->N += 1;
            node->W += value;
            node->Q = node->W / node->N;
            value = -value;
            node = node->parent;
        }
    }
};

// ─── MCTS ───────────────────────────────────────────────────────
class MCTS {
public:
    EvalCallback eval_fn;
    int num_simulations;

    MCTS(EvalCallback fn, int sims = 200)
        : eval_fn(std::move(fn)), num_simulations(sims),
          rng(std::random_device{}()) {}

    // Returns (move_probs, best_move)
    // move_probs: map from Move to probability
    std::pair<std::unordered_map<int, float>, Move>
    get_move_probs(GameState& gs, float temperature = 1.0f) {
        auto root = std::make_unique<MCTSNode>();
        expand_node(root.get(), gs);
        add_dirichlet_noise(root.get());

        for (int sim = 0; sim < num_simulations; ++sim) {
            MCTSNode* node = root.get();
            GameState state = gs; // copy

            // Selection
            while (node->is_expanded && !node->children.empty()) {
                node = node->best_child();
                state.make_move(node->move);
            }

            // Check terminal
            auto valid = state.get_valid_moves();
            auto status = state.get_status(valid);

            float value;
            if (status == "checkmate") {
                value = 1.0f;
            } else if (status == "stalemate") {
                value = 0.0f;
            } else {
                // Expansion + evaluation
                expand_node(node, state);
                value = evaluate(state);
            }

            // Backup
            node->backup(-value);
        }

        return build_policy(root.get(), temperature);
    }

    // Get the list of child moves from a root node (for external use)
    std::vector<Move> get_root_moves(MCTSNode* root) const {
        return root->child_moves;
    }

private:
    std::mt19937 rng;

    float evaluate(const GameState& gs) {
        BoardTensor bt = encode(gs);
        EvalResult result = eval_fn(bt.flat(), BoardTensor::size());
        return result.value;
    }

    std::vector<float> get_priors(const GameState& gs,
                                   const std::vector<Move>& legal_moves) {
        BoardTensor bt = encode(gs);
        EvalResult result = eval_fn(bt.flat(), BoardTensor::size());

        // Mask and normalize
        std::vector<float> priors(legal_moves.size());
        float total = 0.0f;
        for (size_t i = 0; i < legal_moves.size(); ++i) {
            int idx = move_to_index(legal_moves[i]);
            float p = std::exp(result.policy[idx]);
            priors[i] = p;
            total += p;
        }

        if (total < 1e-8f) {
            // Uniform fallback
            float uniform = 1.0f / legal_moves.size();
            for (auto& p : priors) p = uniform;
        } else {
            for (auto& p : priors) p /= total;
        }

        return priors;
    }

    void expand_node(MCTSNode* node, GameState& gs) {
        auto legal = gs.get_valid_moves();
        if (legal.empty()) return;
        auto priors = get_priors(gs, legal);
        node->expand(legal, priors);
    }

    void add_dirichlet_noise(MCTSNode* root) {
        if (root->children.empty()) return;
        size_t n = root->children.size();

        // Generate Dirichlet noise using gamma distribution
        std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);
        std::vector<float> noise(n);
        float noise_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        for (auto& v : noise) v /= noise_sum;

        size_t i = 0;
        for (auto& [idx, child] : root->children) {
            child->P = (1.0f - DIRICHLET_EPS) * child->P + DIRICHLET_EPS * noise[i];
            ++i;
        }
    }

    std::pair<std::unordered_map<int, float>, Move>
    build_policy(MCTSNode* root, float temperature) {
        std::vector<Move> moves;
        std::vector<float> visits;
        moves.reserve(root->children.size());
        visits.reserve(root->children.size());

        for (auto& [idx, child] : root->children) {
            moves.push_back(child->move);
            visits.push_back((float)child->N);
        }

        std::vector<float> probs(moves.size());

        if (temperature < 1e-3f) {
            // Greedy
            int best_idx = std::distance(visits.begin(),
                std::max_element(visits.begin(), visits.end()));
            for (size_t i = 0; i < probs.size(); ++i) probs[i] = 0.0f;
            probs[best_idx] = 1.0f;
        } else {
            // Normalize visits before exponentiation
            float max_v = *std::max_element(visits.begin(), visits.end());
            if (max_v < 1e-8f) max_v = 1.0f;
            float sum = 0.0f;
            for (size_t i = 0; i < visits.size(); ++i) {
                probs[i] = std::pow(visits[i] / max_v, 1.0f / temperature);
                sum += probs[i];
            }
            if (sum < 1e-8f) {
                float uniform = 1.0f / moves.size();
                for (auto& p : probs) p = uniform;
            } else {
                for (auto& p : probs) p /= sum;
            }
        }

        // Sample move
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int sampled = dist(rng);

        std::unordered_map<int, float> move_probs;
        for (size_t i = 0; i < moves.size(); ++i) {
            move_probs[move_to_index(moves[i])] = probs[i];
        }

        return {move_probs, moves[sampled]};
    }
};

} // namespace alphaz0
