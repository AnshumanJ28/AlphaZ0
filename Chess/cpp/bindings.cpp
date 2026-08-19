// bindings.cpp — pybind11 module exposing the C++ core to Python
// Module name: alphaz0_cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "chess_engine.hpp"
#include "board_encoder.hpp"
#include "mcts.hpp"
#include "self_play.hpp"

namespace py = pybind11;
using namespace alphaz0;

// Helper: convert BoardTensor to numpy array
py::array_t<float> board_tensor_to_numpy(const BoardTensor& bt) {
    auto result = py::array_t<float>({NUM_PLANES, BOARD_DIM, BOARD_DIM});
    auto buf = result.mutable_unchecked<3>();
    for (int p = 0; p < NUM_PLANES; ++p)
        for (int r = 0; r < BOARD_DIM; ++r)
            for (int c = 0; c < BOARD_DIM; ++c)
                buf(p, r, c) = bt.data[p][r][c];
    return result;
}

// Helper: encode a GameState to numpy array
py::array_t<float> encode_board_numpy(const GameState& gs) {
    return board_tensor_to_numpy(encode(gs));
}

// Wrap the Python eval function to match C++ EvalCallback signature
EvalCallback wrap_python_eval(py::object py_eval_fn) {
    // Prevent the py::object from dangling — take a shared copy
    auto fn_ptr = std::make_shared<py::object>(py_eval_fn);

    return [fn_ptr](const float* board_data, size_t size) -> EvalResult {
        // Acquire the GIL since we're calling back into Python
        py::gil_scoped_acquire acquire;

        // Create numpy array from the raw board data (no copy)
        auto arr = py::array_t<float>(
            {NUM_PLANES, BOARD_DIM, BOARD_DIM},
            board_data
        );

        // Call the Python function
        py::object result = (*fn_ptr)(arr);

        // Parse result: (log_policy_array, value_float)
        py::tuple tup = result.cast<py::tuple>();
        py::array_t<float> log_policy = tup[0].cast<py::array_t<float>>();
        float value = tup[1].cast<float>();

        EvalResult er;
        er.value = value;
        er.policy.resize(POLICY_SIZE);

        auto policy_buf = log_policy.unchecked<1>();
        for (int i = 0; i < POLICY_SIZE; ++i) {
            er.policy[i] = policy_buf(i);
        }

        return er;
    };
}


PYBIND11_MODULE(alphaz0_cpp, m) {
    m.doc() = "AlphaZ0 C++ core — chess engine, MCTS, board encoding, self-play";

    // ─── CastleRights ───────────────────────────────────────────
    py::class_<CastleRights>(m, "CastleRights")
        .def(py::init<>())
        .def(py::init<bool, bool, bool, bool>())
        .def_readwrite("wks", &CastleRights::wks)
        .def_readwrite("bks", &CastleRights::bks)
        .def_readwrite("wqs", &CastleRights::wqs)
        .def_readwrite("bqs", &CastleRights::bqs)
        .def("copy", &CastleRights::copy);

    // ─── Move ───────────────────────────────────────────────────
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("start_row", &Move::start_row)
        .def_readwrite("start_col", &Move::start_col)
        .def_readwrite("end_row", &Move::end_row)
        .def_readwrite("end_col", &Move::end_col)
        .def_readwrite("piece_moved", &Move::piece_moved)
        .def_readwrite("piece_captured", &Move::piece_captured)
        .def_readwrite("is_en_passant", &Move::is_en_passant)
        .def_readwrite("is_castle", &Move::is_castle)
        .def("to_index", &Move::to_index)
        .def("get_notation", &Move::get_notation)
        .def("__eq__", &Move::operator==)
        .def("__hash__", [](const Move& m) { return std::hash<int>()(m.to_index()); })
        .def("__repr__", [](const Move& m) {
            return "Move(" + m.get_notation() + " " + piece_to_str(m.piece_moved) + ")";
        });

    // ─── GameState ──────────────────────────────────────────────
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def_readwrite("white_to_move", &GameState::white_to_move)
        .def_readwrite("has_ep", &GameState::has_ep)
        .def_readwrite("ep_row", &GameState::ep_row)
        .def_readwrite("ep_col", &GameState::ep_col)
        .def_readwrite("current_castling_rights", &GameState::current_castling_rights)
        .def_property_readonly("move_log", [](const GameState& gs) {
            return gs.move_log;
        })
        .def("make_move", &GameState::make_move,
             py::arg("move"), py::arg("promotion_choice") = 'Q')
        .def("undo_move", &GameState::undo_move)
        .def("get_valid_moves", [](GameState& gs) {
            // Release GIL for expensive computation
            py::gil_scoped_release release;
            return gs.get_valid_moves();
        })
        .def("get_all_possible_moves", &GameState::get_all_possible_moves,
             py::arg("ignore_castling") = false)
        .def("in_check", &GameState::in_check)
        .def("get_status", &GameState::get_status)
        .def("get_board_strings", &GameState::get_board_strings)
        // Provide a board property that returns a list-of-lists of strings
        // matching the Python GameState.board interface
        .def_property_readonly("board", [](const GameState& gs) {
            py::list rows;
            for (int r = 0; r < 8; ++r) {
                py::list row;
                for (int c = 0; c < 8; ++c) {
                    row.append(piece_to_str(gs.board[r][c]));
                }
                rows.append(row);
            }
            return rows;
        })
        // en_passant_possible property matching Python interface
        .def_property_readonly("en_passant_possible", [](const GameState& gs) -> py::object {
            if (gs.has_ep) {
                return py::make_tuple(gs.ep_row, gs.ep_col);
            }
            return py::tuple();
        });

    // ─── Board Encoding ─────────────────────────────────────────
    m.def("encode_board", &encode_board_numpy,
          "Encode a GameState into an (18,8,8) numpy float32 array",
          py::arg("game_state"));

    m.def("move_to_index", [](const Move& m) { return move_to_index(m); },
          "Convert a Move to a policy index (0-4095)");

    // ─── MCTS ───────────────────────────────────────────────────
    py::class_<MCTS>(m, "MCTS")
        .def(py::init([](py::object py_eval_fn, int sims) {
            return std::make_unique<MCTS>(wrap_python_eval(py_eval_fn), sims);
        }), py::arg("eval_fn"), py::arg("num_simulations") = 200)
        .def("get_move_probs", [](MCTS& mcts, GameState& gs, float temperature) {
            // Release GIL for the C++ MCTS loop (it re-acquires for NN callbacks)
            py::gil_scoped_release release;
            auto [move_probs, best_move] = mcts.get_move_probs(gs, temperature);

            // Convert to Python-friendly format
            py::gil_scoped_acquire acquire;
            py::dict py_probs;
            // We need to return Move objects as keys
            // Instead, return the move_probs as dict and best_move separately
            return py::make_tuple(move_probs, best_move);
        }, py::arg("game_state"), py::arg("temperature") = 1.0f);

    // ─── Self-Play ──────────────────────────────────────────────
    py::class_<SelfPlayConfig>(m, "SelfPlayConfig")
        .def(py::init<>())
        .def_readwrite("mcts_sims", &SelfPlayConfig::mcts_sims)
        .def_readwrite("max_game_moves", &SelfPlayConfig::max_game_moves)
        .def_readwrite("temperature_moves", &SelfPlayConfig::temperature_moves);

    py::class_<Experience>(m, "Experience")
        .def(py::init<>())
        .def_property("board_tensor",
            [](const Experience& e) {
                return py::array_t<float>(
                    {NUM_PLANES, BOARD_DIM, BOARD_DIM},
                    e.board_tensor.data()
                );
            },
            [](Experience& e, py::array_t<float> arr) {
                auto buf = arr.unchecked<3>();
                e.board_tensor.resize(BoardTensor::size());
                for (int p = 0; p < NUM_PLANES; ++p)
                    for (int r = 0; r < BOARD_DIM; ++r)
                        for (int c = 0; c < BOARD_DIM; ++c)
                            e.board_tensor[p * 64 + r * 8 + c] = buf(p, r, c);
            })
        .def_property("policy_target",
            [](const Experience& e) {
                return py::array_t<float>(POLICY_SIZE, e.policy_target.data());
            },
            [](Experience& e, py::array_t<float> arr) {
                auto buf = arr.unchecked<1>();
                e.policy_target.resize(POLICY_SIZE);
                for (int i = 0; i < POLICY_SIZE; ++i)
                    e.policy_target[i] = buf(i);
            })
        .def_readwrite("value_target", &Experience::value_target);

    m.def("play_one_game", [](py::object py_eval_fn, const SelfPlayConfig& cfg) {
        auto eval_cb = wrap_python_eval(py_eval_fn);
        // Release GIL — C++ self-play re-acquires for NN callbacks
        py::gil_scoped_release release;
        return play_one_game(eval_cb, cfg);
    }, py::arg("eval_fn"), py::arg("config"),
       "Play one complete self-play game. Returns list of Experience.");

    // ─── Utility ────────────────────────────────────────────────
    m.def("piece_to_str", &piece_to_str);
    m.def("str_to_piece", &str_to_piece);

    m.attr("POLICY_SIZE") = POLICY_SIZE;
}
