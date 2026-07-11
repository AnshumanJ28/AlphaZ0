# AlphaZ0 — Neural Chess Engine

A from-scratch chess engine powered by an AlphaZero-style neural network, Monte Carlo Tree Search (MCTS), and an A3C reinforcement learning training pipeline. Built in Python with PyTorch and Pygame.

**🔗 Live demo: [alphaz0.onrender.com](https://alphaz0.onrender.com)** — play against AlphaZ0 or a friend (Pass & Play) directly in the browser, no install required.

> The web version currently plays at roughly **100 Elo** (chess.com scale) — an early checkpoint, well below the engine's ceiling. Strength scales directly with training: see [How Training Makes the Bot Stronger](#how-training-makes-the-bot-stronger) for how Elo climbs as policy loss drops.

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [Live Demo](#live-demo)
3. [How It Works — The Big Picture](#how-it-works--the-big-picture)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [How Training Makes the Bot Stronger](#how-training-makes-the-bot-stronger)
7. [Training Logs — What They Tell You](#training-logs--what-they-tell-you)
8. [Running the Project](#running-the-project)
9. [Project Structure](#project-structure)
10. [Dependencies](#dependencies)

---

## What Is This?

AlphaZ0 is a self-learning chess bot inspired by DeepMind's AlphaZero. It learns to play chess purely by playing against itself — no hardcoded openings, no Stockfish, no human game databases. It starts knowing only the rules of chess, and gets better every time you train it.

The bot has three brain layers working together:
```
Rules (chesseng.py)  →  Search (mcts.py)  →  Intuition (NeuralNet.py)
        ↑                                              ↓
        └────────── Training (Train.py) teaches this ─┘
```

---

## Live Demo

**[alphaz0.onrender.com](https://alphaz0.onrender.com)**

A lightweight browser interface for the engine — built with plain HTML/CSS/JS (chess.js for rules, chessboard.js for the board), deployed as a static site on Render.

- **Pass & Play** — two people share the board locally
- **vs AlphaZ0** — play against the bot as either color
- Click or drag to move, legal moves highlight automatically, king's square flags red when in check, and pawn promotions let you choose Queen, Rook, Bishop, or Knight

The demo's listed Elo (currently ~100, chess.com scale) reflects the strength of whichever checkpoint is currently wired up to the frontend — it updates as training progresses. A full write-up of what Elo ranges correspond to which policy loss values is in [How Training Makes the Bot Stronger](#how-training-makes-the-bot-stronger).

---

## How It Works — The Big Picture

### Step 1 — The Neural Network evaluates positions

Given any board position, the network outputs two things:

- **Policy** — a probability distribution over all possible moves (which moves look good?)
- **Value** — a single number from -1 to +1 (who is winning right now?)

### Step 2 — MCTS uses the network to search deeper

Raw neural network output is decent but shallow. MCTS runs hundreds of simulated games forward from the current position, using the network's policy to guide which branches to explore and the value head to estimate how good each branch is. The result is a much stronger move than the network alone would pick.

### Step 3 — Self-play generates training data

Two copies of the bot play each other. Every position, MCTS visit distribution, and final game outcome is recorded. This creates `(board, best_move_distribution, who_won)` training triples.

### Step 4 — The network learns from its own games

The network is trained on the self-play data to:
- Match its policy output to what MCTS actually found (smarter search targets)
- Match its value output to the actual game result (better position evaluation)

### Step 5 — A3C workers accelerate this in parallel

Multiple background threads run their own lightweight rollouts simultaneously, computing advantage estimates and pushing gradients to the shared network continuously — even while the main thread is doing self-play.

### Step 6 — Repeat

After each iteration the bot is slightly smarter. It plays better games. Those better games produce better training data. The network gets smarter. This is the self-improvement loop.

---

## Architecture Deep Dive

### Neural Network — NeuralNet.py

```
Input: (18, 8, 8) tensor — 18 planes describing the board
Plane 0-5   : White pieces  (P R N B Q K — one plane each)
Plane 6-11  : Black pieces  (P R N B Q K — one plane each)
Plane 12-15 : Castling rights (wKS, wQS, bKS, bQS)
Plane 16    : En passant target square
Plane 17    : Side to move (1.0 = white, 0.0 = black)

     ┌─────────────────────────────────┐
     │  Stem Conv (18 → 128 channels)  │
     │  BatchNorm + ReLU               │
     └──────────────┬──────────────────┘
                     │
     ┌──────────────▼──────────────────┐
     │   Residual Trunk (10 blocks)    │
     │   Each block: Conv→BN→ReLU      │
     │               Conv→BN→Residual  │
     └──────┬──────────────────┬───────┘
            │                  │
┌───────────▼──────┐  ┌────────▼─────────┐
│   Policy Head    │  │   Value Head     │
│  Conv(2ch) →     │  │  Conv(1ch) →     │
│  Flatten →       │  │  FC(64→256) →    │
│  FC → 4096       │  │  FC(256→1) →     │
│  log_softmax     │  │  tanh            │
└──────────────────┘  └──────────────────┘
     ↓                        ↓
Move probabilities      Position score
over all 64×64          in [-1.0, 1.0]
from-to pairs
```

Move encoding: every possible `(from_square, to_square)` pair maps to a flat index: `row*512 + col*64 + end_row*8 + end_col` giving 4096 buckets. Illegal moves get masked out after the network runs.

### MCTS — mcts.py

Each node in the tree stores:

| Field | Meaning |
|---|---|
| N | Visit count |
| W | Total value accumulated |
| Q | Mean value = W/N |
| P | Prior probability from policy head |

Selection uses PUCT:
```
score(s,a) = Q(s,a) + C_puct × P(s,a) × √(ΣN) / (1 + N(s,a))
```
C_puct = 1.5 balances exploitation (high Q) vs exploration (high P, low N).

Dirichlet noise is added to root priors (α=0.3, ε=0.25) so the bot always explores a little even when confident — critical for generating diverse training data.

### A3C Training — Train.py

```
Main Thread                     Worker Threads (×4)
──────────────                  ──────────────────────
play_one_game()                 Clone global weights
  └─ MCTS × N_games              Run short rollout
  └─ store (s,π,z)               Compute GAE advantages
                                  Actor + Critic + Entropy loss
Replay Buffer                   Push grads → global model
  └─ sample batch
  train_on_batch()
    └─ policy loss: -Σ π_mcts × log π_net
    └─ value loss:  MSE(v, z)
    └─ entropy bonus
```

---

## File-by-File Breakdown

### chesseng.py — The Rules Engine

The foundation everything else sits on. Zero dependencies on the neural network — pure chess logic only.

- `CastleRights` — tracks which sides can still castle, logged every move so undo works perfectly
- `GameState.make_move()` — applies a move, handles en passant, castling, promotion, rights updates
- `GameState.undo_move()` — fully reverses a move including all special cases
- `GameState.get_valid_moves()` — generates all pseudo-legal moves then filters any that leave the king in check
- `GameState.in_check()` — temporarily switches sides and asks if opponent moves hit the king
- `GameState.get_status()` — returns checkmate, stalemate, or ongoing
- `Move` — immutable value object with `__eq__` and `__hash__` so moves work as dict keys in MCTS

### BoardEncoder.py — Board to Tensor

Converts a `GameState` into an `(18, 8, 8)` float32 numpy array. Each plane is a binary grid where 1.0 means the condition is true at that square. The side-to-move plane means the same network handles both colors without needing two separate models.

### NeuralNet.py — The Brain

- `ResBlock` — two convolutions with BatchNorm and ReLU plus a skip connection. BatchNorm stabilizes training when game outcomes vary wildly in early self-play
- `ChessNet` — stem → residual trunk → policy head + value head. The `.float()` cast before FC layers prevents dtype issues on GPUs running mixed precision
- `get_policy_priors()` — masks the 4096-vector to only legal moves, renormalizes, handles the edge case where all priors collapse to zero

### mcts.py — The Search

- `MCTSNode` — uses `__slots__` for memory efficiency since thousands of nodes are created per move
- `MCTS.get_move_probs()` — runs N simulations of select → expand → evaluate → backup, builds final policy from visit counts
- Temperature=1.0 during training (keeps exploration), temperature=1e-3 during play (effectively greedy)
- `_copy_state()` — deep copies game state for each simulation so MCTS branches don't interfere with each other

### Train.py — The Learning Loop

- `TrainConfig` — dataclass holding every hyperparameter, change settings here
- `ReplayBuffer` — thread-safe deque, old experiences evicted when full, random sampling breaks temporal correlations
- `Experience` — one training sample: board tensor, MCTS policy distribution, game outcome z
- `play_one_game()` — plays a complete game, assigns outcomes by propagating the result back through the trajectory
- `A3CWorker` — background thread that clones weights, runs rollout, computes GAE advantages, pushes gradients to global model
- `train_on_batch()` — one supervised gradient step: policy cross-entropy + value MSE + entropy regularization

### chesmain.py — The Desktop UI

- Menu screen with Pass & Play vs AlphaZ0 options, color picker for bot mode
- Board flips automatically when playing as Black
- Bot thinks in a background thread so UI never freezes
- Move log sidebar with SAN notation scrollable with mouse wheel
- Loads `checkpoints/chess_net_best.pt` automatically, falls back to random moves if missing
- Controls: Z = undo (2 plies in bot mode), R/Esc = return to menu

### FE.html — The Web UI

A single-file browser interface mirroring `chesmain.py`'s Pass & Play / vs AlphaZ0 modes, deployed at [alphaz0.onrender.com](https://alphaz0.onrender.com). Uses chess.js for move legality and chessboard.js for rendering; click or drag to move, with legal-move highlighting, check detection, and a promotion picker (Queen/Rook/Bishop/Knight).

---

## How Training Makes the Bot Stronger

### The Improvement Cycle

```
Iteration 1:    Random network → random-ish MCTS → terrible games
                Terrible games → train on terrible data
                Network learns: at least move pieces toward center

Iteration 10:   Slightly better network → slightly better MCTS
                Better MCTS finds better moves → better game data
                Network learns basic tactics

Iteration 100:  Network knows tactics → MCTS finds combinations
                Better combinations → richer training data
                Network learns strategy

Iteration 1000+: Network understands positional play
                 MCTS can plan 10+ moves ahead
                 Bot becomes genuinely strong
```

### Policy Loss as Strength Indicator

| Policy Loss | Meaning | Approx. Elo (chess.com scale) |
|---|---|---|
| ~8.3 | Completely random (log of 4096 possible moves) | < 100 |
| ~6.0 | Learned basic move preferences | ~100–300 |
| ~4.0 | Decent tactical awareness | ~600–900 |
| ~2.0 | Strong play | ~1400–1700 |
| ~0.5 | Near master level | 2000+ |

The live demo's current checkpoint sits around policy loss 6.3, roughly **100 Elo** on chess.com's scale — above random, with MCTS doing most of the heavy lifting. Every 100 additional training iterations with enough games per iteration moves this lower, and the demo's rating will be updated as stronger checkpoints get deployed.

*(These Elo mappings are approximate — derived from typical policy-loss-to-strength correlations in AlphaZero-style engines, not from head-to-head rated games against chess.com opponents. Treat them as a rough guide to relative strength, not a certified rating.)*

### What Each Loss Component Teaches

**Policy loss** — the network learns which moves are good. As this drops, the network's first guess is closer to what MCTS would find, so MCTS needs fewer simulations to find strong moves.

**Value loss** — the network learns who is winning. A good value head means MCTS can cut off unpromising branches early. This is what gives the bot positional understanding — knowing a pawn structure is bad before seeing it lose material.

**Entropy bonus** — prevents the policy from collapsing onto the same 2-3 moves. Keeps the bot exploring different ideas during training, producing more diverse games and richer training data.

**A3C actor loss** — continuous gradient flow between main training iterations, helping the value head improve faster by seeing more diverse position evaluations.

### Recommended Training Progression

```bash
# Phase 1 — Bootstrap (overnight run)
# Goal: get policy loss below 5.0
python Train.py --games 15 --sims 50 --iters 100 --workers 2

# Phase 2 — Strengthen (weekend run)
# Goal: get policy loss below 3.5
python Train.py --games 25 --sims 100 --iters 300 --workers 4 --resume checkpoints/chess_net_best.pt

# Phase 3 — Polish (long run)
# Goal: genuinely challenging opponent
python Train.py --games 50 --sims 200 --iters 1000 --workers 4 --resume checkpoints/chess_net_best.pt
```

| Parameter | Effect | Recommendation |
|---|---|---|
| --games | More games = richer buffer = better training | At least 15 per iter |
| --sims | More MCTS sims = better move quality = better targets | 50-200 depending on speed |
| --iters | More iterations = more improvement cycles | As many as you can run |
| --workers | More A3C threads = faster value head training | Match your CPU core count |

---

## Training Logs — What They Tell You

```
policy_loss = 6.30   ← network move guesses (lower = smarter)
value_loss  = 0.004  ← position evaluation accuracy (lower = better)
entropy     = 7.5    ← move diversity (too low = collapse, too high = random)
actor_loss  = 0.009  ← A3C policy gradient signal
critic_loss = 0.000  ← A3C value baseline (needs terminal states to learn)
buffer_size = 8590   ← total stored positions (more = better batches)
lr          = 0.0005 ← learning rate (cosine decay over time)
elapsed     = 342s   ← time per iteration
```

Warning signs:

- `value_loss` stuck at 0.000 — games not reaching checkmate, increase --games or --max-moves
- `policy_loss` going back up after iter 10 — buffer evicting good data, increase --buffer-size
- `entropy` below 3.0 — policy collapsed, add more Dirichlet noise or reduce training epochs
- `actor_loss` in the millions at iter 1 — normal A3C warmup instability, self-corrects

---

## Running the Project

```bash
# Play the game locally (desktop UI)
cd C:\Users\yourname\Downloads\Chess
python chesmain.py

# Or just open FE.html in a browser for the web UI —
# no install needed, or visit the hosted version:
# https://alphaz0.onrender.com

# Quick test training (30 min)
python Train.py --games 5 --sims 30 --iters 20

# Serious overnight training
python Train.py --games 15 --sims 80 --iters 200 --workers 2

# Resume from checkpoint
python Train.py --resume checkpoints/chess_net_best.pt --iters 300

# Evaluate the bot
python Train.py --mode eval --eval-model checkpoints/chess_net_best.pt --eval-games 20
```

---

## Project Structure

```
Chess/
├── chesmain.py       ← Pygame desktop UI, menu, bot integration
├── FE.html           ← Browser UI (deployed at alphaz0.onrender.com)
├── chesseng.py       ← Chess rules, move generation, game state
├── NeuralNet.py      ← Neural network (policy + value heads)
├── mcts.py           ← Monte Carlo Tree Search with PUCT
├── BoardEncoder.py   ← Board → (18,8,8) tensor conversion
├── Train.py          ← Self-play + A3C training pipeline
├── checkpoints/
│   ├── chess_net_best.pt       ← Best model (loaded by UI)
│   ├── chess_net_final.pt      ← Last model after training
│   └── chess_net_iter_XXXX.pt  ← Per-iteration snapshots
└── Image/
    ├── wP.png  wR.png  wN.png  wB.png  wQ.png  wK.png
    └── bP.png  bR.png  bN.png  bB.png  bQ.png  bK.png
```

---

## Dependencies

```bash
pip install pygame torch numpy
```

| Package | Version | Purpose |
|---|---|---|
| pygame | 2.6+ | UI rendering |
| torch | 2.0+ | Neural network, GPU acceleration |
| numpy | 1.24+ | Board encoding, MCTS policy arrays |

The web UI (`FE.html`) has no Python dependencies — it runs entirely in the browser via CDN-hosted chess.js and chessboard.js.

GPU is optional but recommended for training. The bot runs fine on CPU for playing.

---

*Built from scratch — engine, network, search, training loop, and UI. No chess libraries, no pretrained weights, no external datasets.*
