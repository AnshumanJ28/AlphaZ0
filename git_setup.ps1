# ─────────────────────────────────────────────────────────────────────────────
# git_setup.ps1  —  One-shot Git init + push for AlphaZ0
#
# Run this from the AlphaZ0 root folder (where README.md lives):
#   powershell -ExecutionPolicy Bypass -File git_setup.ps1
#
# What it does:
#   1. Initialises a git repo (if not already one)
#   2. Sets remote to your GitHub repo
#   3. Commits the current v1 Python-only files and tags as v1.0
#   4. Adds v2 C++ files and commits them on main
#   5. Pushes everything including the v1.0 tag
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "`n=== AlphaZ0 Git Setup ===" -ForegroundColor Cyan

# ── Step 1: Init repo ─────────────────────────────────────────────────────────
if (-not (Test-Path ".git")) {
    Write-Host "`n[1/6] Initialising git repo..." -ForegroundColor Yellow
    git init -b main
} else {
    Write-Host "`n[1/6] Git repo already initialised." -ForegroundColor Green
}

# ── Step 2: Set remote ────────────────────────────────────────────────────────
Write-Host "`n[2/6] Setting remote origin..." -ForegroundColor Yellow
$remotes = git remote
if ($remotes -contains "origin") {
    git remote set-url origin https://github.com/AnshumanJ28/AlphaZ0.git
    Write-Host "      Updated existing remote." -ForegroundColor Green
} else {
    git remote add origin https://github.com/AnshumanJ28/AlphaZ0.git
    Write-Host "      Added remote origin." -ForegroundColor Green
}

# ── Step 3: Commit v1 baseline (Python-only files) ───────────────────────────
Write-Host "`n[3/6] Staging v1 files (pure Python)..." -ForegroundColor Yellow
git add Chess/chesseng.py
git add Chess/BoardEncoder.py
git add Chess/NeuralNet.py
git add Chess/mcts.py
git add Chess/Train.py
git add Chess/requirements.txt
git add .gitignore
git add README.md
git add LICENSE

# Check if there's already a commit (repo might already have history from GitHub)
$headRef = git rev-parse --verify HEAD 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      Creating initial v1.0 commit..." -ForegroundColor Yellow
    git commit -m "feat: v1 - Pure Python AlphaZero chess engine

- chesseng.py: complete chess rules engine (move gen, validation, game state)
- BoardEncoder.py: board -> (18,8,8) tensor encoder
- NeuralNet.py: ResNet with policy + value heads (10 blocks, 128 channels)
- mcts.py: Monte Carlo Tree Search with PUCT and Dirichlet noise
- Train.py: self-play + A3C training pipeline"
}

# ── Step 4: Tag v1.0 ──────────────────────────────────────────────────────────
Write-Host "`n[4/6] Tagging v1.0..." -ForegroundColor Yellow
$tagExists = git tag -l "v1.0"
if ($tagExists -eq "v1.0") {
    Write-Host "      Tag v1.0 already exists, skipping." -ForegroundColor Green
} else {
    git tag -a v1.0 -m "v1.0 - Pure Python AlphaZero engine (chesseng + mcts + NeuralNet + Train)"
    Write-Host "      Created tag v1.0" -ForegroundColor Green
}

# ── Step 5: Stage v2 files ────────────────────────────────────────────────────
Write-Host "`n[5/6] Staging v2 files (C++ core + hosting)..." -ForegroundColor Yellow
git add Chess/cpp/
git add Chess/setup.py
git add Chess/CMakeLists.txt
git add Chess/app.py
git add Chess/index.html
git add Chess/Dockerfile
git add Chess/docker-compose.yml
git add Chess/.dockerignore
git add Chess/.gitignore
git add Chess/.env.example
git add render.yaml
git add railway.toml

# Only commit if there are staged changes
$statusOutput = git status --porcelain
if ($statusOutput) {
    git commit -m "feat: v2 - C++ accelerated core + FastAPI backend

- cpp/: pybind11 C++ engine (chess_engine, board_encoder, mcts, self_play)
- app.py: FastAPI server with /get_move REST endpoint
- index.html: updated web UI (talks to FastAPI backend)
- Dockerfile + docker-compose.yml: container build and orchestration
- render.yaml + railway.toml: one-click cloud deploy configs
- .env.example: environment variable template

Falls back to pure-Python stack automatically if C++ extension not built."
} else {
    Write-Host "      No new v2 changes to commit." -ForegroundColor Green
}

# ── Step 6: Push ──────────────────────────────────────────────────────────────
Write-Host "`n[6/6] Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main
git push origin v1.0

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host ""
Write-Host "  v1 tag : https://github.com/AnshumanJ28/AlphaZ0/tree/v1.0" -ForegroundColor Cyan
Write-Host "  v2 main: https://github.com/AnshumanJ28/AlphaZ0/tree/main" -ForegroundColor Cyan
Write-Host "  README : https://github.com/AnshumanJ28/AlphaZ0#readme" -ForegroundColor Cyan
Write-Host ""
