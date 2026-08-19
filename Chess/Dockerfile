# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AlphaZ0 for Render Free Tier
#
# Two-stage build:
#   builder  → compiles the C++ pybind11 extension (alphaz0_cpp)
#   runtime  → lean image, CPU-only PyTorch, runs FastAPI via uvicorn
#
# Render injects $PORT (always 10000 on free tier). The CMD reads it.
# Render builds from repo root; dockerContext is set to ./Chess in render.yaml.
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Build C++ extension
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Install build-time Python deps (layer-cached separately) ─────────────────
RUN pip install --no-cache-dir \
        torch==2.3.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        numpy>=1.24.0 \
        pybind11[global]>=2.11.0 \
        setuptools>=68.0.0 \
        wheel

# ── Copy source and compile C++ extension in-place ───────────────────────────
COPY . .
RUN python setup.py build_ext --inplace


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Lean runtime (everything Render actually runs)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AlphaZ0 Chess Engine"
LABEL org.opencontainers.image.source="https://github.com/AnshumanJ28/AlphaZ0"

# libgomp1 is required by PyTorch CPU kernels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Runtime Python deps ───────────────────────────────────────────────────────
# Install PyTorch CPU separately (different index URL), then the rest
RUN pip install --no-cache-dir \
        torch==2.3.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        numpy>=1.24.0 \
        fastapi>=0.110.0 \
        "uvicorn[standard]>=0.29.0" \
        pydantic>=2.0.0 \
    && pip cache purge

# ── Copy compiled app from builder ───────────────────────────────────────────
COPY --from=builder /build /app

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default MCTS sims — Render free tier has a 30s request timeout.
# 50 sims keeps bot response well under that on CPU.
# Override in Render dashboard: Environment → MCTS_SIMS
ENV MCTS_SIMS=50

# Render always injects PORT=10000 on free tier. Default here is a fallback.
ENV PORT=10000

EXPOSE 10000

# ── Non-root user (Render best practice) ─────────────────────────────────────
RUN useradd --create-home --no-log-init --shell /bin/bash appuser
USER appuser

# ── Start server ──────────────────────────────────────────────────────────────
# Render health checks /health — uvicorn must be up before the 90s grace period
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
