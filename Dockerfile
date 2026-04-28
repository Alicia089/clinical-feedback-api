# ──────────────────────────────────────────────────────────────────────────────
# Clinical Feedback Triage API — Dockerfile
#
# Engineering decisions:
#
# Multi-stage build:
#   WHY: Separates the build environment from the runtime image.
#   The builder stage installs all build tools and compiles dependencies.
#   The runtime stage copies only what's needed. This typically reduces
#   final image size by 60-70% vs. a single-stage build.
#
# Base: python:3.11-slim (not python:3.11-full or ubuntu)
#   WHY: The -slim variant removes non-essential packages (compilers, man pages,
#   locales) while keeping pip and the Python runtime. Smaller image = faster
#   ECR push/pull and reduced attack surface.
#
# Non-root user (appuser):
#   WHY: Running as root in a container is a security risk. If a vulnerability
#   allows container escape, an attacker gains root on the host. Running as a
#   non-privileged user is a CIS Docker Benchmark requirement.
#
# .dockerignore is assumed to exclude: .git, __pycache__, *.pyc, .env,
#   model/fine_tuned (model is mounted at runtime or pulled from S3).
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (won't be in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install to a non-system prefix so we can copy the entire venv cleanly
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Copy fine-tuned model if available locally
# In production, MODEL_PATH env var points to an S3 path or HuggingFace Hub ID
# and the model is downloaded at container init via an entrypoint script.
# COPY model/fine_tuned ./model/fine_tuned

# Switch to non-root user
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=distilbert-base-uncased \
    MAX_TOKEN_LENGTH=256 \
    PORT=8080

EXPOSE 8080

# Health check — used by ECS and App Runner to validate container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Gunicorn with Uvicorn workers:
#   WHY Gunicorn over raw Uvicorn: Gunicorn manages multiple Uvicorn worker
#   processes, providing process-level fault isolation. If one worker crashes,
#   Gunicorn restarts it without taking down the API. This is critical for
#   production uptime.
#   Worker count formula: (2 × CPU_cores) + 1 is the standard heuristic.
#   ECS Fargate tasks with 2 vCPUs → 5 workers.
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
