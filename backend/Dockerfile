# ─── Multi-stage Dockerfile ───────────────────────────────────────────────────
# Stage 1: base — shared Python environment
# Stage 2: api — runs the FastAPI server
# Stage 3: worker — runs Celery worker

FROM python:3.11-slim AS base

# Prevent .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (for asyncpg, cryptography, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install project dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install XGBoost separately without GPU/NCCL dependencies
RUN python -m pip install --no-cache-dir --no-deps xgboost==2.1.1

# Copy application source
COPY . .

# ─── API target ───────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Worker target ────────────────────────────────────────────────────────────
FROM base AS worker

CMD ["celery", "-A", "app.workers.celery_app", "worker", \
     "--loglevel=info", "--queues=training,drift,retrain", "--concurrency=2"]

# ─── Beat target (scheduler) ─────────────────────────────────────────────────
FROM base AS beat

CMD ["celery", "-A", "app.workers.celery_app", "beat", "--loglevel=info"]