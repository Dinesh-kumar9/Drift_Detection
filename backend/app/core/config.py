"""
Central configuration using Pydantic BaseSettings.
All values are read from environment variables / .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "MLOps Observability Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "local"  # local | staging | production
    GIT_COMMIT: Optional[str] = None  # injected at image build time — recorded on every run (reproducibility NFR)

    # ─── Bootstrap ────────────────────────────────────────────────────────────
    RUN_MIGRATIONS_ON_STARTUP: bool = True
    SEED_DEMO_USERS: bool = True  # only honoured when ENVIRONMENT == "local"

    # ─── Serving ──────────────────────────────────────────────────────────────
    MODEL_CACHE_SIZE: int = 32  # model bundles kept in memory per API process

    # ─── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://mlops:mlops@localhost:5432/mlops_db"

    # ─── Redis / Celery ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ─── Object Storage (MinIO / S3) ──────────────────────────────────────────
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"  # None = use real AWS S3
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_DATASETS: str = "mlops-datasets"
    S3_BUCKET_MODELS: str = "mlops-models"
    S3_REGION: str = "us-east-1"

    # ─── Auth / JWT ───────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-in-production-use-a-long-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ─── Drift Detection ──────────────────────────────────────────────────────
    DRIFT_KS_P_VALUE_THRESHOLD: float = 0.05  # family-wise significance per stage (Bonferroni-corrected)
    DRIFT_MIN_EFFECT: float = 0.1  # minimum effect size (KS D / total-variation distance) to call drift
    DRIFT_ADWIN_DELTA: float = 0.002  # ADWIN sensitivity
    DRIFT_CHECK_INTERVAL_SECONDS: int = 60  # Celery Beat dispatcher tick; per-model cadence comes from SLA tier
    DRIFT_WINDOW_MINUTES: int = 60  # how much recent traffic each check examines
    DRIFT_MIN_SAMPLES: int = 50  # skip the check below this many predictions in the window
    DRIFT_MAX_SAMPLES: int = 5000  # most recent predictions examined per check
    DRIFT_ONSET_BUCKETS: int = 6  # window is split into this many buckets to locate drift onset
    DRIFT_BASELINE_ROWS: int = 2000  # reference sample size stored per model at training time
    # SLA tier → seconds between checks (FR16)
    DRIFT_INTERVAL_CRITICAL: int = 60
    DRIFT_INTERVAL_STANDARD: int = 300
    DRIFT_INTERVAL_LOW: int = 1800
    ENABLE_FAULT_INJECTION: bool = True  # /observability/{id}/simulate — disable outside demo/staging

    # ─── Alerting ─────────────────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: Optional[str] = None  # Set to enable Slack alerts
    ALERT_COOLDOWN_MINUTES: int = 30  # suppress repeat alerts for the same model + root cause
    ALERT_EMAIL_FROM: Optional[str] = None
    ALERT_EMAIL_TO: Optional[str] = None

    # ─── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — import this everywhere."""
    return Settings()


settings = get_settings()
