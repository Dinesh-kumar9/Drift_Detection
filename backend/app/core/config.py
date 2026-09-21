"""
Central configuration using Pydantic BaseSettings.
All values are read from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


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

    # ─── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://mlops:mlops@localhost:5432/mlops_db"

    # ─── Redis / Celery ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ─── Object Storage (MinIO / S3) ──────────────────────────────────────────
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"   # None = use real AWS S3
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
    DRIFT_KS_P_VALUE_THRESHOLD: float = 0.05   # KS test significance level
    DRIFT_ADWIN_DELTA: float = 0.002           # ADWIN sensitivity
    DRIFT_CHECK_INTERVAL_SECONDS: int = 300    # Celery Beat default (5 min)

    # ─── Alerting ─────────────────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: Optional[str] = None   # Set to enable Slack alerts
    ALERT_EMAIL_FROM: Optional[str] = None
    ALERT_EMAIL_TO: Optional[str] = None

    # ─── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — import this everywhere."""
    return Settings()


settings = get_settings()
