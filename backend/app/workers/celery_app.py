"""
Celery application instance.
Configured with Redis as both broker and result backend.
Workers are started separately: `celery -A app.workers.celery_app worker`
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "mlops_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.training_tasks",   # Phase 1
        "app.workers.drift_tasks",      # Phase 2
        "app.workers.retrain_tasks",    # Phase 3
    ],
)

celery_app.conf.update(
    # ─── Serialisation ────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ─── Task behaviour ───────────────────────────────────────────────────────
    task_track_started=True,
    task_acks_late=True,            # acknowledge only after task completes
    task_reject_on_worker_lost=True,
    task_soft_time_limit=3600,      # 1h soft limit for training jobs
    task_time_limit=7200,           # 2h hard limit

    # ─── Result expiry ────────────────────────────────────────────────────────
    result_expires=86400,           # keep results for 24h

    # ─── Beat schedule (Phase 2 drift checks) ─────────────────────────────────
    beat_schedule={
        "periodic-drift-check": {
            "task": "app.workers.drift_tasks.run_drift_check_all_models",
            "schedule": settings.DRIFT_CHECK_INTERVAL_SECONDS,
        }
    },

    # ─── Routing ──────────────────────────────────────────────────────────────
    task_routes={
        "app.workers.training_tasks.*": {"queue": "training"},
        "app.workers.drift_tasks.*": {"queue": "drift"},
        "app.workers.retrain_tasks.*": {"queue": "retrain"},
    },
)
