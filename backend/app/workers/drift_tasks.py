"""Phase 2 stub — drift check Celery tasks."""
from app.workers.celery_app import celery_app


@celery_app.task(
    name="app.workers.drift_tasks.run_drift_check_all_models",
    queue="drift",
)
def run_drift_check_all_models():
    """Phase 2: run stage monitors for all production models. Stub."""
    return {"status": "stub", "phase": 2}
