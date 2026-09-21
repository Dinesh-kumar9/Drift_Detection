"""Phase 3 stub — retrain Celery tasks."""
from app.workers.celery_app import celery_app


@celery_app.task(
    name="app.workers.retrain_tasks.run_retrain_job",
    queue="retrain",
)
def run_retrain_job(job_id: str):
    """Phase 3: execute approved retrain job. Stub."""
    return {"status": "stub", "phase": 3, "job_id": job_id}
