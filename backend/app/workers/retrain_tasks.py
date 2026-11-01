"""Celery task: execute an approved retrain job (build data → train candidate → evaluate)."""

import asyncio

from app.services.retrain_service import execute_retrain_job
from app.workers.celery_app import celery_app
from app.workers.db import task_session_factory


@celery_app.task(
    name="app.workers.retrain_tasks.run_retrain_job",
    queue="retrain",
)
def run_retrain_job(job_id: str):
    async def _main():
        async with task_session_factory() as session_factory:
            return await execute_retrain_job(job_id, session_factory)

    out = asyncio.run(_main())
    return {k: out.get(k) for k in ("status", "job_id", "candidate", "production_error", "candidate_error")}
