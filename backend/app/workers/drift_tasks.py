"""
Scheduled drift checks.

Celery Beat ticks `run_drift_check_all_models` every DRIFT_CHECK_INTERVAL_SECONDS.
The dispatcher checks every production model whose SLA-tier interval has
elapsed since its last check (critical 1 min / standard 5 min / low 30 min by
default — FR16) and enqueues one `run_drift_check` task per due model.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models import DriftEvent, ExperimentRun, ModelVersion
from app.services.drift.service import run_drift_check
from app.workers.celery_app import celery_app
from app.workers.db import task_session_factory

logger = logging.getLogger(__name__)


def tier_interval(sla_tier: str | None) -> timedelta:
    seconds = {
        "critical": settings.DRIFT_INTERVAL_CRITICAL,
        "standard": settings.DRIFT_INTERVAL_STANDARD,
        "low": settings.DRIFT_INTERVAL_LOW,
    }.get(sla_tier or "standard", settings.DRIFT_INTERVAL_STANDARD)
    return timedelta(seconds=seconds)


async def due_model_versions(db: AsyncSession, now: datetime) -> list[str]:
    """
    Production versions whose last check is older than their SLA interval (or never checked).
    Versions without a drift baseline (trained before Phase 2) can't be checked and are skipped.
    """
    last_check = (
        select(DriftEvent.model_version_id, func.max(DriftEvent.detected_at).label("last"))
        .group_by(DriftEvent.model_version_id)
        .subquery()
    )
    rows = (
        await db.execute(
            select(ModelVersion.id, ModelVersion.sla_tier, last_check.c.last)
            .join(ExperimentRun, ExperimentRun.id == ModelVersion.run_id)
            .outerjoin(last_check, last_check.c.model_version_id == ModelVersion.id)
            .where(ModelVersion.status == "production", ExperimentRun.baseline_s3_path.is_not(None))
        )
    ).all()
    # small tolerance so a check that ran a few seconds late isn't skipped until the next tick
    return [mv_id for mv_id, tier, last in rows if last is None or now - last >= tier_interval(tier) * 0.9]


async def dispatch_due_checks(session_factory: async_sessionmaker[AsyncSession]) -> list[str]:
    async with session_factory() as db:
        due = await due_model_versions(db, datetime.now(timezone.utc))
    for mv_id in due:
        run_drift_check_task.apply_async(kwargs={"model_version_id": mv_id}, queue="drift")
    return due


async def execute_drift_check(model_version_id: str, session_factory: async_sessionmaker[AsyncSession]) -> dict:
    async with session_factory() as db:
        outcome = await run_drift_check(db, model_version_id)
        await db.commit()
    return outcome


@celery_app.task(name="app.workers.drift_tasks.run_drift_check_all_models", queue="drift")
def run_drift_check_all_models():
    """Beat entry point — enqueue a check for every production model that is due."""

    async def _main():
        async with task_session_factory() as session_factory:
            return await dispatch_due_checks(session_factory)

    due = asyncio.run(_main())
    logger.info("Drift dispatcher: %d model(s) due", len(due))
    return {"dispatched": due}


@celery_app.task(name="app.workers.drift_tasks.run_drift_check", queue="drift")
def run_drift_check_task(model_version_id: str):
    async def _main():
        async with task_session_factory() as session_factory:
            return await execute_drift_check(model_version_id, session_factory)

    outcome = asyncio.run(_main())
    return {k: outcome.get(k) for k in ("status", "drift_detected", "root_cause", "confidence", "n_samples")}
