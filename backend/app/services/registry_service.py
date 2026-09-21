"""
Model Registry Service — Phase 1
Handles staging → production → archived state transitions.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExperimentRun, ModelVersion

logger = logging.getLogger(__name__)


async def get_production_model(db: AsyncSession) -> ModelVersion | None:
    """Return the currently active production model version."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.status == "production"))
    return result.scalars().first()


async def promote_to_production(
    db: AsyncSession,
    model_version_id: str,
    sla_tier: str = "standard",
) -> ModelVersion:
    """
    1. Archive any existing production model
    2. Set the given version to production
    3. Return the updated model version
    """
    # Archive current production model (if any)
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.status == "production")
        .values(status="archived", archived_at=datetime.now(timezone.utc))
    )

    # Promote target model
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.id == model_version_id)
        .values(
            status="production",
            sla_tier=sla_tier,
            deployed_at=datetime.now(timezone.utc),
        )
    )
    await db.flush()

    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_version_id))
    model = result.scalars().first()
    logger.info("Model %s promoted to production (sla_tier=%s)", model_version_id, sla_tier)
    return model


async def get_model_with_run(
    db: AsyncSession,
    model_version_id: str,
) -> tuple[ModelVersion | None, ExperimentRun | None]:
    """Fetch model version + its experiment run in one query."""
    mv_result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_version_id))
    mv = mv_result.scalars().first()
    if not mv:
        return None, None

    run_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == mv.run_id))
    run = run_result.scalars().first()
    return mv, run
