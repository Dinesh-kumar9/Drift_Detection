"""
Model Registry Service
Registered models (lineages) own versions; each lineage has at most one
production version, so many models can be live side by side.
"""

import logging
import re
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExperimentRun, ModelVersion, RegisteredModel

logger = logging.getLogger(__name__)


class RegistryConflictError(ValueError):
    """A registered model with this name already predicts a different target."""


def default_model_name(dataset_name: str, target_column: str) -> str:
    """e.g. ("Credit Card Fraud.csv", "Class") → "credit-card-fraud-class"."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{dataset_name}-{target_column}".lower()).strip("-")
    return slug[:200] or "model"


async def get_or_create_registered_model(
    db: AsyncSession,
    name: str,
    target_column: str,
    task_type: str,
    dataset_id: str | None = None,
    created_by: str | None = None,
) -> RegisteredModel:
    """Find the lineage by name, or create it. Rejects reuse of a name for a different target."""
    result = await db.execute(select(RegisteredModel).where(RegisteredModel.name == name))
    registered = result.scalars().first()
    if registered:
        if registered.target_column != target_column:
            raise RegistryConflictError(
                f"Registered model '{name}' predicts '{registered.target_column}', not '{target_column}'"
            )
        return registered

    registered = RegisteredModel(
        name=name,
        target_column=target_column,
        task_type=task_type,
        dataset_id=dataset_id,
        created_by=created_by,
        created_at=datetime.now(timezone.utc),
    )
    db.add(registered)
    await db.flush()
    logger.info("Registered new model lineage '%s'", name)
    return registered


async def next_version_number(db: AsyncSession, registered_model_id: str) -> int:
    """
    Next version number for a lineage. Locks the registered_models row so two
    training jobs finishing at once can't hand out the same number.
    """
    await db.execute(select(RegisteredModel.id).where(RegisteredModel.id == registered_model_id).with_for_update())
    result = await db.execute(
        select(func.max(ModelVersion.version)).where(ModelVersion.registered_model_id == registered_model_id)
    )
    return (result.scalar() or 0) + 1


async def get_production_version(db: AsyncSession, registered_model_id: str) -> ModelVersion | None:
    """Return the production version of one lineage, if any."""
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.registered_model_id == registered_model_id,
            ModelVersion.status == "production",
        )
    )
    return result.scalars().first()


async def promote_to_production(
    db: AsyncSession,
    model_version_id: str,
    sla_tier: str = "standard",
) -> tuple[ModelVersion, ModelVersion | None]:
    """
    1. Archive the current production version of the *same* lineage (if any)
    2. Set the given version to production
    Returns (promoted_version, previously_live_version_or_None).
    """
    target = (await db.execute(select(ModelVersion).where(ModelVersion.id == model_version_id))).scalars().first()
    if target is None:
        raise ValueError(f"Model version {model_version_id} not found")

    now = datetime.now(timezone.utc)
    lineage_filter = (
        ModelVersion.registered_model_id == target.registered_model_id
        if target.registered_model_id
        else ModelVersion.registered_model_id.is_(None)
    )
    previous = (
        (
            await db.execute(
                select(ModelVersion).where(
                    lineage_filter,
                    ModelVersion.status == "production",
                    ModelVersion.id != model_version_id,
                )
            )
        )
        .scalars()
        .first()
    )

    await db.execute(
        update(ModelVersion)
        .where(lineage_filter, ModelVersion.status == "production", ModelVersion.id != model_version_id)
        .values(status="archived", archived_at=now)
    )
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.id == model_version_id)
        .values(status="production", sla_tier=sla_tier, deployed_at=now, archived_at=None)
    )
    await db.flush()

    await db.refresh(target)
    if previous is not None:
        await db.refresh(previous)
    logger.info("Model %s promoted to production (sla_tier=%s)", model_version_id, sla_tier)
    return target, previous


async def get_model_with_run(
    db: AsyncSession,
    model_version_id: str,
) -> tuple[ModelVersion | None, ExperimentRun | None]:
    """Fetch model version + its experiment run in one query."""
    result = await db.execute(
        select(ModelVersion, ExperimentRun)
        .join(ExperimentRun, ModelVersion.run_id == ExperimentRun.id)
        .where(ModelVersion.id == model_version_id)
    )
    row = result.first()
    if not row:
        return None, None
    return row[0], row[1]
