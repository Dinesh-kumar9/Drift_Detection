"""
Drift Event Persistence

Stores detected drift events in the existing DriftEvent table.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DriftEvent


async def save_drift_event(
    db: AsyncSession,
    model_version_id: str,
    drift_result: dict[str, Any],
) -> DriftEvent:
    """
    Persist one detected drift result as a DriftEvent.

    Only call this when drift_result["drift_detected"] is True.
    """

    event = DriftEvent(
        model_version_id=model_version_id,
        stage=drift_result["stage"],
        drift_score=drift_result["drift_score"],
        detector_type="ks_test",
        feature_name=drift_result.get("feature_name"),
        threshold_used=drift_result.get("threshold"),
    )

    db.add(event)
    await db.flush()

    return event
