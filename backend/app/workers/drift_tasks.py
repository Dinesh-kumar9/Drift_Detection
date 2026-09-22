"""
Phase 2 — Cross-Stage Drift Monitoring

Celery task for S1 ingestion-stage drift detection.
"""

import logging

import pandas as pd
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models import Dataset, ExperimentRun, Prediction
from app.services.drift_persistence import save_drift_event
from app.services.drift_service import detect_s1_drift
from app.services.registry_service import get_production_model
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.workers.drift_tasks.run_drift_check_all_models",
    queue="drift",
)
def run_drift_check_all_models(window_size: int = 100):
    """
    Run S1 drift detection for the current production model.

    The task is synchronous at the Celery boundary and executes the
    asynchronous database workflow using asyncio.run().
    """

    import asyncio

    return asyncio.run(_run_drift_check_all_models(window_size))


async def _run_drift_check_all_models(window_size: int = 100) -> dict:
    """
    Async implementation of the S1 drift monitoring workflow.
    """

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    async with AsyncSessionLocal() as db:

        # 1. Find current production model
        model_version = await get_production_model(db)

        if model_version is None:
            logger.info("No production model found. Skipping drift check.")
            return {
                "status": "skipped",
                "reason": "no_production_model",
            }

        # 2. Find the experiment run associated with the model
        run_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == model_version.run_id))
        experiment_run = run_result.scalars().first()

        if experiment_run is None:
            logger.warning(
                "No experiment run found for model %s",
                model_version.id,
            )
            return {
                "status": "skipped",
                "reason": "experiment_run_not_found",
                "model_version_id": model_version.id,
            }

        # 3. Fetch the training dataset and its baseline statistics
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == experiment_run.dataset_id))
        dataset = dataset_result.scalars().first()

        if dataset is None:
            logger.warning(
                "No dataset found for model %s",
                model_version.id,
            )
            return {
                "status": "skipped",
                "reason": "dataset_not_found",
                "model_version_id": model_version.id,
            }

        if not dataset.baseline_stats:
            logger.warning(
                "Dataset %s has no baseline statistics",
                dataset.id,
            )
            return {
                "status": "skipped",
                "reason": "baseline_stats_missing",
                "model_version_id": model_version.id,
            }

        # 4. Fetch recent production prediction inputs
        prediction_result = await db.execute(
            select(Prediction)
            .where(Prediction.model_version_id == model_version.id)
            .order_by(Prediction.predicted_at.desc())
            .limit(window_size)
        )

        predictions = prediction_result.scalars().all()

        if len(predictions) < 2:
            logger.info(
                "Not enough predictions for drift detection: %s",
                len(predictions),
            )
            return {
                "status": "skipped",
                "reason": "insufficient_predictions",
                "model_version_id": model_version.id,
                "prediction_count": len(predictions),
            }

        # 5. Convert stored input snapshots into a DataFrame
        current_df = pd.DataFrame([prediction.input_snapshot for prediction in predictions])

        if current_df.empty:
            return {
                "status": "skipped",
                "reason": "empty_prediction_data",
                "model_version_id": model_version.id,
            }

        # 6. Run S1 drift detection
        drift_results = detect_s1_drift(
            baseline_stats=dataset.baseline_stats,
            current_df=current_df,
        )

        # 7. Persist detected drift events
        persisted_events = 0

        for drift_result in drift_results:
            if not drift_result["drift_detected"]:
                continue

            await save_drift_event(
                db=db,
                model_version_id=model_version.id,
                drift_result=drift_result,
            )

            persisted_events += 1

        await db.commit()

        logger.info(
            "S1 drift check completed: model=%s predictions=%s " "features_checked=%s events_created=%s",
            model_version.id,
            len(predictions),
            len(drift_results),
            persisted_events,
        )

        return {
            "status": "completed",
            "model_version_id": model_version.id,
            "prediction_count": len(predictions),
            "features_checked": len(drift_results),
            "drift_events_created": persisted_events,
        }
