"""
Celery training task — Phase 1
Orchestrates: load dataset → preprocess → train all models → save to DB.
Runs asynchronously in the `training` queue.
"""

import asyncio
import io
import logging
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select, update

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models import Dataset, ExperimentRun, ModelVersion
from app.services import preprocessing_service, storage_service, training_service
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run a coroutine synchronously inside a Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="app.workers.training_tasks.run_training_job",
    bind=True,
    queue="training",
    max_retries=1,
)
def run_training_job(
    self,
    experiment_group_id: str,
    dataset_id: str,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Main Celery task. Steps:
    1. Fetch dataset from MinIO
    2. Preprocess (impute, encode, scale)
    3. Train 3 models in parallel
    4. Save ExperimentRun + ModelVersion records to DB
    """
    logger.info("Training job started: group=%s dataset=%s", experiment_group_id, dataset_id)

    async def _execute():
        async with AsyncSessionLocal() as db:
            # ── 1. Fetch dataset ────────────────────────────────────────────
            result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalars().first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")

            # Update all runs in this group to "running"
            await db.execute(
                update(ExperimentRun).where(ExperimentRun.id == experiment_group_id).values(status="running")
            )
            await db.commit()

            # Load raw CSV from MinIO
            bucket, key = dataset.s3_path.replace("s3://", "").split("/", 1)
            raw_bytes = storage_service.download_bytes(bucket, key)
            df = pd.read_csv(io.BytesIO(raw_bytes))
            logger.info("Loaded dataset: %d rows × %d cols", len(df), len(df.columns))

            # ── 2. Preprocess ───────────────────────────────────────────────
            X, y, config, transformer_obj = preprocessing_service.build_preprocessing_pipeline(df, target_column)
            actual_task_type = config["task_type"] if task_type == "auto" else task_type

            # Save preprocessed data to MinIO
            preprocessing_service.save_preprocessed(X, y, config, transformer_obj, dataset_id, experiment_group_id)

            # ── 3. Train all models ─────────────────────────────────────────
            results = training_service.train_all_models(
                X,
                y,
                actual_task_type,
                dataset_id,
                experiment_group_id,
                test_size=test_size,
                random_state=random_state,
            )

            # ── 4. Persist to DB ────────────────────────────────────────────
            completed_at = datetime.now(timezone.utc)
            for res in results:
                # Find or create the ExperimentRun for this model
                run_result = await db.execute(
                    select(ExperimentRun).where(
                        ExperimentRun.dataset_id == dataset_id,
                        ExperimentRun.model_type == res["model_name"],
                        ExperimentRun.params.op("->>")(("group_id",)) == experiment_group_id,
                    )
                )
                run = run_result.scalars().first()
                if run:
                    run.status = res["status"]
                    run.metrics = res["metrics"]
                    run.artifact_s3_path = res.get("artifact_s3_path")
                    run.preprocessing_config = config
                    run.completed_at = completed_at
                    # Create a ModelVersion record
                    mv = ModelVersion(
                        id=str(uuid.uuid4()),
                        run_id=run.id,
                        version_tag=f"v1.0.0",
                        status="staging",
                        sla_tier="standard",
                    )
                    db.add(mv)

            await db.commit()
            logger.info("Training job complete for group %s", experiment_group_id)
            return {"status": "completed", "group_id": experiment_group_id}

    return _run_async(_execute())
