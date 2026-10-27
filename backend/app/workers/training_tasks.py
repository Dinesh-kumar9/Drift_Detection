"""
Celery training task.
Orchestrates: load dataset → preprocess → train all models → bundle + store
artifacts → register model versions. Runs asynchronously in the `training` queue.

All job parameters live on the `experiment_groups` row, so the task only needs
the group id. `execute_training_job` is the testable core; the Celery task is a
thin sync wrapper around it.
"""

import asyncio
import io
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models import Dataset, ExperimentGroup, ExperimentRun, ModelVersion, RegisteredModel
from app.services import preprocessing_service, registry_service, storage_service, training_service
from app.services.model_bundle import ModelBundle
from app.workers.celery_app import celery_app
from app.workers.db import task_session_factory

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@celery_app.task(
    name="app.workers.training_tasks.run_training_job",
    bind=True,
    queue="training",
)
def run_training_job(self, experiment_group_id: str):
    """Celery entry point — runs the async job in a fresh event loop with its own DB engine."""

    async def _main():
        async with task_session_factory() as session_factory:
            return await execute_training_job(experiment_group_id, session_factory)

    return asyncio.run(_main())


async def execute_training_job(
    experiment_group_id: str,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict:
    """
    1. Mark group + runs running
    2. Load the raw CSV and fit the preprocessing pipeline
    3. Train the requested models in parallel
    4. Bundle each successful model with its preprocessor, upload, create a ModelVersion
    Any exception marks the group and all unfinished runs as failed (then re-raises).
    """
    logger.info("Training job started: group=%s", experiment_group_id)

    async with session_factory() as db:
        group = await db.get(ExperimentGroup, experiment_group_id)
        if group is None:
            raise ValueError(f"Experiment group {experiment_group_id} not found")
        if group.status == "completed":
            # Redelivered message (acks_late) — never register the same models twice
            logger.warning("Group %s already completed; skipping duplicate delivery", experiment_group_id)
            return {"status": "completed", "group_id": experiment_group_id, "skipped": True}

        await db.execute(
            update(ExperimentRun).where(ExperimentRun.group_id == experiment_group_id).values(status="running")
        )
        group.status = "running"
        group.started_at = _now()
        await db.commit()

        try:
            summary = await _train_and_register(db, group)
        except Exception as exc:
            logger.exception("Training job failed: group=%s", experiment_group_id)
            await db.rollback()
            await _mark_failed(db, experiment_group_id, f"{type(exc).__name__}: {exc}")
            raise

    logger.info("Training job finished: group=%s summary=%s", experiment_group_id, summary)
    return summary


async def _train_and_register(db: AsyncSession, group: ExperimentGroup) -> dict:
    dataset = await db.get(Dataset, group.dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {group.dataset_id} not found")
    runs = (await db.execute(select(ExperimentRun).where(ExperimentRun.group_id == group.id))).scalars().all()

    # ── Load + preprocess ────────────────────────────────────────────────────
    raw_bytes = await asyncio.to_thread(storage_service.download_uri, dataset.s3_path)
    df = await asyncio.to_thread(pd.read_csv, io.BytesIO(raw_bytes))
    logger.info("Loaded dataset: %d rows × %d cols", len(df), len(df.columns))

    task_type = group.resolved_task_type or group.task_type
    X, y, config, transformer = await asyncio.to_thread(
        preprocessing_service.build_preprocessing_pipeline, df, group.target_column, task_type
    )
    task_type = config["task_type"]
    group.resolved_task_type = task_type

    registered = await db.get(RegisteredModel, group.registered_model_id) if group.registered_model_id else None
    if registered is not None:
        if registered.task_type == "auto":
            registered.task_type = task_type
        elif registered.task_type != task_type:
            raise ValueError(
                f"Registered model '{registered.name}' is a {registered.task_type} model; "
                f"this data resolves to {task_type}"
            )

    await asyncio.to_thread(preprocessing_service.save_preprocessed, X, y, config, dataset.id, group.id)

    # ── Train ────────────────────────────────────────────────────────────────
    results = await asyncio.to_thread(
        training_service.train_all_models,
        X,
        y,
        task_type,
        [r.model_type for r in runs],
        group.test_size or 0.2,
        group.random_state if group.random_state is not None else 42,
    )
    by_name = {r["model_name"]: r for r in results}

    # ── Bundle, upload, register ────────────────────────────────────────────
    version = await registry_service.next_version_number(db, registered.id) if registered else 1
    completed_at = _now()
    n_completed = 0
    for run in runs:
        res = by_name.get(run.model_type)
        run.preprocessing_config = config
        run.git_commit = settings.GIT_COMMIT
        run.completed_at = completed_at
        if res is None or res["status"] != "completed":
            run.status = "failed"
            run.error = (res or {}).get("error") or "Model was not trained"
            continue

        bundle = ModelBundle(
            model_type=run.model_type,
            task_type=task_type,
            target_column=group.target_column,
            numeric_cols=config["numeric_cols"],
            cat_cols=config["cat_cols"],
            feature_names=config["feature_names"],
            preprocessor=transformer["ct"],
            model=res["model"],
            label_encoder=transformer["le"],
        )
        data = await asyncio.to_thread(bundle.to_bytes)
        artifact_path = await asyncio.to_thread(
            storage_service.upload_bytes,
            settings.S3_BUCKET_MODELS,
            f"models/{dataset.id}/{group.id}/{run.id}/{run.model_type}.joblib",
            data,
            "application/octet-stream",
        )

        run.status = "completed"
        run.error = None
        run.params = res["params"]
        run.metrics = {**res["metrics"], "model_size_kb": round(len(data) / 1024, 1)}
        run.artifact_s3_path = artifact_path
        db.add(
            ModelVersion(
                run_id=run.id,
                registered_model_id=registered.id if registered else None,
                version=version,
                version_tag=f"v{version}",
                status="staging",
                sla_tier="standard",
            )
        )
        version += 1
        n_completed += 1

    group.status = "completed" if n_completed else "failed"
    group.error = None if n_completed else "All models failed to train — see individual run errors"
    group.completed_at = completed_at
    await db.commit()

    return {"status": group.status, "group_id": group.id, "completed_models": n_completed, "task_type": task_type}


async def _mark_failed(db: AsyncSession, group_id: str, error: str) -> None:
    now = _now()
    await db.execute(
        update(ExperimentRun)
        .where(ExperimentRun.group_id == group_id, ExperimentRun.status != "completed")
        .values(status="failed", error=error, completed_at=now)
    )
    await db.execute(
        update(ExperimentGroup)
        .where(ExperimentGroup.id == group_id)
        .values(status="failed", error=error, completed_at=now)
    )
    await db.commit()
