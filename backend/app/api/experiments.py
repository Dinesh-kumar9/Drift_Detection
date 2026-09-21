"""Experiments API — Phase 1: trigger training, compare metrics."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Dataset, ExperimentRun
from app.schemas.experiment import (
    ComparisonResponse,
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    MetricSet,
)
from app.workers.training_tasks import run_training_job

router = APIRouter()


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Kick off parallel training of 3 models on the given dataset.
    Returns immediately — actual training runs asynchronously in Celery.
    """
    # Validate dataset exists
    ds_result = await db.execute(select(Dataset).where(Dataset.id == payload.dataset_id))
    dataset = ds_result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create a shared group ID for this batch of runs
    experiment_group_id = str(uuid.uuid4())

    # Determine models to train
    from app.services.training_service import CLASSIFICATION_MODELS, REGRESSION_MODELS

    if payload.task_type in ("classification", "auto"):
        model_names = list(CLASSIFICATION_MODELS.keys())
    else:
        model_names = list(REGRESSION_MODELS.keys())

    # Create ExperimentRun records (one per model) in "queued" state
    run_ids = []
    for model_name in model_names:
        run = ExperimentRun(
            id=str(uuid.uuid4()),
            dataset_id=payload.dataset_id,
            model_type=model_name,
            params={"group_id": experiment_group_id, "target_column": payload.target_column},
            status="queued",
            created_at=datetime.now(timezone.utc),
        )
        db.add(run)
        run_ids.append(run.id)

    # We use the group_id as the primary task reference
    primary_run = ExperimentRun(
        id=experiment_group_id,
        dataset_id=payload.dataset_id,
        model_type="__group__",
        params={
            "group_id": experiment_group_id,
            "target_column": payload.target_column,
            "task_type": payload.task_type,
            "model_run_ids": run_ids,
        },
        status="queued",
        created_at=datetime.now(timezone.utc),
    )
    db.add(primary_run)
    await db.flush()

    # Dispatch Celery task
    run_training_job.apply_async(
        kwargs=dict(
            experiment_group_id=experiment_group_id,
            dataset_id=payload.dataset_id,
            target_column=payload.target_column,
            task_type=payload.task_type,
            test_size=payload.test_size,
            random_state=payload.random_state,
        ),
        queue="training",
    )

    return {
        "experiment_group_id": experiment_group_id,
        "dataset_id": payload.dataset_id,
        "target_column": payload.target_column,
        "model_run_ids": run_ids,
        "status": "queued",
        "message": "Training job dispatched to Celery — poll GET /experiments/{id} for status",
    }


@router.get("/{experiment_group_id}/status")
async def get_training_status(
    experiment_group_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Poll training status for an experiment group."""
    result = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.params.op("->>")(("group_id",)) == experiment_group_id,
            ExperimentRun.model_type != "__group__",
        )
    )
    runs = result.scalars().all()
    if not runs:
        raise HTTPException(status_code=404, detail="Experiment group not found")

    statuses = [r.status for r in runs]
    overall = (
        "completed"
        if all(s == "completed" for s in statuses)
        else (
            "failed"
            if any(s == "failed" for s in statuses)
            else "running" if any(s == "running" for s in statuses) else "queued"
        )
    )

    return {
        "experiment_group_id": experiment_group_id,
        "overall_status": overall,
        "models": [{"model_type": r.model_type, "run_id": r.id, "status": r.status} for r in runs],
    }


@router.get("/{experiment_group_id}/compare", response_model=ComparisonResponse)
async def compare_models(
    experiment_group_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Return side-by-side metrics comparison for all models in the experiment group."""
    result = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.params.op("->>")(("group_id",)) == experiment_group_id,
            ExperimentRun.model_type != "__group__",
        )
    )
    runs = result.scalars().all()
    if not runs:
        raise HTTPException(status_code=404, detail="Experiment group not found")

    # Detect task type from first completed run
    task_type = "classification"
    dataset_id = runs[0].dataset_id if runs else ""
    for r in runs:
        if r.metrics and "mae" in r.metrics:
            task_type = "regression"
            break

    # Build MetricSet list
    model_metrics = []
    for r in runs:
        m = r.metrics or {}
        model_metrics.append(
            MetricSet(
                model_type=r.model_type,
                run_id=r.id,
                accuracy=m.get("accuracy"),
                f1=m.get("f1"),
                precision=m.get("precision"),
                recall=m.get("recall"),
                roc_auc=m.get("roc_auc"),
                mae=m.get("mae"),
                rmse=m.get("rmse"),
                r2=m.get("r2"),
                latency_ms=m.get("latency_ms"),
                training_time_s=m.get("training_time_s"),
                status=r.status,
            )
        )

    # Pick best
    best = None
    best_metric = "f1" if task_type == "classification" else "r2"
    completed = [r for r in runs if r.status == "completed" and r.metrics]
    if completed:
        best_run = max(completed, key=lambda r: r.metrics.get(best_metric, -999))
        best = best_run.id

    return ComparisonResponse(
        experiment_group_id=experiment_group_id,
        dataset_id=dataset_id,
        task_type=task_type,
        models=model_metrics,
        best_model_run_id=best,
        best_metric=best_metric,
    )


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    dataset_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List experiment runs, optionally filtered by dataset."""
    query = select(ExperimentRun).where(ExperimentRun.model_type != "__group__")
    if dataset_id:
        query = query.where(ExperimentRun.dataset_id == dataset_id)
    query = query.order_by(ExperimentRun.created_at.desc()).limit(limit).offset(offset)

    count_q = select(func.count()).select_from(ExperimentRun).where(ExperimentRun.model_type != "__group__")
    if dataset_id:
        count_q = count_q.where(ExperimentRun.dataset_id == dataset_id)

    total = (await db.execute(count_q)).scalar_one()
    runs = (await db.execute(query)).scalars().all()

    return ExperimentListResponse(
        items=[ExperimentResponse.model_validate(r) for r in runs],
        total=total,
    )
