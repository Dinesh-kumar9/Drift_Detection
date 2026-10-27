"""Experiments API — trigger training, poll status, compare metrics."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Dataset, ExperimentGroup, ExperimentRun, ModelVersion, RegisteredModel
from app.schemas.experiment import (
    ComparisonResponse,
    ExperimentCreate,
    ExperimentGroupListResponse,
    ExperimentGroupResponse,
    ExperimentListResponse,
    ExperimentResponse,
    GroupStatusResponse,
    MetricSet,
    RunStatus,
    TrainResponse,
)
from app.services import registry_service
from app.services.preprocessing_service import detect_task_type_from_schema
from app.services.training_service import PRIMARY_METRIC, get_catalogue
from app.workers.training_tasks import run_training_job

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train", status_code=status.HTTP_202_ACCEPTED, response_model=TrainResponse)
async def trigger_training(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Kick off parallel training of the model catalogue on the given dataset.
    Returns immediately — actual training runs asynchronously in Celery.
    """
    dataset = await db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    schema = dataset.schema_json or {}
    if payload.target_column not in schema:
        raise HTTPException(status_code=422, detail=f"Column '{payload.target_column}' not found in dataset")

    # Resolve the task type now (from the stored schema) so we know which model catalogue applies
    task_type = (
        detect_task_type_from_schema(schema[payload.target_column])
        if payload.task_type == "auto"
        else payload.task_type
    )
    catalogue = get_catalogue(task_type)
    model_names = payload.models or list(catalogue.keys())
    unknown = [m for m in model_names if m not in catalogue]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown {task_type} model(s): {unknown}. Available: {list(catalogue.keys())}",
        )

    model_name = payload.model_name or registry_service.default_model_name(dataset.name, payload.target_column)
    try:
        registered = await registry_service.get_or_create_registered_model(
            db,
            name=model_name,
            target_column=payload.target_column,
            task_type=task_type,
            dataset_id=dataset.id,
            created_by=current_user.get("sub"),
        )
    except registry_service.RegistryConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if registered.task_type != task_type:
        raise HTTPException(
            status_code=409,
            detail=f"Registered model '{registered.name}' is a {registered.task_type} model, not {task_type}",
        )

    now = datetime.now(timezone.utc)
    group = ExperimentGroup(
        dataset_id=dataset.id,
        registered_model_id=registered.id,
        target_column=payload.target_column,
        task_type=payload.task_type,
        resolved_task_type=task_type,
        requested_models=model_names,
        test_size=payload.test_size,
        random_state=payload.random_state,
        status="queued",
        created_by=current_user.get("sub"),
        created_at=now,
    )
    db.add(group)
    await db.flush()

    runs = [
        ExperimentRun(dataset_id=dataset.id, group_id=group.id, model_type=name, status="queued", created_at=now)
        for name in model_names
    ]
    db.add_all(runs)
    # Commit before dispatching — otherwise a fast worker can start before these rows are visible
    await db.commit()

    try:
        run_training_job.apply_async(kwargs={"experiment_group_id": group.id}, queue="training")
    except Exception as exc:  # broker unreachable — don't leave the group "queued" forever
        logger.exception("Could not dispatch training job for group %s", group.id)
        error = f"Could not reach the task queue: {type(exc).__name__}"
        group.status = "failed"
        group.error = error
        for run in runs:
            run.status = "failed"
            run.error = error
        await db.commit()
        raise HTTPException(status_code=503, detail="Training queue unavailable — try again shortly") from exc

    return TrainResponse(
        experiment_group_id=group.id,
        dataset_id=dataset.id,
        registered_model_id=registered.id,
        registered_model_name=registered.name,
        target_column=payload.target_column,
        task_type=task_type,
        model_run_ids=[r.id for r in runs],
        status="queued",
        message="Training job dispatched — poll GET /experiments/{id}/status",
    )


@router.get("/groups", response_model=ExperimentGroupListResponse)
async def list_groups(
    dataset_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List training requests (experiment groups), newest first."""
    n_models = (
        select(func.count(ExperimentRun.id))
        .where(ExperimentRun.group_id == ExperimentGroup.id)
        .correlate(ExperimentGroup)
        .scalar_subquery()
    )
    query = (
        select(ExperimentGroup, Dataset.name, RegisteredModel.name, n_models)
        .join(Dataset, Dataset.id == ExperimentGroup.dataset_id)
        .outerjoin(RegisteredModel, RegisteredModel.id == ExperimentGroup.registered_model_id)
    )
    count_q = select(func.count()).select_from(ExperimentGroup)
    if dataset_id:
        query = query.where(ExperimentGroup.dataset_id == dataset_id)
        count_q = count_q.where(ExperimentGroup.dataset_id == dataset_id)

    rows = (await db.execute(query.order_by(ExperimentGroup.created_at.desc()).limit(limit).offset(offset))).all()
    total = (await db.execute(count_q)).scalar_one()
    return ExperimentGroupListResponse(
        items=[
            ExperimentGroupResponse(
                id=g.id,
                dataset_id=g.dataset_id,
                dataset_name=ds_name,
                registered_model_id=g.registered_model_id,
                registered_model_name=rm_name,
                target_column=g.target_column,
                task_type=g.task_type,
                resolved_task_type=g.resolved_task_type,
                status=g.status,
                error=g.error,
                n_models=count,
                created_at=g.created_at,
                completed_at=g.completed_at,
            )
            for g, ds_name, rm_name, count in rows
        ],
        total=total,
    )


async def _get_group_runs(db: AsyncSession, experiment_group_id: str) -> tuple[ExperimentGroup, list[ExperimentRun]]:
    group = await db.get(ExperimentGroup, experiment_group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Experiment group not found")
    runs = (
        (
            await db.execute(
                select(ExperimentRun)
                .where(ExperimentRun.group_id == experiment_group_id)
                .order_by(ExperimentRun.model_type)
            )
        )
        .scalars()
        .all()
    )
    return group, list(runs)


@router.get("/{experiment_group_id}/status", response_model=GroupStatusResponse)
async def get_training_status(
    experiment_group_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Poll training status for an experiment group."""
    group, runs = await _get_group_runs(db, experiment_group_id)
    return GroupStatusResponse(
        experiment_group_id=group.id,
        overall_status=group.status,
        error=group.error,
        models=[RunStatus(model_type=r.model_type, run_id=r.id, status=r.status, error=r.error) for r in runs],
    )


@router.get("/{experiment_group_id}/compare", response_model=ComparisonResponse)
async def compare_models(
    experiment_group_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Return side-by-side metrics comparison for all models in the experiment group."""
    group, runs = await _get_group_runs(db, experiment_group_id)
    task_type = group.resolved_task_type or "classification"

    versions = (
        (await db.execute(select(ModelVersion).where(ModelVersion.run_id.in_([r.id for r in runs])))).scalars().all()
    )
    version_by_run = {mv.run_id: mv for mv in versions}

    model_metrics = []
    for r in runs:
        m = r.metrics or {}
        mv = version_by_run.get(r.id)
        model_metrics.append(
            MetricSet(
                model_type=r.model_type,
                run_id=r.id,
                model_version_id=mv.id if mv else None,
                version_tag=mv.version_tag if mv else None,
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
                model_size_kb=m.get("model_size_kb"),
                status=r.status,
                error=r.error,
            )
        )

    best_metric = PRIMARY_METRIC.get(task_type, "f1")
    completed = [r for r in runs if r.status == "completed" and r.metrics and r.metrics.get(best_metric) is not None]
    best = max(completed, key=lambda r: r.metrics[best_metric]).id if completed else None

    return ComparisonResponse(
        experiment_group_id=group.id,
        dataset_id=group.dataset_id,
        registered_model_id=group.registered_model_id,
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
    """List individual experiment runs, optionally filtered by dataset."""
    query = select(ExperimentRun)
    count_q = select(func.count()).select_from(ExperimentRun)
    if dataset_id:
        query = query.where(ExperimentRun.dataset_id == dataset_id)
        count_q = count_q.where(ExperimentRun.dataset_id == dataset_id)
    query = query.order_by(ExperimentRun.created_at.desc()).limit(limit).offset(offset)

    total = (await db.execute(count_q)).scalar_one()
    runs = (await db.execute(query)).scalars().all()

    return ExperimentListResponse(
        items=[ExperimentResponse.model_validate(r) for r in runs],
        total=total,
    )
