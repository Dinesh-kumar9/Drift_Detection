"""Models API — registry, deploy, predict."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import AuditLog, Dataset, ExperimentRun, ModelVersion, RegisteredModel
from app.schemas.model import (
    DeployRequest,
    ModelListResponse,
    ModelVersionResponse,
    PredictRequest,
    PredictResponse,
    RegisteredModelListResponse,
    RegisteredModelResponse,
    SampleInputResponse,
)
from app.services import registry_service, serving_service
from app.services.model_bundle import IncompatibleArtifactError, SchemaValidationError

router = APIRouter()


def _version_response(
    mv: ModelVersion, run: ExperimentRun | None, registered: RegisteredModel | None
) -> ModelVersionResponse:
    resp = ModelVersionResponse.model_validate(mv)
    if run is not None:
        config = run.preprocessing_config or {}
        resp.model_type = run.model_type
        resp.metrics = run.metrics
        resp.dataset_id = run.dataset_id
        resp.task_type = config.get("task_type")
        resp.target_column = config.get("target_col")
        if "numeric_cols" in config:
            resp.input_columns = [*config.get("numeric_cols", []), *config.get("cat_cols", [])]
    if registered is not None:
        resp.registered_model_name = registered.name
    return resp


def _versions_query():
    return (
        select(ModelVersion, ExperimentRun, RegisteredModel)
        .join(ExperimentRun, ModelVersion.run_id == ExperimentRun.id)
        .outerjoin(RegisteredModel, ModelVersion.registered_model_id == RegisteredModel.id)
    )


async def _get_version_or_404(db: AsyncSession, model_version_id: str):
    row = (await db.execute(_versions_query().where(ModelVersion.id == model_version_id))).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model version not found")
    return row


@router.get("/registered", response_model=RegisteredModelListResponse)
async def list_registered_models(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """All model lineages, each with its current production version (if any)."""
    prod = aliased(ModelVersion)
    prod_run = aliased(ExperimentRun)
    version_count = (
        select(func.count(ModelVersion.id))
        .where(ModelVersion.registered_model_id == RegisteredModel.id)
        .correlate(RegisteredModel)
        .scalar_subquery()
    )
    rows = (
        await db.execute(
            select(RegisteredModel, version_count, prod, prod_run.model_type)
            .outerjoin(prod, (prod.registered_model_id == RegisteredModel.id) & (prod.status == "production"))
            .outerjoin(prod_run, prod_run.id == prod.run_id)
            .order_by(RegisteredModel.created_at.desc())
        )
    ).all()

    items = [
        RegisteredModelResponse(
            id=rm.id,
            name=rm.name,
            target_column=rm.target_column,
            task_type=rm.task_type,
            dataset_id=rm.dataset_id,
            created_at=rm.created_at,
            version_count=count,
            production_version_id=pv.id if pv else None,
            production_version_tag=pv.version_tag if pv else None,
            production_model_type=model_type,
            production_sla_tier=pv.sla_tier if pv else None,
            production_deployed_at=pv.deployed_at if pv else None,
        )
        for rm, count, pv, model_type in rows
    ]
    return RegisteredModelListResponse(items=items, total=len(items))


@router.get("/", response_model=ModelListResponse)
async def list_models(
    status_filter: str | None = None,
    registered_model_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List model versions, optionally filtered by status and/or lineage."""
    query = _versions_query()
    count_q = select(func.count()).select_from(ModelVersion)
    if status_filter:
        query = query.where(ModelVersion.status == status_filter)
        count_q = count_q.where(ModelVersion.status == status_filter)
    if registered_model_id:
        query = query.where(ModelVersion.registered_model_id == registered_model_id)
        count_q = count_q.where(ModelVersion.registered_model_id == registered_model_id)
    query = query.order_by(
        ModelVersion.deployed_at.desc().nulls_last(), ExperimentRun.created_at.desc(), ModelVersion.version.desc()
    )

    total = (await db.execute(count_q)).scalar_one()
    rows = (await db.execute(query.limit(limit).offset(offset))).all()
    return ModelListResponse(items=[_version_response(mv, run, rm) for mv, run, rm in rows], total=total)


@router.get("/{model_version_id}", response_model=ModelVersionResponse)
async def get_model(
    model_version_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get model version details."""
    mv, run, rm = await _get_version_or_404(db, model_version_id)
    return _version_response(mv, run, rm)


@router.post("/{model_version_id}/deploy", response_model=ModelVersionResponse)
async def deploy_model(
    model_version_id: str,
    payload: DeployRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Promote a model version to production. Archives the lineage's current production version."""
    mv, run, rm = await _get_version_or_404(db, model_version_id)
    if mv.status == "archived":
        raise HTTPException(status_code=400, detail="Cannot deploy an archived model")
    if not run.artifact_s3_path:
        raise HTTPException(status_code=409, detail="Model has no trained artifact")

    promoted, previous = await registry_service.promote_to_production(db, model_version_id, payload.sla_tier)
    serving_service.invalidate_cache(model_version_id)

    db.add(
        AuditLog(
            actor_id=current_user.get("sub"),
            actor_email=current_user.get("email"),
            action="model.deploy",
            target_type="model_version",
            target_id=model_version_id,
            extra_metadata={
                "sla_tier": payload.sla_tier,
                "model_type": run.model_type,
                "registered_model": rm.name if rm else None,
                "version_tag": mv.version_tag,
                "replaced_version_id": previous.id if previous else None,
            },
            created_at=datetime.now(timezone.utc),
        )
    )
    await db.flush()
    return _version_response(promoted, run, rm)


@router.post("/{model_version_id}/predict", response_model=PredictResponse)
async def predict(
    model_version_id: str,
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Run inference on the given model version.
    Logs every request/response (raw input, transformed features, output) to the predictions table.
    """
    mv = await db.get(ModelVersion, model_version_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    if mv.status == "archived":
        raise HTTPException(status_code=400, detail="Model is archived — deploy a new version first")

    try:
        result = await serving_service.run_inference(db, model_version_id, payload.features)
    except SchemaValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "missing_columns": exc.missing_columns}
        ) from exc
    except (serving_service.ModelNotReadyError, IncompatibleArtifactError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return PredictResponse(**result)


@router.get("/{model_version_id}/sample-input", response_model=SampleInputResponse)
async def sample_input(
    model_version_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """A valid example request body for /predict, built from the training dataset's baseline stats."""
    mv, run, _ = await _get_version_or_404(db, model_version_id)
    try:
        bundle = await serving_service.load_bundle(db, model_version_id)
    except (serving_service.ModelNotReadyError, IncompatibleArtifactError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    dataset = await db.get(Dataset, run.dataset_id)
    return SampleInputResponse(
        model_version_id=mv.id,
        features=bundle.sample_input(dataset.baseline_stats if dataset else None),
    )
