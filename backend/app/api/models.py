"""Models API — Phase 1: list, deploy, predict."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import AuditLog, ExperimentRun, ModelVersion
from app.schemas.model import DeployRequest, ModelListResponse, ModelVersionResponse, PredictRequest, PredictResponse
from app.services import registry_service, serving_service

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    status_filter: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all model versions, optionally filtered by status."""
    query = select(ModelVersion)
    count_q = select(func.count()).select_from(ModelVersion)
    if status_filter:
        query = query.where(ModelVersion.status == status_filter)
        count_q = count_q.where(ModelVersion.status == status_filter)
    query = query.order_by(ModelVersion.deployed_at.desc()).limit(limit).offset(offset)

    total = (await db.execute(count_q)).scalar_one()
    mvs = (await db.execute(query)).scalars().all()

    items = []
    for mv in mvs:
        run_res = await db.execute(select(ExperimentRun).where(ExperimentRun.id == mv.run_id))
        run = run_res.scalars().first()
        resp = ModelVersionResponse.model_validate(mv)
        if run:
            resp.model_type = run.model_type
            resp.metrics = run.metrics
            resp.dataset_id = run.dataset_id
        items.append(resp)

    return ModelListResponse(items=items, total=total)


@router.get("/{model_version_id}", response_model=ModelVersionResponse)
async def get_model(
    model_version_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get model version details."""
    mv, run = await registry_service.get_model_with_run(db, model_version_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    resp = ModelVersionResponse.model_validate(mv)
    if run:
        resp.model_type = run.model_type
        resp.metrics = run.metrics
        resp.dataset_id = run.dataset_id
    return resp


@router.post("/{model_version_id}/deploy", response_model=ModelVersionResponse)
async def deploy_model(
    model_version_id: str,
    payload: DeployRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Promote a staged model version to production. Archives any current production model."""
    mv, run = await registry_service.get_model_with_run(db, model_version_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    if mv.status == "archived":
        raise HTTPException(status_code=400, detail="Cannot deploy an archived model")

    promoted = await registry_service.promote_to_production(db, model_version_id, payload.sla_tier)

    # Invalidate serve cache so next predict reloads the new model
    serving_service.invalidate_cache(model_version_id)

    # Write audit log
    audit = AuditLog(
        id=str(uuid.uuid4()),
        actor_id=current_user.get("sub"),
        actor_email=current_user.get("email", ""),
        action="model.deploy",
        target_type="model_version",
        target_id=model_version_id,
        extra_metadata={"sla_tier": payload.sla_tier, "model_type": run.model_type if run else None},
        created_at=datetime.now(timezone.utc),
    )
    db.add(audit)
    await db.flush()

    resp = ModelVersionResponse.model_validate(promoted)
    if run:
        resp.model_type = run.model_type
        resp.metrics = run.metrics
        resp.dataset_id = run.dataset_id
    return resp


@router.post("/{model_version_id}/predict", response_model=PredictResponse)
async def predict(
    model_version_id: str,
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Run inference on the given model version.
    Logs every request/response to the predictions table.
    """
    mv, _ = await registry_service.get_model_with_run(db, model_version_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    if mv.status == "archived":
        raise HTTPException(status_code=400, detail="Model is archived — deploy a new version first")

    try:
        result = await serving_service.run_inference(db, model_version_id, payload.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    return PredictResponse(**result)


@router.get("/production/current", response_model=ModelVersionResponse)
async def get_production_model(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get the currently active production model."""
    mv = await registry_service.get_production_model(db)
    if not mv:
        raise HTTPException(status_code=404, detail="No production model deployed yet")
    _, run = await registry_service.get_model_with_run(db, mv.id)
    resp = ModelVersionResponse.model_validate(mv)
    if run:
        resp.model_type = run.model_type
        resp.metrics = run.metrics
        resp.dataset_id = run.dataset_id
    return resp
