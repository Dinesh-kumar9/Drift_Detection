"""Retraining API — trigger, list, approve/reject, promote (FR12–FR14)."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, require_role
from app.models import ExperimentRun, ModelVersion, RegisteredModel, RetrainJob
from app.schemas.retrain import RejectRequest, RetrainJobListResponse, RetrainJobResponse, TriggerRequest
from app.services import retrain_service
from app.workers.retrain_tasks import run_retrain_job

logger = logging.getLogger(__name__)
router = APIRouter()

engineer = require_role("engineer", "admin")
admin = require_role("admin")


async def _response(db: AsyncSession, job: RetrainJob) -> RetrainJobResponse:
    resp = RetrainJobResponse.model_validate(job)
    row = (
        await db.execute(
            select(RegisteredModel.name, ModelVersion.version_tag, ExperimentRun.model_type, ModelVersion.sla_tier)
            .select_from(ModelVersion)
            .join(ExperimentRun, ExperimentRun.id == ModelVersion.run_id)
            .outerjoin(RegisteredModel, RegisteredModel.id == ModelVersion.registered_model_id)
            .where(ModelVersion.id == job.model_version_id)
        )
    ).first()
    if row:
        resp.registered_model_name, resp.version_tag, resp.model_type, resp.sla_tier = row
    return resp


def _state_error(exc: Exception) -> HTTPException:
    if isinstance(exc, retrain_service.RetrainStateError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=404, detail=str(exc))


@router.get("/jobs", response_model=RetrainJobListResponse)
async def list_jobs(
    status_filter: str | None = None,
    model_version_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    query = select(RetrainJob)
    count_q = select(func.count()).select_from(RetrainJob)
    if status_filter:
        query = query.where(RetrainJob.status == status_filter)
        count_q = count_q.where(RetrainJob.status == status_filter)
    if model_version_id:
        query = query.where(RetrainJob.model_version_id == model_version_id)
        count_q = count_q.where(RetrainJob.model_version_id == model_version_id)
    jobs = (await db.execute(query.order_by(RetrainJob.created_at.desc()).limit(limit))).scalars().all()
    return RetrainJobListResponse(
        items=[await _response(db, j) for j in jobs], total=(await db.execute(count_q)).scalar_one()
    )


@router.get("/jobs/{job_id}", response_model=RetrainJobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    job = await db.get(RetrainJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Retrain job not found")
    return await _response(db, job)


@router.post("/{model_version_id}/trigger", response_model=RetrainJobResponse, status_code=status.HTTP_201_CREATED)
async def trigger(
    model_version_id: str,
    payload: TriggerRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(engineer),
):
    """
    Manually request a retrain. The cost-benefit gate is evaluated against the latest
    attribution report; if it says no, the job is recorded as rejected unless `force` is set.
    """
    from app.models import AttributionReport

    mv = await db.get(ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    report = (
        (
            await db.execute(
                select(AttributionReport)
                .where(AttributionReport.model_version_id == model_version_id)
                .order_by(AttributionReport.generated_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    root_cause, since = None, None
    if report:
        root_cause = report.root_cause_stage
        top = next((r for r in report.ranked_stages if r.get("stage") == root_cause), {})
        since = datetime.fromisoformat(top["onset"]) if top.get("onset") else report.window_start
    try:
        job = await retrain_service.request_retrain(
            db,
            model_version_id,
            "manual",
            current_user,
            root_cause=root_cause,
            data_since=since,
            attribution_report_id=report.id if report else None,
            note=payload.note,
            force=payload.force,
        )
    except retrain_service.RetrainConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _response(db, job)


@router.post("/jobs/{job_id}/approve", response_model=RetrainJobResponse)
async def approve(job_id: str, db: AsyncSession = Depends(get_db), current_user: dict = Depends(admin)):
    """Human gate 1: approve spending compute on retraining. Starts the retrain job."""
    try:
        job = await retrain_service.approve(db, job_id, current_user)
    except ValueError as exc:
        raise _state_error(exc) from exc
    resp = await _response(db, job)
    await db.commit()  # the worker must see "approved"
    try:
        run_retrain_job.apply_async(kwargs={"job_id": job_id}, queue="retrain")
    except Exception as exc:
        logger.exception("Could not dispatch retrain job %s", job_id)
        raise HTTPException(status_code=503, detail="Retrain queue unavailable — approve again shortly") from exc
    return resp


@router.post("/jobs/{job_id}/reject", response_model=RetrainJobResponse)
async def reject(
    job_id: str, payload: RejectRequest, db: AsyncSession = Depends(get_db), current_user: dict = Depends(admin)
):
    """Reject a pending request, or decline a trained candidate."""
    try:
        job = await retrain_service.reject(db, job_id, current_user, payload.reason)
    except ValueError as exc:
        raise _state_error(exc) from exc
    return await _response(db, job)


@router.post("/jobs/{job_id}/promote", response_model=RetrainJobResponse)
async def promote(job_id: str, db: AsyncSession = Depends(get_db), current_user: dict = Depends(admin)):
    """Human gate 2: promote the retrained candidate to production."""
    try:
        job = await retrain_service.promote(db, job_id, current_user)
    except ValueError as exc:
        raise _state_error(exc) from exc
    return await _response(db, job)
