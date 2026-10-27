"""Observability API — stage-wise drift status, root-cause attribution, checks, fault injection."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import AttributionReport, AuditLog, DriftEvent, ExperimentRun, ModelVersion, Prediction, RegisteredModel
from app.schemas.observability import (
    AttributionHistoryResponse,
    AttributionReportResponse,
    CheckSummary,
    DriftStatusResponse,
    OverviewItem,
    OverviewResponse,
    SeriesPoint,
    SimulateRequest,
    StageStatus,
)
from app.services import serving_service
from app.services.drift import STAGES
from app.services.drift.fault_injection import build_traffic
from app.services.drift.service import load_baseline, run_drift_check

router = APIRouter()


async def _get_version(db: AsyncSession, model_version_id: str) -> ModelVersion:
    mv = await db.get(ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return mv


def _stage_status(e: DriftEvent, with_details: bool = True) -> StageStatus:
    return StageStatus(
        stage=e.stage,
        score=e.drift_score,
        drifted=bool(e.is_drifted),
        p_value=e.p_value,
        detector=e.detector_type,
        top_feature=e.feature_name,
        onset=e.onset_timestamp,
        n_samples=e.n_samples,
        details=e.details if with_details else None,
    )


@router.get("/overview", response_model=OverviewResponse)
async def overview(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Every production model's health on one screen (FR17)."""
    rows = (
        await db.execute(
            select(ModelVersion, ExperimentRun.model_type, ExperimentRun.baseline_s3_path, RegisteredModel.name)
            .join(ExperimentRun, ExperimentRun.id == ModelVersion.run_id)
            .outerjoin(RegisteredModel, RegisteredModel.id == ModelVersion.registered_model_id)
            .where(ModelVersion.status == "production")
            .order_by(RegisteredModel.name)
        )
    ).all()
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    items = []
    for mv, model_type, baseline_path, rm_name in rows:
        last = (
            await db.execute(
                select(DriftEvent.check_id, DriftEvent.detected_at)
                .where(DriftEvent.model_version_id == mv.id)
                .order_by(DriftEvent.detected_at.desc())
                .limit(1)
            )
        ).first()
        stages, root_cause, confidence = {}, None, None
        if last:
            events = (await db.execute(select(DriftEvent).where(DriftEvent.check_id == last.check_id))).scalars()
            stages = {e.stage: {"score": e.drift_score, "drifted": bool(e.is_drifted)} for e in events}
            report = (
                (await db.execute(select(AttributionReport).where(AttributionReport.check_id == last.check_id)))
                .scalars()
                .first()
            )
            if report:
                root_cause, confidence = report.root_cause_stage, report.confidence
        n_preds = (
            await db.execute(
                select(func.count(Prediction.id)).where(
                    Prediction.model_version_id == mv.id, Prediction.predicted_at > since
                )
            )
        ).scalar_one()

        if not baseline_path:
            status = "no_baseline"
        elif not last:
            status = "not_checked"
        else:
            status = "drifting" if any(s["drifted"] for s in stages.values()) else "healthy"
        items.append(
            OverviewItem(
                model_version_id=mv.id,
                registered_model_name=rm_name,
                version_tag=mv.version_tag,
                model_type=model_type,
                sla_tier=mv.sla_tier,
                deployed_at=mv.deployed_at,
                status=status,
                last_check_at=last.detected_at if last else None,
                stages=stages,
                root_cause=root_cause,
                confidence=confidence,
                predictions_24h=n_preds,
            )
        )
    return OverviewResponse(items=items)


@router.get("/{model_version_id}/drift", response_model=DriftStatusResponse)
async def drift_status(
    model_version_id: str,
    hours: int = Query(24, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Current per-stage drift status (latest check) + score time series for trend lines (FR9)."""
    await _get_version(db, model_version_id)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    events = (
        (
            await db.execute(
                select(DriftEvent)
                .where(DriftEvent.model_version_id == model_version_id, DriftEvent.detected_at > since)
                .order_by(DriftEvent.detected_at)
            )
        )
        .scalars()
        .all()
    )

    by_check: dict[str, list[DriftEvent]] = {}
    for e in events:
        by_check.setdefault(e.check_id or e.id, []).append(e)
    series = [
        SeriesPoint(
            detected_at=group[0].detected_at,
            check_id=group[0].check_id,
            scores={e.stage: e.drift_score for e in group},
            drifted={e.stage: bool(e.is_drifted) for e in group},
        )
        for group in by_check.values()
    ]

    latest = None
    if by_check:
        group = list(by_check.values())[-1]
        group.sort(key=lambda e: STAGES.index(e.stage))
        latest = CheckSummary(
            check_id=group[0].check_id,
            detected_at=group[0].detected_at,
            window_start=group[0].window_start,
            window_end=group[0].window_end,
            stages=[_stage_status(e) for e in group],
        )
    return DriftStatusResponse(
        model_version_id=model_version_id, threshold=settings.DRIFT_MIN_EFFECT, latest=latest, series=series
    )


@router.get("/{model_version_id}/attribution", response_model=AttributionHistoryResponse)
async def attribution(
    model_version_id: str,
    limit: int = Query(20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Ranked root-cause reports, newest first (FR10)."""
    await _get_version(db, model_version_id)
    reports = (
        (
            await db.execute(
                select(AttributionReport)
                .where(AttributionReport.model_version_id == model_version_id)
                .order_by(AttributionReport.generated_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    history = [AttributionReportResponse.model_validate(r) for r in reports]
    return AttributionHistoryResponse(
        model_version_id=model_version_id, latest=history[0] if history else None, history=history
    )


@router.post("/{model_version_id}/check")
async def check_now(
    model_version_id: str,
    window_minutes: int | None = Query(None, ge=1, le=24 * 60),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Run a drift check immediately instead of waiting for the scheduler."""
    await _get_version(db, model_version_id)
    return await run_drift_check(db, model_version_id, window_minutes=window_minutes)


@router.post("/{model_version_id}/simulate")
async def simulate(
    model_version_id: str,
    payload: SimulateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Fault injection (demo / testing): log synthetic traffic with a fault at a known
    stage, then (optionally) run a check — the attribution should name that stage.
    """
    if not settings.ENABLE_FAULT_INJECTION:
        raise HTTPException(status_code=403, detail="Fault injection is disabled in this environment")
    mv = await _get_version(db, model_version_id)
    if mv.status == "archived":
        raise HTTPException(status_code=400, detail="Model is archived")
    baseline = await load_baseline(db, model_version_id)
    if baseline is None:
        raise HTTPException(status_code=409, detail="Model has no drift baseline — retrain it to enable monitoring")

    try:
        rows, info = build_traffic(
            baseline,
            n=payload.n,
            fault=payload.fault,
            feature=payload.feature,
            magnitude=payload.magnitude,
            fault_fraction=payload.fault_fraction,
            span_minutes=payload.span_minutes,
            feedback_rate=payload.feedback_rate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await serving_service.log_traffic(db, model_version_id, rows)

    db.add(
        AuditLog(
            actor_id=current_user.get("sub"),
            actor_email=current_user.get("email"),
            action="observability.simulate",
            target_type="model_version",
            target_id=model_version_id,
            extra_metadata=info,
        )
    )
    check = None
    if payload.run_check:
        check = await run_drift_check(db, model_version_id, window_minutes=int(payload.span_minutes) + 1)
    return {"traffic": info, "check": check}
