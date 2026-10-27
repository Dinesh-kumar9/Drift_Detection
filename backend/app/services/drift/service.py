"""
One drift check, end to end:

  load baseline + bundle → read the window of logged predictions → run the S1/S2/S3
  monitors → locate onsets → attribute → persist drift_events (+ attribution report)
  → alert (with cooldown)

Used by the Celery dispatcher (scheduled, per SLA tier) and by the API ("check now").
"""

import asyncio
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.utils import json_safe
from app.models import AttributionReport, DriftEvent, ExperimentRun, ModelVersion, Prediction, RegisteredModel
from app.services import serving_service, storage_service
from app.services.drift import STAGES
from app.services.drift.alerting import build_message, send_alert
from app.services.drift.attribution_engine import AttributionResult, attribute
from app.services.drift.baselines import Baseline, prediction_error
from app.services.drift.onset import bucket_indices, find_onset
from app.services.drift.stage_monitors import StageResult, monitor_s1, monitor_s2, monitor_s3
from app.services.model_bundle import ModelBundle

logger = logging.getLogger(__name__)

_BASELINE_CACHE: "OrderedDict[str, Baseline]" = OrderedDict()


async def load_baseline(db: AsyncSession, model_version_id: str) -> Baseline | None:
    row = (
        await db.execute(
            select(ExperimentRun.id, ExperimentRun.baseline_s3_path)
            .join(ModelVersion, ModelVersion.run_id == ExperimentRun.id)
            .where(ModelVersion.id == model_version_id)
        )
    ).first()
    if not row or not row.baseline_s3_path:
        return None
    if row.id in _BASELINE_CACHE:
        return _BASELINE_CACHE[row.id]
    data = await asyncio.to_thread(storage_service.download_uri, row.baseline_s3_path)
    baseline = await asyncio.to_thread(Baseline.from_bytes, data)
    _BASELINE_CACHE[row.id] = baseline
    while len(_BASELINE_CACHE) > settings.MODEL_CACHE_SIZE:
        _BASELINE_CACHE.popitem(last=False)
    return baseline


# ─── Pure analysis (runs in a worker thread) ─────────────────────────────────


def analyse_window(
    baseline: Baseline,
    bundle: ModelBundle,
    predictions: list[Prediction],
    n_buckets: int,
    alpha: float,
    min_effect: float,
) -> tuple[dict[str, StageResult], dict[str, datetime | None], dict[str, list], AttributionResult]:
    timestamps = [p.predicted_at for p in predictions]
    raw = pd.DataFrame.from_records([p.input_snapshot for p in predictions], columns=baseline.input_columns)
    for col in baseline.numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    feats = pd.DataFrame.from_records(
        [p.features_snapshot or {} for p in predictions], columns=baseline.feature_names
    ).astype(float)
    expected = bundle.transform(raw.astype(object).where(raw.notna(), None).to_dict(orient="records"))
    preds = [p.output.get("prediction") for p in predictions]
    confs = [p.output.get("confidence") for p in predictions]
    errors = np.array(
        [
            prediction_error(baseline.task_type, pr, p.actual) if p.actual is not None else np.nan
            for pr, p in zip(preds, predictions)
        ],
        dtype=float,
    )

    def evaluate(stage: str, idx: np.ndarray) -> StageResult:
        if stage == "S1_ingestion":
            return monitor_s1(baseline, raw.iloc[idx], alpha, min_effect)
        if stage == "S2_preprocessing":
            return monitor_s2(baseline, feats.iloc[idx], expected.iloc[idx], alpha, min_effect)
        errs = errors[idx]
        return monitor_s3(
            baseline,
            [preds[i] for i in idx],
            [confs[i] for i in idx],
            errs[~np.isnan(errs)].tolist(),
            alpha,
            min_effect,
        )

    all_idx = np.arange(len(predictions))
    results = {stage: evaluate(stage, all_idx) for stage in STAGES}

    buckets = bucket_indices(len(predictions), n_buckets)
    onsets: dict[str, datetime | None] = {s: None for s in STAGES}
    timelines: dict[str, list] = {s: [] for s in STAGES}
    for stage in STAGES:
        if results[stage].drifted:
            if buckets:
                onsets[stage], timelines[stage] = find_onset(timestamps, buckets, lambda i, s=stage: evaluate(s, i))
            onsets[stage] = onsets[stage] or timestamps[0]  # drift visible only window-wide

    span = (timestamps[-1] - timestamps[0]).total_seconds()
    bucket_seconds = span / max(len(buckets), 1)
    return results, onsets, timelines, attribute(results, onsets, bucket_seconds)


# ─── Orchestration ───────────────────────────────────────────────────────────


async def run_drift_check(
    db: AsyncSession,
    model_version_id: str,
    now: datetime | None = None,
    window_minutes: int | None = None,
) -> dict[str, Any]:
    mv = await db.get(ModelVersion, model_version_id)
    if mv is None:
        raise ValueError(f"Model version {model_version_id} not found")

    baseline = await load_baseline(db, model_version_id)
    if baseline is None:
        return {
            "status": "no_baseline",
            "model_version_id": model_version_id,
            "message": "Model was trained before drift baselines existed — retrain it to enable monitoring",
        }

    now = now or datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=window_minutes or settings.DRIFT_WINDOW_MINUTES)
    predictions = list(
        reversed(
            (
                await db.execute(
                    select(Prediction)
                    .where(
                        Prediction.model_version_id == model_version_id,
                        Prediction.predicted_at > window_start,
                        Prediction.predicted_at <= now,
                    )
                    .order_by(Prediction.predicted_at.desc())
                    .limit(settings.DRIFT_MAX_SAMPLES)
                )
            )
            .scalars()
            .all()
        )
    )
    if len(predictions) < settings.DRIFT_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "model_version_id": model_version_id,
            "n_samples": len(predictions),
            "message": f"Need at least {settings.DRIFT_MIN_SAMPLES} predictions in the window",
        }

    bundle = await serving_service.load_bundle(db, model_version_id)
    results, onsets, timelines, attribution = await asyncio.to_thread(
        analyse_window,
        baseline,
        bundle,
        predictions,
        settings.DRIFT_ONSET_BUCKETS,
        settings.DRIFT_KS_P_VALUE_THRESHOLD,
        settings.DRIFT_MIN_EFFECT,
    )

    # ── Persist ──────────────────────────────────────────────────────────────
    check_id = str(uuid.uuid4())
    w_start, w_end = predictions[0].predicted_at, predictions[-1].predicted_at
    events = []
    for stage in STAGES:
        res = results[stage]
        event = DriftEvent(
            id=str(uuid.uuid4()),
            model_version_id=model_version_id,
            check_id=check_id,
            stage=stage,
            drift_score=res.score,
            p_value=res.p_value,
            is_drifted=res.drifted,
            detector_type=res.detector,
            feature_name=(res.top_feature or "")[:100] or None,
            threshold_used=settings.DRIFT_MIN_EFFECT,
            n_samples=res.n_current,
            window_start=w_start,
            window_end=w_end,
            onset_timestamp=onsets[stage],
            details=json_safe({**res.details(), "onset_timeline": timelines[stage]}),
            detected_at=now,
        )
        db.add(event)
        events.append(event)

    report_id, alert_channel = None, None
    if attribution.drift_detected:
        report = AttributionReport(
            id=str(uuid.uuid4()),
            model_version_id=model_version_id,
            check_id=check_id,
            drift_event_ids=[e.id for e in events if e.is_drifted],
            ranked_stages=json_safe(attribution.ranked),
            root_cause_stage=attribution.root_cause,
            confidence=attribution.confidence,
            explanation=attribution.explanation,
            window_start=w_start,
            window_end=w_end,
            generated_at=now,
        )
        db.add(report)
        report_id = report.id
        alert_channel = await _maybe_alert(db, mv, report, attribution, now)
    await db.flush()

    logger.info(
        "Drift check %s for %s: drift=%s root_cause=%s",
        check_id,
        model_version_id,
        attribution.drift_detected,
        attribution.root_cause,
    )
    return {
        "status": "completed",
        "check_id": check_id,
        "model_version_id": model_version_id,
        "n_samples": len(predictions),
        "window_start": w_start.isoformat(),
        "window_end": w_end.isoformat(),
        "drift_detected": attribution.drift_detected,
        "root_cause": attribution.root_cause,
        "confidence": attribution.confidence,
        "explanation": attribution.explanation,
        "ranked": json_safe(attribution.ranked),
        "stages": {
            s: {
                "score": r.score,
                "drifted": r.drifted,
                "top_feature": r.top_feature,
                "onset": onsets[s].isoformat() if onsets[s] else None,
            }
            for s, r in results.items()
        },
        "report_id": report_id,
        "alert": alert_channel,
    }


async def _maybe_alert(
    db: AsyncSession, mv: ModelVersion, report: AttributionReport, attribution: AttributionResult, now: datetime
) -> str | None:
    """Alert unless the same root cause was already alerted for this model within the cooldown."""
    cooldown_start = now - timedelta(minutes=settings.ALERT_COOLDOWN_MINUTES)
    recent = (
        await db.execute(
            select(AttributionReport.id).where(
                AttributionReport.model_version_id == mv.id,
                AttributionReport.root_cause_stage == attribution.root_cause,
                AttributionReport.alert_sent_at > cooldown_start,
            )
        )
    ).first()
    if recent:
        return "suppressed"

    rm = await db.get(RegisteredModel, mv.registered_model_id) if mv.registered_model_id else None
    label = f"{rm.name if rm else 'model'} {mv.version_tag or ''}".strip()
    channel = await send_alert(build_message(label, mv.sla_tier, attribution, None))
    if channel != "failed":
        report.alert_sent_at = now
    return channel
