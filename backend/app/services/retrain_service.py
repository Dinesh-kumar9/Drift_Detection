"""
Retraining loop (FR12–FR14).

  drift check → attribution → evaluate_gate() → RetrainJob
      pending   — gate passed (or forced); waits for an admin            (human gate 1: spend)
      approved  — execute_retrain_job(): build training data, train a candidate,
                  compare it with production on held-out recent ground truth
      completed — candidate ready; an admin promotes it or declines it  (human gate 2: promote)
      rejected  — gate suppressed it, or a human said no
      failed    — retraining errored

The cost-benefit gate (in order):
  1. enough ground truth since the drift started?
  2. root cause S2 (pipeline fault) → never retrain: the model would learn the bug
  3. did accuracy actually degrade? drift without accuracy loss isn't worth paying for
  4. expected value (recovered accuracy × $/point for the SLA tier) ≥ compute cost × margin

Training data depends on the root cause: concept drift (S3) means the old labels
are wrong, so the candidate trains only on labelled traffic after the onset;
otherwise the original dataset is extended with the new labelled traffic.
"""

import asyncio
import io
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.utils import json_safe
from app.models import (
    AuditLog,
    Dataset,
    ExperimentGroup,
    ExperimentRun,
    ModelVersion,
    Prediction,
    RegisteredModel,
    RetrainJob,
)
from app.services import ingestion_service, registry_service, serving_service, storage_service
from app.services.drift.baselines import prediction_error

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("pending", "approved", "running")


class RetrainConflictError(ValueError):
    """An open retrain job already exists for this model."""


class RetrainStateError(ValueError):
    """The requested transition isn't valid for the job's current status."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def audit(db: AsyncSession, action: str, target_id: str, actor: dict | None, metadata: dict) -> None:
    db.add(
        AuditLog(
            actor_id=(actor or {}).get("sub"),
            actor_email=(actor or {}).get("email") or ("system" if actor is None else None),
            action=action,
            target_type="retrain_job" if action.startswith("retrain.") else "model_version",
            target_id=target_id,
            extra_metadata=json_safe(metadata),
            created_at=_now(),
        )
    )


# ─── Cost-benefit gate ───────────────────────────────────────────────────────


@dataclass
class GateDecision:
    passed: bool
    reason: str
    mode: str  # post_onset_only | original_plus_recent
    data_since: str | None
    labelled_rows: int = 0
    baseline_error: float | None = None
    current_error: float | None = None
    error_increase: float | None = None
    cost_estimate: float | None = None
    expected_value: float | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return json_safe(asdict(self))


async def _labelled_predictions(db: AsyncSession, mv_id: str, since: datetime | None) -> list[Prediction]:
    query = select(Prediction).where(Prediction.model_version_id == mv_id, Prediction.actual.is_not(None))
    if since is not None:
        query = query.where(Prediction.predicted_at >= since)
    return list((await db.execute(query.order_by(Prediction.predicted_at))).scalars().all())


def _value_per_point(sla_tier: str) -> float:
    return {
        "critical": settings.RETRAIN_VALUE_PER_POINT_CRITICAL,
        "low": settings.RETRAIN_VALUE_PER_POINT_LOW,
    }.get(sla_tier, settings.RETRAIN_VALUE_PER_POINT_STANDARD)


async def evaluate_gate(
    db: AsyncSession,
    mv: ModelVersion,
    run: ExperimentRun,
    root_cause: str | None,
    data_since: datetime | None,
) -> GateDecision:
    task_type = (run.preprocessing_config or {}).get("task_type", "classification")
    mode = "post_onset_only" if root_cause == "S3_output" else "original_plus_recent"
    decision = GateDecision(
        passed=False, reason="", mode=mode, data_since=data_since.isoformat() if data_since else None
    )

    def check(name: str, ok: bool, detail: str) -> bool:
        decision.checks.append({"check": name, "passed": ok, "detail": detail})
        return ok

    labelled = await _labelled_predictions(db, mv.id, data_since)
    decision.labelled_rows = len(labelled)
    if not check(
        "enough ground truth",
        len(labelled) >= settings.RETRAIN_MIN_LABELLED,
        f"{len(labelled)} labelled predictions since onset (need {settings.RETRAIN_MIN_LABELLED})",
    ):
        decision.reason = "Not enough ground truth since the drift started to retrain on"
        return decision

    if not check(
        "not a pipeline fault",
        root_cause != "S2_preprocessing",
        f"root cause: {root_cause or 'unknown'}",
    ):
        decision.reason = (
            "Root cause is the preprocessing pipeline — fix the pipeline; retraining would teach the model the bug"
        )
        return decision

    errors = [prediction_error(task_type, p.output.get("prediction"), p.actual) for p in labelled]
    errors = [e for e in errors if e is not None]
    metrics = run.metrics or {}
    if task_type == "classification":
        baseline = 1 - float(metrics.get("accuracy", 1.0))
        current = float(np.mean(errors)) if errors else 0.0
        increase = current - baseline
        threshold = settings.RETRAIN_MIN_ERROR_INCREASE
        points_lost = max(increase, 0.0) * 100
        detail = f"error rate {baseline:.1%} at training → {current:.1%} now (need +{threshold:.0%})"
    else:
        baseline = float(metrics.get("mae") or 0.0)
        current = float(np.mean(errors)) if errors else 0.0
        increase = (current - baseline) / baseline if baseline else 0.0
        threshold = settings.RETRAIN_MIN_REL_ERROR_INCREASE
        points_lost = max(increase, 0.0) * 100  # relative % of MAE
        detail = f"MAE {baseline:.3g} at training → {current:.3g} now (+{increase:.0%}, need +{threshold:.0%})"
    decision.baseline_error, decision.current_error, decision.error_increase = baseline, current, increase

    if not check("accuracy degraded", increase >= threshold, detail):
        decision.reason = "Drift detected, but accuracy hasn't degraded enough to justify retraining"
        return decision

    rows_scale = max(1.0, len(labelled) / max(int(metrics.get("n_train") or 1), 1))
    hours = float(metrics.get("training_time_s") or 60.0) * rows_scale / 3600
    cost = hours * settings.RETRAIN_VCPUS * settings.RETRAIN_COST_PER_VCPU_HOUR + settings.RETRAIN_FIXED_COST
    value = points_lost * settings.RETRAIN_RECOVERY_FACTOR * _value_per_point(mv.sla_tier)
    decision.cost_estimate, decision.expected_value = round(cost, 4), round(value, 2)
    if not check(
        "worth the cost",
        value >= cost * settings.RETRAIN_VALUE_MARGIN,
        f"expected value ${value:,.2f} vs cost ${cost:,.4f} × {settings.RETRAIN_VALUE_MARGIN:g}",
    ):
        decision.reason = "Expected accuracy recovery isn't worth the compute cost"
        return decision

    decision.passed = True
    decision.reason = (
        "Concept drift: retrain on labelled traffic since the onset"
        if mode == "post_onset_only"
        else "Accuracy degraded after an input shift: retrain on original + recent labelled data"
    )
    return decision


# ─── Job lifecycle ───────────────────────────────────────────────────────────


async def open_job(db: AsyncSession, mv_id: str) -> RetrainJob | None:
    """A job that is still in flight, or a trained candidate awaiting a promote/decline decision."""
    jobs = (
        (
            await db.execute(
                select(RetrainJob).where(
                    RetrainJob.model_version_id == mv_id,
                    RetrainJob.status.in_(OPEN_STATUSES + ("completed",)),
                )
            )
        )
        .scalars()
        .all()
    )
    return next((j for j in jobs if not (j.status == "completed" and j.promoted_at)), None)


async def request_retrain(
    db: AsyncSession,
    mv_id: str,
    trigger_type: str,
    actor: dict | None,
    root_cause: str | None = None,
    data_since: datetime | None = None,
    attribution_report_id: str | None = None,
    note: str | None = None,
    force: bool = False,
) -> RetrainJob:
    mv, run = await registry_service.get_model_with_run(db, mv_id)
    if mv is None:
        raise ValueError(f"Model version {mv_id} not found")
    if (existing := await open_job(db, mv_id)) is not None:
        raise RetrainConflictError(f"Retrain job {existing.id} is already {existing.status} for this model")

    decision = await evaluate_gate(db, mv, run, root_cause, data_since)
    status = "pending" if (decision.passed or force) else "rejected"
    job = RetrainJob(
        model_version_id=mv_id,
        trigger_type=trigger_type,
        trigger_reason=json_safe(
            {"root_cause": root_cause, "data_since": decision.data_since, "note": note, "forced": force}
        ),
        decision=decision.as_dict(),
        cost_estimate=decision.cost_estimate,
        expected_gain=decision.error_increase,
        attribution_report_id=attribution_report_id,
        requested_by=(actor or {}).get("sub"),
        status=status,
        rejection_reason=None if status == "pending" else f"Cost-benefit gate: {decision.reason}",
        created_at=_now(),
    )
    db.add(job)
    await db.flush()
    audit(
        db,
        "retrain.requested" if status == "pending" else "retrain.suppressed",
        job.id,
        actor,
        {"model_version_id": mv_id, "trigger": trigger_type, "gate": decision.reason, "forced": force},
    )
    await db.flush()
    return job


async def consider_auto_retrain(
    db: AsyncSession, mv_id: str, root_cause: str | None, onset: datetime | None, report_id: str | None
) -> dict | None:
    """Called after a drift check that found drift. Never raises; returns what happened (or None)."""
    if not settings.AUTO_RETRAIN_ENABLED or await open_job(db, mv_id) is not None:
        return None
    recently_suppressed = (
        await db.execute(
            select(RetrainJob.id).where(
                RetrainJob.model_version_id == mv_id,
                RetrainJob.trigger_type == "auto",
                RetrainJob.status == "rejected",
                RetrainJob.created_at > _now() - timedelta(minutes=settings.RETRAIN_SUPPRESS_MINUTES),
                RetrainJob.trigger_reason["root_cause"].astext == (root_cause or ""),
            )
        )
    ).first()
    if recently_suppressed:
        return None
    try:
        job = await request_retrain(
            db, mv_id, "auto", None, root_cause=root_cause, data_since=onset, attribution_report_id=report_id
        )
    except RetrainConflictError:
        return None
    return {"job_id": job.id, "status": job.status, "reason": job.decision.get("reason")}


def _get_open(job: RetrainJob | None, *allowed: str) -> RetrainJob:
    if job is None:
        raise ValueError("Retrain job not found")
    if job.status not in allowed:
        raise RetrainStateError(f"Job is {job.status}; expected {' or '.join(allowed)}")
    return job


async def approve(db: AsyncSession, job_id: str, actor: dict) -> RetrainJob:
    job = _get_open(await db.get(RetrainJob, job_id), "pending")
    job.status, job.approved_by, job.approved_at = "approved", actor.get("sub"), _now()
    audit(db, "retrain.approved", job.id, actor, {"model_version_id": job.model_version_id})
    await db.flush()
    return job


async def reject(db: AsyncSession, job_id: str, actor: dict, reason: str) -> RetrainJob:
    job = await db.get(RetrainJob, job_id)
    if job is not None and job.status == "completed" and job.promoted_at:
        raise RetrainStateError("Candidate was already promoted")
    job = _get_open(job, "pending", "completed")
    stage = "candidate declined" if job.status == "completed" else "request rejected"
    job.status, job.rejection_reason = "rejected", f"{stage} by {actor.get('email') or 'admin'}: {reason}"
    audit(db, "retrain.rejected", job.id, actor, {"reason": reason, "stage": stage})
    await db.flush()
    return job


async def promote(db: AsyncSession, job_id: str, actor: dict) -> RetrainJob:
    job = _get_open(await db.get(RetrainJob, job_id), "completed")
    if job.promoted_at:
        raise RetrainStateError("Candidate was already promoted")
    if not job.candidate_model_version_id:
        raise RetrainStateError("Job has no candidate model")
    current = await db.get(ModelVersion, job.model_version_id)
    promoted, previous = await registry_service.promote_to_production(
        db, job.candidate_model_version_id, current.sla_tier if current else "standard"
    )
    serving_service.invalidate_cache(promoted.id)
    job.promoted_at, job.promoted_by = _now(), actor.get("sub")
    audit(
        db,
        "retrain.promoted",
        job.id,
        actor,
        {
            "candidate_model_version_id": promoted.id,
            "version_tag": promoted.version_tag,
            "replaced_version_id": previous.id if previous else None,
            "evaluation": job.evaluation,
        },
    )
    await db.flush()
    return job


# ─── Execution (Celery worker) ───────────────────────────────────────────────


def _records(preds: list[Prediction], cols: list[str]) -> list[dict]:
    return [{c: (p.input_snapshot or {}).get(c) for c in cols} for p in preds]


def _error_rate(task_type: str, outputs: list[dict], actuals: list) -> float | None:
    errs = [prediction_error(task_type, o["prediction"], a) for o, a in zip(outputs, actuals)]
    errs = [e for e in errs if e is not None]
    return round(float(np.mean(errs)), 4) if errs else None


async def execute_retrain_job(job_id: str, session_factory: async_sessionmaker[AsyncSession]) -> dict:
    from app.workers.training_tasks import execute_training_job  # avoid import cycle at module load

    async with session_factory() as db:
        job = await db.get(RetrainJob, job_id)
        if job is None or job.status != "approved":
            return {"status": "skipped", "reason": f"job is {job.status if job else 'missing'}"}
        job.status = "running"
        await db.commit()

    try:
        async with session_factory() as db:
            job = await db.get(RetrainJob, job_id)
            mv, run = await registry_service.get_model_with_run(db, job.model_version_id)
            registered = await db.get(RegisteredModel, mv.registered_model_id) if mv.registered_model_id else None
            original = await db.get(Dataset, run.dataset_id)
            bundle = await serving_service.load_bundle(db, mv.id)
            decision = job.decision or {}
            since = datetime.fromisoformat(decision["data_since"]) if decision.get("data_since") else None
            labelled = await _labelled_predictions(db, mv.id, since)
            if len(labelled) < 10:
                raise ValueError("Not enough labelled predictions to retrain on")

            # newest rows are held back to compare candidate vs production fairly
            n_hold = max(5, int(len(labelled) * settings.RETRAIN_HOLDOUT_FRACTION))
            train_preds, holdout = labelled[:-n_hold], labelled[-n_hold:]
            cols, target = bundle.input_columns, bundle.target_column
            recent = pd.DataFrame(_records(train_preds, cols), columns=cols)
            recent[target] = [p.actual for p in train_preds]
            target_dtype = str((original.schema_json or {}).get(target, {}).get("dtype", "object"))
            if target_dtype.startswith(("int", "float", "uint")):
                recent[target] = pd.to_numeric(recent[target], errors="coerce")

            if decision.get("mode") == "post_onset_only":
                train_df = recent
            else:
                raw = await asyncio.to_thread(storage_service.download_uri, original.s3_path)
                base = await asyncio.to_thread(pd.read_csv, io.BytesIO(raw))
                train_df = pd.concat([base[cols + [target]], recent], ignore_index=True)

            buf = io.BytesIO()
            train_df.to_csv(buf, index=False)
            stamp = _now().strftime("%Y%m%d-%H%M")
            name = f"{registered.name if registered else 'model'}-retrain-{stamp}"
            dataset = await ingestion_service.ingest_dataset(
                db, buf.getvalue(), f"{name}.csv", name, owner_id=job.approved_by
            )
            group = ExperimentGroup(
                dataset_id=dataset.id,
                registered_model_id=mv.registered_model_id,
                target_column=target,
                task_type=bundle.task_type,
                resolved_task_type=bundle.task_type,
                requested_models=[run.model_type],
                test_size=0.2,
                random_state=42,
                status="queued",
                created_by=job.approved_by,
                created_at=_now(),
            )
            db.add(group)
            await db.flush()
            db.add(
                ExperimentRun(
                    dataset_id=dataset.id,
                    group_id=group.id,
                    model_type=run.model_type,
                    status="queued",
                    created_at=_now(),
                )
            )
            job.experiment_group_id = group.id
            holdout_records = _records(holdout, cols)
            holdout_actuals = [p.actual for p in holdout]
            await db.commit()

        await execute_training_job(group.id, session_factory)

        async with session_factory() as db:
            job = await db.get(RetrainJob, job_id)
            candidate = (
                (
                    await db.execute(
                        select(ModelVersion)
                        .join(ExperimentRun, ExperimentRun.id == ModelVersion.run_id)
                        .where(ExperimentRun.group_id == group.id)
                    )
                )
                .scalars()
                .first()
            )
            if candidate is None:
                raise ValueError("Training finished without a candidate model")
            prod_bundle = await serving_service.load_bundle(db, job.model_version_id)
            cand_bundle = await serving_service.load_bundle(db, candidate.id)
            prod_out = await asyncio.to_thread(prod_bundle.predict, holdout_records)
            cand_out = await asyncio.to_thread(cand_bundle.predict, holdout_records)
            prod_err = _error_rate(prod_bundle.task_type, prod_out, holdout_actuals)
            cand_err = _error_rate(cand_bundle.task_type, cand_out, holdout_actuals)
            cand_run = await db.get(ExperimentRun, candidate.run_id)
            job.candidate_model_version_id = candidate.id
            job.evaluation = json_safe(
                {
                    "holdout_rows": len(holdout_records),
                    "metric": "error rate" if prod_bundle.task_type == "classification" else "MAE",
                    "production_error": prod_err,
                    "candidate_error": cand_err,
                    "improvement": None if prod_err is None or cand_err is None else round(prod_err - cand_err, 4),
                    "training_rows": len(train_df),
                    "training_mode": decision.get("mode"),
                    "candidate_version_tag": candidate.version_tag,
                    "candidate_metrics": cand_run.metrics if cand_run else None,
                }
            )
            job.status = "completed"
            audit(db, "retrain.completed", job.id, None, job.evaluation)
            await db.commit()
            return {"status": "completed", "job_id": job_id, "candidate": candidate.id, **job.evaluation}
    except Exception as exc:
        logger.exception("Retrain job %s failed", job_id)
        async with session_factory() as db:
            job = await db.get(RetrainJob, job_id)
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            audit(db, "retrain.failed", job.id, None, {"error": job.error})
            await db.commit()
        raise
