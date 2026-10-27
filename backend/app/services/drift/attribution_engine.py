"""
Root-cause attribution (FR10).

    score(stage) = magnitude × temporal_weight × causal_factor

  magnitude        the stage's largest drifted effect size (0–1)
  temporal_weight  1 / (1 + onset_lag / bucket_width) — the stage that drifted earliest
                   gets 1.0. Bounded version of "1 / onset_lag", which divides by zero
                   for the earliest stage.
  causal_factor    pipeline consistency. All three stages are logged per request, so a
                   real upstream shift appears at S1, S2 and S3 at the same moment with the
                   same KS magnitude (KS is invariant to scaling). Downstream drift that is
                   fully explained upstream is "inherited" and discounted:
                     S2 drift + pipeline divergence → preprocessing fault   (factor 1.0)
                     S2 drift, S1 drifted, no divergence → inherited from S1 (factor 0.3)
                     S3 error rose, inputs stable → concept drift             (factor 1.0)
                     S3 shifted only because inputs shifted → inherited      (factor 0.3)

Confidence = stage score / sum of scores. Ties go to the upstream stage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.services.drift import STAGES
from app.services.drift.stage_monitors import StageResult

INHERITED = 0.3
PARTIALLY_INHERITED = 0.6

LABELS = {"S1_ingestion": "S1 ingestion", "S2_preprocessing": "S2 preprocessing", "S3_output": "S3 output"}


@dataclass
class AttributionResult:
    drift_detected: bool
    root_cause: str | None
    confidence: float | None
    ranked: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = "No drift detected at any stage."


def _causal_factor(stage: str, results: dict[str, StageResult]) -> tuple[float, str]:
    s1, s2, s3 = (results.get(s) for s in STAGES)
    s1_drift = bool(s1 and s1.drifted)
    s2_drift = bool(s2 and s2.drifted)

    if stage == "S1_ingestion":
        return 1.0, "raw input distribution shifted against the training data"

    if stage == "S2_preprocessing":
        if s2 and s2.signals.get("diverged"):
            return 1.0, "features no longer match the training-time transform of the same inputs"
        if s1_drift:
            return INHERITED, "feature shift is explained by the raw-input shift (pipeline unchanged)"
        return 1.0, "feature distribution shifted while raw inputs look stable"

    # S3
    upstream = s1_drift or s2_drift
    if s3 and s3.signals.get("error_drift"):
        if upstream:
            return PARTIALLY_INHERITED, "errors rose, but inputs also shifted"
        return 1.0, "error rate rose on stable inputs — the input→label relationship changed"
    if upstream:
        return INHERITED, "output shift follows the upstream input shift"
    return 1.0, "output distribution shifted on stable inputs"


def attribute(
    results: dict[str, StageResult],
    onsets: dict[str, datetime | None],
    bucket_seconds: float,
) -> AttributionResult:
    drifted = [s for s in STAGES if s in results and results[s].drifted]
    if not drifted:
        return AttributionResult(drift_detected=False, root_cause=None, confidence=None)

    known_onsets = [onsets[s] for s in drifted if onsets.get(s) is not None]
    earliest = min(known_onsets) if known_onsets else None
    bucket_seconds = max(bucket_seconds, 1.0)

    ranked = []
    for stage in drifted:
        res = results[stage]
        magnitude = max((t.statistic for t in res.tests if t.drifted), default=res.score)
        onset = onsets.get(stage)
        lag = (onset - earliest).total_seconds() if onset and earliest else 0.0
        temporal = 1.0 / (1.0 + lag / bucket_seconds)
        factor, reason = _causal_factor(stage, results)
        ranked.append(
            {
                "stage": stage,
                "score": magnitude * temporal * factor,
                "magnitude": round(magnitude, 4),
                "temporal_weight": round(temporal, 4),
                "causal_factor": factor,
                "onset": onset.isoformat() if onset else None,
                "onset_lag_seconds": round(lag, 1),
                "top_feature": res.top_feature,
                "reason": reason,
            }
        )

    total = sum(r["score"] for r in ranked) or 1.0
    ranked.sort(key=lambda r: (-round(r["score"], 6), STAGES.index(r["stage"])))
    for r in ranked:
        r["confidence"] = round(r["score"] / total, 4)
        r["score"] = round(r["score"], 4)

    top = ranked[0]
    feature = f" (most affected: {top['top_feature']})" if top["top_feature"] else ""
    explanation = f"Most likely root cause: {LABELS[top['stage']]} — {top['reason']}{feature}."
    others = [f"{LABELS[r['stage']]} {r['confidence']:.0%}" for r in ranked[1:]]
    if others:
        explanation += " Also drifted: " + ", ".join(others) + "."

    return AttributionResult(
        drift_detected=True,
        root_cause=top["stage"],
        confidence=top["confidence"],
        ranked=ranked,
        explanation=explanation,
    )
