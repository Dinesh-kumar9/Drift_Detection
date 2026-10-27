"""
Independent monitors for the three pipeline stages (FR8).

  S1 ingestion       raw inputs vs training rows: KS / chi-square per column + null-rate shifts
  S2 preprocessing   model features vs training features, plus a pipeline-consistency check:
                     re-run the training-time transform on the logged raw inputs and compare
                     with the features the model actually received. Divergence means the
                     preprocessing step itself changed — the signal that separates
                     "upstream data changed" from "the pipeline broke".
  S3 output          prediction / confidence distribution vs test-set predictions, and — when
                     ground truth has arrived via feedback — whether the error rate rose
                     (concept drift: same inputs, different right answers).

Each monitor returns a StageResult: score = largest effect size among its tests (0–1),
drifted = any test both significant (Bonferroni) and large enough.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.services.drift.baselines import Baseline
from app.services.drift.detectors import (
    FeatureDrift,
    categorical_test,
    mark_drift,
    null_rate_test,
    numeric_test,
    proportion_increase_test,
)

MIN_LABELLED = 20  # error-drift test needs at least this much ground truth
DIVERGENCE_TOLERANCE = 1e-6


@dataclass
class StageResult:
    stage: str
    score: float
    drifted: bool
    p_value: float | None
    n_reference: int
    n_current: int
    tests: list[FeatureDrift] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def top_feature(self) -> str | None:
        return self.tests[0].feature if self.tests else None

    @property
    def detector(self) -> str | None:
        return self.tests[0].test if self.tests else None

    def details(self, max_tests: int = 10) -> dict[str, Any]:
        return {"tests": [t.to_dict() for t in self.tests[:max_tests]], "signals": self.signals}


def _summarize(stage, tests, n_ref, n_cur, alpha, min_effect, signals=None) -> StageResult:
    tests = mark_drift([t for t in tests if t is not None], alpha, min_effect)
    return StageResult(
        stage=stage,
        score=round(max((t.statistic for t in tests), default=0.0), 4),
        drifted=any(t.drifted for t in tests),
        p_value=min((t.p_value for t in tests), default=None),
        n_reference=n_ref,
        n_current=n_cur,
        tests=tests,
        signals=signals or {},
    )


# ─── S1: ingestion ───────────────────────────────────────────────────────────


def monitor_s1(baseline: Baseline, current: pd.DataFrame, alpha: float, min_effect: float) -> StageResult:
    tests: list[FeatureDrift | None] = []
    for col in baseline.numeric_cols:
        cur = current[col] if col in current else pd.Series([None] * len(current))
        tests.append(numeric_test(col, baseline.s1[col], cur))
        tests.append(null_rate_test(col, baseline.s1[col], cur))
    for col in baseline.cat_cols:
        cur = current[col] if col in current else pd.Series([None] * len(current))
        tests.append(categorical_test(col, baseline.s1[col], cur))
        tests.append(null_rate_test(col, baseline.s1[col], cur))
    return _summarize("S1_ingestion", tests, len(baseline.s1), len(current), alpha, min_effect)


# ─── S2: preprocessing ───────────────────────────────────────────────────────


def _is_binary(series: pd.Series) -> bool:
    values = set(pd.unique(series.dropna()))
    return values <= {0.0, 1.0}


def pipeline_divergence(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    """Share of rows whose logged features differ from transform(logged raw input)."""
    cols = [c for c in expected.columns if c in actual.columns]
    if not cols or len(actual) == 0:
        return {"divergent_fraction": 0.0, "features": {}}
    diff = np.abs(actual[cols].to_numpy(dtype=float) - expected[cols].to_numpy(dtype=float))
    diff = np.nan_to_num(diff, nan=0.0)
    divergent = diff > DIVERGENCE_TOLERANCE
    per_feature = {c: round(float(divergent[:, i].mean()), 4) for i, c in enumerate(cols)}
    top = sorted(((c, v) for c, v in per_feature.items() if v > 0), key=lambda kv: -kv[1])[:10]
    return {"divergent_fraction": round(float(divergent.any(axis=1).mean()), 4), "features": dict(top)}


def monitor_s2(
    baseline: Baseline,
    current: pd.DataFrame,
    expected: pd.DataFrame | None,
    alpha: float,
    min_effect: float,
) -> StageResult:
    tests: list[FeatureDrift | None] = []
    for col in baseline.feature_names:
        if col not in current:
            continue
        ref = baseline.s2[col]
        if _is_binary(ref):
            tests.append(categorical_test(col, ref.round(), current[col].round()))
        else:
            tests.append(numeric_test(col, ref, current[col]))

    signals: dict[str, Any] = {}
    if expected is not None:
        div = pipeline_divergence(current, expected)
        frac = div["divergent_fraction"]
        signals["pipeline_divergence"] = div
        tests.append(
            FeatureDrift(
                feature="pipeline consistency",
                test="divergence",
                statistic=frac,
                p_value=0.0 if frac > 0 else 1.0,  # any divergence is a fact, not a sample estimate
                n_reference=len(current),
                n_current=len(current),
                detail={"top_features": div["features"]},
            )
        )
    result = _summarize("S2_preprocessing", tests, len(baseline.s2), len(current), alpha, min_effect, signals)
    result.signals["diverged"] = any(t.test == "divergence" and t.drifted for t in result.tests)
    return result


# ─── S3: output ──────────────────────────────────────────────────────────────


def monitor_s3(
    baseline: Baseline,
    predictions: list[Any],
    confidences: list[float | None],
    errors: list[float],
    alpha: float,
    min_effect: float,
) -> StageResult:
    tests: list[FeatureDrift | None] = []
    if baseline.task_type == "classification":
        tests.append(categorical_test("predicted class", baseline.s3_predictions, predictions))
        if baseline.s3_confidence:
            conf = [c for c in confidences if c is not None]
            tests.append(numeric_test("confidence", baseline.s3_confidence, conf))
    else:
        tests.append(numeric_test("prediction", baseline.s3_predictions, predictions))

    error_test = None
    if len(errors) >= MIN_LABELLED and baseline.s3_errors:
        if baseline.task_type == "classification":
            error_test = proportion_increase_test("error rate", baseline.s3_errors, errors)
        else:
            error_test = numeric_test("absolute error", baseline.s3_errors, errors)
            # only an increase in error is drift worth acting on
            if error_test is not None and np.median(errors) <= np.median(baseline.s3_errors):
                error_test.statistic = 0.0
                error_test.p_value = 1.0
        tests.append(error_test)

    result = _summarize(
        "S3_output",
        tests,
        len(baseline.s3_predictions),
        len(predictions),
        alpha,
        min_effect,
        {"labelled": len(errors)},
    )
    result.signals["error_drift"] = bool(error_test is not None and error_test.drifted)
    return result
