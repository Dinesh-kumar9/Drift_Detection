"""
Synthetic traffic with a fault injected at a known stage — for demos and for
proving attribution works (Phase 2 exit criterion).

Healthy traffic resamples the model's own reference rows (so it matches the
training distribution) and spreads them evenly over the last `span_minutes`.
The last `fault_fraction` of that span is faulty:

  s1_shift    shift one raw input column (numeric: +magnitude σ; categorical: collapse
              onto its rarest category) — e.g. an upstream unit change
  s1_nulls    one raw input column arrives empty — e.g. a broken upstream join
  s2_shift    add +magnitude to one transformed feature after preprocessing — e.g. a
              buggy scaler deployed; raw inputs are untouched
  s3_concept  inputs unchanged, but ground truth flips — e.g. fraudsters change tactics
  none        healthy traffic only

Ground truth is attached to `feedback_rate` of rows, as if it arrived later.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.services.drift.baselines import Baseline

FAULT_TYPES = ("none", "s1_shift", "s1_nulls", "s2_shift", "s3_concept")
FAULT_STAGE = {
    "s1_shift": "S1_ingestion",
    "s1_nulls": "S1_ingestion",
    "s2_shift": "S2_preprocessing",
    "s3_concept": "S3_output",
}


@dataclass
class TrafficRow:
    record: dict[str, Any]
    timestamp: datetime
    actual: Any
    feature_offsets: dict[str, float] | None = None


def _default_feature(baseline: Baseline, fault: str) -> str:
    if baseline.numeric_cols:
        # the numeric column with most spread — a shift there is plainly visible
        stds = baseline.s1[baseline.numeric_cols].std(numeric_only=True)
        return str(stds.idxmax()) if stds.notna().any() else baseline.numeric_cols[0]
    if fault == "s2_shift":
        return baseline.feature_names[0]
    return baseline.cat_cols[0]


def build_traffic(
    baseline: Baseline,
    n: int,
    fault: str = "none",
    feature: str | None = None,
    magnitude: float = 3.0,
    fault_fraction: float = 0.5,
    span_minutes: float = 30.0,
    feedback_rate: float = 1.0,
    now: datetime | None = None,
    seed: int = 0,
) -> tuple[list[TrafficRow], dict[str, Any]]:
    if fault not in FAULT_TYPES:
        raise ValueError(f"fault must be one of {FAULT_TYPES}")
    rng = np.random.default_rng(seed)
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(minutes=span_minutes)

    feature = feature or (_default_feature(baseline, fault) if fault != "none" else None)
    if fault in ("s1_shift", "s1_nulls") and feature not in baseline.input_columns:
        raise ValueError(f"'{feature}' is not a raw input column")
    if fault == "s2_shift" and feature not in baseline.feature_names:
        raise ValueError(f"'{feature}' is not a model feature")

    idx = rng.integers(0, len(baseline.s1), size=n)
    raw = baseline.s1.iloc[idx].reset_index(drop=True)
    records = raw.astype(object).where(raw.notna(), None).to_dict(orient="records")
    actuals = [baseline.s1_target[i] for i in idx]
    first_faulty = int(round(n * (1 - fault_fraction))) if fault != "none" else n

    classes = sorted({str(t) for t in baseline.s1_target if t is not None})
    target_std = float(pd.to_numeric(pd.Series(baseline.s1_target), errors="coerce").std() or 1.0)
    feature_std = float(baseline.s1[feature].std() or 1.0) if feature in baseline.numeric_cols else 1.0
    rarest = (
        baseline.s1[feature].value_counts().idxmin() if fault == "s1_shift" and feature in baseline.cat_cols else None
    )

    rows = []
    for i, (record, actual) in enumerate(zip(records, actuals)):
        ts = start + timedelta(minutes=span_minutes * (i + 0.5) / n)
        offsets = None
        if i >= first_faulty:
            if fault == "s1_shift":
                if feature in baseline.numeric_cols and record[feature] is not None:
                    record[feature] = float(record[feature]) + magnitude * feature_std
                elif feature in baseline.cat_cols:
                    record[feature] = rarest
            elif fault == "s1_nulls":
                record[feature] = None
            elif fault == "s2_shift":
                offsets = {feature: magnitude}
            elif fault == "s3_concept":
                if baseline.task_type == "classification":
                    others = [c for c in classes if c != str(actual)]
                    actual = others[int(rng.integers(len(others)))] if others else actual
                else:
                    actual = float(actual) + magnitude * target_std
        has_label = rng.random() < feedback_rate
        rows.append(
            TrafficRow(record=record, timestamp=ts, actual=actual if has_label else None, feature_offsets=offsets)
        )

    info = {
        "fault": fault,
        "stage": FAULT_STAGE.get(fault),
        "feature": feature,
        "magnitude": magnitude,
        "n": n,
        "n_faulty": n - first_faulty,
        "fault_starts_at": rows[first_faulty].timestamp.isoformat() if first_faulty < n else None,
    }
    return rows, info
