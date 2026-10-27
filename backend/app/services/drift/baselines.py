"""
Per-stage reference samples ("what normal looks like"), captured at training time.

  S1  raw input rows sampled from the training data (+ their true targets)
  S2  the same rows after the fitted preprocessing pipeline
  S3  the model's predictions / confidence / errors on the held-out test split
      (test split, not training rows — predictions on rows the model has seen
      are unrealistically confident and accurate)

Stored as JSON next to the model artifact; small (≤ DRIFT_BASELINE_ROWS rows).
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.utils import json_safe
from app.services.model_bundle import ModelBundle

FORMAT_VERSION = 1


def prediction_error(task_type: str, prediction: Any, actual: Any) -> float | None:
    """0/1 misclassification for classifiers, absolute error for regressors. None if unusable."""
    if prediction is None or actual is None:
        return None
    if task_type == "classification":
        return float(str(prediction) != str(actual))
    try:
        return abs(float(prediction) - float(actual))
    except (TypeError, ValueError):
        return None


@dataclass
class Baseline:
    task_type: str
    target_column: str
    numeric_cols: list[str]
    cat_cols: list[str]
    feature_names: list[str]
    s1: pd.DataFrame
    s1_target: list[Any]
    s2: pd.DataFrame
    s3_predictions: list[Any]
    s3_confidence: list[float] | None
    s3_errors: list[float]

    @property
    def input_columns(self) -> list[str]:
        return [*self.numeric_cols, *self.cat_cols]

    def to_bytes(self) -> bytes:
        payload = {
            "format": FORMAT_VERSION,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "numeric_cols": self.numeric_cols,
            "cat_cols": self.cat_cols,
            "feature_names": self.feature_names,
            "s1": self.s1.to_dict(orient="list"),
            "s1_target": self.s1_target,
            "s2": self.s2.to_dict(orient="list"),
            "s3": {
                "predictions": self.s3_predictions,
                "confidence": self.s3_confidence,
                "errors": self.s3_errors,
            },
        }
        return json.dumps(json_safe(payload)).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Baseline":
        p = json.loads(data)
        s1 = pd.DataFrame(p["s1"], columns=[*p["numeric_cols"], *p["cat_cols"]])
        for col in p["numeric_cols"]:
            s1[col] = pd.to_numeric(s1[col], errors="coerce")
        return cls(
            task_type=p["task_type"],
            target_column=p["target_column"],
            numeric_cols=p["numeric_cols"],
            cat_cols=p["cat_cols"],
            feature_names=p["feature_names"],
            s1=s1,
            s1_target=p["s1_target"],
            s2=pd.DataFrame(p["s2"], columns=p["feature_names"]).astype(float),
            s3_predictions=p["s3"]["predictions"],
            s3_confidence=p["s3"]["confidence"],
            s3_errors=p["s3"]["errors"],
        )


def build_baseline(
    bundle: ModelBundle,
    clean_df: pd.DataFrame,
    test_idx: np.ndarray,
    max_rows: int = 2000,
    seed: int = 42,
) -> Baseline:
    """
    clean_df: the training data after null-target rows were dropped (row order matching
    the preprocessed matrix); test_idx: positions of the held-out split.
    """
    rng = np.random.default_rng(seed)
    cols = bundle.input_columns
    target = bundle.target_column

    ref_idx = np.sort(rng.choice(len(clean_df), size=min(max_rows, len(clean_df)), replace=False))
    s1 = clean_df.iloc[ref_idx][cols].reset_index(drop=True)
    s1_records = s1.astype(object).where(s1.notna(), None).to_dict(orient="records")
    s2 = bundle.transform(s1_records)

    test_idx = np.asarray(test_idx)
    if len(test_idx) > max_rows:
        test_idx = np.sort(rng.choice(test_idx, size=max_rows, replace=False))
    test = clean_df.iloc[test_idx]
    test_records = test[cols].astype(object).where(test[cols].notna(), None).to_dict(orient="records")
    outputs = bundle.predict(test_records) if test_records else []
    actuals = test[target].tolist()
    errors = [prediction_error(bundle.task_type, o["prediction"], a) for o, a in zip(outputs, actuals)]

    return Baseline(
        task_type=bundle.task_type,
        target_column=target,
        numeric_cols=list(bundle.numeric_cols),
        cat_cols=list(bundle.cat_cols),
        feature_names=list(bundle.feature_names),
        s1=s1,
        s1_target=clean_df.iloc[ref_idx][target].tolist(),
        s2=s2,
        s3_predictions=[o["prediction"] for o in outputs],
        s3_confidence=[o["confidence"] for o in outputs] if outputs and outputs[0]["confidence"] is not None else None,
        s3_errors=[e for e in errors if e is not None],
    )
