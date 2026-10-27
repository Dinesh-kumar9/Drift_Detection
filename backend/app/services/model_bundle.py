"""
Model bundle — the single deployable artifact produced per experiment run.

Packs the fitted preprocessing transformer, the estimator, and the label
encoder together, so serving reproduces training-time preprocessing exactly
and every prediction exposes all three observability stages:

    S1 raw input  →  S2 transformed features  →  S3 output
"""

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.utils import json_safe
from app.services.preprocessing_service import transform_features

BUNDLE_FORMAT_VERSION = 1


class SchemaValidationError(ValueError):
    """Request features don't match the columns the model was trained on."""

    def __init__(self, missing_columns: list[str]):
        self.missing_columns = missing_columns
        super().__init__(f"Missing required feature(s): {', '.join(missing_columns)}")


class IncompatibleArtifactError(ValueError):
    """Artifact isn't a ModelBundle (e.g. a bare pickled model from before Phase 1.5)."""


def _to_python(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


@dataclass
class ModelBundle:
    model_type: str
    task_type: str  # classification | regression
    target_column: str
    numeric_cols: list[str]
    cat_cols: list[str]
    feature_names: list[str]
    preprocessor: Any  # fitted ColumnTransformer
    model: Any  # fitted estimator (trained on preprocessor output as a numpy array)
    label_encoder: Any | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = BUNDLE_FORMAT_VERSION

    # ── Schema ──────────────────────────────────────────────────────────────

    @property
    def input_columns(self) -> list[str]:
        return [*self.numeric_cols, *self.cat_cols]

    @property
    def classes(self) -> list[Any] | None:
        if self.label_encoder is None:
            return None
        return [_to_python(c) for c in self.label_encoder.classes_]

    def validate(self, records: list[dict[str, Any]]) -> None:
        """Every training column must be present (null values are fine — they're imputed)."""
        missing = sorted({col for rec in records for col in self.input_columns if col not in rec})
        if missing:
            raise SchemaValidationError(missing)

    # ── Inference ───────────────────────────────────────────────────────────

    def transform(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """S1 raw records → S2 model-ready feature frame."""
        self.validate(records)
        raw = pd.DataFrame.from_records(records, columns=self.input_columns)
        return transform_features(self.preprocessor, raw, self.numeric_cols, self.cat_cols)

    def predict(
        self,
        records: list[dict[str, Any]],
        feature_offsets: list[dict[str, float] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns one dict per record:
          prediction     decoded label (classification) or float (regression)
          confidence     max class probability (classifiers with predict_proba)
          probabilities  {label: p} (classifiers with predict_proba)
          features       S2 feature values, for drift monitoring

        `feature_offsets` (fault injection only) adds per-row offsets to the transformed
        features — simulating a broken preprocessing deploy. Normal serving never passes it.
        """
        features = self.transform(records)
        if feature_offsets:
            for i, offsets in enumerate(feature_offsets):
                for name, delta in (offsets or {}).items():
                    if name in features.columns:
                        features.iat[i, features.columns.get_loc(name)] += delta
        X = features.to_numpy()
        raw_pred = self.model.predict(X)

        probas = None
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            try:
                probas = self.model.predict_proba(X)
            except Exception:  # some estimators expose predict_proba but can't compute it
                probas = None

        if self.label_encoder is not None:
            labels = self.label_encoder.inverse_transform(np.asarray(raw_pred).astype(int))
        else:
            labels = raw_pred

        results = []
        for i, label in enumerate(labels):
            out: dict[str, Any] = {
                "prediction": _to_python(label),
                "confidence": None,
                "probabilities": None,
                "features": json_safe(dict(zip(self.feature_names, features.iloc[i].tolist()))),
            }
            if probas is not None:
                row = probas[i]
                out["confidence"] = round(float(np.max(row)), 4)
                class_labels = self.classes or list(range(len(row)))
                out["probabilities"] = {str(c): round(float(p), 4) for c, p in zip(class_labels, row)}
            results.append(out)
        return results

    def sample_input(self, baseline_stats: dict | None = None) -> dict[str, Any]:
        """A plausible request body built from dataset baseline stats (median / most frequent value)."""
        stats = baseline_stats or {}
        sample: dict[str, Any] = {}
        for col in self.numeric_cols:
            sample[col] = stats.get(col, {}).get("p50", 0.0)
        for col in self.cat_cols:
            counts = stats.get(col, {}).get("value_counts") or {}
            sample[col] = max(counts, key=counts.get) if counts else None
        return sample

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf, compress=3)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ModelBundle":
        obj = joblib.load(io.BytesIO(data))
        if not isinstance(obj, cls):
            raise IncompatibleArtifactError(
                f"Artifact is a {type(obj).__name__}, not a ModelBundle — retrain this model to regenerate it"
            )
        obj._optimise_for_serving()
        return obj

    def _optimise_for_serving(self) -> None:
        """Single-row predictions are slower with thread pools (n_jobs=-1) than without — p95 NFR."""
        try:
            if "n_jobs" in self.model.get_params():
                self.model.set_params(n_jobs=1)
        except Exception:
            pass
