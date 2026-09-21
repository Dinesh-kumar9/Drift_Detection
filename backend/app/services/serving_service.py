"""
Serving / Inference Service — Phase 1
Loads the production model from MinIO, runs inference, logs every prediction.
"""

import io
import logging
import pickle
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ExperimentRun, ModelVersion, Prediction
from app.services import storage_service

logger = logging.getLogger(__name__)

# In-process model cache (keyed by model_version_id)
_MODEL_CACHE: dict[str, Any] = {}


async def load_model(db: AsyncSession, model_version_id: str) -> Any:
    """Load model from cache or MinIO. Returns fitted sklearn/XGB model."""
    if model_version_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_version_id]

    # Fetch artifact path from DB
    result = await db.execute(
        select(ModelVersion, ExperimentRun)
        .join(ExperimentRun, ModelVersion.run_id == ExperimentRun.id)
        .where(ModelVersion.id == model_version_id)
    )
    row = result.first()
    if not row:
        raise ValueError(f"Model version {model_version_id} not found")
    mv, run = row

    if not run.artifact_s3_path:
        raise ValueError(f"No artifact found for model version {model_version_id}")

    # Parse s3://bucket/key
    s3_path = run.artifact_s3_path
    bucket, key = _parse_s3_path(s3_path)
    model_bytes = storage_service.download_bytes(bucket, key)
    model = pickle.loads(model_bytes)

    _MODEL_CACHE[model_version_id] = model
    logger.info("Loaded model %s from %s", model_version_id, s3_path)
    return model


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse 's3://bucket/key/path' → ('bucket', 'key/path')."""
    path = s3_path.replace("s3://", "")
    bucket, _, key = path.partition("/")
    return bucket, key


def invalidate_cache(model_version_id: str) -> None:
    """Remove model from in-process cache (called after promotion)."""
    _MODEL_CACHE.pop(model_version_id, None)


async def run_inference(
    db: AsyncSession,
    model_version_id: str,
    features: dict[str, Any],
) -> dict:
    """
    Run inference on the given model version.
    Logs every prediction to the `predictions` table.
    Returns prediction result dict.
    """
    model = await load_model(db, model_version_id)

    # Convert feature dict → DataFrame (preserves column order)
    X = pd.DataFrame([features])

    t0 = time.perf_counter()
    prediction_raw = model.predict(X)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    prediction_value = prediction_raw[0]
    if hasattr(prediction_value, "item"):
        prediction_value = prediction_value.item()

    # Confidence (proba for classifiers)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            confidence = round(float(np.max(proba)), 4)
        except Exception:
            pass

    # Log to DB
    prediction_id = str(uuid.uuid4())
    pred_record = Prediction(
        id=prediction_id,
        model_version_id=model_version_id,
        input_snapshot=features,
        output={"prediction": prediction_value, "confidence": confidence},
        latency_ms=latency_ms,
        predicted_at=datetime.now(timezone.utc),
    )
    db.add(pred_record)
    await db.flush()

    return {
        "model_version_id": model_version_id,
        "prediction": prediction_value,
        "confidence": confidence,
        "prediction_id": prediction_id,
        "latency_ms": latency_ms,
    }
