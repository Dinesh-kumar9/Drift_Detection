"""
Serving / Inference Service
Loads model bundles from MinIO/S3, runs inference through the training-time
preprocessing pipeline, and logs every prediction with all three observability
stages (raw input, transformed features, output).
"""

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.utils import json_safe
from app.models import ExperimentRun, ModelVersion, Prediction
from app.services import storage_service
from app.services.model_bundle import ModelBundle

logger = logging.getLogger(__name__)


class ModelNotReadyError(ValueError):
    """Model version exists but has no artifact (failed or unfinished run)."""


# In-process LRU cache of deserialised bundles, keyed by model_version_id
_BUNDLE_CACHE: "OrderedDict[str, ModelBundle]" = OrderedDict()


def _cache_put(model_version_id: str, bundle: ModelBundle) -> None:
    _BUNDLE_CACHE[model_version_id] = bundle
    _BUNDLE_CACHE.move_to_end(model_version_id)
    while len(_BUNDLE_CACHE) > settings.MODEL_CACHE_SIZE:
        _BUNDLE_CACHE.popitem(last=False)


def invalidate_cache(model_version_id: str | None = None) -> None:
    """Drop one bundle (or all) from the in-process cache."""
    if model_version_id is None:
        _BUNDLE_CACHE.clear()
    else:
        _BUNDLE_CACHE.pop(model_version_id, None)


async def load_bundle(db: AsyncSession, model_version_id: str) -> ModelBundle:
    """Load a model bundle from cache or object storage."""
    cached = _BUNDLE_CACHE.get(model_version_id)
    if cached is not None:
        _BUNDLE_CACHE.move_to_end(model_version_id)
        return cached

    result = await db.execute(
        select(ExperimentRun.artifact_s3_path)
        .join(ModelVersion, ModelVersion.run_id == ExperimentRun.id)
        .where(ModelVersion.id == model_version_id)
    )
    artifact_path = result.scalar()
    if not artifact_path:
        raise ModelNotReadyError(f"No artifact found for model version {model_version_id}")

    data = await asyncio.to_thread(storage_service.download_uri, artifact_path)
    bundle = await asyncio.to_thread(ModelBundle.from_bytes, data)
    _cache_put(model_version_id, bundle)
    logger.info("Loaded model bundle %s from %s", model_version_id, artifact_path)
    return bundle


async def run_inference(
    db: AsyncSession,
    model_version_id: str,
    features: dict[str, Any],
) -> dict:
    """
    Run inference on the given model version and log the prediction.
    Raises SchemaValidationError if required feature columns are missing.
    """
    bundle = await load_bundle(db, model_version_id)

    t0 = time.perf_counter()
    result = (await asyncio.to_thread(bundle.predict, [features]))[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    output = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }
    pred_record = Prediction(
        model_version_id=model_version_id,
        input_snapshot=json_safe(features),  # S1
        features_snapshot=result["features"],  # S2
        output=json_safe(output),  # S3
        latency_ms=latency_ms,
        predicted_at=datetime.now(timezone.utc),
    )
    db.add(pred_record)
    await db.flush()

    return {
        "model_version_id": model_version_id,
        "prediction_id": pred_record.id,
        "latency_ms": latency_ms,
        **json_safe(output),
    }
