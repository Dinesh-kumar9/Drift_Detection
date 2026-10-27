"""
Ingestion Service — Phase 1
Handles CSV upload, schema inference, baseline stat computation, and S3 storage.
"""

import asyncio
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.utils import json_safe
from app.models import Dataset
from app.services import storage_service

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """The uploaded file can't be used as a dataset at all."""


# ─── Schema inference ────────────────────────────────────────────────────────


def _infer_schema(df: pd.DataFrame) -> dict:
    """Return per-column type and null info."""
    schema = {}
    for col in df.columns:
        schema[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(float(df[col].isnull().mean()) * 100, 2),
            "unique_count": int(df[col].nunique()),
        }
    return schema


def _compute_baseline_stats(df: pd.DataFrame) -> dict:
    """Compute per-column stats used as drift baseline in Phase 2."""
    stats: dict[str, Any] = {}
    for col in df.select_dtypes(include=["number"]).columns:
        s = df[col].dropna()
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "values_sample": s.sample(min(200, len(s)), random_state=42).tolist(),
        }
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts(normalize=True).head(20)
        stats[col] = {
            "value_counts": vc.to_dict(),
            "unique_count": int(df[col].nunique()),
        }
    return stats


def _validate_csv(df: pd.DataFrame) -> list[str]:
    """Return list of validation warnings (not errors — we're permissive)."""
    issues = []
    if df.empty:
        issues.append("CSV is empty")
    if len(df.columns) < 2:
        issues.append("CSV has fewer than 2 columns")
    high_null_cols = [c for c in df.columns if df[c].isnull().mean() > 0.9]
    if high_null_cols:
        issues.append(f"High null rate (>90%) in columns: {high_null_cols}")
    return issues


# ─── Main service function ────────────────────────────────────────────────────


async def ingest_dataset(
    db: AsyncSession,
    file_bytes: bytes,
    filename: str,
    dataset_name: str,
    owner_id: str | None = None,
) -> Dataset:
    """
    1. Parse CSV
    2. Validate schema (fatal problems raise DatasetValidationError)
    3. Upload raw CSV to MinIO/S3
    4. Compute baseline stats
    5. Persist metadata to Postgres
    Returns the created Dataset ORM object.
    """
    # Parse (CPU-bound — keep it off the event loop)
    try:
        df = await asyncio.to_thread(pd.read_csv, io.BytesIO(file_bytes))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"Could not parse CSV: {exc}") from exc
    logger.info("Parsed CSV: %d rows × %d cols", len(df), len(df.columns))

    # Validate
    issues = _validate_csv(df)
    fatal = [i for i in issues if i in ("CSV is empty", "CSV has fewer than 2 columns")]
    if fatal:
        raise DatasetValidationError("; ".join(fatal))
    if issues:
        logger.warning("Validation warnings for %s: %s", filename, issues)

    # Upload raw CSV to MinIO
    dataset_id = str(uuid.uuid4())
    s3_key = f"raw/{dataset_id}/{filename}"
    s3_path = await asyncio.to_thread(
        storage_service.upload_bytes,
        settings.S3_BUCKET_DATASETS,
        s3_key,
        file_bytes,
        "text/csv",
    )

    # Compute schema + baseline stats (NaN/inf → null so JSONB accepts them)
    schema_json = json_safe(_infer_schema(df))
    baseline_stats = json_safe(await asyncio.to_thread(_compute_baseline_stats, df))

    # Persist to Postgres
    dataset = Dataset(
        id=dataset_id,
        name=dataset_name,
        s3_path=s3_path,
        schema_json=schema_json,
        row_count=len(df),
        baseline_stats=baseline_stats,
        uploaded_at=datetime.now(timezone.utc),
        owner_id=owner_id,
    )
    db.add(dataset)
    await db.flush()
    logger.info("Dataset %s persisted to DB", dataset_id)
    return dataset
