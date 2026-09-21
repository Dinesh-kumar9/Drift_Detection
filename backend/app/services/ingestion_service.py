"""
Ingestion Service — Phase 1
Handles CSV upload, schema inference, baseline stat computation, and S3 storage.
"""

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Dataset
from app.services import storage_service

logger = logging.getLogger(__name__)


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
    2. Validate schema
    3. Upload raw CSV to MinIO/S3
    4. Compute baseline stats
    5. Persist metadata to Postgres
    Returns the created Dataset ORM object.
    """
    # Parse
    df = pd.read_csv(io.BytesIO(file_bytes))
    logger.info("Parsed CSV: %d rows × %d cols", len(df), len(df.columns))

    # Validate
    warnings = _validate_csv(df)
    if warnings:
        logger.warning("Validation warnings for %s: %s", filename, warnings)

    # Upload raw CSV to MinIO
    dataset_id = str(uuid.uuid4())
    s3_key = f"raw/{dataset_id}/{filename}"
    s3_path = storage_service.upload_bytes(
        bucket=settings.S3_BUCKET_DATASETS,
        key=s3_key,
        data=file_bytes,
        content_type="text/csv",
    )

    # Compute schema + baseline stats
    schema_json = _infer_schema(df)
    baseline_stats = _compute_baseline_stats(df)

    # Persist to Postgres
    dataset = Dataset(
        id=dataset_id,
        name=dataset_name,
        s3_path=s3_path,
        schema_json=schema_json,
        row_count=str(len(df)),
        baseline_stats=baseline_stats,
        uploaded_at=datetime.now(timezone.utc),
        owner_id=owner_id,
    )
    db.add(dataset)
    await db.flush()
    logger.info("Dataset %s persisted to DB", dataset_id)
    return dataset
