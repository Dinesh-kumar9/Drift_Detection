"""
Preprocessing Service — Phase 1
Versioned pipeline: missing value imputation → encoding → scaling.
Saves preprocessed dataset + pipeline config to MinIO.
"""

import io
import json
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.core.config import settings
from app.services import storage_service

logger = logging.getLogger(__name__)


# ─── Feature detection ────────────────────────────────────────────────────────


def _detect_columns(df: pd.DataFrame, target_col: str):
    """Split columns into numeric and categorical (excluding target)."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, cat_cols


def _detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Auto-detect classification vs regression from target column."""
    target = df[target_col]
    n_unique = target.nunique()
    if target.dtype == object or n_unique <= 20:
        return "classification"
    return "regression"


# ─── Pipeline builder ─────────────────────────────────────────────────────────


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, dict, Any]:
    """
    Returns: (X_processed_df, y_series, config_dict, fitted_transformer)
    """
    numeric_cols, cat_cols = _detect_columns(df, target_col)

    # Build sklearn ColumnTransformer
    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numeric_cols))
    if cat_cols:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if classification
    task_type = _detect_task_type(df, target_col)
    if task_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
    else:
        le = None

    X_transformed = ct.fit_transform(X)

    # Reconstruct column names after transformation
    output_cols = []
    if numeric_cols:
        output_cols.extend(numeric_cols)
    if cat_cols:
        output_cols.extend(cat_cols)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_df = pd.DataFrame(X_transformed, columns=output_cols[: X_transformed.shape[1]])

    config = {
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "task_type": task_type,
        "n_features": X_df.shape[1],
        "n_rows": len(df),
        "label_encoder": le is not None,
    }

    transformer_obj = {"ct": ct, "le": le}
    return X_df, y, config, transformer_obj


# ─── Storage helpers ──────────────────────────────────────────────────────────


def save_preprocessed(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    transformer_obj: Any,
    dataset_id: str,
    experiment_group_id: str,
) -> dict[str, str]:
    """Save preprocessed features + pipeline to MinIO. Returns S3 paths."""
    prefix = f"preprocessed/{dataset_id}/{experiment_group_id}"

    # Save CSV of preprocessed features
    buf = io.BytesIO()
    X.to_csv(buf, index=False)
    x_path = storage_service.upload_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/X.csv", buf.getvalue(), "text/csv")

    # Save target series
    buf = io.BytesIO()
    y.to_csv(buf, index=False, header=True)
    y_path = storage_service.upload_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/y.csv", buf.getvalue(), "text/csv")

    # Save fitted transformer (for Phase 2 — needed to apply to new data)
    transformer_bytes = pickle.dumps(transformer_obj)
    t_path = storage_service.upload_bytes(
        settings.S3_BUCKET_MODELS, f"{prefix}/transformer.pkl", transformer_bytes, "application/octet-stream"
    )

    return {"X": x_path, "y": y_path, "transformer": t_path}


def load_preprocessed(dataset_id: str, experiment_group_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load X, y from MinIO."""
    prefix = f"preprocessed/{dataset_id}/{experiment_group_id}"
    X_bytes = storage_service.download_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/X.csv")
    y_bytes = storage_service.download_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/y.csv")
    X = pd.read_csv(io.BytesIO(X_bytes))
    y = pd.read_csv(io.BytesIO(y_bytes)).iloc[:, 0]
    return X, y
