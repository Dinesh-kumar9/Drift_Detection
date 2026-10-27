"""
Preprocessing Service
Versioned pipeline: missing value imputation → one-hot encoding → scaling.

The fitted ColumnTransformer is shipped inside every model bundle
(see model_bundle.py), so serving applies exactly the training-time transform
and the S2 drift monitor sees exactly the features the model sees.
"""

import io
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.core.config import settings
from app.core.utils import json_safe
from app.services import storage_service

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "2"
MAX_CATEGORIES = 20  # per categorical column; rarer values collapse into one "infrequent" column
CLASSIFICATION_MAX_UNIQUE = 20  # numeric targets with ≤ this many distinct values are treated as classes
TASK_TYPES = ("classification", "regression")


class PreprocessingError(ValueError):
    """The dataset can't be turned into a training matrix (bad target, no usable features, …)."""


# ─── Feature detection ────────────────────────────────────────────────────────


def _detect_columns(df: pd.DataFrame, target_col: str):
    """Split columns into numeric and categorical (excluding target)."""
    feature_cols = [c for c in df.columns if c != target_col]
    features = df[feature_cols]
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    return numeric_cols, cat_cols


def _detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Auto-detect classification vs regression from target column."""
    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        return "classification"
    return "classification" if target.nunique() <= CLASSIFICATION_MAX_UNIQUE else "regression"


def detect_task_type_from_schema(column_info: dict) -> str:
    """
    Same rule as `_detect_task_type`, evaluated on the dataset's stored schema
    (dtype + unique_count) so the API can resolve the task type — and hence
    the model catalogue — before the CSV is ever reloaded.
    """
    dtype = str(column_info.get("dtype", "object"))
    is_numeric = dtype.startswith(("int", "uint", "float", "Int", "UInt", "Float"))
    if not is_numeric:
        return "classification"
    return "classification" if column_info.get("unique_count", 0) <= CLASSIFICATION_MAX_UNIQUE else "regression"


# ─── Input coercion (shared by training and serving) ─────────────────────────


def coerce_inputs(df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """
    Normalise raw feature columns so fit-time and serve-time inputs look identical:
    numeric → float (unparseable → NaN), categorical → str (missing stays NaN).
    OneHotEncoder rejects columns that mix strings and numbers, which raw CSVs
    and JSON payloads produce all the time.
    """
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in cat_cols:
        series = df[col]
        # None (from JSON) and NaN (from CSV) both become np.nan — the only marker SimpleImputer treats as missing
        out[col] = series.astype(str).where(series.notna(), np.nan).astype(object)
    return out


# ─── Pipeline builder ─────────────────────────────────────────────────────────


def _build_column_transformer(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
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
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",  # unseen categories at serve time don't crash
                        max_categories=MAX_CATEGORIES,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "auto",
) -> tuple[pd.DataFrame, pd.Series, dict, Any]:
    """
    Fit the preprocessing pipeline on `df`.

    Returns: (X_processed_df, y_series, config_dict, transformer_obj)
      transformer_obj = {"ct": fitted ColumnTransformer, "le": LabelEncoder | None}
    """
    if target_col not in df.columns:
        raise PreprocessingError(f"Target column '{target_col}' not found in dataset")

    n_input_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if df.empty:
        raise PreprocessingError(f"Target column '{target_col}' has no non-null values")

    resolved_task = _detect_task_type(df, target_col) if task_type == "auto" else task_type
    if resolved_task not in TASK_TYPES:
        raise PreprocessingError(f"Unknown task type '{task_type}'")

    numeric_cols, cat_cols = _detect_columns(df, target_col)
    if not numeric_cols and not cat_cols:
        raise PreprocessingError("Dataset has no usable feature columns besides the target")

    X_raw = coerce_inputs(df, numeric_cols, cat_cols)
    y = df[target_col]

    # Encode every classification target to 0..k-1 — XGBoost rejects anything else,
    # and the encoder lets serving map predictions back to the original labels.
    le = None
    if resolved_task == "classification":
        le = LabelEncoder()
        try:
            y = pd.Series(le.fit_transform(y), name=target_col)
        except TypeError as exc:  # mixed str/number labels
            raise PreprocessingError(f"Target column '{target_col}' mixes text and numeric labels") from exc
        if len(le.classes_) < 2:
            raise PreprocessingError(f"Target column '{target_col}' has a single class — nothing to learn")
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().any():
            raise PreprocessingError(f"Target column '{target_col}' must be numeric for regression")
        y = y.astype(float).rename(target_col)

    ct = _build_column_transformer(numeric_cols, cat_cols)
    X_transformed = ct.fit_transform(X_raw)
    feature_names = [str(n) for n in ct.get_feature_names_out()]
    X_df = pd.DataFrame(np.asarray(X_transformed, dtype=float), columns=feature_names)

    config = {
        "version": PREPROCESSING_VERSION,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "task_type": resolved_task,
        "feature_names": feature_names,
        "n_features": X_df.shape[1],
        "n_rows": len(df),
        "dropped_null_target_rows": n_input_rows - len(df),
        "label_encoder": le is not None,
        "classes": json_safe(le.classes_.tolist()) if le is not None else None,
        "steps": {
            "numeric": ["median_impute", "standard_scale"],
            "categorical": ["most_frequent_impute", f"one_hot(max_categories={MAX_CATEGORIES})"],
        },
    }

    transformer_obj = {"ct": ct, "le": le}
    return X_df, y, config, transformer_obj


def transform_features(
    ct: ColumnTransformer,
    raw_df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Apply a fitted transformer to raw rows (serving / drift S2 stage)."""
    X_raw = coerce_inputs(raw_df, numeric_cols, cat_cols)
    X_transformed = ct.transform(X_raw)
    return pd.DataFrame(np.asarray(X_transformed, dtype=float), columns=[str(n) for n in ct.get_feature_names_out()])


# ─── Storage helpers ──────────────────────────────────────────────────────────


def save_preprocessed(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    dataset_id: str,
    experiment_group_id: str,
) -> dict[str, str]:
    """Persist the preprocessed matrix + config (the "data version" of a run). Returns S3 paths."""
    prefix = f"preprocessed/{dataset_id}/{experiment_group_id}"

    buf = io.BytesIO()
    X.to_csv(buf, index=False)
    x_path = storage_service.upload_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/X.csv", buf.getvalue(), "text/csv")

    buf = io.BytesIO()
    y.to_csv(buf, index=False, header=True)
    y_path = storage_service.upload_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/y.csv", buf.getvalue(), "text/csv")

    config_path = storage_service.upload_bytes(
        settings.S3_BUCKET_DATASETS,
        f"{prefix}/config.json",
        json.dumps(json_safe(config), indent=2).encode(),
        "application/json",
    )
    return {"X": x_path, "y": y_path, "config": config_path}


def load_preprocessed(dataset_id: str, experiment_group_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load X, y from MinIO."""
    prefix = f"preprocessed/{dataset_id}/{experiment_group_id}"
    X_bytes = storage_service.download_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/X.csv")
    y_bytes = storage_service.download_bytes(settings.S3_BUCKET_DATASETS, f"{prefix}/y.csv")
    X = pd.read_csv(io.BytesIO(X_bytes))
    y = pd.read_csv(io.BytesIO(y_bytes)).iloc[:, 0]
    return X, y
