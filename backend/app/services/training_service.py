"""
Training Service
Trains RandomForest, XGBoost, and LogisticRegression/Ridge in parallel on a
shared train/test split and computes comparable metrics.

Pure ML — no storage or DB access. The training worker packages the fitted
models into ModelBundles and persists them.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from app.core.utils import json_safe

logger = logging.getLogger(__name__)


# ─── Model catalogue ──────────────────────────────────────────────────────────
# Unfitted templates — always `clone()` before fitting so concurrent jobs never share state.

CLASSIFICATION_MODELS = {
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBClassifier": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

REGRESSION_MODELS = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Ridge": Ridge(random_state=42),
}

PRIMARY_METRIC = {"classification": "f1", "regression": "r2"}


def get_catalogue(task_type: str) -> dict[str, Any]:
    return CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS


# ─── Metric computation ───────────────────────────────────────────────────────


def _classification_metrics(model, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    latency = (time.perf_counter() - t0) * 1000 / len(X_test)  # ms per sample

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "latency_ms": round(latency, 4),
    }
    # ROC AUC only for binary
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        except Exception:
            pass
    return metrics


def _regression_metrics(model, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    latency = (time.perf_counter() - t0) * 1000 / len(X_test)

    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "latency_ms": round(latency, 4),
    }


# ─── Single model trainer ─────────────────────────────────────────────────────


def _train_single(
    model_name: str,
    template,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> dict:
    """Fit a fresh clone of `template` and compute test metrics."""
    logger.info("Training %s", model_name)
    model = clone(template)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = round(time.perf_counter() - t0, 2)

    if task_type == "classification":
        metrics = _classification_metrics(model, X_test, y_test)
    else:
        metrics = _regression_metrics(model, X_test, y_test)
    metrics["training_time_s"] = training_time
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))

    logger.info("%s → metrics: %s", model_name, metrics)
    return {
        "model_name": model_name,
        "model": model,
        "metrics": json_safe(metrics),
        "params": json_safe(model.get_params()),
        "status": "completed",
        "error": None,
    }


# ─── Parallel training orchestrator ──────────────────────────────────────────


def _split(X: np.ndarray, y: np.ndarray, task_type: str, test_size: float, random_state: int):
    """Stratify classification splits when every class has ≥ 2 rows; otherwise fall back to a plain split."""
    stratify = None
    if task_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        # e.g. test split smaller than the number of classes
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    model_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict]:
    """
    Train the requested models (default: whole catalogue for `task_type`) in parallel.
    Models are fit on plain numpy arrays: XGBoost rejects feature names containing
    `[`, `]` or `<`, which one-hot columns like "income_<=50K" produce.

    Returns one result dict per model; failures are captured, not raised.
    """
    catalogue = get_catalogue(task_type)
    names = model_names or list(catalogue.keys())
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        raise ValueError(f"Unknown {task_type} model(s): {unknown}")

    X_train, X_test, y_train, y_test = _split(
        np.asarray(X, dtype=float), np.asarray(y), task_type, test_size, random_state
    )

    results = []
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {
            pool.submit(_train_single, name, catalogue[name], X_train, X_test, y_train, y_test, task_type): name
            for name in names
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.exception("Training failed for %s", model_name)
                results.append(
                    {
                        "model_name": model_name,
                        "model": None,
                        "metrics": {},
                        "params": {},
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return results


def pick_best_model(results: list[dict], task_type: str) -> dict | None:
    """Return result dict of best model by primary metric."""
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        return None
    key = PRIMARY_METRIC.get(task_type, "f1")
    return max(completed, key=lambda r: r["metrics"].get(key) if r["metrics"].get(key) is not None else -999)
