"""
Training Service — Phase 1
Trains RandomForest, XGBoost, and LogisticRegression/LinearRegression
in parallel, logs metrics, saves artifacts to MinIO.
"""

import io
import logging
import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from app.core.config import settings
from app.services import storage_service

logger = logging.getLogger(__name__)


# ─── Model catalogue ──────────────────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBClassifier": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                                    use_label_encoder=False, verbosity=0),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

REGRESSION_MODELS = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Ridge": Ridge(random_state=42),
}


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
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    dataset_id: str,
    experiment_group_id: str,
) -> dict:
    """Train one model, compute metrics, save artifact. Returns result dict."""
    run_id = str(uuid.uuid4())
    logger.info("Training %s (run_id=%s)", model_name, run_id)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = round(time.perf_counter() - t0, 2)

    # Metrics
    if task_type == "classification":
        metrics = _classification_metrics(model, X_test, y_test)
    else:
        metrics = _regression_metrics(model, X_test, y_test)
    metrics["training_time_s"] = training_time

    # Save artifact to MinIO
    artifact_key = f"models/{dataset_id}/{experiment_group_id}/{run_id}/{model_name}.pkl"
    model_bytes = pickle.dumps(model)
    artifact_s3_path = storage_service.upload_bytes(
        bucket=settings.S3_BUCKET_MODELS,
        key=artifact_key,
        data=model_bytes,
        content_type="application/octet-stream",
    )

    logger.info("%s → metrics: %s", model_name, metrics)
    return {
        "run_id": run_id,
        "model_name": model_name,
        "model": model,
        "metrics": metrics,
        "artifact_s3_path": artifact_s3_path,
        "params": model.get_params(),
        "status": "completed",
    }


# ─── Parallel training orchestrator ──────────────────────────────────────────

def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    dataset_id: str,
    experiment_group_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict]:
    """
    Train all models in parallel using ThreadPoolExecutor.
    Returns list of result dicts (one per model).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    catalogue = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                _train_single,
                name, model, X_train, X_test, y_train, y_test,
                task_type, dataset_id, experiment_group_id,
            ): name
            for name, model in catalogue.items()
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                model_name = futures[future]
                logger.error("Training failed for %s: %s", model_name, exc)
                results.append({
                    "run_id": str(uuid.uuid4()),
                    "model_name": model_name,
                    "metrics": {},
                    "artifact_s3_path": None,
                    "params": {},
                    "status": "failed",
                    "error": str(exc),
                })

    return results


def pick_best_model(results: list[dict], task_type: str) -> dict | None:
    """Return result dict of best model by primary metric."""
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        return None
    key = "f1" if task_type == "classification" else "r2"
    return max(completed, key=lambda r: r["metrics"].get(key, -999))
