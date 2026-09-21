"""
Unit tests for training service.
Tests metric computation, model catalogue, and best-model selection.
No DB or MinIO required — uses mocking for storage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from app.services.training_service import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    _classification_metrics,
    _regression_metrics,
    pick_best_model,
)


@pytest.fixture
def simple_clf_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = pd.Series(np.random.choice([0, 1], n))
    return X, y


@pytest.fixture
def simple_reg_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = pd.Series(np.random.randn(n) * 5 + 10)
    return X, y


@pytest.fixture
def trained_clf(simple_clf_data):
    X, y = simple_clf_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_reg(simple_reg_data):
    X, y = simple_reg_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


class TestClassificationMetrics:
    def test_returns_accuracy(self, trained_clf):
        model, X, y = trained_clf
        metrics = _classification_metrics(model, X, y)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_returns_f1(self, trained_clf):
        model, X, y = trained_clf
        metrics = _classification_metrics(model, X, y)
        assert "f1" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_returns_latency(self, trained_clf):
        model, X, y = trained_clf
        metrics = _classification_metrics(model, X, y)
        assert "latency_ms" in metrics
        assert metrics["latency_ms"] >= 0

    def test_returns_roc_auc_for_binary(self, trained_clf):
        model, X, y = trained_clf
        metrics = _classification_metrics(model, X, y)
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_precision_and_recall_present(self, trained_clf):
        model, X, y = trained_clf
        metrics = _classification_metrics(model, X, y)
        assert "precision" in metrics
        assert "recall" in metrics


class TestRegressionMetrics:
    def test_returns_mae(self, trained_reg):
        model, X, y = trained_reg
        metrics = _regression_metrics(model, X, y)
        assert "mae" in metrics
        assert metrics["mae"] >= 0

    def test_returns_rmse(self, trained_reg):
        model, X, y = trained_reg
        metrics = _regression_metrics(model, X, y)
        assert "rmse" in metrics
        assert metrics["rmse"] >= 0

    def test_returns_r2(self, trained_reg):
        model, X, y = trained_reg
        metrics = _regression_metrics(model, X, y)
        assert "r2" in metrics

    def test_returns_latency(self, trained_reg):
        model, X, y = trained_reg
        metrics = _regression_metrics(model, X, y)
        assert "latency_ms" in metrics
        assert metrics["latency_ms"] >= 0


class TestPickBestModel:
    def test_picks_best_f1_for_classification(self):
        results = [
            {"run_id": "1", "model_name": "RF", "metrics": {"f1": 0.85}, "status": "completed"},
            {"run_id": "2", "model_name": "XGB", "metrics": {"f1": 0.92}, "status": "completed"},
            {"run_id": "3", "model_name": "LR", "metrics": {"f1": 0.78}, "status": "completed"},
        ]
        best = pick_best_model(results, "classification")
        assert best["model_name"] == "XGB"

    def test_picks_best_r2_for_regression(self):
        results = [
            {"run_id": "1", "model_name": "RF", "metrics": {"r2": 0.72}, "status": "completed"},
            {"run_id": "2", "model_name": "Ridge", "metrics": {"r2": 0.81}, "status": "completed"},
        ]
        best = pick_best_model(results, "regression")
        assert best["model_name"] == "Ridge"

    def test_ignores_failed_runs(self):
        results = [
            {"run_id": "1", "model_name": "RF", "metrics": {"f1": 0.95}, "status": "failed"},
            {"run_id": "2", "model_name": "XGB", "metrics": {"f1": 0.80}, "status": "completed"},
        ]
        best = pick_best_model(results, "classification")
        assert best["model_name"] == "XGB"

    def test_returns_none_if_all_failed(self):
        results = [
            {"run_id": "1", "model_name": "RF", "metrics": {}, "status": "failed"},
        ]
        best = pick_best_model(results, "classification")
        assert best is None

    def test_returns_none_if_empty(self):
        assert pick_best_model([], "classification") is None


class TestModelCatalogue:
    def test_classification_has_three_models(self):
        assert len(CLASSIFICATION_MODELS) == 3

    def test_regression_has_three_models(self):
        assert len(REGRESSION_MODELS) == 3

    def test_all_classifiers_have_fit(self):
        for name, model in CLASSIFICATION_MODELS.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"

    def test_all_regressors_have_fit(self):
        for name, model in REGRESSION_MODELS.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
