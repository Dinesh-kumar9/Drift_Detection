"""
Unit tests for preprocessing service.
Tests pipeline building, task-type detection, and column classification.
No DB or MinIO required — fully isolated.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing_service import _detect_columns, _detect_task_type, build_preprocessing_pipeline


@pytest.fixture
def clf_df():
    """Binary classification dataset (credit-card style)."""
    np.random.seed(0)
    n = 500
    return pd.DataFrame(
        {
            "V1": np.random.randn(n),
            "V2": np.random.randn(n),
            "Amount": np.abs(np.random.randn(n)) * 50,
            "Category": np.random.choice(["A", "B"], n),
            "Class": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def reg_df():
    """Regression dataset."""
    np.random.seed(1)
    n = 300
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n) * 2,
            "feature_c": np.abs(np.random.randn(n)),
            "target": np.random.randn(n) * 10 + 5,  # many unique floats → regression
        }
    )


class TestDetectColumns:
    def test_numeric_cols_detected(self, clf_df):
        num, cat = _detect_columns(clf_df, "Class")
        assert "V1" in num
        assert "V2" in num
        assert "Amount" in num

    def test_categorical_cols_detected(self, clf_df):
        num, cat = _detect_columns(clf_df, "Class")
        assert "Category" in cat

    def test_target_excluded(self, clf_df):
        num, cat = _detect_columns(clf_df, "Class")
        assert "Class" not in num
        assert "Class" not in cat

    def test_all_numeric_df(self, reg_df):
        num, cat = _detect_columns(reg_df, "target")
        assert "feature_a" in num
        assert len(cat) == 0


class TestDetectTaskType:
    def test_binary_target_is_classification(self, clf_df):
        task = _detect_task_type(clf_df, "Class")
        assert task == "classification"

    def test_string_target_is_classification(self, clf_df):
        clf_df["label"] = np.where(clf_df["Class"] == 1, "fraud", "normal")
        task = _detect_task_type(clf_df, "label")
        assert task == "classification"

    def test_continuous_float_target_is_regression(self, reg_df):
        task = _detect_task_type(reg_df, "target")
        assert task == "regression"

    def test_many_unique_ints_is_regression(self):
        df = pd.DataFrame({"x": range(1000), "y": range(100, 1100)})
        task = _detect_task_type(df, "y")
        assert task == "regression"


class TestBuildPreprocessingPipeline:
    def test_returns_correct_shapes(self, clf_df):
        X, y, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        assert len(X) == len(clf_df)
        assert len(y) == len(clf_df)

    def test_target_not_in_features(self, clf_df):
        X, y, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        assert "Class" not in X.columns

    def test_no_nulls_after_preprocessing(self, clf_df):
        # Inject some nulls
        clf_df.loc[:10, "V1"] = None
        clf_df.loc[5:15, "Amount"] = None
        X, y, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        assert X.isnull().sum().sum() == 0

    def test_config_has_task_type(self, clf_df):
        _, _, config, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert "task_type" in config
        assert config["task_type"] == "classification"

    def test_config_has_feature_count(self, clf_df):
        _, _, config, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert "n_features" in config
        assert config["n_features"] > 0

    def test_regression_pipeline(self, reg_df):
        X, y, config, transformer = build_preprocessing_pipeline(reg_df, "target")
        assert config["task_type"] == "regression"
        assert len(X) == len(reg_df)
        assert X.isnull().sum().sum() == 0

    def test_transformer_obj_has_ct_key(self, clf_df):
        _, _, _, transformer_obj = build_preprocessing_pipeline(clf_df, "Class")
        assert "ct" in transformer_obj
