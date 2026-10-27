"""
Unit tests for preprocessing service.
Tests pipeline building, task-type detection, encoding and serve-time transforms.
No DB or MinIO required — fully isolated.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.services.preprocessing_service import (
    PreprocessingError,
    _detect_columns,
    _detect_task_type,
    build_preprocessing_pipeline,
    detect_task_type_from_schema,
    transform_features,
)


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

    def test_bool_cols_are_categorical(self, clf_df):
        clf_df["flag"] = clf_df["V1"] > 0
        _, cat = _detect_columns(clf_df, "Class")
        assert "flag" in cat

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

    @pytest.mark.parametrize("target", ["Class", "label", "target"])
    def test_schema_rule_matches_dataframe_rule(self, clf_df, reg_df, target):
        """The API resolves task type from stored schema; it must agree with the worker's rule."""
        clf_df["label"] = np.where(clf_df["Class"] == 1, "fraud", "normal")
        df = reg_df if target == "target" else clf_df
        info = {"dtype": str(df[target].dtype), "unique_count": int(df[target].nunique())}
        assert detect_task_type_from_schema(info) == _detect_task_type(df, target)


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
        clf_df.loc[20:25, "Category"] = None
        X, y, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        assert X.isnull().sum().sum() == 0

    def test_categorical_columns_are_one_hot_encoded(self, clf_df):
        X, _, config, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert "Category" not in X.columns
        assert {"Category_A", "Category_B"} <= set(X.columns)
        assert all(dtype.kind == "f" for dtype in X.dtypes)
        assert config["feature_names"] == list(X.columns)

    def test_output_is_trainable(self, clf_df):
        """Regression test: string categories used to reach the estimators unencoded."""
        X, y, _, _ = build_preprocessing_pipeline(clf_df, "Class")
        RandomForestClassifier(n_estimators=5).fit(X, y)
        LogisticRegression(max_iter=200).fit(X, y)

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
        assert transformer["le"] is None

    def test_transformer_obj_has_ct_key(self, clf_df):
        _, _, _, transformer_obj = build_preprocessing_pipeline(clf_df, "Class")
        assert "ct" in transformer_obj

    def test_numeric_labels_not_starting_at_zero_are_encoded(self, clf_df):
        """XGBoost requires classes 0..k-1; labels {1,2,3} used to crash it."""
        clf_df["grade"] = np.random.choice([1, 2, 3], len(clf_df))
        _, y, config, transformer = build_preprocessing_pipeline(clf_df, "grade")
        assert sorted(y.unique()) == [0, 1, 2]
        assert config["classes"] == [1, 2, 3]
        assert transformer["le"] is not None

    def test_string_labels_are_encoded(self, clf_df):
        clf_df["label"] = np.where(clf_df["Class"] == 1, "fraud", "normal")
        _, y, config, _ = build_preprocessing_pipeline(clf_df, "label")
        assert set(y.unique()) == {0, 1}
        assert config["classes"] == ["fraud", "normal"]

    def test_rows_with_null_target_are_dropped(self, clf_df):
        clf_df["Class"] = clf_df["Class"].astype(float)
        clf_df.loc[:9, "Class"] = None
        X, y, config, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert len(X) == len(y) == len(clf_df) - 10
        assert config["dropped_null_target_rows"] == 10

    def test_missing_target_column_raises(self, clf_df):
        with pytest.raises(PreprocessingError, match="not found"):
            build_preprocessing_pipeline(clf_df, "nope")

    def test_single_class_target_raises(self, clf_df):
        clf_df["Class"] = 1
        with pytest.raises(PreprocessingError, match="single class"):
            build_preprocessing_pipeline(clf_df, "Class", task_type="classification")

    def test_non_numeric_regression_target_raises(self, clf_df):
        clf_df["label"] = "x"
        with pytest.raises(PreprocessingError, match="numeric"):
            build_preprocessing_pipeline(clf_df, "label", task_type="regression")

    def test_mixed_type_categorical_column(self, clf_df):
        clf_df["Mixed"] = [1 if i % 2 else "one" for i in range(len(clf_df))]
        X, _, _, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert X.isnull().sum().sum() == 0

    def test_xgboost_unsafe_category_values_survive(self, clf_df):
        clf_df["income"] = np.random.choice(["<=50K", ">50K"], len(clf_df))
        X, _, config, _ = build_preprocessing_pipeline(clf_df, "Class")
        assert "income_<=50K" in config["feature_names"]


class TestTransformFeatures:
    def test_serve_time_transform_matches_training(self, clf_df):
        X, _, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        raw = clf_df.drop(columns=["Class"]).head(5)
        out = transform_features(transformer["ct"], raw, config["numeric_cols"], config["cat_cols"])
        np.testing.assert_allclose(out.to_numpy(), X.head(5).to_numpy())

    def test_unseen_category_does_not_crash(self, clf_df):
        _, _, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        raw = pd.DataFrame([{"V1": 0.1, "V2": 0.2, "Amount": 10.0, "Category": "NEVER_SEEN"}])
        out = transform_features(transformer["ct"], raw, config["numeric_cols"], config["cat_cols"])
        assert out[["Category_A", "Category_B"]].sum(axis=1).iloc[0] == 0

    def test_none_and_string_numbers_from_json(self, clf_df):
        _, _, config, transformer = build_preprocessing_pipeline(clf_df, "Class")
        raw = pd.DataFrame([{"V1": None, "V2": "0.5", "Amount": 10, "Category": None}])
        out = transform_features(transformer["ct"], raw, config["numeric_cols"], config["cat_cols"])
        assert out.isnull().sum().sum() == 0
