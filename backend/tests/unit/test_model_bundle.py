"""
Unit tests for ModelBundle — the deployable artifact that keeps serving-time
preprocessing identical to training. No DB or MinIO required.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.services.ingestion_service import _compute_baseline_stats
from app.services.model_bundle import IncompatibleArtifactError, ModelBundle, SchemaValidationError
from app.services.preprocessing_service import build_preprocessing_pipeline
from app.services.training_service import train_all_models


def _make_bundle(df: pd.DataFrame, target: str, model_name: str) -> ModelBundle:
    X, y, config, transformer = build_preprocessing_pipeline(df, target)
    result = train_all_models(X, y, config["task_type"], model_names=[model_name])[0]
    assert result["status"] == "completed", result["error"]
    return ModelBundle(
        model_type=model_name,
        task_type=config["task_type"],
        target_column=target,
        numeric_cols=config["numeric_cols"],
        cat_cols=config["cat_cols"],
        feature_names=config["feature_names"],
        preprocessor=transformer["ct"],
        model=result["model"],
        label_encoder=transformer["le"],
    )


@pytest.fixture(scope="module")
def clf_df():
    rng = np.random.default_rng(3)
    n = 400
    amount = rng.exponential(100, n)
    return pd.DataFrame(
        {
            "amount": amount,
            "age": rng.integers(18, 80, n),
            "channel": rng.choice(["web", "pos", "atm"], n),
            "label": np.where(amount > 150, "fraud", "legit"),
        }
    )


@pytest.fixture(scope="module")
def clf_bundle(clf_df):
    return _make_bundle(clf_df, "label", "RandomForestClassifier")


@pytest.fixture(scope="module")
def reg_bundle():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "kind": rng.choice(["a", "b"], n), "y": 3 * x + rng.normal(scale=0.1, size=n)})
    return _make_bundle(df, "y", "Ridge")


class TestPredict:
    def test_classification_returns_decoded_label(self, clf_bundle):
        [out] = clf_bundle.predict([{"amount": 900.0, "age": 40, "channel": "web"}])
        assert out["prediction"] == "fraud"
        assert 0.5 <= out["confidence"] <= 1.0
        assert set(out["probabilities"]) == {"fraud", "legit"}

    def test_exposes_transformed_features_for_s2(self, clf_bundle):
        [out] = clf_bundle.predict([{"amount": 10.0, "age": 30, "channel": "pos"}])
        assert set(out["features"]) == set(clf_bundle.feature_names)
        assert out["features"]["channel_pos"] == 1.0

    def test_raw_inputs_are_scaled_like_training(self, clf_bundle):
        """Regression test: serving used to feed raw values to a model trained on scaled ones."""
        [low] = clf_bundle.predict([{"amount": 5.0, "age": 30, "channel": "web"}])
        [high] = clf_bundle.predict([{"amount": 5000.0, "age": 30, "channel": "web"}])
        assert low["prediction"] == "legit"
        assert high["prediction"] == "fraud"

    def test_regression_returns_float(self, reg_bundle):
        [out] = reg_bundle.predict([{"x": 1.0, "kind": "a"}])
        assert isinstance(out["prediction"], float)
        assert out["prediction"] == pytest.approx(3.0, abs=0.5)
        assert out["confidence"] is None

    def test_missing_column_raises(self, clf_bundle):
        with pytest.raises(SchemaValidationError) as exc:
            clf_bundle.predict([{"amount": 1.0}])
        assert exc.value.missing_columns == ["age", "channel"]

    def test_null_values_are_imputed(self, clf_bundle):
        [out] = clf_bundle.predict([{"amount": None, "age": None, "channel": None}])
        assert out["prediction"] in ("fraud", "legit")

    def test_extra_columns_ignored(self, clf_bundle):
        [out] = clf_bundle.predict([{"amount": 1.0, "age": 20, "channel": "atm", "label": "legit", "junk": 1}])
        assert out["prediction"] in ("fraud", "legit")


class TestSampleInput:
    def test_sample_input_from_baseline_is_predictable(self, clf_bundle, clf_df):
        sample = clf_bundle.sample_input(_compute_baseline_stats(clf_df))
        assert set(sample) == {"amount", "age", "channel"}
        assert clf_bundle.predict([sample])[0]["prediction"] in ("fraud", "legit")

    def test_sample_input_without_stats(self, clf_bundle):
        sample = clf_bundle.sample_input(None)
        assert clf_bundle.predict([sample])


class TestSerialisation:
    def test_round_trip(self, clf_bundle):
        restored = ModelBundle.from_bytes(clf_bundle.to_bytes())
        row = {"amount": 900.0, "age": 40, "channel": "web"}
        assert restored.predict([row])[0]["prediction"] == clf_bundle.predict([row])[0]["prediction"]

    def test_loaded_bundle_uses_single_thread(self, clf_bundle):
        restored = ModelBundle.from_bytes(clf_bundle.to_bytes())
        assert restored.model.get_params()["n_jobs"] == 1

    def test_bare_model_artifact_rejected(self):
        legacy = pickle.dumps(LogisticRegression())
        with pytest.raises(IncompatibleArtifactError):
            ModelBundle.from_bytes(legacy)
