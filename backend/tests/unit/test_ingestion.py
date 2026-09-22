"""
Unit tests for ingestion service.
Tests CSV parsing, schema inference, and baseline stat computation.
No DB or MinIO required — fully isolated.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ingestion_service import _compute_baseline_stats, _infer_schema, _validate_csv


@pytest.fixture
def sample_df():
    """Credit-card-fraud-style DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame(
        {
            "V1": np.random.randn(n),
            "V2": np.random.randn(n),
            "V3": np.random.randn(n),
            "Amount": np.abs(np.random.randn(n)) * 100,
            "Time": np.arange(n, dtype=float),
            "Category": np.random.choice(["A", "B", "C"], n),
            "Class": np.random.choice([0, 1], n, p=[0.998, 0.002]),
        }
    )


class TestInferSchema:
    def test_returns_all_columns(self, sample_df):
        schema = _infer_schema(sample_df)
        assert set(schema.keys()) == set(sample_df.columns)

    def test_numeric_dtype_detected(self, sample_df):
        schema = _infer_schema(sample_df)
        assert "float" in schema["V1"]["dtype"] or "int" in schema["V1"]["dtype"]

    def test_object_dtype_detected(self, sample_df):
        schema = _infer_schema(sample_df)
        assert schema["Category"]["dtype"] == "object"

    def test_null_count_zero_for_clean_data(self, sample_df):
        schema = _infer_schema(sample_df)
        assert schema["V1"]["null_count"] == 0
        assert schema["V1"]["null_pct"] == 0.0

    def test_null_count_detected(self, sample_df):
        sample_df.loc[:50, "V1"] = None
        schema = _infer_schema(sample_df)
        assert schema["V1"]["null_count"] == 51

    def test_unique_count(self, sample_df):
        schema = _infer_schema(sample_df)
        assert schema["Category"]["unique_count"] == 3

    def test_empty_df_returns_empty_schema(self):
        schema = _infer_schema(pd.DataFrame())
        assert schema == {}


class TestComputeBaselineStats:
    def test_numeric_cols_have_mean_std(self, sample_df):
        stats = _compute_baseline_stats(sample_df)
        assert "mean" in stats["V1"]
        assert "std" in stats["V1"]
        assert "p25" in stats["V1"]
        assert "p75" in stats["V1"]

    def test_mean_is_close_to_zero(self, sample_df):
        stats = _compute_baseline_stats(sample_df)
        # Random normal data should have mean ≈ 0
        assert abs(stats["V1"]["mean"]) < 0.5

    def test_categorical_col_has_value_counts(self, sample_df):
        stats = _compute_baseline_stats(sample_df)
        assert "value_counts" in stats["Category"]
        assert "unique_count" in stats["Category"]
        assert stats["Category"]["unique_count"] == 3

    def test_values_sample_length(self, sample_df):
        stats = _compute_baseline_stats(sample_df)
        # Sample should be at most 200 values
        assert len(stats["V1"]["values_sample"]) <= 200

    def test_amount_min_is_non_negative(self, sample_df):
        stats = _compute_baseline_stats(sample_df)
        assert stats["Amount"]["min"] >= 0.0


class TestValidateCsv:
    def test_valid_df_no_issues(self, sample_df):
        issues = _validate_csv(sample_df)
        assert len(issues) == 0

    def test_empty_df_flagged(self):
        issues = _validate_csv(pd.DataFrame())
        assert any("empty" in i.lower() for i in issues)

    def test_single_column_flagged(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        issues = _validate_csv(df)
        assert any("column" in i.lower() for i in issues)

    def test_high_null_col_flagged(self, sample_df):
        # Set 95% of V2 to null
        sample_df.loc[: int(len(sample_df) * 0.95), "V2"] = None
        issues = _validate_csv(sample_df)
        assert any("V2" in i for i in issues)

    def test_all_nulls_flagged(self, sample_df):
        sample_df["V1"] = None
        issues = _validate_csv(sample_df)
        assert any("V1" in i for i in issues)
