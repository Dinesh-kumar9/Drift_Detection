"""
Unit tests for the stage-wise drift engine: detectors, onset, attribution, and
the full analysis on synthetic traffic with a fault injected at each stage.
No DB or MinIO required.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.services.drift.attribution_engine import INHERITED, attribute
from app.services.drift.baselines import Baseline, build_baseline, prediction_error
from app.services.drift.detectors import (
    categorical_test,
    mark_drift,
    null_rate_test,
    numeric_test,
    proportion_increase_test,
)
from app.services.drift.fault_injection import build_traffic
from app.services.drift.onset import bucket_indices, find_onset
from app.services.drift.service import analyse_window
from app.services.drift.stage_monitors import StageResult
from app.services.model_bundle import ModelBundle
from app.services.preprocessing_service import build_preprocessing_pipeline
from app.services.training_service import split_indices, train_all_models

ALPHA, MIN_EFFECT = 0.05, 0.1
NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
rng = np.random.default_rng(0)


# ─── Detectors ───────────────────────────────────────────────────────────────


class TestDetectors:
    def test_numeric_same_distribution_not_drifted(self):
        [r] = mark_drift([numeric_test("x", rng.normal(size=1000), rng.normal(size=500))], ALPHA, MIN_EFFECT)
        assert not r.drifted
        assert r.statistic < MIN_EFFECT

    def test_numeric_shift_drifted(self):
        [r] = mark_drift([numeric_test("x", rng.normal(size=1000), rng.normal(1, 1, size=500))], ALPHA, MIN_EFFECT)
        assert r.drifted
        assert r.statistic > 0.3

    def test_tiny_but_significant_shift_is_not_drift(self):
        """Huge samples make negligible shifts significant — the effect-size floor stops that."""
        ref, cur = rng.normal(size=200_000), rng.normal(0.03, 1, size=200_000)
        [r] = mark_drift([numeric_test("x", ref, cur)], ALPHA, MIN_EFFECT)
        assert r.p_value < 0.001
        assert not r.drifted

    def test_categorical_shift_reports_tvd_and_unseen(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 100 + ["b"] * 100 + ["z"] * 300
        r = categorical_test("c", ref, cur)
        assert r.statistic == pytest.approx(0.6)
        assert r.detail["unseen_categories"] == ["z"]

    def test_null_rate(self):
        r = null_rate_test("x", [1.0] * 200, [None] * 50 + [1.0] * 50)
        assert r.statistic == pytest.approx(0.5)
        assert r.p_value < 0.001
        assert null_rate_test("x", [1.0] * 50, [2.0] * 50) is None

    def test_error_rate_is_one_sided(self):
        worse = proportion_increase_test("err", [0] * 90 + [1] * 10, [0] * 50 + [1] * 50)
        better = proportion_increase_test("err", [0] * 50 + [1] * 50, [0] * 90 + [1] * 10)
        assert worse.statistic == pytest.approx(0.4) and worse.p_value < 0.001
        assert better.statistic == 0 and better.p_value > 0.5

    def test_too_few_samples_skipped(self):
        assert numeric_test("x", [1, 2, 3], [1, 2, 3, 4, 5, 6]) is None

    def test_prediction_error(self):
        assert prediction_error("classification", "fraud", "fraud") == 0.0
        assert prediction_error("classification", 1, "1") == 0.0
        assert prediction_error("regression", 3.0, 5.5) == 2.5
        assert prediction_error("regression", 3.0, None) is None


# ─── Onset ───────────────────────────────────────────────────────────────────


def _result(drifted: bool, score: float = 0.5, **signals) -> StageResult:
    return StageResult("S1_ingestion", score if drifted else 0.02, drifted, 0.0, 100, 100, [], signals)


class TestOnset:
    def test_onset_is_start_of_persistent_drift(self):
        ts = [NOW + timedelta(minutes=i) for i in range(60)]
        buckets = bucket_indices(60, 6)
        onset, timeline = find_onset(ts, buckets, lambda idx: _result(idx[0] >= 30))
        assert onset == ts[30]
        assert [b["drifted"] for b in timeline] == [False, False, False, True, True, True]

    def test_isolated_early_blip_ignored(self):
        ts = [NOW + timedelta(minutes=i) for i in range(60)]
        flags = {0: True, 40: True, 50: True}
        onset, _ = find_onset(ts, bucket_indices(60, 6), lambda idx: _result(flags.get(int(idx[0]), False)))
        assert onset == ts[40]

    def test_too_few_rows_for_buckets(self):
        assert bucket_indices(15, 6) == []


# ─── Attribution rules ───────────────────────────────────────────────────────


class TestAttribution:
    def _stage(self, stage, drifted, score=0.6, **signals):
        r = _result(drifted, score, **signals)
        r.stage = stage
        if drifted:
            r.tests = [SimpleNamespace(feature="f", statistic=score, drifted=True)]
        return r

    def test_no_drift(self):
        res = {s: self._stage(s, False) for s in ("S1_ingestion", "S2_preprocessing", "S3_output")}
        out = attribute(res, {}, 60)
        assert not out.drift_detected and out.root_cause is None

    def test_upstream_shift_wins_over_inherited_downstream(self):
        """Same-moment, same-magnitude drift at S1+S2 (KS is scale-invariant) → S1, S2 discounted."""
        res = {
            "S1_ingestion": self._stage("S1_ingestion", True),
            "S2_preprocessing": self._stage("S2_preprocessing", True),
            "S3_output": self._stage("S3_output", True, 0.3),
        }
        out = attribute(res, {s: NOW for s in res}, 60)
        assert out.root_cause == "S1_ingestion"
        s2 = next(r for r in out.ranked if r["stage"] == "S2_preprocessing")
        assert s2["causal_factor"] == INHERITED

    def test_pipeline_divergence_points_at_s2(self):
        res = {
            "S1_ingestion": self._stage("S1_ingestion", False),
            "S2_preprocessing": self._stage("S2_preprocessing", True, diverged=True),
            "S3_output": self._stage("S3_output", True, 0.4),
        }
        out = attribute(res, {"S2_preprocessing": NOW, "S3_output": NOW}, 60)
        assert out.root_cause == "S2_preprocessing"
        assert "training-time transform" in out.explanation

    def test_error_drift_on_stable_inputs_is_concept_drift(self):
        res = {
            "S1_ingestion": self._stage("S1_ingestion", False),
            "S2_preprocessing": self._stage("S2_preprocessing", False),
            "S3_output": self._stage("S3_output", True, 0.4, error_drift=True),
        }
        out = attribute(res, {"S3_output": NOW}, 60)
        assert out.root_cause == "S3_output"
        assert out.confidence == 1.0

    def test_earlier_onset_outranks_equal_magnitude(self):
        res = {
            "S1_ingestion": self._stage("S1_ingestion", True, 0.5),
            "S2_preprocessing": self._stage("S2_preprocessing", True, 0.5, diverged=True),
        }
        out = attribute(res, {"S1_ingestion": NOW + timedelta(minutes=10), "S2_preprocessing": NOW}, 60)
        assert out.root_cause == "S2_preprocessing"


# ─── Full analysis on injected faults ────────────────────────────────────────


@pytest.fixture(scope="module")
def trained():
    """A real model + baseline on fraud-style data."""
    r = np.random.default_rng(5)
    n = 1500
    amount = r.exponential(100, n)
    channel = r.choice(["web", "pos", "atm"], n)
    df = pd.DataFrame(
        {
            "amount": amount,
            "age": r.integers(18, 80, n).astype(float),
            "channel": channel,
            "label": np.where((amount > 120) | (channel == "atm"), "fraud", "legit"),
        }
    )
    X, y, config, transformer = build_preprocessing_pipeline(df, "label")
    [res] = train_all_models(X, y, "classification", model_names=["RandomForestClassifier"])
    bundle = ModelBundle(
        model_type="RandomForestClassifier",
        task_type="classification",
        target_column="label",
        numeric_cols=config["numeric_cols"],
        cat_cols=config["cat_cols"],
        feature_names=config["feature_names"],
        preprocessor=transformer["ct"],
        model=res["model"],
        label_encoder=transformer["le"],
    )
    _, test_idx = split_indices(y, "classification", 0.2, 42)
    baseline = Baseline.from_bytes(build_baseline(bundle, df, test_idx, max_rows=1000).to_bytes())
    return bundle, baseline


def _analyse(bundle, baseline, fault, **kw):
    rows, info = build_traffic(baseline, n=600, fault=fault, now=NOW, seed=1, **kw)
    outputs = bundle.predict([r.record for r in rows], [r.feature_offsets for r in rows])
    preds = [
        SimpleNamespace(
            predicted_at=r.timestamp,
            input_snapshot=r.record,
            features_snapshot=o["features"],
            output={"prediction": o["prediction"], "confidence": o["confidence"]},
            actual=r.actual,
        )
        for r, o in zip(rows, outputs)
    ]
    results, onsets, _, attribution = analyse_window(baseline, bundle, preds, 6, ALPHA, MIN_EFFECT)
    return results, onsets, attribution, info


class TestInjectedFaults:
    def test_baseline_round_trip(self, trained):
        _, baseline = trained
        assert len(baseline.s1) == 1000
        assert list(baseline.s2.columns) == baseline.feature_names
        assert baseline.s3_confidence and len(baseline.s3_errors) == len(baseline.s3_predictions)

    def test_healthy_traffic_raises_no_drift(self, trained):
        results, _, attribution, _ = _analyse(*trained, "none")
        assert not attribution.drift_detected, {s: r.tests[:2] for s, r in results.items()}

    def test_raw_input_shift_attributed_to_s1(self, trained):
        results, onsets, attribution, info = _analyse(*trained, "s1_shift", feature="age")
        assert attribution.root_cause == "S1_ingestion"
        assert results["S1_ingestion"].top_feature == "age"
        # onset lands near where the fault started (within one bucket = 5 minutes)
        fault_start = datetime.fromisoformat(info["fault_starts_at"])
        assert abs((onsets["S1_ingestion"] - fault_start).total_seconds()) <= 5 * 60

    def test_null_flood_attributed_to_s1(self, trained):
        results, _, attribution, _ = _analyse(*trained, "s1_nulls", feature="channel")
        assert attribution.root_cause == "S1_ingestion"
        assert "null rate" in results["S1_ingestion"].top_feature

    def test_preprocessing_fault_attributed_to_s2(self, trained):
        results, _, attribution, _ = _analyse(*trained, "s2_shift", feature="amount")
        assert not results["S1_ingestion"].drifted
        assert results["S2_preprocessing"].signals["diverged"]
        assert attribution.root_cause == "S2_preprocessing"

    def test_concept_drift_attributed_to_s3(self, trained):
        results, _, attribution, _ = _analyse(*trained, "s3_concept")
        assert not results["S1_ingestion"].drifted
        assert not results["S2_preprocessing"].drifted
        assert results["S3_output"].signals["error_drift"]
        assert attribution.root_cause == "S3_output"
