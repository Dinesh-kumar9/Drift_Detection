import pandas as pd

from app.services.drift_service import (
    detect_numeric_drift,
    detect_s1_drift,
)


def test_numeric_drift_detected():
    baseline = [10, 11, 12, 10, 11, 12, 10, 11, 12, 10]
    current = [20, 21, 22, 20, 21, 22, 20, 21, 22, 20]

    result = detect_numeric_drift(baseline, current)

    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0
    assert result["p_value"] < 0.05


def test_numeric_no_drift():
    baseline = [10, 11, 12, 10, 11, 12, 10, 11, 12, 10]
    current = [10, 11, 12, 10, 11, 12, 10, 11, 12, 10]

    result = detect_numeric_drift(baseline, current)

    assert result["drift_detected"] is False
    assert result["ks_statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_s1_drift_detection():
    baseline = {"temperature": {"values_sample": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10]}}

    current = pd.DataFrame({"temperature": [20, 21, 22, 20, 21, 22, 20, 21, 22, 20]})

    results = detect_s1_drift(baseline, current)

    assert len(results) == 1
    assert results[0]["stage"] == "S1_ingestion"
    assert results[0]["feature_name"] == "temperature"
    assert results[0]["drift_detected"] is True
