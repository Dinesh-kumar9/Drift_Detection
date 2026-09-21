"""
Drift Monitoring Service — Phase 2

S1: Ingestion-stage drift detection.

Compares baseline feature distributions with recent production
input data using the Kolmogorov-Smirnov (KS) test.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _normalize_drift_score(ks_statistic: float) -> float:
    """
    Convert KS statistic (0–1) into a normalized drift score (0–1).
    """
    return round(float(np.clip(ks_statistic, 0.0, 1.0)), 4)


def _severity_from_score(score: float) -> str:
    """
    Convert normalized drift score into a human-readable severity.
    """
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def detect_numeric_drift(
    baseline_values: list[float],
    current_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Detect drift between baseline and current numeric distributions.

    Uses the two-sample Kolmogorov-Smirnov test.

    Returns:
        drift_detected
        ks_statistic
        p_value
        drift_score
        severity
        threshold
    """

    if len(baseline_values) < 2:
        raise ValueError("Baseline data must contain at least 2 values.")

    if len(current_values) < 2:
        raise ValueError("Current data must contain at least 2 values.")

    baseline = np.asarray(baseline_values, dtype=float)
    current = np.asarray(current_values, dtype=float)

    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]

    if len(baseline) < 2 or len(current) < 2:
        raise ValueError("Not enough valid numeric values for drift detection.")

    statistic, p_value = ks_2samp(baseline, current)

    drift_score = _normalize_drift_score(statistic)

    return {
        "drift_detected": bool(p_value < alpha),
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "drift_score": drift_score,
        "severity": _severity_from_score(drift_score),
        "threshold": alpha,
    }


def detect_s1_drift(
    baseline_stats: dict[str, Any],
    current_df: pd.DataFrame,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Run S1 ingestion-stage drift detection for all numeric features.

    baseline_stats comes from Dataset.baseline_stats.
    current_df contains recent production input data.
    """

    results: list[dict[str, Any]] = []

    for feature_name, baseline in baseline_stats.items():

        # S1 numeric drift
        if "values_sample" in baseline and feature_name in current_df.columns:
            current_values = current_df[feature_name].dropna().tolist()

            if len(current_values) < 2:
                continue

            result = detect_numeric_drift(
                baseline_values=baseline["values_sample"],
                current_values=current_values,
                alpha=alpha,
            )

            results.append(
                {
                    "stage": "S1_ingestion",
                    "feature_name": feature_name,
                    **result,
                }
            )

    return results
