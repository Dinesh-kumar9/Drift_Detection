"""
Per-feature drift tests.

Every test returns an effect size in [0, 1] (`statistic`) plus a p-value, so
numeric and categorical features are comparable within a stage:

  numeric      two-sample Kolmogorov–Smirnov — statistic = KS D
  categorical  chi-square on category counts — statistic = total-variation distance
  null rate    chi-square on null / non-null — statistic = |Δ null rate|
  error rate   one-sided two-proportion z-test — statistic = increase in error rate

A test only counts as drift when it is significant AND the effect is large
enough to matter: with thousands of rows, tiny meaningless shifts are
"significant", which is how monitoring turns into alert fatigue.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

MIN_SAMPLES = 5


@dataclass
class FeatureDrift:
    feature: str
    test: str
    statistic: float  # effect size in [0, 1]
    p_value: float
    n_reference: int
    n_current: int
    drifted: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["statistic"] = round(self.statistic, 4)
        out["p_value"] = float(f"{self.p_value:.4g}")
        return out


def _clean_numeric(values: Iterable) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def numeric_test(feature: str, reference: Iterable, current: Iterable) -> FeatureDrift | None:
    ref, cur = _clean_numeric(reference), _clean_numeric(current)
    if len(ref) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
        return None
    result = stats.ks_2samp(ref, cur)
    return FeatureDrift(
        feature=feature,
        test="ks",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_reference=len(ref),
        n_current=len(cur),
        detail={
            "reference_mean": round(float(np.mean(ref)), 4),
            "current_mean": round(float(np.mean(cur)), 4),
        },
    )


def categorical_test(feature: str, reference: Iterable, current: Iterable) -> FeatureDrift | None:
    ref = pd.Series(list(reference), dtype=object).dropna().astype(str)
    cur = pd.Series(list(current), dtype=object).dropna().astype(str)
    if len(ref) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
        return None

    ref_counts, cur_counts = ref.value_counts(), cur.value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_vec = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)
    cur_vec = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)
    ref_p, cur_p = ref_vec / ref_vec.sum(), cur_vec / cur_vec.sum()
    tvd = float(0.5 * np.abs(ref_p - cur_p).sum())

    if len(categories) < 2:
        p_value = 1.0
    else:
        p_value = float(stats.chi2_contingency(np.vstack([ref_vec, cur_vec]))[1])

    shifts = sorted(zip(categories, cur_p - ref_p), key=lambda kv: -abs(kv[1]))[:5]
    return FeatureDrift(
        feature=feature,
        test="chi2",
        statistic=tvd,
        p_value=p_value,
        n_reference=len(ref),
        n_current=len(cur),
        detail={
            "largest_shifts": {c: round(float(d), 4) for c, d in shifts},
            "unseen_categories": [c for c in categories if ref_counts.get(c, 0) == 0][:10],
        },
    )


def null_rate_test(feature: str, reference: Iterable, current: Iterable) -> FeatureDrift | None:
    ref_null = pd.Series(list(reference), dtype=object).isna()
    cur_null = pd.Series(list(current), dtype=object).isna()
    if len(ref_null) < MIN_SAMPLES or len(cur_null) < MIN_SAMPLES:
        return None
    ref_rate, cur_rate = float(ref_null.mean()), float(cur_null.mean())
    if ref_rate == cur_rate:
        return None
    table = np.array(
        [[ref_null.sum(), (~ref_null).sum()], [cur_null.sum(), (~cur_null).sum()]],
        dtype=float,
    )
    p_value = float(stats.chi2_contingency(table)[1]) if (table.sum(axis=0) > 0).all() else 1.0
    return FeatureDrift(
        feature=f"{feature} (null rate)",
        test="null_rate",
        statistic=abs(cur_rate - ref_rate),
        p_value=p_value,
        n_reference=len(ref_null),
        n_current=len(cur_null),
        detail={"reference_null_rate": round(ref_rate, 4), "current_null_rate": round(cur_rate, 4)},
    )


def proportion_increase_test(name: str, reference: Iterable, current: Iterable) -> FeatureDrift | None:
    """One-sided: is the current rate (e.g. error rate) higher than the reference rate?"""
    ref, cur = np.asarray(list(reference), dtype=float), np.asarray(list(current), dtype=float)
    if len(ref) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
        return None
    p1, p2 = ref.mean(), cur.mean()
    pooled = (ref.sum() + cur.sum()) / (len(ref) + len(cur))
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(ref) + 1 / len(cur)))
    p_value = float(stats.norm.sf((p2 - p1) / se)) if se > 0 else (0.0 if p2 > p1 else 1.0)
    return FeatureDrift(
        feature=name,
        test="proportion_increase",
        statistic=float(max(0.0, p2 - p1)),
        p_value=p_value,
        n_reference=len(ref),
        n_current=len(cur),
        detail={"reference_rate": round(float(p1), 4), "current_rate": round(float(p2), 4)},
    )


def mark_drift(results: list[FeatureDrift], alpha: float, min_effect: float) -> list[FeatureDrift]:
    """Bonferroni-correct across the stage's tests, then require a minimum effect size."""
    if not results:
        return results
    alpha_adj = alpha / len(results)
    for r in results:
        r.drifted = r.p_value < alpha_adj and r.statistic >= min_effect
    return sorted(results, key=lambda r: (-r.drifted, -r.statistic))
