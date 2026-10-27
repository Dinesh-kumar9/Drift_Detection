"""Small shared helpers."""

import math
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """
    Recursively convert a value into something Postgres JSONB will accept.

    JSONB rejects NaN/Infinity, which pandas/sklearn produce routinely
    (std of a single value, r2 on a constant target, XGBoost's `missing=nan`
    param) — those become None. numpy scalars/arrays become Python types and
    anything else non-primitive is stringified.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    return str(value)
