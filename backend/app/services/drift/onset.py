"""
Onset detection (FR9: timestamp of first threshold crossing).

The check window is split into equal-count time buckets; each bucket is
scored against the reference with the same stage monitor. The onset is the
start of the latest drift episode: walking back from the most recent drifted
bucket, the episode continues across at most one clean bucket, so a single
noisy bucket early in the window doesn't set the onset.
"""

from datetime import datetime
from typing import Any, Callable

import numpy as np

from app.services.drift.stage_monitors import StageResult

MIN_ROWS_PER_BUCKET = 10
MAX_GAP = 1  # clean buckets tolerated inside one drift episode


def bucket_indices(n_rows: int, n_buckets: int) -> list[np.ndarray]:
    n_buckets = min(n_buckets, n_rows // MIN_ROWS_PER_BUCKET)
    if n_buckets < 2:
        return []
    return [b for b in np.array_split(np.arange(n_rows), n_buckets) if len(b)]


def find_onset(
    timestamps: list[datetime],
    buckets: list[np.ndarray],
    evaluate: Callable[[np.ndarray], StageResult],
) -> tuple[datetime | None, list[dict[str, Any]]]:
    """Returns (onset timestamp or None, per-bucket timeline)."""
    timeline = []
    for idx in buckets:
        res = evaluate(idx)
        timeline.append(
            {
                "start": timestamps[idx[0]],
                "end": timestamps[idx[-1]],
                "score": res.score,
                "drifted": res.drifted,
            }
        )

    drifted = [i for i, b in enumerate(timeline) if b["drifted"]]
    if not drifted:
        return None, timeline
    # Walk back from the most recent drifted bucket; the episode continues across at most
    # MAX_GAP clean buckets (a noisy bucket mid-episode), but an isolated early blip doesn't join it.
    start = drifted[-1]
    for i in reversed(drifted[:-1]):
        if start - i > MAX_GAP + 1:
            break
        start = i
    return timeline[start]["start"], timeline
