"""
Evaluate the drift engine: how reliably does it detect an injected fault and
name the right stage, as the fault gets subtler? Also: false-alarm rate on
healthy traffic and how close the detected onset is to the true fault start.

Runs fully in-process (no database / stack needed):

    python scripts/evaluate_attribution.py [--trials 20] [--rows 600] [--out results.md]

Prints a markdown table — suitable for a project report.
"""

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.drift.baselines import build_baseline  # noqa: E402
from app.services.drift.fault_injection import build_traffic  # noqa: E402
from app.services.drift.service import analyse_window  # noqa: E402
from app.services.model_bundle import ModelBundle  # noqa: E402
from app.services.preprocessing_service import build_preprocessing_pipeline  # noqa: E402
from app.services.training_service import split_indices, train_all_models  # noqa: E402

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
ALPHA, MIN_EFFECT, BUCKETS = 0.05, 0.1, 6

# fault → (feature, severity parameter, grid). Shifts are in σ of the column; the others
# vary the share of traffic affected.
SCENARIOS = {
    "s1_shift": ("age", "magnitude", [0.25, 0.5, 1.0, 2.0, 3.0]),
    "s1_nulls": ("channel", "fault_fraction", [0.05, 0.1, 0.2, 0.35, 0.5]),
    "s2_shift": ("amount", "magnitude", [0.25, 0.5, 1.0, 2.0, 3.0]),
    "s3_concept": (None, "fault_fraction", [0.05, 0.1, 0.2, 0.35, 0.5]),
}
EXPECTED = {
    "s1_shift": "S1_ingestion",
    "s1_nulls": "S1_ingestion",
    "s2_shift": "S2_preprocessing",
    "s3_concept": "S3_output",
}


def train_reference(seed: int = 5):
    rng = np.random.default_rng(seed)
    n = 3000
    amount = rng.exponential(100, n)
    channel = rng.choice(["web", "pos", "atm"], n)
    df = pd.DataFrame(
        {
            "amount": amount,
            "age": rng.integers(18, 80, n).astype(float),
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
    bundle._optimise_for_serving()
    return bundle, build_baseline(bundle, df, test_idx, max_rows=2000)


def run_trial(bundle, baseline, fault, rows, seed, **kw):
    traffic, info = build_traffic(baseline, n=rows, fault=fault, now=NOW, seed=seed, **kw)
    outputs = bundle.predict([t.record for t in traffic], [t.feature_offsets for t in traffic])
    preds = [
        SimpleNamespace(
            predicted_at=t.timestamp,
            input_snapshot=t.record,
            features_snapshot=o["features"],
            output={"prediction": o["prediction"], "confidence": o["confidence"]},
            actual=t.actual,
        )
        for t, o in zip(traffic, outputs)
    ]
    _, onsets, _, attribution = analyse_window(baseline, bundle, preds, BUCKETS, ALPHA, MIN_EFFECT)
    onset_error = None
    if attribution.root_cause and info.get("fault_starts_at"):
        start = datetime.fromisoformat(info["fault_starts_at"])
        onset_error = abs((onsets[attribution.root_cause] - start).total_seconds()) / 60
    return attribution, onset_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--rows", type=int, default=600, help="predictions per check window")
    parser.add_argument("--out", type=Path, help="also write the markdown table to this file")
    args = parser.parse_args()

    t0 = time.time()
    print("Training reference model…", file=sys.stderr)
    bundle, baseline = train_reference()

    lines = [
        f"Window: {args.rows} predictions over 30 min, fault in the last half; "
        f"{args.trials} trials per row; α={ALPHA} (Bonferroni), min effect={MIN_EFFECT}.",
        "",
        "| Fault | Severity | Detected | Correct root cause | Median onset error (min) |",
        "|---|---|---|---|---|",
    ]

    false_alarms = 0
    for seed in range(args.trials):
        attribution, _ = run_trial(bundle, baseline, "none", args.rows, seed)
        false_alarms += attribution.drift_detected
    lines.append(f"| none (healthy) | — | {false_alarms / args.trials:.0%} false alarms | — | — |")

    for fault, (feature, param, grid) in SCENARIOS.items():
        for level in grid:
            detected = correct = 0
            onset_errors = []
            for seed in range(args.trials):
                kw = {param: level}
                if feature:
                    kw["feature"] = feature
                attribution, onset_err = run_trial(bundle, baseline, fault, args.rows, 1000 + seed, **kw)
                detected += attribution.drift_detected
                if attribution.root_cause == EXPECTED[fault]:
                    correct += 1
                    if onset_err is not None:
                        onset_errors.append(onset_err)
            unit = "σ" if param == "magnitude" else " of traffic"
            severity = f"{level:g}{unit}" if param == "magnitude" else f"{level:.0%}{unit}"
            onset = f"{statistics.median(onset_errors):.1f}" if onset_errors else "—"
            lines.append(
                f"| {fault} ({feature or 'labels'}) | {severity} | {detected / args.trials:.0%} | "
                f"{correct / args.trials:.0%} | {onset} |"
            )
            print(lines[-1], file=sys.stderr)

    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    print(f"\n({time.time() - t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
