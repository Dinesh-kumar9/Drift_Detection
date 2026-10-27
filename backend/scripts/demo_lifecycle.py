"""
Drive the full model lifecycle against a running stack (docker compose up):

    login → upload CSV → train 3 models (Celery) → compare → deploy best → predict → feedback

Usage (from backend/):
    python scripts/demo_lifecycle.py [--api http://localhost:8000] [--rows 2000]

Generates a synthetic fraud-style dataset (numeric + categorical features, string
labels) so it runs anywhere with no downloads. Exits non-zero on any failure, so
it doubles as a local end-to-end smoke test.
"""

import argparse
import csv
import io
import math
import random
import sys
import time

import httpx

ENGINEER = ("engineer@example.com", "engineer1234")


def make_dataset(rows: int, seed: int = 7) -> bytes:
    rng = random.Random(seed)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["amount", "hour", "account_age_days", "channel", "country", "label"])
    for _ in range(rows):
        amount = round(rng.expovariate(1 / 120), 2)
        hour = rng.randint(0, 23)
        age = rng.randint(1, 3650)
        channel = rng.choice(["web", "pos", "atm", "mobile"])
        country = rng.choice(["US", "IN", "GB", "DE", "BR"])
        risk = 0.004 * amount + (1.2 if hour < 5 else 0) + (0.8 if channel == "web" else 0) - age / 1500
        label = "fraud" if rng.random() < 1 / (1 + math.exp(-(risk - 2))) else "legit"
        writer.writerow([amount, hour, "" if rng.random() < 0.02 else age, channel, country, label])
    return buf.getvalue().encode()


def step(msg: str) -> None:
    print(f"\n▶ {msg}")


def check(res: httpx.Response, expected: int = 200) -> dict:
    if res.status_code != expected:
        print(f"  ✗ {res.request.method} {res.request.url.path} → {res.status_code}: {res.text[:500]}")
        sys.exit(1)
    return res.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for training")
    args = parser.parse_args()

    with httpx.Client(base_url=args.api, timeout=60) as api:
        step("Health")
        print("  ", check(api.get("/health")))

        step(f"Login as {ENGINEER[0]}")
        token = check(api.post("/auth/login", json={"email": ENGINEER[0], "password": ENGINEER[1]}))["access_token"]
        api.headers["Authorization"] = f"Bearer {token}"

        step(f"Upload synthetic dataset ({args.rows} rows)")
        ds = check(
            api.post(
                "/datasets/",
                files={"file": ("demo_fraud.csv", make_dataset(args.rows), "text/csv")},
                data={"name": "demo-fraud"},
            ),
            201,
        )
        print(f"   dataset {ds['id']} · {ds['row_count']} rows · columns {list(ds['schema_json'])}")

        step("Train RandomForest · XGBoost · LogisticRegression")
        train = check(api.post("/experiments/train", json={"dataset_id": ds["id"], "target_column": "label"}), 202)
        group = train["experiment_group_id"]
        print(f"   group {group} → registered model '{train['registered_model_name']}' ({train['task_type']})")

        deadline = time.time() + args.timeout
        while True:
            status = check(api.get(f"/experiments/{group}/status"))
            states = ", ".join(f"{m['model_type']}={m['status']}" for m in status["models"])
            print(f"   [{status['overall_status']}] {states}")
            if status["overall_status"] in ("completed", "failed"):
                break
            if time.time() > deadline:
                print("  ✗ timed out waiting for training — is the worker running? (docker compose logs worker)")
                sys.exit(1)
            time.sleep(3)
        if status["overall_status"] != "completed":
            print(f"  ✗ training failed: {status['error']}")
            sys.exit(1)

        step("Compare")
        cmp = check(api.get(f"/experiments/{group}/compare"))
        for m in sorted(cmp["models"], key=lambda m: -(m.get(cmp["best_metric"]) or -1)):
            marker = "★" if m["run_id"] == cmp["best_model_run_id"] else " "
            print(
                f"   {marker} {m['model_type']:<24} {m['version_tag']}  f1={m['f1']}  acc={m['accuracy']}"
                f"  auc={m['roc_auc']}  {m['latency_ms']}ms/row  {m['model_size_kb']}KB"
            )
        best = next(m for m in cmp["models"] if m["run_id"] == cmp["best_model_run_id"])

        step(f"Deploy {best['model_type']} {best['version_tag']} to production")
        mv = check(api.post(f"/models/{best['model_version_id']}/deploy", json={"sla_tier": "critical"}))
        print(f"   {mv['registered_model_name']} {mv['version_tag']} → {mv['status']} (sla={mv['sla_tier']})")

        step("Predict")
        sample = check(api.get(f"/models/{mv['id']}/sample-input"))["features"]
        risky = {**sample, "amount": 2500.0, "hour": 3, "channel": "web", "account_age_days": 5}
        latencies = []
        for name, features in (("typical", sample), ("risky", risky)):
            pred = check(api.post(f"/models/{mv['id']}/predict", json={"features": features}))
            latencies.append(pred["latency_ms"])
            print(f"   {name:<8} → {pred['prediction']} (confidence {pred['confidence']}, {pred['latency_ms']} ms)")

        bad = api.post(f"/models/{mv['id']}/predict", json={"features": {"amount": 1}})
        print(f"   missing columns → {bad.status_code} {bad.json()['detail']['missing_columns']}")

        step("Feedback (ground truth)")
        fb = check(api.post(f"/predictions/{pred['prediction_id']}/feedback", json={"actual": "fraud"}))
        print(f"   prediction {fb['id']} labelled '{fb['actual']}'")

        log = check(api.get("/predictions/", params={"model_version_id": mv["id"], "limit": 1}))
        entry = log["items"][0]
        print(
            f"   logged stages: S1 keys={sorted(entry['input_snapshot'])[:3]}…  "
            f"S2 features={len(entry['features_snapshot'])}  S3={entry['output']['prediction']}"
        )

        print("\n✓ Full lifecycle OK")


if __name__ == "__main__":
    main()
