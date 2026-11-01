"""
Load test for the serving endpoint: validates the "Inference API p95 < 300 ms" NFR.

Signs in once, spreads predictions across every production model, and varies the
inputs so each request is a realistic one (each is logged with its S1/S2/S3
stages, exactly like real traffic). The run fails (exit code 1) if p95 is over
the target or more than 1% of requests fail.

    docker compose --profile perf up -d api_perf        # production-mode API on :8001
    pip install -r tests/load/requirements.txt
    locust -f tests/load/locustfile.py --host http://127.0.0.1:8001 \
        --headless --users 20 --spawn-rate 5 --run-time 2m

Notes (results and method: docs/LOAD_TEST.md):
- Target the production-mode API (several workers). The dev server on :8000 is
  one hot-reload process and saturates at ~35 req/s.
- Use 127.0.0.1, not localhost: on Windows, resolving localhost added ~50 ms
  per request on the client side.
- Load-test traffic is logged like real traffic, so drift checks will flag it
  (the inputs are near-identical). Run against a scratch database, or delete the
  run's rows from `predictions` afterwards.
- Every simulated user shares one account, so the per-user predict rate limit
  (RATE_LIMIT_PREDICT_PER_MINUTE, default 6000/min) caps the whole run.

Environment: LOAD_EMAIL / LOAD_PASSWORD (default: the demo engineer),
LOAD_P95_MS (default 300).
"""

import os
import random

import requests
from locust import HttpUser, between, events, task

P95_TARGET_MS = float(os.environ.get("LOAD_P95_MS", "300"))
MAX_FAILURE_RATIO = 0.01

_targets: list[tuple[str, dict]] = []  # (model_version_id, sample features)
_auth: dict[str, str] = {}


@events.test_start.add_listener
def prepare(environment, **_):
    """Log in once and collect a sample input for every production model."""
    host = environment.host.rstrip("/")
    res = requests.post(
        f"{host}/auth/login",
        json={
            "email": os.environ.get("LOAD_EMAIL", "engineer@example.com"),
            "password": os.environ.get("LOAD_PASSWORD", "engineer1234"),
        },
        timeout=10,
    )
    res.raise_for_status()
    _auth["Authorization"] = f"Bearer {res.json()['access_token']}"

    lineages = requests.get(f"{host}/models/registered", headers=_auth, timeout=10).json()["items"]
    for lineage in lineages:
        version_id = lineage.get("production_version_id")
        if not version_id:
            continue
        sample = requests.get(f"{host}/models/{version_id}/sample-input", headers=_auth, timeout=30)
        if sample.ok:
            _targets.append((version_id, sample.json()["features"]))
    if not _targets:
        raise RuntimeError("No production models to load test — deploy one first (see docs/DEMO.md)")
    print(f"Load testing {len(_targets)} production model(s)")


def _vary(features: dict) -> dict:
    """Jitter numeric inputs by ±20% so requests aren't identical."""
    return {k: v * random.uniform(0.8, 1.2) if isinstance(v, (int, float)) else v for k, v in features.items()}


class PredictionClient(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        version_id, features = random.choice(_targets)
        with self.client.post(
            f"/models/{version_id}/predict",
            json={"features": _vary(features)},
            headers=_auth,
            name="/models/[id]/predict",
            catch_response=True,
        ) as res:
            if res.status_code != 200:
                res.failure(f"HTTP {res.status_code}: {res.text[:120]}")


@events.quitting.add_listener
def check_nfr(environment, **_):
    stats = environment.stats.total
    if not stats.num_requests:
        return
    p95 = stats.get_response_time_percentile(0.95)
    verdict = "PASS" if p95 <= P95_TARGET_MS and stats.fail_ratio <= MAX_FAILURE_RATIO else "FAIL"
    print(
        f"\n{verdict}: {stats.num_requests} requests · {stats.total_rps:.1f} req/s · "
        f"p50 {stats.median_response_time:.0f} ms · p95 {p95:.0f} ms · p99 "
        f"{stats.get_response_time_percentile(0.99):.0f} ms · failures {stats.fail_ratio:.2%} "
        f"(target p95 ≤ {P95_TARGET_MS:.0f} ms, failures ≤ {MAX_FAILURE_RATIO:.0%})"
    )
    if verdict == "FAIL":
        environment.process_exit_code = 1
