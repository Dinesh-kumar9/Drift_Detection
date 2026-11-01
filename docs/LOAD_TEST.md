# Load Test: Inference API p95 < 300 ms

The NFR from the build document: **Inference API p95 < 300 ms**. The test is
[`backend/tests/load/locustfile.py`](../backend/tests/load/locustfile.py).

## Method

- **Traffic:** Locust users send `POST /models/{id}/predict` to every production model (3 in this run),
  with numeric inputs jittered ±20%. Each user waits 0.1–0.5 s between requests.
- **Full serving path:** auth, rate limiting (Redis), schema checks, preprocessing, the model,
  and the S1/S2/S3 prediction log written to Postgres. Nothing is mocked.
- **Pass rule:** p95 ≤ 300 ms and ≤ 1% failed requests. Otherwise Locust exits with code 1.
- **Setup:** the Docker Compose stack on one laptop, Windows 11 with Docker Desktop.
  The load generator ran on the same machine.

## Results (2026-11-01)

| Server | Concurrent users | Throughput | p50 | p95 | p99 | Failures | Verdict |
|---|---|---|---|---|---|---|---|
| Production mode, 4 workers (`api_perf`, :8001) | 20 | 55 req/s | 41 ms | **120 ms** | 190 ms | 0% | ✅ pass |
| Production mode, 4 workers (`api_perf`, :8001) | 35 | 82 req/s | 70 ms | **250 ms** | 360 ms | 0% | ✅ pass (near the limit) |
| Dev server, 1 process + hot reload (:8000) | 20 | 37 req/s | 190 ms | 460 ms | 670 ms | 0% | ❌ fail (expected) |

A single request takes about 29 ms end to end. Of that, preprocessing plus the model take 6–12 ms,
and 10–16 ms once the prediction log insert is included.

## What the test found

1. **One process caps throughput at about 35 req/s.** Preprocessing and the model are CPU-bound,
   so threads inside one Python process don't add capacity. The production image now runs
   `API_WORKERS` uvicorn processes (default 4). Rate-limit counters are kept in Redis,
   so all the processes share them.
2. **The dev server's file watcher wasted CPU.** It scanned the whole bind mount, including a local
   `.venv`, and used about 12% CPU at idle. It now watches only `app/`.
3. **Measurement trap:** on Windows, `http://localhost` added about 50 ms per request on the client side
   (79 ms vs 29 ms for the same call). Use `127.0.0.1`.
4. **Load-test traffic looks like drift, and that is correct.** The scheduled drift checks flagged
   the near-identical traffic. The retraining gate refused to retrain ("not enough ground truth"),
   and the refusal was audited. Run load tests against a scratch database,
   or delete that run's rows from `predictions` afterwards.

## Reproduce

```bash
docker compose up -d                                  # dev stack + a deployed model (docs/DEMO.md)
docker compose --profile perf up -d api_perf          # production-mode API on :8001
cd backend && pip install -r tests/load/requirements.txt
locust -f tests/load/locustfile.py --host http://127.0.0.1:8001 \
       --headless --users 20 --spawn-rate 5 --run-time 2m
```
