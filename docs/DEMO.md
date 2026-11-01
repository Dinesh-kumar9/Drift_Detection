# End-to-End Demo Script (≈10 minutes)

A walkthrough of the full lifecycle in the browser. It's ordered so that each
step shows one requirement working. Every step below was exercised in a real
browser against the Docker stack.

## 0. Start

```bash
docker compose up --build        # first build ≈ 10 min; later starts ≈ 1 min
```

Open http://localhost:3000. Accounts (click a row on the login page to fill it in):

| Account | Password | Can |
|---|---|---|
| engineer@example.com | engineer1234 | upload, train, deploy, predict, inject faults, request retrains |
| admin@example.com | admin1234 | everything + **approve retraining, promote models** |
| viewer@example.com | viewer1234 | read-only |

A dataset to upload: `cd backend && python -c "import sys; sys.path.insert(0,'scripts'); from demo_lifecycle import make_dataset; open('fraud.csv','wb').write(make_dataset(2000))"`.
It contains 2,000 transactions with numeric and categorical columns and a `fraud`/`legit` label.

## 1. Core lifecycle (FR1–FR7), as engineer

1. **Datasets:** drop in `fraud.csv`. The schema is inferred and baseline statistics are computed. *(FR1)*
2. **Experiments:** pick the dataset and set the target to `label`, then **Train 3 Models in Parallel**.
   The Celery worker trains RandomForest, XGBoost and LogisticRegression on the same split. *(FR2, FR3)*
3. Compare F1, accuracy, ROC-AUC, latency and model size side by side. The best model is marked. *(FR4)*
4. **Deploy to Production** on the best model. *(FR5, FR6)*
5. **Model Registry:** choose *Fill sample from training data*, then *Run Inference*. Every prediction is
   logged with its raw input, transformed features and output. *(FR7)*

## 2. Stage-wise drift + root cause (FR8–FR11), the differentiator

**Observability:** select the model, then inject each fault in turn with **Inject & check**:

| Inject | What it simulates | Expected root cause | Retraining? |
|---|---|---|---|
| S1 · Upstream data shift (`age`) | a source system changes a column | **S1 Ingestion** | *not proposed*: accuracy didn't degrade |
| S2 · Preprocessing bug | a broken scaler gets deployed | **S2 Preprocessing**, with "features diverge from the training-time transform" | *never*: retraining would learn the bug |
| S3 · Concept drift | fraudsters change tactics; labels flip | **S3 Output**, with "error rate rose against ground truth" | **request opened** |

Point out the stage cards, the trend chart, the ranked attribution with reasons, and
the per-feature evidence table. Drift checks also run automatically on the
model's SLA-tier cadence. *(FR9, FR10, FR16)* Alerts are logged, or posted to Slack
when a webhook is configured. *(FR11)*

## 3. Retraining loop with human gates (FR12–FR14)

1. **Retrain Console** (still as engineer): the request opened by the concept-drift check shows
   the cost-benefit gate. All four checks pass, and the error rate went from ~11% to ~87%.
   The console says *"An admin must approve"*. *(FR12)*
2. Sign out, then sign in as **admin**, and choose **Approve retraining** *(FR13, gate 1: spend)*.
   The worker builds a training set from labelled traffic **after** the drift started,
   trains a candidate, and compares it with production on held-out recent data.
   In the recorded run, held-out error dropped from 84.8% to 18.6%.
3. **Promote candidate to production** *(FR13, gate 2: promotion)*. The new version
   goes live with the old one's SLA tier, and the old version is archived.
4. **Audit trail:** `retrain.requested` (system) → `retrain.approved` (admin) →
   `retrain.completed` → `retrain.promoted` (admin). *(FR14)*

## 4. Talking points

- **Why per-stage monitoring matters:** the same "drift detected" alarm means three different fixes.
  Only one of them is retraining.
- **Evaluation:** 0% false alarms, and the correct stage was named every time drift was detected
  (`docs/DRIFT_ENGINE.md`, `scripts/evaluate_attribution.py`).
- **Safety:** the gate blocks retraining on pipeline bugs and on drift that doesn't hurt accuracy.
  Humans approve both the spend and the promotion, and every decision is audited.

## Without the UI

`python scripts/demo_lifecycle.py` runs upload → train → deploy → predict → feedback against the API.
Every endpoint is also documented at http://localhost:8000/docs.
