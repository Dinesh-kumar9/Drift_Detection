"""
End-to-end lifecycle against real Postgres:
upload → train → status → compare → deploy → predict → feedback.

This is the Phase 1 exit criterion as an automated test. Every bug fixed in
Phase 1.5 (JSONB group lookup, categorical encoding, serve-time preprocessing,
UUID column types, single global production model) fails at least one test here.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from app.models import AuditLog
from tests.integration.helpers import upload


@pytest.fixture
def fraud_df() -> pd.DataFrame:
    """Mixed-type classification data with string labels and XGBoost-unsafe category values."""
    rng = np.random.default_rng(11)
    n = 400
    amount = rng.exponential(100, n)
    df = pd.DataFrame(
        {
            "amount": amount,
            "age": rng.integers(18, 80, n).astype(float),
            "channel": rng.choice(["web", "pos", "atm"], n),
            "income": rng.choice(["<=50K", ">50K"], n),
            "label": np.where(amount > 150, "fraud", "legit"),
        }
    )
    df.loc[:9, "age"] = np.nan  # nulls must be imputed, not crash
    return df


@pytest.fixture
def house_df() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    n = 300
    sqft = rng.uniform(500, 3000, n)
    return pd.DataFrame(
        {
            "sqft": sqft,
            "rooms": rng.integers(1, 6, n),
            "city": rng.choice(["NY", "SF"], n),
            "price": sqft * 300 + rng.normal(0, 1000, n),
        }
    )


async def _train(client, headers, run_training, dataset_id: str, target: str, **extra) -> dict:
    res = await client.post(
        "/experiments/train", headers=headers, json={"dataset_id": dataset_id, "target_column": target, **extra}
    )
    assert res.status_code == 202, res.text
    body = res.json()
    await run_training()
    return body


async def test_full_classification_lifecycle(client, auth_headers, run_training, session_factory, fraud_df):
    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")
    assert ds["row_count"] == 400
    assert ds["schema_json"]["channel"]["dtype"] == "object"

    # ── Train ────────────────────────────────────────────────────────────────
    res = await client.post("/experiments/train", headers=h, json={"dataset_id": ds["id"], "target_column": "label"})
    assert res.status_code == 202, res.text
    train = res.json()
    group_id = train["experiment_group_id"]
    assert train["task_type"] == "classification"
    assert train["registered_model_name"] == "fraud-label"
    assert len(train["model_run_ids"]) == 3

    status = (await client.get(f"/experiments/{group_id}/status", headers=h)).json()
    assert status["overall_status"] == "queued"

    [summary] = await run_training()
    assert summary["completed_models"] == 3

    status = (await client.get(f"/experiments/{group_id}/status", headers=h)).json()
    assert status["overall_status"] == "completed"
    assert {m["status"] for m in status["models"]} == {"completed"}

    groups = (await client.get("/experiments/groups", headers=h)).json()
    assert groups["total"] == 1
    assert groups["items"][0]["dataset_name"] == "fraud"
    assert groups["items"][0]["n_models"] == 3

    # ── Compare ──────────────────────────────────────────────────────────────
    cmp = (await client.get(f"/experiments/{group_id}/compare", headers=h)).json()
    assert cmp["best_metric"] == "f1"
    assert cmp["best_model_run_id"] in train["model_run_ids"]
    assert sorted(m["version_tag"] for m in cmp["models"]) == ["v1", "v2", "v3"]
    best = next(m for m in cmp["models"] if m["run_id"] == cmp["best_model_run_id"])
    assert best["f1"] > 0.8  # the label is a simple function of amount
    assert best["model_size_kb"] > 0

    # ── Deploy ───────────────────────────────────────────────────────────────
    res = await client.post(f"/models/{best['model_version_id']}/deploy", headers=h, json={"sla_tier": "critical"})
    assert res.status_code == 200, res.text
    deployed = res.json()
    assert deployed["status"] == "production"
    assert deployed["sla_tier"] == "critical"
    assert deployed["input_columns"] == ["amount", "age", "channel", "income"]

    registered = (await client.get("/models/registered", headers=h)).json()["items"]
    assert len(registered) == 1
    assert registered[0]["production_version_id"] == best["model_version_id"]
    assert registered[0]["version_count"] == 3

    async with session_factory() as db:
        [audit] = (await db.execute(select(AuditLog).where(AuditLog.action == "model.deploy"))).scalars().all()
    assert audit.actor_email == "engineer@test.io"
    assert audit.target_id == best["model_version_id"]

    # ── Predict ──────────────────────────────────────────────────────────────
    mv_id = best["model_version_id"]
    sample = (await client.get(f"/models/{mv_id}/sample-input", headers=h)).json()["features"]
    assert set(sample) == {"amount", "age", "channel", "income"}

    res = await client.post(
        f"/models/{mv_id}/predict",
        headers=h,
        json={"features": {"amount": 5000.0, "age": 33, "channel": "web", "income": "<=50K"}},
    )
    assert res.status_code == 200, res.text
    pred = res.json()
    assert pred["prediction"] == "fraud"
    assert set(pred["probabilities"]) == {"fraud", "legit"}

    res = await client.post(
        f"/models/{mv_id}/predict",
        headers=h,
        json={"features": {"amount": 2.0, "age": None, "channel": "NEW_CHANNEL", "income": ">50K"}},
    )
    assert res.status_code == 200, res.text
    assert res.json()["prediction"] == "legit"

    res = await client.post(f"/models/{mv_id}/predict", headers=h, json={"features": {"amount": 1.0}})
    assert res.status_code == 422
    assert res.json()["detail"]["missing_columns"] == ["age", "channel", "income"]

    # ── Prediction log + feedback ────────────────────────────────────────────
    log = (await client.get("/predictions/", headers=h, params={"model_version_id": mv_id})).json()
    assert log["total"] == 2
    entry = next(p for p in log["items"] if p["id"] == pred["prediction_id"])
    assert entry["input_snapshot"]["channel"] == "web"  # S1
    assert entry["features_snapshot"]["channel_web"] == 1.0  # S2
    assert entry["output"]["prediction"] == "fraud"  # S3

    res = await client.post(f"/predictions/{pred['prediction_id']}/feedback", headers=h, json={"actual": "fraud"})
    assert res.status_code == 200
    assert res.json()["actual"] == "fraud"
    assert res.json()["feedback_at"] is not None


async def test_regression_lifecycle(client, auth_headers, run_training, house_df):
    h = auth_headers
    ds = await upload(client, h, house_df, "houses")
    train = await _train(client, h, run_training, ds["id"], "price")
    assert train["task_type"] == "regression"

    cmp = (await client.get(f"/experiments/{train['experiment_group_id']}/compare", headers=h)).json()
    assert cmp["best_metric"] == "r2"
    best = next(m for m in cmp["models"] if m["run_id"] == cmp["best_model_run_id"])
    assert best["r2"] > 0.9

    mv_id = best["model_version_id"]
    assert (await client.post(f"/models/{mv_id}/deploy", headers=h, json={})).status_code == 200
    res = await client.post(
        f"/models/{mv_id}/predict", headers=h, json={"features": {"sqft": 1000, "rooms": 2, "city": "SF"}}
    )
    assert res.status_code == 200, res.text
    assert res.json()["prediction"] == pytest.approx(300_000, rel=0.15)
    assert res.json()["confidence"] is None


async def test_each_lineage_keeps_its_own_production_model(client, auth_headers, run_training, fraud_df, house_df):
    """Promotion used to archive every production model on the platform."""
    h = auth_headers
    fraud = await upload(client, h, fraud_df, "fraud")
    houses = await upload(client, h, house_df, "houses")
    g1 = await _train(client, h, run_training, fraud["id"], "label", models=["LogisticRegression"])
    g2 = await _train(client, h, run_training, houses["id"], "price", models=["Ridge"])

    async def version_of(group):
        cmp = (await client.get(f"/experiments/{group['experiment_group_id']}/compare", headers=h)).json()
        return cmp["models"][0]["model_version_id"]

    fraud_v1, house_v1 = await version_of(g1), await version_of(g2)
    for mv in (fraud_v1, house_v1):
        assert (await client.post(f"/models/{mv}/deploy", headers=h, json={})).status_code == 200

    prod = (await client.get("/models/", headers=h, params={"status_filter": "production"})).json()
    assert {m["id"] for m in prod["items"]} == {fraud_v1, house_v1}

    # Retrain the fraud lineage → v2, promote it: only fraud v1 is archived
    g3 = await _train(client, h, run_training, fraud["id"], "label", models=["RandomForestClassifier"])
    fraud_v2 = await version_of(g3)
    v2 = (await client.post(f"/models/{fraud_v2}/deploy", headers=h, json={})).json()
    assert v2["version_tag"] == "v2"
    assert (
        v2["registered_model_id"] == (await client.get(f"/models/{fraud_v1}", headers=h)).json()["registered_model_id"]
    )

    statuses = {mv: (await client.get(f"/models/{mv}", headers=h)).json()["status"] for mv in (fraud_v1, house_v1)}
    assert statuses == {fraud_v1: "archived", house_v1: "production"}

    res = await client.post(f"/models/{fraud_v1}/predict", headers=h, json={"features": {}})
    assert res.status_code == 400  # archived


async def test_training_failure_is_recorded(client, auth_headers, dispatched, session_factory, fraud_df):
    """A crashing job used to leave every run 'queued' forever."""
    from app.workers.training_tasks import execute_training_job

    h = auth_headers
    fraud_df["label"] = "legit"  # single class → preprocessing error in the worker
    ds = await upload(client, h, fraud_df, "one-class")
    res = await client.post(
        "/experiments/train",
        headers=h,
        json={"dataset_id": ds["id"], "target_column": "label", "task_type": "classification"},
    )
    group_id = res.json()["experiment_group_id"]

    with pytest.raises(ValueError, match="single class"):
        await execute_training_job(dispatched.pop()["experiment_group_id"], session_factory)

    status = (await client.get(f"/experiments/{group_id}/status", headers=h)).json()
    assert status["overall_status"] == "failed"
    assert "single class" in status["error"]
    assert {m["status"] for m in status["models"]} == {"failed"}
    assert (await client.get("/models/", headers=h)).json()["total"] == 0


async def test_train_request_validation(client, auth_headers, fraud_df, dispatched):
    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")

    async def train(**body):
        return await client.post("/experiments/train", headers=h, json={"dataset_id": ds["id"], **body})

    assert (await train(target_column="nope")).status_code == 422
    assert (await train(target_column="label", models=["Ridge"])).status_code == 422  # regression model
    assert (await train(target_column="label", test_size=0.9)).status_code == 422

    res = await client.post(
        "/experiments/train", headers=h, json={"dataset_id": "not-a-uuid", "target_column": "label"}
    )
    assert res.status_code == 422

    missing = "00000000-0000-0000-0000-000000000000"
    res = await client.post("/experiments/train", headers=h, json={"dataset_id": missing, "target_column": "label"})
    assert res.status_code == 404

    assert (await train(target_column="label", model_name="shared")).status_code == 202
    res = await train(target_column="amount", model_name="shared")  # same lineage, different target
    assert res.status_code == 409
    assert len(dispatched) == 1


async def test_upload_rejects_unusable_files(client, auth_headers):
    h = auth_headers

    async def post(name: str, content: bytes):
        return await client.post(
            "/datasets/", headers=h, files={"file": (name, content, "text/csv")}, data={"name": "x"}
        )

    assert (await post("a.txt", b"a,b\n1,2\n")).status_code == 400
    assert (await post("a.csv", b"")).status_code == 400
    assert (await post("a.csv", b"only_one_column\n1\n2\n")).status_code == 400
    assert (await post("UPPER.CSV", b"a,b\n1,2\n3,4\n")).status_code == 201


async def test_auth_flow(client):
    res = await client.post("/auth/register", json={"email": "v@test.io", "password": "password123", "role": "viewer"})
    assert res.status_code == 201
    assert (
        await client.post("/auth/register", json={"email": "v@test.io", "password": "password123"})
    ).status_code == 409
    assert (await client.post("/auth/register", json={"email": "s@test.io", "password": "short"})).status_code == 422

    res = await client.post("/auth/login", json={"email": "v@test.io", "password": "password123"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    me = (await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})).json()
    assert me["email"] == "v@test.io"
    assert me["role"] == "viewer"

    assert (await client.post("/auth/login", json={"email": "v@test.io", "password": "wrong"})).status_code == 401
    assert (await client.get("/datasets/")).status_code in (401, 403)


async def test_broker_outage_fails_fast(client, auth_headers, fraud_df, monkeypatch):
    """If Celery can't be reached the group must not sit in 'queued' forever."""
    from app.workers.training_tasks import run_training_job

    def boom(**_):
        raise ConnectionError("redis down")

    monkeypatch.setattr(run_training_job, "apply_async", boom)
    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")
    res = await client.post("/experiments/train", headers=h, json={"dataset_id": ds["id"], "target_column": "label"})
    assert res.status_code == 503

    [group] = (await client.get("/experiments/groups", headers=h)).json()["items"]
    assert group["status"] == "failed"
    assert "task queue" in group["error"]


async def test_redelivered_job_does_not_duplicate_versions(
    client, auth_headers, dispatched, session_factory, run_training, fraud_df
):
    from app.workers.training_tasks import execute_training_job

    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")
    res = await client.post(
        "/experiments/train",
        headers=h,
        json={"dataset_id": ds["id"], "target_column": "label", "models": ["LogisticRegression"]},
    )
    group_id = res.json()["experiment_group_id"]
    await run_training()

    again = await execute_training_job(group_id, session_factory)  # e.g. Celery redelivery
    assert again["skipped"] is True
    assert (await client.get("/models/", headers=h)).json()["total"] == 1
