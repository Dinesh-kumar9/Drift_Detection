"""
Phase 3 exit criterion, end to end: a drift event opens a retrain request that waits
for approval; approving it trains a candidate; promoting it deploys a new version —
with every step in the audit trail. Plus the cost-benefit gate's refusals and RBAC.
"""

import numpy as np
import pandas as pd
import pytest

from tests.integration.helpers import upload


@pytest.fixture
def fraud_df() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 1500
    amount = rng.exponential(100, n)
    channel = rng.choice(["web", "pos", "atm"], n)
    return pd.DataFrame(
        {
            "amount": amount,
            "age": rng.integers(18, 80, n).astype(float),
            "channel": channel,
            "label": np.where((amount > 120) | (channel == "atm"), "fraud", "legit"),
        }
    )


@pytest.fixture
async def deployed(client, auth_headers, run_training, fraud_df):
    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")
    res = await client.post(
        "/experiments/train",
        headers=h,
        json={"dataset_id": ds["id"], "target_column": "label", "models": ["RandomForestClassifier"]},
    )
    await run_training()
    cmp = (await client.get(f"/experiments/{res.json()['experiment_group_id']}/compare", headers=h)).json()
    mv_id = cmp["models"][0]["model_version_id"]
    assert (await client.post(f"/models/{mv_id}/deploy", headers=h, json={"sla_tier": "critical"})).status_code == 200
    return mv_id


async def _simulate(client, h, mv_id, **body):
    res = await client.post(f"/observability/{mv_id}/simulate", headers=h, json=body)
    assert res.status_code == 200, res.text
    return res.json()["check"]


async def test_concept_drift_full_retraining_loop(
    client, auth_headers, admin_headers, viewer_headers, deployed, retrain_dispatched, session_factory
):
    from app.services.retrain_service import execute_retrain_job

    # ── drift → automatic retrain request (gate passed, waiting for a human) ──
    check = await _simulate(client, auth_headers, deployed, fault="s3_concept")
    assert check["root_cause"] == "S3_output"
    assert check["retrain"]["status"] == "pending", check["retrain"]
    job_id = check["retrain"]["job_id"]

    job = (await client.get(f"/retrain/jobs/{job_id}", headers=viewer_headers)).json()
    assert job["trigger_type"] == "auto"
    assert job["decision"]["passed"] and job["decision"]["mode"] == "post_onset_only"
    assert job["decision"]["error_increase"] > 0.3
    assert job["registered_model_name"] == "fraud-label"

    # ── human gate 1: only admins approve spend ──
    for headers in (viewer_headers, auth_headers):
        assert (await client.post(f"/retrain/jobs/{job_id}/approve", headers=headers)).status_code == 403
    res = await client.post(f"/retrain/jobs/{job_id}/approve", headers=admin_headers)
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "approved"
    assert retrain_dispatched == [{"job_id": job_id}]

    # ── worker: build post-onset training data, train candidate, evaluate on holdout ──
    out = await execute_retrain_job(job_id, session_factory)
    assert out["status"] == "completed"
    job = (await client.get(f"/retrain/jobs/{job_id}", headers=admin_headers)).json()
    ev = job["evaluation"]
    assert job["status"] == "completed" and job["candidate_model_version_id"]
    assert ev["training_mode"] == "post_onset_only"
    assert ev["candidate_error"] < ev["production_error"], ev  # the new model learned the new relationship
    assert ev["candidate_version_tag"] == "v2"

    # ── human gate 2: promote the candidate ──
    assert (await client.post(f"/retrain/jobs/{job_id}/promote", headers=auth_headers)).status_code == 403
    res = await client.post(f"/retrain/jobs/{job_id}/promote", headers=admin_headers)
    assert res.status_code == 200, res.text
    assert res.json()["promoted_at"]
    candidate = (await client.get(f"/models/{job['candidate_model_version_id']}", headers=viewer_headers)).json()
    old = (await client.get(f"/models/{deployed}", headers=viewer_headers)).json()
    assert (candidate["status"], old["status"]) == ("production", "archived")
    assert candidate["sla_tier"] == "critical"  # inherited from the model it replaced
    assert (await client.post(f"/retrain/jobs/{job_id}/promote", headers=admin_headers)).status_code == 409

    # ── audit trail: trigger → approver → outcome → promotion ──
    logs = (await client.get("/audit-logs/", headers=viewer_headers, params={"action": "retrain."})).json()
    trail = {log["action"]: log for log in logs["items"] if log["target_id"] == job_id}
    assert {"retrain.requested", "retrain.approved", "retrain.completed", "retrain.promoted"} <= set(trail)
    assert trail["retrain.requested"]["actor_email"] == "system"
    assert trail["retrain.approved"]["actor_email"] == "admin@test.io"
    assert trail["retrain.promoted"]["metadata"]["replaced_version_id"] == deployed


async def test_pipeline_fault_is_never_retrained(client, auth_headers, deployed):
    check = await _simulate(client, auth_headers, deployed, fault="s2_shift", feature="amount")
    assert check["root_cause"] == "S2_preprocessing"
    assert check["retrain"]["status"] == "rejected"
    assert "preprocessing pipeline" in check["retrain"]["reason"]


async def test_drift_without_accuracy_loss_is_not_worth_retraining(client, auth_headers, deployed):
    # 'age' doesn't drive the label, so shifting it changes inputs but not accuracy
    check = await _simulate(client, auth_headers, deployed, fault="s1_shift", feature="age")
    assert check["root_cause"] == "S1_ingestion"
    assert check["retrain"]["status"] == "rejected"
    assert "hasn't degraded" in check["retrain"]["reason"]

    # the suppression isn't repeated on every scheduled check
    again = (await client.post(f"/observability/{deployed}/check", headers=auth_headers)).json()
    assert again["retrain"] is None


async def test_manual_trigger_force_reject_and_conflict(client, auth_headers, admin_headers, deployed):
    await _simulate(client, auth_headers, deployed, fault="none")  # some labelled traffic, no drift

    res = await client.post(f"/retrain/{deployed}/trigger", headers=auth_headers, json={"note": "quarterly refresh"})
    assert res.status_code == 201
    assert res.json()["status"] == "rejected"  # gate: accuracy fine

    res = await client.post(f"/retrain/{deployed}/trigger", headers=auth_headers, json={"force": True})
    assert res.status_code == 201
    forced = res.json()
    assert forced["status"] == "pending" and forced["trigger_reason"]["forced"]

    assert (await client.post(f"/retrain/{deployed}/trigger", headers=auth_headers, json={})).status_code == 409

    res = await client.post(
        f"/retrain/jobs/{forced['id']}/reject", headers=admin_headers, json={"reason": "not needed this quarter"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "rejected" and "not needed" in res.json()["rejection_reason"]

    jobs = (await client.get("/retrain/jobs", headers=auth_headers, params={"model_version_id": deployed})).json()
    assert jobs["total"] == 2


async def test_viewer_is_read_only(client, viewer_headers, fraud_df):
    from tests.integration.helpers import csv_bytes

    res = await client.post(
        "/datasets/",
        headers=viewer_headers,
        files={"file": ("x.csv", csv_bytes(fraud_df), "text/csv")},
        data={"name": "x"},
    )
    assert res.status_code == 403
    assert (await client.get("/models/", headers=viewer_headers)).status_code == 200
    assert (await client.get("/observability/overview", headers=viewer_headers)).status_code == 200
