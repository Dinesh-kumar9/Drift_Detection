"""
Phase 2 exit criterion, end to end through the API and real Postgres:
inject synthetic drift into a known stage of a deployed model; one check must
rank that stage as the top root cause. Plus the dashboard endpoints and the
SLA-tier scheduler.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import update

from app.models import DriftEvent
from tests.integration.helpers import upload


@pytest.fixture
def fraud_df() -> pd.DataFrame:
    rng = np.random.default_rng(21)
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
    """A production model (with drift baseline) — returns its model_version_id."""
    h = auth_headers
    ds = await upload(client, h, fraud_df, "fraud")
    res = await client.post(
        "/experiments/train",
        headers=h,
        json={"dataset_id": ds["id"], "target_column": "label", "models": ["RandomForestClassifier"]},
    )
    assert res.status_code == 202, res.text
    await run_training()
    cmp = (await client.get(f"/experiments/{res.json()['experiment_group_id']}/compare", headers=h)).json()
    mv_id = cmp["models"][0]["model_version_id"]
    res = await client.post(f"/models/{mv_id}/deploy", headers=h, json={"sla_tier": "critical"})
    assert res.status_code == 200, res.text
    return mv_id


async def _simulate(client, h, mv_id, **body):
    res = await client.post(f"/observability/{mv_id}/simulate", headers=h, json=body)
    assert res.status_code == 200, res.text
    return res.json()


@pytest.mark.parametrize(
    "fault, feature, expected",
    [
        ("s1_shift", "age", "S1_ingestion"),
        ("s1_nulls", "channel", "S1_ingestion"),
        ("s2_shift", "amount", "S2_preprocessing"),
        ("s3_concept", None, "S3_output"),
    ],
)
async def test_injected_fault_is_ranked_root_cause(client, auth_headers, deployed, fault, feature, expected):
    out = await _simulate(client, auth_headers, deployed, fault=fault, feature=feature)
    check = out["check"]
    assert out["traffic"]["stage"] == expected
    assert check["status"] == "completed"
    assert check["drift_detected"]
    assert check["root_cause"] == expected, check["explanation"]
    assert check["ranked"][0]["confidence"] >= 0.5
    assert check["alert"] == "log"  # no Slack webhook configured in tests

    # dashboard endpoints reflect the same check
    drift = (await client.get(f"/observability/{deployed}/drift", headers=auth_headers)).json()
    stages = {s["stage"]: s for s in drift["latest"]["stages"]}
    assert stages[expected]["drifted"]
    assert stages[expected]["onset"] is not None
    assert len(drift["series"]) == 1

    attribution = (await client.get(f"/observability/{deployed}/attribution", headers=auth_headers)).json()
    assert attribution["latest"]["root_cause_stage"] == expected
    assert attribution["latest"]["explanation"].startswith("Most likely root cause")


async def test_healthy_traffic_is_quiet(client, auth_headers, deployed):
    out = await _simulate(client, auth_headers, deployed, fault="none")
    assert out["check"]["status"] == "completed"
    assert not out["check"]["drift_detected"]
    attribution = (await client.get(f"/observability/{deployed}/attribution", headers=auth_headers)).json()
    assert attribution["latest"] is None  # reports are only written when something drifted


async def test_repeat_alert_is_suppressed(client, auth_headers, deployed):
    first = await _simulate(client, auth_headers, deployed, fault="s1_shift", feature="age")
    second = (await client.post(f"/observability/{deployed}/check", headers=auth_headers)).json()
    assert first["check"]["alert"] == "log"
    assert second["root_cause"] == "S1_ingestion"
    assert second["alert"] == "suppressed"


async def test_check_needs_enough_traffic(client, auth_headers, deployed):
    res = await client.post(f"/observability/{deployed}/check", headers=auth_headers)
    assert res.json()["status"] == "insufficient_data"


async def test_overview_lists_production_models(client, auth_headers, deployed):
    h = auth_headers
    [item] = (await client.get("/observability/overview", headers=h)).json()["items"]
    assert item["status"] == "not_checked"

    await _simulate(client, h, deployed, fault="s2_shift", feature="amount")
    [item] = (await client.get("/observability/overview", headers=h)).json()["items"]
    assert item["status"] == "drifting"
    assert item["root_cause"] == "S2_preprocessing"
    assert item["sla_tier"] == "critical"
    assert item["predictions_24h"] == 600


async def test_simulate_validation(client, auth_headers, deployed):
    res = await client.post(
        f"/observability/{deployed}/simulate", headers=auth_headers, json={"fault": "s1_shift", "feature": "nope"}
    )
    assert res.status_code == 422


async def test_scheduler_respects_sla_tier(client, auth_headers, deployed, session_factory):
    from app.workers.drift_tasks import due_model_versions

    async with session_factory() as db:
        assert await due_model_versions(db, datetime.now(timezone.utc)) == [deployed]  # never checked

    await _simulate(client, auth_headers, deployed, fault="none")
    async with session_factory() as db:
        assert await due_model_versions(db, datetime.now(timezone.utc)) == []  # just checked (critical = 60s)
        await db.execute(update(DriftEvent).values(detected_at=datetime.now(timezone.utc) - timedelta(minutes=2)))
        await db.commit()
        assert await due_model_versions(db, datetime.now(timezone.utc)) == [deployed]
