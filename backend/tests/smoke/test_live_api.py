"""
Smoke tests against a deployed API (staging) — run by deploy.yml after each deploy.

Pure HTTP: no app imports, so the deploy job only needs httpx + pytest.
Skipped unless TEST_API_URL is set; endpoints needing auth also need TEST_API_TOKEN.
"""

import os

import httpx
import pytest

API_URL = os.environ.get("TEST_API_URL", "").rstrip("/")
API_TOKEN = os.environ.get("TEST_API_TOKEN")

pytestmark = pytest.mark.skipif(not API_URL, reason="TEST_API_URL not set")


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=API_URL, timeout=15) as client:
        yield client


@pytest.fixture(scope="module")
def auth():
    if not API_TOKEN:
        pytest.skip("TEST_API_TOKEN not set")
    return {"Authorization": f"Bearer {API_TOKEN}"}


def test_health(http):
    res = http.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_openapi_lists_core_routes(http):
    paths = http.get("/openapi.json").json()["paths"]
    for route in ("/datasets/", "/experiments/train", "/models/{model_version_id}/predict"):
        assert route in paths


def test_protected_routes_reject_anonymous(http):
    assert http.get("/datasets/").status_code in (401, 403)


def test_registry_reachable_with_token(http, auth):
    res = http.get("/models/registered", headers=auth)
    assert res.status_code == 200
    assert "items" in res.json()
