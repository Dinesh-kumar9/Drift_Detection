"""Rate limiter: bucket selection, per-user/per-IP counting, 429 responses, fail-open."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core import rate_limit
from app.core.config import settings
from app.core.security import create_access_token


def make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(rate_limit.RateLimitMiddleware)

    @app.post("/auth/login")
    async def login():
        return {"ok": True}

    @app.post("/models/{mid}/predict")
    async def predict(mid: str):
        return {"ok": True}

    @app.get("/models/registered")
    async def registered():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=make_app()), base_url="http://test") as ac:
        yield ac


def bearer(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id, role='engineer')}"}


@pytest.mark.parametrize(
    "method,path,scope",
    [
        ("POST", "/auth/login", "auth"),
        ("POST", "/auth/register", "auth"),
        ("POST", "/models/abc/predict", "predict"),
        ("POST", "/datasets/", "jobs"),
        ("POST", "/experiments/train", "jobs"),
        ("POST", "/observability/abc/check", "jobs"),
        ("POST", "/observability/abc/simulate", "jobs"),
        ("POST", "/retrain/abc/trigger", "jobs"),
        ("POST", "/retrain/jobs/abc/approve", "jobs"),
        ("POST", "/retrain/jobs/abc/promote", "api"),
        ("POST", "/predictions/abc/feedback", "api"),
        ("GET", "/models/abc/predict", "api"),
        ("GET", "/observability/overview", "api"),
    ],
)
def test_each_route_maps_to_one_bucket(method, path, scope):
    assert rate_limit.match_rule(method, path).scope == scope


async def test_login_is_limited_per_ip_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3)
    for remaining in (2, 1, 0):
        res = await client.post("/auth/login")
        assert res.status_code == 200
        assert res.headers["X-RateLimit-Remaining"] == str(remaining)

    res = await client.post("/auth/login")
    assert res.status_code == 429
    assert 1 <= int(res.headers["Retry-After"]) <= 60
    assert "sign-in attempts" in res.json()["detail"]
    # a token doesn't buy extra login attempts — auth is always counted per IP
    assert (await client.post("/auth/login", headers=bearer("u1"))).status_code == 429


async def test_users_have_separate_buckets_and_bad_tokens_count_per_ip(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PREDICT_PER_MINUTE", 2)
    for _ in range(2):
        assert (await client.post("/models/m/predict", headers=bearer("alice"))).status_code == 200
    assert (await client.post("/models/m/predict", headers=bearer("alice"))).status_code == 429
    assert (await client.post("/models/m/predict", headers=bearer("bob"))).status_code == 200

    junk = {"Authorization": "Bearer not-a-jwt"}
    assert (await client.post("/models/m/predict", headers=junk)).status_code == 200
    assert (await client.post("/models/m/predict")).status_code == 200  # same IP bucket
    assert (await client.post("/models/m/predict", headers=junk)).status_code == 429


async def test_buckets_are_independent(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PREDICT_PER_MINUTE", 1)
    h = bearer("alice")
    await client.post("/models/m/predict", headers=h)
    assert (await client.post("/models/m/predict", headers=h)).status_code == 429
    assert (await client.get("/models/registered", headers=h)).status_code == 200


async def test_health_preflight_and_disabled_are_not_counted(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_DEFAULT_PER_MINUTE", 0)
    assert (await client.get("/health")).status_code == 200
    assert (await client.options("/models/registered")).status_code != 429
    assert (await client.get("/models/registered")).status_code == 429

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    assert (await client.get("/models/registered")).status_code == 200


async def test_fails_open_when_the_store_is_down(client, monkeypatch):
    class Broken:
        async def hit(self, key):
            raise ConnectionError("redis down")

    monkeypatch.setattr(settings, "RATE_LIMIT_DEFAULT_PER_MINUTE", 0)
    monkeypatch.setattr(rate_limit, "get_store", lambda: Broken())
    res = await client.get("/models/registered")
    assert res.status_code == 200
    assert "X-RateLimit-Limit" not in res.headers


async def test_forwarded_for_is_only_trusted_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1)
    spoof = {"X-Forwarded-For": "10.0.0.9"}
    assert (await client.post("/auth/login", headers=spoof)).status_code == 200
    assert (await client.post("/auth/login", headers={"X-Forwarded-For": "10.0.0.10"})).status_code == 429

    monkeypatch.setattr(settings, "RATE_LIMIT_TRUST_FORWARDED_FOR", True)
    assert (await client.post("/auth/login", headers={"X-Forwarded-For": "10.0.0.11, 172.16.0.1"})).status_code == 200


async def test_real_app_returns_429_with_cors_headers(monkeypatch):
    """CORS must wrap the limiter, or the browser reports a CORS error instead of the 429."""
    from app.main import app

    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 0)
    origin = settings.CORS_ORIGINS[0]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/auth/login", json={"email": "a@b.co", "password": "x"}, headers={"Origin": origin})
    assert res.status_code == 429
    assert res.headers["access-control-allow-origin"] == origin
    assert "Retry-After" in res.headers.get("access-control-expose-headers", "")


async def test_memory_store_resets_each_window(monkeypatch):
    store = rate_limit.MemoryStore()
    clock = iter([0.0, 1.0, 61.0])
    monkeypatch.setattr(rate_limit.time, "time", lambda: next(clock))
    assert await store.hit("k") == 1
    assert await store.hit("k") == 2
    assert await store.hit("k") == 1
