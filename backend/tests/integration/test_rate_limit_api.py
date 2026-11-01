"""Rate limiting through the real app, database and auth flow."""

from app.core.config import settings


async def test_password_guessing_is_blocked_even_for_the_right_password(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 5)
    creds = {"email": "victim@test.io", "password": "password123"}
    assert (await client.post("/auth/register", json={**creds, "role": "viewer"})).status_code == 201

    for _ in range(4):  # register + 4 guesses = the 5 allowed attempts
        res = await client.post("/auth/login", json={**creds, "password": "wrong-guess"})
        assert res.status_code == 401

    res = await client.post("/auth/login", json=creds)
    assert res.status_code == 429
    assert "Retry-After" in res.headers


async def test_authenticated_traffic_carries_quota_headers(client, auth_headers):
    res = await client.get("/models/registered", headers=auth_headers)
    assert res.status_code == 200
    assert res.headers["X-RateLimit-Limit"] == str(settings.RATE_LIMIT_DEFAULT_PER_MINUTE)
    assert int(res.headers["X-RateLimit-Remaining"]) == settings.RATE_LIMIT_DEFAULT_PER_MINUTE - 1
