"""pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def rate_limit_in_memory(monkeypatch):
    """Keep rate limiting on in tests, but count in memory and start every test at zero."""
    from app.core import rate_limit
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_STORAGE", "memory")
    rate_limit.reset_memory_store()
    yield
    rate_limit.reset_memory_store()
