"""pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
