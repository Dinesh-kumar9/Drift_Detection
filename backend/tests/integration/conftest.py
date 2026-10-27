"""
Integration test fixtures: real Postgres (migrated with Alembic), S3 mocked
with moto, the FastAPI app in-process, and Celery replaced by a recorder so
tests can run training jobs inline.

Requires TEST_DATABASE_URL (CI provides one); the whole package is skipped otherwise.
WARNING: tables in that database are truncated before every test.
"""

import os
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from moto import mock_aws
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def pytest_collection_modifyitems(config, items):
    if TEST_DATABASE_URL:
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set — integration tests need Postgres")
    for item in items:
        if "tests/integration" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_db() -> str:
    from app.bootstrap import run_migrations

    run_migrations(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


@pytest.fixture
async def session_factory(migrated_db) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    from app.models import Base

    engine = create_async_engine(migrated_db, poolclass=NullPool)
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def s3(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "S3_ENDPOINT_URL", None)  # moto only intercepts real AWS endpoints
    with mock_aws():
        yield


@pytest.fixture
def dispatched(monkeypatch) -> list[dict]:
    """Records Celery dispatches instead of sending them to Redis."""
    from app.workers.training_tasks import run_training_job

    calls: list[dict] = []
    monkeypatch.setattr(run_training_job, "apply_async", lambda kwargs=None, **_: calls.append(kwargs or {}))
    return calls


@pytest.fixture
async def client(session_factory, s3, dispatched) -> AsyncIterator[AsyncClient]:
    from app.core.database import get_db
    from app.main import app
    from app.services import serving_service

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    serving_service.invalidate_cache()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    serving_service.invalidate_cache()


@pytest.fixture
async def auth_headers(client) -> dict[str, str]:
    res = await client.post(
        "/auth/register", json={"email": "engineer@test.io", "password": "password123", "role": "engineer"}
    )
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def run_training(dispatched, session_factory):
    """Execute every training job dispatched so far, as the Celery worker would."""
    from app.workers.training_tasks import execute_training_job

    async def _run() -> list[dict]:
        results = []
        while dispatched:
            kwargs = dispatched.pop(0)
            results.append(await execute_training_job(kwargs["experiment_group_id"], session_factory))
        return results

    return _run
