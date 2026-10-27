"""
Database access for Celery tasks.

Each task runs its coroutine in a fresh event loop (`asyncio.run`). asyncpg
connections are bound to the loop that created them, so reusing the API's
module-level pooled engine across tasks fails with "attached to a different
loop" on the second task. Tasks get their own NullPool engine instead.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings


@asynccontextmanager
async def task_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    try:
        yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
