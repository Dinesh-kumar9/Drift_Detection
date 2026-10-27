"""
Alembic environment configuration.
Uses async SQLAlchemy engine — required for asyncpg driver.
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import settings

# Import all models so Alembic can detect changes
from app.models import Base  # noqa: F401

# ─── Alembic Config ──────────────────────────────────────────────────────────
config = context.config
if config.config_file_name and not config.attributes.get("skip_logging_config"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)
target_metadata = Base.metadata

# URL precedence: programmatic (tests / bootstrap) → DATABASE_URL env → app settings (.env)
db_url = config.attributes.get("sqlalchemy.url") or os.environ.get("DATABASE_URL") or settings.DATABASE_URL
config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))  # escape ConfigParser interpolation


# ─── Offline Mode ────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ─── Online Mode (async) ─────────────────────────────────────────────────────
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
