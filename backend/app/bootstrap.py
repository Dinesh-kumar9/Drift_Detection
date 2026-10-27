"""
Container bootstrap: `python -m app.bootstrap`

1. Apply Alembic migrations (`alembic upgrade head`) — the only way tables get created
2. In local dev, seed demo users so a fresh clone is usable immediately

Idempotent; safe to run on every API container start.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command
from alembic.config import Config
from app.core.config import settings
from app.core.security import hash_password

logger = logging.getLogger("bootstrap")

BACKEND_DIR = Path(__file__).resolve().parents[1]

DEMO_USERS = [
    ("admin@example.com", "admin1234", "admin"),
    ("engineer@example.com", "engineer1234", "engineer"),
    ("viewer@example.com", "viewer1234", "viewer"),
]


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    # alembic.ini's logging config would reset the root logger to WARN and swallow our own INFO output
    cfg.attributes["skip_logging_config"] = True
    if database_url:
        cfg.attributes["sqlalchemy.url"] = database_url
    return cfg


def run_migrations(database_url: str | None = None) -> None:
    logger.info("Applying database migrations…")
    command.upgrade(alembic_config(database_url), "head")


async def seed_demo_users(database_url: str | None = None) -> list[str]:
    """Create demo users that don't exist yet. Returns the emails created."""
    from app.models import User  # imported late so alembic's logging config applies first

    engine = create_async_engine(database_url or settings.DATABASE_URL, poolclass=NullPool)
    created = []
    try:
        async with engine.begin() as conn:
            existing = set((await conn.execute(select(User.email))).scalars().all())
            for email, password, role in DEMO_USERS:
                if email in existing:
                    continue
                await conn.execute(
                    User.__table__.insert().values(
                        id=str(uuid.uuid4()),
                        email=email,
                        hashed_password=hash_password(password),
                        role=role,
                    )
                )
                created.append(email)
    finally:
        await engine.dispose()
    return created


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s [%(name)s] %(message)s")
    if settings.RUN_MIGRATIONS_ON_STARTUP:
        run_migrations()
    if settings.ENVIRONMENT == "local" and settings.SEED_DEMO_USERS:
        created = asyncio.run(seed_demo_users())
        if created:
            logger.info("Seeded demo users: %s", ", ".join(created))
        logger.info(
            "Demo logins (local only): %s",
            " | ".join(f"{email} / {password}" for email, password, _ in DEMO_USERS),
        )


if __name__ == "__main__":
    main()
