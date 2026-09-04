from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if settings.is_sqlite else {},
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    import app.models  # noqa: F401 — register metadata including new tables

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_patch_model_versions)


def _patch_model_versions(sync_conn) -> None:
    """Add Phase 7 registry columns on existing SQLite/Postgres databases."""
    from sqlalchemy import inspect, text

    inspector = inspect(sync_conn)
    if "model_versions" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("model_versions")}
    additions = [
        ("model_id", "VARCHAR(64) DEFAULT ''"),
        ("dataset", "VARCHAR(64)"),
        ("feature_set", "JSON"),
        ("training_rows", "INTEGER DEFAULT 0"),
        ("positive_rows", "INTEGER DEFAULT 0"),
        ("evaluation_rows", "INTEGER DEFAULT 0"),
        ("status", "VARCHAR(16) DEFAULT 'CANDIDATE'"),
    ]
    missing_status = "status" not in cols
    for name, spec in additions:
        if name in cols:
            continue
        sync_conn.execute(text(f"ALTER TABLE model_versions ADD COLUMN {name} {spec}"))
    if missing_status:
        sync_conn.execute(text("UPDATE model_versions SET status = 'ACTIVE' WHERE is_active = 1"))
        sync_conn.execute(text("UPDATE model_versions SET status = 'CANDIDATE' WHERE is_active = 0 OR is_active IS NULL"))
        sync_conn.execute(text("UPDATE model_versions SET model_id = version WHERE model_id IS NULL OR model_id = ''"))
