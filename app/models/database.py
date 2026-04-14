"""
SQLAlchemy ORM models for persisting forecast data.
Uses async SQLite via aiosqlite — zero infrastructure required.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, text, func, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# ─── Engine & Session ─────────────────────────────────────────────────────────

# NullPool: every request gets a fresh connection — avoids stale file handles
# on macOS when the DB file is recreated between server reloads.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── ORM Models ───────────────────────────────────────────────────────────────

class ForecastRun(Base):
    """Stores one complete forecast run (summary + raw JSON)."""
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_id = Column(String(36), unique=True, nullable=False, index=True)
    energy_type = Column(String(20), nullable=False)           # solar | wind | combined
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_name = Column(String(200), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    horizon_hours = Column(Integer, default=72)
    model_used = Column(String(20), nullable=False)
    total_energy_kwh = Column(Float, nullable=False)
    peak_power_kw = Column(Float, nullable=False)
    capacity_factor = Column(Float, nullable=False)
    raw_response_json = Column(Text, nullable=True)            # Full JSON for retrieval

    def set_raw_response(self, data: dict) -> None:
        self.raw_response_json = json.dumps(data, default=str)

    def get_raw_response(self) -> dict | None:
        if self.raw_response_json:
            return json.loads(self.raw_response_json)
        return None


class ForecastHour(Base):
    """Individual hourly predictions within a forecast run."""
    __tablename__ = "forecast_hours"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    energy_kwh = Column(Float, nullable=False)
    energy_kwh_lower = Column(Float, nullable=False)
    energy_kwh_upper = Column(Float, nullable=False)
    power_kw = Column(Float, nullable=False)
    temperature_2m = Column(Float)
    cloud_cover = Column(Float)
    wind_speed_10m = Column(Float)
    shortwave_radiation = Column(Float)


# ─── DB Lifecycle ─────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Create all tables and indexes on startup.
    Fully idempotent — safe to call multiple times (handles hot-reloads).

    Uses `checkfirst=True` for table creation and `IF NOT EXISTS` for indexes.
    """
    import os
    os.makedirs("data", exist_ok=True)

    async with engine.begin() as conn:
        # Create tables idempotently
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True)
        )
        # Create indexes idempotently — SQLAlchemy's create_all does NOT handle
        # separately-defined Index objects idempotently; use raw SQL instead.
        index_stmts = [
            "CREATE INDEX IF NOT EXISTS ix_forecast_runs_generated_at ON forecast_runs (generated_at)",
            "CREATE INDEX IF NOT EXISTS ix_forecast_runs_energy_type  ON forecast_runs (energy_type)",
            "CREATE INDEX IF NOT EXISTS ix_forecast_hours_forecast_id ON forecast_hours (forecast_id)",
            "CREATE INDEX IF NOT EXISTS ix_forecast_hours_timestamp   ON forecast_hours (timestamp)",
        ]
        for stmt in index_stmts:
            await conn.execute(text(stmt))

    log.info("Database tables and indexes are ready.")


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
