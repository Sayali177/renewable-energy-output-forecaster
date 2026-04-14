"""
History router — query past forecast runs stored in the database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db, ForecastRun
from app.models.schemas import ForecastRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get(
    "",
    response_model=list[ForecastRecord],
    summary="List Forecast History",
    description="Returns a paginated list of past forecasts, newest first.",
)
async def list_history(
    energy_type: Optional[str] = Query(None, description="Filter by 'solar', 'wind', or 'combined'"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip (for pagination)"),
    db: AsyncSession = Depends(get_db),
) -> list[ForecastRecord]:
    q = select(ForecastRun).order_by(desc(ForecastRun.generated_at))

    if energy_type:
        q = q.where(ForecastRun.energy_type == energy_type)

    q = q.limit(limit).offset(offset)
    result = await db.execute(q)
    records = result.scalars().all()

    return [ForecastRecord.model_validate(r) for r in records]


@router.get(
    "/count",
    summary="Count Forecasts",
    description="Returns the total number of stored forecasts, optionally filtered by type.",
)
async def count_history(
    energy_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    q = select(func.count(ForecastRun.id))
    if energy_type:
        q = q.where(ForecastRun.energy_type == energy_type)

    result = await db.execute(q)
    count = result.scalar_one()
    return {"count": count, "energy_type": energy_type or "all"}


@router.get(
    "/{forecast_id}",
    response_model=dict,
    summary="Get Forecast by ID",
    description="Returns the complete stored JSON for a specific forecast run.",
)
async def get_forecast_by_id(
    forecast_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    q = select(ForecastRun).where(ForecastRun.forecast_id == forecast_id)
    result = await db.execute(q)
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast ID '{forecast_id}' not found",
        )

    raw = record.get_raw_response()
    if raw is None:
        return {"forecast_id": forecast_id, "detail": "Raw response not stored"}
    return raw


@router.delete(
    "/{forecast_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Forecast",
    description="Permanently delete a forecast record from the database.",
)
async def delete_forecast(
    forecast_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    q = select(ForecastRun).where(ForecastRun.forecast_id == forecast_id)
    result = await db.execute(q)
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast ID '{forecast_id}' not found",
        )

    await db.delete(record)
