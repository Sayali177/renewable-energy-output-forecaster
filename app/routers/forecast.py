"""
Forecast router — core prediction endpoints.

All forecast endpoints:
  1. Validate the incoming request (Pydantic does this automatically)
  2. Fetch weather from Open-Meteo
  3. Run the appropriate forecast model(s)
  4. Persist the result to the database
  5. Return the structured response
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db, ForecastRun
from app.models.schemas import (
    ForecastRequest, ForecastResponse, CombinedForecastResponse,
    ForecastSummary, EnergyType, ModelType, LocationRequest,
)
from app.services.forecast_engine import forecast_engine
from app.services.weather_service import WeatherService, WeatherServiceError

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/forecast", tags=["Forecast"])


async def _fetch_weather(location: LocationRequest, hours: int) -> list:
    """Shared helper: fetch weather data and raise HTTP 502 on failure."""
    async with WeatherService() as svc:
        try:
            return await svc.get_forecast(location.latitude, location.longitude, hours)
        except WeatherServiceError as e:
            logger.error("Weather API error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch weather data: {str(e)}",
            )


async def _save_forecast(
    db: AsyncSession,
    response: ForecastResponse,
) -> None:
    """Persist forecast summary to database."""
    try:
        record = ForecastRun(
            forecast_id=response.forecast_id,
            energy_type=response.energy_type.value,
            latitude=response.location.latitude,
            longitude=response.location.longitude,
            location_name=response.location.name,
            generated_at=response.generated_at,
            horizon_hours=response.horizon_hours,
            model_used=response.model_used.value,
            total_energy_kwh=response.summary.total_energy_kwh,
            peak_power_kw=response.summary.peak_power_kw,
            capacity_factor=response.summary.capacity_factor,
        )
        record.set_raw_response(response.model_dump(mode="json"))
        db.add(record)
        await db.flush()
    except Exception as e:
        logger.warning("Failed to persist forecast to DB (non-fatal): %s", e)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/solar",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Solar Energy Forecast",
    description="""
    Generate a 72-hour solar energy forecast for a photovoltaic farm.

    The forecast uses a weighted ensemble of:
    - **XGBoost** (physics-informed, primary model, R² ≈ 0.97)
    - **Prophet** (Meta's time-series model for seasonality)
    - **Physics model** (PV equation with temperature correction)

    Returns hourly kWh predictions with 90% confidence intervals.
    """,
)
async def solar_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db),
) -> ForecastResponse:
    weather = await _fetch_weather(request.location, request.horizon_hours)
    response = await forecast_engine.forecast_solar(request, weather)
    await _save_forecast(db, response)
    return response


@router.post(
    "/wind",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Wind Energy Forecast",
    description="""
    Generate a 72-hour wind energy forecast for a wind turbine installation.

    Uses an ensemble of XGBoost, Prophet, and the Betz-limit aerodynamic
    power equation with real air density correction.

    Returns hourly kWh predictions with confidence intervals that widen
    at high wind speeds due to atmospheric turbulence.
    """,
)
async def wind_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db),
) -> ForecastResponse:
    weather = await _fetch_weather(request.location, request.horizon_hours)
    response = await forecast_engine.forecast_wind(request, weather)
    await _save_forecast(db, response)
    return response


@router.post(
    "/combined",
    response_model=CombinedForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Combined Solar + Wind Forecast",
    description="""
    Generate both solar and wind forecasts simultaneously for a hybrid energy park.

    Returns individual solar and wind forecasts plus a combined summary
    showing total generation potential.
    """,
)
async def combined_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db),
) -> CombinedForecastResponse:
    # Single weather fetch shared by both models
    weather = await _fetch_weather(request.location, request.horizon_hours)

    # Run both forecasts concurrently
    import asyncio
    solar_task = forecast_engine.forecast_solar(request, weather)
    wind_task = forecast_engine.forecast_wind(request, weather)
    solar_response, wind_response = await asyncio.gather(solar_task, wind_task)

    # Persist both
    await _save_forecast(db, solar_response)
    await _save_forecast(db, wind_response)

    # Build combined hourly view
    hourly_combined = []
    for s, w in zip(solar_response.hourly, wind_response.hourly):
        hourly_combined.append({
            "timestamp": s.timestamp.isoformat(),
            "solar_kwh": s.energy_kwh,
            "wind_kwh": w.energy_kwh,
            "total_kwh": round(s.energy_kwh + w.energy_kwh, 3),
            "solar_kw": s.power_kw,
            "wind_kw": w.power_kw,
            "total_kw": round(s.power_kw + w.power_kw, 3),
        })

    import numpy as np
    total_arr = [h["total_kwh"] for h in hourly_combined]
    combined_summary = ForecastSummary(
        total_energy_kwh=round(float(sum(total_arr)), 2),
        avg_power_kw=round(float(np.mean(total_arr)), 2),
        peak_power_kw=round(float(max(total_arr)), 2),
        peak_hour=solar_response.hourly[int(np.argmax(total_arr))].timestamp,
        capacity_factor=round(
            float(np.mean(total_arr)) / (
                solar_response.metadata.get("rated_capacity_kw", 1)
                + wind_response.metadata.get("rated_capacity_kw", 1)
            ),
            4,
        ),
    )

    from uuid import uuid4
    return CombinedForecastResponse(
        forecast_id=str(uuid4()),
        location=request.location,
        generated_at=datetime.now(timezone.utc),
        horizon_hours=request.horizon_hours,
        solar=solar_response,
        wind=wind_response,
        combined_summary=combined_summary,
        hourly_combined=hourly_combined,
    )


@router.get(
    "/latest",
    response_model=dict,
    summary="Latest Stored Forecast",
    description="Returns the most recently generated and stored forecast from the database.",
)
async def get_latest_forecast(
    energy_type: str = "solar",
    db: AsyncSession = Depends(get_db),
) -> dict:
    from sqlalchemy import select, desc

    q = (
        select(ForecastRun)
        .where(ForecastRun.energy_type == energy_type)
        .order_by(desc(ForecastRun.generated_at))
        .limit(1)
    )
    result = await db.execute(q)
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {energy_type} forecasts found in database",
        )

    return record.get_raw_response() or {}
