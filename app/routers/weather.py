"""
Weather router — standalone weather data endpoints.
Useful for testing the weather integration separately from forecasting.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.config import get_settings
from app.services.weather_service import WeatherService, WeatherServiceError

router = APIRouter(prefix="/api/weather", tags=["Weather"])
settings = get_settings()


@router.get(
    "/current",
    summary="Current Weather",
    description="Return the current hour's weather conditions at a given location or city.",
)
async def current_weather(
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
    city: str | None = Query(None, description="City name to search for (e.g. 'New York')"),
) -> dict:
    # Default to Mumbai if nothing provided
    if not city and latitude is None and longitude is None:
        latitude = settings.default_latitude
        longitude = settings.default_longitude

    async with WeatherService() as svc:
        try:
            return await svc.get_current_conditions(latitude=latitude, longitude=longitude, city=city)
        except WeatherServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            )


@router.get(
    "/forecast",
    summary="Raw Weather Forecast",
    description="Return raw hourly weather forecast data without energy prediction.",
)
async def weather_forecast(
    latitude: float = Query(settings.default_latitude, ge=-90, le=90),
    longitude: float = Query(settings.default_longitude, ge=-180, le=180),
    hours: int = Query(72, ge=1, le=168),
) -> dict:
    async with WeatherService() as svc:
        try:
            weather_list = await svc.get_forecast(latitude, longitude, hours)
            return {
                "latitude": latitude,
                "longitude": longitude,
                "hours": hours,
                "data": [
                    {
                        "timestamp": w.timestamp.isoformat(),
                        "temperature_c": w.temperature_2m,
                        "cloud_cover_pct": w.cloud_cover,
                        "wind_speed_ms": w.wind_speed_10m,
                        "wind_direction_deg": w.wind_direction_10m,
                        "solar_ghi_wm2": w.shortwave_radiation,
                        "solar_dni_wm2": w.direct_normal_irradiance,
                        "precipitation_mm": w.precipitation,
                        "humidity_pct": w.relative_humidity_2m,
                        "pressure_hpa": w.surface_pressure,
                    }
                    for w in weather_list
                ],
            }
        except WeatherServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            )
