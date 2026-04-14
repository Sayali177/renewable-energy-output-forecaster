"""
Health check router.
Used by load balancers, CI/CD pipelines, and monitoring systems.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.forecast_engine import forecast_engine

router = APIRouter(prefix="/api", tags=["Health"])
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Check",
    description="Returns the current health status of the API and its dependent services.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
        services={
            "solar_xgb_model": "ready" if forecast_engine.solar_xgb.is_trained else "not_ready",
            "wind_xgb_model": "ready" if forecast_engine.wind_xgb.is_trained else "not_ready",
            "solar_prophet": "ready" if forecast_engine.solar_prophet.is_fitted else "not_ready",
            "wind_prophet": "ready" if forecast_engine.wind_prophet.is_fitted else "not_ready",
            "weather_api": "open-meteo (no auth required)",
            "database": "sqlite (async)",
        },
    )
