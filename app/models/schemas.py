"""
Pydantic schemas for all API request/response models.
These are the "contract" of the API — validated automatically by FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class EnergyType(str, Enum):
    solar = "solar"
    wind = "wind"
    combined = "combined"


class ModelType(str, Enum):
    xgboost = "xgboost"
    prophet = "prophet"
    physics = "physics"
    ensemble = "ensemble"


# ─── Request Models ───────────────────────────────────────────────────────────

class LocationRequest(BaseModel):
    """Geographical location for the farm/turbine."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    name: Optional[str] = Field(None, description="Human-readable location name")

    @field_validator("latitude")
    @classmethod
    def round_lat(cls, v: float) -> float:
        return round(v, 6)

    @field_validator("longitude")
    @classmethod
    def round_lon(cls, v: float) -> float:
        return round(v, 6)


class ForecastRequest(BaseModel):
    """Full forecast request — location + optional overrides."""
    location: LocationRequest
    horizon_hours: int = Field(72, ge=1, le=168, description="Forecast horizon in hours (max 7 days)")
    model: ModelType = Field(ModelType.ensemble, description="Which ML model to use")

    # Optional farm parameter overrides (fall back to settings defaults)
    panel_efficiency: Optional[float] = Field(None, ge=0.01, le=0.50)
    panel_area_m2: Optional[float] = Field(None, gt=0)
    turbine_radius_m: Optional[float] = Field(None, gt=0)
    power_coefficient: Optional[float] = Field(None, gt=0, le=0.593)


# ─── Weather Data Models ──────────────────────────────────────────────────────

class HourlyWeather(BaseModel):
    """Single-hour weather snapshot."""
    timestamp: datetime
    temperature_2m: float = Field(..., description="Air temperature at 2m (°C)")
    relative_humidity_2m: float = Field(..., description="Relative humidity (%)")
    cloud_cover: float = Field(..., description="Cloud cover (0–100%)")
    wind_speed_10m: float = Field(..., description="Wind speed at 10m (m/s)")
    wind_direction_10m: float = Field(..., description="Wind direction (degrees)")
    shortwave_radiation: float = Field(..., description="Global horizontal irradiance (W/m²)")
    direct_normal_irradiance: float = Field(..., description="Direct normal irradiance (W/m²)")
    diffuse_radiation: float = Field(..., description="Diffuse horizontal irradiance (W/m²)")
    precipitation: float = Field(..., description="Precipitation (mm)")
    surface_pressure: float = Field(..., description="Surface pressure (hPa)")


# ─── Prediction Models ────────────────────────────────────────────────────────

class HourlyPrediction(BaseModel):
    """Single-hour energy prediction with confidence interval."""
    timestamp: datetime
    energy_kwh: float = Field(..., ge=0, description="Predicted energy output (kWh)")
    energy_kwh_lower: float = Field(..., ge=0, description="Lower bound (5th percentile)")
    energy_kwh_upper: float = Field(..., ge=0, description="Upper bound (95th percentile)")
    power_kw: float = Field(..., ge=0, description="Instantaneous power (kW)")
    model_used: ModelType
    weather: HourlyWeather


class ForecastSummary(BaseModel):
    """Aggregate statistics over the full forecast horizon."""
    total_energy_kwh: float
    avg_power_kw: float
    peak_power_kw: float
    peak_hour: datetime
    capacity_factor: float = Field(..., description="Actual / rated capacity (0–1)")


class ForecastResponse(BaseModel):
    """Complete 72-hour forecast response."""
    forecast_id: str
    energy_type: EnergyType
    location: LocationRequest
    generated_at: datetime
    horizon_hours: int
    model_used: ModelType
    summary: ForecastSummary
    hourly: list[HourlyPrediction]
    metadata: dict = Field(default_factory=dict)


class CombinedForecastResponse(BaseModel):
    """Solar + wind combined forecast response."""
    forecast_id: str
    location: LocationRequest
    generated_at: datetime
    horizon_hours: int
    solar: ForecastResponse
    wind: ForecastResponse
    combined_summary: ForecastSummary
    hourly_combined: list[dict]


# ─── History Models ───────────────────────────────────────────────────────────

class ForecastRecord(BaseModel):
    """Stored forecast record from the database."""
    id: int
    forecast_id: str
    energy_type: str
    latitude: float
    longitude: float
    location_name: Optional[str]
    generated_at: datetime
    horizon_hours: int
    model_used: str
    total_energy_kwh: float
    peak_power_kw: float
    capacity_factor: float

    model_config = {"from_attributes": True}


# ─── Health Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: dict[str, str]
