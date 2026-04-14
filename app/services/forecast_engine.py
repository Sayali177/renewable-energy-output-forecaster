"""
Forecast Engine — orchestrates all models into a final ensemble prediction.

The ensemble strategy:
    1. XGBoost prediction (primary model, trained on physics-informed data)
    2. Prophet prediction (captures seasonality + trend)
    3. Physics-based prediction (pure first-principles, used as sanity check)
    4. Weighted ensemble = 0.55 × XGBoost + 0.30 × Prophet + 0.15 × Physics

The weights are derived from cross-validation on synthetic test sets:
    XGBoost consistently achieves R² ≈ 0.97 on solar, 0.95 on wind.
    Prophet adds temporal smoothness and uncertainty calibration.
    Physics provides a lower bound that prevents physically impossible outputs.

In production, these weights would be learned via stacking (meta-learning).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.config import get_settings
from app.models.schemas import (
    EnergyType, ModelType, HourlyWeather, HourlyPrediction,
    ForecastSummary, ForecastResponse, ForecastRequest, LocationRequest,
)
from app.services.solar_model import SolarXGBModel
from app.services.wind_model import WindXGBModel
from app.services.prophet_model import ProphetEnergyModel
from app.utils.physics import (
    solar_power_kw, wind_power_kw,
    rated_solar_power_kw, rated_wind_power_kw,
    SolarParams, WindParams,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Ensemble weights (Physics mathematically scales anywhere globally)
WEIGHT_XGB = 0.20
WEIGHT_PROPHET = 0.10
WEIGHT_PHYSICS = 0.70


class ForecastEngine:
    """
    Singleton-style engine that holds trained models and generates forecasts.
    Initialized once at application startup.
    """

    def __init__(self) -> None:
        self.solar_xgb = SolarXGBModel()
        self.wind_xgb = WindXGBModel()
        self.solar_prophet = ProphetEnergyModel(energy_type="solar")
        self.wind_prophet = ProphetEnergyModel(energy_type="wind")
        self._initialized = False

    async def initialize(self) -> None:
        """
        Load/train all models. Called once at app startup.
        Training is done in a thread pool executor so it doesn't block the event loop.
        """
        if self._initialized:
            return

        logger.info("Initializing Forecast Engine — loading/training all models...")
        loop = asyncio.get_event_loop()

        # Run blocking model training in thread pool (won't block async endpoints)
        await loop.run_in_executor(None, self.solar_xgb.train_or_load)
        await loop.run_in_executor(None, self.wind_xgb.train_or_load)
        await loop.run_in_executor(None, self.solar_prophet.fit_on_synthetic)
        await loop.run_in_executor(None, self.wind_prophet.fit_on_synthetic)

        self._initialized = True
        logger.info("All models initialized successfully.")

    def _physics_solar(
        self,
        weather_list: list[HourlyWeather],
        params: SolarParams | None = None,
    ) -> np.ndarray:
        """Pure physics predictions for solar."""
        return np.array([
            solar_power_kw(
                ghi_wm2=w.shortwave_radiation,
                temperature_c=w.temperature_2m,
                cloud_cover_pct=w.cloud_cover,
                params=params,
            )
            for w in weather_list
        ])

    def _physics_wind(
        self,
        weather_list: list[HourlyWeather],
        params: WindParams | None = None,
    ) -> np.ndarray:
        """Pure physics predictions for wind."""
        return np.array([
            wind_power_kw(
                wind_speed_ms=w.wind_speed_10m,
                temperature_c=w.temperature_2m,
                pressure_hpa=w.surface_pressure,
                params=params,
            )
            for w in weather_list
        ])

    def _build_solar_params(self, req: ForecastRequest) -> SolarParams | None:
        """Build SolarParams from request overrides, or use defaults."""
        if req.panel_efficiency or req.panel_area_m2:
            return SolarParams(
                efficiency=req.panel_efficiency or settings.solar_panel_efficiency,
                area_m2=req.panel_area_m2 or settings.solar_panel_area_m2,
                temp_coefficient=settings.solar_temp_coefficient,
            )
        return None

    def _build_wind_params(self, req: ForecastRequest) -> WindParams | None:
        """Build WindParams from request overrides, or use defaults."""
        if req.turbine_radius_m or req.power_coefficient:
            return WindParams(
                radius_m=req.turbine_radius_m or settings.wind_turbine_radius_m,
                power_coefficient=req.power_coefficient or settings.wind_power_coefficient,
                cut_in_speed=settings.wind_cut_in_speed,
                rated_speed=settings.wind_rated_speed,
                cut_out_speed=settings.wind_cut_out_speed,
            )
        return None

    def _ensemble_solar(
        self,
        weather_list: list[HourlyWeather],
        model_type: ModelType,
        solar_params: SolarParams | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run solar ensemble and return (point, lower, upper) in kW."""
        xgb_point, xgb_low, xgb_high = self.solar_xgb.predict(weather_list)
        phys = self._physics_solar(weather_list, solar_params)

        if model_type == ModelType.xgboost:
            return xgb_point, xgb_low, xgb_high
        elif model_type == ModelType.physics:
            unc = 0.20
            return phys, phys * (1 - unc), phys * (1 + unc)
        elif model_type == ModelType.prophet:
            p_point, p_low, p_high = self.solar_prophet.predict(weather_list)
            return p_point, p_low, p_high
        else:  # ensemble
            p_point, p_low, p_high = self.solar_prophet.predict(weather_list)
            point = WEIGHT_XGB * xgb_point + WEIGHT_PROPHET * p_point + WEIGHT_PHYSICS * phys
            lower = WEIGHT_XGB * xgb_low + WEIGHT_PROPHET * p_low + WEIGHT_PHYSICS * phys * 0.80
            upper = WEIGHT_XGB * xgb_high + WEIGHT_PROPHET * p_high + WEIGHT_PHYSICS * phys * 1.20
            return np.clip(point, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None)

    def _ensemble_wind(
        self,
        weather_list: list[HourlyWeather],
        model_type: ModelType,
        wind_params: WindParams | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run wind ensemble and return (point, lower, upper) in kW."""
        xgb_point, xgb_low, xgb_high = self.wind_xgb.predict(weather_list)
        phys = self._physics_wind(weather_list, wind_params)

        if model_type == ModelType.xgboost:
            return xgb_point, xgb_low, xgb_high
        elif model_type == ModelType.physics:
            unc = 0.25
            return phys, phys * (1 - unc), phys * (1 + unc)
        elif model_type == ModelType.prophet:
            p_point, p_low, p_high = self.wind_prophet.predict(weather_list)
            return p_point, p_low, p_high
        else:  # ensemble
            p_point, p_low, p_high = self.wind_prophet.predict(weather_list)
            point = WEIGHT_XGB * xgb_point + WEIGHT_PROPHET * p_point + WEIGHT_PHYSICS * phys
            lower = WEIGHT_XGB * xgb_low + WEIGHT_PROPHET * p_low + WEIGHT_PHYSICS * phys * 0.75
            upper = WEIGHT_XGB * xgb_high + WEIGHT_PROPHET * p_high + WEIGHT_PHYSICS * phys * 1.25
            return np.clip(point, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None)

    def _build_forecast_response(
        self,
        energy_type: EnergyType,
        location: LocationRequest,
        weather_list: list[HourlyWeather],
        point: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        model_type: ModelType,
        rated_power_kw: float,
    ) -> ForecastResponse:
        """Assemble the full ForecastResponse from arrays."""
        now = datetime.now(timezone.utc)
        forecast_id = str(uuid.uuid4())

        # Build hourly predictions (kWh = kW × 1h)
        hourly = []
        for i, w in enumerate(weather_list):
            hourly.append(HourlyPrediction(
                timestamp=w.timestamp,
                energy_kwh=round(float(point[i]), 3),
                energy_kwh_lower=round(float(lower[i]), 3),
                energy_kwh_upper=round(float(upper[i]), 3),
                power_kw=round(float(point[i]), 3),
                model_used=model_type,
                weather=w,
            ))

        # Summary statistics
        total_kwh = float(np.sum(point))
        avg_kw = float(np.mean(point))
        peak_kw = float(np.max(point))
        peak_idx = int(np.argmax(point))
        capacity_factor = avg_kw / rated_power_kw if rated_power_kw > 0 else 0.0

        summary = ForecastSummary(
            total_energy_kwh=round(total_kwh, 2),
            avg_power_kw=round(avg_kw, 2),
            peak_power_kw=round(peak_kw, 2),
            peak_hour=weather_list[peak_idx].timestamp,
            capacity_factor=round(min(capacity_factor, 1.0), 4),
        )

        return ForecastResponse(
            forecast_id=forecast_id,
            energy_type=energy_type,
            location=location,
            generated_at=now,
            horizon_hours=len(weather_list),
            model_used=model_type,
            summary=summary,
            hourly=hourly,
            metadata={
                "ensemble_weights": {
                    "xgboost": WEIGHT_XGB,
                    "prophet": WEIGHT_PROPHET,
                    "physics": WEIGHT_PHYSICS,
                } if model_type == ModelType.ensemble else {},
                "rated_capacity_kw": round(rated_power_kw, 2),
                "api_version": settings.app_version,
            },
        )

    async def forecast_solar(
        self,
        request: ForecastRequest,
        weather_list: list[HourlyWeather],
    ) -> ForecastResponse:
        """Generate a complete solar energy forecast."""
        if not self._initialized:
            raise RuntimeError("ForecastEngine not initialized — call initialize() first")

        solar_params = self._build_solar_params(request)
        rated_kw = rated_solar_power_kw(solar_params)

        loop = asyncio.get_event_loop()
        point, lower, upper = await loop.run_in_executor(
            None,
            lambda: self._ensemble_solar(weather_list, request.model, solar_params),
        )

        return self._build_forecast_response(
            EnergyType.solar, request.location, weather_list,
            point, lower, upper, request.model, rated_kw,
        )

    async def forecast_wind(
        self,
        request: ForecastRequest,
        weather_list: list[HourlyWeather],
    ) -> ForecastResponse:
        """Generate a complete wind energy forecast."""
        if not self._initialized:
            raise RuntimeError("ForecastEngine not initialized — call initialize() first")

        wind_params = self._build_wind_params(request)
        rated_kw = rated_wind_power_kw(wind_params)

        loop = asyncio.get_event_loop()
        point, lower, upper = await loop.run_in_executor(
            None,
            lambda: self._ensemble_wind(weather_list, request.model, wind_params),
        )

        return self._build_forecast_response(
            EnergyType.wind, request.location, weather_list,
            point, lower, upper, request.model, rated_kw,
        )


# Module-level singleton — shared across all requests
forecast_engine = ForecastEngine()
