"""
Feature engineering: convert raw HourlyWeather data into ML-ready features.

All features are documented with their units and derivation so the code
is readable in a code review.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.models.schemas import HourlyWeather


def weather_to_dataframe(weather_list: list[HourlyWeather]) -> pd.DataFrame:
    """
    Convert a list of HourlyWeather objects into a pandas DataFrame
    with raw features plus derived engineered features.
    """
    rows = []
    for w in weather_list:
        rows.append({
            "timestamp": w.timestamp,
            "temperature_2m": w.temperature_2m,
            "relative_humidity_2m": w.relative_humidity_2m,
            "cloud_cover": w.cloud_cover,
            "wind_speed_10m": w.wind_speed_10m,
            "wind_direction_10m": w.wind_direction_10m,
            "shortwave_radiation": w.shortwave_radiation,
            "direct_normal_irradiance": w.direct_normal_irradiance,
            "diffuse_radiation": w.diffuse_radiation,
            "precipitation": w.precipitation,
            "surface_pressure": w.surface_pressure,
        })

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")
    df = _add_temporal_features(df)
    df = _add_solar_features(df)
    df = _add_wind_features(df)
    df = _add_interaction_features(df)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encoding of hour, day-of-year, month."""
    idx = df.index

    # Hour of day: encode as sin/cos so 23:00 and 00:00 are close
    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hour / 24)

    # Day of year (seasonality)
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

    # Month (coarser seasonality)
    month = idx.month
    df["month_sin"] = np.sin(2 * math.pi * month / 12)
    df["month_cos"] = np.cos(2 * math.pi * month / 12)

    # Is daytime (rough: hour 6–20)
    df["is_daytime"] = ((idx.hour >= 6) & (idx.hour < 20)).astype(float)

    return df


def _add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features specific to solar generation."""
    # Cloud transmittance (clear = 1, overcast = ~0.15)
    df["cloud_transmittance"] = 1.0 - 0.85 * (df["cloud_cover"] / 100.0)

    # Effective GHI after cloud attenuation
    df["ghi_effective"] = df["shortwave_radiation"] * df["cloud_transmittance"]

    # Solar cell temperature (NOCT model approximation)
    df["cell_temperature"] = df["temperature_2m"] + 25.0

    # Temperature derating factor (for standard -0.4%/°C coefficient)
    df["temp_derating"] = 1.0 + (-0.004) * (df["cell_temperature"] - 25.0)
    df["temp_derating"] = df["temp_derating"].clip(lower=0.0)

    # DNI clearness index (ratio of DNI to GHI)
    df["clearness_index"] = np.where(
        df["shortwave_radiation"] > 10,
        df["direct_normal_irradiance"] / df["shortwave_radiation"].clip(lower=1),
        0.0,
    )
    return df


def _add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features specific to wind generation."""
    v = df["wind_speed_10m"]

    # Wind power density (W/m²) — proportional to v³ × rho
    # Use constant air density for simplicity here
    df["wind_power_density"] = 0.5 * 1.225 * (v ** 3)

    # Wind speed squared (often useful for linear models)
    df["wind_speed_sq"] = v ** 2

    # Wind speed cubed (physic relationship)
    df["wind_speed_cubed"] = v ** 3

    # Wind components (north-south, east-west)
    dir_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -v * np.sin(dir_rad)   # zonal (east+)
    df["wind_v"] = -v * np.cos(dir_rad)   # meridional (north+)

    # Hub-height wind speed extrapolation (power law, α = 1/7)
    # v_hub = v_10m × (h_hub / 10)^(1/7), with h_hub = 80m
    df["wind_speed_80m"] = v * ((80.0 / 10.0) ** (1.0 / 7.0))

    # Air density from pressure and temperature
    df["air_density"] = (df["surface_pressure"] * 100.0) / (287.05 * (df["temperature_2m"] + 273.15))

    # Density-corrected wind power
    df["wind_power_density_corrected"] = 0.5 * df["air_density"] * (df["wind_speed_80m"] ** 3)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-variable interaction terms."""
    # Temperature × cloud cover (hot cloudy days vs cold clear days)
    df["temp_x_cloud"] = df["temperature_2m"] * df["cloud_cover"]

    # Solar radiation × temperature (PV output interaction)
    df["ghi_x_temp"] = df["ghi_effective"] * df["temp_derating"]

    # Humidity effect on air density (reduces wind power slightly)
    df["humidity_factor"] = 1.0 - 0.001 * df["relative_humidity_2m"]

    return df


def get_solar_feature_columns() -> list[str]:
    """Return the ordered list of features used for solar ML model."""
    return [
        "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
        "month_sin", "month_cos",
        "is_daytime",
        "temperature_2m",
        "cloud_cover",
        "cloud_transmittance",
        "ghi_effective",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cell_temperature",
        "temp_derating",
        "clearness_index",
        "temp_x_cloud",
        "ghi_x_temp",
    ]


def get_wind_feature_columns() -> list[str]:
    """Return the ordered list of features used for wind ML model."""
    return [
        "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_sq",
        "wind_speed_cubed",
        "wind_speed_80m",
        "wind_u",
        "wind_v",
        "wind_power_density",
        "wind_power_density_corrected",
        "air_density",
        "surface_pressure",
        "humidity_factor",
        "precipitation",
    ]
