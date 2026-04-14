"""
Application configuration — loaded from .env file.
Pydantic Settings automatically reads environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Renewable Energy Forecaster API"
    app_version: str = "1.0.0"
    debug: bool = True

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/energy.db"

    # Weather
    weather_api_base: str = "https://api.open-meteo.com/v1"
    weatherstack_api_key: str = "5a92b56d1fc0de82b53028f8eb4b23c7"

    # Default location (Mumbai, India)
    default_latitude: float = 19.0760
    default_longitude: float = 72.8777

    # Solar farm config
    solar_panel_efficiency: float = 0.20        # 20% efficiency
    solar_panel_area_m2: float = 10_000.0       # 10,000 m² (1 hectare farm)
    solar_temp_coefficient: float = -0.004      # -0.4%/°C above 25°C

    # Wind turbine config
    wind_turbine_radius_m: float = 40.0         # 40m blade radius
    wind_power_coefficient: float = 0.45        # Cp (Betz limit ≈ 0.593)
    wind_cut_in_speed: float = 3.0              # m/s — below this: no power
    wind_rated_speed: float = 12.0             # m/s — full rated power
    wind_cut_out_speed: float = 25.0           # m/s — safety shutdown


@lru_cache()
def get_settings() -> Settings:
    return Settings()
