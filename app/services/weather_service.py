"""
Open-Meteo weather API client.

Open-Meteo is completely free, no API key needed, and provides:
- Historical weather data (ERA5)
- 72-hour+ hourly forecasts
- Variables: temperature, cloud cover, wind speed, solar radiation, etc.

Docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.models.schemas import HourlyWeather

settings = get_settings()

# Variables we request from Open-Meteo
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "precipitation",
    "surface_pressure",
]


class WeatherServiceError(Exception):
    """Raised when the weather API returns an unexpected response."""
    pass


class WeatherService:
    """
    Async HTTP client for the Open-Meteo Forecast API.
    Fetches hourly weather data for a given lat/lon.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float = 30.0) -> None:
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> "WeatherService":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 72,
    ) -> list[HourlyWeather]:
        """
        Fetch hourly weather forecast from Open-Meteo.

        Args:
            latitude: Decimal degrees (-90 to 90)
            longitude: Decimal degrees (-180 to 180)
            hours: Number of hours to forecast (default 72)

        Returns:
            List of HourlyWeather objects, one per hour.

        Raises:
            WeatherServiceError: If the API call fails or data is malformed.
        """
        if self._client is None:
            raise WeatherServiceError("WeatherService must be used as an async context manager")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": max(3, (hours // 24) + 1),
            "timezone": "UTC",
            "wind_speed_unit": "ms",          # we want m/s not km/h
        }

        try:
            response = await self._client.get(self.BASE_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise WeatherServiceError(
                f"Open-Meteo API returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise WeatherServiceError(f"Network error calling Open-Meteo: {e}") from e

        data = response.json()
        return self._parse_response(data, hours)

    def _parse_response(self, data: dict, hours: int) -> list[HourlyWeather]:
        """Parse the Open-Meteo JSON response into typed HourlyWeather objects."""
        try:
            hourly = data["hourly"]
            times = hourly["time"]
        except KeyError as e:
            raise WeatherServiceError(f"Unexpected API response structure: missing {e}") from e

        results: list[HourlyWeather] = []

        for i, ts_str in enumerate(times[:hours]):
            # Open-Meteo returns ISO 8601 strings like "2024-01-01T00:00"
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)

            def safe_get(key: str, idx: int = i) -> float:
                val = hourly.get(key, [None])
                if idx < len(val) and val[idx] is not None:
                    return float(val[idx])
                return 0.0

            results.append(
                HourlyWeather(
                    timestamp=ts,
                    temperature_2m=safe_get("temperature_2m"),
                    relative_humidity_2m=safe_get("relative_humidity_2m"),
                    cloud_cover=safe_get("cloud_cover"),
                    wind_speed_10m=safe_get("wind_speed_10m"),
                    wind_direction_10m=safe_get("wind_direction_10m"),
                    shortwave_radiation=safe_get("shortwave_radiation"),
                    direct_normal_irradiance=safe_get("direct_normal_irradiance"),
                    diffuse_radiation=safe_get("diffuse_radiation"),
                    precipitation=safe_get("precipitation"),
                    surface_pressure=safe_get("surface_pressure"),
                )
            )

        return results

    async def get_current_conditions(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        city: str | None = None,
    ) -> dict[str, Any]:
        """
        Fetch current hour's weather conditions using Open-Meteo API.
        Can be queried either by city name or by latitude/longitude.
        """
        if self._client is None:
            raise WeatherServiceError("WeatherService must be used as an async context manager")

        loc_name = "Unknown"
        if city:
            try:
                geo_resp = await self._client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "format": "json"}
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()
                if "results" not in geo_data or not geo_data["results"]:
                    raise WeatherServiceError(f"City '{city}' not found.")
                result = geo_data["results"][0]
                latitude = result["latitude"]
                longitude = result["longitude"]
                loc_name = result.get("name", city)
            except Exception as e:
                raise WeatherServiceError(f"Geocoding API request failed: {e}") from e
        elif latitude is not None and longitude is not None:
            loc_name = f"Grid {latitude:.2f}, {longitude:.2f}"
        else:
            raise WeatherServiceError("Must provide either 'city' or both 'latitude' and 'longitude'")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,wind_direction_10m,precipitation,shortwave_radiation",
            "wind_speed_unit": "ms",
            "timezone": "GMT"
        }

        try:
            response = await self._client.get("https://api.open-meteo.com/v1/forecast", params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise WeatherServiceError(f"Open-Meteo API request failed: {e}") from e

        try:
            current = data["current"]

            return {
                "timestamp": current.get("time", datetime.utcnow().isoformat()),
                "temperature_c": float(current.get("temperature_2m", 0)),
                "cloud_cover_pct": float(current.get("cloud_cover", 0)),
                "wind_speed_ms": float(current.get("wind_speed_10m", 0)),
                "wind_direction_deg": float(current.get("wind_direction_10m", 0)),
                "solar_radiation_wm2": float(current.get("shortwave_radiation", 0)),
                "precipitation_mm": float(current.get("precipitation", 0)),
                "humidity_pct": float(current.get("relative_humidity_2m", 0)),
                "resolved_location": {
                    "name": loc_name,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                }
            }
        except KeyError as e:
            raise WeatherServiceError(f"Unexpected Open-Meteo response structure: {e}") from e
