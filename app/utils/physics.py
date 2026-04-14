"""
Physics-based energy calculation utilities.

These formulas are used as a ground-truth baseline and to generate training
data for the ML models. All equations are derived from first principles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()


# ─── Solar Physics ───────────────────────────────────────────────────────────

@dataclass
class SolarParams:
    efficiency: float         # Panel efficiency (0–1)
    area_m2: float            # Total panel area (m²)
    temp_coefficient: float   # Power temperature coefficient (/°C)


def solar_power_kw(
    ghi_wm2: float,           # Global Horizontal Irradiance (W/m²)
    temperature_c: float,     # Ambient temperature (°C)
    cloud_cover_pct: float,   # Cloud cover (0–100)
    params: SolarParams | None = None,
) -> float:
    """
    Calculate solar PV power output using a standard temperature-corrected model.

    Formula:
        P = η × A × GHI_eff × [1 + γ(T - 25)]

    Where:
        η  = panel efficiency
        A  = panel area (m²)
        GHI_eff = effective GHI adjusted for cloud cover
        γ  = temperature coefficient (negative for crystalline silicon)
        T  = cell temperature ≈ ambient + 25°C (NOCT approximation)

    Args:
        ghi_wm2: Global horizontal irradiance (W/m²)
        temperature_c: Ambient air temperature (°C)
        cloud_cover_pct: Cloud cover percentage (0–100)
        params: Solar farm parameters (defaults to settings)

    Returns:
        Power output in kilowatts (kW)
    """
    if params is None:
        params = SolarParams(
            efficiency=settings.solar_panel_efficiency,
            area_m2=settings.solar_panel_area_m2,
            temp_coefficient=settings.solar_temp_coefficient,
        )

    # Cloud attenuation: clear sky is 100%, overcast ≈ 15% transmittance
    cloud_fraction = cloud_cover_pct / 100.0
    transmittance = 1.0 - 0.85 * cloud_fraction
    ghi_effective = max(0.0, ghi_wm2 * transmittance)

    # Cell temperature (simplified NOCT model)
    t_cell = temperature_c + 25.0  # NOCT correction

    # Temperature derating
    temp_factor = 1.0 + params.temp_coefficient * (t_cell - 25.0)
    temp_factor = max(0.0, temp_factor)

    power_w = params.efficiency * params.area_m2 * ghi_effective * temp_factor
    return power_w / 1000.0  # Convert W → kW


# ─── Wind Physics ────────────────────────────────────────────────────────────

@dataclass
class WindParams:
    radius_m: float          # Rotor radius (m)
    power_coefficient: float # Cp (max Betz limit = 0.593)
    cut_in_speed: float      # Wind speed below which turbine is off (m/s)
    rated_speed: float       # Wind speed at full rated power (m/s)
    cut_out_speed: float     # Wind speed above which turbine shuts down (m/s)


# Air density at STP (kg/m³) — corrected for altitude/pressure below
AIR_DENSITY_STP = 1.225


def air_density(pressure_hpa: float, temperature_c: float) -> float:
    """
    Calculate actual air density from temperature and pressure.
    ρ = P / (R_specific × T)
    R_specific for dry air = 287.05 J/(kg·K)
    """
    pressure_pa = pressure_hpa * 100.0
    temperature_k = temperature_c + 273.15
    return pressure_pa / (287.05 * temperature_k)


def wind_power_kw(
    wind_speed_ms: float,      # Wind speed at hub height (m/s)
    temperature_c: float = 15.0,
    pressure_hpa: float = 1013.25,
    params: WindParams | None = None,
) -> float:
    """
    Calculate wind turbine power output using the aerodynamic power equation.

    The wind power curve has three regions:
        1. Below cut-in speed: P = 0
        2. Between cut-in and rated: P = 0.5 × ρ × Cp × A × v³  (cubic law)
        3. Between rated and cut-out: P = P_rated  (constant)
        4. Above cut-out: P = 0  (emergency shutdown)

    Args:
        wind_speed_ms: Wind speed at hub height in m/s
        temperature_c: Air temperature (°C) for density correction
        pressure_hpa: Surface pressure (hPa) for density correction
        params: Wind turbine parameters (defaults to settings)

    Returns:
        Power output in kW
    """
    if params is None:
        params = WindParams(
            radius_m=settings.wind_turbine_radius_m,
            power_coefficient=settings.wind_power_coefficient,
            cut_in_speed=settings.wind_cut_in_speed,
            rated_speed=settings.wind_rated_speed,
            cut_out_speed=settings.wind_cut_out_speed,
        )

    v = max(0.0, wind_speed_ms)

    # Below cut-in or above cut-out → no power
    if v < params.cut_in_speed or v > params.cut_out_speed:
        return 0.0

    # Swept area A = π r²
    swept_area_m2 = math.pi * (params.radius_m ** 2)

    # Air density from ambient conditions
    rho = air_density(pressure_hpa, temperature_c)

    # Rated power at rated wind speed
    p_rated_kw = 0.5 * rho * params.power_coefficient * swept_area_m2 * (params.rated_speed ** 3) / 1000.0

    if v >= params.rated_speed:
        # Constant output at rated power
        return p_rated_kw
    else:
        # Cubic region
        power_w = 0.5 * rho * params.power_coefficient * swept_area_m2 * (v ** 3)
        return min(power_w / 1000.0, p_rated_kw)


def rated_solar_power_kw(params: SolarParams | None = None) -> float:
    """Return peak rated solar power (at STC: 1000 W/m², 25°C, no cloud)."""
    if params is None:
        params = SolarParams(
            efficiency=settings.solar_panel_efficiency,
            area_m2=settings.solar_panel_area_m2,
            temp_coefficient=settings.solar_temp_coefficient,
        )
    return params.efficiency * params.area_m2 * 1000.0 / 1000.0  # kW


def rated_wind_power_kw(params: WindParams | None = None) -> float:
    """Return rated wind turbine power (at rated wind speed, STP air density)."""
    if params is None:
        params = WindParams(
            radius_m=settings.wind_turbine_radius_m,
            power_coefficient=settings.wind_power_coefficient,
            cut_in_speed=settings.wind_cut_in_speed,
            rated_speed=settings.wind_rated_speed,
            cut_out_speed=settings.wind_cut_out_speed,
        )
    swept_area_m2 = math.pi * (params.radius_m ** 2)
    power_w = 0.5 * AIR_DENSITY_STP * params.power_coefficient * swept_area_m2 * (params.rated_speed ** 3)
    return power_w / 1000.0
