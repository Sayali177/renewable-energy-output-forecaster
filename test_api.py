"""
Integration test script for the Renewable Energy Output Forecaster API.

Run this AFTER starting the server with:
    uvicorn app.main:app --reload

This script hits all major endpoints and prints results.
"""

import json
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"

# Test location: Mumbai, India
MUMBAI = {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "name": "Mumbai Test Farm",
}

FORECAST_REQUEST = {
    "location": MUMBAI,
    "horizon_hours": 24,  # Use 24h for faster test
    "model": "ensemble",
}


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = "") -> None:
    icon = "✅" if condition else "❌"
    print(f"  {icon} {label}", f"({detail})" if detail else "")
    if not condition:
        sys.exit(1)


def main() -> None:
    print("\n🌱 Renewable Energy Forecaster — API Test Suite")
    print(f"   Target: {BASE_URL}")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:

        # ── Health Check ──────────────────────────────────────────────────
        print_section("1. Health Check")
        r = client.get("/api/health")
        check("Status 200", r.status_code == 200, f"got {r.status_code}")
        data = r.json()
        check("Status is healthy", data["status"] == "healthy")
        check("Version present", "version" in data)
        check("Services reported", "services" in data)
        print(f"   Services: {json.dumps(data['services'], indent=4)}")

        # ── Root Endpoint ─────────────────────────────────────────────────
        print_section("2. Root Endpoint")
        r = client.get("/")
        check("Status 200", r.status_code == 200, f"got {r.status_code}")
        data = r.json()
        check("Endpoints listed", "endpoints" in data)

        # ── Weather Endpoint ──────────────────────────────────────────────
        print_section("3. Weather — Current Conditions")
        r = client.get("/api/weather/current", params={"latitude": 19.076, "longitude": 72.877})
        check("Status 200", r.status_code == 200, f"got {r.status_code}")
        data = r.json()
        check("Has temperature", "temperature_c" in data)
        check("Has wind speed", "wind_speed_ms" in data)
        print(f"   Temperature: {data['temperature_c']:.1f}°C")
        print(f"   Wind Speed:  {data['wind_speed_ms']:.1f} m/s")
        print(f"   Cloud Cover: {data['cloud_cover_pct']:.0f}%")

        # ── Solar Forecast ────────────────────────────────────────────────
        print_section("4. Solar Energy Forecast (24h Ensemble)")
        t0 = time.time()
        r = client.post("/api/forecast/solar", json=FORECAST_REQUEST)
        elapsed = time.time() - t0
        check("Status 200", r.status_code == 200, f"got {r.status_code}\n{r.text[:500]}")
        data = r.json()
        check("Has forecast_id", "forecast_id" in data)
        check("Energy type is solar", data["energy_type"] == "solar")
        check(f"24 hourly predictions", len(data["hourly"]) >= 20)
        check("Has summary", "summary" in data)
        check("Hourly has confidence interval", "energy_kwh_lower" in data["hourly"][0])
        
        summary = data["summary"]
        print(f"   Total Energy:    {summary['total_energy_kwh']:.1f} kWh")
        print(f"   Avg Power:       {summary['avg_power_kw']:.1f} kW")
        print(f"   Peak Power:      {summary['peak_power_kw']:.1f} kW")
        print(f"   Capacity Factor: {summary['capacity_factor']:.1%}")
        print(f"   Response time:   {elapsed:.1f}s")
        
        solar_forecast_id = data["forecast_id"]

        # ── Wind Forecast ─────────────────────────────────────────────────
        print_section("5. Wind Energy Forecast (24h Ensemble)")
        t0 = time.time()
        r = client.post("/api/forecast/wind", json=FORECAST_REQUEST)
        elapsed = time.time() - t0
        check("Status 200", r.status_code == 200, f"got {r.status_code}\n{r.text[:500]}")
        data = r.json()
        check("Energy type is wind", data["energy_type"] == "wind")
        check("Has hourly data", len(data["hourly"]) >= 20)
        
        summary = data["summary"]
        print(f"   Total Energy:    {summary['total_energy_kwh']:.1f} kWh")
        print(f"   Avg Power:       {summary['avg_power_kw']:.1f} kW")
        print(f"   Peak Power:      {summary['peak_power_kw']:.1f} kW")
        print(f"   Response time:   {elapsed:.1f}s")

        # ── Combined Forecast ─────────────────────────────────────────────
        print_section("6. Combined Solar + Wind Forecast")
        r = client.post("/api/forecast/combined", json=FORECAST_REQUEST)
        check("Status 200", r.status_code == 200, f"got {r.status_code}\n{r.text[:500]}")
        data = r.json()
        check("Has solar sub-forecast", "solar" in data)
        check("Has wind sub-forecast", "wind" in data)
        check("Has combined summary", "combined_summary" in data)
        check("Has hourly combined", "hourly_combined" in data)
        check("Hourly combined has total_kw", "total_kw" in data["hourly_combined"][0])
        print(f"   Combined Total: {data['combined_summary']['total_energy_kwh']:.1f} kWh")

        # ── History ───────────────────────────────────────────────────────
        print_section("7. History Endpoints")
        r = client.get("/api/history", params={"energy_type": "solar", "limit": 5})
        check("Status 200", r.status_code == 200, f"got {r.status_code}")
        records = r.json()
        check("Has stored records", len(records) >= 1, f"{len(records)} records")
        print(f"   Stored solar records: {len(records)}")

        # ── Retrieve specific forecast ────────────────────────────────────
        r = client.get(f"/api/history/{solar_forecast_id}")
        check("Status 200 for specific forecast", r.status_code == 200, f"got {r.status_code}")
        check("Correct forecast_id returned", r.json().get("forecast_id") == solar_forecast_id)

        # ── Latest forecast ────────────────────────────────────────────────
        r = client.get("/api/forecast/latest", params={"energy_type": "solar"})
        check("Status 200 for latest", r.status_code == 200, f"got {r.status_code}")

        # ── Model variants ────────────────────────────────────────────────
        print_section("8. Model Variants (XGBoost / Prophet / Physics)")
        for model in ["xgboost", "prophet", "physics"]:
            req = {**FORECAST_REQUEST, "model": model, "horizon_hours": 6}
            r = client.post("/api/forecast/solar", json=req)
            check(f"Model '{model}' returns 200", r.status_code == 200, f"got {r.status_code}")
            data = r.json()
            check(f"Model '{model}' correct tag", data["model_used"] == model)
            peak = data["summary"]["peak_power_kw"]
            print(f"   {model:10s}: peak = {peak:.1f} kW")

    print("\n" + "="*60)
    print("  ✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
