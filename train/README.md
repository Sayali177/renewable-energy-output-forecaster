# Renewable Energy Output Forecaster — Backend API

A production-grade FastAPI backend that predicts how much energy a **solar farm** or **wind turbine** will generate over the next 72 hours using real weather forecast data and ML models.

---

## 🌱 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI (async) |
| ML Models | XGBoost + Prophet (Meta) |
| Physics Models | PV equation (solar), Betz limit (wind) |
| Weather API | Open-Meteo (**free, no API key needed**) |
| Database | SQLite + SQLAlchemy (async) |
| HTTP Client | httpx (async) |
| Serialization | Pydantic v2 |

---

## 🚀 Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env

# 4. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **First run:** The server will train XGBoost and Prophet models on synthetic data. This takes **30–120 seconds**. Subsequent runs load pre-trained models instantly.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root — lists all endpoints |
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/forecast/solar` | 72-hour solar energy forecast |
| `POST` | `/api/forecast/wind` | 72-hour wind energy forecast |
| `POST` | `/api/forecast/combined` | Both solar + wind together |
| `GET` | `/api/forecast/latest` | Most recent stored forecast |
| `GET` | `/api/history` | Paginated forecast history |
| `GET` | `/api/history/{id}` | Single forecast by ID |
| `DELETE` | `/api/history/{id}` | Delete a forecast |
| `GET` | `/api/weather/current` | Current weather at location |
| `GET` | `/api/weather/forecast` | Raw weather forecast data |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

---

## 🔬 ML Architecture

### Ensemble Model (default)

```
Weather Data (72h)
      │
      ├──► XGBoost ──────── 55% weight ─┐
      │    (physics-informed)            │
      │                                  ├──► Weighted Ensemble ──► 72h kWh Forecast
      ├──► Prophet ──────── 30% weight ─┤    + Confidence Intervals
      │    (seasonality)                 │
      │                                  │
      └──► Physics Model ── 15% weight ─┘
           (first principles)
```

### Solar Physics Model
```
P_solar = η × A × GHI_eff × [1 + γ(T_cell - 25)]

Where:
  η        = panel efficiency (default: 20%)
  A        = panel area (default: 10,000 m²)
  GHI_eff  = GHI × (1 - 0.85 × cloud_fraction)
  γ        = temperature coefficient (-0.4%/°C)
  T_cell   = T_ambient + 25°C (NOCT approximation)
```

### Wind Physics Model
```
P_wind = 0.5 × ρ × Cp × A × v³    (cubic region)

Where:
  ρ   = air density (corrected for T and P)
  Cp  = power coefficient (default: 0.45, Betz max = 0.593)
  A   = π × r²  (swept area with r = 40m)
  v   = hub-height wind speed (extrapolated from 10m via power law)

Power curve regions:
  v < 3 m/s    → P = 0  (below cut-in)
  3–12 m/s     → cubic law
  12–25 m/s    → P = P_rated  (constant)
  v > 25 m/s   → P = 0  (safety shutdown)
```

---

## 📊 Example Request

```bash
# Solar farm in Berlin, Germany — 72h ensemble forecast
curl -X POST http://localhost:8000/api/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "latitude": 52.5200,
      "longitude": 13.4050,
      "name": "Berlin Solar Farm"
    },
    "horizon_hours": 72,
    "model": "ensemble"
  }'
```

```json
{
  "forecast_id": "a1b2c3d4-...",
  "energy_type": "solar",
  "location": {"latitude": 52.52, "longitude": 13.405, "name": "Berlin Solar Farm"},
  "generated_at": "2024-01-15T10:30:00Z",
  "horizon_hours": 72,
  "model_used": "ensemble",
  "summary": {
    "total_energy_kwh": 12540.3,
    "avg_power_kw": 174.2,
    "peak_power_kw": 1843.5,
    "peak_hour": "2024-01-16T11:00:00Z",
    "capacity_factor": 0.087
  },
  "hourly": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "energy_kwh": 210.5,
      "energy_kwh_lower": 178.9,
      "energy_kwh_upper": 242.1,
      "power_kw": 210.5,
      "model_used": "ensemble",
      "weather": { ... }
    }
  ]
}
```

---

## ⚙️ Configuration

Edit `.env` to customize farm parameters:

```env
# Solar Farm
SOLAR_PANEL_EFFICIENCY=0.22       # 22% efficiency panels
SOLAR_PANEL_AREA_M2=50000        # 5 hectare farm

# Wind Turbine  
WIND_TURBINE_RADIUS_M=63          # 63m radius (5 MW class turbine)
WIND_POWER_COEFFICIENT=0.48
WIND_RATED_SPEED=14.0
```

Or override per-request:
```json
{
  "location": {"latitude": 51.5, "longitude": -0.1},
  "panel_efficiency": 0.22,
  "panel_area_m2": 50000
}
```

---

## 🧪 Run Tests

```bash
python test_api.py
```

---

## 📁 Project Structure

```
app/
├── main.py                 # FastAPI app, middleware, lifespan
├── config.py               # Pydantic Settings
├── models/
│   ├── schemas.py          # Pydantic request/response models
│   └── database.py         # SQLAlchemy ORM + async engine
├── services/
│   ├── weather_service.py  # Open-Meteo API client
│   ├── solar_model.py      # XGBoost solar forecaster
│   ├── wind_model.py       # XGBoost wind forecaster
│   ├── prophet_model.py    # Prophet time-series model
│   └── forecast_engine.py  # Ensemble orchestrator
├── routers/
│   ├── forecast.py         # /api/forecast/* endpoints
│   ├── history.py          # /api/history/* endpoints
│   ├── health.py           # /api/health endpoint
│   └── weather.py          # /api/weather/* endpoints
└── utils/
    ├── physics.py          # Solar PV + wind aerodynamic equations
    └── feature_engineering.py  # Weather → ML feature pipeline
data/
└── energy.db               # SQLite database (auto-created)
```
