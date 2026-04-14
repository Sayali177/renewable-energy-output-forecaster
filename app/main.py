"""
Renewable Energy Output Forecaster — FastAPI Application Entry Point

Architecture Overview:
    - FastAPI with async/await throughout (no blocking I/O on the event loop)
    - SQLAlchemy async (aiosqlite) for persistence
    - httpx async for weather API calls
    - XGBoost + Prophet ML ensemble (trained in thread pool on startup)
    - Open-Meteo weather API (free, no key needed)

Startup sequence:
    1. Create SQLite tables (idempotent)
    2. Train/load ML models (XGBoost solar, XGBoost wind)
    3. Fit Prophet models on synthetic data
    4. Ready to serve requests
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.database import init_db
from app.routers import forecast, health, history, weather as weather_router
from app.services.forecast_engine import forecast_engine

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Code before `yield` runs on startup; after `yield` runs on shutdown.
    """
    logger.info("=" * 60)
    logger.info("  %s v%s — Starting up", settings.app_name, settings.app_version)
    logger.info("=" * 60)

    # 1. Initialize database tables
    logger.info("Step 1/2: Initializing database...")
    await init_db()
    logger.info("Database initialized.")

    # 2. Initialize ML models (trains or loads from disk)
    logger.info("Step 2/2: Initializing ML forecast engine...")
    logger.info("  This may take 30–120 seconds on first run (training).")
    logger.info("  Subsequent runs load pre-trained models instantly.")
    t0 = time.time()
    await forecast_engine.initialize()
    elapsed = time.time() - t0
    logger.info("Forecast engine ready in %.1f seconds.", elapsed)

    logger.info("=" * 60)
    logger.info("  API is ready. Visit http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # Server is running

    logger.info("Shutting down %s...", settings.app_name)


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Renewable Energy Output Forecaster

A production-grade API that predicts how much energy a **solar farm** or **wind turbine**
will generate over the next 72 hours, using real weather forecast data.

### How It Works

1. You provide a geographic location (latitude/longitude).
2. We fetch live 72-hour weather forecasts from **Open-Meteo** (free, no API key needed).
3. We run your weather data through a **weighted ML ensemble**:
   - **XGBoost** (55%) — physics-informed gradient boosted trees
   - **Prophet** (30%) — Meta's time-series model capturing daily/yearly seasonality
   - **Physics Model** (15%) — first-principles PV/aerodynamic equations
4. We return hourly energy predictions (kWh) with **90% confidence intervals**.

### Green Tech Techniques Used

| Technique | Application |
|-----------|-------------|
| XGBoost | Primary regression model for both solar & wind |
| Prophet | Temporal seasonality decomposition |
| Physics-informed ML | Training data generated from PV/Betz equations |
| Feature Engineering | Cyclic temporal encoding, cloud transmittance, air density correction |
| Confidence Intervals | Uncertainty propagation from cloud cover and wind turbulence |

### Quick Start

```bash
# Solar forecast for Mumbai, India
curl -X POST /api/forecast/solar \\
  -H "Content-Type: application/json" \\
  -d '{"location": {"latitude": 19.0760, "longitude": 72.8777, "name": "Mumbai Solar Farm"}, "horizon_hours": 72, "model": "ensemble"}'
```
""",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Log request timing for every incoming request."""
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info("%-6s %-40s → %d  (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.0f}"
    return response


# ─── Global Exception Handlers ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "path": str(request.url.path),
        },
    )


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(weather_router.router)


# ─── Root Redirect ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "solar_forecast": "POST /api/forecast/solar",
            "wind_forecast": "POST /api/forecast/wind",
            "combined_forecast": "POST /api/forecast/combined",
            "latest_forecast": "GET /api/forecast/latest",
            "history": "GET /api/history",
            "current_weather": "GET /api/weather/current",
        },
    }
