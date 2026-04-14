"""
XGBoost-based wind energy forecaster.

Same physics-informed approach as the solar model. The wind power curve
has a distinctive cubic relationship with wind speed, but XGBoost can
capture the cut-in / rated / cut-out non-linearities that piecewise
physics models sometimes miss under turbulent conditions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from app.config import get_settings
from app.models.schemas import HourlyWeather
from app.utils.feature_engineering import weather_to_dataframe, get_wind_feature_columns
from app.utils.physics import wind_power_kw, WindParams

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_PATH = Path("data/wind_xgb_model.joblib")
SCALER_PATH = Path("data/wind_xgb_scaler.joblib")


class WindXGBModel:
    """
    XGBoost wind forecaster with physics-informed synthetic training.

    Key difference from solar: wind is highly non-linear and location-specific.
    The power curve is shaped by:
        - Cut-in speed (below → 0 output)
        - Rated speed (above → constant rated output)
        - Cut-out speed (above → emergency shutdown, 0 output)

    Training noise includes turbulence intensity, wake effects, and
    mechanical losses that pure aerodynamic models don't capture.
    """

    XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    def __init__(self) -> None:
        self.model: xgb.XGBRegressor | None = None
        self.scaler: StandardScaler | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train_or_load(self) -> None:
        """Load a pre-trained model from disk or train a new one from scratch."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            logger.info("Loading pre-trained wind XGBoost model from disk...")
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self._is_trained = True
            return

        logger.info("No pre-trained wind model found — training from synthetic data...")
        self._train_from_synthetic_data()

    def _generate_synthetic_dataset(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Generate 8760 hours of synthetic wind data and physics-based labels.
        Wind speed follows a Weibull distribution (standard in wind energy).
        """
        logger.info("Generating synthetic wind training dataset (8760 hours)...")
        rng = np.random.default_rng(42)

        n_hours = 8760
        hours = np.arange(n_hours)
        doy = (hours // 24) + 1
        hour_of_day = hours % 24

        # ── Synthetic wind generation (Weibull shape k=2, scale varies) ──
        # Seasonal variation: stronger winds in winter/spring
        scale_seasonal = 7.0 + 3.0 * np.cos(2 * np.pi * (doy + 30) / 365)
        wind_speed = rng.weibull(2.0, n_hours) * scale_seasonal

        # Add diurnal pattern: peak afternoon for onshore wind
        diurnal = 1.5 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
        wind_speed = np.clip(wind_speed + diurnal, 0, 30)

        # Gusts and turbulence
        turbulence = rng.normal(0, 0.8, n_hours)
        wind_speed_turbulent = np.clip(wind_speed + turbulence, 0, 35)

        wind_direction = rng.uniform(0, 360, n_hours)

        # Temperature (for air density)
        temp_seasonal = 12 + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
        temperature = temp_seasonal + rng.normal(0, 3, n_hours)

        # Other variables
        humidity = 60 + 20 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 10, n_hours)
        humidity = np.clip(humidity, 10, 100)
        pressure = 1013 + rng.normal(0, 8, n_hours)
        ghi = np.zeros(n_hours)  # Not relevant for wind but needed for schema
        precipitation = np.where(rng.random(n_hours) > 0.85, rng.exponential(1, n_hours), 0)

        # ── Build HourlyWeather objects ────────────────────────────────────
        import pandas as pd
        timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")
        from app.models.schemas import HourlyWeather
        weather_list = []
        for i in range(n_hours):
            weather_list.append(HourlyWeather(
                timestamp=timestamps[i].to_pydatetime(),
                temperature_2m=float(temperature[i]),
                relative_humidity_2m=float(humidity[i]),
                cloud_cover=float(rng.uniform(0, 100)),
                wind_speed_10m=float(wind_speed_turbulent[i]),
                wind_direction_10m=float(wind_direction[i]),
                shortwave_radiation=0.0,
                direct_normal_irradiance=0.0,
                diffuse_radiation=0.0,
                precipitation=float(precipitation[i]),
                surface_pressure=float(pressure[i]),
            ))

        df = weather_to_dataframe(weather_list)

        # ── Physics labels ─────────────────────────────────────────────────
        labels = []
        params = WindParams(
            radius_m=settings.wind_turbine_radius_m,
            power_coefficient=settings.wind_power_coefficient,
            cut_in_speed=settings.wind_cut_in_speed,
            rated_speed=settings.wind_rated_speed,
            cut_out_speed=settings.wind_cut_out_speed,
        )
        for i in range(n_hours):
            p_kw = wind_power_kw(
                wind_speed_ms=float(wind_speed_turbulent[i]),
                temperature_c=float(temperature[i]),
                pressure_hpa=float(pressure[i]),
                params=params,
            )
            # Realistic losses
            wake_loss = rng.uniform(0.88, 1.0)            # 0–12% wake/array losses
            mechanical_loss = rng.uniform(0.96, 0.99)     # 1–4% mechanical losses
            availability = 1.0 if rng.random() > 0.02 else 0.0  # 2% downtime
            p_realistic = p_kw * wake_loss * mechanical_loss * availability
            labels.append(max(0.0, p_realistic))

        return df, np.array(labels)

    def _train_from_synthetic_data(self) -> None:
        df, labels = self._generate_synthetic_dataset()

        feature_cols = get_wind_feature_columns()
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].values
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Training XGBoost wind model on %d samples...", len(X_train))
        self.model = xgb.XGBRegressor(**self.XGB_PARAMS)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        y_pred = np.clip(self.model.predict(X_test_scaled), 0, None)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info("Wind XGBoost — MAE: %.2f kW | R²: %.4f", mae, r2)

        os.makedirs("data", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info("Wind XGBoost model saved to disk.")
        self._is_trained = True

    def predict(
        self,
        weather_list: list[HourlyWeather],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict wind power output for a list of weather observations.

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound) in kW.
            Wind uncertainty is higher than solar due to turbulence.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained/loaded before predicting")

        df = weather_to_dataframe(weather_list)
        feature_cols = get_wind_feature_columns()
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        point = np.clip(self.model.predict(X_scaled), 0, None)

        # Wind uncertainty: larger at high speeds (turbulence) and near cut-out
        wind_arr = df["wind_speed_10m"].values
        turbulence_uncertainty = 0.12 + 0.10 * np.clip((wind_arr / settings.wind_rated_speed), 0, 1)
        lower = np.clip(point * (1 - turbulence_uncertainty), 0, None)
        upper = point * (1 + turbulence_uncertainty)

        return point, lower, upper
