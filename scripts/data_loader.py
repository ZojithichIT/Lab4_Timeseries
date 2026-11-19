"""
Data loading, UTC alignment, outlier cleaning, and EDA diagnostics module
for Lab 4 Multivariate Time Series Forecasting (Electricity Load + Weather).
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd

from .config import Config


class MultivariateDataLoader:
    """Loads, merges, cleans, and audits hourly Spain electricity load and weather data."""

    def __init__(
        self,
        energy_path: Union[str, Path] = Config.RAW_ENERGY_PATH,
        weather_path: Union[str, Path] = Config.RAW_WEATHER_PATH,
    ):
        self.energy_path = Path(energy_path)
        self.weather_path = Path(weather_path)

    def load_clean_multivariate(
        self, export_path: Union[str, Path, None] = Config.CLEAN_DATA_PATH
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge hourly electricity load and multi-city weather features, handle outliers
        and missing values via time interpolation, and return (clean_df, data_audit).
        """
        if not self.energy_path.exists():
            raise FileNotFoundError(f"Energy dataset not found: {self.energy_path}")
        if not self.weather_path.exists():
            raise FileNotFoundError(f"Weather dataset not found: {self.weather_path}")

        df_energy = pd.read_csv(self.energy_path)
        df_weather = pd.read_csv(self.weather_path)

        df_energy["dt"] = pd.to_datetime(df_energy["time"], utc=True)
        df_weather["dt"] = pd.to_datetime(df_weather["dt_iso"], utc=True)
        df_weather["city_name"] = df_weather["city_name"].astype(str).str.strip()

        raw_wind_outliers = int((df_weather["wind_speed"] > Config.WIND_SPEED_MAX_MS).sum())
        df_weather["temperature"] = df_weather["temp"] - 273.15
        df_weather.loc[df_weather["wind_speed"] > Config.WIND_SPEED_MAX_MS, "wind_speed"] = np.nan

        energy_hourly = df_energy.groupby("dt")["total load actual"].mean().to_frame(name="load")
        weather_hourly = df_weather.groupby("dt")[["temperature", "humidity", "wind_speed"]].mean()

        df_merged = energy_hourly.join(weather_hourly, how="inner").sort_index().asfreq("h")
        raw_nan_counts = {col: int(df_merged[col].isna().sum()) for col in Config.FEATURES}

        # IQR outlier counts on merged hourly series (diagnostic)
        iqr_outliers: Dict[str, int] = {}
        for col in Config.FEATURES:
            q1 = df_merged[col].quantile(0.25)
            q3 = df_merged[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df_merged[col] < q1 - 1.5 * iqr) | (df_merged[col] > q3 + 1.5 * iqr)
            iqr_outliers[col] = int(mask.sum())

        # Physical wind outlier check on hourly level + time-based interpolation
        hourly_wind_outliers = int((df_merged["wind_speed"] > Config.WIND_SPEED_MAX_MS).sum())
        df_merged.loc[df_merged["wind_speed"] > Config.WIND_SPEED_MAX_MS, "wind_speed"] = np.nan

        df_clean = df_merged[Config.FEATURES].interpolate(method="time").bfill().ffill()
        clean_nan_counts = {col: int(df_clean[col].isna().sum()) for col in Config.FEATURES}

        if export_path is not None:
            out_file = Path(export_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(out_file)

        stats_table: Dict[str, Dict[str, float]] = {}
        for col in Config.FEATURES:
            s = df_clean[col]
            stats_table[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }

        audit: Dict[str, Any] = {
            "raw_energy_rows": int(len(df_energy)),
            "raw_weather_rows": int(len(df_weather)),
            "cities": sorted(df_weather["city_name"].unique().tolist()),
            "clean_rows": int(len(df_clean)),
            "start_time": str(df_clean.index.min()),
            "end_time": str(df_clean.index.max()),
            "raw_nan_counts": raw_nan_counts,
            "clean_nan_counts": clean_nan_counts,
            "raw_wind_outliers_gt40ms": raw_wind_outliers,
            "hourly_wind_outliers_gt40ms": hourly_wind_outliers,
            "iqr_outlier_counts": iqr_outliers,
            "summary_stats": stats_table,
            "pearson_corr": df_clean.corr(method="pearson").round(4).to_dict(),
            "spearman_corr": df_clean.corr(method="spearman").round(4).to_dict(),
            "hourly_profile": df_clean.groupby(df_clean.index.hour)["load"].mean().round(2).to_dict(),
            "weekday_profile": df_clean.groupby(df_clean.index.dayofweek)["load"].mean().round(2).to_dict(),
        }
        return df_clean, audit
