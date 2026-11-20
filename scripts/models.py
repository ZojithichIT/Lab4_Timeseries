"""
Model wrappers and Walk-Forward Validation for Lab 4 Multivariate Time Series Forecasting:
- VARForecaster (Statistical Baseline with ADF test & AIC/BIC lag selection)
- XGBoostMultivariateForecaster (Machine Learning with Early Stopping Tuning)
- LSTMForecaster (Deep Learning: LSTM(64) -> LSTM(32) -> Dense(1))
- Walk-Forward Validation (Full Test + Periodic-Refit Backtest Simulation)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
import xgboost as xgb

from .config import Config
from .evaluation import evaluate_multivariate_forecast
from .feature_engineering import MultivariateFeatureBuilder


class VARForecaster:
    """Vector Autoregression (VAR) forecaster with ADF stationarity check and AIC/BIC order selection."""

    def __init__(self, maxlags: int = Config.VAR_MAXLAGS):
        self.maxlags = maxlags
        self.diff_order: int = 0
        self.selected_lag: int = 1
        self.results: Any = None
        self.feature_names_: List[str] = list(Config.FEATURES)
        self.diagnostics_: Dict[str, Any] = {}

    @staticmethod
    def run_adf_tests(df_train_scaled: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Run Augmented Dickey-Fuller test on level and first-difference series."""
        adf_summary: Dict[str, Dict[str, Any]] = {}
        for col in df_train_scaled.columns:
            ser = df_train_scaled[col].dropna()
            lvl_res = adfuller(ser.values, autolag="AIC")
            diff_ser = ser.diff().dropna()
            diff_res = adfuller(diff_ser.values, autolag="AIC")
            adf_summary[col] = {
                "level_stat": float(lvl_res[0]),
                "level_pvalue": float(lvl_res[1]),
                "level_stationary": bool(lvl_res[1] < 0.05),
                "diff1_stat": float(diff_res[0]),
                "diff1_pvalue": float(diff_res[1]),
                "diff1_stationary": bool(diff_res[1] < 0.05),
            }
        return adf_summary

    def fit(self, df_train_scaled: pd.DataFrame) -> "VARForecaster":
        """Check stationarity, select optimal lag p via AIC/BIC, and fit VAR."""
        self.feature_names_ = list(df_train_scaled.columns)
        adf_results = self.run_adf_tests(df_train_scaled)

        all_level_stationary = all(v["level_stationary"] for v in adf_results.values())
        self.diff_order = 0 if all_level_stationary else 1

        train_data = (
            df_train_scaled if self.diff_order == 0 else df_train_scaled.diff().dropna()
        )
        # Ensure pure numeric numpy matrix with clean RangeIndex for statsmodels stability
        train_df = pd.DataFrame(train_data.values, columns=self.feature_names_)
        model = VAR(train_df)
        order_sel = model.select_order(maxlags=self.maxlags)

        # statsmodels stores order selection ICS dictionary with keys 'aic', 'bic', etc. (1..maxlags or 0..maxlags)
        ics = order_sel.ics
        lag_grid = list(range(1, self.maxlags + 1))
        aic_vals = [float(ics["aic"][k]) for k in lag_grid]
        bic_vals = [float(ics["bic"][k]) for k in lag_grid]

        best_aic_lag = int(lag_grid[int(np.argmin(aic_vals))])
        best_bic_lag = int(lag_grid[int(np.argmin(bic_vals))])
        self.selected_lag = best_aic_lag

        self.results = model.fit(self.selected_lag)

        granger_pvals: Dict[str, float] = {}
        for col in self.feature_names_:
            if col == Config.TARGET_COL:
                continue
            try:
                gc = self.results.test_causality(Config.TARGET_COL, [col], kind="f")
                granger_pvals[col] = float(gc.pvalue)
            except Exception:
                granger_pvals[col] = 1.0

        self.diagnostics_ = {
            "diff_order": self.diff_order,
            "adf_results": adf_results,
            "lag_grid": lag_grid,
            "aic_values": aic_vals,
            "bic_values": bic_vals,
            "selected_lag_aic": best_aic_lag,
            "selected_lag_bic": best_bic_lag,
            "selected_lag_used": self.selected_lag,
            "granger_pvalues_on_load": granger_pvals,
            "n_equations": len(self.feature_names_),
            "n_params_per_eq": int(1 + len(self.feature_names_) * self.selected_lag),
        }
        return self

    def predict_walk_forward(
        self,
        all_scaled_values: np.ndarray,
        target_positions: np.ndarray,
        scaler: StandardScaler,
    ) -> np.ndarray:
        """
        Generate 1-step-ahead forecasts at specified target positions using rolling true
        lag windows X_{t-p+1:t} in standardized scale, returning original MW predictions.
        """
        p = self.selected_lag
        # Extract linear VAR matrices (coefs: (p, K, K), intercept: (K,)) for ultra-fast vectorized stepping
        coefs = self.results.coefs  # shape (p, K, K): lag_0 is t, lag_1 is t-1, ..., lag_{p-1} is t-(p-1)
        intercept = self.results.intercept  # shape (K,)

        preds_scaled = np.empty(len(target_positions), dtype=np.float64)
        for i, pos in enumerate(target_positions):
            if self.diff_order == 0:
                # Most recent p observations: pos-1 (lag 1), pos-2 (lag 2), ..., pos-p (lag p)
                window = all_scaled_values[pos - p : pos][::-1]
                # Only compute load (index 0) prediction
                y_hat = float(intercept[0] + np.sum(coefs[:, 0, :] * window))
            else:
                # Most recent p+1 levels -> p first differences
                lvl_win = all_scaled_values[pos - p - 1 : pos]
                diff_win = np.diff(lvl_win, axis=0)[::-1]
                delta_hat = float(intercept[0] + np.sum(coefs[:, 0, :] * diff_win))
                y_hat = float(all_scaled_values[pos - 1, 0] + delta_hat)
            preds_scaled[i] = y_hat

        return MultivariateFeatureBuilder.inverse_transform_load(preds_scaled, scaler)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "VARForecaster":
        return joblib.load(filepath)


class XGBoostMultivariateForecaster:
    """XGBoost Regressor wrapper for tabular multivariate lag + rolling features."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = (params or Config.XGB_PARAMS).copy()
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_scaled: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val_scaled: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "XGBoostMultivariateForecaster":
        eval_set = [(X_train, y_train_scaled)]
        if X_val is not None and y_val_scaled is not None:
            eval_set.append((X_val, y_val_scaled))
        self.model.fit(X_train, y_train_scaled, eval_set=eval_set, verbose=verbose)
        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        return self

    def predict_mw(self, X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
        scaled_pred = self.model.predict(X)
        return MultivariateFeatureBuilder.inverse_transform_load(scaled_pred, scaler)

    @property
    def best_iteration(self) -> Optional[int]:
        return getattr(self.model, "best_iteration", None)

    @property
    def evals_result(self) -> Dict[str, Any]:
        return self.model.evals_result()

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        importances = {col: float(score_dict.get(col, 0.0)) for col in self.feature_names_}
        return pd.Series(importances).sort_values(ascending=False)

    def get_group_importance_share(self) -> Dict[str, float]:
        """Aggregate Gain importance percentage by variable group."""
        imp = self.get_feature_importance(importance_type="gain")
        total = float(imp.sum()) if float(imp.sum()) > 0 else 1.0
        groups = {
            "load_lags (24)": 0.0,
            "temperature_lags (24)": 0.0,
            "humidity_lags (24)": 0.0,
            "wind_speed_lags (24)": 0.0,
            "rolling_and_trend (3)": 0.0,
        }
        for col, val in imp.items():
            if col.startswith("load_lag_"):
                groups["load_lags (24)"] += float(val)
            elif col.startswith("temperature_lag_"):
                groups["temperature_lags (24)"] += float(val)
            elif col.startswith("humidity_lag_"):
                groups["humidity_lags (24)"] += float(val)
            elif col.startswith("wind_speed_lag_"):
                groups["wind_speed_lags (24)"] += float(val)
            else:
                groups["rolling_and_trend (3)"] += float(val)
        return {k: round(v / total * 100.0, 2) for k, v in groups.items()}

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "XGBoostMultivariateForecaster":
        return joblib.load(filepath)


def tune_xgboost_multivariate(
    X_train: pd.DataFrame,
    y_train_scaled: np.ndarray,
    X_val: pd.DataFrame,
    y_val_scaled: np.ndarray,
    y_val_mw: np.ndarray,
    scaler: StandardScaler,
    grid: List[Dict[str, Any]] = Config.XGB_TUNING_GRID,
) -> Tuple[XGBoostMultivariateForecaster, List[Dict[str, Any]], float]:
    """Tune max_depth & n_estimators with early stopping on Validation set."""
    tuning_results: List[Dict[str, Any]] = []
    best_model: Optional[XGBoostMultivariateForecaster] = None
    best_val_rmse = float("inf")
    best_time = 0.0

    for cfg in grid:
        params = Config.XGB_PARAMS.copy()
        params["max_depth"] = cfg["max_depth"]
        params["n_estimators"] = cfg["n_estimators"]

        t0 = time.time()
        candidate = XGBoostMultivariateForecaster(params)
        candidate.fit(X_train, y_train_scaled, X_val, y_val_scaled, verbose=False)
        elapsed = time.time() - t0

        val_pred_mw = candidate.predict_mw(X_val, scaler)
        val_m = evaluate_multivariate_forecast(y_val_mw, val_pred_mw)
        tuning_results.append({
            "name": cfg["name"],
            "max_depth": cfg["max_depth"],
            "n_estimators": cfg["n_estimators"],
            "best_iteration": int((candidate.best_iteration or 0) + 1),
            "val_rmse_mw": val_m["RMSE"],
            "val_mae_mw": val_m["MAE"],
            "val_mape": val_m["MAPE"],
            "train_time_s": elapsed,
        })

        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_model = candidate
            best_time = elapsed

    assert best_model is not None
    return best_model, tuning_results, best_time


class LSTMForecaster:
    """Stacked LSTM network: LSTM(64) -> LSTM(32) -> Dense(1) with MSE loss and Adam."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = (params or Config.LSTM_PARAMS).copy()
        self.model: Optional[tf.keras.Model] = None
        self.history_: Dict[str, List[float]] = {}

    def build_model(self, lookback: int = Config.LOOKBACK, n_features: int = 4) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.params.get("random_state", 42))
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(self.params["units_1"], return_sequences=True),
            tf.keras.layers.LSTM(self.params["units_2"], return_sequences=False),
            tf.keras.layers.Dense(self.params["dense_units"]),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"])
        model.compile(optimizer=optimizer, loss=self.params["loss"], metrics=["mae"])
        self.model = model
        return model

    def fit(
        self,
        X_train_3d: np.ndarray,
        y_train_scaled: np.ndarray,
        X_val_3d: np.ndarray,
        y_val_scaled: np.ndarray,
    ) -> "LSTMForecaster":
        if self.model is None:
            self.build_model(lookback=X_train_3d.shape[1], n_features=X_train_3d.shape[2])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.params["patience"],
            restore_best_weights=True,
        )
        hist = self.model.fit(
            X_train_3d,
            y_train_scaled,
            validation_data=(X_val_3d, y_val_scaled),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            shuffle=False,  # preserve sequence batch ordering during optimization
            callbacks=[early_stop],
            verbose=0,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def predict_mw(self, X_3d: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        if self.model is None:
            raise ValueError("LSTM model is not fitted or loaded.")
        scaled_pred = self.model.predict(X_3d, batch_size=512, verbose=0).reshape(-1)
        return MultivariateFeatureBuilder.inverse_transform_load(scaled_pred, scaler)

    def save(self, filepath: Path) -> None:
        if self.model is None:
            raise ValueError("No LSTM model to save.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: Path) -> "LSTMForecaster":
        inst = cls()
        inst.model = tf.keras.models.load_model(filepath)
        return inst


def run_walk_forward_backtest(
    prepared: Any,
    var_model: VARForecaster,
    xgb_model: XGBoostMultivariateForecaster,
    lstm_model: LSTMForecaster,
    preds_test_map: Dict[str, np.ndarray],
    eval_hours: int = Config.WALK_FORWARD_EVAL_HOURS,
    refit_every: int = Config.WALK_FORWARD_REFIT_EVERY,
) -> Dict[str, Any]:
    """
    Execute Walk-Forward Validation diagnostics:
    1) Rolling 7-day RMSE across the entire Test set using observed actual lags at each hour t.
    2) Adaptive daily-refit simulation over the trailing `eval_hours` (168h) of Test set
       (`Train -> predict t+1..t+24 -> update chain & refit -> predict next day`), compared
       against static walk-forward over the exact same 168-hour horizon.
    """
    idx_test = prepared.idx_test
    y_test_mw = prepared.y_test_mw

    # 1. Full Test Set 7-day (168h) rolling RMSE per model
    rolling_cols: Dict[str, np.ndarray] = {}
    for m_name, y_pred in preds_test_map.items():
        sq_err = pd.Series((y_test_mw - y_pred) ** 2, index=idx_test)
        rolling_cols[m_name] = np.sqrt(sq_err.rolling(window=168, min_periods=24).mean()).values
    daily_rmse_df = pd.DataFrame(rolling_cols, index=idx_test)

    # 2. Adaptive Daily-Refit Simulation over last 168 hours of Test set
    n_total = len(prepared.df_all_scaled)
    sim_start_pos = n_total - eval_hours
    y_sim_true = prepared.df_clean[Config.TARGET_COL].values[sim_start_pos:n_total]

    static_var_tail = preds_test_map["VAR"][-eval_hours:]
    static_xgb_tail = preds_test_map["XGBoost (Multivariate 99f)"][-eval_hours:]
    static_lstm_tail = preds_test_map["LSTM (64->32->1)"][-eval_hours:]

    # Build combined pre-simulation XGB/LSTM arrays for clean expanding slicing
    X_xgb_all = pd.concat([prepared.X_xgb_train, prepared.X_xgb_val, prepared.X_xgb_test])
    y_all_scaled = np.concatenate([prepared.y_train_scaled, prepared.y_val_scaled, prepared.y_test_scaled])
    X_lstm_all = np.concatenate([prepared.X_lstm_train, prepared.X_lstm_val, prepared.X_lstm_test], axis=0)
    n_all_samples = len(X_xgb_all)
    sim_sample_start = n_all_samples - eval_hours

    refit_var_preds: List[float] = []
    refit_xgb_preds: List[float] = []
    refit_lstm_preds: List[float] = []

    # Clone LSTM weights so main model isn't mutated
    lstm_sim = tf.keras.models.clone_model(lstm_model.model)
    lstm_sim.set_weights(lstm_model.model.get_weights())
    lstm_sim.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")

    n_blocks = int(np.ceil(eval_hours / refit_every))
    for blk in range(n_blocks):
        blk_offset = blk * refit_every
        blk_end = min(eval_hours, blk_offset + refit_every)
        cur_sample_split = sim_sample_start + blk_offset
        cur_sample_end = sim_sample_start + blk_end
        cur_raw_split = sim_start_pos + blk_offset
        cur_raw_end = sim_start_pos + blk_end

        # Daily VAR refit on history up to day start
        var_hist_df = pd.DataFrame(
            prepared.df_all_scaled.iloc[:cur_raw_split].values,
            columns=Config.FEATURES,
        )
        var_step = VARForecaster(maxlags=var_model.selected_lag)
        var_step.diff_order = var_model.diff_order
        var_step.selected_lag = var_model.selected_lag
        train_v = var_hist_df if var_step.diff_order == 0 else var_hist_df.diff().dropna()
        var_step.results = VAR(pd.DataFrame(train_v.values, columns=Config.FEATURES)).fit(var_step.selected_lag)
        blk_positions = np.arange(cur_raw_split, cur_raw_end)
        refit_var_preds.extend(
            var_step.predict_walk_forward(prepared.df_all_scaled.values, blk_positions, prepared.scaler).tolist()
        )

        # Daily XGBoost refit (without early stopping on fixed optimal best_iteration)
        xgb_params_sim = xgb_model.params.copy()
        xgb_params_sim.pop("early_stopping_rounds", None)
        xgb_params_sim["n_estimators"] = max(100, int((xgb_model.best_iteration or 300) + 1))
        xgb_sim = xgb.XGBRegressor(**xgb_params_sim)
        xgb_sim.fit(X_xgb_all.iloc[:cur_sample_split], y_all_scaled[:cur_sample_split], verbose=False)
        xgb_blk_scaled = xgb_sim.predict(X_xgb_all.iloc[cur_sample_split:cur_sample_end])
        refit_xgb_preds.extend(
            MultivariateFeatureBuilder.inverse_transform_load(xgb_blk_scaled, prepared.scaler).tolist()
        )

        # Daily lightweight LSTM adaptation on most recent 720 hours (30 days)
        recent_start = max(0, cur_sample_split - 720)
        if blk > 0:
            lstm_sim.fit(
                X_lstm_all[recent_start:cur_sample_split],
                y_all_scaled[recent_start:cur_sample_split],
                epochs=1,
                batch_size=128,
                shuffle=False,
                verbose=0,
            )
        lstm_blk_scaled = lstm_sim.predict(X_lstm_all[cur_sample_split:cur_sample_end], verbose=0).reshape(-1)
        refit_lstm_preds.extend(
            MultivariateFeatureBuilder.inverse_transform_load(lstm_blk_scaled, prepared.scaler).tolist()
        )

    adaptive_168h_summary = {
        "VAR": {
            "static_rmse": evaluate_multivariate_forecast(y_sim_true, static_var_tail)["RMSE"],
            "periodic_refit_rmse": evaluate_multivariate_forecast(y_sim_true, np.array(refit_var_preds))["RMSE"],
        },
        "XGBoost (Multivariate 99f)": {
            "static_rmse": evaluate_multivariate_forecast(y_sim_true, static_xgb_tail)["RMSE"],
            "periodic_refit_rmse": evaluate_multivariate_forecast(y_sim_true, np.array(refit_xgb_preds))["RMSE"],
        },
        "LSTM (64->32->1)": {
            "static_rmse": evaluate_multivariate_forecast(y_sim_true, static_lstm_tail)["RMSE"],
            "periodic_refit_rmse": evaluate_multivariate_forecast(y_sim_true, np.array(refit_lstm_preds))["RMSE"],
        },
    }

    return {
        "daily_rmse_df": daily_rmse_df,
        "eval_hours": eval_hours,
        "refit_every": refit_every,
        "adaptive_168h_summary": adaptive_168h_summary,
    }
