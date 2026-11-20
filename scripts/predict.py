"""
Standalone inference and artifact verification script for Lab 4 Multivariate Forecasting.

Loads saved StandardScaler, VAR, XGBoost, and LSTM artifacts from Lab4_TS/models/,
verifies test-set and trailing 168-hour accuracy, and performs 24-hour recursive future
forecasting beyond the latest observation.

Outputs:
- logs/inference.log (append mode)
- logs/latest_predictions.csv
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.data_loader import MultivariateDataLoader
from scripts.evaluation import evaluate_multivariate_forecast
from scripts.feature_engineering import MultivariateFeatureBuilder
from scripts.models import LSTMForecaster, VARForecaster, XGBoostMultivariateForecaster


def setup_inference_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n")

    logger = logging.getLogger("Lab4InferencePipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def forecast_future_24h(
    scaler: Any,
    var_model: VARForecaster,
    xgb_model: XGBoostMultivariateForecaster,
    lstm_model: LSTMForecaster,
    df_clean: pd.DataFrame,
    df_all_scaled: pd.DataFrame,
    horizon: int,
    generated_at: str,
) -> pd.DataFrame:
    """
    Generate multi-step recursive forecasts for VAR, XGBoost, and LSTM beyond the dataset end.
    VAR forecasts the full 4-variable vector natively; its weather forecasts provide coherent
    future weather covariates when stepping XGBoost and LSTM recursively across t+1..t+H.
    """
    last_ts = df_clean.index.max()
    hist_var = df_all_scaled.values.copy()
    p = var_model.selected_lag

    # 1. Step VAR natively in full 4D multivariate space
    var_future_scaled: List[np.ndarray] = []
    coefs = var_model.results.coefs
    intercept = var_model.results.intercept
    for _ in range(horizon):
        if var_model.diff_order == 0:
            win = hist_var[-p:][::-1]
            # Exact matrix multiplication: sum_{l=0..p-1} coefs[l] @ win[l]
            next_vec = intercept.copy()
            for l_idx in range(p):
                next_vec = next_vec + coefs[l_idx] @ win[l_idx]
        else:
            diffs = np.diff(hist_var[-(p + 1) :], axis=0)[::-1]
            delta = intercept.copy()
            for l_idx in range(p):
                delta = delta + coefs[l_idx] @ diffs[l_idx]
            next_vec = hist_var[-1] + delta
        var_future_scaled.append(next_vec)
        hist_var = np.vstack([hist_var, next_vec])

    var_future_arr = np.asarray(var_future_scaled)
    var_future_mw = MultivariateFeatureBuilder.inverse_transform_load(var_future_arr[:, 0], scaler)

    # 2. Step XGBoost (99f) and LSTM (24x4) using VAR weather covariates + own recursive load
    hist_xgb = df_all_scaled.values.copy()
    hist_lstm = df_all_scaled.values.copy()
    full_cols, _ = MultivariateFeatureBuilder.get_xgb_feature_names()

    xgb_future_scaled: List[float] = []
    lstm_future_scaled: List[float] = []

    for h in range(horizon):
        weather_h = var_future_arr[h, 1:]

        # XGBoost 1-step feature vector
        win_xgb = hist_xgb[-Config.LOOKBACK :][::-1].reshape(-1)
        r6 = float(np.mean(hist_xgb[-Config.ROLLING_MEAN_WINDOW :, 0]))
        r24 = float(np.std(hist_xgb[-Config.ROLLING_STD_WINDOW :, 0], ddof=1))
        tr24 = float(hist_xgb[-1, 0] - hist_xgb[-1 - Config.TREND_LAG, 0])
        x_row = np.concatenate([win_xgb, [r6, r24, tr24]])
        y_xgb_sc = float(xgb_model.model.predict(pd.DataFrame([x_row], columns=full_cols))[0])
        xgb_future_scaled.append(y_xgb_sc)
        hist_xgb = np.vstack([hist_xgb, np.concatenate([[y_xgb_sc], weather_h])])

        # LSTM 1-step 3D tensor
        win_lstm = hist_lstm[-Config.LOOKBACK :].reshape(1, Config.LOOKBACK, 4)
        y_lstm_sc = float(lstm_model.model.predict(win_lstm, verbose=0).reshape(-1)[0])
        lstm_future_scaled.append(y_lstm_sc)
        hist_lstm = np.vstack([hist_lstm, np.concatenate([[y_lstm_sc], weather_h])])

    xgb_future_mw = MultivariateFeatureBuilder.inverse_transform_load(np.array(xgb_future_scaled), scaler)
    lstm_future_mw = MultivariateFeatureBuilder.inverse_transform_load(np.array(lstm_future_scaled), scaler)

    rows: List[Dict[str, Any]] = []
    for step in range(1, horizon + 1):
        ts = str(last_ts + pd.Timedelta(hours=step))
        for m_name, pred_arr in [
            ("XGBoost (Multivariate 99f)", xgb_future_mw),
            ("LSTM (64->32->1)", lstm_future_mw),
            ("VAR", var_future_mw),
        ]:
            rows.append({
                "Datetime": ts,
                "horizon_step": step,
                "actual_load_MW": np.nan,
                "predicted_load_MW": round(float(pred_arr[step - 1]), 2),
                "abs_error_MW": np.nan,
                "record_type": "future_forecast",
                "model_name": m_name,
                "generated_at": generated_at,
            })
    return pd.DataFrame(rows)


def run_inference(eval_hours: int = 168, horizon: int = 24) -> None:
    logger = setup_inference_logger(Config.INFERENCE_LOG_PATH)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 90)
    logger.info(f"STARTING LAB 4 STANDALONE INFERENCE PIPELINE (eval_hours={eval_hours}, horizon={horizon}h)")
    logger.info("=" * 90)

    scaler = joblib.load(Config.SCALER_PATH)
    var_model = VARForecaster.load(Config.MODELS_DIR / Config.MODEL_FILENAMES["VAR"])
    xgb_model = XGBoostMultivariateForecaster.load(Config.MODELS_DIR / Config.MODEL_FILENAMES["XGBoost (Multivariate 99f)"])
    lstm_model = LSTMForecaster.load(Config.MODELS_DIR / Config.MODEL_FILENAMES["LSTM (64->32->1)"])

    if Config.MODEL_REGISTRY_PATH.exists():
        reg = json.loads(Config.MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        logger.info(f"Loaded Model Registry ({reg['models_count']} models, default: {reg['default_model']})")

    loader = MultivariateDataLoader()
    df_clean, _ = loader.load_clean_multivariate(export_path=None)
    prepared = MultivariateFeatureBuilder.prepare_all(df_clean, export_dir=None, scaler_path=None)

    n_total = len(df_clean)
    val_end = len(prepared.df_train_raw) + len(prepared.df_val_raw)
    test_positions = np.arange(val_end, n_total)

    preds_test = {
        "VAR": var_model.predict_walk_forward(prepared.df_all_scaled.values, test_positions, scaler),
        "XGBoost (Multivariate 99f)": xgb_model.predict_mw(prepared.X_xgb_test, scaler),
        "LSTM (64->32->1)": lstm_model.predict_mw(prepared.X_lstm_test, scaler),
    }

    trailing_frames: List[pd.DataFrame] = []
    tail_idx = prepared.idx_test[-eval_hours:]
    y_tail_true = prepared.y_test_mw[-eval_hours:]

    for m_name, y_pred_all in preds_test.items():
        m_full = evaluate_multivariate_forecast(prepared.y_test_mw, y_pred_all)
        m_tail = evaluate_multivariate_forecast(y_tail_true, y_pred_all[-eval_hours:])
        logger.info(
            f"  -> {m_name:<28} | Full Test RMSE: {m_full['RMSE']:6.2f} MW (MAE: {m_full['MAE']:6.2f}, MAPE: {m_full['MAPE']:5.2f}%) | "
            f"Last {eval_hours}h RMSE: {m_tail['RMSE']:6.2f} MW"
        )
        trailing_frames.append(pd.DataFrame({
            "Datetime": [str(ts) for ts in tail_idx],
            "horizon_step": np.nan,
            "actual_load_MW": np.round(y_tail_true, 2),
            "predicted_load_MW": np.round(y_pred_all[-eval_hours:], 2),
            "abs_error_MW": np.round(np.abs(y_tail_true - y_pred_all[-eval_hours:]), 2),
            "record_type": "trailing_eval",
            "model_name": m_name,
            "generated_at": generated_at,
        }))

    future_df = forecast_future_24h(
        scaler, var_model, xgb_model, lstm_model, df_clean, prepared.df_all_scaled, horizon, generated_at
    )
    for m_name in ["XGBoost (Multivariate 99f)", "LSTM (64->32->1)", "VAR"]:
        sub = future_df[future_df["model_name"] == m_name]
        logger.info(
            f"  -> Future {horizon}h [{m_name:<26}]: Min={sub['predicted_load_MW'].min():,.1f} MW | "
            f"Max={sub['predicted_load_MW'].max():,.1f} MW | Mean={sub['predicted_load_MW'].mean():,.1f} MW"
        )

    out_df = pd.concat(trailing_frames + [future_df], ignore_index=True)
    Config.PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(Config.PREDICTIONS_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved {len(out_df):,} prediction rows to: {Config.PREDICTIONS_CSV_PATH}")
    logger.info("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 4 Standalone Inference Pipeline")
    parser.add_argument("--eval-hours", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    run_inference(eval_hours=args.eval_hours, horizon=args.horizon)
