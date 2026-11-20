"""
Automated CLI training, evaluation, Walk-Forward backtesting, and artifact logging pipeline
for Lab 4 Multivariate Time Series Forecasting (Electricity Load + Weather).

Implements 100% of practice 4.pdf (Sections 1-11):
- Data understanding, outlier & missing value handling
- StandardScaler + Chronological split (70% / 15% / 15%)
- Feature engineering (L=24 multivariate window, rolling/trend features, 3D LSTM tensor)
- Model 1: VAR (ADF stationarity check, AIC/BIC lag order selection)
- Model 2: XGBoost (Hyperparameter tuning with Early Stopping + Weather ablation)
- Model 3: LSTM(64) -> LSTM(32) -> Dense(1) (MSE + Adam + EarlyStopping)
- Walk-forward validation & 6 analytical diagnostics

Usage:
    python scripts/train.py
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn
import statsmodels
import tensorflow as tf
import xgboost

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.data_loader import MultivariateDataLoader
from scripts.evaluation import evaluate_multivariate_forecast, residual_diagnostics
from scripts.feature_engineering import MultivariateFeatureBuilder
from scripts.models import (
    LSTMForecaster,
    VARForecaster,
    XGBoostMultivariateForecaster,
    run_walk_forward_backtest,
    tune_xgboost_multivariate,
)
from scripts.visualization import (
    plot_eda_load_temp_and_corr,
    plot_eda_timeseries,
    plot_future_forecast_and_summary,
    plot_lstm_learning_curves,
    plot_missing_and_outliers_audit,
    plot_models_comparison_metrics,
    plot_predictions_vs_actual,
    plot_residual_and_hourly_error,
    plot_rolling_and_trend_features,
    plot_seasonality_and_split,
    plot_standard_scaler_distributions,
    plot_var_stationarity_and_lag_selection,
    plot_walk_forward_backtest,
    plot_xgboost_learning_and_ablation,
    plot_xgboost_tuning_and_importance,
)


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure logger in append mode writing to file and stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n")

    logger = logging.getLogger("Lab4TrainingPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _evaluate_model_entry(
    model_name: str,
    feature_set: str,
    feature_count: int,
    train_time_s: float,
    y_train_mw: np.ndarray,
    y_val_mw: np.ndarray,
    y_test_mw: np.ndarray,
    pred_train_mw: np.ndarray,
    pred_val_mw: np.ndarray,
    pred_test_mw: np.ndarray,
    artifact_path: str = "",
) -> Dict[str, Any]:
    tr = evaluate_multivariate_forecast(y_train_mw, pred_train_mw)
    va = evaluate_multivariate_forecast(y_val_mw, pred_val_mw)
    te = evaluate_multivariate_forecast(y_test_mw, pred_test_mw)
    diag = residual_diagnostics(y_test_mw, pred_test_mw, lags=24)
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "feature_count": int(feature_count),
        "train_time_s": float(train_time_s),
        "train_metrics": tr,
        "val_metrics": va,
        "test_metrics": te,
        "overfit_gap_rmse": float(te["RMSE"] - tr["RMSE"]),
        "residual_diagnostics": diag,
        "pred_test_mw": pred_test_mw,
        "artifact_path": artifact_path,
    }


def _log_eval(logger: logging.Logger, entry: Dict[str, Any]) -> None:
    tr, va, te = entry["train_metrics"], entry["val_metrics"], entry["test_metrics"]
    d = entry["residual_diagnostics"]
    logger.info(f"--- Evaluation: {entry['model_name']} ({entry['feature_set']}, {entry['train_time_s']:.2f}s) ---")
    logger.info(f"  [Train] RMSE: {tr['RMSE']:7.2f} MW | MAE: {tr['MAE']:6.2f} MW | MAPE: {tr['MAPE']:5.2f}% | R2: {tr['R2']:.4f}")
    logger.info(f"  [Val]   RMSE: {va['RMSE']:7.2f} MW | MAE: {va['MAE']:6.2f} MW | MAPE: {va['MAPE']:5.2f}% | R2: {va['R2']:.4f}")
    logger.info(f"  [Test]  RMSE: {te['RMSE']:7.2f} MW | MAE: {te['MAE']:6.2f} MW | MAPE: {te['MAPE']:5.2f}% | R2: {te['R2']:.4f}")
    logger.info(f"  [Gap & Residuals] Overfit Gap: {entry['overfit_gap_rmse']:+.2f} MW | Bias: {d['residual_mean']:+.2f} MW | LB(24) p={d['ljung_box_pvalue']:.2e}")


def write_summary_report(
    summary_path: Path,
    env_info: Dict[str, Any],
    audit: Dict[str, Any],
    split_summary: Dict[str, Dict[str, Any]],
    var_diag: Dict[str, Any],
    xgb_tuning: List[Dict[str, Any]],
    xgb_groups: Dict[str, float],
    xgb_top10: pd.Series,
    lstm_hist: Dict[str, List[float]],
    scaler_load_std: float,
    wf_res: Dict[str, Any],
    leaderboard: List[Dict[str, Any]],
    q_diagnostics: Dict[str, Any],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 104
    sub = "-" * 104
    lines: List[str] = [
        sep,
        "LAB 4: MULTIVARIATE TIME SERIES FORECASTING (ELECTRICITY LOAD + WEATHER) - BENCHMARK REPORT",
        sep,
        f"Generated Timestamp : {env_info['timestamp']}",
        f"Environment         : Python {env_info['python_version']} | TF {env_info['tf_version']} | XGB {env_info['xgboost_version']} | Statsmodels {env_info['statsmodels_version']}",
        "",
        sep,
        "1. DATASET AUDIT & PREPROCESSING SUMMARY (SECTIONS 2-4)",
        sep,
        f"- Merged Dataset    : {audit['clean_rows']:,} hourly rows ({audit['start_time']} -> {audit['end_time']})",
        f"- Cities Aggregated : {', '.join(audit['cities'])}",
        f"- Raw Missing Values: load={audit['raw_nan_counts']['load']} NaNs (0 after time interpolation)",
        f"- Wind Outliers     : {audit['raw_wind_outliers_gt40ms']} raw records > 40 m/s cleaned before interpolation",
        "- Pearson Corr Load : " + ", ".join([f"{k}={v:+.4f}" for k, v in audit["pearson_corr"]["load"].items() if k != "load"]),
        "- Spearman Corr Load: " + ", ".join([f"{k}={v:+.4f}" for k, v in audit["spearman_corr"]["load"].items() if k != "load"]),
        sub,
        f"{'Split (70/15/15)':<14} | {'Raw Rows':>8} | {'Samples':>8} | {'Start UTC':<25} | {'End UTC':<25} | {'Mean (MW)':>10}",
        sub,
    ]
    for s_name, meta in split_summary.items():
        lines.append(
            f"{s_name:<14} | {meta['raw_rows']:>8,} | {meta['sample_rows']:>8,} | {meta['start']:<25} | {meta['end']:<25} | {meta['load_mean_mw']:>10,.2f}"
        )
    lines.extend([
        sub,
        "",
        sep,
        "2. COMPLETE MODEL PERFORMANCE LEADERBOARD ON TEST SET (RMSE / MAE / MAPE)",
        sep,
        f"{'Rank':<4} | {'Model Name':<28} | {'Features':<18} | {'Train RMSE':>10} | {'Val RMSE':>8} | {'Test RMSE':>9} | {'Test MAE':>8} | {'MAPE':>6} | {'R2':>6}",
        sub,
    ])
    for row in leaderboard:
        lines.append(
            f"#{row['rank']:<3} | {row['model_name']:<28} | {row['feature_set']:<18} | "
            f"{row['train_rmse']:>10.2f} | {row['val_rmse']:>8.2f} | {row['test_rmse']:>9.2f} | "
            f"{row['test_mae']:>8.2f} | {row['test_mape']:>5.2f}% | {row['test_r2']:>6.4f}"
        )

    lines.extend([
        sub,
        "",
        sep,
        "3. VAR DIAGNOSTICS (ADF STATIONARITY, AIC/BIC LAG ORDER, GRANGER CAUSALITY)",
        sep,
        f"- Differencing Applied: d = {var_diag['diff_order']} | Selected Lag Order (AIC) = {var_diag['selected_lag_aic']} | BIC = {var_diag['selected_lag_bic']}",
        f"- Parameters per Eq.  : {var_diag['n_params_per_eq']} (4 equations x {var_diag['n_params_per_eq']} = {4 * var_diag['n_params_per_eq']} total parameters)",
    ])
    for var_name, adf in var_diag["adf_results"].items():
        lines.append(
            f"  * ADF {var_name:<12}: Level stat={adf['level_stat']:7.2f} (p={adf['level_pvalue']:.4e}, stationary={adf['level_stationary']}) | "
            f"Diff1 p={adf['diff1_pvalue']:.4e}"
        )
    lines.append("- Granger Causality on Load (p-values): " + ", ".join([f"{k}={v:.4e}" for k, v in var_diag["granger_pvalues_on_load"].items()]))

    lines.extend([
        "",
        sep,
        "4. XGBOOST TUNING & FEATURE GROUP IMPORTANCE",
        sep,
    ])
    for cfg in xgb_tuning:
        lines.append(
            f"  * {cfg['name']:<20} | Best Trees={cfg['best_iteration']:>3d} | Val RMSE={cfg['val_rmse_mw']:6.2f} MW | Val MAE={cfg['val_mae_mw']:6.2f} MW"
        )
    lines.append("- Gain Share by Variable Group: " + ", ".join([f"{k}={v:.2f}%" for k, v in xgb_groups.items()]))
    lines.append("- Top 10 Features (Gain): " + ", ".join([f"{k}={v:.1f}" for k, v in xgb_top10.items()]))

    tr_end_rmse = float(np.sqrt(lstm_hist["loss"][-1]) * scaler_load_std)
    va_best_rmse = float(np.sqrt(min(lstm_hist["val_loss"])) * scaler_load_std)
    lines.extend([
        "",
        sep,
        "5. LSTM ARCHITECTURE & OVERFITTING DIAGNOSIS",
        sep,
        f"- Architecture : Input(24, 4) -> LSTM(64, return_sequences=True) -> LSTM(32) -> Dense(1)",
        f"- Epochs Run   : {len(lstm_hist['loss'])} | Final Train RMSE={tr_end_rmse:.2f} MW | Best Val RMSE={va_best_rmse:.2f} MW",
        "",
        sep,
        "6. WALK-FORWARD BACKTESTING (168H STATIC VS DAILY REFIT)",
        sep,
    ])
    for m_name, scores in wf_res["adaptive_168h_summary"].items():
        lines.append(
            f"  * {m_name:<28}: Static RMSE={scores['static_rmse']:.2f} MW | Daily-Refit RMSE={scores['periodic_refit_rmse']:.2f} MW"
        )

    lines.extend([
        "",
        sep,
        "7. ANALYTICAL QUESTIONS QUANTITATIVE EVIDENCE (Q1-Q6)",
        sep,
        f"- Q1 Strongest Weather Var : humidity (|r|={abs(audit['pearson_corr']['load']['humidity']):.4f}), temperature Granger p={var_diag['granger_pvalues_on_load']['temperature']:.2e}, Gain={xgb_groups['temperature_lags (24)']:.2f}% | Weather Ablation Delta RMSE={q_diagnostics['weather_ablation_delta_rmse_mw']:+.2f} MW",
        f"- Q2 VAR vs Non-Linear ML  : Test RMSE gap VAR-XGB={q_diagnostics['var_vs_xgb_rmse_gap_mw']:+.2f} MW | VAR-LSTM={q_diagnostics['var_vs_lstm_rmse_gap_mw']:+.2f} MW",
        f"- Q5 LSTM Overfit Gap      : Test RMSE - Train RMSE = {q_diagnostics['lstm_overfit_gap_rmse']:+.2f} MW",
        sep,
        "END OF LAB 4 BENCHMARK REPORT",
        sep,
    ])

    report_text = "\n".join(lines) + "\n"
    if summary_path.exists() and summary_path.stat().st_size > 0:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + report_text)
    else:
        summary_path.write_text(report_text, encoding="utf-8")


def run_pipeline() -> None:
    logger = setup_logger(Config.PIPELINE_LOG_PATH)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    env_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_platform": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "statsmodels_version": statsmodels.__version__,
        "tf_version": tf.__version__,
    }

    logger.info("=" * 92)
    logger.info("STARTING LAB 4 MULTIVARIATE TIME SERIES PIPELINE (VAR vs XGBoost vs LSTM)")
    logger.info("=" * 92)

    # [1/8] Load & Clean
    logger.info("[1/8] Loading, merging & cleaning Spain Electricity Load + Weather dataset...")
    loader = MultivariateDataLoader()
    df_clean, audit = loader.load_clean_multivariate()
    logger.info(f"-> Merged clean dataset: {audit['clean_rows']:,} hours ({audit['start_time']} -> {audit['end_time']})")
    logger.info(f"-> Raw NaNs in load: {audit['raw_nan_counts']['load']} | Cleaned Wind Outliers (>40 m/s): {audit['raw_wind_outliers_gt40ms']}")
    logger.info("-> Pearson corr with load: " + ", ".join([f"{k}={v:+.4f}" for k, v in audit["pearson_corr"]["load"].items() if k != "load"]))

    # [2/8] Preprocess, Scale, Build Windows
    logger.info("-" * 92)
    logger.info("[2/8] Splitting chronologically (70/15/15), fitting StandardScaler on Train & building 2D/3D windows...")
    prepared = MultivariateFeatureBuilder.prepare_all(df_clean)
    for s_name, meta in prepared.split_summary.items():
        logger.info(f"-> {s_name:<10}: {meta['raw_rows']:,} raw rows | {meta['sample_rows']:,} window samples | [{meta['start']} -> {meta['end']}]")
    logger.info(f"-> Shapes: X_xgb_train={prepared.X_xgb_train.shape}, X_lstm_train={prepared.X_lstm_train.shape}, X_lstm_test={prepared.X_lstm_test.shape}")

    model_entries: Dict[str, Dict[str, Any]] = {}
    hyperparams_map: Dict[str, Dict[str, Any]] = {}

    # [3/8] Naive Baselines & Model 1: VAR
    logger.info("-" * 92)
    logger.info("[3/8] Evaluating Naive Baselines & training Model 1: VAR...")
    for lag_col, b_name in [
        ("load_lag_1", "Persistence Naive (lag_1)"),
        ("load_lag_24", "Seasonal Naive 24h (lag_24)"),
    ]:
        p_tr = MultivariateFeatureBuilder.inverse_transform_load(prepared.X_xgb_train[lag_col].values, prepared.scaler)
        p_va = MultivariateFeatureBuilder.inverse_transform_load(prepared.X_xgb_val[lag_col].values, prepared.scaler)
        p_te = MultivariateFeatureBuilder.inverse_transform_load(prepared.X_xgb_test[lag_col].values, prepared.scaler)
        entry = _evaluate_model_entry(b_name, f"1 Lag ({lag_col})", 1, 0.0, prepared.y_train_mw, prepared.y_val_mw, prepared.y_test_mw, p_tr, p_va, p_te)
        model_entries[b_name] = entry
        _log_eval(logger, entry)

    t0 = time.time()
    var_model = VARForecaster(maxlags=Config.VAR_MAXLAGS).fit(prepared.df_train_scaled)
    var_time = time.time() - t0
    var_diag = var_model.diagnostics_
    logger.info(f"-> VAR ADF stationary flags (level): " + ", ".join([f"{k}={v['level_stationary']}(p={v['level_pvalue']:.3e})" for k, v in var_diag["adf_results"].items()]))
    logger.info(f"-> VAR diff_order={var_diag['diff_order']} | Selected lag p={var_diag['selected_lag_used']} (AIC={var_diag['selected_lag_aic']}, BIC={var_diag['selected_lag_bic']})")
    logger.info("-> VAR Granger causality on load p-vals: " + ", ".join([f"{k}={v:.3e}" for k, v in var_diag["granger_pvalues_on_load"].items()]))

    n_total = len(df_clean)
    train_end = len(prepared.df_train_raw)
    val_end = train_end + len(prepared.df_val_raw)
    min_pos = max(Config.LOOKBACK, Config.TREND_LAG + 1)
    all_sc = prepared.df_all_scaled.values
    var_tr_mw = var_model.predict_walk_forward(all_sc, np.arange(min_pos, train_end), prepared.scaler)
    var_va_mw = var_model.predict_walk_forward(all_sc, np.arange(train_end, val_end), prepared.scaler)
    var_te_mw = var_model.predict_walk_forward(all_sc, np.arange(val_end, n_total), prepared.scaler)

    var_path = Config.MODELS_DIR / Config.MODEL_FILENAMES["VAR"]
    var_model.save(var_path)
    hyperparams_map["VAR"] = {"maxlags": Config.VAR_MAXLAGS, "selected_lag": var_diag["selected_lag_used"], "diff_order": var_diag["diff_order"]}
    model_entries["VAR"] = _evaluate_model_entry(
        "VAR", f"4 Vars x p={var_diag['selected_lag_used']}", 4 * var_diag["selected_lag_used"], var_time,
        prepared.y_train_mw, prepared.y_val_mw, prepared.y_test_mw, var_tr_mw, var_va_mw, var_te_mw, str(var_path),
    )
    _log_eval(logger, model_entries["VAR"])

    # [4/8] Model 2: XGBoost (Load-Only Ablation + Multivariate Tuned)
    logger.info("-" * 92)
    logger.info("[4/8] Training Model 2: XGBoost (Tuning max_depth & n_estimators + Weather Ablation)...")
    t0 = time.time()
    xgb_load_only = XGBoostMultivariateForecaster(Config.XGB_PARAMS).fit(
        prepared.X_xgb_load_train, prepared.y_train_scaled, prepared.X_xgb_load_val, prepared.y_val_scaled, verbose=False
    )
    xgb_lo_time = time.time() - t0
    xgb_lo_path = Config.MODELS_DIR / Config.MODEL_FILENAMES["XGBoost (Load-Only 27f)"]
    xgb_load_only.save(xgb_lo_path)
    hyperparams_map["XGBoost (Load-Only 27f)"] = Config.XGB_PARAMS.copy()
    model_entries["XGBoost (Load-Only 27f)"] = _evaluate_model_entry(
        "XGBoost (Load-Only 27f)", "24 Load Lags + 3 Roll", 27, xgb_lo_time,
        prepared.y_train_mw, prepared.y_val_mw, prepared.y_test_mw,
        xgb_load_only.predict_mw(prepared.X_xgb_load_train, prepared.scaler),
        xgb_load_only.predict_mw(prepared.X_xgb_load_val, prepared.scaler),
        xgb_load_only.predict_mw(prepared.X_xgb_load_test, prepared.scaler),
        str(xgb_lo_path),
    )
    _log_eval(logger, model_entries["XGBoost (Load-Only 27f)"])

    xgb_model, xgb_tuning, xgb_time = tune_xgboost_multivariate(
        prepared.X_xgb_train, prepared.y_train_scaled, prepared.X_xgb_val, prepared.y_val_scaled, prepared.y_val_mw, prepared.scaler
    )
    for cfg in xgb_tuning:
        logger.info(f"  -> Tuning {cfg['name']:<18} | Best Iter={cfg['best_iteration']:>3d} | Val RMSE={cfg['val_rmse_mw']:6.2f} MW")

    xgb_path = Config.MODELS_DIR / Config.MODEL_FILENAMES["XGBoost (Multivariate 99f)"]
    xgb_model.save(xgb_path)
    hyperparams_map["XGBoost (Multivariate 99f)"] = xgb_model.params.copy()
    model_entries["XGBoost (Multivariate 99f)"] = _evaluate_model_entry(
        "XGBoost (Multivariate 99f)", "96 Lags + 3 Roll/Trend", 99, xgb_time,
        prepared.y_train_mw, prepared.y_val_mw, prepared.y_test_mw,
        xgb_model.predict_mw(prepared.X_xgb_train, prepared.scaler),
        xgb_model.predict_mw(prepared.X_xgb_val, prepared.scaler),
        xgb_model.predict_mw(prepared.X_xgb_test, prepared.scaler),
        str(xgb_path),
    )
    _log_eval(logger, model_entries["XGBoost (Multivariate 99f)"])
    xgb_groups = xgb_model.get_group_importance_share()
    xgb_top10 = xgb_model.get_feature_importance().head(10)
    logger.info("-> XGBoost Gain Share by Group: " + ", ".join([f"{k}={v:.2f}%" for k, v in xgb_groups.items()]))

    # [5/8] Model 3: LSTM (64 -> 32 -> 1)
    logger.info("-" * 92)
    logger.info("[5/8] Training Model 3: Stacked LSTM(64) -> LSTM(32) -> Dense(1)...")
    t0 = time.time()
    lstm_model = LSTMForecaster(Config.LSTM_PARAMS).fit(
        prepared.X_lstm_train, prepared.y_train_scaled, prepared.X_lstm_val, prepared.y_val_scaled
    )
    lstm_time = time.time() - t0
    lstm_path = Config.MODELS_DIR / Config.MODEL_FILENAMES["LSTM (64->32->1)"]
    lstm_model.save(lstm_path)
    hyperparams_map["LSTM (64->32->1)"] = Config.LSTM_PARAMS.copy()
    model_entries["LSTM (64->32->1)"] = _evaluate_model_entry(
        "LSTM (64->32->1)", "3D Tensor (24x4)", 96, lstm_time,
        prepared.y_train_mw, prepared.y_val_mw, prepared.y_test_mw,
        lstm_model.predict_mw(prepared.X_lstm_train, prepared.scaler),
        lstm_model.predict_mw(prepared.X_lstm_val, prepared.scaler),
        lstm_model.predict_mw(prepared.X_lstm_test, prepared.scaler),
        str(lstm_path),
    )
    _log_eval(logger, model_entries["LSTM (64->32->1)"])
    logger.info(f"-> LSTM Epochs Completed: {len(lstm_model.history_['loss'])}")

    # [6/8] Walk-Forward Validation
    logger.info("-" * 92)
    logger.info("[6/8] Running Walk-Forward Validation (Full Test Rolling RMSE + 168h Adaptive Refit Simulation)...")
    preds_test_map = {
        "VAR": model_entries["VAR"]["pred_test_mw"],
        "XGBoost (Multivariate 99f)": model_entries["XGBoost (Multivariate 99f)"]["pred_test_mw"],
        "LSTM (64->32->1)": model_entries["LSTM (64->32->1)"]["pred_test_mw"],
    }
    wf_res = run_walk_forward_backtest(prepared, var_model, xgb_model, lstm_model, preds_test_map)
    for m_name, sc in wf_res["adaptive_168h_summary"].items():
        logger.info(f"  -> 168h Walk-Forward {m_name:<28}: Static RMSE={sc['static_rmse']:.2f} MW | Daily-Refit RMSE={sc['periodic_refit_rmse']:.2f} MW")

    # [7/8] Analytical Q1-Q6 Diagnostics
    q_diagnostics = {
        "weather_ablation_delta_rmse_mw": float(model_entries["XGBoost (Load-Only 27f)"]["test_metrics"]["RMSE"] - model_entries["XGBoost (Multivariate 99f)"]["test_metrics"]["RMSE"]),
        "var_vs_xgb_rmse_gap_mw": float(model_entries["VAR"]["test_metrics"]["RMSE"] - model_entries["XGBoost (Multivariate 99f)"]["test_metrics"]["RMSE"]),
        "var_vs_lstm_rmse_gap_mw": float(model_entries["VAR"]["test_metrics"]["RMSE"] - model_entries["LSTM (64->32->1)"]["test_metrics"]["RMSE"]),
        "lstm_overfit_gap_rmse": float(model_entries["LSTM (64->32->1)"]["overfit_gap_rmse"]),
    }

    # [8/8] Export Artifacts & Plots
    logger.info("-" * 92)
    logger.info("[8/8] Exporting structured leaderboard CSV, model registry JSON, summary report & 10 plots...")
    ranked = sorted(model_entries.values(), key=lambda x: x["test_metrics"]["RMSE"])
    leaderboard_rows: List[Dict[str, Any]] = []
    registry_models: List[Dict[str, Any]] = []

    for idx, res in enumerate(ranked, start=1):
        tr, va, te, d = res["train_metrics"], res["val_metrics"], res["test_metrics"], res["residual_diagnostics"]
        row = {
            "rank": idx,
            "model_name": res["model_name"],
            "feature_set": res["feature_set"],
            "feature_count": res["feature_count"],
            "train_rmse": round(tr["RMSE"], 4),
            "train_mae": round(tr["MAE"], 4),
            "val_rmse": round(va["RMSE"], 4),
            "val_mae": round(va["MAE"], 4),
            "test_rmse": round(te["RMSE"], 4),
            "test_mae": round(te["MAE"], 4),
            "test_mape": round(te["MAPE"], 4),
            "test_r2": round(te["R2"], 6),
            "overfit_gap_rmse": round(res["overfit_gap_rmse"], 4),
            "residual_mean": round(d["residual_mean"], 4),
            "ljung_box_pvalue": float(d["ljung_box_pvalue"]),
            "train_time_s": round(res["train_time_s"], 4),
            "artifact_path": res["artifact_path"],
        }
        leaderboard_rows.append(row)
        if res["artifact_path"]:
            registry_models.append(_json_safe({
                "model_name": res["model_name"],
                "filename": Path(res["artifact_path"]).name,
                "path": res["artifact_path"],
                "feature_set": res["feature_set"],
                "feature_count": res["feature_count"],
                "hyperparameters": hyperparams_map.get(res["model_name"], {}),
                "test_metrics": {k: round(float(v), 4) for k, v in te.items()},
                "train_time_s": round(res["train_time_s"], 4),
                "timestamp": env_info["timestamp"],
            }))

    pd.DataFrame(leaderboard_rows).to_csv(Config.LEADERBOARD_CSV_PATH, index=False, encoding="utf-8")
    reg_payload = _json_safe({
        "generated_at": env_info["timestamp"],
        "scaler_path": str(Config.SCALER_PATH),
        "default_model": ranked[0]["model_name"],
        "models_count": len(registry_models),
        "models": registry_models,
    })
    Config.MODEL_REGISTRY_PATH.write_text(json.dumps(reg_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    history: List[Dict[str, Any]] = []
    if Config.EXPERIMENTS_JSON_PATH.exists() and Config.EXPERIMENTS_JSON_PATH.stat().st_size > 0:
        loaded = json.loads(Config.EXPERIMENTS_JSON_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            history = loaded
    total_elapsed = time.time() - t_start
    history.append(_json_safe({
        "run_id": f"run_{len(history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": env_info["timestamp"],
        "duration_s": round(total_elapsed, 2),
        "environment": env_info,
        "split_summary": prepared.split_summary,
        "leaderboard": leaderboard_rows,
        "var_diagnostics": {"diff_order": var_diag["diff_order"], "selected_lag": var_diag["selected_lag_used"], "granger_pvals": var_diag["granger_pvalues_on_load"]},
        "walk_forward_168h": wf_res["adaptive_168h_summary"],
        "analytical_diagnostics": q_diagnostics,
    }))
    Config.EXPERIMENTS_JSON_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_summary_report(
        Config.SUMMARY_REPORT_PATH, env_info, audit, prepared.split_summary, var_diag,
        xgb_tuning, xgb_groups, xgb_top10, lstm_model.history_, float(prepared.scaler.scale_[0]),
        wf_res, leaderboard_rows, q_diagnostics,
    )

    # Generate all 15 diagnostic & comparison plots
    plot_eda_timeseries(df_clean, Config.IMAGE_DIR / "01_eda_load_and_weather_timeseries.png")
    plot_eda_load_temp_and_corr(df_clean, Config.IMAGE_DIR / "02_eda_load_vs_temperature_and_corr.png")
    plot_seasonality_and_split(df_clean, prepared.df_train_raw, prepared.df_val_raw, prepared.df_test_raw, Config.IMAGE_DIR / "03_seasonality_and_split.png")
    plot_var_stationarity_and_lag_selection(var_diag, Config.IMAGE_DIR / "04_var_stationarity_and_aic_bic.png")
    plot_xgboost_tuning_and_importance(xgb_tuning, xgb_groups, xgb_model.get_feature_importance(), Config.IMAGE_DIR / "05_xgboost_tuning_and_feature_importance.png")
    plot_lstm_learning_curves(lstm_model.history_, float(prepared.scaler.scale_[0]), Config.IMAGE_DIR / "06_lstm_learning_curves.png")
    plot_models_comparison_metrics(leaderboard_rows, Config.IMAGE_DIR / "07_models_comparison_metrics.png")
    plot_predictions_vs_actual(prepared.idx_test, prepared.y_test_mw, preds_test_map, Config.IMAGE_DIR / "08_predictions_vs_actual_test.png")
    plot_walk_forward_backtest(wf_res, Config.IMAGE_DIR / "09_walk_forward_backtest.png")
    plot_residual_and_hourly_error(prepared.idx_test, prepared.y_test_mw, prepared.df_test_raw["temperature"].values, preds_test_map, Config.IMAGE_DIR / "10_residual_and_hourly_error_analysis.png")
    plot_missing_and_outliers_audit(Config.RAW_ENERGY_PATH, Config.RAW_WEATHER_PATH, df_clean, Config.IMAGE_DIR / "11_missing_values_and_outliers_audit.png")
    plot_standard_scaler_distributions(prepared.df_train_raw, prepared.df_train_scaled, Config.IMAGE_DIR / "12_standard_scaler_distributions.png")
    plot_rolling_and_trend_features(df_clean, Config.IMAGE_DIR / "13_rolling_and_load_trend_features.png")
    plot_xgboost_learning_and_ablation(xgb_model, float(prepared.scaler.scale_[0]), df_clean, leaderboard_rows, wf_res["adaptive_168h_summary"], Config.IMAGE_DIR / "14_xgboost_learning_curves_and_weather_ablation.png")
    plot_future_forecast_and_summary(df_clean, Config.PREDICTIONS_CSV_PATH, leaderboard_rows, wf_res["adaptive_168h_summary"], Config.IMAGE_DIR / "15_future_24h_forecast_and_lab4_summary.png")

    logger.info("FINAL TEST LEADERBOARD (VAR vs XGBoost vs LSTM):")
    for row in leaderboard_rows:
        logger.info(f"  #{row['rank']} {row['model_name']:<28} | RMSE={row['test_rmse']:6.2f} MW | MAE={row['test_mae']:6.2f} MW | MAPE={row['test_mape']:5.2f}%")
    logger.info("=" * 92)
    logger.info(f"LAB 4 PIPELINE COMPLETED IN {time.time() - t_start:.2f} SECONDS!")
    logger.info("=" * 92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 4 Multivariate Time Series Pipeline")
    parser.parse_args()
    run_pipeline()
