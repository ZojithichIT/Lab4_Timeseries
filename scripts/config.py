"""
Centralized configuration for Lab 4 Multivariate Time Series Forecasting
(Electricity Load + Weather: VAR vs XGBoost vs LSTM).
"""

from pathlib import Path


class Config:
    # 1. Directory & File Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    RAW_ENERGY_PATH = RAW_DATA_DIR / "energy_dataset.csv"
    RAW_WEATHER_PATH = RAW_DATA_DIR / "weather_features.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "electricity_weather_clean.csv"

    DOCS_DIR = BASE_DIR / "docs"
    IMAGE_DIR = DOCS_DIR / "images"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    SCRIPTS_DIR = BASE_DIR / "scripts"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_training.log"
    SUMMARY_REPORT_PATH = LOGS_DIR / "model_comparison_summary.txt"
    LEADERBOARD_CSV_PATH = LOGS_DIR / "metrics_leaderboard.csv"
    EXPERIMENTS_JSON_PATH = LOGS_DIR / "experiments_history.json"
    MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
    INFERENCE_LOG_PATH = LOGS_DIR / "inference.log"
    PREDICTIONS_CSV_PATH = LOGS_DIR / "latest_predictions.csv"
    SCALER_PATH = MODELS_DIR / "standard_scaler.joblib"

    MODEL_FILENAMES = {
        "VAR": "var_model.joblib",
        "XGBoost (Load-Only 27f)": "xgboost_load_only.joblib",
        "XGBoost (Multivariate 99f)": "xgboost_multivariate.joblib",
        "LSTM (64->32->1)": "lstm_multivariate.keras",
    }

    # 2. Multivariate Features & Target
    FEATURES = ["load", "temperature", "humidity", "wind_speed"]
    WEATHER_FEATURES = ["temperature", "humidity", "wind_speed"]
    TARGET_COL = "load"
    LOOKBACK = 24  # L = 24 hours
    WIND_SPEED_MAX_MS = 40.0  # Physical threshold to flag corrupt 133 m/s outliers
    ROLLING_MEAN_WINDOW = 6
    ROLLING_STD_WINDOW = 24
    TREND_LAG = 24

    # 3. Temporal Train / Val / Test Split (Strictly Sequential, shuffle=False)
    SPLIT_RATIOS = {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    }

    # 4. Model 1: VAR Hyperparameters
    VAR_MAXLAGS = 24

    # 5. Model 2: XGBoost Hyperparameters & Tuning Grid
    XGB_PARAMS = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 40,
        "random_state": 42,
        "n_jobs": -1,
    }

    XGB_TUNING_GRID = [
        {"name": "Depth=4, Trees=500", "max_depth": 4, "n_estimators": 500},
        {"name": "Depth=4, Trees=800", "max_depth": 4, "n_estimators": 800},
        {"name": "Depth=6, Trees=500", "max_depth": 6, "n_estimators": 500},
        {"name": "Depth=6, Trees=800", "max_depth": 6, "n_estimators": 800},
        {"name": "Depth=8, Trees=500", "max_depth": 8, "n_estimators": 500},
        {"name": "Depth=8, Trees=800", "max_depth": 8, "n_estimators": 800},
    ]

    # 6. Model 3: LSTM Hyperparameters
    LSTM_PARAMS = {
        "units_1": 64,
        "units_2": 32,
        "dense_units": 1,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 20,
        "patience": 4,
        "random_state": 42,
    }

    # 7. Walk-Forward Periodic Refit Simulation Settings
    WALK_FORWARD_EVAL_HOURS = 168  # 7 days
    WALK_FORWARD_REFIT_EVERY = 24  # Daily refit

    RANDOM_STATE = 42
