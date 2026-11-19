"""
Preprocessing, chronological splitting, StandardScaler normalization,
and causal multivariate feature engineering (XGBoost 2D + LSTM 3D tensors).
"""

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config


class MultivariatePreparedData(NamedTuple):
    df_clean: pd.DataFrame
    df_train_raw: pd.DataFrame
    df_val_raw: pd.DataFrame
    df_test_raw: pd.DataFrame
    df_train_scaled: pd.DataFrame
    df_val_scaled: pd.DataFrame
    df_test_scaled: pd.DataFrame
    df_all_scaled: pd.DataFrame
    scaler: StandardScaler
    # XGBoost 99-feature supervised tables
    X_xgb_train: pd.DataFrame
    X_xgb_val: pd.DataFrame
    X_xgb_test: pd.DataFrame
    # XGBoost Load-Only 27-feature ablation tables
    X_xgb_load_train: pd.DataFrame
    X_xgb_load_val: pd.DataFrame
    X_xgb_load_test: pd.DataFrame
    # LSTM 3D tensors (samples, 24, 4)
    X_lstm_train: np.ndarray
    X_lstm_val: np.ndarray
    X_lstm_test: np.ndarray
    # Aligned targets (scaled and original MW)
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_train_mw: np.ndarray
    y_val_mw: np.ndarray
    y_test_mw: np.ndarray
    idx_train: pd.DatetimeIndex
    idx_val: pd.DatetimeIndex
    idx_test: pd.DatetimeIndex
    split_summary: Dict[str, Dict[str, Any]]


class MultivariateFeatureBuilder:
    """Prepares leak-free scaled splits and aligned 2D/3D feature windows."""

    @staticmethod
    def inverse_transform_load(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """Convert standardized load values back to original MW scale."""
        load_mean = float(scaler.mean_[0])
        load_std = float(scaler.scale_[0])
        return np.asarray(y_scaled, dtype=float) * load_std + load_mean

    @staticmethod
    def get_xgb_feature_names(
        features: List[str] = Config.FEATURES,
        lookback: int = Config.LOOKBACK,
    ) -> Tuple[List[str], List[str]]:
        """Return ordered column names for full 99-feature XGBoost and 27-feature Load-Only baseline."""
        lag_cols: List[str] = []
        for k in range(1, lookback + 1):
            for col in features:
                lag_cols.append(f"{col}_lag_{k}")
        rolling_cols = [
            f"rolling_mean_{Config.ROLLING_MEAN_WINDOW}",
            f"rolling_std_{Config.ROLLING_STD_WINDOW}",
            f"load_trend_{Config.TREND_LAG}",
        ]
        full_cols = lag_cols + rolling_cols
        load_only_cols = [f"load_lag_{k}" for k in range(1, lookback + 1)] + rolling_cols
        return full_cols, load_only_cols

    @staticmethod
    def _build_windows_from_scaled(
        all_scaled_values: np.ndarray,
        all_raw_load: np.ndarray,
        all_index: pd.DatetimeIndex,
        start_pos: int,
        end_pos: int,
        full_cols: List[str],
        lookback: int = Config.LOOKBACK,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Build causal 2D XGBoost features and 3D LSTM tensors for target positions [start_pos, end_pos)."""
        n_vars = all_scaled_values.shape[1]
        n_samples = end_pos - start_pos
        n_xgb_features = lookback * n_vars + 3

        X_2d = np.empty((n_samples, n_xgb_features), dtype=np.float32)
        X_3d = np.empty((n_samples, lookback, n_vars), dtype=np.float32)
        y_scaled = np.empty(n_samples, dtype=np.float32)
        y_mw = np.empty(n_samples, dtype=np.float64)

        for i, pos in enumerate(range(start_pos, end_pos)):
            # 24 steps: [pos - 24, ..., pos - 1] -> t-23 .. t predicting t+1 (pos)
            window = all_scaled_values[pos - lookback : pos]
            X_3d[i] = window

            # Lag features ordered k=1 (t, pos-1) .. k=24 (t-23, pos-24)
            rev_window = window[::-1].reshape(-1)
            load_recent_6 = all_scaled_values[pos - Config.ROLLING_MEAN_WINDOW : pos, 0]
            load_recent_24 = all_scaled_values[pos - Config.ROLLING_STD_WINDOW : pos, 0]
            roll_mean_6 = float(np.mean(load_recent_6))
            roll_std_24 = float(np.std(load_recent_24, ddof=1))
            # load_t - load_{t-24} = scaled[pos-1] - scaled[pos-25]
            load_trend_24 = float(all_scaled_values[pos - 1, 0] - all_scaled_values[pos - 1 - Config.TREND_LAG, 0])

            X_2d[i, : lookback * n_vars] = rev_window
            X_2d[i, lookback * n_vars :] = [roll_mean_6, roll_std_24, load_trend_24]

            y_scaled[i] = float(all_scaled_values[pos, 0])
            y_mw[i] = float(all_raw_load[pos])

        target_index = all_index[start_pos:end_pos]
        df_xgb = pd.DataFrame(X_2d, index=target_index, columns=full_cols)
        return df_xgb, X_3d, y_scaled, y_mw, target_index

    @classmethod
    def prepare_all(
        cls,
        df_clean: pd.DataFrame,
        ratios: Dict[str, float] = Config.SPLIT_RATIOS,
        lookback: int = Config.LOOKBACK,
        export_dir: Optional[Path] = Config.PROCESSED_DATA_DIR,
        scaler_path: Optional[Path] = Config.SCALER_PATH,
    ) -> MultivariatePreparedData:
        """
        Split chronologically (70/15/15), fit StandardScaler only on Train,
        export scaled CSVs, and construct aligned 2D & 3D tensors.
        """
        n_total = len(df_clean)
        train_end = int(n_total * ratios.get("train", 0.70))
        val_size = int(round(n_total * ratios.get("val", 0.15)))
        val_end = train_end + val_size

        df_train_raw = df_clean.iloc[:train_end].copy()
        df_val_raw = df_clean.iloc[train_end:val_end].copy()
        df_test_raw = df_clean.iloc[val_end:].copy()

        scaler = StandardScaler()
        scaler.fit(df_train_raw[Config.FEATURES].values)

        df_train_scaled = pd.DataFrame(
            scaler.transform(df_train_raw[Config.FEATURES].values),
            index=df_train_raw.index,
            columns=Config.FEATURES,
        )
        df_val_scaled = pd.DataFrame(
            scaler.transform(df_val_raw[Config.FEATURES].values),
            index=df_val_raw.index,
            columns=Config.FEATURES,
        )
        df_test_scaled = pd.DataFrame(
            scaler.transform(df_test_raw[Config.FEATURES].values),
            index=df_test_raw.index,
            columns=Config.FEATURES,
        )
        df_all_scaled = pd.concat([df_train_scaled, df_val_scaled, df_test_scaled])

        if export_dir is not None:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            df_train_scaled.to_csv(export_dir / "train_scaled.csv")
            df_val_scaled.to_csv(export_dir / "val_scaled.csv")
            df_test_scaled.to_csv(export_dir / "test_scaled.csv")

        if scaler_path is not None:
            scaler_path = Path(scaler_path)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)

        full_cols, load_only_cols = cls.get_xgb_feature_names(Config.FEATURES, lookback)
        all_scaled_arr = df_all_scaled.values
        all_raw_load = df_clean[Config.TARGET_COL].values
        all_idx = df_clean.index

        # Need pos - 1 - TREND_LAG >= 0 -> start_pos = max(lookback, TREND_LAG + 1) = 25
        min_pos = max(lookback, Config.TREND_LAG + 1)
        X_xgb_tr, X_lstm_tr, y_tr_sc, y_tr_mw, idx_tr = cls._build_windows_from_scaled(
            all_scaled_arr, all_raw_load, all_idx, min_pos, train_end, full_cols, lookback
        )
        X_xgb_va, X_lstm_va, y_va_sc, y_va_mw, idx_va = cls._build_windows_from_scaled(
            all_scaled_arr, all_raw_load, all_idx, train_end, val_end, full_cols, lookback
        )
        X_xgb_te, X_lstm_te, y_te_sc, y_te_mw, idx_te = cls._build_windows_from_scaled(
            all_scaled_arr, all_raw_load, all_idx, val_end, n_total, full_cols, lookback
        )

        split_summary = {
            "Train": {
                "ratio": ratios["train"],
                "raw_rows": len(df_train_raw),
                "sample_rows": len(idx_tr),
                "start": str(df_train_raw.index.min()),
                "end": str(df_train_raw.index.max()),
                "load_mean_mw": float(df_train_raw["load"].mean()),
                "load_std_mw": float(df_train_raw["load"].std()),
            },
            "Validation": {
                "ratio": ratios["val"],
                "raw_rows": len(df_val_raw),
                "sample_rows": len(idx_va),
                "start": str(df_val_raw.index.min()),
                "end": str(df_val_raw.index.max()),
                "load_mean_mw": float(df_val_raw["load"].mean()),
                "load_std_mw": float(df_val_raw["load"].std()),
            },
            "Test": {
                "ratio": ratios["test"],
                "raw_rows": len(df_test_raw),
                "sample_rows": len(idx_te),
                "start": str(df_test_raw.index.min()),
                "end": str(df_test_raw.index.max()),
                "load_mean_mw": float(df_test_raw["load"].mean()),
                "load_std_mw": float(df_test_raw["load"].std()),
            },
        }

        return MultivariatePreparedData(
            df_clean=df_clean,
            df_train_raw=df_train_raw,
            df_val_raw=df_val_raw,
            df_test_raw=df_test_raw,
            df_train_scaled=df_train_scaled,
            df_val_scaled=df_val_scaled,
            df_test_scaled=df_test_scaled,
            df_all_scaled=df_all_scaled,
            scaler=scaler,
            X_xgb_train=X_xgb_tr,
            X_xgb_val=X_xgb_va,
            X_xgb_test=X_xgb_te,
            X_xgb_load_train=X_xgb_tr[load_only_cols].copy(),
            X_xgb_load_val=X_xgb_va[load_only_cols].copy(),
            X_xgb_load_test=X_xgb_te[load_only_cols].copy(),
            X_lstm_train=X_lstm_tr,
            X_lstm_val=X_lstm_va,
            X_lstm_test=X_lstm_te,
            y_train_scaled=y_tr_sc,
            y_val_scaled=y_va_sc,
            y_test_scaled=y_te_sc,
            y_train_mw=y_tr_mw,
            y_val_mw=y_va_mw,
            y_test_mw=y_te_mw,
            idx_train=idx_tr,
            idx_val=idx_va,
            idx_test=idx_te,
            split_summary=split_summary,
        )
