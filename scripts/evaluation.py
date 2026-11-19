"""
Quantitative evaluation metrics (RMSE, MAE, MAPE, R2) and Ljung-Box
residual diagnostics for multivariate electricity load forecasting.
"""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (%) safely avoiding zero division."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = y_t != 0
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def evaluate_multivariate_forecast(
    y_true_mw: Union[pd.Series, np.ndarray],
    y_pred_mw: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Return RMSE, MAE, MAPE (%), and R2 evaluated in original MW units."""
    y_t = np.asarray(y_true_mw, dtype=float)
    y_p = np.asarray(y_pred_mw, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "MAPE": calculate_mape(y_t, y_p),
        "R2": float(r2_score(y_t, y_p)),
    }


def residual_diagnostics(
    y_true_mw: Union[pd.Series, np.ndarray],
    y_pred_mw: Union[pd.Series, np.ndarray],
    lags: int = 24,
) -> Dict[str, Any]:
    """Compute residual bias, dispersion, and Ljung-Box autocorrelation statistics."""
    residuals = np.asarray(y_true_mw, dtype=float) - np.asarray(y_pred_mw, dtype=float)
    lb_df = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = float(lb_df["lb_pvalue"].iloc[0])
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "ljung_box_stat": float(lb_df["lb_stat"].iloc[0]),
        "ljung_box_pvalue": p_value,
        "is_white_noise": bool(p_value > 0.05),
    }
