"""
Lab 4 Multivariate Time Series Forecasting Package.
Provides modular components for electricity load + weather forecasting:
- config: Centralized paths, hyperparameters, and split constants
- data_loader: Raw CSV loading, UTC alignment, outlier cleaning, and EDA stats
- feature_engineering: Temporal split, StandardScaler, 99-feature XGBoost table, and 3D LSTM tensors
- evaluation: RMSE, MAE, MAPE, R2, and Ljung-Box residual diagnostics
- visualization: Diagnostic and model comparison plots
- models: VAR, XGBoost, LSTM forecasters and Walk-Forward Validation
"""

from .config import Config

__all__ = ["Config"]
