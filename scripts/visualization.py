"""
Visualization module generating 10 diagnostic, EDA, model training,
walk-forward backtest, and analytical comparison plots for Lab 4.
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_eda_timeseries(df_clean: pd.DataFrame, save_path: Path) -> None:
    """01: Full 4-year Load & Weather series with Mean/Max/Min thresholds + 2-week zoom (Lab 3 style)."""
    fig = plt.figure(figsize=(16, 13.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 1.15], hspace=0.34)

    # Panel (A): Full Electricity Load with Mean / Max / Min lines and callouts
    ax_load = fig.add_subplot(gs[0])
    s_load = df_clean["load"]
    mean_l = float(s_load.mean())
    max_l, min_l = float(s_load.max()), float(s_load.min())
    max_ts_l, min_ts_l = s_load.idxmax(), s_load.idxmin()

    ax_load.plot(s_load.index, s_load.values, color="#1f77b4", lw=0.6, alpha=0.80, label="Phụ tải thực tế hàng giờ (Load MW)")
    ax_load.axhline(mean_l, color="#d62728", ls="--", lw=1.8, label=f"Mean (Trung bình): {mean_l:,.1f} MW")
    ax_load.axhline(max_l, color="#800080", ls=":", lw=1.8, label=f"Max (Cao nhất): {max_l:,.0f} MW ({max_ts_l.strftime('%d/%m/%Y %H:%M')})")
    ax_load.axhline(min_l, color="#16a34a", ls=":", lw=1.8, label=f"Min (Thấp nhất): {min_l:,.0f} MW ({min_ts_l.strftime('%d/%m/%Y %H:%M')})")

    ax_load.annotate(
        f"Đỉnh cực đại: {max_l:,.0f} MW\n({max_ts_l.strftime('%d/%m/%Y %H:%M')})",
        xy=(max_ts_l, max_l),
        xytext=(max_ts_l - pd.Timedelta(days=480), max_l + 1500),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdf4ff", ec="#800080", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.3),
    )
    ax_load.annotate(
        f"Đáy cực tiểu: {min_l:,.0f} MW\n({min_ts_l.strftime('%d/%m/%Y %H:%M')})",
        xy=(min_ts_l, min_l),
        xytext=(min_ts_l + pd.Timedelta(days=110), min_l - 2900),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f0fdf4", ec="#16a34a", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.3),
    )
    ax_load.set_ylim(min_l - 4200, max_l + 5200)
    ax_load.set_title("(A) Chuỗi Phụ Tải Điện Năng Hàng Giờ Tây Ban Nha (2015 – 2018) & Các Đường Ngưỡng Thống Kê", fontsize=12.5, fontweight="bold", pad=8)
    ax_load.set_ylabel("Phụ tải (MW)")
    ax_load.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", ncol=2)

    # Panel (B): Full 4-year Weather Variables (Temp, Humidity, Wind) with Mean/Max/Min
    sub_gs = gs[1].subgridspec(1, 3, wspace=0.22)
    weather_specs = [
        ("temperature", "Nhiệt độ trung bình (°C)", "°C", "#d97706"),
        ("humidity", "Độ ẩm tương đối (%)", "%", "#2563eb"),
        ("wind_speed", "Tốc độ gió (m/s)", "m/s", "#0f766e"),
    ]
    for idx_w, (col, title, unit, base_col) in enumerate(weather_specs):
        ax_w = fig.add_subplot(sub_gs[0, idx_w])
        sw = df_clean[col]
        w_mean, w_max, w_min = float(sw.mean()), float(sw.max()), float(sw.min())
        w_max_ts, w_min_ts = sw.idxmax(), sw.idxmin()
        span = w_max - w_min

        ax_w.plot(sw.index, sw.values, color=base_col, lw=0.45, alpha=0.68)
        ax_w.axhline(w_mean, color="#d62728", ls="--", lw=1.4, label=f"Mean: {w_mean:.1f} {unit}")
        ax_w.axhline(w_max, color="#800080", ls=":", lw=1.4, label=f"Max: {w_max:.1f} {unit} ({w_max_ts.strftime('%d/%m/%Y')})")
        ax_w.axhline(w_min, color="#16a34a", ls=":", lw=1.4, label=f"Min: {w_min:.1f} {unit} ({w_min_ts.strftime('%d/%m/%Y')})")
        ax_w.set_ylim(w_min - span * 0.14, w_max + span * 0.40)
        ax_w.set_title(f"(B{idx_w+1}) {title}", fontsize=10.5, fontweight="bold")
        ax_w.set_ylabel(unit)
        ax_w.xaxis.set_major_locator(mdates.YearLocator())
        ax_w.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_w.tick_params(axis="x", labelsize=8.5)
        ax_w.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="white")

    # Panel (C): Two-Week Zoom (02/07/2018 - 15/07/2018, 2 Full Weeks) matching Lab 3 02_aep_zoomin_2weeks.png
    ax_z = fig.add_subplot(gs[2])
    z_raw = df_clean.loc["2018-07-02":"2018-07-15", "load"]
    z_2h = z_raw.resample("2h").mean()
    wkday_mean = float(z_raw[z_raw.index.dayofweek < 5].mean())
    wkend_mean = float(z_raw[z_raw.index.dayofweek >= 5].mean())

    ax_z.plot(z_raw.index, z_raw.values, color="#cbd5e1", lw=1.2, label="Dữ liệu gốc 1h/điểm (336 điểm)")
    ax_z.plot(z_2h.index, z_2h.values, color="#d95f02", lw=1.8, marker="o", ms=3.0, label="Chuỗi nén 2h/điểm (168 điểm)")
    ax_z.axhline(wkday_mean, color="#2563eb", ls="--", lw=1.5, label=f"TB Ngày thường (T2–T6): {wkday_mean:,.0f} MW")
    ax_z.axhline(wkend_mean, color="#16a34a", ls="-.", lw=1.5, label=f"TB Cuối tuần (T7–CN): {wkend_mean:,.0f} MW (Giảm {wkend_mean - wkday_mean:,.0f} MW)")

    for wk_start, wk_end in [("2018-07-07 00:00:00+00:00", "2018-07-09 00:00:00+00:00"), ("2018-07-14 00:00:00+00:00", "2018-07-16 00:00:00+00:00")]:
        ax_z.axvspan(pd.Timestamp(wk_start), pd.Timestamp(wk_end), color="#e2e8f0", alpha=0.55)

    z_max_ts, z_min_ts = z_raw.idxmax(), z_raw.idxmin()
    ax_z.annotate(
        f"Đỉnh 2 tuần: {z_raw.max():,.0f} MW",
        xy=(z_max_ts, z_raw.max()),
        xytext=(z_max_ts + pd.Timedelta(hours=18), z_raw.max() + 700),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec="#d95f02", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.1),
    )
    ax_z.annotate(
        f"Đáy cuối tuần: {z_raw.min():,.0f} MW",
        xy=(z_min_ts, z_raw.min()),
        xytext=(z_min_ts - pd.Timedelta(days=1.5), z_raw.min() - 1800),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#16a34a", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.1),
    )
    ax_z.set_ylim(z_raw.min() - 2800, z_raw.max() + 2600)
    ax_z.set_title("(C) Thu Phóng 14 Ngày Liên Tiếp (02/07 – 15/07/2018): Nhịp Điệu Ngày & Sụt Giảm Cuối Tuần (T7–CN)", fontsize=12, fontweight="bold", pad=8)
    ax_z.set_ylabel("Phụ tải (MW)")
    ax_z.legend(loc="upper right", fontsize=8.3, frameon=True, facecolor="white", ncol=2)

    _save_fig(fig, save_path)


def plot_eda_load_temp_and_corr(df_clean: pd.DataFrame, save_path: Path) -> None:
    """02: Load vs Temperature non-linear scatter + Pearson Correlation Matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    sample = df_clean.sample(min(8000, len(df_clean)), random_state=42)
    axes[0].scatter(sample["temperature"], sample["load"], s=8, alpha=0.25, color="#1f4e79")
    # Polynomial trend curve highlighting non-linear U-shape relationship
    z = np.polyfit(sample["temperature"], sample["load"], 2)
    p = np.poly1d(z)
    xs = np.linspace(sample["temperature"].min(), sample["temperature"].max(), 200)
    axes[0].plot(xs, p(xs), color="#dc2626", lw=2.5, label="Quadratic Fit (Non-linear)")
    axes[0].set_title("Load vs Temperature (Heating + Cooling Effect)", fontweight="bold")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Electricity Load (MW)")
    axes[0].legend()

    corr = df_clean.corr(method="pearson")
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=axes[1],
    )
    axes[1].set_title("Pearson Correlation Matrix (Multivariate Variables)", fontweight="bold")

    _save_fig(fig, save_path)


def plot_seasonality_and_split(
    df_clean: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    save_path: Path,
) -> None:
    """03: Hourly/Weekly seasonality profiles + Chronological 70/15/15 Split."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax_split = fig.add_subplot(gs[0, :])
    ax_split.plot(df_train.index, df_train["load"], color="#1f4e79", lw=0.6, label=f"Train (70% · {len(df_train):,}h)")
    ax_split.plot(df_val.index, df_val["load"], color="#7c3aed", lw=0.6, label=f"Validation (15% · {len(df_val):,}h)")
    ax_split.plot(df_test.index, df_test["load"], color="#d97706", lw=0.6, label=f"Test (15% · {len(df_test):,}h)")
    ax_split.set_title("Chronological Train / Validation / Test Split (70% / 15% / 15%, Shuffle=False)", fontweight="bold")
    ax_split.set_ylabel("Load (MW)")
    ax_split.legend(loc="upper left", ncol=3)

    ax_hr = fig.add_subplot(gs[1, 0])
    hr_mean = df_clean.groupby(df_clean.index.hour)["load"].mean()
    ax_hr.plot(hr_mean.index, hr_mean.values, marker="o", color="#0f766e", lw=2)
    ax_hr.set_title("Diurnal Cycle: Average Load by Hour of Day", fontweight="bold")
    ax_hr.set_xlabel("Hour (UTC)")
    ax_hr.set_ylabel("Mean Load (MW)")
    ax_hr.set_xticks(range(0, 24, 3))

    ax_dow = fig.add_subplot(gs[1, 1])
    dow_mean = df_clean.groupby(df_clean.index.dayofweek)["load"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#1f4e79"] * 5 + ["#d97706", "#d97706"]
    ax_dow.bar(days, dow_mean.values, color=colors)
    ax_dow.set_title("Weekly Seasonality: Weekday vs Weekend Load", fontweight="bold")
    ax_dow.set_ylabel("Mean Load (MW)")
    ax_dow.set_ylim(dow_mean.min() * 0.9, dow_mean.max() * 1.05)

    _save_fig(fig, save_path)


def plot_var_stationarity_and_lag_selection(
    var_info: Dict[str, Any],
    save_path: Path,
) -> None:
    """04: VAR AIC/BIC lag selection curve and ADF test p-values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    lags = var_info["lag_grid"]
    aics = var_info["aic_values"]
    bics = var_info["bic_values"]

    axes[0].plot(lags, aics, marker="o", ms=4, color="#1f4e79", label="AIC")
    axes[0].plot(lags, bics, marker="s", ms=4, color="#d97706", label="BIC")
    axes[0].axvline(var_info["selected_lag_aic"], color="#1f4e79", ls="--", alpha=0.7, label=f"Best AIC p={var_info['selected_lag_aic']}")
    axes[0].axvline(var_info["selected_lag_bic"], color="#d97706", ls=":", alpha=0.8, label=f"Best BIC p={var_info['selected_lag_bic']}")
    axes[0].set_title("VAR Lag Order Selection (AIC vs BIC)", fontweight="bold")
    axes[0].set_xlabel("Lag Order p (Hours)")
    axes[0].set_ylabel("Information Criterion")
    axes[0].legend(fontsize=8.5)

    adf_vars = list(var_info["adf_results"].keys())
    lvl_pvals = [var_info["adf_results"][v]["level_pvalue"] for v in adf_vars]
    diff_pvals = [var_info["adf_results"][v]["diff1_pvalue"] for v in adf_vars]
    x = np.arange(len(adf_vars))
    w = 0.36
    axes[1].bar(x - w / 2, lvl_pvals, width=w, color="#2563eb", label="Level Series")
    axes[1].bar(x + w / 2, diff_pvals, width=w, color="#16a34a", label="First Difference")
    axes[1].axhline(0.05, color="#dc2626", ls="--", lw=1.5, label="alpha = 0.05")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(adf_vars)
    axes[1].set_title("ADF Stationarity Test p-values (Before Training VAR)", fontweight="bold")
    axes[1].set_ylabel("ADF p-value")
    axes[1].legend()

    _save_fig(fig, save_path)


def plot_xgboost_tuning_and_importance(
    tuning_results: List[Dict[str, Any]],
    group_importance: Dict[str, float],
    top_features: pd.Series,
    save_path: Path,
) -> None:
    """05: XGBoost hyperparameter grid + variable group Gain + Top 15 features."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    names = [r["name"] for r in tuning_results]
    val_rmses = [r["val_rmse_mw"] for r in tuning_results]
    best_idx = int(np.argmin(val_rmses))
    bar_cols = ["#16a34a" if i == best_idx else "#64748b" for i in range(len(names))]
    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, val_rmses, color=bar_cols)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Validation RMSE (MW)")
    axes[0].set_title("XGBoost Tuning (Early Stopping)", fontweight="bold")
    axes[0].set_xlim(min(val_rmses) * 0.96, max(val_rmses) * 1.02)

    grp_names = list(group_importance.keys())
    grp_vals = [group_importance[k] for k in grp_names]
    axes[1].barh(grp_names, grp_vals, color=["#1f4e79", "#d97706", "#0f766e", "#2563eb", "#7c3aed"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Share of Total Split Gain (%)")
    axes[1].set_title("XGBoost Gain Share by Feature Group", fontweight="bold")

    top12 = top_features.head(12)
    axes[2].barh(top12.index, top12.values, color="#1f4e79")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Gain Score")
    axes[2].set_title("Top 12 Features in XGBoost (99f)", fontweight="bold")
    axes[2].tick_params(axis="y", labelsize=8)

    _save_fig(fig, save_path)


def plot_lstm_learning_curves(
    history_dict: Dict[str, List[float]],
    scaler_std_load: float,
    save_path: Path,
) -> None:
    """06: LSTM Train vs Validation loss (scaled MSE & converted MW RMSE)."""
    epochs = np.arange(1, len(history_dict["loss"]) + 1)
    tr_rmse = np.sqrt(np.asarray(history_dict["loss"])) * scaler_std_load
    va_rmse = np.sqrt(np.asarray(history_dict["val_loss"])) * scaler_std_load

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(epochs, history_dict["loss"], marker="o", ms=4, color="#1f4e79", label="Train Loss (MSE)")
    axes[0].plot(epochs, history_dict["val_loss"], marker="s", ms=4, color="#d97706", label="Val Loss (MSE)")
    axes[0].set_title("LSTM Learning Curve: Scaled MSE Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (Standardized)")
    axes[0].legend()

    axes[1].plot(epochs, tr_rmse, marker="o", ms=4, color="#1f4e79", label="Train RMSE (MW)")
    axes[1].plot(epochs, va_rmse, marker="s", ms=4, color="#d97706", label="Val RMSE (MW)")
    best_ep = int(np.argmin(va_rmse)) + 1
    axes[1].axvline(best_ep, color="#16a34a", ls="--", label=f"Best Epoch = {best_ep}")
    axes[1].set_title("LSTM Overfitting Check: RMSE in MW", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (MW)")
    axes[1].legend()

    _save_fig(fig, save_path)


def plot_models_comparison_metrics(
    leaderboard: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    """07: Side-by-side RMSE, MAE, MAPE comparison across VAR, XGBoost, LSTM."""
    df_lb = pd.DataFrame(leaderboard)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [("test_rmse", "Test RMSE (MW)"), ("test_mae", "Test MAE (MW)"), ("test_mape", "Test MAPE (%)")]
    colors = ["#16a34a" if "Multivariate" in n else ("#2563eb" if "LSTM" in n else ("#d97706" if "VAR" in n else "#94a3b8")) for n in df_lb["model_name"]]

    for ax, (col, title) in zip(axes, metrics):
        bars = ax.barh(df_lb["model_name"], df_lb[col], color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        for b, val in zip(bars, df_lb[col]):
            fmt = f"{val:.2f}%" if col == "test_mape" else f"{val:.1f}"
            ax.text(val + df_lb[col].max() * 0.01, b.get_y() + b.get_height() / 2, fmt, va="center", fontsize=8.5)
        ax.set_xlim(0, df_lb[col].max() * 1.18)

    _save_fig(fig, save_path)


def plot_predictions_vs_actual(
    idx_test: pd.DatetimeIndex,
    y_test_mw: np.ndarray,
    preds_map: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """08: Actual vs VAR / XGBoost / LSTM predictions over 1-week Test window + full test scatter."""
    window_n = min(168, len(idx_test))
    ts = idx_test[:window_n]
    y_true_w = y_test_mw[:window_n]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8))
    axes[0].plot(ts, y_true_w, color="#0f172a", lw=2.0, label="Actual Load (MW)")
    palette = {"VAR": "#d97706", "XGBoost (Multivariate 99f)": "#16a34a", "LSTM (64->32->1)": "#2563eb"}
    styles = {"VAR": ":", "XGBoost (Multivariate 99f)": "-", "LSTM (64->32->1)": "--"}
    for name, col in palette.items():
        if name in preds_map:
            axes[0].plot(ts, preds_map[name][:window_n], color=col, ls=styles[name], lw=1.6, label=name)
    axes[0].set_title("First 7 Days of Test Set: Actual vs VAR, XGBoost & LSTM", fontweight="bold")
    axes[0].set_ylabel("Load (MW)")
    axes[0].legend(ncol=4, fontsize=9)

    for name, col in palette.items():
        if name in preds_map:
            err = y_test_mw - preds_map[name]
            axes[1].hist(err, bins=60, alpha=0.45, color=col, label=f"{name} (std={np.std(err):.1f} MW)")
    axes[1].axvline(0, color="#0f172a", lw=1.2, ls="--")
    axes[1].set_title("Test Set Residual Distribution Comparison (e_t = Actual - Predicted)", fontweight="bold")
    axes[1].set_xlabel("Prediction Error (MW)")
    axes[1].set_ylabel("Frequency (Hours)")
    axes[1].legend()

    _save_fig(fig, save_path)


def plot_walk_forward_backtest(
    wf_results: Dict[str, Any],
    save_path: Path,
) -> None:
    """09: Walk-Forward Validation rolling daily RMSE & 168h adaptive vs static comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    daily_df: pd.DataFrame = wf_results["daily_rmse_df"]
    palette = {"VAR": "#d97706", "XGBoost (Multivariate 99f)": "#16a34a", "LSTM (64->32->1)": "#2563eb"}
    for col_name, color in palette.items():
        if col_name in daily_df.columns:
            axes[0].plot(daily_df.index, daily_df[col_name], lw=1.2, alpha=0.85, color=color, label=col_name)
    axes[0].set_title("Walk-Forward Validation: Weekly Rolling RMSE Over Entire Test Set", fontweight="bold")
    axes[0].set_xlabel("Date (Test Period)")
    axes[0].set_ylabel("7-Day Rolling RMSE (MW)")
    axes[0].legend(fontsize=8.5)

    adaptive = wf_results["adaptive_168h_summary"]
    models = list(adaptive.keys())
    static_vals = [adaptive[m]["static_rmse"] for m in models]
    refit_vals = [adaptive[m]["periodic_refit_rmse"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    axes[1].bar(x - w / 2, static_vals, width=w, color="#64748b", label="Static Expanding Window")
    axes[1].bar(x + w / 2, refit_vals, width=w, color="#0f766e", label="Daily Periodic Refit (24h)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=12, fontsize=9)
    axes[1].set_title("168h Backtest Simulation: Static vs Daily Refit RMSE", fontweight="bold")
    axes[1].set_ylabel("RMSE (MW)")
    axes[1].legend()

    _save_fig(fig, save_path)


def plot_residual_and_hourly_error(
    idx_test: pd.DatetimeIndex,
    y_test_mw: np.ndarray,
    temp_test: np.ndarray,
    preds_map: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """10: Hourly RMSE profile & Temperature regime error breakdown for Q1-Q6 analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    palette = {"VAR": "#d97706", "XGBoost (Multivariate 99f)": "#16a34a", "LSTM (64->32->1)": "#2563eb"}

    df_eval = pd.DataFrame({"actual": y_test_mw, "hour": idx_test.hour, "temp": temp_test})
    df_eval["temp_bin"] = pd.cut(df_eval["temp"], bins=[-5, 10, 18, 25, 45], labels=["Cold (<10°C)", "Mild (10-18°C)", "Warm (18-25°C)", "Hot (>25°C)"])

    for name, color in palette.items():
        if name in preds_map:
            sq_err = (df_eval["actual"].values - preds_map[name]) ** 2
            df_eval[f"sq_{name}"] = sq_err
            hr_rmse = np.sqrt(df_eval.groupby("hour")[f"sq_{name}"].mean())
            axes[0].plot(hr_rmse.index, hr_rmse.values, marker="o", ms=3.5, lw=1.7, color=color, label=name)
    axes[0].set_title("Test RMSE by Hour of Day (Morning Ramp & Evening Peak)", fontweight="bold")
    axes[0].set_xlabel("Hour (UTC)")
    axes[0].set_ylabel("RMSE (MW)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=8.5)

    x = np.arange(4)
    w = 0.25
    for idx_m, (name, color) in enumerate(palette.items()):
        if name in preds_map:
            bin_rmse = np.sqrt(df_eval.groupby("temp_bin", observed=False)[f"sq_{name}"].mean())
            axes[1].bar(x + (idx_m - 1) * w, bin_rmse.values, width=w, color=color, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Cold (<10°C)", "Mild (10-18°C)", "Warm (18-25°C)", "Hot (>25°C)"], fontsize=8.5)
    axes[1].set_title("Test RMSE Across Temperature Regimes (Weather Impact)", fontweight="bold")
    axes[1].set_ylabel("RMSE (MW)")
    axes[1].legend(fontsize=8.5)

    _save_fig(fig, save_path)


def plot_missing_and_outliers_audit(
    raw_energy_path: Path,
    raw_weather_path: Path,
    df_clean: pd.DataFrame,
    save_path: Path,
) -> None:
    """11: Data Quality Audit (Boxplots + Wind Outliers >40 m/s & Interpolated Load NaNs)."""
    df_e = pd.read_csv(raw_energy_path, usecols=["time", "total load actual"])
    df_w = pd.read_csv(raw_weather_path, usecols=["dt_iso", "wind_speed"])
    df_e["dt"] = pd.to_datetime(df_e["time"], utc=True)
    df_w["dt"] = pd.to_datetime(df_w["dt_iso"], utc=True)
    raw_wind_hourly = df_w.groupby("dt")["wind_speed"].max()
    nan_load_dt = df_e.loc[df_e["total load actual"].isna(), "dt"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    z_df = (df_clean - df_clean.mean()) / df_clean.std()
    sns.boxplot(data=z_df, ax=axes[0], palette=["#1f4e79", "#d97706", "#2563eb", "#0f766e"])
    axes[0].axhline(0, color="#64748b", ls="--", lw=0.8)
    axes[0].set_title("Standardized Feature Distributions & IQR Outliers (Cleaned)", fontweight="bold")
    axes[0].set_ylabel("Z-Score (Standard Deviations)")

    axes[1].plot(raw_wind_hourly.index, raw_wind_hourly.values, color="#dc2626", lw=0.8, alpha=0.75, label="Raw City-Max Wind Speed")
    axes[1].plot(df_clean.index, df_clean["wind_speed"], color="#0f766e", lw=0.8, label="Cleaned Hourly Wind Mean")
    axes[1].axhline(40, color="#0f172a", ls="--", lw=1.1, label="Physical Threshold (40 m/s)")
    axes[1].set_title(f"Outlier Removal (Wind > 40 m/s up to {raw_wind_hourly.max():.0f} m/s) & {len(nan_load_dt)} Load NaNs Fixed", fontweight="bold")
    axes[1].set_ylabel("Wind Speed (m/s)")
    axes[1].legend(fontsize=8.5, loc="upper left")

    _save_fig(fig, save_path)


def plot_standard_scaler_distributions(
    df_train_raw: pd.DataFrame,
    df_train_scaled: pd.DataFrame,
    save_path: Path,
) -> None:
    """12: Raw unit distributions vs StandardScaler (mu=0, sigma=1) on Train Set."""
    cols = list(df_train_raw.columns)
    units = ["MW", "°C", "%", "m/s"]
    colors = ["#1f4e79", "#d97706", "#2563eb", "#0f766e"]

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8))
    for j, (col, unit, color) in enumerate(zip(cols, units, colors)):
        axes[0, j].hist(df_train_raw[col], bins=45, color=color, alpha=0.8)
        axes[0, j].set_title(f"Raw {col} ({unit})", fontweight="bold", fontsize=10)
        axes[0, j].set_xlabel(f"μ={df_train_raw[col].mean():.1f}, σ={df_train_raw[col].std():.1f}", fontsize=8.5)

        axes[1, j].hist(df_train_scaled[col], bins=45, color=color, alpha=0.65, edgecolor="#0f172a", linewidth=0.2)
        axes[1, j].axvline(0, color="#dc2626", ls="--", lw=1.1)
        axes[1, j].set_title(f"Scaled {col} (Train)", fontweight="bold", fontsize=10)
        axes[1, j].set_xlabel(f"μ={df_train_scaled[col].mean():.2f}, σ={df_train_scaled[col].std(ddof=0):.2f}", fontsize=8.5)

    axes[0, 0].set_ylabel("Raw Frequency", fontweight="bold")
    axes[1, 0].set_ylabel("Scaled Frequency", fontweight="bold")
    fig.suptitle("StandardScaler Normalization (Fit on Train Only -> Mean=0, Std=1)", fontweight="bold", fontsize=12, y=1.01)

    _save_fig(fig, save_path)


def plot_rolling_and_trend_features(df_clean: pd.DataFrame, save_path: Path) -> None:
    """13: Section 5.2 Rolling Mean (6h), Rolling Std (24h), and Load Trend (t - t-24)."""
    load_ser = df_clean["load"]
    df_feat = pd.DataFrame({
        "load": load_ser,
        "rolling_mean_6": load_ser.rolling(6).mean(),
        "rolling_std_24": load_ser.rolling(24).std(),
        "load_trend_24": load_ser - load_ser.shift(24),
    }).dropna()

    win = df_feat.loc["2018-07-02":"2018-07-11"]
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.2), sharex=True)

    axes[0].plot(win.index, win["load"], color="#1f4e79", lw=1.4, label="Actual Load (MW)")
    axes[0].plot(win.index, win["rolling_mean_6"], color="#d97706", lw=1.8, ls="--", label="rolling_mean_6 (6h Moving Avg)")
    axes[0].set_title("XGBoost Rolling & Trend Features (10-Day Window, July 2018)", fontweight="bold")
    axes[0].set_ylabel("Load (MW)")
    axes[0].legend(loc="upper right")

    axes[1].plot(win.index, win["rolling_std_24"], color="#7c3aed", lw=1.7)
    axes[1].fill_between(win.index, 0, win["rolling_std_24"], color="#7c3aed", alpha=0.15)
    axes[1].set_title("rolling_std_24: Intraday Volatility (Higher on Weekdays, Lower on Weekends)", fontweight="bold")
    axes[1].set_ylabel("Std Dev (MW)")

    pos = win["load_trend_24"].clip(lower=0)
    neg = win["load_trend_24"].clip(upper=0)
    axes[2].fill_between(win.index, 0, pos, color="#16a34a", alpha=0.45, label="Positive 24h Growth (+)")
    axes[2].fill_between(win.index, 0, neg, color="#dc2626", alpha=0.45, label="Negative 24h Drop (-)")
    axes[2].axhline(0, color="#0f172a", lw=0.9)
    axes[2].set_title("load_trend_24 = load(t) - load(t-24): Day-over-Day Momentum", fontweight="bold")
    axes[2].set_ylabel("Delta Load (MW)")
    axes[2].legend(loc="upper right")

    _save_fig(fig, save_path)


def plot_xgboost_learning_and_ablation(
    xgb_model: Any,
    scaler_std_load: float,
    df_clean: pd.DataFrame,
    leaderboard_rows: List[Dict[str, Any]],
    wf_summary: Dict[str, Any],
    save_path: Path,
) -> None:
    """14: XGBoost Learning Curves + Weather Cross-Correlation (0..24h) + Weather Ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Left: Learning Curves converted to MW RMSE
    ev = xgb_model.evals_result
    tr_rmse = np.asarray(ev["validation_0"]["rmse"]) * scaler_std_load
    va_rmse = np.asarray(ev["validation_1"]["rmse"]) * scaler_std_load
    rounds = np.arange(1, len(tr_rmse) + 1)
    axes[0].plot(rounds, tr_rmse, color="#1f4e79", lw=1.6, label="Train RMSE (MW)")
    axes[0].plot(rounds, va_rmse, color="#d97706", lw=1.6, label="Val RMSE (MW)")
    if xgb_model.best_iteration is not None:
        axes[0].axvline(xgb_model.best_iteration + 1, color="#16a34a", ls="--", label=f"Best Iter = {xgb_model.best_iteration + 1}")
    axes[0].set_title("XGBoost Convergence (99 Features)", fontweight="bold")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("RMSE (MW)")
    axes[0].set_ylim(0, max(va_rmse[50], tr_rmse[50]) * 1.25)
    axes[0].legend(fontsize=8.5)

    # Middle: Lagged Cross-Correlation (0..24h) between Weather & Load
    lags = np.arange(0, 25)
    for col, color in [("temperature", "#d97706"), ("humidity", "#2563eb"), ("wind_speed", "#0f766e")]:
        ccfs = [df_clean["load"].corr(df_clean[col].shift(k)) for k in lags]
        axes[1].plot(lags, ccfs, marker="o", ms=3, lw=1.7, color=color, label=col)
    axes[1].axhline(0, color="#64748b", ls="--", lw=0.9)
    axes[1].set_title("Cross-Correlation: load(t) vs weather(t-k)", fontweight="bold")
    axes[1].set_xlabel("Lag k (Hours)")
    axes[1].set_ylabel("Pearson Correlation r")
    axes[1].legend(fontsize=8.5)

    # Right: Load-Only (27f) vs Multivariate (99f)
    lb_map = {r["model_name"]: r for r in leaderboard_rows}
    lo_rmse = lb_map["XGBoost (Load-Only 27f)"]["test_rmse"]
    mv_rmse = lb_map["XGBoost (Multivariate 99f)"]["test_rmse"]
    wf_refit_mv = wf_summary.get("XGBoost (Multivariate 99f)", {}).get("periodic_refit_rmse", mv_rmse)

    labels = ["Load-Only\n(27f Static)", "Multivariate\n(99f Static)", "Multivariate\n(168h Daily Refit)"]
    vals = [lo_rmse, mv_rmse, wf_refit_mv]
    bars = axes[2].bar(labels, vals, color=["#64748b", "#1f4e79", "#16a34a"], width=0.55)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.1f} MW", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].set_ylim(0, max(vals) * 1.18)
    axes[2].set_title("Weather Ablation & Adaptive Refit RMSE", fontweight="bold")
    axes[2].set_ylabel("Test RMSE (MW)")

    _save_fig(fig, save_path)


def plot_future_forecast_and_summary(
    df_clean: pd.DataFrame,
    predictions_csv_path: Path,
    leaderboard_rows: List[Dict[str, Any]],
    wf_summary: Dict[str, Any],
    save_path: Path,
) -> None:
    """15: Future 24h Forecast Comparison + Executive Lab 4 Summary Table."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    hist_tail = df_clean["load"].iloc[-48:]
    axes[0].plot(hist_tail.index, hist_tail.values, color="#0f172a", lw=2.0, label="Actual History (Last 48h)")

    if predictions_csv_path.exists():
        df_pred = pd.read_csv(predictions_csv_path)
        fut = df_pred[df_pred["record_type"] == "future_forecast"].copy()
        if not fut.empty:
            fut["dt"] = pd.to_datetime(fut["Datetime"])
            palette = {"XGBoost (Multivariate 99f)": "#16a34a", "LSTM (64->32->1)": "#2563eb", "VAR": "#d97706"}
            styles = {"XGBoost (Multivariate 99f)": "-", "LSTM (64->32->1)": "--", "VAR": ":"}
            for m_name, col in palette.items():
                sub = fut[fut["model_name"] == m_name].sort_values("horizon_step")
                if not sub.empty:
                    axes[0].plot(sub["dt"], sub["predicted_load_MW"], color=col, ls=styles[m_name], lw=1.9, marker="o", ms=3, label=f"24h Forecast: {m_name}")
            axes[0].axvline(hist_tail.index.max(), color="#dc2626", ls="--", lw=1.1, label="Forecast Origin")

    axes[0].set_title("Standalone Inference: 24-Hour Future Forecast Beyond Dataset End", fontweight="bold")
    axes[0].set_xlabel("UTC Timestamp")
    axes[0].set_ylabel("Electricity Load (MW)")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].axis("off")
    axes[1].set_title("Lab 4 Executive Summary: Multivariate Benchmark (Test Set 5,260h)", fontweight="bold", pad=12)
    cell_rows = []
    for r in leaderboard_rows:
        m = r["model_name"]
        wf_str = f"{wf_summary[m]['periodic_refit_rmse']:.1f}" if m in wf_summary else "-"
        cell_rows.append([
            f"#{r['rank']}",
            m,
            f"{r['test_rmse']:.2f}",
            f"{r['test_mae']:.2f}",
            f"{r['test_mape']:.2f}%",
            f"{r['test_r2']:.4f}",
            wf_str,
        ])

    tbl = axes[1].table(
        cellText=cell_rows,
        colLabels=["Rank", "Model", "RMSE", "MAE", "MAPE", "R²", "168h Refit"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.05, 1.8)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i <= 2:
            cell.set_facecolor("#eef6ff")

    _save_fig(fig, save_path)
