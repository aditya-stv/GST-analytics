"""
forecasting.py - National GST forecasting with rolling-origin backtests.

Target: national *core* GST (CGST + SGST + IGST). Cess is left out because it
is corrupted in the 2025-26 source rows and is being phased out, so it cannot
be validated out of sample.

Four models are compared on fiscal-year holdouts, then refit on the full
window (Aug 2017 - Mar 2025) to forecast FY 2025-26. The two FY 2025-26 months
present in the source (Aug 2025, Jan 2026) serve as a true out-of-sample check.
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

COVID_LOCKDOWN = ("2020-04-01", "2020-06-01")
HOLDOUT_FYS = {"2022-23": "2022-03-01", "2023-24": "2023-03-01", "2024-25": "2024-03-01"}
MODELS = ["Seasonal naive", "Seasonal naive + drift", "Holt-Winters (log, damped)", "SARIMA (log) + lockdown dummy"]


def _lockdown(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"lockdown": index.to_series().between(*COVID_LOCKDOWN).astype(float).values}, index=index)


def _forecast(train: pd.Series, horizon: int, model: str, alpha: float = 0.2) -> pd.DataFrame:
    """Returns a frame with columns forecast / lower / upper for `horizon` months after `train`."""
    idx = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    out = pd.DataFrame(index=idx, columns=["forecast", "lower", "upper"], dtype=float)
    last_year = train.iloc[-12:].values
    reps = int(np.ceil(horizon / 12))

    if model == "Seasonal naive":
        out["forecast"] = np.tile(last_year, reps)[:horizon]
    elif model == "Seasonal naive + drift":
        growth = train.iloc[-12:].sum() / train.iloc[-24:-12].sum()
        out["forecast"] = np.tile(last_year, reps)[:horizon] * growth
    elif model == "Holt-Winters (log, damped)":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(np.log(train), trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=12, initialization_method="estimated").fit()
        out["forecast"] = np.exp(fit.forecast(horizon).values)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(np.log(train), exog=_lockdown(train.index), order=(1, 1, 1),
                          seasonal_order=(0, 1, 1, 12)).fit(disp=False)
            pred = fit.get_forecast(horizon, exog=_lockdown(idx))
        ci = pred.conf_int(alpha=alpha)
        out["forecast"] = np.exp(pred.predicted_mean.values)
        out["lower"] = np.exp(ci.iloc[:, 0].values)
        out["upper"] = np.exp(ci.iloc[:, 1].values)
    return out


def backtest(series: pd.Series) -> pd.DataFrame:
    """Rolling-origin backtest: train to March of year t, forecast the 12 months of FY t+1."""
    rows = []
    for fy, cutoff in HOLDOUT_FYS.items():
        train = series[:cutoff]
        test = series[pd.Timestamp(cutoff) + pd.DateOffset(months=1):].iloc[:12]
        for model in MODELS:
            fc = _forecast(train, len(test), model)["forecast"]
            ratio = test.values / fc.values
            rows.append({
                "holdout_fy": fy, "model": model,
                "mape_pct": float(np.mean(np.abs(test.values - fc.values) / test.values) * 100),
                "annual_error_pct": float((fc.sum() / test.sum() - 1) * 100),
                "ratios": ratio,
            })
    return pd.DataFrame(rows)


def run_forecasting(national_all: pd.DataFrame, analysis_end: pd.Timestamp) -> Dict:
    """
    national_all: national monthly frame including the post-window months.
    Returns backtest table, FY 2025-26 forecast and the out-of-sample check.
    """
    s = national_all.set_index("date")["core_gst"]
    history = s[:analysis_end].asfreq("MS")
    actual_post = s[analysis_end + pd.DateOffset(months=1):].dropna()

    bt = backtest(history)
    summary = (
        bt.assign(abs_annual=bt["annual_error_pct"].abs())
          .groupby("model").agg(mean_mape_pct=("mape_pct", "mean"), worst_mape_pct=("mape_pct", "max"),
                                mean_abs_annual_error_pct=("abs_annual", "mean"))
          .reindex(MODELS).reset_index()
    )

    # Primary model = lowest mean backtest MAPE. Its 80% interval is split-conformal: the
    # 80th percentile of absolute percentage error across all 36 holdout months, so the band
    # reflects how wrong the model has actually been rather than a distributional assumption.
    primary = summary.sort_values("mean_mape_pct").iloc[0]["model"]
    ratios = np.concatenate(bt.loc[bt["model"] == primary, "ratios"].values)
    q80 = float(np.quantile(np.abs(ratios - 1), 0.80))
    fc = _forecast(history, 12, primary)
    fc["lower"], fc["upper"] = fc["forecast"] * (1 - q80), fc["forecast"] * (1 + q80)
    all_fc = pd.DataFrame({m: _forecast(history, 12, m)["forecast"] for m in MODELS})
    validation = pd.DataFrame({"actual": actual_post, "forecast": fc["forecast"].reindex(actual_post.index),
                               "lower_80": fc["lower"].reindex(actual_post.index),
                               "upper_80": fc["upper"].reindex(actual_post.index)})
    validation["error_pct"] = (validation["actual"] / validation["forecast"] - 1) * 100
    validation["inside_80_pi"] = validation["actual"].between(validation["lower_80"], validation["upper_80"])

    return {
        "history": history,
        "backtest": bt.drop(columns="ratios"),
        "all_forecasts": all_fc,
        "backtest_summary": summary,
        "primary_model": primary,
        "interval_halfwidth_pct": q80 * 100,
        "holdout_bias_pct": float((np.median(ratios) - 1) * 100),
        "forecast": fc,
        "fy26_forecast_total": float(fc["forecast"].sum()),
        "fy25_actual_total": float(history["2024-04-01":"2025-03-01"].sum()),
        "validation": validation,
    }
