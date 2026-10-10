"""
synthetic_model.py - Econometric Imputation, Precasting (Backcasting), and Forecasting Engine
Synthesizes the complete 2017-2026 timeline for GSDP and GST with P10/P90 confidence bounds.
"""

import os
import sys
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.data_engine import (
    load_and_clean_gst,
    load_and_clean_gsdp,
    aggregate_annual_gst,
    BENCHMARK_28_STATES,
    normalize_state_name,
    assign_fiscal_year
)

NATIONAL_NOMINAL_GROWTH_BENCHMARKS = {
    "2017-18": 0.110,
    "2018-19": 0.105,
    "2019-20": 0.064,
    "2020-21": -0.014, # COVID contraction
    "2021-22": 0.185, # Post-lockdown rebound
    "2022-23": 0.142,
    "2023-24": 0.106,
    "2024-25": 0.102, # Forward projection
    "2025-26": 0.098  # Forward projection
}

def impute_missing_gsdp_2024(gsdp_panel: pd.DataFrame, annual_gst: pd.DataFrame) -> pd.DataFrame:
    """Imputes 6 missing states in FY 2023-24 using state CAGR + GST growth + MoSPI national benchmark."""
    df = gsdp_panel.copy()
    piv = df.pivot(index="state", columns="fiscal_year", values="per_capita_gsdp_raw")
    
    for state in piv.index:
        if pd.isna(piv.loc[state, "2023-24"]):
            v_2020 = piv.loc[state, "2019-20"]
            v_2023 = piv.loc[state, "2022-23"]
            
            if pd.notna(v_2020) and pd.notna(v_2023) and v_2020 > 0:
                hist_cagr = (v_2023 / v_2020) ** (1.0 / 3.0) - 1.0
            else:
                hist_cagr = NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2023-24"]
                
            gst_state = annual_gst[annual_gst["state"] == state].set_index("fiscal_year")
            gst_growth = 0.11
            if "2022-23" in gst_state.index and "2023-24" in gst_state.index:
                g22 = gst_state.loc["2022-23", "annualized_total_gst"]
                g23 = gst_state.loc["2023-24", "annualized_total_gst"]
                if g22 > 0:
                    gst_growth = (g23 - g22) / g22
                    
            implied_growth = 0.50 * hist_cagr + 0.30 * NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2023-24"] + 0.20 * (gst_growth * 0.8)
            implied_growth = float(np.clip(implied_growth, 0.05, 0.22))
            imputed_val = v_2023 * (1.0 + implied_growth)
            
            mask = (df["state"] == state) & (df["fiscal_year"] == "2023-24")
            df.loc[mask, "per_capita_gsdp_raw"] = round(imputed_val, 2)
            df.loc[mask, "is_imputed"] = True
            
    return df

def fit_tax_elasticity_model(merged_panel: pd.DataFrame) -> Tuple[HuberRegressor, float, Dict[str, float]]:
    """Fits Huber regression: ln(annualized_total_gst) = alpha_i + beta * ln(per_capita_gsdp)."""
    valid = merged_panel.dropna(subset=["per_capita_gsdp_raw", "annualized_total_gst"]).copy()
    valid = valid[(valid["per_capita_gsdp_raw"] > 0) & (valid["annualized_total_gst"] > 0)]
    
    valid["log_gsdp"] = np.log(valid["per_capita_gsdp_raw"])
    valid["log_gst"] = np.log(valid["annualized_total_gst"])
    
    state_dummies = pd.get_dummies(valid["state"], drop_first=True, dtype=float)
    X = pd.concat([valid[["log_gsdp"]], state_dummies], axis=1)
    y = valid["log_gst"]
    
    reg = HuberRegressor(max_iter=300).fit(X, y)
    beta_elasticity = reg.coef_[0]
    
    state_intercepts = {}
    base_state = sorted(valid["state"].unique())[0]
    base_intercept = reg.intercept_
    state_intercepts[base_state] = base_intercept
    
    for i, col in enumerate(state_dummies.columns):
        state_intercepts[col] = base_intercept + reg.coef_[i + 1]
        
    return reg, beta_elasticity, state_intercepts

def generate_complete_gsdp_panel(gsdp_imputed: pd.DataFrame, annual_gst: pd.DataFrame) -> pd.DataFrame:
    """Precasts FY18-FY19 and Forecasts FY25-FY26 with P10/P90 confidence bounds."""
    merged = pd.merge(gsdp_imputed, annual_gst, on=["state", "fiscal_year"], how="inner")
    _, beta_elasticity, state_intercepts = fit_tax_elasticity_model(merged)
    beta_elasticity = float(np.clip(beta_elasticity, 0.75, 1.40))
    
    all_fys = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    records = []
    
    for state in BENCHMARK_28_STATES:
        state_gsdp = gsdp_imputed[gsdp_imputed["state"] == state].set_index("fiscal_year")
        val_2020 = state_gsdp.loc["2019-20", "per_capita_gsdp_raw"] if "2019-20" in state_gsdp.index else 150000.0
        val_2024 = state_gsdp.loc["2023-24", "per_capita_gsdp_raw"] if "2023-24" in state_gsdp.index else 250000.0
        
        # 1. Precasting / Backcasting (FY18 and FY19)
        val_2019_est = val_2020 / (1.0 + NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2019-20"])
        val_2018_est = val_2019_est / (1.0 + NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2018-19"])
        
        # 2. Forward Forecasting (FY25 and FY26)
        state_recent_cagr = (val_2024 / val_2020) ** (1.0 / 4.0) - 1.0 if val_2020 > 0 else 0.10
        fwd_g25 = np.clip(0.65 * state_recent_cagr + 0.35 * NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2024-25"], 0.06, 0.18)
        val_2025_est = val_2024 * (1.0 + fwd_g25)
        
        fwd_g26 = np.clip(0.60 * state_recent_cagr + 0.40 * NATIONAL_NOMINAL_GROWTH_BENCHMARKS["2025-26"], 0.05, 0.16)
        val_2026_est = val_2025_est * (1.0 + fwd_g26)
        
        for fy in all_fys:
            if fy in ["2017-18", "2018-19"]:
                val = val_2018_est if fy == "2017-18" else val_2019_est
                status = "Synthetic (Precast/Backcast)"
                sigma_pct = 0.075
            elif fy in ["2024-25", "2025-26"]:
                val = val_2025_est if fy == "2024-25" else val_2026_est
                status = "Synthetic (Forecast)"
                sigma_pct = 0.065 if fy == "2024-25" else 0.095
            else:
                is_imp = state_gsdp.loc[fy, "is_imputed"] if fy in state_gsdp.index else False
                val = state_gsdp.loc[fy, "per_capita_gsdp_raw"] if fy in state_gsdp.index else np.nan
                status = "Synthetic (Imputed)" if is_imp else "Empirical (Real MoSPI)"
                sigma_pct = 0.04 if is_imp else 0.01
                
            p10 = val * (1.0 - 1.28 * sigma_pct)
            p90 = val * (1.0 + 1.28 * sigma_pct)
            
            records.append({
                "state": state,
                "fiscal_year": fy,
                "per_capita_gsdp": round(val, 2),
                "per_capita_gsdp_p10": round(p10, 2),
                "per_capita_gsdp_p90": round(p90, 2),
                "data_regime": status,
                "is_synthetic": status != "Empirical (Real MoSPI)"
            })
            
    return pd.DataFrame(records)

def forecast_monthly_gst_to_2026(monthly_gst: pd.DataFrame) -> pd.DataFrame:
    """Closes monthly GST series through March 2026 using Seasonal Holt-Winters."""
    all_series = []
    target_end_date = pd.Timestamp("2026-03-01")
    
    for state, group in monthly_gst.groupby("state_name_clean"):
        ts_df = group.set_index("date").sort_index()
        full_idx = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq="MS")
        ts_reindexed = ts_df.reindex(full_idx)
        total_ts = ts_reindexed["total_gst"].interpolate(method="time").bfill()
        
        max_date = ts_reindexed.index.max()
        if max_date < target_end_date:
            horizon = len(pd.date_range(start=max_date + pd.DateOffset(months=1), end=target_end_date, freq="MS"))
            try:
                hw_model = ExponentialSmoothing(
                    total_ts, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
                ).fit()
                fwd_preds = hw_model.forecast(horizon)
            except Exception:
                drift = (total_ts.iloc[-1] - total_ts.iloc[0]) / max(len(total_ts), 1)
                fwd_preds = pd.Series([total_ts.iloc[-1] + drift * (i + 1) for i in range(horizon)],
                                      index=pd.date_range(start=max_date + pd.DateOffset(months=1), periods=horizon, freq="MS"))
                
            future_rows = []
            cgst_ratio = (ts_reindexed["cgst"] / ts_reindexed["total_gst"]).median() or 0.25
            sgst_ratio = (ts_reindexed["sgst"] / ts_reindexed["total_gst"]).median() or 0.32
            igst_ratio = (ts_reindexed["igst"] / ts_reindexed["total_gst"]).median() or 0.38
            cess_ratio = (ts_reindexed["cess"] / ts_reindexed["total_gst"]).median() or 0.05
            
            for dt, val in fwd_preds.items():
                val_clean = max(1.0, float(val))
                future_rows.append({
                    "id": 9999000,
                    "date": dt,
                    "state_name": state,
                    "state_code": group["state_code"].iloc[0],
                    "cgst": round(val_clean * cgst_ratio, 2),
                    "sgst": round(val_clean * sgst_ratio, 2),
                    "igst": round(val_clean * igst_ratio, 2),
                    "cess": round(val_clean * cess_ratio, 2),
                    "state_name_clean": state,
                    "total_gst": round(val_clean, 2),
                    "domestic_gst": round(val_clean * (cgst_ratio + sgst_ratio), 2),
                    "fiscal_year": assign_fiscal_year(dt),
                    "fiscal_quarter": f"Q{((dt.month - 4) % 12) // 3 + 1}",
                    "year": dt.year,
                    "month": dt.month,
                    "is_benchmark_state": state in BENCHMARK_28_STATES,
                    "is_synthetic_forecast": True
                })
            
            hist_df = group.copy()
            hist_df["is_synthetic_forecast"] = False
            all_series.append(pd.concat([hist_df, pd.DataFrame(future_rows)], ignore_index=True))
        else:
            hist_df = group.copy()
            hist_df["is_synthetic_forecast"] = False
            all_series.append(hist_df)
            
    return pd.concat(all_series, ignore_index=True).sort_values(["state_name_clean", "date"]).reset_index(drop=True)

def build_harmonized_macro_panel() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates the unified 2017-2026 panel."""
    raw_gst = load_and_clean_gst()
    raw_gsdp = load_and_clean_gsdp()
    annual_gst_base = aggregate_annual_gst(raw_gst)
    
    gsdp_imputed = impute_missing_gsdp_2024(raw_gsdp, annual_gst_base)
    complete_gsdp = generate_complete_gsdp_panel(gsdp_imputed, annual_gst_base)
    complete_monthly_gst = forecast_monthly_gst_to_2026(raw_gst)
    complete_annual_gst = aggregate_annual_gst(complete_monthly_gst)
    
    macro_panel = pd.merge(complete_annual_gst, complete_gsdp, on=["state", "fiscal_year"], how="inner")
    macro_panel["gst_per_capita_proxy"] = macro_panel["annualized_total_gst"] * 1e7 / macro_panel["per_capita_gsdp"].replace(0, np.nan)
    macro_panel["tax_productivity_pct"] = (macro_panel["annualized_total_gst"] / macro_panel["per_capita_gsdp"]) * 1000.0
    
    return complete_monthly_gst, complete_gsdp, macro_panel
