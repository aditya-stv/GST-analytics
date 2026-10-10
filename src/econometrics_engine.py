"""
econometrics_engine.py - Panel Fixed Effects, Tax Buoyancy, and Inequality Engine
"""

import os
import sys
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

def compute_gini_and_lorenz(values: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes Gini Coefficient and Lorenz curve cumulative points."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    vals = vals[vals >= 0]
    if len(vals) == 0:
        return 0.0, np.array([0, 1]), np.array([0, 1])
        
    sorted_vals = np.sort(vals)
    n = len(sorted_vals)
    cum_vals = np.cumsum(sorted_vals)
    total_val = cum_vals[-1]
    
    if total_val == 0:
        return 0.0, np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
        
    lorenz_curve = np.insert(cum_vals / total_val, 0, 0)
    pop_percentiles = np.linspace(0, 1, n + 1)
    
    area_under_lorenz = np.trapz(lorenz_curve, pop_percentiles)
    gini = 1.0 - 2.0 * area_under_lorenz
    return float(np.clip(gini, 0.0, 1.0)), pop_percentiles, lorenz_curve

def run_panel_econometrics(macro_panel: pd.DataFrame) -> Dict[str, Any]:
    """Estimates Panel Fixed Effects vs Pooled OLS for Tax Buoyancy."""
    df = macro_panel.copy()
    df = df.dropna(subset=["annualized_total_gst", "per_capita_gsdp"])
    df = df[(df["annualized_total_gst"] > 0) & (df["per_capita_gsdp"] > 0)]
    
    df["log_gst"] = np.log(df["annualized_total_gst"])
    df["log_gsdp"] = np.log(df["per_capita_gsdp"])
    
    post_covid_fys = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    df["post_covid"] = df["fiscal_year"].isin(post_covid_fys).astype(float)
    
    # 1. Pooled OLS
    X_pooled = sm.add_constant(df[["log_gsdp", "post_covid"]])
    y = df["log_gst"]
    pooled_model = OLS(y, X_pooled).fit()
    
    # 2. Entity Fixed Effects (Within transformation)
    state_dummies = pd.get_dummies(df["state"], drop_first=True, dtype=float)
    X_fe = sm.add_constant(pd.concat([df[["log_gsdp", "post_covid"]], state_dummies], axis=1))
    fe_model = OLS(y, X_fe).fit()
    
    return {
        "pooled_ols": {
            "r_squared": float(pooled_model.rsquared),
            "beta_gsdp": float(pooled_model.params.get("log_gsdp", 0.0)),
            "se_gsdp": float(pooled_model.bse.get("log_gsdp", 0.0)),
            "pval_gsdp": float(pooled_model.pvalues.get("log_gsdp", 0.0)),
            "beta_covid": float(pooled_model.params.get("post_covid", 0.0))
        },
        "fixed_effects": {
            "r_squared": float(fe_model.rsquared),
            "beta_gsdp": float(fe_model.params.get("log_gsdp", 0.0)),
            "se_gsdp": float(fe_model.bse.get("log_gsdp", 0.0)),
            "pval_gsdp": float(fe_model.pvalues.get("log_gsdp", 0.0)),
            "beta_covid": float(fe_model.params.get("post_covid", 0.0))
        },
        "n_observations": int(len(df)),
        "n_states": int(df["state"].nunique())
    }

def calculate_state_buoyancy_profile(macro_panel: pd.DataFrame) -> pd.DataFrame:
    """Computes state-level empirical Tax Buoyancy Index."""
    df = macro_panel.copy()
    piv_gst = df.pivot(index="state", columns="fiscal_year", values="annualized_total_gst")
    piv_gsdp = df.pivot(index="state", columns="fiscal_year", values="per_capita_gsdp")
    
    records = []
    base_fy = "2018-19" if "2018-19" in piv_gst.columns else piv_gst.columns[0]
    term_fy = "2023-24" if "2023-24" in piv_gst.columns else piv_gst.columns[-2]
    
    for state in piv_gst.index:
        gst_start = piv_gst.loc[state, base_fy]
        gst_end = piv_gst.loc[state, term_fy]
        gsdp_start = piv_gsdp.loc[state, base_fy]
        gsdp_end = piv_gsdp.loc[state, term_fy]
        
        pct_delta_gst = (gst_end - gst_start) / gst_start if gst_start > 0 else np.nan
        pct_delta_gsdp = (gsdp_end - gsdp_start) / gsdp_start if gsdp_start > 0 else np.nan
        
        buoyancy = pct_delta_gst / pct_delta_gsdp if (pd.notna(pct_delta_gst) and pd.notna(pct_delta_gsdp) and abs(pct_delta_gsdp) > 0.001) else 1.0
        
        if buoyancy >= 1.30:
            classification = "Hyper-Buoyant (Progressive)"
        elif buoyancy >= 1.00:
            classification = "Buoyant (Elastic)"
        elif buoyancy >= 0.70:
            classification = "Moderate (Unitary)"
        else:
            classification = "Sub-Buoyant (Lagging/Leakage)"
            
        records.append({
            "state": state,
            "base_fy": base_fy,
            "terminal_fy": term_fy,
            "gst_growth_pct": round(pct_delta_gst * 100.0, 1),
            "gsdp_growth_pct": round(pct_delta_gsdp * 100.0, 1),
            "tax_buoyancy_index": round(float(buoyancy), 2),
            "fiscal_classification": classification,
            "avg_annual_gst_cr": round(float(piv_gst.loc[state].mean()), 1),
            "latest_gsdp_pc": round(float(piv_gsdp.loc[state, term_fy]), 0)
        })
        
    return pd.DataFrame(records).sort_values("tax_buoyancy_index", ascending=False).reset_index(drop=True)
