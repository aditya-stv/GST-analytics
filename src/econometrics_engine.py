"""
econometrics_engine.py - Panel elasticity models, correlations, buoyancy and inequality.

All estimation runs on the empirical panel only (states x FY 2019-20..2023-24,
both series observed). Standard errors are clustered by state because the
same state appears in up to five years.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compute_gini_and_lorenz(values) -> Tuple[float, np.ndarray, np.ndarray]:
    """Gini coefficient and Lorenz-curve points for a vector of non-negative values."""
    vals = np.asarray(values, dtype=float)
    vals = np.sort(vals[~np.isnan(vals) & (vals >= 0)])
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    lorenz = np.insert(np.cumsum(vals) / vals.sum(), 0, 0.0)
    population = np.linspace(0.0, 1.0, n + 1)
    # Trapezoid area under the Lorenz curve (works on every NumPy version).
    area = float(np.sum((lorenz[1:] + lorenz[:-1]) / 2.0 * np.diff(population)))
    return float(1.0 - 2.0 * area), population, lorenz


def _residualize(frame: pd.DataFrame, cols, absorb) -> pd.DataFrame:
    """Frisch-Waugh-Lovell: residualise `cols` on dummies for the `absorb` factors."""
    if not absorb:
        return frame[cols] - frame[cols].mean()
    dummies = pd.get_dummies(frame[absorb].astype(str), drop_first=False, dtype=float)
    beta, *_ = np.linalg.lstsq(dummies.values, frame[cols].values, rcond=None)
    return pd.DataFrame(frame[cols].values - dummies.values @ beta, columns=cols, index=frame.index)


def _fit(y: pd.Series, x: pd.Series, groups: pd.Series):
    return sm.OLS(y, sm.add_constant(x.to_frame())).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def run_panel_models(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates the GST-income elasticity  ln(GST_it) = a + b ln(GSDPpc_it) + e  four ways.

    * Pooled OLS          - cross-state comparison, confounded by state size.
    * State FE            - within-state variation; still shares the common nominal trend.
    * Two-way FE          - state + year effects; removes inflation/COVID shocks common to all states.
    * First differences   - year-on-year growth regression.
    R-squared for the FE models is the within (partial) R-squared after absorbing the effects,
    not the inflated LSDV R-squared that includes the dummies.
    """
    df = panel.copy()
    rows = []

    specs = [
        ("Pooled OLS", [], "Between-state; confounded by population size"),
        ("State fixed effects", ["state"], "Within-state; shares common nominal trend"),
        ("Two-way fixed effects", ["state", "fiscal_year"], "Within-state, net of year shocks"),
    ]
    for name, absorb, note in specs:
        X = df[["log_gsdp_pc"]]
        if absorb:
            X = pd.concat([X, pd.get_dummies(df[absorb].astype(str), drop_first=True, dtype=float)], axis=1)
        res = sm.OLS(df["log_gst"], sm.add_constant(X)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(df["state"])[0]})
        row = _summarise(name, res, "log_gsdp_pc", len(df), note)
        if absorb:
            r = _residualize(df, ["log_gst", "log_gsdp_pc"], absorb)
            row["r_squared"] = float(np.corrcoef(r["log_gst"], r["log_gsdp_pc"])[0, 1] ** 2)
        rows.append(row)

    df = df.sort_values(["state", "fiscal_year"])
    df["d_log_gst"] = df.groupby("state")["log_gst"].diff()
    df["d_log_gsdp"] = df.groupby("state")["log_gsdp_pc"].diff()
    fd = df.dropna(subset=["d_log_gst", "d_log_gsdp"])
    res = _fit(fd["d_log_gst"], fd["d_log_gsdp"], fd["state"])
    rows.append(_summarise("First differences", res, "d_log_gsdp", len(fd), "Year-on-year growth response"))
    return pd.DataFrame(rows)


def _summarise(name, res, var, n, note) -> Dict:
    lo, hi = res.conf_int().loc[var]
    return {
        "model": name,
        "elasticity": float(res.params[var]),
        "std_error": float(res.bse[var]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(res.pvalues[var]),
        "r_squared": float(res.rsquared),
        "n_obs": int(n),
        "interpretation": note,
    }


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Correlates per-capita GSDP with GST level and composition two ways:
    pooled state-years (as originally reported, observations not independent)
    and one observation per state (period averages, n = 28), which is the valid test.
    """
    df = panel.copy()
    state = df.groupby("state").agg(
        per_capita_gsdp=("per_capita_gsdp", "mean"), total_gst=("total_gst", "mean"),
        igst=("igst", "sum"), cess=("cess", "sum"), gst_sum=("total_gst", "sum"),
    )
    state["igst_share"] = state["igst"] / state["gst_sum"]
    state["cess_share"] = state["cess"] / state["gst_sum"]

    rows = []
    for label, col in [("Total GST (Rs Cr)", "total_gst"), ("IGST share", "igst_share"), ("Cess share", "cess_share")]:
        r_p, p_p = stats.pearsonr(df["per_capita_gsdp"], df[col])
        r_s, p_s = stats.pearsonr(state["per_capita_gsdp"], state[col])
        rho, p_rho = stats.spearmanr(state["per_capita_gsdp"], state[col])
        rows.append({
            "metric": label,
            "pooled_r": r_p, "pooled_p": p_p, "pooled_n": len(df),
            "state_r": r_s, "state_p": p_s, "state_spearman": rho, "state_spearman_p": p_rho,
            "state_n": len(state),
        })
    return pd.DataFrame(rows)


def calculate_state_buoyancy(panel: pd.DataFrame) -> pd.DataFrame:
    """
    State tax buoyancy = log-growth of GST / log-growth of per-capita GSDP,
    from FY 2019-20 to the latest year with official GSDP (FY 2023-24, else FY 2022-23).

    Caveat: the denominator is per-capita income while the numerator is total revenue,
    so population growth (~1% a year) pushes every ratio slightly upward.
    """
    records = []
    for state, grp in panel.groupby("state"):
        grp = grp.set_index("fiscal_year")
        base = "2019-20"
        term = "2023-24" if "2023-24" in grp.index else grp.index.max()
        years = int(term[:4]) - int(base[:4])
        g_gst = np.log(grp.loc[term, "total_gst"] / grp.loc[base, "total_gst"])
        g_gsdp = np.log(grp.loc[term, "per_capita_gsdp"] / grp.loc[base, "per_capita_gsdp"])
        buoyancy = g_gst / g_gsdp if g_gsdp > 0.01 else np.nan
        records.append({
            "state": state,
            "period": f"{base} to {term}",
            "gst_cagr_pct": (np.exp(g_gst / years) - 1) * 100,
            "gsdp_pc_cagr_pct": (np.exp(g_gsdp / years) - 1) * 100,
            "buoyancy": buoyancy,
            "class": "Buoyant (>1)" if buoyancy >= 1 else "Sub-buoyant (<1)",
            "latest_gst_cr": grp.loc[term, "total_gst"],
            "latest_gsdp_pc": grp.loc[term, "per_capita_gsdp"],
        })
    return pd.DataFrame(records).sort_values("buoyancy", ascending=False).reset_index(drop=True)
