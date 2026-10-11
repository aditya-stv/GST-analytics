"""
analysis.py - Single entry point that runs the full pipeline and computes every
headline number. The app, the tests and the README all read from here, so no
finding is ever hard-coded.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.data_engine import (
    ANALYSIS_END, COMPONENTS, COMPLETE_FYS, aggregate_annual, build_empirical_panel,
    load_gsdp, load_gst, national_monthly,
)
from src.econometrics_engine import (
    calculate_state_buoyancy, compute_gini_and_lorenz, correlation_table, run_panel_models,
)
from src.forecasting import run_forecasting
from src.ml_engine import build_state_features, detect_fiscal_anomalies, perform_state_clustering


def run_pipeline() -> Dict:
    monthly, all_months, quality_log = load_gst()
    gsdp = load_gsdp()
    annual = aggregate_annual(monthly)
    panel = build_empirical_panel(monthly, gsdp)
    nat = national_monthly(monthly)
    nat_all = national_monthly(all_months[all_months["date"] >= monthly["date"].min()])

    features = build_state_features(annual, gsdp)
    clusters, cluster_meta = perform_state_clustering(features)

    bundle = {
        "monthly": monthly,
        "all_months": all_months,
        "quality_log": pd.DataFrame(quality_log),
        "gsdp": gsdp,
        "annual": annual,
        "panel": panel,
        "national": nat,
        "models": run_panel_models(panel),
        "correlations": correlation_table(panel),
        "buoyancy": calculate_state_buoyancy(panel),
        "forecast": run_forecasting(nat_all, ANALYSIS_END),
        "clusters": clusters,
        "cluster_meta": cluster_meta,
        "anomalies": detect_fiscal_anomalies(monthly),
    }
    bundle["findings"] = compute_findings(bundle)
    return bundle


def compute_findings(b: Dict) -> Dict:
    f: Dict = {}
    monthly, nat, annual, gsdp = b["monthly"], b["national"], b["annual"], b["gsdp"]

    # 1. Revenue magnitude and mix (Aug 2017 - Mar 2025)
    f["n_months"] = int(monthly["date"].nunique())
    f["period"] = f"{monthly['date'].min():%b %Y} - {monthly['date'].max():%b %Y}"
    f["total_cr"] = float(monthly["total_gst"].sum())
    f["component_cr"] = {c: float(monthly[c].sum()) for c in COMPONENTS}
    f["component_share"] = {c: f["component_cr"][c] / f["total_cr"] * 100 for c in COMPONENTS}

    fy = annual.groupby("fiscal_year")["total_gst"].sum()
    f["fy_totals"] = fy[COMPLETE_FYS]
    f["fy19_cr"], f["fy25_cr"] = float(fy["2018-19"]), float(fy["2024-25"])
    f["cagr_fy19_fy25_pct"] = ((fy["2024-25"] / fy["2018-19"]) ** (1 / 6) - 1) * 100

    # 2. Spatial concentration (states + UTs; centrally booked heads excluded from ranking)
    terr = monthly[monthly["entity_type"] != "Non-territorial"]
    ranking = terr.groupby("state")["total_gst"].sum().sort_values(ascending=False)
    share = ranking / f["total_cr"] * 100
    f["state_share"] = share
    f["top1_state"], f["top1_share"] = share.index[0], float(share.iloc[0])
    f["top1_cr"] = float(ranking.iloc[0])
    f["top5_states"], f["top5_share"] = list(share.index[:5]), float(share.iloc[:5].sum())
    f["top10_share"] = float(share.iloc[:10].sum())
    f["bottom10_share"] = float(share.iloc[-10:].sum())
    f["n_territories"] = int(len(share))

    # 3. COVID shock
    s = nat.set_index("date")["total_gst"]
    pre = s["2019-04-01":"2020-02-01"].mean()
    f["precovid_mean_cr"] = float(pre)
    f["apr20_cr"] = float(s["2020-04-01"])
    f["apr20_drop_pct"] = float((s["2020-04-01"] / pre - 1) * 100)
    f["apr20_yoy_pct"] = float((s["2020-04-01"] / s["2019-04-01"] - 1) * 100)
    after = s["2020-04-01":]
    f["recovery_month"] = after[after >= pre].index[0]
    f["recovery_lag_months"] = int((f["recovery_month"].year - 2020) * 12 + f["recovery_month"].month - 4)

    # 4. Per-capita income divide (official MoSPI rows only)
    piv = gsdp.pivot(index="state", columns="fiscal_year", values="per_capita_gsdp")
    ratio = []
    for col in piv.columns:
        c = piv[col].dropna()
        ratio.append({"fiscal_year": col, "richest": c.idxmax(), "richest_value": c.max(),
                      "poorest": c.idxmin(), "poorest_value": c.min(), "ratio": c.max() / c.min()})
    f["income_ratio"] = pd.DataFrame(ratio)
    f["fy21_contracted"] = int((piv["2020-21"] < piv["2019-20"]).sum())

    # 5. Tax mix by state (28 states, full window)
    st = annual[annual["entity_type"] == "State"].groupby("state")[COMPONENTS + ["total_gst"]].sum()
    mix = st[COMPONENTS].div(st["total_gst"], axis=0) * 100
    f["state_mix"] = mix
    f["top_cess"] = mix["cess"].sort_values(ascending=False).head(5)
    f["top_igst"] = mix["igst"].sort_values(ascending=False).head(5)

    # 6. Correlations
    f["correlations"] = b["correlations"].set_index("metric")

    # 7. Inequality (FY 2022-23: latest year with official GSDP for all 28 states)
    p23 = b["panel"][b["panel"]["fiscal_year"] == "2022-23"]
    f["gini_gst"] = compute_gini_and_lorenz(p23["total_gst"])[0]
    f["gini_gsdp_pc"] = compute_gini_and_lorenz(p23["per_capita_gsdp"])[0]

    # 8. Models
    m = b["models"].set_index("model")
    f["models"] = m
    f["buoyant_states"] = int((b["buoyancy"]["buoyancy"] >= 1).sum())

    # 9. Forecast
    fc = b["forecast"]
    best = fc["backtest_summary"].sort_values("mean_mape_pct").iloc[0]
    f["best_model"], f["best_mape"] = best["model"], float(best["mean_mape_pct"])
    f["sarima_mape"] = float(fc["backtest_summary"].set_index("model").loc["SARIMA (log) + lockdown dummy", "mean_mape_pct"])
    f["interval_halfwidth_pct"] = fc["interval_halfwidth_pct"]
    f["validation"] = fc["validation"]
    f["fy26_forecast_cr"] = fc["fy26_forecast_total"]
    f["fy26_growth_pct"] = (fc["fy26_forecast_total"] / fc["fy25_actual_total"] - 1) * 100

    # 10. Anomalies
    an = b["anomalies"]
    flagged = an[an["is_anomaly"]]
    f["n_anomalies"] = int(len(flagged))
    f["anomaly_covid_share"] = float(flagged["date"].between("2020-04-01", "2021-06-01").mean() * 100)
    f["cluster_k"] = b["cluster_meta"]["k"]
    f["cluster_silhouette"] = b["cluster_meta"]["silhouette"]

    f.update(compute_deep_dive(b, f))
    return f


def compute_deep_dive(b: Dict, f: Dict) -> Dict:
    """Second-level insights: trends in concentration, growth, mix, seasonality and volatility."""
    d: Dict = {}
    monthly, annual, nat = b["monthly"], b["annual"], b["national"].set_index("date")
    fy_tot = f["fy_totals"]

    # Concentration over time (share of the national total, states + UTs)
    terr = annual[(annual["entity_type"] != "Non-territorial") & annual["fiscal_year"].isin(COMPLETE_FYS)]
    shares = terr.pivot(index="state", columns="fiscal_year", values="total_gst").div(fy_tot) * 100
    d["share_by_year"] = shares
    d["hhi_by_year"] = (shares ** 2).sum()
    d["top5_by_year"] = shares.apply(lambda c: c.nlargest(5).sum())
    d["mh_share_by_year"] = shares.loc["Maharashtra"]
    d["share_change_pp"] = (shares["2024-25"] - shares["2018-19"]).dropna().sort_values()

    # State growth FY19 -> FY25
    st = annual[annual["entity_type"] == "State"].pivot(index="state", columns="fiscal_year", values="total_gst")
    cagr = ((st["2024-25"] / st["2018-19"]) ** (1 / 6) - 1) * 100
    d["state_cagr"] = cagr.sort_values()
    r_size, p_size = stats.pearsonr(np.log(st["2018-19"]), cagr)
    gsdp0 = b["gsdp"].pivot(index="state", columns="fiscal_year", values="per_capita_gsdp")["2019-20"]
    r_inc, p_inc = stats.pearsonr(np.log(gsdp0[cagr.index]), cagr)
    d["cagr_vs_size"] = (float(r_size), float(p_size))
    d["cagr_vs_income"] = (float(r_inc), float(p_inc))

    # Growth normalisation and the post-COVID level shift
    yoy = fy_tot.pct_change() * 100
    d["fy_yoy"] = yoy
    pre_growth = fy_tot["2019-20"] / fy_tot["2018-19"] - 1
    trend_fy25 = fy_tot["2019-20"] * (1 + pre_growth) ** 5
    d["pre_covid_growth_pct"] = pre_growth * 100
    d["fy25_vs_precovid_trend_pct"] = (fy_tot["2024-25"] / trend_fy25 - 1) * 100

    # National tax-mix drift
    mix = annual[annual["fiscal_year"].isin(COMPLETE_FYS)].groupby("fiscal_year")[COMPONENTS + ["total_gst"]].sum()
    d["mix_by_year"] = mix[COMPONENTS].div(mix["total_gst"], axis=0) * 100
    d["fy25_cess_cr"] = float(mix.loc["2024-25", "cess"])

    # SGST vs CGST: equal statutory rates, unequal cash collections
    sc = annual[(annual["entity_type"] == "State") & annual["fiscal_year"].isin(COMPLETE_FYS)] \
        .groupby("state")[["sgst", "cgst"]].sum()
    ratio = (sc["sgst"] / sc["cgst"]).sort_values()
    d["sgst_cgst_ratio"] = ratio
    d["sgst_cgst_national"] = float(f["component_cr"]["sgst"] / f["component_cr"]["cgst"])

    # Seasonality: ratio to centred 12-month moving average, FY 2021-22 onward (post-COVID)
    s = nat["total_gst"]
    centred = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    idx = (s / centred)["2021-07-01":"2024-12-01"]
    d["seasonal_index"] = idx.groupby(idx.index.month).mean()

    # Volatility of YoY monthly growth, COVID window excluded
    m = monthly[monthly["entity_type"] == "State"].sort_values(["state", "date"]).copy()
    m["yoy"] = m.groupby("state")["total_gst"].pct_change(12)
    vol = m[~m["date"].between("2020-03-01", "2021-06-30")].groupby("state")["yoy"].std() * 100
    d["volatility"] = vol.sort_values()
    rho, p_rho = stats.spearmanr(vol, np.log(st.loc[vol.index, "2024-25"]))
    d["volatility_vs_size"] = (float(rho), float(p_rho))

    # Cess concentration in FY 2024-25
    c25 = annual[(annual["fiscal_year"] == "2024-25") & (annual["entity_type"] == "State")].set_index("state")
    d["cess_share_fy25"] = (c25["cess"] / c25["total_gst"] * 100).sort_values(ascending=False)
    return d
