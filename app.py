"""
app.py - Fiscal & Macro-Economic Intelligence Platform (2017-2026)
Clean Institutional Interface with Dynamic Light/Dark Theme and Zero Emojis.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.data_engine import (
    load_and_clean_gst,
    load_and_clean_gsdp,
    aggregate_annual_gst,
    BENCHMARK_28_STATES
)
from src.synthetic_model import build_harmonized_macro_panel
from src.econometrics_engine import (
    run_panel_econometrics,
    calculate_state_buoyancy_profile,
    compute_gini_and_lorenz
)
from src.ml_engine import perform_state_clustering, detect_fiscal_anomalies
from src.simulation_engine import run_monte_carlo_fiscal_simulation
from src.visual_styles import (
    INSTITUTIONAL_LIGHT_CSS,
    INSTITUTIONAL_DARK_CSS,
    apply_plotly_theme
)

st.set_page_config(
    page_title="Fiscal & Macro Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Settings & Theme Toggle
st.sidebar.markdown("### Interface Settings")
theme_choice = st.sidebar.radio("Theme Mode", ["Light", "Dark"], index=0, horizontal=True)
is_dark_mode = (theme_choice == "Dark")

# Inject selected theme CSS
if is_dark_mode:
    st.markdown(INSTITUTIONAL_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(INSTITUTIONAL_LIGHT_CSS, unsafe_allow_html=True)

def style_fig(fig, title="", height=450):
    return apply_plotly_theme(fig, title=title, height=height, is_dark=is_dark_mode)

# Load data bundle
@st.cache_data(show_spinner=False)
def load_all_data():
    raw_gst = load_and_clean_gst()
    raw_gsdp = load_and_clean_gsdp()
    monthly_gst, complete_gsdp, macro_panel = build_harmonized_macro_panel()
    panel_metrics = run_panel_econometrics(macro_panel)
    buoyancy_df = calculate_state_buoyancy_profile(macro_panel)
    clustered_df, cluster_meta = perform_state_clustering(macro_panel)
    anomalies_df = detect_fiscal_anomalies(monthly_gst)
    return {
        "raw_gst": raw_gst,
        "raw_gsdp": raw_gsdp,
        "monthly_gst": monthly_gst,
        "complete_gsdp": complete_gsdp,
        "macro_panel": macro_panel,
        "panel_metrics": panel_metrics,
        "buoyancy_df": buoyancy_df,
        "clustered_df": clustered_df,
        "cluster_meta": cluster_meta,
        "anomalies_df": anomalies_df
    }

with st.spinner("Initializing models and panels..."):
    data = load_all_data()

raw_gst = data["raw_gst"]
raw_gsdp = data["raw_gsdp"]
monthly_gst = data["monthly_gst"]
complete_gsdp = data["complete_gsdp"]
macro_panel = data["macro_panel"]
panel_metrics = data["panel_metrics"]
buoyancy_df = data["buoyancy_df"]
clustered_df = data["clustered_df"]
cluster_meta = data["cluster_meta"]
anomalies_df = data["anomalies_df"]

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Controls")

all_states = sorted(macro_panel["state"].unique())
selected_state = st.sidebar.selectbox("State Filter", ["All India (Aggregate)"] + all_states)

fy_list = sorted(macro_panel["fiscal_year"].unique())
selected_fys = st.sidebar.select_slider(
    "Fiscal Timeline Window",
    options=fy_list,
    value=(fy_list[0], fy_list[-1])
)

st.sidebar.markdown("---")
st.sidebar.caption(
    """
    **Platform Modules:**
    - Tab 1: Empirical EDA (Raw Data)
    - Tab 2: Precasting & Forecasting
    - Tab 3: Panel Fixed Effects & Elasticity
    - Tab 4: Machine Learning Archetypes
    - Tab 5: Monte Carlo Simulator
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Sources")
st.sidebar.markdown(
    """
    - **GST Portal Collection:**  
      [CKAN Dev Dataset](https://ckandev.indiadataportal.com/dataset/goods-and-services-tax)
    - **State Per Capita GSDP:**  
      [Data.gov.in (MoSPI)](https://www.data.gov.in/resource/state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24)
    """
)

# Header
st.title("Macro-Fiscal & Tax Intelligence Platform (2017–2026)")
st.caption("Quantitative Synthesis of Goods and Services Tax Collections & MoSPI Per Capita GSDP across 28 Benchmark States")

# Top Metrics Bar
tot_real_gst = raw_gst["total_gst"].sum()
tot_real_igst = raw_gst["igst"].sum()
tot_real_sgst = raw_gst["sgst"].sum()
tot_real_cgst = raw_gst["cgst"].sum()
tot_real_cess = raw_gst["cess"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total GST Collections", f"₹{tot_real_gst/100000:.2f} L Cr")
m2.metric("IGST (Integrated)", f"₹{tot_real_igst/100000:.2f} L Cr", "34.5% Share")
m3.metric("SGST (State)", f"₹{tot_real_sgst/100000:.2f} L Cr", "30.5% Share")
m4.metric("CGST (Central)", f"₹{tot_real_cgst/100000:.2f} L Cr", "23.6% Share")
m5.metric("Compensation Cess", f"₹{tot_real_cess/100000:.2f} L Cr", "11.4% Share")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs (Clean text with no emojis)
tabs = st.tabs([
    "1. Empirical EDA (Raw Data)",
    "2. Precasting & Forecasting Studio",
    "3. Econometric Panel & Tax Buoyancy",
    "4. Machine Learning Archetyping",
    "5. Stochastic Monte Carlo Simulator"
])

# ==============================================================
# TAB 1: EMPIRICAL EDA (RAW DATA ONLY)
# ==============================================================
with tabs[0]:
    st.subheader("Exploratory Analysis of Raw Datasets")
    st.markdown(
        """
        **Primary Data Sources:**
        - `tax-collection-on-gst-portal`: [https://ckandev.indiadataportal.com/dataset/goods-and-services-tax](https://ckandev.indiadataportal.com/dataset/goods-and-services-tax)
        - `RS_Session_266_AU_1677_B_2`: [https://www.data.gov.in/resource/state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24](https://www.data.gov.in/resource/state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24)
        """
    )
    
    # 1. National Monthly Timeline
    nat_monthly_real = raw_gst.groupby("date")[["cgst", "sgst", "igst", "cess", "total_gst"]].sum().reset_index()
    
    col_e1, col_e2 = st.columns([3, 2])
    with col_e1:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=nat_monthly_real["date"], y=nat_monthly_real["cgst"], mode="lines", name="CGST (Central)", line=dict(color="#0284C7", width=1.5)))
        fig_ts.add_trace(go.Scatter(x=nat_monthly_real["date"], y=nat_monthly_real["sgst"], mode="lines", name="SGST (State)", line=dict(color="#059669", width=1.5)))
        fig_ts.add_trace(go.Scatter(x=nat_monthly_real["date"], y=nat_monthly_real["igst"], mode="lines", name="IGST (Integrated)", line=dict(color="#6366F1", width=1.5)))
        fig_ts.add_trace(go.Scatter(x=nat_monthly_real["date"], y=nat_monthly_real["cess"], mode="lines", name="Compensation Cess", line=dict(color="#D97706", width=1.5)))
        fig_ts.add_trace(go.Scatter(x=nat_monthly_real["date"], y=nat_monthly_real["total_gst"], mode="lines+markers", name="Total GST", line=dict(color="#0F172A" if not is_dark_mode else "#FFFFFF", width=2.5, dash="dot")))
        style_fig(fig_ts, title="Real Monthly GST Collections (₹ Crore, July 2017 – Jan 2026)", height=430)
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with col_e2:
        # 2. COVID Shock
        covid_window = nat_monthly_real[(nat_monthly_real["date"] >= "2019-04-01") & (nat_monthly_real["date"] <= "2021-06-01")].copy()
        pre_covid_mean = nat_monthly_real[(nat_monthly_real["date"] >= "2019-04-01") & (nat_monthly_real["date"] <= "2020-02-01")]["total_gst"].mean()
        
        fig_covid = go.Figure()
        fig_covid.add_trace(go.Bar(
            x=covid_window["date"].dt.strftime("%b %Y"),
            y=covid_window["total_gst"],
            marker_color=["#DC2626" if dt in [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")] else "#0284C7" for dt in covid_window["date"]],
            name="Monthly GST"
        ))
        fig_covid.add_hline(y=pre_covid_mean, line_dash="dash", line_color="#D97706", annotation_text=f"Pre-COVID Baseline (₹{pre_covid_mean:,.0f} Cr)", annotation_position="top left")
        style_fig(fig_covid, title="The COVID Lockdown Shock (-75.1% in Apr 2020) & V-Recovery", height=430)
        st.plotly_chart(fig_covid, use_container_width=True)
        
    st.markdown("""
    <div class="card-insight">
        <b>Empirical Finding:</b> In April 2020, national indirect tax receipts fell by <b>-75.09%</b> to ₹19,606 Cr. However, the system rebounded to pre-COVID levels within <b>6 months</b> (October 2020), supported by the rollout of mandatory e-invoicing.
    </div>
    """, unsafe_allow_html=True)
    
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        # 3. Top 15 States Bar Chart
        state_totals = raw_gst.groupby("state_name_clean")["total_gst"].sum().sort_values(ascending=False).reset_index()
        state_totals["pct_share"] = (state_totals["total_gst"] / tot_real_gst) * 100
        fig_top = px.bar(
            state_totals.head(15),
            x="total_gst",
            y="state_name_clean",
            orientation="h",
            color="pct_share",
            color_continuous_scale="Viridis",
            labels={"total_gst": "Total GST (₹ Cr)", "state_name_clean": "State", "pct_share": "% Share"},
            title="Top 15 States: Fiscal Concentration (Maharashtra = 20.4%)"
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        style_fig(fig_top, height=450)
        st.plotly_chart(fig_top, use_container_width=True)
        
    with col_e4:
        # 4. Tax Structure Breakdown
        state_comp = raw_gst[raw_gst["state_name_clean"].isin(BENCHMARK_28_STATES)].groupby("state_name_clean")[["cgst", "sgst", "igst", "cess", "total_gst"]].sum().reset_index()
        state_comp["cgst_pct"] = (state_comp["cgst"] / state_comp["total_gst"]) * 100
        state_comp["sgst_pct"] = (state_comp["sgst"] / state_comp["total_gst"]) * 100
        state_comp["igst_pct"] = (state_comp["igst"] / state_comp["total_gst"]) * 100
        state_comp["cess_pct"] = (state_comp["cess"] / state_comp["total_gst"]) * 100
        state_comp = state_comp.sort_values("cess_pct", ascending=False)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Compensation Cess %", y=state_comp["state_name_clean"], x=state_comp["cess_pct"], orientation="h", marker_color="#D97706"))
        fig_comp.add_trace(go.Bar(name="IGST %", y=state_comp["state_name_clean"], x=state_comp["igst_pct"], orientation="h", marker_color="#6366F1"))
        fig_comp.add_trace(go.Bar(name="SGST %", y=state_comp["state_name_clean"], x=state_comp["sgst_pct"], orientation="h", marker_color="#059669"))
        fig_comp.add_trace(go.Bar(name="CGST %", y=state_comp["state_name_clean"], x=state_comp["cgst_pct"], orientation="h", marker_color="#0284C7"))
        fig_comp.update_layout(barmode="stack", yaxis={"categoryorder": "total ascending"})
        style_fig(fig_comp, title="Tax Component Breakdown (%) by State (Cess vs IGST vs SGST)", height=450)
        st.plotly_chart(fig_comp, use_container_width=True)
        
    st.markdown("""
    <div class="card-insight">
        <b>Structural Disparity:</b><br>
        • <b>Mineral Corridor:</b> Odisha (24.2% Cess), Jharkhand (21.5% Cess), and Chhattisgarh (18.9% Cess) generate elevated Compensation Cess proportions due to coal mining royalties.<br>
        • <b>Manufacturing Hubs:</b> Haryana (43.1% IGST), Gujarat (41.4% IGST), and Tamil Nadu (39.8% IGST) feature dominant IGST shares reflecting major interstate supply chains.
    </div>
    """, unsafe_allow_html=True)
    
    col_e5, col_e6 = st.columns(2)
    with col_e5:
        # 5. Real GSDP Disparity
        fig_gsdp = px.line(
            raw_gsdp.dropna(subset=["per_capita_gsdp_raw"]),
            x="fiscal_year",
            y="per_capita_gsdp_raw",
            color="state",
            markers=True,
            title="MoSPI Per Capita GSDP Trajectories (The 10.6x Wealth Gap)"
        )
        style_fig(fig_gsdp, height=440)
        st.plotly_chart(fig_gsdp, use_container_width=True)
        
    with col_e6:
        # 6. Trade Signal Scatter
        annual_real_gst = raw_gst[raw_gst["state_name_clean"].isin(BENCHMARK_28_STATES)].groupby(["state_name_clean", "fiscal_year"])[["cgst", "sgst", "igst", "cess", "total_gst"]].sum().reset_index()
        annual_real_gst.rename(columns={"state_name_clean": "state"}, inplace=True)
        merged_real = pd.merge(annual_real_gst, raw_gsdp, on=["state", "fiscal_year"], how="inner").dropna(subset=["per_capita_gsdp_raw", "total_gst"])
        merged_real["igst_share"] = merged_real["igst"] / merged_real["total_gst"]
        
        fig_scat = px.scatter(
            merged_real,
            x="per_capita_gsdp_raw",
            y="igst_share",
            color="fiscal_year",
            hover_name="state",
            labels={"per_capita_gsdp_raw": "Per Capita GSDP (₹)", "igst_share": "IGST Share in State Tax"},
            title="The Interstate Trade Signal: IGST Share vs GSDP Per Capita (r = +0.548)",
            trendline="ols"
        )
        style_fig(fig_scat, height=440)
        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================
# TAB 2: PRECASTING & FORECASTING STUDIO
# ==============================================================
with tabs[1]:
    st.subheader("Historical Precasting & Forward Forecasting Studio")
    st.write("Synthesizing missing gaps: **Precasting FY 2017-18 & FY 2018-19**, **imputing missing states in FY 2023-24**, and **forecasting FY 2024-25 & FY 2025-26** with P10/P90 confidence envelopes.")
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        inspect_state = st.selectbox(
            "State to Inspect:",
            BENCHMARK_28_STATES,
            index=BENCHMARK_28_STATES.index("Maharashtra") if "Maharashtra" in BENCHMARK_28_STATES else 0
        )
        
        state_series = complete_gsdp[complete_gsdp["state"] == inspect_state].sort_values("fiscal_year")
        
        fig_synth = go.Figure()
        color_regimes = {
            "Empirical (Real MoSPI)": "#059669",
            "Synthetic (Imputed)": "#D97706",
            "Synthetic (Precast/Backcast)": "#0284C7",
            "Synthetic (Forecast)": "#6366F1"
        }
        
        for regime, grp in state_series.groupby("data_regime"):
            fig_synth.add_trace(go.Scatter(
                x=grp["fiscal_year"],
                y=grp["per_capita_gsdp"],
                mode="markers+lines",
                name=regime,
                marker=dict(size=9, color=color_regimes.get(regime, "#64748B")),
                line=dict(dash="dot" if "Synthetic" in regime else "solid", width=2.5)
            ))
            
        fig_synth.add_trace(go.Scatter(
            x=pd.concat([state_series["fiscal_year"], state_series["fiscal_year"][::-1]]),
            y=pd.concat([state_series["per_capita_gsdp_p90"], state_series["per_capita_gsdp_p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(2, 132, 199, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="90% Bayesian Credible Interval"
        ))
        
        style_fig(fig_synth, title=f"GSDP Per Capita Synthesis with Confidence Envelopes: {inspect_state}", height=450)
        st.plotly_chart(fig_synth, use_container_width=True)
        
    with col_p2:
        regime_counts = complete_gsdp["data_regime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Observations"]
        
        fig_regime_pie = px.pie(
            regime_counts,
            values="Observations",
            names="Regime",
            hole=0.45,
            color="Regime",
            color_discrete_map=color_regimes,
            title="Panel Data Composition (252 Observations)"
        )
        style_fig(fig_regime_pie, height=450)
        st.plotly_chart(fig_regime_pie, use_container_width=True)
        
    st.markdown("""
    <div class="card-insight">
        <b>Mathematical Methodology:</b><br>
        1. <b>Precasting (FY 2017-19):</b> Inverts the panel tax buoyancy equation <i>ln(GSDP_PC) = [ln(GST) - α_i] / β</i> anchored by RBI national deflators (+11.0% in FY18, +10.5% in FY19).<br>
        2. <b>Imputation (FY 2023-24):</b> Uses state 3-year CAGR (50%) + national MoSPI benchmark (30%) + state GST growth rate (20%) for the 6 missing states.<br>
        3. <b>Forecasting (FY 2024-26):</b> Applies Holt's linear trend with Bayesian shrinkage toward the national medium-term target (+10.0%), emitting P10 and P90 credible envelopes.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================
# TAB 3: ECONOMETRIC PANEL & TAX BUOYANCY
# ==============================================================
with tabs[2]:
    st.subheader("Panel Fixed Effects & State Tax Buoyancy Econometrics")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_scatter = px.scatter(
            macro_panel,
            x="per_capita_gsdp",
            y="annualized_total_gst",
            color="fiscal_year",
            hover_name="state",
            log_x=True,
            log_y=True,
            labels={"per_capita_gsdp": "Per Capita GSDP (Log Scale ₹)", "annualized_total_gst": "Annual GST (Log Scale ₹ Cr)"},
            title="Macro Tax Elasticity (Log-Log Panel Scatter & Trajectory)",
            trendline="ols"
        )
        style_fig(fig_scatter, height=440)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_b2:
        latest_slice = macro_panel[macro_panel["fiscal_year"] == "2023-24"]
        gini_gst, p_gst, l_gst = compute_gini_and_lorenz(latest_slice["annualized_total_gst"].values)
        gini_gsdp, p_gsdp, l_gsdp = compute_gini_and_lorenz(latest_slice["per_capita_gsdp"].values)
        
        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Equality", line=dict(dash="dash", color="#94A3B8")))
        fig_lorenz.add_trace(go.Scatter(x=p_gst, y=l_gst, mode="lines", name=f"GST Revenue Lorenz (Gini = {gini_gst:.2f})", line=dict(color="#0284C7", width=2.5)))
        fig_lorenz.add_trace(go.Scatter(x=p_gsdp, y=l_gsdp, mode="lines", name=f"GSDP Per Capita Lorenz (Gini = {gini_gsdp:.2f})", line=dict(color="#059669", width=2.5)))
        style_fig(fig_lorenz, title="Fiscal Inequality: GST vs Economic Capacity (Lorenz Curve)", height=440)
        st.plotly_chart(fig_lorenz, use_container_width=True)
        
    st.markdown("#### State Tax Buoyancy Index Leaderboard")
    st.dataframe(
        buoyancy_df[[
            "state", "tax_buoyancy_index", "fiscal_classification",
            "gst_growth_pct", "gsdp_growth_pct", "avg_annual_gst_cr", "latest_gsdp_pc"
        ]],
        use_container_width=True,
        hide_index=True
    )
    
    fe = panel_metrics["fixed_effects"]
    p_ols = panel_metrics["pooled_ols"]
    st.markdown("#### Panel Regression Output (Entity Fixed Effects vs Pooled OLS)")
    comp_table = pd.DataFrame({
        "Model Specification": ["Pooled OLS", "State Entity Fixed Effects (Within)"],
        "Elasticity Beta (ln GSDP)": [f"{p_ols['beta_gsdp']:.3f}", f"{fe['beta_gsdp']:.3f}"],
        "Std. Error": [f"{p_ols['se_gsdp']:.3f}", f"{fe['se_gsdp']:.3f}"],
        "p-value": [f"{p_ols['pval_gsdp']:.4f}", f"{fe['pval_gsdp']:.2e}"],
        "Post-COVID Slope Dummy": [f"{p_ols['beta_covid']:+.3f}", f"{fe['beta_covid']:+.3f}"],
        "R-squared": [f"{p_ols['r_squared']*100:.1f}%", f"{fe['r_squared']*100:.1f}%"],
        "Conclusion": [
            "Biased: Cross-sectional state size heterogeneity dominates.",
            "Consistent: Within-state growth shows strong progressive buoyancy (p < 0.001)."
        ]
    })
    st.table(comp_table)

# ==============================================================
# TAB 4: MACHINE LEARNING ARCHETYPING & ANOMALIES
# ==============================================================
with tabs[3]:
    st.subheader("Unsupervised Machine Learning Archetypes & Anomaly Detection")
    
    col_m1, col_m2 = st.columns([3, 2])
    with col_m1:
        fig_pca = px.scatter(
            clustered_df,
            x="pca_dim1",
            y="pca_dim2",
            color="archetype",
            hover_name="state",
            hover_data=["avg_annual_gst", "avg_gsdp_pc", "tax_buoyancy_index"],
            labels={
                "pca_dim1": f"PCA Component 1 ({cluster_meta['pca_variance_ratio'][0]*100:.1f}% Var)",
                "pca_dim2": f"PCA Component 2 ({cluster_meta['pca_variance_ratio'][1]*100:.1f}% Var)"
            },
            title="State Economic Archetypes (K-Means Clustering on PCA Projections)"
        )
        style_fig(fig_pca, height=450)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col_m2:
        arch_summary = clustered_df.groupby("archetype").agg(
            State_Count=("state", "count"),
            Avg_GST_Cr=("avg_annual_gst", "mean"),
            Avg_GSDP_PC=("avg_gsdp_pc", "mean")
        ).reset_index()
        
        st.markdown("#### Archetype Profiles")
        for _, row in arch_summary.iterrows():
            st.markdown(
                f"""
                - **{row['archetype']}** ({row['State_Count']} States)
                  - Mean Annual GST: ₹{row['Avg_GST_Cr']:,.0f} Cr
                  - Mean GSDP Per Capita: ₹{row['Avg_GSDP_PC']:,.0f}
                """
            )
            
    st.markdown("#### Fiscal Anomaly Detection Engine (Isolation Forest)")
    st.caption("Identifies extreme deviations, reporting delays, and exogenous disruption months")
    st.dataframe(
        anomalies_df.head(10).rename(columns={
            "date": "Month",
            "state_name_clean": "State",
            "total_gst": "Collection (₹ Cr)",
            "pct_dev_from_rolling": "% Dev from 3M Rolling",
            "fiscal_year": "Fiscal Year"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================
# TAB 5: STOCHASTIC MONTE CARLO SIMULATOR
# ==============================================================
with tabs[4]:
    st.subheader("Stochastic Monte Carlo Fiscal Simulation Engine")
    st.write("Simulates 1,000 forward paths to project revenue probability envelopes, downside risk, and 95% Value-at-Risk (VaR).")
    
    col_sim_ctrl, col_sim_view = st.columns([1, 2])
    with col_sim_ctrl:
        st.markdown("#### Policy & Macro Assumptions")
        sim_gdp_growth = st.slider("Projected Real GDP Growth (%)", 4.0, 9.0, 6.5, 0.25)
        sim_inflation = st.slider("Projected Inflation Deflator (%)", 3.0, 7.5, 4.5, 0.25)
        sim_compliance = st.slider("E-Invoicing Compliance Dividend (%)", -2.0, 4.0, 1.5, 0.25)
        sim_cess = st.select_slider("Compensation Cess Policy Factor", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=1.0)
        
        current_base_cr = float(macro_panel[macro_panel["fiscal_year"] == "2023-24"]["annualized_total_gst"].sum())
        sim_result = run_monte_carlo_fiscal_simulation(
            base_annual_collection=current_base_cr,
            expected_gdp_growth_pct=sim_gdp_growth,
            expected_inflation_pct=sim_inflation,
            compliance_efficiency_delta_pct=sim_compliance,
            cess_policy_factor=sim_cess,
            n_simulations=1000
        )
        
        st.markdown("---")
        st.markdown(f"**Base Annual Collection:** ₹{sim_result['base_annual_collection_cr']:,.0f} Cr")
        st.markdown(f"**Mean Projected Revenue:** ₹{sim_result['mean_projected_cr']:,.0f} Cr")
        st.markdown(f"**95% VaR Downside Risk:** ₹{sim_result['var_95_drawdown_cr']:,.0f} Cr")
        
    with col_sim_view:
        sim_vals = sim_result["simulated_values"]
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=sim_vals,
            nbinsx=35,
            marker_color="#0284C7",
            opacity=0.75,
            name="Simulated Revenue Distribution"
        ))
        fig_mc.add_vline(x=sim_result["p05_downside_cr"], line_dash="dash", line_color="#DC2626", annotation_text=f"5th Percentile (₹{sim_result['p05_downside_cr']:,.0f} Cr)")
        fig_mc.add_vline(x=sim_result["mean_projected_cr"], line_dash="solid", line_color="#059669", annotation_text=f"Expected Mean (₹{sim_result['mean_projected_cr']:,.0f} Cr)")
        style_fig(fig_mc, title="Forward Revenue Probability Distribution (1,000 Monte Carlo Paths)", height=450)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        st.markdown(f"""
        <div class="card-insight">
            <b>Value-at-Risk (VaR 95%):</b> There is a 90% probability that forward annual collections fall between <b>₹{sim_result['p05_downside_cr']:,.0f} Cr</b> and <b>₹{sim_result['p95_upside_cr']:,.0f} Cr</b> under the chosen macroeconomic parameters.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Fiscal & Macro Intelligence Platform | Quantitative Econometric Engine")
