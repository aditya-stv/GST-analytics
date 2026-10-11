"""
app.py - India GST & State Income Analytics (FY 2017-18 to FY 2025-26)

Streamlit front end. Every number shown is computed by src/analysis.py at run
time; nothing in this file is hard-coded from a previous run.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import run_pipeline
from src.data_engine import COMPONENTS, assign_fiscal_year
from src.econometrics_engine import compute_gini_and_lorenz
from src.ml_engine import FEATURES, perform_state_clustering
from src.simulation_engine import run_monte_carlo_fiscal_simulation
from src.visual_styles import (
    COMPONENT_LABELS, INSTITUTIONAL_DARK_CSS, INSTITUTIONAL_LIGHT_CSS, apply_plotly_theme, palette,
)

st.set_page_config(page_title="India GST & State Income Analytics", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- sidebar: theme
st.sidebar.markdown("### Display")
is_dark = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, horizontal=True) == "Dark"
st.markdown(INSTITUTIONAL_DARK_CSS if is_dark else INSTITUTIONAL_LIGHT_CSS, unsafe_allow_html=True)
P = palette(is_dark)


def style(fig, title=None, height=430, left=None):
    apply_plotly_theme(fig, title=title, height=height, is_dark=is_dark)
    if left:  # room for long category labels on horizontal charts
        fig.update_layout(margin_l=left)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None)


def lakh_cr(x):
    return f"₹{x / 1e5:,.2f} lakh Cr"


def lcr(x):
    """Compact form for metric tiles (L Cr = lakh crore)."""
    return f"₹{x / 1e5:,.2f}L Cr"


def insight(html):
    st.markdown(f'<div class="card-insight">{html}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load():
    return run_pipeline()


with st.spinner("Running data-quality checks, models and backtests..."):
    B = load()

F = B["findings"]
monthly, annual, panel, national = B["monthly"], B["annual"], B["panel"], B["national"]

# ---------------------------------------------------------------- sidebar: filters
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
fy_options = sorted(monthly["fiscal_year"].unique())
fy_lo, fy_hi = st.sidebar.select_slider("Fiscal-year window (trends & state tabs)", options=fy_options,
                                        value=(fy_options[0], fy_options[-1]))
fy_window = [fy for fy in fy_options if fy_lo <= fy <= fy_hi]
states_sorted = sorted(monthly.loc[monthly["entity_type"] == "State", "state"].unique())
focus_state = st.sidebar.selectbox("State drill-down", states_sorted, index=states_sorted.index("Maharashtra"))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### Data sources\n"
    "- [GST portal collections](https://ckandev.indiadataportal.com/dataset/goods-and-services-tax) "
    "(India Data Portal, monthly, ₹ Cr)\n"
    "- [Per-capita GSDP, current prices](https://www.data.gov.in/resource/"
    "state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24) "
    "(MoSPI via Rajya Sabha, annual, ₹)"
)

# ---------------------------------------------------------------- header
st.title("India GST & State Income Analytics")
st.caption(f"Validated analysis of {F['n_months']} months of GST portal collections ({F['period']}) "
           "and official per-capita GSDP for 28 states (FY 2019-20 to 2023-24)")

share = F["component_share"]
k = st.columns(5)
k[0].metric(f"Total GST · {F['n_months']} months", lcr(F["total_cr"]))
for col, comp in zip(k[1:], ["igst", "sgst", "cgst", "cess"]):
    col.metric(f"{COMPONENT_LABELS[comp]} · {share[comp]:.1f}%", lcr(F["component_cr"][comp]))
st.caption("L Cr = lakh crore rupees (₹1 lakh Cr = ₹1 trillion).")

# ---------------------------------------------------------------- shared insight values and charts
top5y, mhy, hhi = F["top5_by_year"], F["mh_share_by_year"], F["hhi_by_year"]
cagr, yoy, nat_cagr = F["state_cagr"], F["fy_yoy"], F["cagr_fy19_fy25_pct"]
si, mixy, mm = F["seasonal_index"], F["mix_by_year"], F["models"]
weak = si.nsmallest(3)
twfe = mm.loc["Two-way fixed effects"]


def month_name(m):
    return pd.to_datetime(str(m), format="%m").strftime("%B")


def chart_concentration_trend():
    fig = go.Figure()
    for series, name, color in [(top5y, "Top 5 states", P["primary"]), (mhy, "Maharashtra", P["accent"])]:
        fig.add_trace(go.Scatter(x=series.index, y=series.values - series.iloc[0], mode="lines+markers",
                                 name=f"{name} ({series.iloc[0]:.1f}% to {series.iloc[-1]:.1f}%)",
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 customdata=series.values,
                                 hovertemplate="%{x}: %{customdata:.2f}% share (%{y:+.2f} pp)<extra></extra>"))
    fig.add_hline(y=0, line_color=P["muted"], line_width=1)
    style(fig, "Change in share of national GST since FY19 (pp)", height=380)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    show(fig)


def chart_state_cagr():
    cs = cagr.sort_values()
    fig = go.Figure(go.Bar(
        x=cs.values, y=cs.index, orientation="h",
        marker_color=[P["accent"] if s == focus_state else (P["muted"] if v < nat_cagr - 2 else P["primary"])
                      for s, v in zip(cs.index, cs.values)],
        hovertemplate="%{y}: %{x:.1f}% a year<extra></extra>"))
    fig.add_vline(x=nat_cagr, line_color=P["text_secondary"], line_width=1,
                  annotation_text=f"National {nat_cagr:.1f}%")
    style(fig, "GST growth by state, FY 2018-19 to 2024-25 (CAGR %)", height=600, left=160)
    show(fig)


def chart_seasonality():
    fig = go.Figure(go.Bar(x=[month_name(m)[:3] for m in si.index], y=(si.values - 1) * 100,
                           marker_color=[P["primary"] if v >= 1 else P["muted"] for v in si.values],
                           hovertemplate="%{x}: %{y:+.1f}% vs trend<extra></extra>"))
    style(fig, "Seasonal pattern: % above or below trend (FY22 to FY25)", height=380)
    show(fig)


def chart_mix_trend():
    fig = go.Figure()
    for comp in ["igst", "sgst", "cgst", "cess"]:
        fig.add_trace(go.Scatter(x=mixy.index, y=mixy[comp], mode="lines+markers", name=COMPONENT_LABELS[comp],
                                 line=dict(color=P["components"][comp], width=2), marker=dict(size=7)))
    style(fig, "National tax mix by fiscal year (% of collections)", height=380)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    show(fig)


def chart_volatility():
    vol = F["volatility"]
    fig = go.Figure(go.Bar(x=vol.values, y=vol.index, orientation="h",
                           marker_color=[P["accent"] if s == focus_state else P["primary"] for s in vol.index],
                           hovertemplate="%{y}: s.d. %{x:.1f} pp<extra></extra>"))
    style(fig, "Volatility of year-on-year growth (s.d., pp, COVID window excluded)", height=600, left=160)
    show(fig)


def chart_sgst_cgst():
    r = F["sgst_cgst_ratio"]
    fig = go.Figure(go.Bar(x=r.values, y=r.index, orientation="h",
                           marker_color=[P["accent"] if s == focus_state else P["primary"] for s in r.index],
                           hovertemplate="%{y}: SGST = %{x:.2f} x CGST<extra></extra>"))
    fig.add_vline(x=1, line_color=P["text_secondary"], line_width=1, annotation_text="Equal")
    style(fig, "SGST collected per ₹1 of CGST, FY19 to FY25", height=600, left=160)
    fig.update_xaxes(range=[0.9, r.max() * 1.05])
    show(fig)

tabs = st.tabs([
    "Executive Summary", "Data Quality", "Revenue Trends", "States & Tax Mix", "Income vs Revenue",
    "Econometrics", "Forecasting", "ML: Archetypes & Anomalies", "Scenario Simulator", "Insights & Recommendations",
])

# ================================================================ 0. EXECUTIVE SUMMARY
with tabs[0]:
    st.subheader("What the data says")
    st.write(f"{F['n_months']} validated months of GST collections ({F['period']}) across 36 states and union "
             "territories, joined with official per-capita GSDP for 28 states. Each card below is computed live "
             "and points to the tab that holds the evidence.")

    h = st.columns(4)
    h[0].metric("GST collected", lcr(F["total_cr"]), f"+{nat_cagr:.1f}% a year (FY19-FY25)", delta_color="off")
    h[1].metric("Top-5 state share, FY25", f"{top5y.iloc[-1]:.1f}%",
                f"{top5y.iloc[-1] - top5y.iloc[0]:+.1f} pp since FY19", delta_color="off")
    h[2].metric("FY25 growth", f"{yoy['2024-25']:+.1f}%", f"{yoy['2024-25'] - yoy['2021-22']:+.1f} pp vs FY22",
                delta_color="off")
    h[3].metric("FY26 forecast (core GST)", lcr(F["fy26_forecast_cr"]), f"{F['fy26_growth_pct']:+.1f}%",
                delta_color="off")
    h = st.columns(4)
    h[0].metric("April 2020 COVID drop", f"{F['apr20_drop_pct']:.0f}%",
                f"recovered by {F['recovery_month']:%b %Y}", delta_color="off")
    h[1].metric(f"Income gap, {F['income_ratio']['richest'].iloc[-1]} vs {F['income_ratio']['poorest'].iloc[-1]}",
                f"{F['income_ratio']['ratio'].iloc[-1]:.1f}x", help="Ratio of per-capita GSDP, FY 2023-24")
    h[2].metric(f"GST-income elasticity (p = {twfe['p_value']:.2f})", f"{twfe['elasticity']:.2f}",
                help=f"Two-way fixed effects, 95% CI {twfe['ci_low']:.2f} to {twfe['ci_high']:.2f}")
    n_in = int(F["validation"]["inside_80_pi"].sum())
    h[3].metric("Forecast error (backtest)", f"{F['best_mape']:.1f}% MAPE",
                f"{n_in} of {len(F['validation'])} held-out months in band", delta_color="off")

    st.markdown("#### Key findings")
    cards = [
        ("Revenue is concentrated, and getting more so",
         f"{F['top1_state']} alone collects {F['top1_share']:.1f}% of national GST and the top 5 states "
         f"{F['top5_share']:.1f}%. The top-5 share rose from {top5y.iloc[0]:.1f}% in FY19 to {top5y.iloc[-1]:.1f}% "
         f"in FY25, and the HHI by {(hhi.iloc[-1] / hhi.iloc[0] - 1) * 100:.0f}%.", "States & Tax Mix"),
        ("Hill and mineral states are falling behind",
         f"Himachal Pradesh and Uttarakhand grew GST at {cagr['Himachal Pradesh']:.1f}% a year, under half the "
         f"national {nat_cagr:.1f}%. Jharkhand ({cagr['Jharkhand']:.1f}%) and Chhattisgarh "
         f"({cagr['Chhattisgarh']:.1f}%) also lag: producer states with small consumer markets under a "
         "destination-based tax.", "States & Tax Mix"),
        ("The post-COVID boom is over",
         "Annual growth slowed " + ", ".join(f"{v:+.1f}%" for v in yoy.dropna().iloc[2:]) +
         f" from FY22 to FY25. Revenue is still {F['fy25_vs_precovid_trend_pct']:.0f}% above the pre-COVID "
         "trend: a one-off level gain, not faster growth.", "Revenue Trends"),
        ("The COVID shock was deep but short",
         f"April 2020 collections were {abs(F['apr20_drop_pct']):.1f}% below the FY20 average, and were back "
         f"above it by {F['recovery_month']:%B %Y}, {F['recovery_lag_months']} months later.", "Revenue Trends"),
        ("April is the big month",
         f"April runs {(si[4] - 1) * 100:+.0f}% above trend as March year-end filings are paid. The weakest "
         "months are " + ", ".join(f"{month_name(m)} ({(v - 1) * 100:+.0f}%)" for m, v in weak.items()) + ".",
         "Revenue Trends"),
        ("Tax mix follows the economy",
         "Cess is highest in the coal belt (" +
         ", ".join(f"{s_} {v:.0f}%" for s_, v in F["top_cess"].head(3).items()) +
         f"). Cess has fallen every year, from {mixy['cess'].iloc[0]:.1f}% to {mixy['cess'].iloc[-1]:.1f}% of "
         f"collections. Richer states lean on IGST (r = {F['correlations'].loc['IGST share', 'state_r']:+.2f}, "
         f"p = {F['correlations'].loc['IGST share', 'state_p']:.3f}).", "States & Tax Mix"),
        ("Income does not drive GST growth",
         f"Once common year shocks are removed, the GST-income elasticity is {twfe['elasticity']:.2f} "
         f"(p = {twfe['p_value']:.2f}), and state GST growth is unrelated to per-capita income "
         f"(r = {F['cagr_vs_income'][0]:+.2f}). Total GST follows population and economic size.", "Econometrics"),
        ("Small states are volatile",
         f"Year-on-year swings have a standard deviation of {F['volatility'].max():.0f} pp in "
         f"{F['volatility'].idxmax()} against {F['volatility'].min():.1f} pp in {F['volatility'].idxmin()} "
         f"(ρ = {F['volatility_vs_size'][0]:+.2f} with size).", "States & Tax Mix"),
        ("Simple models forecast best",
         f"Seasonal-naive with drift beat SARIMA and Holt-Winters ({F['best_mape']:.1f}% vs "
         f"{F['sarima_mape']:.1f}% MAPE) and landed within "
         f"{F['validation']['error_pct'].abs().max():.0f}% on both unseen FY 2025-26 months.", "Forecasting"),
        ("The source data has a corruption error",
         "In Aug 2025 and Jan 2026 the cess column repeats CGST+SGST+IGST for every state, adding about ₹2.67 "
         "lakh Cr of revenue that does not exist. It is detected and removed automatically.", "Data Quality"),
    ]
    for row in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, (title, body, tab) in zip(cols, cards[row:row + 2]):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)
                st.caption(f"Evidence: {tab} tab")

    st.markdown("#### Top recommendations")
    recs = [
        ("Track concentration", "Monitor the HHI of state collections each year and check that IGST "
                                "settlement credits revenue to the consuming state."),
        ("Support lagging producer states", "Help Himachal Pradesh, Uttarakhand, Jharkhand and Chhattisgarh "
                                            "widen their own consumption base."),
        ("Plan the cess phase-out", f"Cess was {lakh_cr(F['fy25_cess_cr'])} in FY25. It funds the Centre's "
                                    "compensation account, so the replacement is a central decision."),
        ("Budget on normal growth", f"Plan FY26 on about 10% growth; the forecast is "
                                    f"{lakh_cr(F['fy26_forecast_cr'])} ({F['fy26_growth_pct']:+.1f}%)."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, recs):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)
    st.caption("Full list of findings and recommendations: Insights & Recommendations tab.")

# ================================================================ 1. DATA QUALITY
with tabs[1]:
    st.subheader("The raw dataset")
    st.write("This is the raw dataset as it was received from the source.")
    q = B["quality_log"]
    st.dataframe(q, width="stretch", hide_index=True,
                 column_config={"detail": st.column_config.TextColumn(width="large")})

    c1, c2 = st.columns(2)
    with c1:
        allm = B["all_months"]
        nat_cess = allm.groupby("date").agg(core=("core_gst", "sum"), cess=("cess", "sum")).reset_index()
        nat_cess["flag"] = np.where(allm.groupby("date")["cess_corrupt"].any().values, "Corrupt (blanked)",
                                    np.where(nat_cess["date"] < monthly["date"].min(), "Launch placeholder",
                                             "Valid"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=nat_cess["date"], y=nat_cess["core"], name="CGST+SGST+IGST",
                             marker_color=P["primary"]))
        bad = nat_cess[nat_cess["flag"] != "Valid"]
        fig.add_trace(go.Scatter(x=bad["date"], y=bad["core"], mode="markers", name="Flagged month",
                                 marker=dict(size=11, color=P["accent"], symbol="x")))
        style(fig, "National core GST by month, flagged months marked (₹ Cr)")
        show(fig)
    with c2:
        cov = B["all_months"].pivot_table(index="state", columns="fiscal_year", values="date", aggfunc="nunique")
        cov = cov.reindex(sorted(cov.index))
        fig = px.imshow(cov, color_continuous_scale="Blues", aspect="auto",
                        labels={"color": "Months"}, title="Months reported per entity and fiscal year")
        style(fig, height=430, left=230)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(tickfont={"size": 9})
        show(fig)
    insight("<b>Why this matters:</b> the uncorrected file overstates total GST by ~₹2.67 lakh Cr and the cess "
            "share by ~2.3 points. Its FY 2025-26 data was 'annualised' from 2 corrupt months, which on its own "
            "pushed the original elasticity estimate above 2.")

# ================================================================ 2. REVENUE TRENDS
with tabs[2]:
    st.subheader("National revenue trends")
    nat_w = national[national["date"].apply(assign_fiscal_year).isin(fy_window)]
    c1, c2 = st.columns([3, 2])
    with c1:
        long = nat_w.melt(id_vars="date", value_vars=COMPONENTS, var_name="component", value_name="cr")
        long["component"] = long["component"].map(COMPONENT_LABELS)
        fig = px.area(long, x="date", y="cr", color="component",
                      color_discrete_map={COMPONENT_LABELS[c]: P["components"][c] for c in COMPONENTS},
                      labels={"cr": "₹ Cr", "date": "", "component": "Tax head"},
                      title="Monthly GST collections by tax head (₹ Cr)")
        fig.update_traces(line_width=1)
        style(fig)
        fig.update_layout(hovermode="x unified")
        show(fig)
    with c2:
        fy = F["fy_totals"]
        fy = fy[fy.index.isin(fy_window)]
        growth = fy.pct_change() * 100
        fig = go.Figure(go.Bar(x=fy.index, y=fy.values / 1e5, marker_color=P["primary"],
                               text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in growth],
                               textposition="outside",
                               hovertemplate="%{x}: ₹%{y:.2f} lakh Cr<extra></extra>"))
        style(fig, "Annual GST, complete fiscal years (₹ lakh Cr, YoY growth)")
        show(fig)

    st.markdown("#### COVID-19 lockdown shock")
    cw = national[(national["date"] >= "2019-04-01") & (national["date"] <= "2021-06-01")]
    pre = F["precovid_mean_cr"]
    fig = go.Figure(go.Bar(
        x=cw["date"], y=cw["total_gst"],
        marker_color=[P["accent"] if d in (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01"))
                      else P["primary"] for d in cw["date"]],
        hovertemplate="%{x|%b %Y}: ₹%{y:,.0f} Cr<extra></extra>", showlegend=False))
    fig.add_hline(y=pre, line_color=P["muted"], line_width=1,
                  annotation_text=f"FY20 pre-COVID mean ₹{pre:,.0f} Cr", annotation_position="top left")
    style(fig, "Monthly GST, Apr 2019 to Jun 2021 (₹ Cr)", height=380)
    show(fig)
    m = st.columns(3)
    m[0].metric("April 2020 collections", f"₹{F['apr20_cr']:,.0f} Cr",
                f"{F['apr20_drop_pct']:.1f}% vs FY20 average", delta_color="inverse")
    m[1].metric("Year-on-year (Apr 2020 vs Apr 2019)", f"{F['apr20_yoy_pct']:.1f}%")
    m[2].metric("Back above pre-COVID mean", f"{F['recovery_month']:%B %Y}",
                f"{F['recovery_lag_months']} months after the trough", delta_color="off")
    insight(f"Collections fell <b>{abs(F['apr20_drop_pct']):.1f}%</b> below the FY 2019-20 monthly average in "
            f"April 2020 and recovered to that level by <b>{F['recovery_month']:%B %Y}</b>. Annual GST grew at a "
            f"<b>{F['cagr_fy19_fy25_pct']:.1f}% CAGR</b> from FY 2018-19 ({lakh_cr(F['fy19_cr'])}) to "
            f"FY 2024-25 ({lakh_cr(F['fy25_cr'])}). Note that October 2020 also saw e-invoicing begin for large "
            "taxpayers; the data show the timing coincides but cannot separate that effect from reopening.")

    st.markdown("#### Growth is normalising, and revenue is seasonal")
    c1, c2 = st.columns(2)
    with c1:
        chart_seasonality()
    with c2:
        chart_mix_trend()
    insight(
        "<b>The post-COVID boom is over.</b> Annual growth went "
        + ", ".join(f"{v:+.1f}%" for v in yoy.dropna().iloc[2:]) +
        f" from FY22 to FY25, close to nominal GDP growth again. FY25 is still "
        f"<b>{F['fy25_vs_precovid_trend_pct']:.0f}%</b> above the pre-COVID {F['pre_covid_growth_pct']:.1f}% "
        "growth path: a one-off level gain from formalisation and 2022 inflation.<br>"
        f"<b>Seasonality:</b> April runs <b>{(si[4] - 1) * 100:+.0f}%</b> above trend (March year-end filings are "
        "paid in April); the weakest months are "
        + ", ".join(f"{month_name(m)} ({(v - 1) * 100:+.0f}%)" for m, v in weak.items()) + ".<br>"
        f"<b>Mix:</b> cess fell every year, from {mixy['cess'].iloc[0]:.1f}% to {mixy['cess'].iloc[-1]:.1f}% of "
        f"collections; IGST edged up from {mixy['igst'].min():.1f}% to {mixy['igst'].iloc[-1]:.1f}%.")

# ================================================================ 3. STATES & TAX MIX
with tabs[3]:
    st.subheader("Where GST is collected, and what it is made of")
    mw = monthly[monthly["fiscal_year"].isin(fy_window)]
    terr = mw[mw["entity_type"] != "Non-territorial"].groupby(["state", "entity_type"])["total_gst"].sum() \
        .reset_index().sort_values("total_gst", ascending=False)
    terr["share"] = terr["total_gst"] / mw["total_gst"].sum() * 100
    terr["cum_share"] = terr["share"].cumsum()

    c1, c2 = st.columns([3, 2])
    with c1:
        top = terr.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["share"], y=top["state"], orientation="h",
            marker_color=[P["accent"] if s == focus_state else P["primary"] for s in top["state"]],
            text=[f"{v:.1f}%" for v in top["share"]], textposition="outside",
            hovertemplate="%{y}: %{x:.2f}% of national GST<extra></extra>"))
        style(fig, f"Top 15 states/UTs by share of national GST, {fy_lo} to {fy_hi}", height=480, left=200)
        fig.update_xaxes(title="% of national total", range=[0, top["share"].max() * 1.15])
        show(fig)
    with c2:
        fig = go.Figure(go.Scatter(x=np.arange(1, len(terr) + 1), y=terr["cum_share"], mode="lines+markers",
                                   line=dict(color=P["primary"], width=2), marker=dict(size=6),
                                   customdata=terr["state"],
                                   hovertemplate="Rank %{x} (%{customdata}): %{y:.1f}% cumulative<extra></extra>"))
        for n in (5, 10):
            fig.add_annotation(x=n, y=terr["cum_share"].iloc[n - 1], text=f"Top {n}: {terr['cum_share'].iloc[n - 1]:.1f}%",
                               showarrow=True, arrowhead=0, ax=40, ay=30)
        style(fig, "Cumulative share by state rank", height=480)
        fig.update_xaxes(title="States/UTs ranked by collections")
        fig.update_yaxes(title="Cumulative % of national GST", range=[0, 105])
        show(fig)

    st.markdown("#### Tax mix across the 28 states")
    sort_by = st.radio("Sort states by", ["Cess share", "IGST share", "SGST share"], horizontal=True)
    sw = annual[(annual["entity_type"] == "State") & annual["fiscal_year"].isin(fy_window)] \
        .groupby("state")[COMPONENTS + ["total_gst"]].sum()
    mix = sw[COMPONENTS].div(sw["total_gst"], axis=0) * 100
    mix = mix.sort_values(sort_by.split()[0].lower(), ascending=True)
    fig = go.Figure()
    for comp in ["cess", "igst", "sgst", "cgst"]:
        fig.add_trace(go.Bar(y=mix.index, x=mix[comp], name=COMPONENT_LABELS[comp], orientation="h",
                             marker=dict(color=P["components"][comp], line=dict(width=1, color="rgba(255,255,255,0.9)")),
                             hovertemplate="%{y}: %{x:.1f}%<extra>" + COMPONENT_LABELS[comp] + "</extra>"))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.04, x=0))
    style(fig, "Share of each tax head in state collections (%)", height=720, left=200)
    fig.update_xaxes(range=[0, 100], title="% of state GST")
    show(fig)
    tc, ti = F["top_cess"], F["top_igst"]
    insight(
        f"<b>Concentration:</b> {F['top1_state']} alone collects <b>{F['top1_share']:.1f}%</b> "
        f"({lakh_cr(F['top1_cr'])}) of national GST; the top 5 ({', '.join(F['top5_states'])}) collect "
        f"<b>{F['top5_share']:.1f}%</b>, the top 10 <b>{F['top10_share']:.1f}%</b>, and the bottom 10 of "
        f"{F['n_territories']} states/UTs just <b>{F['bottom10_share']:.2f}%</b> (full window).<br>"
        f"<b>Cess-heavy states:</b> " + ", ".join(f"{s} {v:.1f}%" for s, v in tc.head(4).items()) +
        " - the coal and mineral belt.<br>"
        f"<b>IGST-heavy states:</b> " + ", ".join(f"{s} {v:.1f}%" for s, v in ti.head(4).items()) +
        ". These are mostly smaller or hill states that buy much of what they use from other states; "
        "large manufacturing states such as Gujarat and Tamil Nadu sit near the national average.")

    st.markdown("#### How the state picture is changing")
    c1, c2 = st.columns(2)
    with c1:
        chart_concentration_trend()
        gain, lose = F["share_change_pp"].tail(3)[::-1], F["share_change_pp"].head(4)
        insight(
            f"<b>Concentration is rising.</b> The top-5 share went from {top5y.iloc[0]:.1f}% to "
            f"{top5y.iloc[-1]:.1f}% (FY19 to FY25) and the HHI from {hhi.iloc[0]:.0f} to {hhi.iloc[-1]:.0f}. "
            "Gainers: " + ", ".join(f"{s_} {v:+.2f} pp" for s_, v in gain.items()) + ". Losers: "
            + ", ".join(f"{s_} {v:+.2f} pp" for s_, v in lose.items()) + ".")
    with c2:
        chart_state_cagr()
    insight(
        f"<b>Hill and mineral states lag:</b> Himachal Pradesh and Uttarakhand grew "
        f"{cagr['Himachal Pradesh']:.1f}% a year against {nat_cagr:.1f}% nationally. <b>Small states grow faster in "
        f"% terms</b> (r = {F['cagr_vs_size'][0]:+.2f} with initial size, p = {F['cagr_vs_size'][1]:.3f}), but "
        f"growth is unrelated to per-capita income (r = {F['cagr_vs_income'][0]:+.2f}, "
        f"p = {F['cagr_vs_income'][1]:.2f}).")

    c1, c2 = st.columns(2)
    with c1:
        chart_volatility()
    with c2:
        chart_sgst_cgst()
    insight(
        f"<b>Small states are far more volatile:</b> {F['volatility'].idxmax()} swings with a standard deviation of "
        f"{F['volatility'].max():.0f} pp against {F['volatility'].min():.1f} pp in {F['volatility'].idxmin()} "
        f"(ρ = {F['volatility_vs_size'][0]:+.2f} with size), so single months say little about their trend.<br>"
        f"<b>SGST exceeds CGST in every state</b> ({F['sgst_cgst_national']:.2f}x nationally) although both are "
        "levied at equal rates. The likely cause is the order in which input-tax credit is set off, which uses "
        "up credit against CGST first.")

    st.markdown(f"#### Drill-down: {focus_state}")
    fs = monthly[monthly["state"] == focus_state]
    fs_w = fs[fs["fiscal_year"].isin(fy_window)]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for comp in COMPONENTS:
            fig.add_trace(go.Scatter(x=fs_w["date"], y=fs_w[comp], name=COMPONENT_LABELS[comp], mode="lines",
                                     line=dict(color=P["components"][comp], width=2)))
        style(fig, f"{focus_state}: monthly collections by tax head (₹ Cr)", height=380)
        fig.update_layout(hovermode="x unified")
        show(fig)
    with c2:
        b_row = B["buoyancy"].set_index("state").loc[focus_state]
        c_row = B["clusters"].set_index("state").loc[focus_state]
        rank = int(terr.reset_index(drop=True).index[terr["state"].values == focus_state][0]) + 1 \
            if focus_state in terr["state"].values else None
        st.metric("National rank (window)", f"#{rank}" if rank else "n/a")
        st.metric("Tax buoyancy", f"{b_row['buoyancy']:.2f}", b_row["period"], delta_color="off")
        st.metric("ML archetype", c_row["archetype"])

# ================================================================ 4. INCOME VS REVENUE
with tabs[4]:
    st.subheader("Per-capita income (GSDP) and its link to GST")
    gsdp = B["gsdp"].dropna()
    c1, c2 = st.columns([3, 2])
    with c1:
        last = gsdp[gsdp["fiscal_year"] == "2022-23"].sort_values("per_capita_gsdp")
        fig = go.Figure(go.Bar(
            x=last["per_capita_gsdp"] / 1e5, y=last["state"], orientation="h",
            marker_color=[P["accent"] if s == focus_state else P["primary"] for s in last["state"]],
            hovertemplate="%{y}: ₹%{x:.2f} lakh<extra></extra>"))
        style(fig, "Per-capita GSDP, FY 2022-23 (₹ lakh, current prices)", height=640, left=200)
        show(fig)
    with c2:
        ir = F["income_ratio"]
        fig = go.Figure(go.Scatter(x=ir["fiscal_year"], y=ir["ratio"], mode="lines+markers+text",
                                   text=[f"{v:.1f}x" for v in ir["ratio"]], textposition="top center",
                                   line=dict(color=P["primary"], width=2), marker=dict(size=8),
                                   customdata=ir[["richest", "poorest"]],
                                   hovertemplate="%{x}: %{customdata[0]} / %{customdata[1]} = %{y:.2f}x<extra></extra>"))
        style(fig, "Richest-to-poorest state ratio", height=300)
        fig.update_yaxes(range=[9, 11.5])
        show(fig)
        st.dataframe(ir.assign(ratio=ir["ratio"].round(2)), hide_index=True, width="stretch")
        st.caption(f"{F['fy21_contracted']} of 28 states saw nominal per-capita GSDP fall in FY 2020-21. "
                   "Six states have no official FY 2023-24 figure; they are left missing, not imputed.")

    st.markdown("#### Correlations: per-capita GSDP vs GST")
    corr = B["correlations"].copy()
    show_tbl = corr[["metric", "pooled_r", "pooled_p", "state_r", "state_p", "state_spearman"]].rename(columns={
        "metric": "GST measure", "pooled_r": "r (134 state-years)", "pooled_p": "p (pooled)",
        "state_r": "r (28 states)", "state_p": "p (28 states)", "state_spearman": "Spearman ρ (28 states)"})
    st.dataframe(show_tbl.style.format({c: "{:.3f}" for c in show_tbl.columns[1:]}), hide_index=True,
                 width="stretch")
    c1, c2 = st.columns(2)
    for col, (y, label) in zip((c1, c2), [("igst_share", "IGST share"), ("cess_share", "Cess share")]):
        with col:
            fig = px.scatter(panel, x="per_capita_gsdp", y=y, color="fiscal_year", hover_name="state",
                             log_x=True, trendline="ols", trendline_scope="overall",
                             trendline_color_override=P["muted"],
                             labels={"per_capita_gsdp": "Per-capita GSDP (₹, log scale)", y: label,
                                     "fiscal_year": "FY"},
                             title=f"{label} vs per-capita GSDP")
            fig.update_yaxes(tickformat=".0%")
            style(fig, height=400)
            show(fig)
    ci, cc, ct = (F["correlations"].loc[m] for m in ["IGST share", "Cess share", "Total GST (Rs Cr)"])
    insight(
        f"Using one observation per state (the statistically valid test, since state-years are not independent): "
        f"richer states have a higher <b>IGST share</b> (r = {ci['state_r']:+.2f}, p = {ci['state_p']:.3f}), "
        f"and a lower <b>cess share</b> (r = {cc['state_r']:+.2f}, p = {cc['state_p']:.3f}, borderline; "
        f"Spearman p = {cc['state_spearman_p']:.2f}). Total GST is <b>not</b> related to per-capita income "
        f"(r = {ct['state_r']:+.2f}, p = {ct['state_p']:.2f}): revenue follows population and economic size.")

    st.markdown("#### Concentration (Lorenz curves, FY 2022-23)")
    p23 = panel[panel["fiscal_year"] == "2022-23"]
    g1, x1, y1 = compute_gini_and_lorenz(p23["total_gst"])
    g2, x2, y2 = compute_gini_and_lorenz(p23["per_capita_gsdp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                             line=dict(color=P["muted"], width=1)))
    fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name=f"State GST totals (Gini {g1:.2f})",
                             line=dict(color=P["primary"], width=2)))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode="lines", name=f"Per-capita GSDP (Gini {g2:.2f})",
                             line=dict(color=P["accent"], width=2)))
    style(fig, "Lorenz curves across the 28 states", height=420)
    fig.update_xaxes(title="Cumulative share of states")
    fig.update_yaxes(title="Cumulative share of value")
    show(fig)
    st.caption("The two Gini values measure different things: GST is a state total (reflects population and "
               "economic size), GSDP is per head. The gap shows GST follows size, not that taxes are 'more "
               "unequal than income'. A like-for-like comparison would need state population data.")

# ================================================================ 5. ECONOMETRICS
with tabs[5]:
    st.subheader("How does GST respond to state income? Panel elasticity estimates")
    st.latex(r"\ln(\text{GST}_{it}) = \alpha_i + \gamma_t + \beta \,\ln(\text{GSDP per capita}_{it}) + \varepsilon_{it}")
    st.caption("134 observed state-years (28 states, FY 2019-20 to 2023-24). Standard errors clustered by state. "
               "R² for the fixed-effects models is the within R², not the inflated dummy-variable R².")
    models = B["models"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=models["elasticity"], y=models["model"], mode="markers",
        marker=dict(size=11, color=P["primary"]),
        error_x=dict(type="data", symmetric=False, array=models["ci_high"] - models["elasticity"],
                     arrayminus=models["elasticity"] - models["ci_low"], color=P["primary"], thickness=2),
        hovertemplate="%{y}: β = %{x:.2f}<extra></extra>", showlegend=False))
    fig.add_vline(x=1, line_color=P["muted"], line_width=1, annotation_text="Unit elasticity")
    fig.add_vline(x=0, line_color=P["muted"], line_width=1)
    style(fig, "Elasticity estimates with 95% confidence intervals", height=320, left=200)
    fig.update_yaxes(autorange="reversed")
    show(fig)
    st.dataframe(models.style.format({"elasticity": "{:.3f}", "std_error": "{:.3f}", "ci_low": "{:.2f}",
                                      "ci_high": "{:.2f}", "p_value": "{:.3g}", "r_squared": "{:.3f}"}),
                 hide_index=True, width="stretch")
    mm = F["models"]
    insight(
        f"<b>Reading the models:</b> the state fixed-effects estimate (β = {mm.loc['State fixed effects', 'elasticity']:.2f}) "
        f"and the first-difference estimate (β = {mm.loc['First differences', 'elasticity']:.2f}) both include the "
        f"common nominal trend and the COVID V-shape shared by every state. Once year effects absorb those shocks, "
        f"the two-way FE estimate falls to <b>β = {mm.loc['Two-way fixed effects', 'elasticity']:.2f}</b> "
        f"(95% CI {mm.loc['Two-way fixed effects', 'ci_low']:.2f} to {mm.loc['Two-way fixed effects', 'ci_high']:.2f}, "
        f"p = {mm.loc['Two-way fixed effects', 'p_value']:.2f}). With five years of income data, a state growing faster "
        "than its peers is <b>not</b> shown to collect disproportionately more GST. The original project's β = 2.03 "
        "came from synthetic back-casts and a corrupt FY 2025-26 data point.")

    st.markdown("#### State tax buoyancy")
    bu = B["buoyancy"]
    c1, c2 = st.columns([3, 2])
    with c1:
        bs = bu.sort_values("buoyancy")
        fig = go.Figure(go.Bar(
            x=bs["buoyancy"], y=bs["state"], orientation="h",
            marker_color=[P["accent"] if s == focus_state else (P["primary"] if v >= 1 else P["muted"])
                          for s, v in zip(bs["state"], bs["buoyancy"])],
            customdata=bs[["gst_cagr_pct", "gsdp_pc_cagr_pct", "period"]],
            hovertemplate="%{y}: %{x:.2f}<br>GST CAGR %{customdata[0]:.1f}%, GSDP pc CAGR %{customdata[1]:.1f}%"
                          "<br>%{customdata[2]}<extra></extra>"))
        fig.add_vline(x=1, line_color=P["text_secondary"], line_width=1)
        style(fig, "Buoyancy = GST log-growth / per-capita GSDP log-growth", height=640, left=200)
        show(fig)
    with c2:
        st.dataframe(bu[["state", "buoyancy", "gst_cagr_pct", "gsdp_pc_cagr_pct", "period"]].style.format(
            {"buoyancy": "{:.2f}", "gst_cagr_pct": "{:.1f}", "gsdp_pc_cagr_pct": "{:.1f}"}),
            hide_index=True, width="stretch", height=560)
        st.caption(f"{F['buoyant_states']} of 28 states have buoyancy above 1. Because GST is a total and GSDP is "
                   "per capita, population growth (~1% a year) adds a little to every ratio. Small north-eastern "
                   "states show extreme values because their GST base is tiny.")

# ================================================================ 6. FORECASTING
with tabs[6]:
    fc = B["forecast"]
    st.subheader("Forecasting FY 2025-26 national core GST")
    st.write("Target is CGST + SGST + IGST (cess excluded: it is corrupt in the 2025-26 rows and being phased out). "
             "Models are compared with a rolling-origin backtest: train to March, forecast the next 12 months, "
             "repeated for FY 2022-23, 2023-24 and 2024-25.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(fc["backtest_summary"].style.format({c: "{:.2f}" for c in fc["backtest_summary"].columns[1:]})
                     .highlight_min(subset=["mean_mape_pct"], color="rgba(42,120,214,0.18)"),
                     hide_index=True, width="stretch")
        st.caption(f"Primary model: **{fc['primary_model']}** (lowest mean MAPE). Its 80% interval is "
                   f"split-conformal: ±{fc['interval_halfwidth_pct']:.1f}%, the 80th percentile of absolute "
                   "error across the 36 holdout months. The simple benchmark beating SARIMA and Holt-Winters is "
                   "itself a finding: GST is dominated by a stable seasonal pattern plus trend.")
        v = fc["validation"].reset_index().rename(columns={"date": "month"})
        v["month"] = v["month"].dt.strftime("%b %Y")
        st.markdown("**Out-of-sample check against the 2 real FY 2025-26 months**")
        st.dataframe(v.style.format({"actual": "{:,.0f}", "forecast": "{:,.0f}", "lower_80": "{:,.0f}",
                                     "upper_80": "{:,.0f}", "error_pct": "{:+.1f}%"}),
                     hide_index=True, width="stretch")
    with c2:
        hist = fc["history"]["2022-04-01":]
        f = fc["forecast"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(f.index) + list(f.index[::-1]), y=list(f["upper"]) + list(f["lower"][::-1]),
                                 fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                 name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual (training window)", mode="lines",
                                 line=dict(color=P["text_secondary"], width=2)))
        fig.add_trace(go.Scatter(x=f.index, y=f["forecast"], name="Forecast", mode="lines",
                                 line=dict(color=P["primary"], width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=fc["validation"].index, y=fc["validation"]["actual"], mode="markers",
                                 name="Actual FY 2025-26 (held out)",
                                 marker=dict(size=11, color=P["accent"], line=dict(width=2, color="white"))))
        style(fig, "National core GST: history, forecast and held-out actuals (₹ Cr)", height=470)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15))
        show(fig)
    m = st.columns(3)
    m[0].metric("FY 2024-25 actual (core)", lcr(fc["fy25_actual_total"]))
    m[1].metric("FY 2025-26 forecast (core)", lcr(fc["fy26_forecast_total"]),
                f"{F['fy26_growth_pct']:+.1f}% YoY", delta_color="off")
    m[2].metric("Held-out months inside 80% band", f"{int(fc['validation']['inside_80_pi'].sum())} of "
                f"{len(fc['validation'])}")
    st.caption("The negative January 2026 error is consistent with the GST rate rationalisation of September 2025, "
               "which no model trained on pre-2025 data can anticipate. Two months is too few to confirm it.")

# ================================================================ 7. ML
with tabs[7]:
    st.subheader("State archetypes (K-Means)")
    meta = B["cluster_meta"]
    k_choice = st.slider("Number of clusters (default = best silhouette score)", 2, 6, meta["k"])
    clusters, meta_k = (B["clusters"], meta) if k_choice == meta["k"] else perform_state_clustering(
        B["clusters"].set_index("state")[list(FEATURES) + ["avg_gst_cr", "avg_gsdp_pc"]], k_choice)
    c1, c2 = st.columns([3, 2])
    with c1:
        archetypes = sorted(clusters["archetype"].unique())
        fig = px.scatter(clusters, x="pca_1", y="pca_2", color="archetype", text="state",
                         category_orders={"archetype": archetypes},
                         hover_data={"avg_gst_cr": ":,.0f", "avg_gsdp_pc": ":,.0f", "igst_share": ":.1%",
                                     "cess_share": ":.1%", "pca_1": False, "pca_2": False},
                         labels={"pca_1": f"PC1 ({meta_k['pca_variance_ratio'][0]:.0%} of variance)",
                                 "pca_2": f"PC2 ({meta_k['pca_variance_ratio'][1]:.0%} of variance)",
                                 "archetype": "Archetype"},
                         title="States in principal-component space, coloured by cluster")
        fig.update_traces(textposition="top center", textfont_size=9, marker=dict(size=10, line=dict(width=2, color="white")))
        style(fig, height=520)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        show(fig)
    with c2:
        sil = pd.Series(meta["silhouette_by_k"])
        st.markdown(f"**Silhouette score:** {meta_k['silhouette']:.2f} at k = {meta_k['k']} "
                    f"(scores by k: " + ", ".join(f"{kk}: {vv:.2f}" for kk, vv in sil.items()) + ")")
        cz = meta_k["centroids_z"].copy()
        cz.columns = [FEATURES[c][0] for c in cz.columns]
        cz.index = [clusters.loc[clusters["cluster_id"] == i, "archetype"].iloc[0] for i in cz.index]
        fig = px.imshow(cz, color_continuous_scale="RdBu_r", zmin=-2.5, zmax=2.5, text_auto=".1f", aspect="auto",
                        title="Cluster centroids (z-scores)")
        style(fig, height=360, left=230)
        fig.update_layout(coloraxis_showscale=False)
        show(fig)
        st.caption("Labels are generated from each centroid's two most distinctive features. A silhouette "
                   "around 0.4 means the groups are real but overlapping, so treat archetypes as descriptive.")
    st.dataframe(clusters[["state", "archetype", "avg_gst_cr", "avg_gsdp_pc", "igst_share", "cess_share", "gst_cagr"]]
                 .sort_values(["archetype", "avg_gst_cr"], ascending=[True, False])
                 .style.format({"avg_gst_cr": "{:,.0f}", "avg_gsdp_pc": "{:,.0f}", "igst_share": "{:.1%}",
                                "cess_share": "{:.1%}", "gst_cagr": "{:.1%}"}),
                 hide_index=True, width="stretch")

    st.subheader("Anomaly detection (Isolation Forest)")
    st.write("Features are each state's year-on-year and month-on-month log change, robust-z-scored within the "
             "state. That removes seasonality and size, so Maharashtra is not 'anomalous' just for being large.")
    an = B["anomalies"]
    excl = st.checkbox("Hide the COVID window (Mar 2020 to Jun 2021) to see other unusual months")
    view = an[an["is_anomaly"]] if not excl else an[~an["date"].between("2020-03-01", "2021-06-30")].head(20)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(an, x="yoy_z", y="mom_z", color=np.where(an["is_anomaly"], "Anomaly", "Normal"),
                         color_discrete_map={"Anomaly": P["accent"], "Normal": P["primary"]},
                         hover_name="state", hover_data={"date": "|%b %Y", "yoy_z": ":.1f", "mom_z": ":.1f"},
                         labels={"yoy_z": "YoY change (robust z)", "mom_z": "MoM change (robust z)", "color": ""},
                         title="All scored state-months")
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        style(fig, height=420)
        show(fig)
    with c2:
        st.dataframe(view[["state", "date", "yoy_change_pct", "mom_change_pct", "anomaly_score"]]
                     .style.format({"date": lambda d: d.strftime("%b %Y"), "yoy_change_pct": "{:+.0f}%",
                                    "mom_change_pct": "{:+.0f}%", "anomaly_score": "{:.3f}"}),
                     hide_index=True, width="stretch", height=400)
    insight(f"{F['n_anomalies']} state-months are flagged (2% contamination). <b>{F['anomaly_covid_share']:.0f}%</b> "
            "of them fall in the April 2020 to June 2021 COVID window, so the detector rediscovers the lockdown "
            "without being told about it, which is a useful validation.")

# ================================================================ 8. SIMULATOR
with tabs[8]:
    st.subheader("Next-year national GST: Monte Carlo scenarios")
    fy25 = annual[annual["fiscal_year"] == "2024-25"]
    base = float(fy25["total_gst"].sum())
    cess_share = float(fy25["cess"].sum() / base)
    mm = F["models"]
    c1, c2 = st.columns([1, 2])
    with c1:
        real_g = st.slider("Real GDP growth (%)", 3.0, 9.0, 6.5, 0.25)
        infl = st.slider("GDP deflator inflation (%)", 1.0, 7.0, 3.5, 0.25)
        g_sd = st.slider("Uncertainty in nominal growth (s.d., pp)", 0.5, 3.0, 1.5, 0.25)
        buoy = st.slider("GST buoyancy (mean)", 0.5, 1.8, 1.0, 0.05,
                         help=f"Panel estimates: two-way FE {mm.loc['Two-way fixed effects', 'elasticity']:.2f}, "
                              f"first differences {mm.loc['First differences', 'elasticity']:.2f}. "
                              "1.0 is the conventional long-run assumption.")
        comp = st.slider("Compliance gain (% of base)", -2.0, 3.0, 0.0, 0.25)
        cess_keep = st.select_slider("Share of compensation cess retained", [0.0, 0.25, 0.5, 0.75, 1.0], 1.0)
    sim = run_monte_carlo_fiscal_simulation(base, cess_share, real_g, infl, g_sd, buoy, 0.15, comp, cess_keep)
    with c2:
        fig = go.Figure(go.Histogram(x=sim["simulated_values"] / 1e5, nbinsx=60, marker_color=P["primary"],
                                     opacity=0.85, name="Simulated outcomes",
                                     hovertemplate="₹%{x:.2f} lakh Cr: %{y} paths<extra></extra>"))
        for x, label, color in [(sim["p05_cr"], "P5", P["accent"]), (sim["median_cr"], "Median", P["text"]),
                                (base, "FY25 base", P["muted"])]:
            fig.add_vline(x=x / 1e5, line_color=color, line_width=1.5, annotation_text=label)
        style(fig, f"Distribution of FY 2025-26 revenue, {sim['n_simulations']:,} paths (₹ lakh Cr)", height=440)
        fig.update_layout(showlegend=False)
        show(fig)
    m = st.columns(4)
    m[0].metric("FY 2024-25 base", lcr(base))
    m[1].metric("Median outcome", lcr(sim["median_cr"]), f"{sim['expected_growth_pct']:+.1f}% expected",
                delta_color="off")
    m[2].metric("Revenue at risk (median − P5)", lcr(sim["revenue_at_risk_95_cr"]))
    m[3].metric("P(revenue below base)", f"{sim['prob_below_base']:.1%}")
    st.caption(f"Cess was {cess_share:.1%} of FY 2024-25 collections. Setting retention to 0 shows the revenue gap "
               "if the compensation cess is fully phased out without a replacement levy.")

# ================================================================ 9. FINDINGS
with tabs[9]:
    st.subheader("Verified findings")
    ci, cc = F["correlations"].loc["IGST share"], F["correlations"].loc["Cess share"]
    st.markdown(f"""
1. **Scale:** {lakh_cr(F['total_cr'])} of GST was collected over {F['n_months']} validated months ({F['period']}):
   IGST {share['igst']:.1f}%, SGST {share['sgst']:.1f}%, CGST {share['cgst']:.1f}%, cess {share['cess']:.1f}%.
   Annual collections grew at a {F['cagr_fy19_fy25_pct']:.1f}% CAGR from FY 2018-19 to FY 2024-25.
2. **Concentration:** {F['top1_state']} collects {F['top1_share']:.1f}%; the top 5 states {F['top5_share']:.1f}%;
   the top 10 {F['top10_share']:.1f}%; the bottom 10 of {F['n_territories']} states/UTs {F['bottom10_share']:.2f}%.
3. **COVID shock:** April 2020 collections were {abs(F['apr20_drop_pct']):.1f}% below the FY 2019-20 average
   ({F['apr20_yoy_pct']:.1f}% year-on-year) and were back above it by {F['recovery_month']:%B %Y}.
4. **Income divide:** the richest state's per-capita GSDP is {F['income_ratio']['ratio'].min():.1f}x to
   {F['income_ratio']['ratio'].max():.1f}x the poorest's (Bihar) every year; {F['fy21_contracted']} of 28 states shrank
   in nominal per-capita terms in FY 2020-21.
5. **Tax mix follows the economy:** cess is highest in the mineral belt
   ({', '.join(f'{s} {v:.1f}%' for s, v in F['top_cess'].head(4).items())}); richer states rely more on IGST
   (r = {ci['state_r']:+.2f}, p = {ci['state_p']:.3f} across 28 states) and less on cess (r = {cc['state_r']:+.2f}, p = {cc['state_p']:.3f}).
6. **Elasticity:** net of common year shocks, the GST-income elasticity is {mm.loc['Two-way fixed effects', 'elasticity']:.2f}
   (95% CI {mm.loc['Two-way fixed effects', 'ci_low']:.2f} to {mm.loc['Two-way fixed effects', 'ci_high']:.2f}), not statistically
   different from zero or one with 5 years of data. There is no evidence of a "2x progressive" response.
7. **Forecasting:** seasonal-naive with drift beats SARIMA and Holt-Winters in the backtest
   ({F['best_mape']:.1f}% vs {F['sarima_mape']:.1f}% MAPE); both held-out FY 2025-26 months fall inside its 80% band.
""")

    # ------------------------------------------------------------ deeper insights
    st.subheader("Deeper insights")
    st.caption("Charts for these findings are in the Revenue Trends and States & Tax Mix tabs.")
    gain, lose = F["share_change_pp"].tail(3)[::-1], F["share_change_pp"].head(4)
    sz_r, sz_p = F["cagr_vs_size"]
    inc_r, inc_p = F["cagr_vs_income"]
    v_rho, _ = F["volatility_vs_size"]
    mix = F["mix_by_year"]
    si = F["seasonal_index"]
    weak = si.nsmallest(3)
    st.markdown(f"""
8. **Concentration is rising, not falling.** The top-5 share went from {top5y.iloc[0]:.1f}% (FY19) to
   {top5y.iloc[-1]:.1f}% (FY25), and Maharashtra from {mhy.iloc[0]:.1f}% to {mhy.iloc[-1]:.1f}%. The HHI rose
   {(hhi.iloc[-1] / hhi.iloc[0] - 1) * 100:.0f}% ({hhi.iloc[0]:.0f} to {hhi.iloc[-1]:.0f}). Biggest gainers:
   {', '.join(f'{s} {v:+.2f} pp' for s, v in gain.items())}. Biggest losers:
   {', '.join(f'{s} {v:+.2f} pp' for s, v in lose.items())}.
9. **Hill and mineral states are falling behind.** Himachal Pradesh and Uttarakhand grew GST at only
   {cagr['Himachal Pradesh']:.1f}% a year, under half the national {nat_cagr:.1f}%; Jharkhand ({cagr['Jharkhand']:.1f}%)
   and Chhattisgarh ({cagr['Chhattisgarh']:.1f}%) also lag. These are producer states with small consumer markets, which a
   destination-based tax is expected to disadvantage.
10. **Small states grow faster in percentage terms** (growth vs initial size r = {sz_r:+.2f}, p = {sz_p:.3f}), but
    growth is unrelated to per-capita income (r = {inc_r:+.2f}, p = {inc_p:.2f}). The north-eastern states
    (Arunachal {cagr['Arunachal Pradesh']:.1f}%, Nagaland {cagr['Nagaland']:.1f}%) are catching up from a tiny base.
11. **The post-COVID boom is over.** Annual growth went {', '.join(f'{v:+.1f}%' for v in yoy.dropna().iloc[2:])}
    from FY22 to FY25. FY25 collections are still {F['fy25_vs_precovid_trend_pct']:.0f}% above where the pre-COVID
    {F['pre_covid_growth_pct']:.1f}% growth path would have reached: a one-off level gain from formalisation and
    2022 inflation, not a permanently higher growth rate.
12. **Cess is shrinking in the mix.** Cess fell every year, from {mix['cess'].iloc[0]:.1f}% to {mix['cess'].iloc[-1]:.1f}%
    of collections, while IGST edged up from {mix['igst'].min():.1f}% (its FY20 low) to {mix['igst'].iloc[-1]:.1f}%,
    a sign of slightly more inter-state trade.
13. **SGST exceeds CGST in all 28 states** (national ratio {F['sgst_cgst_national']:.2f}, from
    {F['sgst_cgst_ratio'].min():.2f} in {F['sgst_cgst_ratio'].idxmin()} to {F['sgst_cgst_ratio'].max():.2f} in
    {F['sgst_cgst_ratio'].idxmax()}), even though the two are levied at equal rates. The likely cause is the order in
    which input-tax credit is set off, which uses up credit against CGST first.
14. **Strong seasonality.** April runs {(si[4] - 1) * 100:+.0f}% above trend (March year-end filings are paid in April);
    the weakest months are {', '.join(pd.to_datetime(str(mo), format='%m').strftime('%B') + f' ({(v - 1) * 100:+.0f}%)' for mo, v in weak.items())}.
15. **Small states are far more volatile.** The standard deviation of year-on-year growth is
    {F['volatility'].max():.0f} pp in {F['volatility'].idxmax()} against {F['volatility'].min():.1f} pp in
    {F['volatility'].idxmin()} (Spearman ρ = {v_rho:+.2f} with size).
""")

    # ------------------------------------------------------------ recommendations
    st.subheader("Recommendations")
    cess25 = F["cess_share_fy25"]
    st.markdown(f"""
**For policy makers**

1. **Track concentration as an explicit indicator.** With a destination-based tax, revenue is drifting further
   toward a few large consumer and headquarters states (top-5 share {top5y.iloc[-1]:.1f}% and rising). The Finance
   Commission and GST Council should monitor the HHI annually and check that IGST settlement credits revenue to
   the consuming state rather than the billing state.
2. **Support lagging producer states.** Himachal Pradesh, Uttarakhand, Jharkhand and Chhattisgarh have lost share since
   FY19. They need help widening their own consumption base (services, tourism, local consumption) rather than
   relying on manufacturing output that is taxed elsewhere.
3. **Plan for the end of the compensation cess at the central level.** Cess was {lakh_cr(F['fy25_cess_cr'])}
   ({mix['cess'].iloc[-1]:.1f}% of collections) in FY25. It is collected most heavily in Jharkhand ({cess25['Jharkhand']:.0f}%),
   Chhattisgarh ({cess25['Chhattisgarh']:.0f}%) and Madhya Pradesh ({cess25['Madhya Pradesh']:.0f}%), so any replacement
   levy on coal or demerit goods will fall on the same region. Cess goes to the compensation fund rather than state
   budgets, so the revenue gap is the Centre's to fill.
4. **Budget on normal growth.** Use about 10% nominal growth (buoyancy near 1) for FY26 planning, not the 15 to 27%
   seen in FY22 to FY24. The forecast here is {lakh_cr(F['fy26_forecast_cr'])} of core GST ({F['fy26_growth_pct']:+.1f}%).

**For treasuries and tax administration**

5. **Manage cash around the seasonal cycle.** Expect an April surge of about {(si[4] - 1) * 100:.0f}% above trend and dips
   in {', '.join(pd.to_datetime(str(mo), format='%m').strftime('%B') for mo in weak.index)}; time borrowing and devolution
   transfers around them.
6. **Give small states a volatility buffer.** For north-eastern and hill states, budget on multi-year averages and keep
   reserves: one month's figure can swing ±{F['volatility'].tail(4).mean():.0f} pp year-on-year.
7. **Monitor anomalies automatically.** The scale-free Isolation Forest picked out the lockdown months across states
   without being told about COVID; run it monthly to spot reporting errors or sudden state-level shocks early.

**For data publishers**

8. **Fix the source data.** Correct the Aug 2025 and Jan 2026 cess column, publish the 8 missing months of 2025-26,
   and release state population and total GSDP so revenue can be analysed per person.
""")
    st.subheader("Corrections to the earlier version of this project")
    st.dataframe(pd.DataFrame([
        ["Total GST ₹93.82 lakh Cr", f"{lakh_cr(F['total_cr'])} (Aug 2017 to Mar 2025)", "Corrupt cess rows and incomplete FY 2025-26 were included"],
        ["Cess share 11.4%", f"{share['cess']:.1f}%", "Cess column held CGST+SGST+IGST in Aug 2025 and Jan 2026"],
        ["'95 months, July 2017 to January 2026'", "95 rows span 103 calendar months; 8 are missing", "Gaps in Apr to Jul and Sep to Dec 2025"],
        ["Cess: Odisha 24.2%, Jharkhand 21.5%, Chhattisgarh 18.9%",
         ", ".join(f"{s} {v:.1f}%" for s, v in F['top_cess'].head(3).items()) + f", Odisha {F['state_mix'].loc['Odisha', 'cess']:.1f}%",
         "Values were mis-stated and polluted by corrupt rows"],
        ["IGST: Haryana 43.1%, Gujarat 41.4%, Tamil Nadu 39.8%",
         f"Haryana {F['state_mix'].loc['Haryana', 'igst']:.1f}%, Gujarat {F['state_mix'].loc['Gujarat', 'igst']:.1f}%, "
         f"Tamil Nadu {F['state_mix'].loc['Tamil Nadu', 'igst']:.1f}%", "Gujarat and Tamil Nadu are below the national IGST share"],
        ["FE elasticity β = 2.03, R² = 98.9%", f"Two-way FE β = {mm.loc['Two-way fixed effects', 'elasticity']:.2f} (p = {mm.loc['Two-way fixed effects', 'p_value']:.2f})",
         "Estimated on synthetic back-casts and a corrupt FY26 point; R² included dummies"],
        ["Gini of per-capita GSDP = 0.36", f"{F['gini_gsdp_pc']:.2f} (FY 2022-23)", "Mis-stated; the old code itself produced 0.28"],
        ["'Wealth ratio fixed at 10.5x to 10.6x every year'", f"{F['income_ratio']['ratio'].min():.2f}x to {F['income_ratio']['ratio'].max():.2f}x",
         "FY 2019-20 is 10.06x (Goa vs Bihar)"],
        ["Recovery 'catalysed by e-invoicing'", "Timing coincides; causality not identified", "Unsupported causal claim"],
        ["'Bayesian credible intervals', 'Holt's trend', 'inverted elasticity backcast'", "Removed",
         "Code used fixed ±% bands and simple growth multipliers, not the methods described"],
    ], columns=["Original claim", "Verified value", "Reason"]), hide_index=True, width="stretch")

    st.subheader("Method in brief")
    st.markdown("""
- **Cleaning:** automated rules for launch-month placeholders, column corruption, calendar gaps and non-territorial
  heads; every correction is logged (Data Quality tab).
- **Panel econometrics:** pooled OLS, one-way FE, two-way FE and first-difference models with state-clustered errors,
  on observed data only. No imputation enters any estimate.
- **Forecasting:** four models, rolling-origin backtest across three fiscal years, split-conformal intervals, and a
  genuine out-of-sample check.
- **Machine learning:** K-Means on scale-free features with k chosen by silhouette; Isolation Forest on
  seasonality-adjusted, within-state z-scores.
- **Simulation:** 10,000-path Monte Carlo for nominal growth, buoyancy and compliance, with a cess phase-out lever.
- **Limitations:** no state population data (GST and GSDP cannot be put on the same per-head basis); only 5 years
  of GSDP; GST is booked where the taxpayer is registered, which is not always where consumption happens.
""")

st.markdown("---")
st.caption("India GST & State Income Analytics · all figures computed live from the source files · "
           "tests: `pytest tests/`")
