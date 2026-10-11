# India GST & State Income Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gst-analytics.streamlit.app/)

An end-to-end analytics project on India's Goods and Services Tax. It takes 3,570 monthly
state-level GST records and official per-capita GSDP for 28 states, then:

- **validates the raw data** with automated rules (and finds a column-corruption error in the source)
- **measures** concentration, tax mix and the COVID-19 shock
- **estimates** how GST responds to state income with four panel models and clustered standard errors
- **forecasts** FY 2025-26 revenue after a rolling-origin backtest, then checks the forecast against real held-out months
- **segments** states with K-Means and **flags anomalies** with an Isolation Forest
- **simulates** next-year revenue under macro and policy scenarios (Monte Carlo)

Every number in the app and in this README is computed by the pipeline at run time
(`src/analysis.py`) and pinned by tests (`tests/`).

---

## Data

| Dataset | Source | Coverage | Unit |
|---|---|---|---|
| `tax-collection-on-gst-portal.csv` | [India Data Portal (CKAN)](https://ckandev.indiadataportal.com/dataset/goods-and-services-tax) | 95 monthly snapshots, Jul 2017 to Jan 2026, 39 entities | ₹ crore |
| `RS_Session_266_AU_1677_B_2.csv` | [data.gov.in (MoSPI, Rajya Sabha)](https://www.data.gov.in/resource/state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24) | 28 states, FY 2019-20 to 2023-24 | ₹ per person, current prices |

### Data-quality issues found and handled

| Issue | Rows | Handling |
|---|---|---|
| **Cess column contains CGST+SGST+IGST** in Aug 2025 and Jan 2026, for every entity (e.g. Maharashtra Aug 2025: 7,840.27 + 9,400.00 + 9,505.44 = 26,745.71 = "cess") | 76 | Cess set to missing; detected by rule, not hard-coded |
| July 2017 is a launch placeholder (₹53 Cr nationally vs ~₹70,000 Cr in later months) | 34 | Excluded |
| 8 calendar months missing (Apr to Jul and Sep to Dec 2025) | n/a | FY 2025-26 excluded from annual analysis; its 2 real months are used only as a forecast check |
| CBIC, OIDAR and "Other Territory" are centrally booked heads, not states | 182 | Kept in national totals, excluded from state rankings |
| 6 states have no official FY 2023-24 GSDP | 6 | Left missing; no imputation enters any estimate |

**Analysis window:** Aug 2017 to Mar 2025 (92 complete, validated months).

---

## Key findings

1. **Scale and mix.** ₹88.48 lakh crore was collected over the 92 months: IGST 35.4%, SGST 31.4%,
   CGST 24.2%, cess 9.1%. Annual collections grew at an 11.4% CAGR from FY 2018-19 (₹8.77 lakh Cr)
   to FY 2024-25 (₹16.76 lakh Cr).
2. **Concentration.** Maharashtra collects 20.3% of national GST. The top 5 states (Maharashtra,
   Karnataka, Gujarat, Tamil Nadu, Uttar Pradesh) collect 52.6%, the top 10 collect 75.4%, and the
   bottom 10 of 36 states/UTs collect 0.63%.
3. **COVID shock.** April 2020 collections (₹19,606 Cr) were 75.1% below the FY 2019-20 monthly
   average (−78.1% year on year). Collections were back above that average by October 2020, 6 months later.
4. **Income divide.** The richest state's per-capita GSDP is 10.1x to 10.6x Bihar's in every year.
   20 of 28 states shrank in nominal per-capita terms in FY 2020-21.
5. **Tax mix tracks the economy.** Cess is highest in the coal and mineral belt (Chhattisgarh 23.7%,
   Jharkhand 20.8%, Madhya Pradesh 19.1%, Odisha 18.7%). Across the 28 states, richer states rely more on
   IGST (r = +0.57, p = 0.002) and less on cess (r = −0.38, p = 0.046). Total GST is not related to
   per-capita income (r = +0.09, p = 0.65); it follows population and economic size.
6. **Elasticity.** Once year effects remove the common inflation trend and COVID swing, the GST-income
   elasticity is **0.51 (95% CI −0.11 to 1.12, p = 0.11)**. With 5 years of GSDP, there is no evidence that
   faster-growing states collect disproportionately more GST.

   | Model (134 state-years, state-clustered SE) | β | 95% CI | Within R² |
   |---|---|---|---|
   | Pooled OLS | 0.31 | −0.76 to 1.38 | 0.01 |
   | State fixed effects | 1.29 | 1.18 to 1.40 | 0.91 |
   | Two-way fixed effects (state + year) | 0.51 | −0.11 to 1.12 | 0.08 |
   | First differences | 1.51 | 1.32 to 1.70 | 0.71 |

7. **Forecasting.** In a 3-year rolling-origin backtest, a seasonal-naive-with-drift model beat SARIMA
   and Holt-Winters:

   | Model | Mean MAPE | Worst year MAPE |
   |---|---|---|
   | Seasonal naive | 13.3% | 17.6% |
   | **Seasonal naive + drift** | **6.6%** | 10.1% |
   | Holt-Winters (log, damped) | 9.3% | 12.8% |
   | SARIMA(1,1,1)(0,1,1)₁₂ + lockdown dummy | 8.2% | 12.4% |

   Forecast for FY 2025-26 core GST (CGST+SGST+IGST): ₹16.92 lakh Cr (+10.3%). Against the two real
   FY 2025-26 months, which the model never saw, the errors were −0.0% (Aug 2025) and −5.0% (Jan 2026),
   both inside the split-conformal 80% interval (±7.7%).
8. **Anomaly detection.** On seasonality-adjusted, within-state z-scores, the Isolation Forest flags
   45 state-months. All of them fall in the COVID window, so the detector recovers the lockdown
   without being told about it.

### Deeper insights

9. **Concentration is rising.** The top-5 states' share rose from 51.8% (FY19) to 54.1% (FY25), and
   Maharashtra's from 19.4% to 21.5%. The Herfindahl index rose 10% (806 to 886). Biggest gainers:
   Maharashtra +2.05 pp, Haryana +0.82 pp, Odisha +0.56 pp. Biggest losers: West Bengal −0.55 pp,
   Jharkhand −0.53 pp, Uttarakhand −0.49 pp, Chhattisgarh −0.40 pp.
10. **Hill and mineral states are falling behind.** Himachal Pradesh and Uttarakhand grew GST at only 5.3%
    a year, under half the national 11.4%; Jharkhand (7.5%) and Chhattisgarh (8.4%) also lag. These are
    producer states with small consumer markets, which a destination-based tax is expected to disadvantage.
11. **Small states grow faster in percentage terms** (growth vs initial size r = −0.51, p = 0.006). Growth is
    unrelated to per-capita income (r = +0.07, p = 0.74). Arunachal Pradesh and Nagaland grew about 20% a year
    from a tiny base.
12. **The post-COVID boom is over.** Annual growth slowed from +26.8% (FY22) to +20.7%, +15.0% and +10.0% (FY25).
    FY25 is still 22% above the pre-COVID 7.7% growth path: a one-off level gain from formalisation and
    2022 inflation, not a permanently higher growth rate.
13. **Cess is shrinking in the mix.** Its share fell every year, from 10.0% to 8.5%, while IGST edged up
    from 33.8% (FY20) to 36.0%.
14. **SGST exceeds CGST in all 28 states** (1.30x nationally; 1.19x in Maharashtra to 1.73x in Bihar), even
    though the two are levied at equal rates. The likely cause is the order in which input-tax credit is set
    off, which uses up credit against CGST first.
15. **Strong seasonality.** April runs 27% above trend because March year-end filings are paid in April.
    The weakest months are August (−8%), December (−6%) and September (−5%).
16. **Small states are far more volatile.** The standard deviation of year-on-year growth is 65 pp in Mizoram
    against 8.7 pp in Uttar Pradesh (Spearman ρ = −0.70 with state size).

---

## Recommendations

**Policy**

1. **Track concentration as an explicit indicator.** Revenue is drifting toward a few large consumer and
   headquarters states. The Finance Commission and GST Council should monitor the HHI annually and check that
   IGST settlement credits revenue to the consuming state rather than the billing state.
2. **Support lagging producer states.** Himachal Pradesh, Uttarakhand, Jharkhand and Chhattisgarh need help
   widening their own consumption base (services, tourism) rather than relying on output that is taxed
   where it is consumed.
3. **Plan the cess phase-out at the central level.** Cess was ₹1.42 lakh Cr (8.5%) in FY25 and is collected
   most heavily in Jharkhand (21%), Chhattisgarh (20%) and Madhya Pradesh (18%). It goes to the compensation
   fund, not state budgets, so the gap falls on the Centre. Any replacement levy on coal or demerit goods
   will land on the same region.
4. **Budget on normal growth.** Plan FY26 on about 10% nominal growth (buoyancy near 1), not the 15 to 27% of
   FY22 to FY24. This project's forecast is ₹16.92 lakh Cr of core GST (+10.3%).

**Treasury and tax administration**

5. **Manage cash around the seasonal cycle.** Time borrowing and devolution transfers around the April surge
   (+27%) and the August, December and September dips.
6. **Give small states a volatility buffer.** For north-eastern and hill states, budget on multi-year averages
   and hold reserves; single-month year-on-year swings average around ±50 pp.
7. **Monitor anomalies automatically.** The scale-free Isolation Forest picked out the lockdown without
   labels; run it monthly to catch reporting errors and state-level shocks early.

**Data publishers**

8. **Fix the source data.** Correct the Aug 2025 and Jan 2026 cess column, publish the 8 missing months of
   2025-26, and release state population and total GSDP so revenue can be analysed per person.

---


## Project structure

```
├── app.py                      Streamlit app (9 tabs, light/dark theme, working filters)
├── src/
│   ├── data_engine.py          loading, name harmonisation, data-quality rules, aggregation
│   ├── econometrics_engine.py  panel models (clustered SE), correlations, buoyancy, Gini/Lorenz
│   ├── forecasting.py          4 models, rolling-origin backtest, conformal intervals, OOS check
│   ├── ml_engine.py            K-Means (silhouette-selected k), Isolation Forest anomalies
│   ├── simulation_engine.py    Monte Carlo revenue scenarios
│   ├── analysis.py             runs the pipeline and computes every headline number
│   └── visual_styles.py        CSS themes and a colour-blind-safe chart palette
├── tests/test_pipeline.py      13 tests covering data rules, findings and models
├── tax-collection-on-gst-portal.csv
└── RS_Session_266_AU_1677_B_2.csv
```

## Running locally

```bash
git clone https://github.com/aditya-stv/GST-analytics.git
cd GST-analytics
pip install -r requirements-dev.txt
pytest                 # 13 tests
streamlit run app.py   # http://localhost:8501
```

## Limitations

- No state population data, so GST and GSDP cannot be compared on the same per-person basis. Buoyancy
  ratios therefore carry a small upward bias from population growth.
- Only 5 years of official GSDP, so state-level elasticities are imprecise.
- GST is booked where the taxpayer is registered, which is not always where goods are consumed.
- The September 2025 rate rationalisation falls outside the training window. The −5% January 2026 miss is
  consistent with it, but two months cannot confirm that.
