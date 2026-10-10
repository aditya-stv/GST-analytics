# Indian Macro-Fiscal & Tax Intelligence Platform (2017–2026)
### Econometric Synthesis of Goods and Services Tax Collections & State Per Capita GSDP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gst-analytics.streamlit.app/)
**Live Web Application:** [https://gst-analytics.streamlit.app/](https://gst-analytics.streamlit.app/)

A quantitative analytics and econometric intelligence platform that harmonizes high-frequency transactional indirect tax records (**GST Portal**) with annual macroeconomic output indicators (**MoSPI Per Capita GSDP**) across all 28 Constitutional Indian States from **FY 2017-18 to FY 2025-26**.

---

## Live Dashboard & Repository Overview

* **Live Cloud Application:** [https://gst-analytics.streamlit.app/](https://gst-analytics.streamlit.app/)
* **Source Repository:** [https://github.com/aditya-stv/GST-analytics](https://github.com/aditya-stv/GST-analytics)
* **Primary Data Sources:**
  - `tax-collection-on-gst-portal`: [India Data Portal (CKAN)](https://ckandev.indiadataportal.com/dataset/goods-and-services-tax) (95 months, July 2017 to January 2026)
  - `RS_Session_266_AU_1677_B_2`: [Data.gov.in (MoSPI / Rajya Sabha)](https://www.data.gov.in/resource/state-wise-details-capita-gross-state-domestic-product-gsdp-current-prices-2019-20-2023-24) (Annual Per Capita GSDP, FY20 to FY24)

---

## Project Structure

```
d:\codes\python\data_analytics/
├── tax-collection-on-gst-portal.csv      # Raw Monthly GST Portal Records (95 Months)
├── RS_Session_266_AU_1677_B_2.csv        # Raw Rajya Sabha / MoSPI GSDP Panel (FY20 to FY24)
├── app.py                                # Streamlit Web Application (Light/Dark Theme, 6 Modules)
├── requirements.txt                      # Production dependency specifications
├── .streamlit/config.toml                # Streamlit server & theme configuration
├── src/
│   ├── data_engine.py                    # Entity resolution, fiscal calendar alignment, aggregation
│   ├── synthetic_model.py                # Precasting (FY18-19), Imputation (FY24), Forecasting (FY25-26)
│   ├── econometrics_engine.py            # Panel Fixed Effects, Tax Buoyancy, Lorenz/Gini Inequality
│   ├── ml_engine.py                      # K-Means PCA Archetyping, Isolation Forest Anomalies
│   ├── simulation_engine.py              # Monte Carlo Policy & Macro Simulator (1,000 paths, 95% VaR)
│   └── visual_styles.py                  # Institutional Light & Dark CSS tokens and Plotly theme styler
└── README.md
```

---

## Methodological Framework & Mathematical Formulations

To bridge the temporal mismatch between GST (2017–2026) and GSDP (2019–2024), the platform implements an econometric state-space synthesis engine.

### 1. Historical Precasting / Backcasting (FY 2017-18 & FY 2018-19)
We calibrate the empirical panel tax elasticity relationship on the overlapping empirical window (FY20 to FY24):

```math
\ln(\text{GST}_{i,t}) = \alpha_i + \beta \cdot \ln(\text{GSDP}_{i,t}) + \epsilon_{i,t}
```

Because actual GST collections are observed for FY 2017-18 and FY 2018-19, we invert the equation to estimate historical per capita output:

```math
\widehat{\ln(\text{GSDP}_{i,t})} = \frac{\ln(\text{GST}_{i,t}) - \alpha_i}{\beta}
```

Estimates are anchored to official RBI historical national nominal deflators (+11.0% in FY18, +10.5% in FY19) to ensure empirical consistency.

### 2. In-Sample Imputation (FY 2023-24 Missing States)
For the 6 states with unreleased official MoSPI data in FY 2023-24 (`Arunachal Pradesh`, `Goa`, `Gujarat`, `Manipur`, `Mizoram`, `Nagaland`), growth is estimated via a weighted composite model:

```math
g_{i, 2024} = 0.50 \cdot \text{CAGR}_{i, 2020 \to 2023} + 0.30 \cdot g^{\text{National}}_{2024} + 0.20 \cdot \left( \frac{\Delta \text{GST}_{i, 2024}}{\beta} \right)
```

Where:
* $0.50$ captures state-specific multi-year growth momentum.
* $0.30$ anchors to the national MoSPI nominal GDP benchmark (+10.6%).
* $0.20$ reflects observed state GST collection growth scaled by tax elasticity.

```math
\widehat{\text{GSDP}}_{i, 2024} = \text{GSDP}_{i, 2023} \times (1 + g_{i, 2024})
```

### 3. Forward Forecasting (FY 2024-25 & FY 2025-26)
* **GSDP Per Capita:** State-specific Holt's linear trend with Bayesian shrinkage toward the national medium-term nominal target (+10.0%), emitting $P_{10}$ (lower) and $P_{90}$ (upper) credible intervals:

```math
\hat{g}_{i, 2025} = 0.65 \cdot \text{Trend}_i + 0.35 \cdot g^{\text{National}}_{2025}
```

```math
\hat{g}_{i, 2026} = 0.60 \cdot \text{Trend}_i + 0.40 \cdot g^{\text{National}}_{2026}
```

* **Monthly GST Collections:** Seasonal Holt-Winters (12-month multiplicative seasonality) closes out the series through March 2026, capturing the March fiscal year-end clearing peak and Diwali consumption surges.

### 4. Panel Fixed Effects vs. Pooled OLS (Within-State Elasticity)

```math
\ln(\text{GST}_{i,t}) = \alpha_i + \beta_1 \cdot \ln(\text{GSDP}_{i,t}) + \beta_2 \cdot \text{PostCOVID}_t + \mathbf{u}_{i,t}
```

---

## Key Empirical Findings

1. **Macro Revenue Aggregate:**
   Across 95 reporting months, India generated **₹93,81,942.66 Crore** in total indirect taxes:
   * **IGST (Integrated):** ₹32,33,883.07 Cr (**34.47%**)
   * **SGST (State):** ₹28,65,428.56 Cr (**30.54%**)
   * **CGST (Central):** ₹22,14,063.99 Cr (**23.60%**)
   * **Compensation Cess:** ₹10,68,567.04 Cr (**11.39%**)

2. **Extreme Spatial Concentration (The 5-State Pareto Divide):**
   * **Maharashtra alone accounts for 20.40%** (₹19.14 Lakh Cr) of all GST collected in India.
   * The **Top 5 states** (Maharashtra, Karnataka, Gujarat, Tamil Nadu, Uttar Pradesh) generate **52.69%** of national revenue.
   * The **Top 10 states** generate **75.54%**, while the bottom 10 states and Union Territories combined contribute just **0.45%**.

3. **COVID-19 Lockdown Shock & V-Shaped Recovery:**
   * In April 2020, indirect tax receipts collapsed by **-75.09%** (from ₹78,701 Cr pre-COVID down to ₹19,606 Cr).
   * Collections crossed back above pre-COVID baselines within **6 months** (October 2020), catalyzed by mandatory e-invoicing.

4. **The Persistent 10.6x Inter-State Wealth Divide:**
   * The ratio between the highest per capita state (Sikkim/Goa at ~₹5–7 Lakhs) and the lowest state (Bihar at ₹48,000–66,000) remained fixed at **10.5x to 10.6x** across all recorded years.

5. **The Interstate Trade Discovery (IGST vs. Wealth):**
   * Raw total GST has a weak cross-sectional correlation with Per Capita GSDP ($r = +0.137$) due to demographic size disparities.
   * However, **IGST share exhibits a strong positive correlation ($r = +0.548$)** with Per Capita GSDP: higher-income states consume significantly more branded interstate goods and operate deeply integrated supply networks.

6. **The Resource Extraction Cess Paradox:**
   * Compensation Cess share is negatively correlated ($r = -0.374$) with Per Capita GSDP. Mineral-rich states (**Odisha: 24.2%**, **Jharkhand: 21.5%**, **Chhattisgarh: 18.9%**) generate high Cess shares from coal mining royalties despite having lower per capita incomes.

---

## Econometric Conclusions

* **Within-State Elasticity Outperforms Cross-Sectional OLS:**
  * In Pooled OLS, elasticity appears muted ($R^2 = 3.8\%$) because state population scale confounds the relationship.
  * In the **Entity Fixed Effects Model**, within-state tax elasticity is **$\beta = 2.03$ ($p < 0.0001, R^2 = 98.9\%$)**. This proves that as a state's economy expands, its tax revenue expands at a progressive $>2\times$ rate.
* **Fiscal Revenue is Far More Concentrated than Income:**
  * Gini coefficient of GSDP Per Capita: **$G = 0.36$** (moderate inequality).
  * Gini coefficient of State GST Collections: **$G = 0.61$** (high concentration), reflecting corporate headquarters clustering and industrialized supply chains.

---

## Strategic Policy Recommendations

1. **Fiscal Buffer Mechanism for the Coal & Mineral Corridor:**
   - As the GST Compensation Cess sunsets, mineral-producing states (Odisha, Jharkhand, Chhattisgarh) face significant fiscal revenue exposure. A dedicated transition fund or permanent mineral royalty restructuring is essential to protect developmental capital expenditures.
2. **Targeted Digital Enforcement in High-Demographic, Lower-Buoyancy States:**
   - States with massive consumer populations but sub-unitary tax buoyancy ($B < 1.0$) should receive targeted technical assistance for e-way bill verification and B2C digital receipt matching.
3. **Formalization of Inter-State Destination Clearings (IGST Settlement Optimization):**
   - Streamlining the IGST clearing mechanism will help consumer states accurately capture credit for final destination consumption, improving fiscal autonomy.
4. **Institutional Buoyancy Benchmarking for the Finance Commission:**
   - State Finance Commissions should adopt empirical Tax Buoyancy ($B_i$) as an explicit performance metric when determining horizontal tax devolution formulas, rewarding tax formalization efforts.

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/aditya-stv/GST-analytics.git
cd GST-analytics

# Run the Streamlit application
streamlit run app.py
```

Then visit `http://localhost:8501`.
