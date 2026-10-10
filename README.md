# Indian Macro-Fiscal & Tax Intelligence Platform (2017–2026)
### Econometric Synthesis of Goods and Services Tax Collections & State Per Capita GSDP

A senior-level quantitative analytics and econometric platform that harmonizes high-frequency transactional indirect tax data (**GST Portal**) with annual macroeconomic output indicators (**MoSPI Per Capita GSDP**) across all 28 Constitutional Indian States from **FY 2017-18 to FY 2025-26**.

---

## Project Architecture

```
d:\codes\python\data_analytics/
├── tax-collection-on-gst-portal.csv      # Raw Monthly GST Portal Records (95 Months)
├── RS_Session_266_AU_1677_B_2.csv        # Raw Rajya Sabha / MoSPI GSDP Panel (FY20 to FY24)
├── app.py                                # Unified 5-Tab Interactive Streamlit Dashboard (Light/Dark Theme)
├── src/
│   ├── data_engine.py                    # Entity resolution, fiscal year assignment, aggregation
│   ├── synthetic_model.py                # Precasting (FY18-19), Imputation (FY24), Forecasting (FY25-26)
│   ├── econometrics_engine.py            # Panel Fixed Effects, Tax Buoyancy, Lorenz/Gini Inequality
│   ├── ml_engine.py                      # K-Means PCA Archetyping, Isolation Forest Anomalies
│   ├── simulation_engine.py              # Monte Carlo Policy & Macro Simulator (1,000 paths, 95% VaR)
│   └── visual_styles.py                  # Institutional Light & Dark CSS tokens and Plotly theme styler
└── README.md
```

---

## Methodological Framework

### 1. 100% Real Empirical Exploratory Data Analysis
- Analyzes **₹93.82 Lakh Crore** in indirect tax collections across 95 monthly reporting cycles.
- Disentangles the tax mix: **IGST (34.5%)**, **SGST (30.5%)**, **CGST (23.6%)**, and **Compensation Cess (11.4%)**.
- Quantifies the **COVID-19 lockdown shock** ($-75.09\%$ monthly collapse in April 2020) and subsequent V-shaped recovery within 6 months.
- Uncovers the **Spatial Concentration**: Maharashtra alone generates **20.40%**; the top 5 states generate **52.69%**; while the bottom 10 states and UTs contribute just **0.45%**.

### 2. Timeline Gap Synthesis: Precasting, Imputation & Forecasting
To bridge the timeline gap between GST (2017–2026) and GSDP (2019–2024):
* **Historical Precasting (FY 2017-18 & FY 2018-19):** Inverts the empirical panel elasticity equation $\ln(\text{GST}_{i,t}) = \alpha_i + \beta \ln(\text{GSDP\_PC}_{i,t}) + \epsilon_{i,t}$ conditioned on historical RBI national nominal deflators (+11.0% in FY18, +10.5% in FY19).
* **In-Sample Imputation (FY 2023-24):** Synthesizes records for the 6 missing states (`Arunachal Pradesh`, `Goa`, `Gujarat`, `Manipur`, `Mizoram`, `Nagaland`) using a weighted composite model:
  $$g_{i, 2024} = 0.50 \cdot \text{CAGR}_{i, 2020\to 2023} + 0.30 \cdot g^{\text{National}}_{2024} + 0.20 \cdot \left( \frac{\% \Delta \text{GST}_{i, 2024}}{\hat{\beta}} \right)$$
* **Forward Forecasting (FY 2024-25 & FY 2025-26):** Projects state growth using Holt's Linear Trend with Bayesian shrinkage towards the medium-term national target (+10.0%), emitting $P_{10}$ (downside) and $P_{90}$ (upside) credible intervals.
* **High-Frequency GST Completion:** Uses Seasonal Holt-Winters (12-month periodicity) to forecast the remaining months through March 2026 (closing FY 2025-26).

### 3. Econometric Findings & Elasticity
* **The Population/Size Confounder:** Total GST has a low raw correlation with Per Capita GSDP ($r = +0.137$) because total GST is dominated by demographic mass (e.g. UP collects high taxes with lower per-capita income).
* **The Interstate Trade Discovery:** IGST share exhibits a **strong positive correlation ($r = +0.548$)** with Per Capita GSDP—wealthier states consume significantly more interstate branded goods.
* **The Mineral Belt Cess Paradox:** Cess share is negatively correlated ($r = -0.374$) with Per Capita GSDP—mineral-rich states (Odisha: 24.2%, Jharkhand: 21.5%, Chhattisgarh: 18.9%) generate high Cess from coal mining despite lower per-capita incomes.
* **Within-State Elasticity (Panel Fixed Effects):** When controlling for state size using an Entity Fixed Effects model, within-state elasticity is **$\beta = 2.03$ ($p < 0.0001, R^2 = 98.9\%$)**, proving that economic expansion yields progressive revenue expansion.

---

## Running the Platform Locally

```bash
# Navigate to the repository directory
cd d:\codes\python\data_analytics

# Run the Streamlit application
streamlit run app.py
```

Open browser to `http://localhost:8501`.

---

## Platform Modules

1. **Tab 1: Empirical EDA (Raw Data):** Interactive exploration of national components, COVID lockdown shock, state Pareto curve, state tax mix, real GSDP spread, and empirical cross-dataset correlations.
2. **Tab 2: Precasting & Forecasting Studio:** State-level diagnostic comparing real vs precast vs forecast trajectories with $P_{10}/P_{90}$ confidence ribbons.
3. **Tab 3: Econometric Panel & Tax Buoyancy:** Log-log elasticity regressions, Fixed Effects vs Pooled OLS comparison table, buoyancy rankings, and Lorenz curves.
4. **Tab 4: Machine Learning Archetypes:** 2D PCA projection of K-Means state clusters and Isolation Forest anomaly detector.
5. **Tab 5: Stochastic Monte Carlo Simulator:** Real-time 1,000-path forward simulation with Value-at-Risk ($95\%$ VaR) downside stress testing.
