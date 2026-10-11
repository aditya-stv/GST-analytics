"""
simulation_engine.py - Monte Carlo scenario model for next-year national GST revenue.

Revenue is split into a non-cess base, which grows with nominal GDP times a
buoyancy draw plus a compliance effect, and the compensation cess, which grows
with nominal GDP and is scaled by the share of cess retained under the policy
scenario.
"""

from typing import Dict

import numpy as np


def run_monte_carlo_fiscal_simulation(
    base_revenue_cr: float,
    cess_share: float,
    real_gdp_growth_pct: float = 6.5,
    inflation_pct: float = 4.0,
    growth_sd_pct: float = 1.5,
    buoyancy_mean: float = 1.0,
    buoyancy_sd: float = 0.15,
    compliance_gain_pct: float = 0.0,
    cess_retained: float = 1.0,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    nominal_mu = (1 + real_gdp_growth_pct / 100) * (1 + inflation_pct / 100) - 1
    nominal = rng.normal(nominal_mu, growth_sd_pct / 100, n_simulations)
    buoyancy = np.clip(rng.normal(buoyancy_mean, buoyancy_sd, n_simulations), 0.3, 2.5)
    compliance = rng.normal(compliance_gain_pct / 100, 0.005, n_simulations)

    non_cess = base_revenue_cr * (1 - cess_share) * (1 + buoyancy * nominal + compliance)
    cess = base_revenue_cr * cess_share * (1 + nominal) * cess_retained
    revenue = non_cess + cess

    median = float(np.median(revenue))
    p05, p95 = np.percentile(revenue, [5, 95])
    return {
        "n_simulations": n_simulations,
        "base_revenue_cr": float(base_revenue_cr),
        "nominal_growth_mean_pct": nominal_mu * 100,
        "mean_cr": float(revenue.mean()),
        "median_cr": median,
        "p05_cr": float(p05),
        "p95_cr": float(p95),
        "revenue_at_risk_95_cr": median - float(p05),
        "expected_growth_pct": (revenue.mean() / base_revenue_cr - 1) * 100,
        "prob_below_base": float((revenue < base_revenue_cr).mean()),
        "simulated_values": revenue,
    }
