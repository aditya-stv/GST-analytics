"""
simulation_engine.py - Stochastic Monte Carlo Fiscal Simulation Engine
"""

from typing import Dict, Any
import numpy as np

def run_monte_carlo_fiscal_simulation(
    base_annual_collection: float,
    expected_gdp_growth_pct: float = 6.5,
    expected_inflation_pct: float = 4.5,
    compliance_efficiency_delta_pct: float = 1.5,
    cess_policy_factor: float = 1.0,
    n_simulations: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Simulates 1,000 forward paths to evaluate Value-at-Risk (VaR 95%)."""
    np.random.seed(random_seed)
    
    mu_nominal_growth = (expected_gdp_growth_pct + expected_inflation_pct) / 100.0
    sigma_nominal_growth = 0.022
    simulated_nominal_growth = np.random.normal(mu_nominal_growth, sigma_nominal_growth, n_simulations)
    
    simulated_elasticity = np.random.normal(1.08, 0.12, n_simulations)
    simulated_elasticity = np.clip(simulated_elasticity, 0.70, 1.45)
    compliance_shock = np.random.normal(compliance_efficiency_delta_pct / 100.0, 0.008, n_simulations)
    
    gross_growth_factor = 1.0 + (simulated_elasticity * simulated_nominal_growth) + compliance_shock
    cess_adjustment = 1.0 - (1.0 - cess_policy_factor) * 0.075
    simulated_revenues = base_annual_collection * gross_growth_factor * cess_adjustment
    
    mean_rev = np.mean(simulated_revenues)
    median_rev = np.median(simulated_revenues)
    p05 = np.percentile(simulated_revenues, 5)
    p95 = np.percentile(simulated_revenues, 95)
    var_95_drawdown = max(0.0, base_annual_collection - p05)
    
    return {
        "n_simulations": n_simulations,
        "base_annual_collection_cr": round(float(base_annual_collection), 1),
        "mean_projected_cr": round(float(mean_rev), 1),
        "median_projected_cr": round(float(median_rev), 1),
        "p05_downside_cr": round(float(p05), 1),
        "p95_upside_cr": round(float(p95), 1),
        "var_95_drawdown_cr": round(float(var_95_drawdown), 1),
        "simulated_values": simulated_revenues
    }
