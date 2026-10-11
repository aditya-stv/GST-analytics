"""
Regression tests for the data-quality rules and the headline findings.
Run with:  pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import run_pipeline
from src.data_engine import (
    BENCHMARK_28_STATES, assign_fiscal_year, load_gsdp, load_raw_gst, normalize_state_name, run_quality_checks,
)
from src.econometrics_engine import compute_gini_and_lorenz
from src.simulation_engine import run_monte_carlo_fiscal_simulation


@pytest.fixture(scope="module")
def bundle():
    return run_pipeline()


# ---------------------------------------------------------------- helpers
def test_fiscal_year_boundaries():
    assert assign_fiscal_year(pd.Timestamp("2021-03-01")) == "2020-21"
    assert assign_fiscal_year(pd.Timestamp("2021-04-01")) == "2021-22"
    assert assign_fiscal_year(pd.Timestamp("2099-12-01")) == "2099-00"


def test_state_names_harmonise_across_sources():
    assert normalize_state_name("Chhatisgarh") == "Chhattisgarh"
    assert set(load_gsdp()["state"]) == set(BENCHMARK_28_STATES)


def test_gini_reference_values():
    assert compute_gini_and_lorenz([1, 1, 1, 1])[0] == pytest.approx(0.0)
    # one holder of everything among n: Gini = (n-1)/n
    assert compute_gini_and_lorenz([0, 0, 0, 10])[0] == pytest.approx(0.75)


# ---------------------------------------------------------------- data quality
def test_cess_corruption_detected_and_blanked():
    checked, log = run_quality_checks(load_raw_gst())
    bad = checked[checked["cess_corrupt"]]
    assert set(bad["date"].dt.strftime("%Y-%m")) == {"2025-08", "2026-01"}
    assert bad["cess"].isna().all()
    assert any("Cess column" in row["check"] for row in log)


def test_analysis_window_is_clean(bundle):
    m = bundle["monthly"]
    assert m["date"].min() == pd.Timestamp("2017-08-01")
    assert m["date"].max() == pd.Timestamp("2025-03-01")
    assert m["date"].nunique() == 92
    assert m["cess"].notna().all()
    assert not m.duplicated(["state", "date"]).any()
    # components add up to the total
    assert np.allclose(m[["cgst", "sgst", "igst", "cess"]].sum(axis=1), m["total_gst"])


def test_empirical_panel_has_no_imputed_values(bundle):
    p = bundle["panel"]
    assert len(p) == 134                      # 28 x 5 minus 6 unpublished FY24 values
    assert p["per_capita_gsdp"].notna().all()
    assert p["state"].nunique() == 28


# ---------------------------------------------------------------- findings
def test_headline_findings(bundle):
    f = bundle["findings"]
    assert f["total_cr"] == pytest.approx(8_848_486, rel=1e-4)
    assert f["component_share"]["cess"] == pytest.approx(9.06, abs=0.01)
    assert f["top1_state"] == "Maharashtra"
    assert f["top5_share"] == pytest.approx(52.56, abs=0.01)
    assert f["apr20_drop_pct"] == pytest.approx(-75.09, abs=0.01)
    assert f["recovery_month"] == pd.Timestamp("2020-10-01")
    assert f["fy21_contracted"] == 20
    assert f["gini_gsdp_pc"] == pytest.approx(0.276, abs=0.001)


def test_two_way_fe_is_not_significant(bundle):
    m = bundle["models"].set_index("model")
    twfe = m.loc["Two-way fixed effects"]
    assert twfe["ci_low"] < 0 < twfe["ci_high"]
    assert m.loc["State fixed effects", "elasticity"] < 2.0


def test_forecast_backtest_and_validation(bundle):
    fc = bundle["forecast"]
    assert len(fc["backtest"]) == 12          # 3 holdout years x 4 models
    assert fc["backtest_summary"]["mean_mape_pct"].min() < 10
    assert fc["validation"]["inside_80_pi"].all()


def test_anomaly_detector_finds_lockdown(bundle):
    flagged = bundle["anomalies"].query("is_anomaly")
    assert (flagged["date"] == "2020-04-01").sum() >= 15


# ---------------------------------------------------------------- simulation
def test_simulation_is_reproducible_and_cess_lever_works():
    a = run_monte_carlo_fiscal_simulation(100.0, 0.1, seed=1)
    b = run_monte_carlo_fiscal_simulation(100.0, 0.1, seed=1)
    assert a["median_cr"] == b["median_cr"]
    no_cess = run_monte_carlo_fiscal_simulation(100.0, 0.1, cess_retained=0.0, seed=1)
    assert no_cess["median_cr"] < a["median_cr"] - 9


# ---------------------------------------------------------------- deeper insights
def test_concentration_rising_and_growth_normalising(bundle):
    f = bundle["findings"]
    assert f["top5_by_year"]["2024-25"] > f["top5_by_year"]["2018-19"] + 2
    assert f["hhi_by_year"].iloc[-1] > f["hhi_by_year"].iloc[0] * 1.05
    yoy = f["fy_yoy"]
    assert yoy["2021-22"] > yoy["2022-23"] > yoy["2023-24"] > yoy["2024-25"]
    assert f["mix_by_year"]["cess"].is_monotonic_decreasing


def test_seasonality_and_volatility(bundle):
    f = bundle["findings"]
    assert f["seasonal_index"].idxmax() == 4                    # April peak
    assert (f["sgst_cgst_ratio"] > 1).all()                     # SGST > CGST in every state
    assert f["volatility_vs_size"][0] < -0.5                    # small states more volatile
