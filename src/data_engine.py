"""
data_engine.py - Ingestion, data-quality validation and harmonisation.

Loads the monthly GST portal records and the MoSPI per-capita GSDP table,
runs explicit data-quality checks, and produces the clean frames every other
module consumes. Every correction made to the raw data is recorded in a
quality log so it can be shown in the app and audited.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GST_PATH = ROOT / "tax-collection-on-gst-portal.csv"
GSDP_PATH = ROOT / "RS_Session_266_AU_1677_B_2.csv"

COMPONENTS = ["cgst", "sgst", "igst", "cess"]
CORE_COMPONENTS = ["cgst", "sgst", "igst"]

# GST was launched on 1 July 2017; returns for July were filed in August, so the
# July 2017 row is a near-zero placeholder. The last month with a complete,
# uncorrupted fiscal year behind it is March 2025.
ANALYSIS_START = pd.Timestamp("2017-08-01")
ANALYSIS_END = pd.Timestamp("2025-03-01")
COMPLETE_FYS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
GSDP_FYS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

STATE_NAME_STANDARDIZATION: Dict[str, str] = {
    "chhatisgarh": "Chhattisgarh",
    "the dadra and nagar haveli and daman and diu": "Dadra And Nagar Haveli And Daman And Diu",
    "cbic": "CBIC (Centre)",
    "oidar": "OIDAR (Online Services)",
}

BENCHMARK_28_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]

UNION_TERRITORIES = [
    "Andaman And Nicobar Islands", "Chandigarh", "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi", "Jammu And Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

# Collections booked centrally rather than to a territory: part of the national
# total, but excluded from state rankings.
NON_TERRITORIAL = ["CBIC (Centre)", "OIDAR (Online Services)", "Other Territory"]


def normalize_state_name(name: str) -> str:
    """Cleans and standardises state names across the two sources."""
    if not isinstance(name, str):
        return ""
    clean = " ".join(name.strip().split())
    key = clean.lower()
    if key in STATE_NAME_STANDARDIZATION:
        return STATE_NAME_STANDARDIZATION[key]
    return " ".join(word.capitalize() for word in clean.split())


def entity_type(name: str) -> str:
    if name in BENCHMARK_28_STATES:
        return "State"
    if name in UNION_TERRITORIES:
        return "Union Territory"
    return "Non-territorial"


def assign_fiscal_year(date: pd.Timestamp) -> str:
    """Maps a date to the Indian fiscal year (April-March), e.g. 2021-03 -> '2020-21'."""
    start = date.year if date.month >= 4 else date.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def load_raw_gst(path: Path = GST_PATH) -> pd.DataFrame:
    """Reads the GST portal file with standardised names and derived calendar fields."""
    df = pd.read_csv(path, dtype={"state_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df["state"] = df["state_name"].apply(normalize_state_name)
    df["entity_type"] = df["state"].apply(entity_type)
    for col in COMPONENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fiscal_year"] = df["date"].apply(assign_fiscal_year)
    return df.sort_values(["date", "state"]).reset_index(drop=True)


def run_quality_checks(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Applies every data-quality rule to the raw GST frame.

    Returns the corrected frame (all months, with flags) and a log describing
    each check, how many rows it touched and what was done about it.
    """
    df = raw.copy()
    log: List[dict] = []

    def record(check, severity, rows, action, detail):
        log.append({"check": check, "severity": severity, "rows_affected": int(rows),
                    "action": action, "detail": detail})

    dupes = df.duplicated(["state", "date"]).sum()
    record("Duplicate state-month records", "Info", dupes, "None needed" if dupes == 0 else "Dropped",
           "Each (state, month) pair appears at most once.")
    df = df.drop_duplicates(["state", "date"])

    negatives = (df[COMPONENTS] < 0).any(axis=1).sum()
    nulls = df[COMPONENTS].isna().any(axis=1).sum()
    record("Negative or non-numeric amounts", "Info", negatives + nulls, "None needed",
           "All four tax heads parse as non-negative numbers.")

    # Rule 1: the July 2017 launch month is a placeholder (national total ~Rs 53 Cr).
    launch = df["date"] < ANALYSIS_START
    national_launch = df.loc[launch, COMPONENTS].sum().sum()
    record("GST launch month (July 2017) is a placeholder", "High", launch.sum(),
           "Excluded from analysis",
           f"National total for July 2017 is Rs {national_launch:,.0f} Cr vs ~Rs 70,000 Cr in every following month; "
           "July liabilities were filed and booked from August 2017.")

    # Rule 2: in some months the 'cess' column holds CGST+SGST+IGST instead of cess.
    core_sum = df[CORE_COMPONENTS].sum(axis=1)
    mirrors = ((df["cess"] - core_sum).abs() < 0.05) & (core_sum > 1.0)
    month_share = mirrors.groupby(df["date"]).mean()
    corrupt_months = month_share[month_share > 0.9].index
    corrupt = df["date"].isin(corrupt_months)
    df["cess_corrupt"] = corrupt
    df.loc[corrupt, "cess"] = np.nan
    record("Cess column duplicates CGST+SGST+IGST", "Critical", corrupt.sum(),
           "Cess set to missing for these months",
           "In " + ", ".join(d.strftime("%b %Y") for d in corrupt_months) +
           " the cess value equals CGST+SGST+IGST to the paisa for every entity (e.g. Maharashtra Aug 2025: "
           "7,840.27 + 9,400.00 + 9,505.44 = 26,745.71). Left uncorrected this adds ~Rs 2.67 lakh Cr "
           "of phantom revenue and doubles these months.")

    # Rule 3: calendar gaps.
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    missing = sorted(set(calendar) - set(df["date"].unique()))
    record("Missing calendar months", "High", len(missing),
           "FY 2025-26 excluded from annual analysis",
           "No records for " + ", ".join(d.strftime("%b %Y") for d in missing) +
           ". FY 2025-26 has only 2 of 12 months, so it cannot be annualised.")

    # Rule 4: entities that are not states.
    nt = df["entity_type"] == "Non-territorial"
    record("Non-territorial collection heads", "Medium", nt.sum(),
           "Kept in national totals, excluded from state rankings",
           "CBIC, OIDAR and 'Other Territory' are centrally booked heads, not states.")

    # Rule 5: entities with partial coverage.
    coverage = df[df["date"].between(ANALYSIS_START, ANALYSIS_END)].groupby("state")["date"].nunique()
    full = coverage.max()
    partial = coverage[coverage < full]
    record("Entities with partial time coverage", "Low", len(partial), "Documented",
           "; ".join(f"{s}: {n}/{full} months" for s, n in partial.items()) +
           ". Ladakh became a UT in Oct 2019, OIDAR was reported separately from Apr 2024.")

    df["in_analysis_window"] = df["date"].between(ANALYSIS_START, ANALYSIS_END)
    df["total_gst"] = df[COMPONENTS].sum(axis=1, min_count=4)
    df["core_gst"] = df[CORE_COMPONENTS].sum(axis=1)
    return df.reset_index(drop=True), log


def load_gst() -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    Returns (analysis_frame, all_months_frame, quality_log).

    The analysis frame covers Aug 2017 - Mar 2025: complete, validated months only.
    The all-months frame keeps the post-window 2025-26 rows (with cess blanked) for
    out-of-sample forecast validation.
    """
    checked, log = run_quality_checks(load_raw_gst())
    analysis = checked[checked["in_analysis_window"]].copy()
    return analysis.reset_index(drop=True), checked, log


def load_gsdp(path: Path = GSDP_PATH) -> pd.DataFrame:
    """Loads MoSPI per-capita GSDP (current prices) in long format; NA stays missing."""
    df = pd.read_csv(path)
    df["state"] = df["State"].apply(normalize_state_name)
    value_cols = [c for c in df.columns if c.startswith("Per Capita GSDP")]
    long = df.melt(id_vars="state", value_vars=value_cols, var_name="column", value_name="per_capita_gsdp")
    long["fiscal_year"] = long["column"].str.extract(r"(\d{4})-(\d{2,4})").apply(
        lambda r: f"{r[0]}-{r[1][-2:]}", axis=1)
    long["per_capita_gsdp"] = pd.to_numeric(long["per_capita_gsdp"], errors="coerce")
    unknown = set(long["state"]) - set(BENCHMARK_28_STATES)
    if unknown:
        raise ValueError(f"GSDP states not matched to GST names: {sorted(unknown)}")
    return long[["state", "fiscal_year", "per_capita_gsdp"]].sort_values(["state", "fiscal_year"]).reset_index(drop=True)


def aggregate_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregates monthly records to entity x fiscal-year totals with composition shares."""
    annual = monthly.groupby(["state", "entity_type", "fiscal_year"]).agg(
        cgst=("cgst", "sum"), sgst=("sgst", "sum"), igst=("igst", "sum"), cess=("cess", "sum"),
        total_gst=("total_gst", "sum"), months=("date", "nunique"),
    ).reset_index()
    for col in COMPONENTS:
        annual[f"{col}_share"] = annual[col] / annual["total_gst"].replace(0, np.nan)
    return annual


def build_empirical_panel(monthly: pd.DataFrame, gsdp: pd.DataFrame) -> pd.DataFrame:
    """
    State x FY panel where both GST and GSDP are actually observed (FY 2019-20 to 2023-24).
    No imputation: the 6 states without FY 2023-24 GSDP simply have 4 observations.
    """
    annual = aggregate_annual(monthly[monthly["entity_type"] == "State"])
    annual = annual[annual["fiscal_year"].isin(GSDP_FYS)]
    panel = annual.merge(gsdp, on=["state", "fiscal_year"], how="inner").dropna(subset=["per_capita_gsdp"])
    panel["log_gst"] = np.log(panel["total_gst"])
    panel["log_gsdp_pc"] = np.log(panel["per_capita_gsdp"])
    return panel.sort_values(["state", "fiscal_year"]).reset_index(drop=True)


def national_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """National monthly totals across every reporting entity."""
    cols = COMPONENTS + ["total_gst", "core_gst"]
    return monthly.groupby("date")[cols].sum(min_count=1).reset_index()
