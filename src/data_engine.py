"""
data_engine.py - Fiscal and Macroeconomic Data Ingestion & Harmonization Pipeline
Harmonizes monthly GST transactional data with annual MoSPI GSDP per capita data.
"""

from typing import Dict, Tuple, List, Optional
import os
import sys
import numpy as np
import pandas as pd

# Standardized state name normalization mapping
STATE_NAME_STANDARDIZATION: Dict[str, str] = {
    "chhatisgarh": "Chhattisgarh",
    "chhattisgarh": "Chhattisgarh",
    "andaman and nicobar islands": "Andaman And Nicobar Islands",
    "the dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "jammu and kashmir": "Jammu And Kashmir",
    "delhi": "Delhi",
    "puducherry": "Puducherry",
    "chandigarh": "Chandigarh",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
}

# 28 Constitutional Indian States present in the MoSPI GSDP benchmark
BENCHMARK_28_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

def normalize_state_name(name: str) -> str:
    """Cleans and standardizes state names across disparate data sources."""
    if not isinstance(name, str):
        return ""
    clean = name.strip()
    key = clean.lower()
    if key in STATE_NAME_STANDARDIZATION:
        return STATE_NAME_STANDARDIZATION[key]
    return " ".join(word.capitalize() for word in clean.split())

def assign_fiscal_year(date: pd.Timestamp) -> str:
    """
    Maps calendar date to Indian Fiscal Year (April 1 to March 31).
    Example: 2021-03-01 -> '2020-21', 2021-04-01 -> '2021-22'
    """
    year = date.year
    month = date.month
    if month >= 4:
        start_year = year
        end_year = (year + 1) % 100
    else:
        start_year = year - 1
        end_year = year % 100
    return f"{start_year}-{end_year:02d}"

def load_and_clean_gst(gst_path: str = "tax-collection-on-gst-portal.csv") -> pd.DataFrame:
    """Loads and standardizes monthly GST collection dataset."""
    df = pd.read_csv(gst_path)
    df["date"] = pd.to_datetime(df["date"])
    df["state_name_clean"] = df["state_name"].apply(normalize_state_name)
    
    for col in ["cgst", "sgst", "igst", "cess"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        
    df["total_gst"] = df["cgst"] + df["sgst"] + df["igst"] + df["cess"]
    df["domestic_gst"] = df["cgst"] + df["sgst"]
    df["fiscal_year"] = df["date"].apply(assign_fiscal_year)
    df["fiscal_quarter"] = df["date"].apply(lambda d: f"Q{((d.month - 4) % 12) // 3 + 1}")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["is_benchmark_state"] = df["state_name_clean"].isin(BENCHMARK_28_STATES)
    
    return df.sort_values(["state_name_clean", "date"]).reset_index(drop=True)

def load_and_clean_gsdp(gsdp_path: str = "RS_Session_266_AU_1677_B_2.csv") -> pd.DataFrame:
    """Loads and transforms MoSPI Per Capita GSDP dataset into long format."""
    df = pd.read_csv(gsdp_path)
    df["State_Clean"] = df["State"].apply(normalize_state_name)
    
    fy_mapping = {
        "Per Capita GSDP - 2019-2020": "2019-20",
        "Per Capita GSDP - 2020-2021": "2020-21",
        "Per Capita GSDP - 2021-22": "2021-22",
        "Per Capita GSDP - 2022-23": "2022-23",
        "Per Capita GSDP - 2023-24": "2023-24"
    }
    
    records = []
    for _, row in df.iterrows():
        state = row["State_Clean"]
        for raw_col, fy in fy_mapping.items():
            val = row.get(raw_col, np.nan)
            try:
                val_numeric = float(val) if pd.notna(val) and str(val).strip().upper() != "NA" else np.nan
            except (ValueError, TypeError):
                val_numeric = np.nan
                
            records.append({
                "state": state,
                "fiscal_year": fy,
                "per_capita_gsdp_raw": val_numeric,
                "is_imputed": False
            })
            
    return pd.DataFrame(records)

def aggregate_annual_gst(clean_gst: pd.DataFrame) -> pd.DataFrame:
    """Aggregates monthly GST collections into annual fiscal year totals."""
    annual = clean_gst.groupby(["state_name_clean", "fiscal_year"]).agg(
        cgst_sum=("cgst", "sum"),
        sgst_sum=("sgst", "sum"),
        igst_sum=("igst", "sum"),
        cess_sum=("cess", "sum"),
        total_gst_sum=("total_gst", "sum"),
        months_recorded=("date", "count")
    ).reset_index()
    
    annual.rename(columns={"state_name_clean": "state"}, inplace=True)
    annual["annualization_factor"] = 12.0 / annual["months_recorded"].clip(lower=1)
    annual["annualized_total_gst"] = annual["total_gst_sum"] * annual["annualization_factor"]
    annual["igst_share"] = annual["igst_sum"] / annual["total_gst_sum"].replace(0, np.nan)
    annual["cess_share"] = annual["cess_sum"] / annual["total_gst_sum"].replace(0, np.nan)
    annual["sgst_to_cgst_ratio"] = annual["sgst_sum"] / annual["cgst_sum"].replace(0, np.nan)
    
    return annual
