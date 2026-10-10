"""
ml_engine.py - Unsupervised State Archetyping & Fiscal Anomaly Detection Engine
"""

import os
import sys
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.econometrics_engine import calculate_state_buoyancy_profile

def perform_state_clustering(macro_panel: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs PCA and K-Means Clustering across the 28 benchmark states."""
    buoyancy_df = calculate_state_buoyancy_profile(macro_panel).set_index("state")
    
    state_agg = macro_panel.groupby("state").agg(
        avg_annual_gst=("annualized_total_gst", "mean"),
        avg_gsdp_pc=("per_capita_gsdp", "mean"),
        avg_igst_share=("igst_share", "mean"),
        avg_cess_share=("cess_share", "mean")
    )
    
    combined = state_agg.join(buoyancy_df[["tax_buoyancy_index"]])
    feature_cols = ["avg_annual_gst", "avg_gsdp_pc", "avg_igst_share", "avg_cess_share", "tax_buoyancy_index"]
    X = combined[feature_cols].fillna(combined[feature_cols].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = kmeans.fit_predict(X_scaled)
    
    combined["pca_dim1"] = pca_coords[:, 0]
    combined["pca_dim2"] = pca_coords[:, 1]
    combined["cluster_id"] = raw_labels
    
    cluster_means = combined.groupby("cluster_id")[["avg_annual_gst", "avg_gsdp_pc", "avg_cess_share"]].mean()
    highest_gst_id = cluster_means["avg_annual_gst"].idxmax()
    remaining = [c for c in cluster_means.index if c != highest_gst_id]
    highest_gsdp_id = cluster_means.loc[remaining, "avg_gsdp_pc"].idxmax()
    remaining = [c for c in remaining if c != highest_gsdp_id]
    highest_cess_id = cluster_means.loc[remaining, "avg_cess_share"].idxmax()
    remaining = [c for c in remaining if c != highest_cess_id]
    catchup_id = remaining[0] if len(remaining) > 0 else 3
    
    archetype_map = {
        highest_gst_id: "Industrial & Megacity Powerhouses",
        highest_gsdp_id: "High-Income & Tourism Frontrunners",
        highest_cess_id: "Resource & Mineral Corridor",
        catchup_id: "Demographic & Catch-Up Consumer Economies"
    }
    
    combined["archetype"] = combined["cluster_id"].map(archetype_map)
    
    metadata = {
        "pca_variance_ratio": [float(v) for v in var_explained],
        "total_pca_variance": float(np.sum(var_explained))
    }
    
    return combined.reset_index(), metadata

def detect_fiscal_anomalies(monthly_gst: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """Detects anomalous monthly state reports using Isolation Forest."""
    df = monthly_gst[monthly_gst["is_benchmark_state"]].copy()
    
    df["rolling_3m_mean"] = df.groupby("state_name_clean")["total_gst"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["pct_dev_from_rolling"] = ((df["total_gst"] - df["rolling_3m_mean"]) / df["rolling_3m_mean"].replace(0, np.nan)).fillna(0.0)
    
    features = df[["total_gst", "pct_dev_from_rolling"]].fillna(0.0)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(features)
    
    df["is_anomaly"] = preds == -1
    df["anomaly_score"] = iso_forest.decision_function(features)
    
    anomalies = df[df["is_anomaly"]].sort_values("anomaly_score", ascending=True)
    return anomalies[["date", "state_name_clean", "total_gst", "pct_dev_from_rolling", "fiscal_year"]].reset_index(drop=True)
