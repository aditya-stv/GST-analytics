"""
ml_engine.py - Unsupervised state archetypes and fiscal anomaly detection.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "log_avg_gst": ("revenue scale", "Large", "Small"),
    "log_gsdp_pc": ("per-capita income", "High", "Low"),
    "igst_share": ("IGST share", "High", "Low"),
    "cess_share": ("cess share", "High", "Low"),
    "gst_cagr": ("GST growth", "Fast", "Slow"),
}


def build_state_features(annual: pd.DataFrame, gsdp: pd.DataFrame) -> pd.DataFrame:
    """One row per state, all features scale-free (logs, shares, growth rates)."""
    states = annual[annual["entity_type"] == "State"]
    window = states[states["fiscal_year"].isin(["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"])]
    agg = window.groupby("state").agg(avg_gst=("total_gst", "mean"), igst=("igst", "sum"),
                                      cess=("cess", "sum"), gst=("total_gst", "sum"))
    feats = pd.DataFrame(index=agg.index)
    feats["log_avg_gst"] = np.log(agg["avg_gst"])
    feats["igst_share"] = agg["igst"] / agg["gst"]
    feats["cess_share"] = agg["cess"] / agg["gst"]
    feats["log_gsdp_pc"] = np.log(gsdp.groupby("state")["per_capita_gsdp"].mean())
    piv = states.pivot(index="state", columns="fiscal_year", values="total_gst")
    feats["gst_cagr"] = (piv["2024-25"] / piv["2018-19"]) ** (1 / 6) - 1
    feats["avg_gst_cr"] = agg["avg_gst"]
    feats["avg_gsdp_pc"] = np.exp(feats["log_gsdp_pc"])
    return feats


def _label_clusters(centroids_z: pd.DataFrame) -> Dict[int, str]:
    """Names each cluster after its two most distinctive features (largest |z| of the centroid)."""
    labels = {}
    for cid, row in centroids_z.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        parts = []
        for feat in top:
            noun, hi, lo = FEATURES[feat]
            parts.append(f"{hi if row[feat] > 0 else lo} {noun}")
        label = " / ".join(parts)
        labels[cid] = label[0].upper() + label[1:]
    return labels


def perform_state_clustering(features: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, Dict]:
    """
    K-Means on standardised features. If n_clusters is None, k in 2..6 is chosen by
    silhouette score. PCA is used only for the 2-D view, not for clustering.
    """
    cols = list(FEATURES)
    X = StandardScaler().fit_transform(features[cols])

    scores = {}
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(X)
        scores[k] = float(silhouette_score(X, lab))
    k = n_clusters or max(scores, key=scores.get)

    km = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X)
    pca = PCA(n_components=2, random_state=42).fit(X)
    coords = pca.transform(X)

    out = features.copy()
    out["cluster_id"] = km.labels_
    out["pca_1"], out["pca_2"] = coords[:, 0], coords[:, 1]
    centroids_z = pd.DataFrame(km.cluster_centers_, columns=cols)
    out["archetype"] = out["cluster_id"].map(_label_clusters(centroids_z))

    meta = {
        "k": k,
        "silhouette_by_k": scores,
        "silhouette": scores[k],
        "pca_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_loadings": pd.DataFrame(pca.components_.T, index=cols, columns=["PC1", "PC2"]),
        "centroids_z": centroids_z,
    }
    return out.reset_index(), meta


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad if mad > 0 else x * 0


def detect_fiscal_anomalies(monthly: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Flags unusual state-months. Features are scale-free and seasonality-adjusted, so a large
    state is not anomalous merely for being large:
      * year-on-year log change (same month last year removes seasonality), robust-z within state
      * month-on-month log change, robust-z within state
    Returns every scored state-month with an `is_anomaly` flag.
    """
    df = monthly[monthly["entity_type"] == "State"][["state", "date", "fiscal_year", "total_gst"]].copy()
    df = df.sort_values(["state", "date"])
    df["log_gst"] = np.log(df["total_gst"].clip(lower=1.0))
    g = df.groupby("state")["log_gst"]
    df["yoy_change"] = g.diff(12)
    df["mom_change"] = g.diff(1)
    df = df.dropna(subset=["yoy_change", "mom_change"]).copy()
    df["yoy_z"] = df.groupby("state")["yoy_change"].transform(_robust_z)
    df["mom_z"] = df.groupby("state")["mom_change"].transform(_robust_z)

    feats = df[["yoy_z", "mom_z"]].clip(-25, 25)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=300).fit(feats)
    df["anomaly_score"] = -iso.decision_function(feats)
    df["is_anomaly"] = iso.predict(feats) == -1
    df["yoy_change_pct"] = (np.exp(df["yoy_change"]) - 1) * 100
    df["mom_change_pct"] = (np.exp(df["mom_change"]) - 1) * 100
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
