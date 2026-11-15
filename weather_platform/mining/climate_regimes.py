"""Data mining: group locations into climate regimes from their monthly climate profile.

Features per location (24): mean temperature of each calendar month (T01..T12) and mean total rainfall of
each calendar month (R01..R12), computed over complete months only. Features are standardised, then:
  * KMeans for k = 2..K, choosing k by silhouette score (reported for every k, so the choice is auditable);
  * Ward hierarchical clustering at the same k as an independent check (agreement = adjusted Rand index);
  * stability across random restarts (adjusted Rand index between different seeds);
  * PCA to 2-D for plotting.
Cluster names are DESCRIPTIVE labels from simple thresholds, not a formal classification such as Koppen.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from weather_platform.aggregate import aggregate, monthly_climatology

TCOLS = [f"T{m:02d}" for m in range(1, 13)]
RCOLS = [f"R{m:02d}" for m in range(1, 13)]


def city_profile(df: pd.DataFrame, min_years: int = 10) -> pd.Series | None:
    """24-value climate profile, or None if any month has fewer than `min_years` complete years."""
    agg = aggregate(df, ["temperature_2m_mean", "precipitation_sum"], "monthly")
    t = monthly_climatology(agg, "temperature_2m_mean")
    r = monthly_climatology(agg, "precipitation_sum")
    if (t["n_years"].fillna(0) < min_years).any() or (r["n_years"].fillna(0) < min_years).any():
        return None
    return pd.Series(list(t["mean"]) + list(r["mean"]), index=TCOLS + RCOLS)


def build_feature_matrix(datasets: dict) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = {}, []
    for name, df in datasets.items():
        p = city_profile(df)
        if p is None or p.isna().any():
            skipped.append(name)
        else:
            rows[name] = p
    return pd.DataFrame(rows).T, skipped


MONSOON_COLS = ["R06", "R07", "R08", "R09"]


def summary_features(X: pd.DataFrame) -> pd.DataFrame:
    """Four interpretable numbers per location: mean temperature, seasonal temperature range, log annual rainfall, share of rain in June-September."""
    annual = X[RCOLS].sum(axis=1)
    return pd.DataFrame({"mean_temp": X[TCOLS].mean(axis=1), "temp_range": X[TCOLS].max(axis=1) - X[TCOLS].min(axis=1),
                         "log_annual_rain": np.log1p(annual), "monsoon_share": (X[MONSOON_COLS].sum(axis=1) / annual.replace(0, np.nan)).fillna(0.0)}, index=X.index)


def describe_cluster(profile_means: pd.Series) -> str:
    temp = float(profile_means[TCOLS].mean())
    rain = float(profile_means[RCOLS].sum())
    t = "cool" if temp < 15 else "warm" if temp < 25 else "hot"
    r = "dry" if rain < 600 else "moderately wet" if rain < 1500 else "very wet" if rain < 3000 else "extremely wet"
    return f"{t}, {r}"


def cluster_regimes(X: pd.DataFrame, k_range=range(2, 9), k: int | None = None, seed: int = 0, min_auto_k: int = 3, log_rain: bool = True, features: str = "monthly") -> dict:
    """Cluster locations. Rainfall is log-transformed by default (it is extremely skewed: a few very wet places would otherwise dominate).
    The silhouette score is computed for EVERY k; the automatic choice only considers k >= min_auto_k because k = 2 usually just splits
    off the coldest hills, which is a trivial finding (the best k overall is still reported)."""
    n = len(X)
    if n < 6:
        raise ValueError("Need at least 6 locations with complete profiles to cluster.")
    if features not in ("monthly", "summary"):
        raise ValueError("features must be 'monthly' or 'summary'")
    if features == "summary":
        F = summary_features(X)
    else:
        F = X.copy()
        if log_rain:
            F[RCOLS] = np.log1p(F[RCOLS])
    Z = StandardScaler().fit_transform(F.values)
    ks = [kk for kk in k_range if 2 <= kk < n]
    table = []
    for kk in ks:
        lab = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(Z)
        table.append({"k": kk, "silhouette": float(silhouette_score(Z, lab))})
    sil = pd.DataFrame(table)
    best_overall = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    eligible = sil[sil["k"] >= min_auto_k]
    best_k = int(eligible.loc[eligible["silhouette"].idxmax(), "k"]) if len(eligible) else best_overall
    chosen = int(k) if k else best_k
    km = KMeans(n_clusters=chosen, n_init=20, random_state=seed).fit(Z)
    labels = km.labels_
    ward = AgglomerativeClustering(n_clusters=chosen, linkage="ward").fit_predict(Z)
    other = [KMeans(n_clusters=chosen, n_init=5, random_state=s).fit_predict(Z) for s in (seed + 1, seed + 2, seed + 3)]
    stability = float(np.mean([adjusted_rand_score(labels, o) for o in other]))
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    coords = pca.transform(Z)
    prof = X.copy()
    prof["cluster"] = labels
    summary_rows = []
    for c in sorted(set(labels)):
        sub = prof[prof["cluster"] == c].drop(columns="cluster")
        means = sub.mean()
        annual = sub[RCOLS].sum(axis=1)
        mixed = len(sub) > 1 and (annual.max() - annual.min()) > annual.mean()
        summary_rows.append({"cluster": c, "n_locations": len(sub), "mean_temp_C": round(float(means[TCOLS].mean()), 1),
                             "annual_rain_mm": round(float(means[RCOLS].sum())), "rain_range_mm": f"{annual.min():,.0f} to {annual.max():,.0f}",
                             "wettest_month": calendar.month_abbr[int(np.argmax(means[RCOLS].values)) + 1],
                             "description": describe_cluster(means) + (" (mixed group)" if mixed else ""), "members": ", ".join(sub.index)})
    return {"labels": dict(zip(X.index, map(int, labels))), "k": chosen, "best_k_by_silhouette": best_k, "best_k_overall": best_overall, "log_rain": log_rain, "features": features,
            "silhouette_by_k": sil, "silhouette_chosen": float(sil.loc[sil["k"] == chosen, "silhouette"].iloc[0]) if chosen in set(sil["k"]) else None,
            "ward_agreement_ari": float(adjusted_rand_score(labels, ward)), "stability_ari": stability,
            "pca_coords": pd.DataFrame(coords, index=X.index, columns=["PC1", "PC2"]),
            "pca_explained": [float(v) for v in pca.explained_variance_ratio_],
            "summary": pd.DataFrame(summary_rows),
            "caveats": ["Clusters describe similarity of long-term monthly climate profiles from ERA5 grid cells.",
                        ("Four summary features were used: mean temperature, seasonal temperature range, log annual rainfall and the June-September share of rain." if features == "summary"
                         else "Rainfall is log-transformed before clustering because it is very skewed." if log_rain else "Rainfall was NOT log-transformed."),
                        "Silhouette scores below about 0.5 mean the groups are not sharply separated; climate varies gradually, so groups are a convenient partition, not natural classes.",
                        "k = 2 is excluded from the automatic choice because it mostly just separates the coldest hills; its silhouette is still shown.",
                        "The number of clusters is a modelling choice; silhouette scores for every k are shown.",
                        "Names are descriptive threshold labels, not a formal climate classification.",
                        "Locations near each other can share a grid cell and therefore look identical."]}
