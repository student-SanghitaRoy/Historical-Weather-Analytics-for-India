"""Weather anomaly detection against an EXPLICIT historical baseline.

Temperature (and other roughly symmetric variables): robust z-score
    z = (x - median) / (1.4826 * MAD)
  where median and MAD come from baseline-year values within +/- `window_days` of the same calendar day
  (so the seasonal cycle is removed). |z| >= 3.5 is the Iglewicz-Hoaglin rule of thumb for outliers.
Rainfall (zero-inflated, so MAD is usually 0): a day is an EXTREME if it exceeds the q-quantile (default 99th) of baseline
  WET-day amounts (>= wet_mm) within +/- `window_days` of the same calendar day.
Isolation Forest is provided ONLY as a comparison: it is unsupervised, has no built-in notion of baseline or of
"unusual for this time of year", and its flag rate is set by the contamination parameter, not by the data.
Missing values are never flagged and never filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from weather_platform import config

Z_THRESHOLD = 3.5


def doy_aligned(idx: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year 1..365 with Feb 29 folded onto Feb 28 so calendar days line up across leap years."""
    doy = idx.day_of_year.to_numpy().copy()
    return np.where(idx.is_leap_year & (doy > 59), doy - 1, doy)


def _circ_dist(a: np.ndarray, d: int) -> np.ndarray:
    diff = np.abs(a - d)
    return np.minimum(diff, 365 - diff)


def _baseline(s: pd.Series, baseline_years: tuple[int, int]) -> pd.Series:
    return s[(s.index.year >= baseline_years[0]) & (s.index.year <= baseline_years[1])].dropna()


def robust_zscores(s: pd.Series, baseline_years: tuple[int, int] = (1991, 2020), window_days: int = 7, min_n: int = 60) -> pd.DataFrame:
    """Per-day robust z-score. Columns: value, baseline_median, robust_scale, z (NaN where undefined)."""
    base = _baseline(s, baseline_years)
    bdoy, bvals = doy_aligned(base.index), base.to_numpy(float)
    med, scale = np.full(366, np.nan), np.full(366, np.nan)
    for d in range(1, 366):
        v = bvals[_circ_dist(bdoy, d) <= window_days]
        if len(v) >= min_n:
            med[d] = np.median(v)
            scale[d] = 1.4826 * np.median(np.abs(v - med[d]))
    sd = doy_aligned(s.index)
    m, sc = med[sd], scale[sd]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sc > 0, (s.to_numpy(float) - m) / sc, np.nan)
    return pd.DataFrame({"value": s.to_numpy(float), "baseline_median": m, "robust_scale": sc, "z": z}, index=s.index)


def flag_zscore(zdf: pd.DataFrame, threshold: float = Z_THRESHOLD) -> pd.Series:
    return (zdf["z"].abs() >= threshold).fillna(False)


def rainfall_extremes(s: pd.Series, baseline_years: tuple[int, int] = (1991, 2020), window_days: int = 15,
                      wet_mm: float = config.WET_DAY_MM, q: float = 0.99, min_wet: int = 30) -> pd.DataFrame:
    """Per-day threshold = q-quantile of baseline wet-day amounts near the same calendar day. Columns: value, threshold, flag."""
    base = _baseline(s, baseline_years)
    base = base[base >= wet_mm]
    bdoy, bvals = doy_aligned(base.index), base.to_numpy(float)
    thr = np.full(366, np.nan)
    for d in range(1, 366):
        v = bvals[_circ_dist(bdoy, d) <= window_days]
        if len(v) >= min_wet:
            thr[d] = np.quantile(v, q)
    t = thr[doy_aligned(s.index)]
    vals = s.to_numpy(float)
    flag = np.where(np.isnan(t) | np.isnan(vals), False, (vals > t) & (vals >= wet_mm))
    return pd.DataFrame({"value": vals, "threshold": t, "flag": flag.astype(bool)}, index=s.index)


def group_events(flag: pd.Series, score: pd.Series, min_len: int = 1) -> pd.DataFrame:
    """Merge consecutive flagged days into events: start, end, length_days, peak_score, peak_date."""
    rows, start, prev = [], None, None
    for d, f in flag.items():
        if f:
            start = start if start is not None else d
            prev = d
        elif start is not None:
            rows.append((start, prev))
            start = None
    if start is not None:
        rows.append((start, prev))
    out = []
    for a, b in rows:
        seg = score.loc[a:b]
        if len(seg) < min_len:
            continue
        pk = seg.abs().idxmax()
        out.append({"start": a.date(), "end": b.date(), "length_days": (b - a).days + 1, "peak_score": float(seg.loc[pk]), "peak_date": pk.date()})
    return pd.DataFrame(out, columns=["start", "end", "length_days", "peak_score", "peak_date"])


def isolation_forest_flags(df: pd.DataFrame, baseline_years: tuple[int, int] = (1991, 2020), contamination: float = 0.01,
                           seed: int = 0) -> pd.Series:
    """Comparison detector on [seasonally adjusted temperature, log(1 + rainfall)]. Rows with missing values are never flagged."""
    z = robust_zscores(df["temperature_2m_mean"], baseline_years)
    X = pd.DataFrame({"t_resid": z["value"] - z["baseline_median"], "rain": np.log1p(df["precipitation_sum"].to_numpy(float))}, index=df.index)
    ok = X.notna().all(axis=1)
    flags = pd.Series(False, index=df.index)
    if ok.sum() < 100:
        return flags
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200).fit(X[ok])
    flags.loc[ok] = model.predict(X[ok]) == -1
    return flags


def compare_detectors(a: pd.Series, b: pd.Series) -> dict:
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {"n_a": int(a.sum()), "n_b": int(b.sum()), "both": inter, "jaccard": (inter / union) if union else None}


def precision_recall(flag: pd.Series, truth: pd.Series) -> dict:
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}
