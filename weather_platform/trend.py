"""Trend analysis: Mann-Kendall test, Sen's slope with a confidence interval, trend-free pre-whitening, FDR correction.

Applied to ANNUAL values (one number per complete year). Annual aggregation removes the seasonal cycle.
Serial dependence: the variance of the Mann-Kendall statistic is corrected for significant autocorrelation with the
Hamed-Rao (1998) method (an effective-sample-size style correction on the ranks of the detrended series). I first tried
trend-free pre-whitening, but a simulation test showed it made false positives WORSE on autocorrelated noise, so it was
dropped. Across many locations the p-values are adjusted with
Benjamini-Hochberg so that 'significant' is not inflated by testing 71 places.
A significant trend in ERA5 describes the reanalysis grid cell; changes in the observing system assimilated into
reanalysis can also create apparent trends (rainfall especially). It is not proof of climate change by itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

from weather_platform.stats.baseline import window_value_by_year

MIN_YEARS = 10


def mann_kendall(x, var_factor: float = 1.0) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    s = 0.0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0 * var_factor
    if var <= 0:
        z = 0.0
    else:
        z = (s - 1) / np.sqrt(var) if s > 0 else (s + 1) / np.sqrt(var) if s < 0 else 0.0
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return {"S": float(s), "var_S": float(var), "Z": float(z), "p_value": float(p), "tau": float(s / (n * (n - 1) / 2)), "n": n}


def sens_slope(x, t=None, alpha: float = 0.05, var_factor: float = 1.0) -> dict:
    """Median of pairwise slopes, with the Gilbert (1987) confidence interval based on Var(S)."""
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, float)
    slopes = np.array([(x[j] - x[i]) / (t[j] - t[i]) for i in range(len(x) - 1) for j in range(i + 1, len(x)) if t[j] != t[i]])
    slopes.sort()
    nn = len(slopes)
    var = mann_kendall(x, var_factor)["var_S"]
    c = sst.norm.ppf(1 - alpha / 2) * np.sqrt(var)
    lo_rank, hi_rank = (nn - c) / 2.0, (nn + c) / 2.0 + 1.0
    ranks = np.arange(1, nn + 1)
    lo = float(np.interp(np.clip(lo_rank, 1, nn), ranks, slopes))
    hi = float(np.interp(np.clip(hi_rank, 1, nn), ranks, slopes))
    slope = float(np.median(slopes))
    return {"slope": slope, "ci_low": lo, "ci_high": hi, "intercept": float(np.median(x - slope * t))}


def _acf(y: np.ndarray, nlags: int) -> np.ndarray:
    y = y - y.mean()
    denom = (y * y).sum()
    return np.array([(y[: len(y) - k] * y[k:]).sum() / denom if denom > 0 else 0.0 for k in range(nlags + 1)])


def hamed_rao_factor(x, t=None, alpha: float = 0.05) -> float:
    """Variance inflation factor n/n* of Hamed and Rao (1998), from significant autocorrelation of the detrended ranks."""
    x = np.asarray(x, float)
    n = len(x)
    t = np.arange(1, n + 1, dtype=float) if t is None else np.asarray(t, float)
    t = t - t[0] + 1.0   # centre the time axis: results must not depend on the calendar offset
    slope = sens_slope(x, t, alpha)["slope"]
    # Sen's slope is a median of pairwise slopes, so one pair of detrended values is mathematically tied;
    # rounding makes ties rank as ties instead of depending on floating-point noise.
    ranks = sst.rankdata(np.round(x - slope * t, 9))
    acf = _acf(ranks, n - 1)
    bound = sst.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    sni = sum((n - k) * (n - k - 1) * (n - k - 2) * acf[k] for k in range(1, n) if abs(acf[k]) > bound)
    return float(1 + 2.0 / (n * (n - 1) * (n - 2)) * sni)


def trend_test(values, years, alpha: float = 0.05, correct_autocorrelation: bool = True) -> dict:
    """Full trend result for one series (values per year). Returns status 'insufficient_data' if fewer than MIN_YEARS valid years."""
    v = pd.Series(np.asarray(values, float), index=np.asarray(years, float)).dropna()
    if len(v) < MIN_YEARS:
        return {"status": "insufficient_data", "reason": f"Only {len(v)} usable year(s); at least {MIN_YEARS} are needed.", "n_years": int(len(v))}
    t, x = v.index.to_numpy(float), v.to_numpy(float)
    factor = hamed_rao_factor(x, t, alpha) if correct_autocorrelation else 1.0
    factor = max(factor, 1e-6)
    mk = mann_kendall(x, factor)
    sen = sens_slope(x, t, alpha, factor)
    sig = mk["p_value"] < alpha
    return {"status": "ok", "n_years": int(len(x)), "first_year": int(t[0]), "last_year": int(t[-1]), "S": mk["S"], "Z": mk["Z"],
            "p_value": mk["p_value"], "tau": mk["tau"], "slope_per_year": sen["slope"], "slope_ci_low": sen["ci_low"], "slope_ci_high": sen["ci_high"],
            "slope_per_decade": 10 * sen["slope"], "intercept": sen["intercept"], "autocorr_factor": factor,
            "significant": bool(sig), "direction": ("increasing" if sen["slope"] > 0 else "decreasing" if sen["slope"] < 0 else "none") if sig else "no significant trend",
            "alpha": alpha}


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-adjusted p-values (false-discovery-rate control); NaN inputs stay NaN."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def annual_series(s: pd.Series, stat: str, min_valid_frac: float = 0.9) -> pd.Series:
    years = sorted(set(s.index.year))
    w = window_value_by_year(s, "01-01", "12-31", years, stat, min_valid_frac)
    return w.loc[w["used"], "value"]


def trend_table(datasets: dict, cities: list[dict], variable: str, alpha: float = 0.05, min_years: int = 20) -> pd.DataFrame:
    """Trend of the annual value (mean for temperature, sum for rainfall) at every location, with FDR adjustment."""
    stat = "sum" if variable == "precipitation_sum" else "mean"
    meta = {c["name"]: c for c in cities}
    rows = []
    for name, df in datasets.items():
        a = annual_series(df[variable], stat)
        r = trend_test(a.values, a.index.values, alpha) if len(a) >= min_years else {"status": "insufficient_data"}
        row = {"name": name, "state": meta[name]["state"], "n_years": int(len(a)), "status": r["status"]}
        if r["status"] == "ok":
            row.update(slope_per_decade=r["slope_per_decade"], ci_low_decade=10 * r["slope_ci_low"], ci_high_decade=10 * r["slope_ci_high"],
                       p_value=r["p_value"], autocorr_factor=r["autocorr_factor"])
        else:
            row.update(slope_per_decade=np.nan, ci_low_decade=np.nan, ci_high_decade=np.nan, p_value=np.nan, autocorr_factor=np.nan)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["p_fdr"] = benjamini_hochberg(t["p_value"].to_numpy())
    t["significant_raw"] = t["p_value"] < alpha
    t["significant_fdr"] = t["p_fdr"] < alpha
    return t
