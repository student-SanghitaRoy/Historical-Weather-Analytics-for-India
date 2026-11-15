"""Compare a selected period with a historical baseline, with uncertainty.

Method (stated in the report methodology)
-----------------------------------------
1. For every year, take the SAME calendar window (e.g. 1 Jun - 31 Aug) and reduce it to
   one number (mean for temperature, sum for rainfall). Using annual values avoids treating
   strongly autocorrelated daily values as independent samples.
2. Years with too many missing days (default: < 90 % valid) are excluded, not zero-filled.
3. Selected years that also fall inside the baseline are removed from the baseline
   (otherwise the period would be compared partly with itself).
4. One selected year  -> compare with the baseline distribution: difference, z-score,
   percentile rank, and a 95 % interval for the difference based on year-to-year variability
   (t-distribution, uses sd * sqrt(1 + 1/n)).
   Several years     -> Welch two-sample t-test: difference, 95 % CI, p-value, Cohen's d,
   plus a bootstrap CI for the difference in means.

Limitations (returned with every result): assumes baseline years are roughly independent and
approximately normal; rainfall totals are skewed, so rely on the percentile rank and bootstrap
CI; a trend inside the baseline period inflates its variability; a significant difference
describes this grid-cell reanalysis, not necessarily a station.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

CAVEATS = [
    "Assumes baseline years are approximately independent and normally distributed.",
    "Rainfall totals are skewed; prefer the percentile rank and bootstrap interval for rainfall.",
    "A trend inside the baseline period increases its spread and weakens the comparison.",
    "Results describe ERA5 reanalysis for one grid cell, not a weather station.",
]
MIN_BASELINE_YEARS = 5


def _safe_date(year: int, md: str) -> pd.Timestamp:
    month, day = int(md[:2]), int(md[3:])
    last = pd.Timestamp(year=year, month=month, day=1).days_in_month
    return pd.Timestamp(year=year, month=month, day=min(day, last))


def window_value_by_year(s: pd.Series, start_md: str, end_md: str, years, stat: str = "mean",
                         min_valid_frac: float = 0.9) -> pd.DataFrame:
    """One value per year for the calendar window start_md..end_md ('MM-DD'). Columns:
    value, n_valid, n_expected, used. Years below min_valid_frac get value NaN / used False."""
    if stat not in ("mean", "sum"):
        raise ValueError("stat must be 'mean' or 'sum'")
    if start_md > end_md:
        raise ValueError("Windows that cross the year boundary are not supported.")
    rows = {}
    for y in years:
        a, b = _safe_date(y, start_md), _safe_date(y, end_md)
        seg = s.loc[a:b]
        expected = (b - a).days + 1
        n_valid = int(seg.notna().sum())
        ok = expected > 0 and n_valid / expected >= min_valid_frac
        val = (seg.mean() if stat == "mean" else seg.sum()) if ok else np.nan
        rows[y] = {"value": float(val) if ok else np.nan, "n_valid": n_valid, "n_expected": expected, "used": bool(ok)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")


def _welch(sel: np.ndarray, base: np.ndarray, alpha: float) -> dict:
    n1, n2 = len(sel), len(base)
    m1, m2 = sel.mean(), base.mean()
    v1, v2 = sel.var(ddof=1), base.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    diff = m1 - m2
    if se == 0:
        return {"difference": float(diff), "ci": [float(diff), float(diff)], "p_value": 1.0 if diff == 0 else 0.0,
                "cohens_d": None, "df": None}
    dfree = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = sst.t.ppf(1 - alpha / 2, dfree)
    t_stat = diff / se
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return {"difference": float(diff), "ci": [float(diff - tcrit * se), float(diff + tcrit * se)],
            "p_value": float(2 * sst.t.sf(abs(t_stat), dfree)),
            "cohens_d": float(diff / pooled) if pooled > 0 else None, "df": float(dfree)}


def compare_to_baseline(s: pd.Series, period_start, period_end, baseline_years: tuple[int, int],
                        stat: str = "mean", relative: bool = False, min_valid_frac: float = 0.9,
                        alpha: float = 0.05, n_boot: int = 5000, seed: int = 0) -> dict:
    ps, pe = pd.Timestamp(period_start).normalize(), pd.Timestamp(period_end).normalize()
    start_md, end_md = ps.strftime("%m-%d"), pe.strftime("%m-%d")
    sel_years = [ps.year] if ps.year == pe.year else list(range(ps.year, pe.year + 1))
    if ps.year == pe.year:
        end_md = pe.strftime("%m-%d")
    out = {"status": "ok", "stat": stat, "period": [str(ps.date()), str(pe.date())],
           "calendar_window": [start_md, end_md], "baseline_years_requested": list(baseline_years),
           "alpha": alpha, "caveats": CAVEATS}

    base_years = [y for y in range(baseline_years[0], baseline_years[1] + 1) if y not in sel_years]
    out["baseline_years_excluded_overlap"] = [y for y in range(baseline_years[0], baseline_years[1] + 1) if y in sel_years]
    sel_tbl = window_value_by_year(s, start_md, end_md, sel_years, stat, min_valid_frac)
    base_tbl = window_value_by_year(s, start_md, end_md, base_years, stat, min_valid_frac)
    sel = sel_tbl.loc[sel_tbl["used"], "value"].to_numpy(float)
    base = base_tbl.loc[base_tbl["used"], "value"].to_numpy(float)
    out["selected_years_used"] = [int(y) for y in sel_tbl.index[sel_tbl["used"]]]
    out["baseline_years_used"] = [int(y) for y in base_tbl.index[base_tbl["used"]]]
    out["baseline_years_dropped_missing_data"] = [int(y) for y in base_tbl.index[~base_tbl["used"]]]

    if len(sel) == 0:
        return {**out, "status": "insufficient_data", "reason": "The selected period has too many missing days."}
    if len(base) < MIN_BASELINE_YEARS:
        return {**out, "status": "insufficient_data",
                "reason": f"Only {len(base)} usable baseline year(s); at least {MIN_BASELINE_YEARS} are required."}

    bm, bsd = float(base.mean()), float(base.std(ddof=1))
    sm = float(sel.mean())
    diff = sm - bm
    out.update({"selected_mean": sm, "selected_values": {int(y): float(v) for y, v in zip(out["selected_years_used"], sel)},
                "baseline_mean": bm, "baseline_sd": bsd, "n_baseline_years": int(len(base)),
                "difference": diff,
                "percent_difference": (100 * diff / bm) if (relative and bm != 0) else None})
    n = len(base)
    tcrit = sst.t.ppf(1 - alpha / 2, n - 1)
    out["baseline_mean_ci"] = [float(bm - tcrit * bsd / np.sqrt(n)), float(bm + tcrit * bsd / np.sqrt(n))]

    if len(sel) == 1:
        half = tcrit * bsd * np.sqrt(1 + 1 / n)
        less, equal = (base < sm).sum(), (base == sm).sum()
        out.update({"method": "single period vs baseline distribution (t-based)",
                    "z_score": float(diff / bsd) if bsd > 0 else None,
                    "percentile_in_baseline": float(100 * (less + 0.5 * equal) / n),
                    "difference_interval": [float(diff - half), float(diff + half)],
                    "outside_normal_variability": bool(abs(diff) > half),
                    "interval_meaning": "Range of differences expected from normal year-to-year variability; "
                                        "a difference outside it is unusual relative to the baseline."})
    else:
        w = _welch(sel, base, alpha)
        rng = np.random.default_rng(seed)
        boots = [rng.choice(sel, len(sel)).mean() - rng.choice(base, len(base)).mean() for _ in range(n_boot)]
        out.update({"method": "Welch two-sample t-test on annual window values",
                    "difference_ci": w["ci"], "p_value": w["p_value"], "cohens_d": w["cohens_d"], "welch_df": w["df"],
                    "bootstrap_ci": [float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))],
                    "n_selected_years": int(len(sel))})
    return out
