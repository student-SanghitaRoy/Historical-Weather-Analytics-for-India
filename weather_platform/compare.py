"""Compare several locations over a MATCHING period with consistent units and stated spatial limits.

Matching rules
  * Only calendar years that are complete (>= min_valid_frac valid days) at EVERY selected location are used, so every
    location is described by exactly the same years.
  * Units must be identical for every location, otherwise the comparison is refused.
Pairwise differences use the year-matched (paired) annual values: mean difference, 95% t-interval, paired t-test and a
Wilcoxon signed-rank p-value (more robust for skewed rainfall).
Spatial limits (returned with every result): each point is the ERA5 grid cell (~25 km) around the city centre, so elevation
and local effects are smoothed; points closer than ~0.3 degrees can share a grid cell.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sst

from weather_platform import config
from weather_platform.aggregate import stat_for
from weather_platform.data.geonames_check import haversine_km
from weather_platform.stats.baseline import window_value_by_year

NOTES = [
    "Each location is a ~25 km reanalysis grid cell around the city centre, not the city itself; hills, coasts and urban effects are smoothed.",
    "Locations closer than about 0.3 degrees (~33 km) can fall in the same grid cell and then show almost identical values.",
    "Only years that are complete at every selected location are compared, so results can differ from single-location pages.",
    "Differences describe this period only; with few years the intervals are wide.",
]
MIN_COMMON_YEARS = 5


def annual_matrix(datasets: dict, names: list[str], variable: str, start, end, min_valid_frac: float = 0.9) -> pd.DataFrame:
    """Rows = calendar years inside [start, end]; columns = locations; NaN where the year is incomplete."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    years = list(range(s.year, e.year + 1))
    stat = stat_for(variable)
    cols = {}
    for n in names:
        ser = datasets[n][variable].loc[s:e]
        w = window_value_by_year(ser, "01-01", "12-31", years, stat, min_valid_frac)
        cols[n] = w["value"]
    return pd.DataFrame(cols)


def _paired(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    d = a - b
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = sst.t.ppf(1 - alpha / 2, n - 1)
    out = {"mean_difference": mean, "ci_low": float(mean - tcrit * se), "ci_high": float(mean + tcrit * se), "n_years": n,
           "p_value_ttest": float(sst.ttest_rel(a, b).pvalue) if sd > 0 else 1.0, "p_value_wilcoxon": None}
    try:
        if np.any(d != 0):
            out["p_value_wilcoxon"] = float(sst.wilcoxon(a, b).pvalue)
    except ValueError:
        pass
    return out


def compare_locations(datasets: dict, cities: list[dict], names: list[str], start, end, variables: list[str] | None = None,
                      min_valid_frac: float = 0.9) -> dict:
    variables = variables or ["temperature_2m_mean", "precipitation_sum"]
    meta = {c["name"]: c for c in cities}
    names = [n for n in names if n in datasets]
    if len(names) < 2:
        return {"status": "insufficient_data", "reason": "Select at least two locations that have downloaded data."}
    out = {"status": "ok", "locations": names, "period": [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())], "notes": NOTES,
           "variables": {}, "distances_km": {}, "shared_grid_warnings": []}
    for a, b in itertools.combinations(names, 2):
        d = haversine_km(meta[a]["lat"], meta[a]["lon"], meta[b]["lat"], meta[b]["lon"])
        out["distances_km"][f"{a} - {b}"] = round(d, 1)
        if abs(meta[a]["lat"] - meta[b]["lat"]) < 0.3 and abs(meta[a]["lon"] - meta[b]["lon"]) < 0.3:
            out["shared_grid_warnings"].append(f"{a} and {b} are about {d:.0f} km apart and may share a reanalysis grid cell.")
    for v in variables:
        units = {n: datasets[n].attrs.get("units", {}).get(v) for n in names}
        if len(set(units.values())) != 1 or None in units.values():
            return {"status": "refused", "reason": f"Units are not consistent across locations for {v}: {units}. Comparison refused."}
        am = annual_matrix(datasets, names, v, start, end, min_valid_frac)
        common = am.dropna()
        unit = config.VARIABLES[v]["unit"] if stat_for(v) == "mean" else "mm"
        if len(common) < MIN_COMMON_YEARS:
            out["variables"][v] = {"status": "insufficient_data", "unit": unit,
                                   "reason": f"Only {len(common)} calendar year(s) are complete at every selected location; at least {MIN_COMMON_YEARS} are needed.",
                                   "years_available": {n: int(am[n].notna().sum()) for n in names}}
            continue
        summary = pd.DataFrame({"location": names, "state": [meta[n]["state"] for n in names],
                                "mean_annual_value": [float(common[n].mean()) for n in names], "sd_between_years": [float(common[n].std(ddof=1)) for n in names],
                                "min_year_value": [float(common[n].min()) for n in names], "max_year_value": [float(common[n].max()) for n in names]})
        pairs = {f"{a} minus {b}": _paired(common[a].to_numpy(), common[b].to_numpy()) for a, b in itertools.combinations(names, 2)}
        out["variables"][v] = {"status": "ok", "unit": unit, "stat": stat_for(v), "years_used": [int(y) for y in common.index],
                               "annual": common, "summary": summary, "pairs": pairs}
    return out
