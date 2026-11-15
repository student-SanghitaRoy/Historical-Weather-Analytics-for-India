"""India-wide helpers: cached multi-city datasets, location tables, anomaly maps, synthetic demo data.

Real data comes from `scripts/prefetch_india.py`, which downloads each city's daily ERA5 series once and
caches it. Everything here reads that cache; it never makes network calls.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from weather_platform.data.fetch import FetchError, fetch_daily, load_from_cache
from weather_platform.quality.checks import clean_daily
from weather_platform.stats.baseline import compare_to_baseline, window_value_by_year

PREFETCH_START = "1991-01-01"
PREFETCH_END = "2025-12-31"
PREFETCH_VARS = ["temperature_2m_mean", "precipitation_sum"]
# Always next to the code (project root), no matter which folder a command is run from.
CACHE_DIR = str(Path(__file__).resolve().parent.parent / "data_cache" / "india")

_HILLS = {"Shimla", "Darjeeling", "Shillong", "Gangtok", "Leh", "Srinagar", "Dehradun", "Cherrapunji", "Itanagar", "Kohima"}
_DESERT = {"Jaisalmer", "Bikaner", "Jodhpur", "Leh"}
_WETNESS = {"Northeast": 2.6, "South": 1.4, "West": 1.3, "East": 1.8, "Central": 1.2, "North": 0.9, "Islands": 2.2}


# ---------------------------------------------------------------------------------- cache access
def load_datasets(cities: list[dict], cache_dir: str | None = None, start: str = PREFETCH_START, end: str = PREFETCH_END,
                  variables: list[str] | None = None) -> tuple[dict, list[str]]:
    """Cleaned daily DataFrames for every city found in the cache; returns (datasets, names_not_cached)."""
    variables = variables or PREFETCH_VARS
    cache_dir = cache_dir or CACHE_DIR
    out, missing = {}, []
    for c in cities:
        hit = load_from_cache(c["lat"], c["lon"], start, end, variables, cache_dir)
        if hit is None:
            missing.append(c["name"])
            continue
        raw, meta = hit
        df, _ = clean_daily(raw, variables, start, end, units=meta.get("units"))
        out[c["name"]] = df
    return out, missing


def prefetch(cities: list[dict], cache_dir: str | None = None, delay: float = 3.0, max_new: int | None = None,
             session=None, today=None, retry_wait: float = 60.0, attempts: int = 3, sleep=None, log=print,
             start: str = PREFETCH_START, end: str = PREFETCH_END, variables: list[str] | None = None) -> dict:
    """Download missing cities one by one (resumable). Returns {'cached': [...], 'fetched': [...], 'failed': {...}, 'skipped': [...]}."""
    sleep = sleep or time.sleep
    variables = variables or PREFETCH_VARS
    cache_dir = cache_dir or CACHE_DIR
    res = {"cached": [], "fetched": [], "failed": {}, "skipped": [], "limit_hit": False}
    for i, c in enumerate(cities, 1):
        if load_from_cache(c["lat"], c["lon"], start, end, variables, cache_dir) is not None:
            res["cached"].append(c["name"])
            log(f"[{i}/{len(cities)}] {c['name']}: already cached")
            continue
        if max_new is not None and len(res["fetched"]) + len(res["failed"]) >= max_new:
            res["skipped"].append(c["name"])
            continue
        last, limit_hit = None, False
        for attempt in range(1, attempts + 1):
            try:
                fetch_daily(c["lat"], c["lon"], start, end, variables, cache_dir=cache_dir, session=session, today=today, retries=1)
                res["fetched"].append(c["name"])
                log(f"[{i}/{len(cities)}] {c['name']}: downloaded")
                last = None
                break
            except FetchError as exc:
                last = exc
                log(f"[{i}/{len(cities)}] {c['name']}: attempt {attempt} failed ({exc})")
                msg = str(exc).lower()
                if "hourly" in msg or "daily" in msg:   # waiting a minute will not help; stop and resume later
                    limit_hit = True
                    break
                sleep(retry_wait)
        if limit_hit:
            res["limit_hit"] = True
            res["skipped"].extend(x["name"] for x in cities[i - 1:] if x["name"] not in res["cached"] + res["fetched"])
            log("Hourly/daily request limit reached. Stopping now; run the same command again later to continue.")
            break
        if last is not None:
            res["failed"][c["name"]] = str(last)
        sleep(delay)
    return res


def prefetch_until_done(cities: list[dict], cache_dir: str | None = None, delay: float = 20.0, wait_minutes: float = 65.0,
                        max_rounds: int = 12, session=None, today=None, sleep=None, log=print, **kw) -> dict:
    """Keep downloading, sleeping through hourly API limits, until every location is cached (or max_rounds is reached).
    Safe to stop with Ctrl+C and to start again: finished locations are never downloaded twice."""
    sleep = sleep or time.sleep
    total = {"fetched": [], "rounds": 0, "remaining": [c["name"] for c in cities], "complete": False}
    for rnd in range(1, max_rounds + 1):
        res = prefetch(cities, cache_dir, delay=delay, session=session, today=today, sleep=sleep, log=log, **kw)
        total["fetched"] += res["fetched"]
        total["rounds"] = rnd
        total["remaining"] = res["skipped"] + list(res["failed"])
        if not total["remaining"]:
            total["complete"] = True
            log(f"All {len(cities)} locations are cached. Done.")
            return total
        log(f"Round {rnd}: {len(total['remaining'])} location(s) left ({', '.join(total['remaining'][:6])}...). "
            f"Waiting {wait_minutes:.0f} minutes for the API limit to reset; leave this window open.")
        sleep(wait_minutes * 60)
    log(f"Stopped after {max_rounds} rounds with {len(total['remaining'])} location(s) left. Run the command again to continue.")
    return total


# ---------------------------------------------------------------------------------- tables
def city_table(datasets: dict, cities: list[dict], min_valid_frac: float = 0.9) -> pd.DataFrame:
    """Per-location mean annual temperature and rainfall over COMPLETE years only."""
    meta = {c["name"]: c for c in cities}
    rows = []
    for name, df in datasets.items():
        years = sorted(set(df.index.year))
        t = window_value_by_year(df["temperature_2m_mean"], "01-01", "12-31", years, "mean", min_valid_frac)
        r = window_value_by_year(df["precipitation_sum"], "01-01", "12-31", years, "sum", min_valid_frac)
        c = meta[name]
        rows.append({"name": name, "state": c["state"], "region": c["region"],
                     "mean_annual_temp_C": float(t.loc[t["used"], "value"].mean()) if t["used"].any() else np.nan,
                     "mean_annual_rain_mm": float(r.loc[r["used"], "value"].mean()) if r["used"].any() else np.nan,
                     "n_years_temp": int(t["used"].sum()), "n_years_rain": int(r["used"].sum()),
                     "pct_missing_rain": round(100 * float(df["precipitation_sum"].isna().mean()), 2)})
    return pd.DataFrame(rows)


def anomaly_table(datasets: dict, year: int, variable: str, baseline=(1991, 2020)) -> pd.DataFrame:
    """Annual anomaly of `year` against the baseline for every location (same method as the single-city page)."""
    stat = "sum" if variable == "precipitation_sum" else "mean"
    rows = []
    for name, df in datasets.items():
        r = compare_to_baseline(df[variable], f"{year}-01-01", f"{year}-12-31", baseline, stat=stat, relative=(stat == "sum"))
        if r["status"] != "ok" or "z_score" not in r:
            rows.append({"name": name, "status": "insufficient_data", "difference": np.nan, "percent_difference": np.nan, "z_score": np.nan, "percentile": np.nan})
        else:
            rows.append({"name": name, "status": "ok", "difference": r["difference"], "percent_difference": r["percent_difference"],
                         "z_score": r["z_score"], "percentile": r["percentile_in_baseline"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- synthetic demo data
def synthetic_datasets(cities: list[dict], start: str = PREFETCH_START, end: str = PREFETCH_END, seed: int = 0) -> dict:
    """SYNTHETIC location-specific climates (hill/desert/monsoon shapes). NOT real weather; for demos and tests."""
    idx = pd.date_range(start, end, freq="D", name="date")
    doy = idx.day_of_year.to_numpy()
    out = {}
    for i, c in enumerate(cities):
        rng = np.random.default_rng(seed + i)
        base = 31.0 - 0.5 * (c["lat"] - 8.0) - (9.0 if c["name"] in _HILLS else 0.0)
        amp = 1.5 + 0.25 * (c["lat"] - 8.0)
        temp = base + amp * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 1.2, len(idx))
        annual = 700.0 * _WETNESS[c["region"]] * (0.15 if c["name"] in _DESERT else 1.0) * (3.0 if c["name"] == "Cherrapunji" else 1.0)
        monsoon = np.exp(-0.5 * ((doy - 200) / 45.0) ** 2)
        p_wet = 0.04 + 0.6 * monsoon
        scale = annual / (p_wet.sum() / len(idx) * 365.25 * 0.8)
        wet = rng.random(len(idx)) < p_wet
        rain = np.where(wet, rng.gamma(0.8, scale, len(idx)), 0.0)
        df = pd.DataFrame({"temperature_2m_mean": temp.round(1), "precipitation_sum": rain.round(1)}, index=idx)
        df.attrs["units"] = {"temperature_2m_mean": "°C", "precipitation_sum": "mm"}
        out[c["name"]] = df
    return out
