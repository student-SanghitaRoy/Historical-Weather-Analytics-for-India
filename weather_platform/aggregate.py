"""Aggregation to daily / monthly / seasonal values, climatology and year-to-year tables.

Rules
-----
* Temperature and wind variables are AVERAGED; precipitation is SUMMED.
* A period is 'complete' only if >= min_valid_frac of its CALENDAR days have valid values.
  Incomplete periods get value NaN (never a partial sum presented as a total). This also makes
  partial months/seasons at the edges of the requested range show as incomplete.
* Seasons are meteorological, northern-hemisphere naming: DJF, MAM, JJA, SON. December belongs
  to the DJF season of the FOLLOWING year (DJF 2023 = Dec 2022 + Jan 2023 + Feb 2023).
  Southern-hemisphere users should read DJF as summer, JJA as winter.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from weather_platform import config

SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
FREQS = ("daily", "monthly", "seasonal")


def stat_for(var: str) -> str:
    return "sum" if config.VARIABLES[var]["kind"] == "precipitation" else "mean"


def season_bounds(season_year: int, season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = {"DJF": pd.Timestamp(season_year - 1, 12, 1), "MAM": pd.Timestamp(season_year, 3, 1),
             "JJA": pd.Timestamp(season_year, 6, 1), "SON": pd.Timestamp(season_year, 9, 1)}[season]
    last_month = {"DJF": (season_year, 2), "MAM": (season_year, 5), "JJA": (season_year, 8), "SON": (season_year, 11)}[season]
    end = pd.Timestamp(last_month[0], last_month[1], calendar.monthrange(*last_month)[1])
    return start, end


def aggregate(df: pd.DataFrame, variables: list[str], freq: str = "monthly", min_valid_frac: float = 0.9) -> pd.DataFrame:
    """Tidy table: period_start, period_label, year, sub, variable, value, stat, unit, n_valid, n_expected, complete."""
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {FREQS}")
    rows = []
    for var in variables:
        s, st, unit = df[var], stat_for(var), config.VARIABLES[var]["unit"]
        if freq == "daily":
            for d, v in s.items():
                rows.append((d, d.strftime("%Y-%m-%d"), d.year, d.month, var, v, st, unit, int(pd.notna(v)), 1, bool(pd.notna(v))))
            continue
        if freq == "monthly":
            keys = s.index.to_period("M")
            groups = s.groupby(keys)
        else:
            sy = s.index.year + (s.index.month == 12)
            season = pd.Series(s.index.month, index=s.index).map(SEASON_OF_MONTH)
            groups = s.groupby([np.asarray(sy), season.to_numpy()])
        for key, g in groups:
            if freq == "monthly":
                start = key.start_time
                expected, label, year, sub = key.days_in_month, str(key), key.year, key.month
            else:
                start, end = season_bounds(int(key[0]), key[1])
                expected, label, year, sub = (end - start).days + 1, f"{int(key[0])} {key[1]}", int(key[0]), key[1]
            n_valid = int(g.notna().sum())
            complete = n_valid / expected >= min_valid_frac
            if complete:
                value = float(g.sum(min_count=1)) if st == "sum" else float(g.mean())
            else:
                value = np.nan
            rows.append((start, label, year, sub, var, value, st, unit, n_valid, expected, bool(complete)))
    cols = ["period_start", "period_label", "year", "sub", "variable", "value", "stat", "unit", "n_valid", "n_expected", "complete"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["variable", "period_start"]).reset_index(drop=True)


def monthly_climatology(agg_monthly: pd.DataFrame, var: str) -> pd.DataFrame:
    """Average (and spread) of each calendar month across years, from COMPLETE months only."""
    a = agg_monthly[(agg_monthly["variable"] == var) & agg_monthly["complete"]]
    g = a.groupby("sub")["value"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "min": g.min(), "max": g.max(), "n_years": g.count()})
    out = out.reindex(range(1, 13))
    out.insert(0, "month", [calendar.month_abbr[m] for m in out.index])
    out.index.name = "month_number"
    return out


def pivot_by_year(agg: pd.DataFrame, var: str) -> pd.DataFrame:
    """Rows = month number (monthly) or season (seasonal); columns = year; NaN where incomplete/absent."""
    a = agg[agg["variable"] == var]
    p = a.pivot_table(index="sub", columns="year", values="value", aggfunc="first")
    if set(p.index) <= set(SEASON_ORDER):
        p = p.reindex([s for s in SEASON_ORDER if s in p.index])
    return p
