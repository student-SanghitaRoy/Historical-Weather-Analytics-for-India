"""Rainfall metrics: totals, wet/dry day counts, dry spells.

Definitions (also printed in the report methodology):
  * wet day  : valid day with precipitation >= wet_mm   (default config.WET_DAY_MM = 1.0 mm)
  * dry day  : valid day with precipitation <  wet_mm   (includes real 0.0 mm days)
  * dry spell: run of CONSECUTIVE dry days. A missing day BREAKS a run, because we cannot
               know whether it was dry. This can only shorten spells, never invent them.
  * Missing days are never counted as dry, wet or zero.
"""
from __future__ import annotations

import pandas as pd

from weather_platform import config


def rainfall_total(s: pd.Series) -> dict:
    valid = s.dropna()
    return {
        "total_mm": float(valid.sum()),
        "n_valid_days": int(valid.size),
        "n_missing_days": int(s.size - valid.size),
        # If any days are missing the true total could be higher: flag it so nobody over-reads it.
        "is_lower_bound": bool(s.size - valid.size > 0),
    }


def wet_dry_counts(s: pd.Series, wet_mm: float = config.WET_DAY_MM) -> dict:
    valid = s.dropna()
    wet = int((valid >= wet_mm).sum())
    dry = int((valid < wet_mm).sum())
    return {
        "wet_mm_threshold": wet_mm,
        "wet_days": wet,
        "dry_days": dry,
        "zero_rain_days": int((valid == 0).sum()),
        "missing_days": int(s.size - valid.size),
        "wet_day_fraction": (wet / valid.size) if valid.size else None,
    }


def dry_spells(s: pd.Series, wet_mm: float = config.WET_DAY_MM) -> pd.DataFrame:
    """All dry spells as a DataFrame: start, end, length_days (sorted by start)."""
    rows, run_start, prev = [], None, None
    for date, val in s.items():
        is_dry = pd.notna(val) and val < wet_mm
        if is_dry:
            if run_start is None:
                run_start = date
            prev = date
        else:
            if run_start is not None:
                rows.append((run_start, prev, (prev - run_start).days + 1))
                run_start = None
    if run_start is not None:
        rows.append((run_start, prev, (prev - run_start).days + 1))
    return pd.DataFrame(rows, columns=["start", "end", "length_days"])


def longest_dry_spell(s: pd.Series, wet_mm: float = config.WET_DAY_MM) -> dict:
    sp = dry_spells(s, wet_mm)
    if sp.empty:
        return {"length_days": 0, "start": None, "end": None, "n_spells": 0, "wet_mm_threshold": wet_mm}
    top = sp.loc[sp["length_days"].idxmax()]
    return {"length_days": int(top["length_days"]), "start": str(top["start"].date()),
            "end": str(top["end"].date()), "n_spells": int(len(sp)), "wet_mm_threshold": wet_mm}


def rainfall_summary(s: pd.Series, wet_mm: float = config.WET_DAY_MM) -> dict:
    return {**rainfall_total(s), **wet_dry_counts(s, wet_mm), "longest_dry_spell": longest_dry_spell(s, wet_mm)}
