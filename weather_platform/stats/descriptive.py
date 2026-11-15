"""Descriptive statistics. All functions ignore missing values (NaN) and REPORT how many were ignored."""
from __future__ import annotations

import pandas as pd

from weather_platform import config


def describe(s: pd.Series) -> dict:
    """Mean, median, min, max, sample standard deviation (ddof=1) of the valid values."""
    valid = s.dropna()
    n_valid, n_missing = int(valid.size), int(s.size - valid.size)
    base = {"n_valid": n_valid, "n_missing": n_missing}
    if n_valid == 0:
        return {**base, "mean": None, "median": None, "min": None, "max": None, "std": None,
                "min_date": None, "max_date": None}
    return {
        **base,
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "std": float(valid.std(ddof=1)) if n_valid > 1 else None,
        "min_date": str(valid.idxmin().date()),
        "max_date": str(valid.idxmax().date()),
    }


def describe_all(df: pd.DataFrame, variables: list[str]) -> dict:
    return {v: {"label": config.VARIABLES[v]["label"], "unit": config.VARIABLES[v]["unit"], **describe(df[v])}
            for v in variables}
