"""One call that produces every statistic for a cleaned DataFrame (the 'evidence' for later steps)."""
from __future__ import annotations

import pandas as pd

from weather_platform import config
from weather_platform.stats.descriptive import describe_all
from weather_platform.stats.rainfall import rainfall_summary


def summarize(df: pd.DataFrame, variables: list[str], wet_mm: float = config.WET_DAY_MM) -> dict:
    out = {"period": [str(df.index.min().date()), str(df.index.max().date())], "n_days": int(len(df)),
           "descriptive": describe_all(df, variables)}
    if "precipitation_sum" in variables:
        out["rainfall"] = rainfall_summary(df["precipitation_sum"], wet_mm)
    return out
