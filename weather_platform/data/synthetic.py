"""SYNTHETIC weather generator for offline development and tests.

NOT real data. Never present output from this module as observations. It exists so
the dashboard, statistics and tests can be developed without network access.
Optionally injects the defects the quality module must catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_platform import config


def make_synthetic(start, end, variables: list[str], seed: int = 0, inject_defects: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D", name="date")
    doy = idx.day_of_year.to_numpy()
    season = np.sin(2 * np.pi * (doy - 100) / 365.25)
    mean = 24 + 6 * season + rng.normal(0, 1.5, len(idx))
    data = {
        "temperature_2m_mean": mean,
        "temperature_2m_max": mean + 4 + rng.uniform(0, 2, len(idx)),
        "temperature_2m_min": mean - 4 - rng.uniform(0, 2, len(idx)),
        "wind_speed_10m_max": np.abs(rng.normal(14, 5, len(idx))),
    }
    wet = rng.random(len(idx)) < (0.25 + 0.3 * (season > 0.3))
    data["precipitation_sum"] = np.where(wet, rng.gamma(0.8, 9.0, len(idx)), 0.0)
    df = pd.DataFrame({v: data[v] for v in variables}, index=idx).round(1)
    df.attrs["units"] = {v: config.VARIABLES[v]["unit"] for v in variables}

    if inject_defects and len(df) > 20:
        if "precipitation_sum" in df:
            df.iloc[3:6, df.columns.get_loc("precipitation_sum")] = np.nan  # missing, not zero
            df.iloc[10, df.columns.get_loc("precipitation_sum")] = -5.0  # impossible
        df = pd.concat([df, df.iloc[[7]]])  # duplicate date
        df = df.drop(df.index[12])  # missing date
        df.attrs["units"] = {v: config.VARIABLES[v]["unit"] for v in variables}
    return df
