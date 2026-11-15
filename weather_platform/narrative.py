"""Plain-language descriptions generated from computed numbers (nothing is invented: every sentence is filled from data)."""
from __future__ import annotations

import calendar

import pandas as pd

from weather_platform.aggregate import aggregate, monthly_climatology


def _temp_word(t: float) -> str:
    return "cool" if t < 15 else "warm" if t < 25 else "hot"


def _rain_word(r: float) -> str:
    return "dry" if r < 600 else "moderately wet" if r < 1500 else "very wet" if r < 3000 else "extremely wet"


def climate_profile(df: pd.DataFrame) -> dict:
    """Annual means and seasonal highlights from complete months only. Missing pieces are simply absent."""
    out: dict = {}
    cols = [c for c in ("temperature_2m_mean", "precipitation_sum") if c in df.columns]
    if not cols:
        return out
    agg = aggregate(df, cols, "monthly")
    if "temperature_2m_mean" in cols:
        t = monthly_climatology(agg, "temperature_2m_mean")["mean"]
        if t.notna().all():
            out.update(annual_temp=float(t.mean()), hottest_month=int(t.idxmax()), coolest_month=int(t.idxmin()))
    if "precipitation_sum" in cols:
        r = monthly_climatology(agg, "precipitation_sum")["mean"]
        if r.notna().all() and r.sum() > 0:
            out.update(annual_rain=float(r.sum()), wettest_month=int(r.idxmax()), driest_month=int(r.idxmin()),
                       monsoon_share=float(r.loc[6:9].sum() / r.sum()))
    return out


def describe_climate(name: str, p: dict) -> str:
    if not p:
        return f"There is not enough complete monthly data to describe the climate of {name}."
    parts = []
    if "annual_temp" in p and "annual_rain" in p:
        parts.append(f"{name} has a {_temp_word(p['annual_temp'])}, {_rain_word(p['annual_rain'])} climate: the average temperature is about "
                     f"{p['annual_temp']:.1f} °C and about {p['annual_rain']:,.0f} mm of rain falls in a typical year.")
    elif "annual_temp" in p:
        parts.append(f"The average temperature at {name} is about {p['annual_temp']:.1f} °C ({_temp_word(p['annual_temp'])}).")
    elif "annual_rain" in p:
        parts.append(f"About {p['annual_rain']:,.0f} mm of rain falls at {name} in a typical year ({_rain_word(p['annual_rain'])}).")
    if "annual_rain" in p:
        wm = calendar.month_name[p["wettest_month"]]
        parts.append(f"{wm} is typically the wettest month and about {100 * p['monsoon_share']:.0f}% of the yearly rain falls between June and September.")
    if "annual_temp" in p:
        parts.append(f"{calendar.month_name[p['hottest_month']]} is typically the warmest month and {calendar.month_name[p['coolest_month']]} the coolest.")
    return " ".join(parts)


def india_key_findings(table: pd.DataFrame, k: int = 3) -> list[str]:
    """Top/bottom locations from the city table (annual mean temperature and rainfall)."""
    out = []
    t = table.dropna(subset=["mean_annual_rain_mm"])
    if len(t) >= 2 * k:
        wet = t.nlargest(k, "mean_annual_rain_mm")
        dry = t.nsmallest(k, "mean_annual_rain_mm")
        out.append("Wettest: " + ", ".join(f"{r.name} ({r.mean_annual_rain_mm:,.0f} mm)" for r in wet.itertuples()))
        out.append("Driest: " + ", ".join(f"{r.name} ({r.mean_annual_rain_mm:,.0f} mm)" for r in dry.itertuples()))
    t = table.dropna(subset=["mean_annual_temp_C"])
    if len(t) >= 2 * k:
        hot = t.nlargest(k, "mean_annual_temp_C")
        cool = t.nsmallest(k, "mean_annual_temp_C")
        out.append("Hottest on average: " + ", ".join(f"{r.name} ({r.mean_annual_temp_C:.1f} °C)" for r in hot.itertuples()))
        out.append("Coolest on average: " + ", ".join(f"{r.name} ({r.mean_annual_temp_C:.1f} °C)" for r in cool.itertuples()))
    return out
