"""Shared state and data-loading helpers used by every page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from weather_platform import india, ui
from weather_platform.data.cities import load_cities
from weather_platform.data.fetch import fetch_daily, load_from_cache
from weather_platform.data.synthetic import make_synthetic
from weather_platform.quality.checks import clean_daily, validate_request

MIN_VALID = 0.9
DEFAULT_VARS = ["temperature_2m_mean", "precipitation_sum"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def settings() -> dict:
    return st.session_state.setdefault("settings", {"freq": "monthly", "wet_mm": 1.0, "baseline": (1991, 2020)})


def baseline() -> tuple[int, int]:
    b = settings()["baseline"]
    return int(b[0]), int(b[1])


def loaded() -> dict | None:
    return st.session_state.get("data")


def synthetic_mode() -> bool:
    return bool(st.session_state.get("use_synth", False))


def cache_count() -> int:
    d = Path(india.CACHE_DIR)
    return len(list(d.glob("*.meta.json"))) if d.exists() else 0


def can_use_downloaded(variables: list[str], start, end) -> bool:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return (bool(variables) and set(variables) <= set(india.PREFETCH_VARS) and s >= pd.Timestamp(india.PREFETCH_START)
            and e <= pd.Timestamp(india.PREFETCH_END))


def load_location(city: dict, start, end, variables: list[str], use_synth: bool = False) -> dict:
    """Validate, load (downloaded dataset -> live API fallback, or synthetic), clean, and package the result."""
    s, e = validate_request(start, end, variables)
    label = f"{city['name']}, {city['state']}"
    if use_synth:
        raw = make_synthetic(s, e, variables, seed=1, inject_defects=False)
        meta = {"source": "SYNTHETIC demo data (NOT real weather)", "source_url": "n/a", "data_type": "Synthetic", "latitude": city["lat"],
                "longitude": city["lon"], "start": str(s.date()), "end": str(e.date()), "from_cache": False, "units": raw.attrs["units"], "origin": "synthetic demo data"}
    else:
        hit = load_from_cache(city["lat"], city["lon"], india.PREFETCH_START, india.PREFETCH_END, india.PREFETCH_VARS, india.CACHE_DIR) if can_use_downloaded(variables, s, e) else None
        if hit is not None:
            full, meta = hit
            raw = full.loc[s:e, variables].copy()
            raw.attrs["units"] = {v: meta["units"][v] for v in variables}
            meta = {**meta, "start": str(s.date()), "end": str(e.date()), "variables": variables, "units": raw.attrs["units"], "origin": "downloaded dataset (offline)"}
        else:
            raw, meta = fetch_daily(city["lat"], city["lon"], s, e, variables, cache_dir=str(PROJECT_ROOT / ".cache"))
            meta["origin"] = "live Open-Meteo request" if not meta.get("from_cache") else "previous live request (cached)"
    meta["location"] = label
    df, rep = clean_daily(raw, variables, s, e, units=meta.get("units"))
    return {"df": df, "rep": rep, "meta": meta, "variables": variables, "city": city}


@st.cache_data(show_spinner="Preparing demo data...")
def _synthetic_all():
    return india.synthetic_datasets(load_cities())


@st.cache_data(show_spinner="Loading the downloaded data for all locations...")
def _real_all(stamp: int):
    return india.load_datasets(load_cities())


def india_datasets() -> tuple[dict, list[str]]:
    if synthetic_mode():
        return _synthetic_all(), []
    return _real_all(cache_count())


def require_india_data() -> tuple[dict, list[str]]:
    datasets, missing = india_datasets()
    if not datasets:
        ui.callout("warn", "No weather data downloaded yet",
                   "Double-click download_data.bat in the project folder (or run: python scripts/prefetch_india.py --auto) and reload this page. "
                   f"The data is stored in {india.CACHE_DIR}. To preview the platform now, switch on Demo mode in the left sidebar.")
        st.stop()
    if synthetic_mode():
        ui.callout("warn", "Demo mode", "These are SYNTHETIC data, invented for demonstration. The patterns are not real weather.")
    return datasets, missing


def require_loaded() -> dict:
    D = loaded()
    if not D:
        ui.callout("info", "No location loaded yet", "Open Location analysis, choose a place and press Load analysis. Then come back to this page.")
        st.page_link("views/location.py", label="Go to Location analysis", icon=":material/location_on:")
        st.stop()
    return D
