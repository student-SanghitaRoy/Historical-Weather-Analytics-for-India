"""Reproducibility: save the analysis settings + a results fingerprint, then re-run and confirm the results are identical."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from weather_platform import config
from weather_platform.data.cities import get_city

SCHEMA_VERSION = 1
REQUIRED = ["location", "start", "end", "variables", "aggregation", "wet_day_threshold_mm"]


def results_fingerprint(df: pd.DataFrame, variables: list[str]) -> str:
    """Hash of key statistics (rounded), so a re-run can prove it reproduced the same numbers."""
    parts = {}
    for v in variables:
        s = df[v]
        parts[v] = {"n_valid": int(s.notna().sum()), "mean": round(float(s.mean()), 6) if s.notna().any() else None,
                    "sum": round(float(s.sum()), 4) if s.notna().any() else None}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def validate_config(cfg) -> list[str]:
    """Return a list of problems (empty = usable)."""
    if not isinstance(cfg, dict):
        return ["Not a valid analysis configuration file."]
    problems = [f"Missing field: {k}" for k in REQUIRED if k not in cfg]
    if problems:
        return problems
    try:
        s, e = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
        if s > e:
            problems.append("Start date is after end date.")
    except (ValueError, TypeError):
        problems.append("Dates in the file cannot be read.")
    bad = [v for v in cfg["variables"] if v not in config.VARIABLES]
    if bad or not cfg["variables"]:
        problems.append(f"Unsupported or empty variable list: {cfg['variables']}")
    if cfg["aggregation"] not in ("daily", "monthly", "seasonal"):
        problems.append(f"Unsupported aggregation: {cfg['aggregation']}")
    try:
        get_city(str(cfg["location"]).split(",")[0])
    except KeyError:
        problems.append(f"Location not supported: {cfg['location']}")
    if not isinstance(cfg["wet_day_threshold_mm"], (int, float)) or cfg["wet_day_threshold_mm"] <= 0:
        problems.append("Invalid wet-day threshold.")
    return problems


def config_to_state(cfg: dict) -> dict:
    """Map a saved configuration onto the app's widget values (keys match app.py widget keys)."""
    city = get_city(str(cfg["location"]).split(",")[0])
    by = cfg.get("baseline_years") or [1991, 2020]
    return {"state": "All", "city_query": "", "city": city, "start": pd.Timestamp(cfg["start"]).date(), "end": pd.Timestamp(cfg["end"]).date(),
            "vars": list(cfg["variables"]), "freq": cfg["aggregation"], "wet": float(cfg["wet_day_threshold_mm"]),
            "use_synth": "SYNTHETIC" in str(cfg.get("data_source", "")), "b0": int(by[0]), "b1": int(by[1])}


def compare_fingerprints(saved: str | None, now: str) -> tuple[bool | None, str]:
    if not saved:
        return None, "The saved file has no results fingerprint, so exact reproduction cannot be confirmed."
    if saved == now:
        return True, "Reproduced: the key statistics are identical to the saved analysis."
    return False, ("The key statistics differ from the saved analysis. Likely causes: the data source was updated since then, "
                   "the dates or variables were changed, or synthetic data was used instead of real data.")
