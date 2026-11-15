"""Supported Indian locations (data/india_cities.csv) + simple search.

Coordinates are approximate city-centre points. ERA5 is a ~0.25 degree (~25 km) grid, so a city's
values describe the grid cell around that point, not the city itself. Verify the list against
GeoNames with scripts/verify_cities_geonames.py.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).with_name("india_cities.csv")


@lru_cache(maxsize=1)
def _load() -> tuple:
    with open(_PATH, encoding="utf-8", newline="") as f:
        rows = [{"name": r["name"], "state": r["state"], "kind": r["kind"], "region": r["region"],
                 "country": "India", "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(f)]
    return tuple(rows)


def load_cities() -> list[dict]:
    return [dict(c) for c in _load()]


def search_cities(query: str) -> list[dict]:
    """Case-insensitive substring search on city, state, region or country."""
    q = query.strip().lower()
    if not q:
        return load_cities()
    return [c for c in load_cities() if q in f"{c['name']} {c['state']} {c['region']} {c['country']}".lower()]


def get_city(name: str) -> dict:
    for c in _load():
        if c["name"].lower() == name.strip().lower():
            return dict(c)
    raise KeyError(f"Unsupported city: {name!r}. Supported: {[c['name'] for c in _load()]}")


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
