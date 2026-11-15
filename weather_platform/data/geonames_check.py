"""Cross-check our city coordinates against GeoNames (https://download.geonames.org/export/dump/IN.zip).

GeoNames is CC BY 4.0; attribute it if you republish. The pure functions here are unit-tested offline;
scripts/verify_cities_geonames.py does the download.
"""
from __future__ import annotations

import math


def parse_geonames(lines) -> list[dict]:
    """Parse tab-separated GeoNames dump lines (IN.txt format)."""
    recs = []
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 15:
            continue
        try:
            recs.append({"name": p[1], "ascii": p[2], "alt": [a.lower() for a in p[3].split(",") if a], "lat": float(p[4]), "lon": float(p[5]),
                         "fclass": p[6], "population": int(p[14] or 0)})
        except ValueError:
            continue
    return recs


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def find_match(city: dict, recs: list[dict]) -> dict | None:
    """Most populous populated-place record (feature class P) whose name matches the city name."""
    n = city["name"].lower()
    cands = [r for r in recs if r["fclass"] == "P" and (r["ascii"].lower() == n or r["name"].lower() == n or n in r["alt"])]
    return max(cands, key=lambda r: r["population"]) if cands else None


def check_cities(cities: list[dict], recs: list[dict], tol_km: float = 25.0) -> list[dict]:
    out = []
    for c in cities:
        m = find_match(c, recs)
        if m is None:
            out.append({"name": c["name"], "status": "no match", "distance_km": None})
        else:
            d = haversine_km(c["lat"], c["lon"], m["lat"], m["lon"])
            out.append({"name": c["name"], "status": "ok" if d <= tol_km else "CHECK", "distance_km": round(d, 1)})
    return out
