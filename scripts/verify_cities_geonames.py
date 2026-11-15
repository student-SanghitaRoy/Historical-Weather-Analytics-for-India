"""Verify data/india_cities.csv against GeoNames (needs internet). Prints cities >25 km from the GeoNames match."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_platform.data.cities import load_cities  # noqa: E402
from weather_platform.data.geonames_check import check_cities, parse_geonames  # noqa: E402

URL = "https://download.geonames.org/export/dump/IN.zip"

print("Downloading GeoNames India dump (CC BY 4.0)...")
with urllib.request.urlopen(URL, timeout=120) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
recs = parse_geonames(io.TextIOWrapper(z.open("IN.txt"), encoding="utf-8"))
rows = check_cities(load_cities(), recs)
bad = [r for r in rows if r["status"] != "ok"]
for r in rows:
    print(f"{r['name']:<22} {r['status']:<9} {r['distance_km']}")
print(f"\n{len(rows) - len(bad)} OK, {len(bad)} need a manual look (name spelling differs or >25 km away).")
