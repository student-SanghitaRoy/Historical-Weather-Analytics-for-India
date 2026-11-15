"""Step 1 demo.

  python demo_step1.py            -> offline run on SYNTHETIC data with injected defects
  python demo_step1.py --live     -> real Open-Meteo (ERA5) data for Guwahati (needs internet)
"""
import sys

from weather_platform.data.cities import get_city
from weather_platform.data.fetch import fetch_daily
from weather_platform.data.synthetic import make_synthetic
from weather_platform.quality.checks import clean_daily

VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
START, END = "2020-01-01", "2020-12-31"

if "--live" in sys.argv:
    city = get_city("Guwahati")
    raw, meta = fetch_daily(city["lat"], city["lon"], START, END, VARS)
    print(f"SOURCE: {meta['source']}  (from_cache={meta['from_cache']})")
else:
    raw = make_synthetic(START, END, VARS, inject_defects=True)
    print("SOURCE: SYNTHETIC test data (not real weather)")

clean, rep = clean_daily(raw, VARS, START, END)
print("\nPROCESSING STEPS")
for s in rep.steps:
    print(" ", s)
print("\nCOMPLETENESS")
print(rep.summary_table().to_string(index=False))
print("\nFIRST ROWS OF CLEAN DATA")
print(clean.head())
