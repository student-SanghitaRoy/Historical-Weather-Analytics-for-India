"""Step 2 demo: statistics + baseline comparison.

  python demo_step2.py          -> offline, SYNTHETIC data (not real weather)
  python demo_step2.py --live   -> real Guwahati data from Open-Meteo (internet needed)
"""
import json
import sys

from weather_platform.data.cities import get_city
from weather_platform.data.fetch import fetch_daily
from weather_platform.data.synthetic import make_synthetic
from weather_platform.quality.checks import clean_daily
from weather_platform.stats.baseline import compare_to_baseline
from weather_platform.stats.summary import summarize

VARS = ["temperature_2m_mean", "precipitation_sum"]
START, END = "1991-01-01", "2023-12-31"          # baseline 1991-2020 + the year we examine
PERIOD = ("2023-06-01", "2023-08-31")            # monsoon-season window to examine

if "--live" in sys.argv:
    c = get_city("Guwahati")
    raw, meta = fetch_daily(c["lat"], c["lon"], START, END, VARS)
    print("SOURCE:", meta["source"], "| from_cache:", meta["from_cache"])
else:
    raw = make_synthetic(START, END, VARS)
    print("SOURCE: SYNTHETIC test data (not real weather)")

df, rep = clean_daily(raw, VARS, START, END)
print(f"Data quality: {rep.summary_table()[['variable','pct_missing']].to_dict('records')}")

period_df = df.loc[PERIOD[0]:PERIOD[1]]
print("\n=== SUMMARY STATISTICS for", PERIOD)
print(json.dumps(summarize(period_df, VARS), indent=2, default=str))

for var, stat, rel in [("temperature_2m_mean", "mean", False), ("precipitation_sum", "sum", True)]:
    r = compare_to_baseline(df[var], *PERIOD, baseline_years=(1991, 2020), stat=stat, relative=rel)
    print(f"\n=== {var} ({stat}) vs 1991-2020 baseline")
    for k in ["status", "selected_mean", "baseline_mean", "baseline_sd", "difference", "percent_difference", "z_score",
              "percentile_in_baseline", "difference_interval", "outside_normal_variability", "n_baseline_years",
              "baseline_years_dropped_missing_data"]:
        if k in r:
            print(f"  {k}: {r[k]}")
