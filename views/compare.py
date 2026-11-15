import pandas as pd
import streamlit as st

from weather_platform import state, ui
from weather_platform.compare import compare_locations
from weather_platform.data.cities import load_cities
from weather_platform.viz import charts

def fmt_p(p):
    return "" if p is None else ("<0.0001" if p < 0.0001 else f"{p:.4f}")


ui.page_header("Compare locations", "Put two to six places side by side over exactly the same years, with the uncertainty of every difference.")
datasets, missing = state.require_india_data()
avail = [c for c in load_cities() if c["name"] in datasets]
names_all = [c["name"] for c in avail]

ui.explainer("Yearly averages and totals for each chosen place, using only years that are complete at every one of them.",
             "Lines that sit apart show real differences. The table of pairs gives the average yearly difference with a 95% range; if the range excludes zero, the difference is statistically clear.")
with st.container(border=True):
    _default = [n for n in ("Guwahati", "Jaisalmer") if n in names_all]
    _default += [n for n in names_all if n not in _default][: max(0, 2 - len(_default))]       # always start with two available locations
    sel = st.multiselect("Locations (2 to 6)", names_all, default=_default, max_selections=6, key="cmp_locs")
    d0 = min(df.index.min() for df in datasets.values()).date()
    d1 = max(df.index.max() for df in datasets.values()).date()
    c1, c2 = st.columns(2)
    cs0 = c1.date_input("Period start", d0, min_value=d0, max_value=d1, key="cmp_start")
    cs1 = c2.date_input("Period end", d1, min_value=d0, max_value=d1, key="cmp_end")
res = compare_locations(datasets, avail, sel, cs0, cs1)
if res["status"] != "ok":
    ui.callout("warn", "Cannot compare yet", res["reason"])
    st.stop()
for w in res["shared_grid_warnings"]:
    ui.callout("warn", "Possible shared grid cell", w)
for v, lab in (("temperature_2m_mean", "Average temperature"), ("precipitation_sum", "Yearly rainfall")):
    r = res["variables"][v]
    st.subheader(f"{lab} ({r['unit']})")
    if r["status"] != "ok":
        ui.callout("info", "", r["reason"])
        continue
    st.plotly_chart(charts.compare_lines(r["annual"], f"{lab}: the same {len(r['years_used'])} complete years at every location", r["unit"]), width="stretch", key=f"cmp_{v}")
    st.dataframe(r["summary"].rename(columns={"location": "Location", "state": "State", "mean_annual_value": "Average per year", "sd_between_years": "Year-to-year spread (sd)",
                                              "min_year_value": "Lowest year", "max_year_value": "Highest year"}).round(2), width="stretch", hide_index=True)
    rows = [{"Pair": k, "Mean difference": round(p["mean_difference"], 2), "95% range low": round(p["ci_low"], 2), "95% range high": round(p["ci_high"], 2),
             "p (paired t)": fmt_p(p["p_value_ttest"]), "p (Wilcoxon)": fmt_p(p["p_value_wilcoxon"]), "Years": p["n_years"]}
            for k, p in r["pairs"].items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.caption("Distances (km): " + "; ".join(f"{k}: {v}" for k, v in res["distances_km"].items()))
with st.expander("Spatial limitations and method"):
    st.write("Pairwise differences use year-matched annual values (paired t-test with a 95% interval, plus the Wilcoxon test). Only years complete at every location are used, and units are checked to be identical.")
    for n in res["notes"]:
        st.write("- " + n)
