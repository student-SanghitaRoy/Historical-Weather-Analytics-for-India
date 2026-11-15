import streamlit as st

from weather_platform import india, narrative, state, ui
from weather_platform.data.cities import load_cities
from weather_platform.trend import trend_table
from weather_platform.viz import maps

ui.page_header("India overview", "How climate differs across the country, which years stood out, and where long-term trends are statistically real.")
datasets, missing = state.require_india_data()
cities = load_cities()
avail = [c for c in cities if c["name"] in datasets]
table = india.city_table(datasets, avail)

ui.chips([(f"{len(datasets)} of {len(cities)} locations with data", "good" if not missing else "warn"),
          (f"Record {india.PREFETCH_START[:4]} to {india.PREFETCH_END[:4]}", ""), ("ERA5 reanalysis, ~25 km grid", "")])
if missing:
    with st.expander(f"{len(missing)} location(s) not downloaded yet"):
        st.write(", ".join(missing))
findings = narrative.india_key_findings(table)
if findings:
    ui.callout_list("tip", "Key findings from the data", findings)

t_map, t_anom, t_trend = st.tabs(["Climate map", "Yearly anomalies", "Long-term trends"])

with t_map:
    ui.explainer("Each dot is a location, coloured by its long-term average (complete years only).",
                 "Hover over a dot for its value. For temperature, darker red is hotter; for rainfall, darker blue is wetter. Grey dots have no usable data.")
    metric = st.radio("Show", ["Average temperature (°C)", "Average yearly rainfall (mm)"], horizontal=True, key="ind_metric")
    col, unit = ("mean_annual_temp_C", "°C") if "temperature" in metric else ("mean_annual_rain_mm", "mm")
    vals = dict(zip(table["name"], table[col]))
    st.plotly_chart(maps.value_map(avail, vals, metric, unit, scale="YlOrRd" if "temperature" in metric else "YlGnBu"), width="stretch", key="ind_map")
    with st.expander("See the numbers"):
        st.dataframe(table.rename(columns={"name": "Location", "state": "State", "region": "Region", "mean_annual_temp_C": "Avg temp (°C)",
                                           "mean_annual_rain_mm": "Avg rain (mm/yr)", "n_years_temp": "Years (temp)", "n_years_rain": "Years (rain)",
                                           "pct_missing_rain": "% rain missing"}).round(1), width="stretch", hide_index=True)
    st.caption("Years with less than 90% valid days are excluded. Grid-cell values from reanalysis can differ from station records, especially for rainfall in hills and coastal areas.")

with t_anom:
    ui.explainer("How much a chosen year differed from each location's own long-term normal.",
                 "For rainfall, blue is wetter than normal and red is drier. For temperature, red is warmer and blue is cooler. Compare the size with how much that place normally varies from year to year (see the table). One unusual year is not climate change.")
    c1, c2, c3, c4 = st.columns(4)
    var_label = c1.radio("Measurement", ["Yearly rainfall", "Yearly mean temperature"], key="ind_avar")
    var = "precipitation_sum" if "rainfall" in var_label else "temperature_2m_mean"
    years = sorted({y for df in datasets.values() for y in df.index.year})
    year = c2.selectbox("Year", years, index=len(years) - 2 if len(years) > 1 else 0, key="ind_year")
    b0 = c3.number_input("Baseline from", 1940, 2100, state.baseline()[0], key="ind_b0")
    b1 = c4.number_input("Baseline to", 1940, 2100, state.baseline()[1], key="ind_b1")
    an = india.anomaly_table(datasets, int(year), var, (int(b0), int(b1)))
    col = "percent_difference" if var == "precipitation_sum" else "difference"
    unit = "%" if var == "precipitation_sum" else "°C"
    st.plotly_chart(maps.value_map(avail, dict(zip(an["name"], an[col])), f"{var_label}: {year} compared with {int(b0)}-{int(b1)}", unit, diverging=True, warm_is_red=(var == "temperature_2m_mean")),
                    width="stretch", key="ind_anom_map")
    ok = an[an["status"] == "ok"]
    if len(ok):
        hi, lo = ok.loc[ok[col].idxmax()], ok.loc[ok[col].idxmin()]
        ui.callout("info", "Reading this year", f"The largest positive difference was at {hi['name']} ({hi[col]:+.1f}{unit}) and the largest negative at {lo['name']} ({lo[col]:+.1f}{unit}).")
    with st.expander("See the numbers"):
        st.dataframe(an.drop(columns="status").dropna(axis=1, how="all").rename(columns={"name": "Location", "difference": "Difference", "percent_difference": "% difference", "z_score": "z-score", "percentile": "Percentile"}).round(2),
                     width="stretch", hide_index=True)
    st.caption("Each location is compared with its own baseline using the same method as the single-location page. A year inside the baseline is removed from it. Grey = not enough valid data.")

with t_trend:
    ui.explainer("The long-term change per decade at each location, using annual values from complete years.",
                 "Red means warming or increasing, blue means cooling or decreasing (for rainfall, blue means wetter). Because 71 places are tested, some would look significant by chance, so use the corrected result: tick the box to show only trends that survive that correction.")
    tl = st.radio("Measurement", ["Average temperature", "Yearly rainfall"], horizontal=True, key="ind_tvar")
    tvar = "temperature_2m_mean" if "temperature" in tl else "precipitation_sum"
    tt = trend_table(datasets, avail, tvar)
    only_sig = st.checkbox("Show only statistically significant trends (corrected for testing many places)", value=False, key="ind_sig")
    unit = "°C per decade" if tvar == "temperature_2m_mean" else "mm per decade"
    vals = {r.name: (r.slope_per_decade if (not only_sig or r.significant_fdr) else None) for r in tt.itertuples()}
    st.plotly_chart(maps.value_map(avail, vals, f"Trend in {tl.lower()} (Sen's slope)", unit, diverging=True, warm_is_red=(tvar == "temperature_2m_mean")), width="stretch", key="ind_trend_map")
    ok = tt[tt["status"] == "ok"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Locations tested", len(ok))
    m2.metric("Significant (uncorrected)", int(ok["significant_raw"].sum()), help="p < 0.05 without correcting for testing many places.")
    m3.metric("Significant after correction", int(ok["significant_fdr"].sum()), help="Benjamini-Hochberg false-discovery-rate control at 5%.")
    if len(ok):
        if tvar == "precipitation_sum":
            ui.callout("warn", "Treat rainfall trends with extra caution",
                       f"{int(ok['significant_fdr'].sum())} of {len(ok)} locations ({ok['significant_fdr'].mean():.0%}) show a statistically significant rainfall trend after correction. "
                       "Station-based studies usually find far fewer significant rainfall trends. Reanalysis rainfall can contain artificial shifts when the observations fed into it change, "
                       "so validate these against station or IMD gridded data before treating them as climate change.")
        top = ok.assign(abs_slope=ok["slope_per_decade"].abs()).nlargest(3, "abs_slope")
        ui.callout_list("info", "Largest changes per decade", [f"{r.name}: {r.slope_per_decade:+.2f} {unit} ({'significant' if r.significant_fdr else 'not significant'} after correction)" for r in top.itertuples()])
        if tvar == "temperature_2m_mean":
            st.caption("High mountain grid cells (for example Ladakh) are poorly resolved by reanalysis, so very large changes there deserve extra scrutiny.")
    with st.expander("See the numbers"):
        st.dataframe(tt.drop(columns="status").rename(columns={"name": "Location", "state": "State", "n_years": "Years", "slope_per_decade": "Slope per decade", "ci_low_decade": "95% low",
                                        "ci_high_decade": "95% high", "p_value": "p", "autocorr_factor": "Autocorr. factor", "p_fdr": "p corrected", "significant_raw": "Sig. (raw)", "significant_fdr": "Sig. (corrected)"}).round(4),
                     width="stretch", hide_index=True)
    st.caption("Mann-Kendall test with Hamed-Rao autocorrelation correction. Reanalysis trends (especially rainfall) can partly reflect changes in the observations that were assimilated; this is not evidence of a cause.")
