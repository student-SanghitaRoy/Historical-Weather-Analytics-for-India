from datetime import date

import pandas as pd
import streamlit as st

from weather_platform import config, india, narrative, state, ui
from weather_platform.aggregate import aggregate, monthly_climatology, pivot_by_year, stat_for
from weather_platform.anomaly import (Z_THRESHOLD, compare_detectors, flag_zscore, group_events, isolation_forest_flags, rainfall_extremes, robust_zscores)
from weather_platform.data.cities import load_cities
from weather_platform.stats.baseline import compare_to_baseline
from weather_platform.stats.descriptive import describe_all
from weather_platform.stats.rainfall import rainfall_summary
from weather_platform.trend import MIN_YEARS, annual_series, trend_test
from weather_platform.viz import charts, maps

MIN_VALID = state.MIN_VALID
cities = load_cities()
PRESETS = ["Full record", "Last 10 years", "Last 5 years", "Custom dates"]
END = pd.Timestamp(india.PREFETCH_END)

ui.page_header("Location analysis", "Choose a place and a period, then read a plain-language summary backed by charts, statistics, unusual days and trends.")

# ---- a re-run from a saved configuration pre-fills the controls (must happen before the widgets are created)
_pend = st.session_state.pop("_pending_cfg_state", None)
if _pend:
    for k in ("state", "city", "start", "end", "vars", "freq", "wet", "b0", "b1"):
        if k in _pend:
            st.session_state[k] = _pend[k]
    st.session_state["preset"] = "Custom dates"

# ---------------------------------------------------------------- control panel
with st.container(border=True):
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
    with c1:
        st.markdown("**1. Where**")
        state_sel = st.selectbox("State / UT", ["All"] + sorted({c["state"] for c in cities}), key="state")
        pool = [c for c in cities if state_sel == "All" or c["state"] == state_sel]
        default_idx = next((i for i, c in enumerate(pool) if c["name"] == "Guwahati"), 0)
        city = st.selectbox("Location", pool, index=default_idx, format_func=lambda c: f"{c['name']}, {c['state']}", key="city")
    with c2:
        st.markdown("**2. When**")
        preset = st.radio("Period", PRESETS, key="preset", help="'Full record' uses all downloaded years (1991 to 2025), which is best for baselines and trends.")
        if preset == "Custom dates":
            start = st.date_input("Start date", date(2015, 1, 1), min_value=date(1940, 1, 1), max_value=date.today(), key="start")
            end = st.date_input("End date", date(2023, 12, 31), min_value=date(1940, 1, 1), max_value=date.today(), key="end")
        else:
            start = pd.Timestamp(india.PREFETCH_START).date() if preset == "Full record" else date(END.year - (9 if preset == "Last 10 years" else 4), 1, 1)
            end = END.date()
    with c3:
        st.markdown("**3. What**")
        variables = st.multiselect("Measurements", list(config.VARIABLES), default=state.DEFAULT_VARS, key="vars",
                                   format_func=lambda v: f"{config.VARIABLES[v]['label']} ({config.VARIABLES[v]['unit']})")
        if state.synthetic_mode():
            st.caption("Demo mode: synthetic data (not real).")
        elif state.can_use_downloaded(variables, start, end) and state.cache_count():
            st.caption("Uses your downloaded dataset: instant and works offline.")
        else:
            st.caption("Needs a live request to Open-Meteo (internet; the free service has hourly limits).")
    load_clicked = st.button("Load analysis", type="primary", key="load")

with st.expander("Analysis settings (optional)"):
    s1, s2, s3, s4 = st.columns(4)
    freq = s1.selectbox("Rainfall chart grouping", ["daily", "monthly", "seasonal"], index=1, key="freq")
    wet_mm = s2.number_input("Wet-day threshold (mm)", 0.1, 20.0, config.WET_DAY_MM, 0.1, key="wet", help="A day with at least this much rain counts as a wet day.")
    b0 = s3.number_input("Baseline from (year)", 1940, 2100, 1991, key="b0")
    b1 = s4.number_input("Baseline to (year)", 1940, 2100, 2020, key="b1", help="The 'normal' period that other years are compared with. 30 years is standard.")
if b0 >= b1:
    st.error("The baseline start year must be before the end year.")
    b0, b1 = 1991, 2020
S = state.settings()
S.update(freq=freq, wet_mm=float(wet_mm), baseline=(int(b0), int(b1)))

# ---------------------------------------------------------------- loading
auto = st.session_state.pop("_auto_load", False)
if state.loaded() is None and not st.session_state.get("_loc_visited") and (state.cache_count() or state.synthetic_mode()):
    auto = True
st.session_state["_loc_visited"] = True
if load_clicked or auto:
    try:
        with st.spinner("Loading and checking the data..."):
            st.session_state["data"] = state.load_location(city, start, end, variables, state.synthetic_mode())
        st.session_state.pop("pdf_bytes", None)
        st.session_state.pop("eval_out", None)
    except ValueError as exc:
        st.error(f"Invalid selection: {exc}")
    except Exception as exc:
        st.error(f"Could not load data: {exc}")

D = state.loaded()
if not D:
    ui.callout("info", "Nothing loaded yet", "Choose a location, period and measurements above, then press Load analysis. "
               "If you have not downloaded the data yet, run download_data.bat, or switch on Demo mode in the sidebar to try the platform.")
    st.stop()
df, rep, meta, VARS = D["df"], D["rep"], D["meta"], D["variables"]
name = D["city"]["name"]
if D["city"]["name"] != city["name"] or D["meta"]["start"] != str(pd.Timestamp(start).date()) or D["meta"]["end"] != str(pd.Timestamp(end).date()) or list(VARS) != list(variables):
    ui.callout("warn", "Your selection has changed", "The analysis below still shows the previous selection. Press Load analysis to update it.")
if "SYNTHETIC" in meta["source"]:
    ui.callout("warn", "Demo data", "This analysis uses SYNTHETIC demo data. It is not real weather.")
if not rep.units_ok:
    st.error("Unit validation failed: " + " | ".join(rep.unit_issues))
completeness = 100 - sum(v["pct_missing"] for v in rep.per_variable.values()) / max(1, len(rep.per_variable))
ui.chips([(meta["location"], ""), (f"{meta['start']} to {meta['end']} ({rep.n_expected_days:,} days)", ""), (meta.get("origin", meta["source"]), ""),
          (f"{completeness:.1f}% complete", "good" if completeness >= 98 else "warn")])

if st.session_state.get("_saved_fp") is not None or st.session_state.get("_cfg_sig"):
    from weather_platform.reproduce import compare_fingerprints, results_fingerprint
    _ok, _msg = compare_fingerprints(st.session_state.get("_saved_fp"), results_fingerprint(df, VARS))
    (st.success if _ok else st.warning)("Re-run check: " + _msg)

lab = lambda v: config.VARIABLES[v]["label"]  # noqa: E731
unit_of = lambda v: config.VARIABLES[v]["unit"]  # noqa: E731
t_sum, t_charts, t_stats, t_anom, t_quality = st.tabs(["Summary", "Charts", "Statistics", "Unusual days and trends", "Data quality"])

# ================================================================ SUMMARY
with t_sum:
    prof = narrative.climate_profile(df)
    ui.callout("tip", "In plain words", narrative.describe_climate(name, prof))
    items = []
    if "annual_temp" in prof:
        items.append(("Average temperature", f"{prof['annual_temp']:.1f} °C", "across a typical year"))
    if "annual_rain" in prof:
        items.append(("Rainfall in a typical year", f"{prof['annual_rain']:,.0f} mm", "sum of the monthly averages"))
    if "precipitation_sum" in VARS:
        rs0 = rainfall_summary(df["precipitation_sum"], wet_mm)
        valid = rs0["wet_days"] + rs0["dry_days"]
        if valid:
            items.append(("Wet days per year", f"{rs0['wet_days'] / valid * 365.25:.0f}", f"days with at least {wet_mm:g} mm"))
    items.append(("Data completeness", f"{completeness:.1f}%", "of days have a valid value"))
    ui.kpis(items)
    agg_m = aggregate(df, [v for v in ("temperature_2m_mean", "precipitation_sum") if v in VARS], "monthly", MIN_VALID)
    tclim = monthly_climatology(agg_m, "temperature_2m_mean") if "temperature_2m_mean" in VARS else None
    rclim = monthly_climatology(agg_m, "precipitation_sum") if "precipitation_sum" in VARS else None
    if tclim is not None or rclim is not None:
        ui.explainer("The average temperature (orange line) and rainfall (blue bars) for each month of the year, averaged over all complete years.",
                     "Tall bars show the rainy season. Where the line is highest is the hottest part of the year.")
        st.plotly_chart(charts.climate_at_a_glance(tclim, rclim), width="stretch", key="sum_glance")
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(maps.location_map(cities, selected=name, title="Where this location is"), width="stretch", key="sum_map")
    with right:
        st.markdown("**What to explore next**")
        ui.cards([("Charts", "Open the Charts tab for time series, seasons and year-to-year comparison."),
                  ("Statistics", "See averages, rainfall counts and a comparison with the 1991 to 2020 normal."),
                  ("Unusual days and trends", "Find extreme days and test for long-term change.")])
        st.page_link("views/ask.py", label="Ask a question about this data", icon=":material/chat:")
        st.page_link("views/export.py", label="Download a PDF report", icon=":material/download:")

# ================================================================ CHARTS
with t_charts:
    temp_cols = [v for v in VARS if config.VARIABLES[v]["kind"] == "temperature"]
    if temp_cols:
        st.subheader("Temperature over time")
        ui.explainer("Daily temperatures (thin lines) and their 30-day average (thick lines).", "The wave is the seasons. A line that is lower or higher than in other years stands out. Gaps mean missing data.")
        st.plotly_chart(charts.temperature_trend(df[temp_cols]), width="stretch", key="ch_temp")
    if "precipitation_sum" in VARS:
        st.subheader("Rainfall")
        ui.explainer(f"{freq.capitalize()} rainfall. Change the grouping in Analysis settings.", "Taller bars mean more rain. A missing bar is a period with too little valid data, which is different from a dry period.")
        agg = aggregate(df, ["precipitation_sum"], freq, MIN_VALID)
        st.plotly_chart(charts.rainfall_chart(agg, freq), width="stretch", key="ch_rain")
    for v in [x for x in VARS if config.VARIABLES[x]["kind"] == "wind"]:
        st.subheader(lab(v))
        st.line_chart(df[v])
    st.subheader("Seasonal pattern and year-to-year comparison")
    cv = st.selectbox("Measurement", VARS, format_func=lab, key="cmp_var")
    cf = st.radio("Group by", ["monthly", "seasonal"], horizontal=True, key="cmp_freq")
    cagg = aggregate(df, [cv], cf, MIN_VALID)
    unit = unit_of(cv)
    word = "total" if stat_for(cv) == "sum" else "mean"
    piv = pivot_by_year(cagg, cv)
    ui.explainer(f"The {word} of {lab(cv).lower()} for each {'month' if cf == 'monthly' else 'season'}, one line per year.",
                 "Lines that follow each other show a regular pattern. A line far from the others is an unusual year.")
    st.plotly_chart(charts.year_comparison(piv, f"Year-to-year comparison ({cf} {word})", f"{lab(cv)} ({unit})"), width="stretch", key="ch_years")
    if cf == "monthly":
        clim = monthly_climatology(cagg, cv)
        st.plotly_chart(charts.climatology_chart(clim, f"Typical year: monthly {word} (average over years; bars show 1 standard deviation)", unit), width="stretch", key="ch_clim")
        with st.expander("See the numbers"):
            st.dataframe(clim.reset_index(drop=True).rename(columns={"month": "Month", "mean": f"Average ({unit})", "std": "Std. deviation", "min": "Lowest year",
                                                                    "max": "Highest year", "n_years": "Years"}).round(2), width="stretch", hide_index=True)
    with st.expander("Table of values by year"):
        _p = piv.round(1).copy()
        _p.index.name = "Month number" if cf == "monthly" else "Season"
        st.dataframe(_p, width="stretch")

# ================================================================ STATISTICS
with t_stats:
    st.subheader("Summary statistics")
    ui.explainer("Average, middle value, extremes and spread of the daily values over the whole period.",
                 "The standard deviation tells you how much values typically vary. Missing days are ignored and counted separately.")
    d = describe_all(df, VARS)
    dtab = pd.DataFrame(d).T[["label", "unit", "mean", "median", "min", "max", "std", "n_valid", "n_missing"]]
    dtab.columns = ["Measurement", "Unit", "Average", "Median", "Lowest", "Highest", "Std. deviation", "Valid days", "Missing days"]
    for _c in ["Average", "Median", "Lowest", "Highest", "Std. deviation"]:
        dtab[_c] = pd.to_numeric(dtab[_c]).round(2)
    for _c in ["Valid days", "Missing days"]:
        dtab[_c] = pd.to_numeric(dtab[_c]).astype(int)
    st.dataframe(dtab, width="stretch", hide_index=True)
    if "precipitation_sum" in VARS:
        st.subheader("Rainfall in detail")
        rs = rainfall_summary(df["precipitation_sum"], wet_mm)
        ds = rs["longest_dry_spell"]
        ui.kpis([("Total rainfall", f"{rs['total_mm']:,.0f} mm", "lower bound: days missing" if rs["is_lower_bound"] else "whole period"),
                 ("Wet days", f"{rs['wet_days']:,}", f"at least {wet_mm:g} mm"), ("Zero-rain days", f"{rs['zero_rain_days']:,}", "a real measurement of 0 mm"),
                 ("Longest dry spell", f"{ds['length_days']} days", f"{ds['start']} to {ds['end']}" if ds["start"] else "none")])
        st.caption("A dry spell is a run of consecutive days below the wet-day threshold. A missing day ends a spell because we cannot know whether it was dry.")
        if rs["is_lower_bound"]:
            ui.callout("warn", "Some days are missing", f"{rs['n_missing_days']} day(s) have no value, so the total is a lower bound.")

    st.subheader("Compare a period with the normal (baseline)")
    ui.explainer("How a chosen period compares with the same time of year in the baseline years, including how large and how uncertain the difference is.",
                 "The percentile says where it ranks (50 is typical, above 90 is high). The 95% range shows what normal year-to-year variation would give.")
    bv = st.selectbox("Measurement", VARS, format_func=lab, key="base_var")
    last_year = df.index.max().year if df.index.max() >= pd.Timestamp(df.index.max().year, 12, 25) else df.index.max().year - 1
    p1, p2 = st.columns(2)
    ps = p1.date_input("Period start", date(last_year, 1, 1), min_value=df.index.min().date(), max_value=df.index.max().date(), key="p_start")
    pe = p2.date_input("Period end", date(last_year, 12, 31), min_value=df.index.min().date(), max_value=df.index.max().date(), key="p_end")
    if pe < ps:
        st.error("Period end is before period start.")
    else:
        res = compare_to_baseline(df[bv], ps, pe, (int(b0), int(b1)), stat=stat_for(bv), relative=stat_for(bv) == "sum", min_valid_frac=MIN_VALID)
        if res["status"] != "ok":
            ui.callout("warn", "Not enough data for a reliable comparison", f"{res['reason']} Load the Full record period so the baseline years are included.")
        else:
            u = unit_of(bv) if stat_for(bv) == "mean" else "mm"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Selected period ({u})", f"{res['selected_mean']:.1f}")
            m2.metric(f"Normal ({u})", f"{res['baseline_mean']:.1f}", help=f"Baseline mean; sd {res['baseline_sd']:.2f}; {res['n_baseline_years']} years")
            m3.metric("Difference", f"{res['difference']:+.2f}", f"{res['percent_difference']:+.1f}%" if res["percent_difference"] is not None else None)
            if "z_score" in res:
                m4.metric("Percentile", f"{res['percentile_in_baseline']:.0f}", help="Where this period ranks among the baseline years.")
                lo, hi = res["difference_interval"]
                verdict = "outside" if res["outside_normal_variability"] else "within"
                direction = "above" if res["difference"] > 0 else "below"
                ui.callout("good" if verdict == "within" else "warn", "In plain words",
                           f"The selected period was {abs(res['difference']):.2f} {u} {direction} the normal. Normal year-to-year variation would give a difference between {lo:+.2f} and {hi:+.2f} {u}, "
                           f"so this difference is {verdict} the usual range.")
            else:
                m4.metric("p-value", f"{res['p_value']:.3f}")
                ui.callout("info", "In plain words", f"The 95% range for the difference is {res['difference_ci'][0]:+.2f} to {res['difference_ci'][1]:+.2f} {u} "
                           f"(bootstrap: {res['bootstrap_ci'][0]:+.2f} to {res['bootstrap_ci'][1]:+.2f}); the effect size (Cohen's d) is {res['cohens_d']:.2f}.")
            if res["n_baseline_years"] < 20:
                st.caption(f"Short baseline: only {res['n_baseline_years']} usable year(s); a normal baseline has about 30, so treat this with caution.")
            with st.expander("Method, assumptions and full output"):
                st.write(res["method"])
                for c in res["caveats"]:
                    st.write("- " + c)
                st.json({k: v for k, v in res.items() if k != "caveats"}, expanded=False)

# ================================================================ UNUSUAL DAYS AND TRENDS
with t_anom:
    ab0, ab1 = int(b0), int(b1)
    st.subheader("Unusual days")
    ui.explainer(f"Days that were very different from what is normal for that time of year, judged against the baseline {ab0} to {ab1}.",
                 "Red dots are flagged days. Neighbouring flagged days are merged into one event. Change the baseline in Analysis settings.")
    av = st.selectbox("Measurement", VARS, format_func=lab, key="an_var")
    au = unit_of(av)
    if av == "precipitation_sum":
        rdf = rainfall_extremes(df[av], (ab0, ab1), wet_mm=wet_mm)
        if rdf["threshold"].notna().sum() == 0:
            ui.callout("warn", "Not enough baseline data", "There are too few wet days in the baseline years to define extreme rain. Load the Full record period.")
        else:
            flags = rdf["flag"]
            ev = group_events(flags, rdf["value"])
            k1, k2 = st.columns(2)
            k1.metric("Extreme-rain days", int(flags.sum()))
            k2.metric("Events (consecutive days merged)", len(ev))
            st.plotly_chart(charts.rain_extreme_chart(rdf, "Daily precipitation and extreme-day threshold"), width="stretch", key="an_chart")
            st.caption("An extreme day exceeds the 99th percentile of baseline wet-day amounts for that time of year, so about 1% of normal wet days exceed it by definition.")
            st.dataframe(ev.reindex(ev["peak_score"].abs().sort_values(ascending=False).index).head(25).rename(columns={"start": "Start", "end": "End", "length_days": "Days", "peak_score": "Peak (mm)", "peak_date": "Peak date"}),
                         width="stretch", hide_index=True)
    else:
        zdf = robust_zscores(df[av], (ab0, ab1))
        if zdf["z"].notna().sum() == 0:
            ui.callout("warn", "Not enough baseline data", "There are too few baseline years to judge what is unusual. Load the Full record period.")
        else:
            flags = flag_zscore(zdf)
            ev = group_events(flags, zdf["z"].where(flags).fillna(0.0))
            k1, k2, k3 = st.columns(3)
            k1.metric("Flagged days", int(flags.sum()))
            k2.metric("Events (consecutive days merged)", len(ev))
            k3.metric("Valid days checked", int(zdf["z"].notna().sum()))
            st.plotly_chart(charts.anomaly_chart(zdf, flags, f"{lab(av)}: unusual days", f"{lab(av)} ({au})"), width="stretch", key="an_chart")
            st.caption(f"Rule: a day is flagged when its robust z-score is at least {Z_THRESHOLD} in size, where z = (value minus baseline median) divided by (1.4826 x MAD). Missing days are never flagged.")
            st.dataframe(ev.reindex(ev["peak_score"].abs().sort_values(ascending=False).index).head(25).rename(columns={"start": "Start", "end": "End", "length_days": "Days", "peak_score": "Peak z", "peak_date": "Peak date"}),
                         width="stretch", hide_index=True)
            if "temperature_2m_mean" in VARS and "precipitation_sum" in VARS and st.checkbox("Cross-check with Isolation Forest (machine learning)", key="an_iso"):
                iso = isolation_forest_flags(df, (ab0, ab1))
                cmp_ = compare_detectors(flag_zscore(robust_zscores(df["temperature_2m_mean"], (ab0, ab1))), iso)
                st.write(f"The statistical rule flagged {cmp_['n_a']} days, Isolation Forest flagged {cmp_['n_b']}, and {cmp_['both']} were flagged by both (overlap {cmp_['jaccard']:.0%}).")
                st.caption("Isolation Forest looks at temperature and rainfall together but has no baseline, and its flag rate is set by a setting (1%). Treat it as a cross-check, not as proof that either method is better.")

    st.subheader("Long-term trends")
    ui.explainer("Whether the yearly average (or total) has been rising or falling over the years, tested statistically.",
                 "The dashed line is the estimated change. If the 95% interval for the slope includes zero, the trend is not distinguishable from no trend.")
    tv = st.selectbox("Measurement", VARS, format_func=lab, key="tr_var")
    tst = stat_for(tv)
    ann = annual_series(df[tv], tst)
    if len(ann) < MIN_YEARS:
        ui.callout("warn", "Not enough years", f"Only {len(ann)} complete year(s) are loaded; at least {MIN_YEARS} are needed for a trend test. Load the Full record period.")
    else:
        tr = trend_test(ann.values, ann.index.values)
        tu = unit_of(tv) if tst == "mean" else "mm"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Change per decade ({tu})", f"{tr['slope_per_decade']:+.2f}")
        m2.metric("95% interval (per decade)", f"{10 * tr['slope_ci_low']:+.2f} to {10 * tr['slope_ci_high']:+.2f}")
        m3.metric("p-value", f"{tr['p_value']:.3f}")
        m4.metric("Result", {"increasing": "Increasing", "decreasing": "Decreasing"}.get(tr["direction"], "Not significant"))
        st.plotly_chart(charts.trend_chart(ann.index.astype(int), ann.values, tr["slope_per_year"], tr["intercept"], f"Yearly {'total' if tst == 'sum' else 'average'}: {lab(tv)}", tu),
                        width="stretch", key="tr_chart")
        ui.callout("good" if tr["significant"] else "info", "In plain words",
                   f"Over {tr['n_years']} complete years ({tr['first_year']} to {tr['last_year']}) the estimated change is {tr['slope_per_decade']:+.2f} {tu} per decade. "
                   + ("This is statistically significant at the 5% level." if tr["significant"] else "This is not statistically distinguishable from no change at the 5% level."))
        with st.expander("Method and limitations"):
            st.write("Annual values from complete years are tested with the Mann-Kendall test; the slope is Sen's slope with a Gilbert confidence interval; the variance is corrected for serial correlation (Hamed-Rao). "
                     f"Autocorrelation correction factor here: {tr['autocorr_factor']:.2f} (1 means none needed).")
            for c in ["A short record gives wide intervals; do not over-read the sign of a weak trend.",
                      "Reanalysis trends can partly reflect changes in the observations assimilated over time, especially for rainfall.", "A statistically significant trend is not proof of a cause."]:
                st.write("- " + c)

# ================================================================ DATA QUALITY
with t_quality:
    ui.explainer("What was checked and fixed before any number was calculated.", "Green means nothing needed fixing. Missing days are shown separately from real zero-rain days.")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows received", rep.n_rows_raw)
    q2.metric("Duplicates removed", rep.duplicate_rows_removed)
    q3.metric("Missing dates added", rep.missing_dates_added)
    q4.metric("Impossible values removed", sum(rep.impossible_values.values()))
    _q = rep.summary_table().copy()
    if "n_zero_days" in _q:
        _q["n_zero_days"] = _q["n_zero_days"].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    st.dataframe(_q.rename(columns={"variable": "Variable", "label": "Measurement", "unit": "Unit", "n_expected": "Expected days", "n_valid": "Valid",
                                                    "n_missing": "Missing", "pct_missing": "% missing", "longest_missing_run_days": "Longest gap (days)", "n_zero_days": "Zero-rain days"}),
                 width="stretch", hide_index=True)
    st.markdown("**Processing steps**")
    for step in rep.steps:
        st.write(step)
    with st.expander("Preview of the cleaned daily data"):
        _prev = df.head(200).copy()
        _prev.index = _prev.index.strftime("%Y-%m-%d")
        st.dataframe(_prev, width="stretch")
        st.caption("Empty cells are missing values, not zeros.")
