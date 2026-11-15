import hashlib
import json

import pandas as pd
import streamlit as st

from weather_platform import config, state, ui
from weather_platform.aggregate import aggregate, stat_for
from weather_platform.export import aggregated_csv_bytes, daily_csv_bytes
from weather_platform.report.pdf import analysis_config, build_report
from weather_platform.reproduce import config_to_state, results_fingerprint, validate_config
from weather_platform.stats.baseline import compare_to_baseline
from weather_platform.trend import MIN_YEARS, annual_series, trend_test

ui.page_header("Reports and export", "Take your results with you: a PDF report, the data as CSV, and a settings file that lets anyone repeat the analysis.")

# ---- re-run from a saved configuration (works even before anything is loaded)
with st.container(border=True):
    st.markdown("**Re-run a saved analysis**")
    st.caption("Upload a settings file (JSON) saved earlier. The platform restores every setting, loads the data, and checks whether the results are identical.")
    cfg_file = st.file_uploader("Saved analysis configuration (JSON)", type="json", key="cfg_file")
    if cfg_file is not None:
        raw = cfg_file.getvalue()
        sig = hashlib.md5(raw).hexdigest()
        if st.session_state.get("_cfg_sig") != sig:
            try:
                cfg_obj = json.loads(raw)
            except ValueError:
                cfg_obj = None
                st.error("That file is not valid JSON.")
            if cfg_obj is not None:
                problems = validate_config(cfg_obj)
                if problems:
                    st.error("Cannot use this file: " + " | ".join(problems))
                else:
                    st.session_state["_pending_cfg_state"] = config_to_state(cfg_obj)
                    st.session_state["_saved_fp"] = cfg_obj.get("results_fingerprint")
                    st.session_state["_cfg_sig"] = sig
                    st.session_state["_auto_load"] = True
                    st.switch_page("views/location.py")

D = state.require_loaded()
df, rep, meta, VARS = D["df"], D["rep"], D["meta"], D["variables"]
S = state.settings()
freq, wet_mm, b = S["freq"], S["wet_mm"], state.baseline()
agg = aggregate(df, VARS, freq, state.MIN_VALID)
ui.chips([(f"Exporting {meta['location']}", ""), (f"{meta['start']} to {meta['end']}", "")])

st.subheader("PDF report")
ui.explainer("A report for this location and period: scope, data source, quality checks, statistics, charts, comparison with the normal, methods, limitations and the settings used.",
             "Every page states that it is a historical analysis, not a forecast. Generate it, then download.")
if st.button("Generate PDF report", key="make_pdf", type="primary"):
    ly = df.index.max().year if df.index.max() >= pd.Timestamp(df.index.max().year, 12, 25) else df.index.max().year - 1
    results, note = [], None
    for v in [x for x in VARS if config.VARIABLES[x]["kind"] in ("temperature", "precipitation")]:
        r = compare_to_baseline(df[v], f"{ly}-01-01", f"{ly}-12-31", b, stat=stat_for(v), relative=stat_for(v) == "sum", min_valid_frac=state.MIN_VALID)
        if r["status"] == "ok" and "z_score" in r:
            r.update(label=config.VARIABLES[v]["label"], unit=config.VARIABLES[v]["unit"] if stat_for(v) == "mean" else "mm")
            results.append(r)
    if not results:
        note = (f"No baseline comparison: the loaded data do not contain enough complete baseline years ({b[0]}-{b[1]}) for the comparison year {ly}. "
                "Load the Full record period to include one.")
    trends = []
    for v in [x for x in VARS if config.VARIABLES[x]["kind"] in ("temperature", "precipitation")]:
        ann = annual_series(df[v], stat_for(v))
        if len(ann) >= MIN_YEARS:
            tr = trend_test(ann.values, ann.index.values)
            tr.update(label=config.VARIABLES[v]["label"], unit=config.VARIABLES[v]["unit"] if stat_for(v) == "mean" else "mm")
            trends.append(tr)
    with st.spinner("Building the report..."):
        st.session_state["pdf_bytes"] = build_report(df, rep, meta, VARS, freq, wet_mm, results, note, state.MIN_VALID, b, trend_results=trends)
if st.session_state.get("pdf_bytes"):
    st.download_button("Download PDF report", st.session_state["pdf_bytes"], file_name="weather_report.pdf", mime="application/pdf", key="dl_pdf")

st.subheader("Data as CSV")
ui.explainer("The cleaned data with a header that records the source, location, period and units.", "Empty cells are missing values, never zeros.")
d1, d2 = st.columns(2)
d1.download_button("Download daily data (CSV)", daily_csv_bytes(df, meta), file_name="weather_daily.csv", mime="text/csv", key="dl_daily", width="stretch")
d2.download_button(f"Download {freq} data (CSV)", aggregated_csv_bytes(agg, meta, freq, state.MIN_VALID), file_name=f"weather_{freq}.csv", mime="text/csv", key="dl_agg", width="stretch")

st.subheader("Analysis settings (to repeat this analysis)")
ui.explainer("A small file holding the location, period, measurements, thresholds, baseline, software versions and a fingerprint of the key results.",
             "Upload it above (or on another computer) to re-run the analysis; the platform confirms whether the results are identical.")
cfg = analysis_config(meta, VARS, freq, wet_mm, state.MIN_VALID, b, fingerprint=results_fingerprint(df, VARS))
st.download_button("Download analysis configuration (JSON)", json.dumps(cfg, indent=2), file_name="analysis_config.json", mime="application/json", key="dl_cfg")
