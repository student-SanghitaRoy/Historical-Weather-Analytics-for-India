"""PDF report: location, period, units, source attribution, data quality, statistics, charts, methodology, limitations.

Charts are drawn with matplotlib (no browser needed). Text sticks to characters in the standard PDF fonts.
Every page carries a footer stating that this is historical analysis, not a forecast.
"""
from __future__ import annotations

import io
import json
import platform
import sys
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether  # noqa: E402

from weather_platform import config  # noqa: E402
from weather_platform.aggregate import aggregate, monthly_climatology, pivot_by_year  # noqa: E402
from weather_platform.data.fetch import make_cache_key  # noqa: E402
from weather_platform.stats.descriptive import describe_all  # noqa: E402
from weather_platform.stats.rainfall import rainfall_summary  # noqa: E402

NOT_A_FORECAST = ("This report is a HISTORICAL analysis of reanalysis data. It is NOT a weather forecast, not an official weather "
                  "record, and must not be used for safety-critical decisions. For warnings and forecasts use the official national "
                  "weather service (in India, the India Meteorological Department).")

LIMITATIONS = [
    "Source data are ERA5 reanalysis values on a grid of roughly 25 km. They describe the grid cell around the stated coordinates, not a "
    "weather station, and can differ from observations, especially for rainfall, in hills and near coasts.",
    "Reanalysis rainfall tends to produce many light-rain days, so wet-day counts depend strongly on the chosen threshold.",
    "Missing or invalid values are left missing (never replaced by zero). Totals over periods with missing days are lower bounds; "
    "periods with too much missing data are left blank.",
    "Baseline comparisons use one value per year and assume roughly independent, approximately normal baseline years. Rainfall totals "
    "are skewed, so the percentile rank is more reliable than the z-score for rainfall.",
    "A difference from the baseline or a single unusual year is not evidence of long-term climate change. Trend claims need dedicated "
    "tests and longer records.",
    "Results depend on analyst choices (baseline years, wet-day threshold, completeness rule). They are stated here so the analysis can be repeated.",
]


# ------------------------------------------------------------------------------------------------ helpers
def analysis_config(meta: dict, variables: list[str], freq: str, wet_mm: float, min_valid_frac: float,
                    baseline_years=None, period=None, fingerprint: str | None = None) -> dict:
    """Everything needed to re-run this analysis (also saved as JSON for reproducibility)."""
    return {
        "location": meta.get("location"), "latitude": meta.get("latitude"), "longitude": meta.get("longitude"),
        "start": meta.get("start"), "end": meta.get("end"), "variables": variables, "aggregation": freq,
        "wet_day_threshold_mm": wet_mm, "min_valid_fraction": min_valid_frac,
        "baseline_years": list(baseline_years) if baseline_years else None, "comparison_period": list(period) if period else None,
        "data_source": meta.get("source"), "fetched_at_utc": meta.get("fetched_at_utc"),
        "cache_key": make_cache_key(meta["latitude"], meta["longitude"], pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"]), variables)
        if meta.get("latitude") is not None and meta.get("start") else None,
        "schema_version": 1, "results_fingerprint": fingerprint,
        "software": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.system()},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _img(buf: io.BytesIO, width_cm: float = 17.0) -> Image:
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def _chart_temperature(df, variables):
    cols = [v for v in variables if config.VARIABLES[v]["kind"] == "temperature"]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for v in cols:
        ax.plot(df.index, df[v], lw=0.5, alpha=0.35)
        ax.plot(df.index, df[v].rolling(30, min_periods=24).mean(), lw=1.4, label=f"{config.VARIABLES[v]['label']} (30-day mean)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature (thin lines: daily values; gaps = missing data)", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    return _png(fig)


def _chart_rainfall(df):
    a = aggregate(df, ["precipitation_sum"], "monthly")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ok = a[a["complete"]]
    ax.bar(ok["period_start"], ok["value"], width=25, color="#1f77b4")
    n_inc = int((~a["complete"]).sum())
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_title("Monthly precipitation" + (f" ({n_inc} incomplete month(s) left blank, not zero)" if n_inc else ""), fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    return _png(fig)


def _chart_climatology(df, variables):
    agg = aggregate(df, [v for v in variables if v in ("temperature_2m_mean", "precipitation_sum")], "monthly")
    panels = [v for v in ("temperature_2m_mean", "precipitation_sum") if v in variables]
    if not panels:
        return None, {}
    fig, axes = plt.subplots(1, len(panels), figsize=(9, 3.0))
    axes = np.atleast_1d(axes)
    clims = {}
    for ax, v in zip(axes, panels):
        c = monthly_climatology(agg, v)
        clims[v] = c
        ax.bar(c["month"], c["mean"], yerr=c["std"].fillna(0), color="#2ca02c" if v != "precipitation_sum" else "#1f77b4", capsize=2)
        word = "total" if v == "precipitation_sum" else "mean"
        ax.set_title(f"Monthly {word} (average over years, bars = 1 sd)", fontsize=9)
        ax.set_ylabel(f"{config.VARIABLES[v]['label']} ({config.VARIABLES[v]['unit']})", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _png(fig), clims


def _chart_years(df):
    a = aggregate(df, ["precipitation_sum"], "monthly")
    p = pivot_by_year(a, "precipitation_sum")
    if p.shape[1] < 2 or p.shape[1] > 12:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.2))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for yr in p.columns:
        ax.plot([months[int(m) - 1] for m in p.index], p[yr], marker="o", ms=3, lw=1, label=str(yr))
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_title("Year-to-year comparison of monthly precipitation (gaps = incomplete months)", fontsize=10)
    ax.legend(fontsize=6, ncol=6)
    ax.grid(alpha=0.3)
    return _png(fig)


def _table(data, col_widths=None, header=True, font=8):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _f(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{nd}f}"


# ------------------------------------------------------------------------------------------------ main
def build_report(df: pd.DataFrame, rep, meta: dict, variables: list[str], freq: str = "monthly", wet_mm: float = config.WET_DAY_MM,
                 baseline_results: list | None = None, baseline_note: str | None = None, min_valid_frac: float = 0.9,
                 baseline_years=None, trend_results: list | None = None) -> bytes:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1f3864"), keepWithNext=1)
    warn = ParagraphStyle("warn", parent=body, backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor("#d9822b"),
                          borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=8)
    mono = ParagraphStyle("mono", parent=small, fontName="Courier", fontSize=6.5, leading=8)
    P = lambda text, st=body: Paragraph(text, st)  # noqa: E731

    story = []
    loc = escape(str(meta.get("location", "Unknown location")))
    story += [P("Historical Weather Analysis Report", h1), P(f"<b>{loc}</b>", ParagraphStyle("t", parent=body, fontSize=12)),
              P(NOT_A_FORECAST, warn)]

    synthetic = "SYNTHETIC" in str(meta.get("source", ""))
    if synthetic:
        story.append(P("<b>WARNING: this report was produced from SYNTHETIC demonstration data, not real weather.</b>", warn))

    from weather_platform.reproduce import results_fingerprint
    cfg = analysis_config(meta, variables, freq, wet_mm, min_valid_frac, baseline_years, fingerprint=results_fingerprint(df, variables))
    units_txt = "; ".join(f"{config.VARIABLES[v]['label']}: {config.VARIABLES[v]['unit']}" for v in variables)
    info = [["Item", "Value"],
            ["Location", loc], ["Coordinates", f"{meta.get('latitude')}, {meta.get('longitude')} (grid cell of about 25 km)"],
            ["Period analysed", f"{meta.get('start')} to {meta.get('end')} ({rep.n_expected_days} days)"],
            ["Variables and units", escape(units_txt)], ["Aggregation shown", freq],
            ["Wet-day threshold", f"{wet_mm} mm of precipitation per day"],
            ["Report generated (UTC)", cfg["generated_utc"]]]
    story += [P("1. Scope", h2), _table([[P(escape(str(a)), small), P(str(b), small)] for a, b in info], [4.2 * cm, 12.8 * cm])]

    story += [P("2. Data source and attribution", h2),
              P(f"<b>Source:</b> {escape(str(meta.get('source', config.DATA_SOURCE['name'])))}<br/>"
                f"<b>URL:</b> {escape(str(meta.get('source_url', config.DATA_SOURCE['url'])))}<br/>"
                f"<b>Data type:</b> {escape(str(meta.get('data_type', config.DATA_SOURCE['type'])))}<br/>"
                f"<b>Retrieved (UTC):</b> {escape(str(meta.get('fetched_at_utc', 'n/a')))}<br/>"
                f"<b>Attribution:</b> weather data by Open-Meteo.com (CC BY 4.0), based on ERA5 reanalysis produced by Copernicus Climate "
                f"Change Service / ECMWF. Check the current terms of use before publishing or commercial use.")]

    # data quality
    story.append(P("3. Data quality and processing", h2))
    q = [["Variable", "Expected days", "Valid", "Missing", "% missing", "Longest gap (days)"]]
    for v, st in rep.per_variable.items():
        q.append([st["label"], st["n_expected"], st["n_valid"], st["n_missing"], f"{st['pct_missing']}%", st["longest_missing_run_days"]])
    story.append(_table(q, [5 * cm, 2.6 * cm, 2 * cm, 2 * cm, 2.2 * cm, 3.2 * cm]))
    extra = [f"Duplicate-date rows removed: {rep.duplicate_rows_removed}", f"Missing dates added as missing: {rep.missing_dates_added}",
             f"Impossible values set to missing: {sum(rep.impossible_values.values())}",
             f"Unit validation: {'passed' if rep.units_ok else 'FAILED - ' + escape('; '.join(rep.unit_issues))}"]
    if "precipitation_sum" in rep.per_variable:
        extra.append(f"Days with a real measurement of zero rainfall: {rep.per_variable['precipitation_sum'].get('n_zero_days', 0)} "
                     "(counted separately from missing days)")
    story += [Spacer(1, 4), P("<br/>".join(extra), small), Spacer(1, 4), P("<b>Processing steps</b>", small)]
    story += [P(escape(s), small) for s in rep.steps]

    # statistics
    story.append(P("4. Summary statistics", h2))
    d = describe_all(df, variables)
    rows = [["Variable", "Unit", "Mean", "Median", "Min", "Max", "Std dev", "Valid days"]]
    for v in variables:
        s = d[v]
        rows.append([s["label"], s["unit"], _f(s["mean"]), _f(s["median"]), _f(s["min"]), _f(s["max"]), _f(s["std"]), s["n_valid"]])
    story.append(_table(rows, [4.6 * cm, 1.4 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2 * cm]))
    story.append(P("Standard deviation is the sample standard deviation of valid daily values.", small))
    if "precipitation_sum" in variables:
        r = rainfall_summary(df["precipitation_sum"], wet_mm)
        ds = r["longest_dry_spell"]
        rr = [["Rainfall metric", "Value"], ["Total precipitation (mm)", _f(r["total_mm"], 1) + (" (lower bound: days missing)" if r["is_lower_bound"] else "")],
              [f"Wet days (>= {wet_mm} mm)", r["wet_days"]], ["Dry days", r["dry_days"]], ["Zero-rain days", r["zero_rain_days"]],
              ["Missing days (not counted)", r["missing_days"]],
              ["Longest dry spell", f"{ds['length_days']} days" + (f" ({ds['start']} to {ds['end']})" if ds["start"] else "")]]
        story += [Spacer(1, 6), _table(rr, [8 * cm, 9 * cm])]

    # charts
    story.append(PageBreak())
    story.append(P("5. Charts", h2))
    for fn in (lambda: _chart_temperature(df, variables), (lambda: _chart_rainfall(df)) if "precipitation_sum" in variables else (lambda: None)):
        b = fn()
        if b:
            story += [_img(b), Spacer(1, 4)]
    cb, clims = _chart_climatology(df, variables)
    if cb:
        story += [_img(cb), Spacer(1, 4)]
    yb = _chart_years(df) if "precipitation_sum" in variables else None
    if yb:
        story.append(_img(yb))
    story.append(P("Monthly values use only complete months (at least 90% valid days). Blank periods are missing data, not zero.", small))

    if clims:
        story.append(P("6. Monthly averages across years", h2))
        for v, c in clims.items():
            word = "total" if v == "precipitation_sum" else "mean"
            rows = [["Month", f"Average {word} ({config.VARIABLES[v]['unit']})", "Std dev", "Min", "Max", "Years"]]
            for _, rw in c.iterrows():
                rows.append([rw["month"], _f(rw["mean"], 1), _f(rw["std"], 1), _f(rw["min"], 1), _f(rw["max"], 1), _f(rw["n_years"], 0)])
            story.append(KeepTogether([P(f"<b>{escape(config.VARIABLES[v]['label'])}</b>", small), _table(rows, [2 * cm, 5 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2 * cm], font=7)]))

    # baseline
    story.append(P("7. Comparison with a baseline", h2))
    if baseline_results:
        for res in baseline_results:
            u = res.get("unit", "")
            story.append(P(f"<b>{escape(res.get('label', ''))}</b>: {escape(res['period'][0])} to {escape(res['period'][1])} versus baseline years "
                           f"{escape(str(res['baseline_years_requested'][0]))}-{escape(str(res['baseline_years_requested'][1]))} "
                           f"({res['n_baseline_years']} usable years).", body))
            rows = [["Measure", "Value"], ["Selected period value", f"{_f(res['selected_mean'])} {u}"], ["Baseline mean", f"{_f(res['baseline_mean'])} {u}"],
                    ["Baseline std dev", _f(res["baseline_sd"])], ["Difference", f"{res['difference']:+.2f} {u}"]]
            if res.get("percent_difference") is not None:
                rows.append(["Percent difference", f"{res['percent_difference']:+.1f}%"])
            if "z_score" in res:
                lo, hi = res["difference_interval"]
                rows += [["z-score", _f(res["z_score"])], ["Percentile within baseline years", f"{res['percentile_in_baseline']:.0f}"],
                         ["95% range of normal year-to-year variation", f"{lo:+.2f} to {hi:+.2f} {u}"],
                         ["Outside that range?", "yes" if res["outside_normal_variability"] else "no"]]
            else:
                rows += [["95% CI of difference (Welch)", f"{res['difference_ci'][0]:+.2f} to {res['difference_ci'][1]:+.2f} {u}"],
                         ["Bootstrap 95% CI", f"{res['bootstrap_ci'][0]:+.2f} to {res['bootstrap_ci'][1]:+.2f} {u}"], ["p-value", _f(res["p_value"], 3)]]
            story += [_table([[P(escape(str(a)), small), P(escape(str(b)), small)] for a, b in rows], [8 * cm, 9 * cm]), Spacer(1, 6)]
        story.append(P("Assumptions: " + " ".join(escape(c) for c in baseline_results[0].get("caveats", [])), small))
    else:
        story.append(P(escape(baseline_note or "No baseline comparison is included in this report."), body))

    # long-term trend
    story.append(P("8. Long-term trend", h2))
    if trend_results:
        rows = [["Measurement", "Change per decade", "95% interval (per decade)", "p-value", "Years", "Result"]]
        for r in trend_results:
            u = r.get("unit", "")
            rows.append([r["label"], f"{r['slope_per_decade']:+.2f} {u}", f"{10 * r['slope_ci_low']:+.2f} to {10 * r['slope_ci_high']:+.2f}", f"{r['p_value']:.3f}",
                         f"{r['first_year']}-{r['last_year']} ({r['n_years']})", "significant" if r["significant"] else "not significant"])
        story.append(_table([[P(escape(str(c)), small) for c in row] for row in rows], [3.6 * cm, 2.8 * cm, 3.6 * cm, 1.6 * cm, 2.8 * cm, 2.6 * cm]))
        story.append(P("Annual values (complete years only) tested with the Mann-Kendall test; Sen's slope with a Gilbert 95% interval; variance corrected for serial correlation "
                       "(Hamed-Rao). Reanalysis trends can partly reflect changes in the observations assimilated over time, especially for rainfall, and a significant trend is not proof of a cause.", small))
    else:
        story.append(P("No trend test is included: fewer than 10 complete years are available for the loaded period. Load the full record to include one.", body))

    # methodology
    story.append(P("9. Methodology", h2))
    meth = [
        "<b>Data cleaning.</b> Duplicate dates are removed (first kept); missing calendar days are added as missing; values outside physical "
        "bounds and inconsistent temperature triples (min, mean, max) are set to missing; units are validated against the expected unit.",
        "<b>Missing versus zero.</b> Missing values are never filled with zero. A day with 0.0 mm rainfall is a valid measurement; a missing day is not counted as wet, dry or zero.",
        f"<b>Wet day and dry spell.</b> A wet day has at least {wet_mm} mm of precipitation. A dry spell is a run of consecutive valid days below that threshold; a missing day ends a spell.",
        "<b>Aggregation.</b> Temperature is averaged and precipitation is summed. A month or season is reported only if at least 90% of its calendar days are valid. "
        "Seasons are meteorological: DJF, MAM, JJA, SON (December belongs to the next year's DJF).",
        "<b>Baseline comparison.</b> For each year the same calendar window is reduced to one number (mean or sum); years with under 90% valid days are excluded. "
        "A selected year that lies in the baseline is removed from it. One selected year is compared with the baseline distribution (z-score, percentile, t-based "
        "95% range using sd x sqrt(1 + 1/n)); several selected years use Welch's t-test with Cohen's d and a bootstrap interval.",
        "<b>Verification.</b> The statistics code is covered by automated tests whose expected values are computed independently.",
    ]
    story += [P(m, body) for m in meth]

    story.append(P("10. Limitations", h2))
    story += [P(f"{i}. {escape(t)}", body) for i, t in enumerate(LIMITATIONS, 1)]

    story.append(P("11. Reproducibility", h2))
    story.append(KeepTogether([P("Analysis configuration (save this to repeat the analysis):", small),
                               P(escape(json.dumps(cfg, indent=1)).replace("\n", "<br/>").replace(" ", "&nbsp;"), mono)]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.8 * cm, 1.1 * cm, "Historical analysis of ERA5 reanalysis data. Not a weather forecast.")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                      title=f"Historical Weather Analysis - {meta.get('location', '')}", author="Historical Weather Analytics").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
