"""Plotly figures. Every chart carries units in the axis titles."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from weather_platform import config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


COLORWAY = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#8c6d1f", "#444444"]   # Okabe-Ito: colour-blind safe


def _layout(fig: go.Figure, title: str, ytitle: str) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color="#0f2a4a")), yaxis_title=ytitle, xaxis_title=None, hovermode="x unified",
                      template="plotly_white", colorway=COLORWAY, font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12),
                      legend=dict(orientation="h", y=-0.22), margin=dict(l=50, r=20, t=60, b=40), height=380)
    return fig


def temperature_trend(df: pd.DataFrame, rolling_days: int = 30) -> go.Figure:
    fig = go.Figure()
    for v in [c for c in df.columns if config.VARIABLES[c]["kind"] == "temperature"]:
        lab = config.VARIABLES[v]["label"]
        fig.add_trace(go.Scatter(x=df.index, y=df[v], mode="lines", name=lab, opacity=0.35, line=dict(width=1), connectgaps=False))
        fig.add_trace(go.Scatter(x=df.index, y=df[v].rolling(rolling_days, min_periods=int(rolling_days * 0.8)).mean(),
                                 mode="lines", name=f"{lab} ({rolling_days}-day mean)", connectgaps=False))
    return _layout(fig, "Temperature trend", "Temperature (°C)")


def rainfall_chart(agg: pd.DataFrame, freq: str) -> go.Figure:
    a = agg[agg["variable"] == "precipitation_sum"]
    fig = go.Figure(go.Bar(x=a["period_start"], y=a["value"], name="Precipitation",
                           customdata=a[["n_valid", "n_expected"]], hovertemplate="%{y:.1f} mm<br>%{customdata[0]}/%{customdata[1]} valid days<extra></extra>"))
    n_inc = int((~a["complete"]).sum())
    sub = f" ({n_inc} incomplete period(s) left blank)" if n_inc else ""
    return _layout(fig, f"{freq.capitalize()} precipitation{sub}", "Precipitation (mm per day)" if freq == "daily" else f"Precipitation total (mm per {freq[:-2]})")


def year_comparison(pivot: pd.DataFrame, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    seasonal = not all(isinstance(i, (int, float)) for i in pivot.index)
    x = list(pivot.index) if seasonal else [MONTHS[int(m) - 1] for m in pivot.index]
    for yr in pivot.columns:
        fig.add_trace(go.Scatter(x=x, y=pivot[yr], mode="lines+markers", name=str(yr), connectgaps=False))
    return _layout(fig, title, unit)


def climatology_chart(clim: pd.DataFrame, title: str, unit: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=clim["month"], y=clim["mean"], error_y=dict(type="data", array=clim["std"].fillna(0)), name="Mean ± 1 sd"))
    return _layout(fig, title, unit)


def anomaly_chart(zdf: pd.DataFrame, flags: pd.Series, title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zdf.index, y=zdf["value"], mode="lines", name="Daily value", line=dict(width=1), connectgaps=False))
    fig.add_trace(go.Scatter(x=zdf.index, y=zdf["baseline_median"], mode="lines", name="Baseline median for the day of year", line=dict(dash="dot", width=1)))
    f = zdf[flags.reindex(zdf.index, fill_value=False)]
    fig.add_trace(go.Scatter(x=f.index, y=f["value"], mode="markers", name="Flagged as unusual", marker=dict(color="#d62728", size=7)))
    return _layout(fig, title, ytitle)


def rain_extreme_chart(rdf: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rdf.index, y=rdf["value"], name="Daily precipitation", marker_color="#9ecae1"))
    fig.add_trace(go.Scatter(x=rdf.index, y=rdf["threshold"], mode="lines", name="Extreme threshold (baseline 99th percentile of wet days)", line=dict(dash="dot")))
    f = rdf[rdf["flag"]]
    fig.add_trace(go.Scatter(x=f.index, y=f["value"], mode="markers", name="Flagged extreme day", marker=dict(color="#d62728", size=8)))
    return _layout(fig, title, "Precipitation (mm per day)")


def trend_chart(years, values, slope: float, intercept: float, title: str, ytitle: str) -> go.Figure:
    years = list(years)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=list(values), mode="lines+markers", name="Annual value"))
    fig.add_trace(go.Scatter(x=[years[0], years[-1]], y=[intercept + slope * years[0], intercept + slope * years[-1]], mode="lines",
                             name="Sen's slope line", line=dict(dash="dash")))
    return _layout(fig, title, ytitle)


def compare_lines(annual: pd.DataFrame, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    for col in annual.columns:
        fig.add_trace(go.Scatter(x=annual.index.astype(int), y=annual[col], mode="lines+markers", name=col))
    return _layout(fig, title, unit)


def climate_at_a_glance(temp_clim: pd.DataFrame | None, rain_clim: pd.DataFrame | None, title: str = "Climate at a glance (typical year)") -> go.Figure:
    """Monthly rainfall bars with the monthly temperature line on a second axis."""
    fig = go.Figure()
    if rain_clim is not None:
        fig.add_trace(go.Bar(x=rain_clim["month"], y=rain_clim["mean"], name="Rainfall (mm)", marker_color="#8fb8ea", yaxis="y"))
    if temp_clim is not None:
        fig.add_trace(go.Scatter(x=temp_clim["month"], y=temp_clim["mean"], name="Temperature (°C)", mode="lines+markers", line=dict(color="#e07b00", width=3), yaxis="y2"))
    fig.update_layout(title=dict(text=title, font=dict(size=15, color="#0f2a4a")), template="plotly_white", colorway=COLORWAY, height=380,
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12), hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=50, r=50, t=60, b=40), yaxis=dict(title="Rainfall (mm per month)"),
                      yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right", showgrid=False, tickmode="auto", nticks=7, tickformat=".0f"))
    return fig
