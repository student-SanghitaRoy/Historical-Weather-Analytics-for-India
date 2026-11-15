"""India maps (Plotly Scattermap; the browser loads OpenStreetMap tiles, so internet is needed to view)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

CENTER = {"lat": 21.5, "lon": 81.0}
# Okabe-Ito palette: distinguishable for the common forms of colour blindness (including red-green)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#444444"]


def cluster_color_map(labels) -> dict:
    """One colour per group, identical on every chart that shows groups (keys are strings, as plotly expects)."""
    return {str(int(l)): PALETTE[int(l) % len(PALETTE)] for l in sorted({int(v) for v in labels})}


def _base(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, map=dict(style="open-street-map", center=CENTER, zoom=3.2), uirevision=title,
                      margin=dict(l=0, r=0, t=50, b=0), height=560, legend=dict(orientation="h", y=-0.02))
    return fig


def location_map(cities: list[dict], selected: str | None = None, title: str = "Supported locations") -> go.Figure:
    fig = go.Figure()
    others = [c for c in cities if c["name"] != selected]
    fig.add_trace(go.Scattermap(lat=[c["lat"] for c in others], lon=[c["lon"] for c in others], mode="markers",
                                marker=dict(size=8, color="#7f7f7f"), text=[f"{c['name']}, {c['state']}" for c in others],
                                hoverinfo="text", name="Locations"))
    sel = [c for c in cities if c["name"] == selected]
    if sel:
        fig.add_trace(go.Scattermap(lat=[sel[0]["lat"]], lon=[sel[0]["lon"]], mode="markers",
                                    marker=dict(size=16, color="#d62728"), text=[f"{sel[0]['name']}, {sel[0]['state']}"],
                                    hoverinfo="text", name="Selected"))
    return _base(fig, title)


def value_map(cities: list[dict], values: dict, title: str, unit: str, diverging: bool = False, warm_is_red: bool = False, scale: str | None = None) -> go.Figure:
    """Colour each city by a numeric value. Cities without a value are grey (never coloured as zero).
    diverging: centred on 0. warm_is_red: for temperature, positive values are red (default diverging scale shows positive as blue, right for rainfall)."""
    have = [c for c in cities if values.get(c["name"]) is not None and np.isfinite(values[c["name"]])]
    none = [c for c in cities if c not in have]
    fig = go.Figure()
    if none:
        fig.add_trace(go.Scattermap(lat=[c["lat"] for c in none], lon=[c["lon"] for c in none], mode="markers",
                                    marker=dict(size=8, color="#bdbdbd"), text=[f"{c['name']}: no data" for c in none],
                                    hoverinfo="text", name="No data"))
    if have:
        vals = [float(values[c["name"]]) for c in have]
        kw = dict(colorscale="RdBu_r" if warm_is_red else "RdBu", cmid=0) if diverging else dict(colorscale=scale or "Viridis")
        if diverging:
            # one extreme value must not wash out every other dot: cap the colour range at the 95th percentile of |value|
            lim = float(np.percentile(np.abs(vals), 95))
            beyond = [c["name"] for c, v in zip(have, vals) if abs(v) > lim]
            if lim > 0 and max(abs(v) for v in vals) > 1.5 * lim:
                kw = dict(colorscale=kw["colorscale"], cmin=-lim, cmax=lim)
                title = f"{title} (colour scale capped at ±{lim:.1f} {unit}; beyond it: {', '.join(beyond[:3])}{'...' if len(beyond) > 3 else ''})"
        fig.add_trace(go.Scattermap(lat=[c["lat"] for c in have], lon=[c["lon"] for c in have], mode="markers", marker=dict(size=17, color="#4a5b6d"),
                                    hoverinfo="skip", showlegend=False))       # dark ring under each dot so pale values stay visible on a light map
        fig.add_trace(go.Scattermap(lat=[c["lat"] for c in have], lon=[c["lon"] for c in have], mode="markers",
                                    showlegend=False, marker=dict(size=13, color=vals, colorbar=dict(title=unit), **kw),
                                    text=[f"{c['name']}, {c['state']}: {v:.1f} {unit}" for c, v in zip(have, vals)],
                                    hoverinfo="text", name="Value"))
    return _base(fig, title)


def category_map(cities: list[dict], labels: dict, title: str, names: dict | None = None) -> go.Figure:
    """Colour cities by a categorical label (for example a climate-regime cluster)."""
    fig = go.Figure()
    for lab in sorted({v for v in labels.values()}):
        cs = [c for c in cities if labels.get(c["name"]) == lab]
        nm = (names or {}).get(lab, f"Cluster {lab}")
        fig.add_trace(go.Scattermap(lat=[c["lat"] for c in cs], lon=[c["lon"] for c in cs], mode="markers",
                                    marker=dict(size=13, color=PALETTE[int(lab) % len(PALETTE)]),
                                    text=[f"{c['name']}, {c['state']} ({nm})" for c in cs], hoverinfo="text", name=nm))
    return _base(fig, title)
