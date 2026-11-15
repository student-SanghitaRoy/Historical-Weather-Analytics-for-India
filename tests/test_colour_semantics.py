"""Colour meaning is part of the platform's correctness: reads the colours out of the figures the real pages produce.

  * differences / trends (diverging): temperature warmer = RED, cooler = BLUE; rainfall wetter = BLUE, drier = RED
  * climate map (sequential): temperature darker = hotter (red), rainfall darker = wetter (blue); low values are pale
"""
import colorsys
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from plotly.colors import sample_colorscale
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def hue_name(rgb: str) -> str:
    r, g, b = [int(float(x)) / 255 for x in re.findall(r"[\d.]+", rgb)[:3]]
    h, s, _ = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:
        return "neutral"
    if h < 0.05 or h > 0.9:
        return "RED"
    if 0.5 < h < 0.7:
        return "BLUE"
    return "other"                                   # yellow / orange / green


def colour_at(trace, value) -> str:
    m = trace["marker"]
    lo, hi = min(m["color"]), max(m["color"])
    if m.get("cmin") is not None and m.get("cmax") is not None:               # capped diverging scale
        pos = (value - m["cmin"]) / (m["cmax"] - m["cmin"])
    elif m.get("cmid") is not None:                    # diverging scale centred on cmid
        span = max(abs(lo - m["cmid"]), abs(hi - m["cmid"])) or 1
        pos = 0.5 + (value - m["cmid"]) / (2 * span)
    else:
        pos = (value - lo) / ((hi - lo) or 1)
    return sample_colorscale(m["colorscale"], [min(max(pos, 0), 1)])[0]


def coloured_trace(at, title_part):
    for el in at.get("plotly_chart"):
        fig = json.loads(el.proto.spec)
        title = json.dumps(fig["layout"].get("title", {}), ensure_ascii=False)
        if title_part in title:
            for tr in fig["data"]:
                if "colorbar" in tr.get("marker", {}):
                    return tr
    raise AssertionError(f"no coloured map with title containing {title_part!r}")


@pytest.fixture(scope="module")
def india_page():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.checkbox(key="use_synth").check().run()
    at.switch_page("views/india.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("measurement,pos,neg", [("Yearly rainfall", "BLUE", "RED"), ("Yearly mean temperature", "RED", "BLUE")])
def test_anomaly_map_colours(india_page, measurement, pos, neg):
    india_page.radio(key="ind_avar").set_value(measurement).run()
    tr = coloured_trace(india_page, "compared with")
    assert hue_name(colour_at(tr, max(tr["marker"]["color"]))) == pos
    assert hue_name(colour_at(tr, min(tr["marker"]["color"]))) == neg


@pytest.mark.parametrize("measurement,pos,neg", [("Yearly rainfall", "BLUE", "RED"), ("Average temperature", "RED", "BLUE")])
def test_trend_map_colours(india_page, measurement, pos, neg):
    india_page.radio(key="ind_tvar").set_value(measurement).run()
    tr = coloured_trace(india_page, "Trend in")
    assert hue_name(colour_at(tr, max(tr["marker"]["color"]))) == pos
    assert hue_name(colour_at(tr, min(tr["marker"]["color"]))) == neg


@pytest.mark.parametrize("metric,high", [("Average temperature (°C)", "RED"), ("Average yearly rainfall (mm)", "BLUE")])
def test_climate_map_darker_means_more(india_page, metric, high):
    india_page.radio(key="ind_metric").set_value(metric).run()
    tr = coloured_trace(india_page, metric)
    assert hue_name(colour_at(tr, max(tr["marker"]["color"]))) == high            # highest value: dark red / dark blue
    assert hue_name(colour_at(tr, min(tr["marker"]["color"]))) in ("other", "neutral")   # lowest value: pale yellow / cream, never red or blue
