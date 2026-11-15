"""End-to-end test of the REAL-DATA code path (not demo mode): a cache is built with the same downloader the user runs,
then every page is driven from it. A partial cache (24 of 71 locations, with deliberate gaps) also tests graceful handling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from weather_platform import india
from weather_platform.data.cities import get_city, load_cities

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TODAY = pd.Timestamp("2026-10-30")
N = 24


class SynthSession:
    """Stands in for the Open-Meteo API: serves synthetic daily data for whichever coordinates are requested."""

    def __init__(self, cities):
        self.data = india.synthetic_datasets(cities, india.PREFETCH_START, india.PREFETCH_END, seed=5)
        self.by_coord = {(round(c["lat"], 4), round(c["lon"], 4)): c["name"] for c in cities}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        name = self.by_coord[(round(params["latitude"], 4), round(params["longitude"], 4))]
        df = self.data[name].loc[params["start_date"]:params["end_date"]]
        if name == "Delhi":                      # a real-looking gap: a month of missing rainfall in 2005
            df = df.copy()
            df.loc["2005-03-01":"2005-03-31", "precipitation_sum"] = np.nan
        daily = {"time": [d.strftime("%Y-%m-%d") for d in df.index]}
        for v in params["daily"].split(","):
            daily[v] = [None if pd.isna(x) else float(x) for x in df[v]]
        payload = {"daily_units": {"temperature_2m_mean": "°C", "precipitation_sum": "mm"}, "daily": daily}
        return type("R", (), {"status_code": 200, "json": lambda s: payload, "text": ""})()


@pytest.fixture(scope="module")
def real_cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("india_cache")
    cities = load_cities()[:N - 1] + [get_city("Delhi")]
    names = {c["name"] for c in cities}
    assert "Delhi" in names and "Guwahati" in names
    res = india.prefetch(cities, str(d), delay=0, session=SynthSession(cities), today=TODAY, sleep=lambda *_: None, log=lambda *_: None)
    assert len(res["fetched"]) == N and not res["failed"]
    mp = pytest.MonkeyPatch()
    mp.setattr("weather_platform.india.CACHE_DIR", str(d))
    yield d
    mp.undo()


def md(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.subheader] + [h.value for h in at.header])


def start():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def goto(at, page):
    at.switch_page(f"views/{page}.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_reports_how_much_is_downloaded(real_cache):
    assert f"{N} of 71" in md(start())


def test_location_page_loads_from_the_downloaded_dataset_without_the_network(real_cache):
    at = goto(start(), "location")
    d = at.session_state["data"]
    assert d["meta"]["origin"] == "downloaded dataset (offline)" and d["city"]["name"] == "Guwahati"
    assert len(d["df"]) == len(pd.date_range(india.PREFETCH_START, india.PREFETCH_END)) and "downloaded dataset" in md(at)
    assert not at.error and len(at.tabs) == 5


def test_missing_data_is_reported_and_never_zero_in_real_mode(real_cache):
    at = goto(start(), "location")
    at.selectbox(key="state").set_value("Delhi")
    at.run()
    at.button(key="load").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    d = at.session_state["data"]
    assert d["city"]["name"] == "Delhi"
    st = d["rep"].per_variable["precipitation_sum"]
    assert st["n_missing"] == 31 and st["pct_missing"] > 0 and d["df"].loc["2005-03", "precipitation_sum"].isna().all()
    assert d["rep"].per_variable["temperature_2m_mean"]["n_missing"] == 0


def test_period_presets_and_custom_dates_use_the_cache(real_cache):
    at = goto(start(), "location")
    at.radio(key="preset").set_value("Last 5 years")
    at.run()
    at.button(key="load").click()
    at.run()
    d = at.session_state["data"]
    assert d["meta"]["origin"] == "downloaded dataset (offline)" and str(d["df"].index.min().date()) == "2021-01-01"
    assert not at.exception, [e.value for e in at.exception]


def test_variables_outside_the_cache_need_the_network_and_fail_gracefully(real_cache, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no internet")
    monkeypatch.setattr("weather_platform.state.fetch_daily", boom)
    at = goto(start(), "location")
    at.multiselect(key="vars").set_value(["temperature_2m_mean", "temperature_2m_max"])
    at.run()
    at.button(key="load").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Could not load data" in e.value for e in at.error)


def test_all_pages_work_with_a_partial_download(real_cache):
    at = start()
    goto(at, "location")
    goto(at, "india")
    assert "not downloaded yet" in " ".join(e.label for e in at.expander) and len(at.tabs) == 3
    goto(at, "compare")
    assert any("Average temperature" in s.value for s in at.subheader)
    goto(at, "regimes")
    assert any("Groups" in m.label for m in at.metric)
    goto(at, "methods")
    assert len(at.tabs) == 5


def test_ask_evaluate_and_export_work_in_real_mode(real_cache):
    at = goto(start(), "location")
    goto(at, "ask")
    at.text_input(key="question").set_value("What was the total rainfall in July 2022?").run()
    assert any("total precipitation" in s.value for s in at.success)
    at.button(key="run_eval").click()
    at.run()
    assert at.session_state["eval_out"]["metrics"]["overall_pass_rate"] == 1.0
    goto(at, "export")
    at.button(key="make_pdf").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_evaluation_script_runs_on_the_cache(real_cache, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_qa", Path(APP).parent / "scripts" / "evaluate_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--cities", "Guwahati,Delhi", "--out", str(tmp_path / "ev")]) == 0
    assert (tmp_path / "ev" / "qa_report.md").exists() and (tmp_path / "ev" / "qa_results.csv").exists()
