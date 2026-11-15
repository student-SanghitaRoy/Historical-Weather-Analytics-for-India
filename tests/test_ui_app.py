"""Headless smoke tests of the multi-page app. All use SYNTHETIC data (no network, no downloads needed)."""
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def start(synthetic=True, timeout=240):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if synthetic:
        at.checkbox(key="use_synth").check()
        at.run()
    return at


def goto(at, page):
    at.switch_page(f"views/{page}.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def md(at) -> str:
    """All visible text: markdown blocks plus headings."""
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.subheader] + [h.value for h in at.header] + [t.value for t in at.title])


def test_home_page_explains_the_platform():
    at = start(synthetic=False)
    text = md(at)
    for must in ["Historical Weather Analytics for India", "Start here", "Why you can trust the results", "not a weather forecast"]:
        assert must in text, must


def test_location_page_autoloads_in_demo_mode_and_shows_five_tabs():
    at = goto(start(), "location")
    assert not at.error
    assert at.session_state["data"]["city"]["name"] == "Guwahati"
    assert len(at.tabs) == 5
    assert "In plain words" in md(at) and "Demo data" in md(at)


def test_full_record_supports_baselines_and_trends():
    at = goto(start(), "location")
    assert any("Change per decade" in m.label for m in at.metric)      # trend computed
    assert any("Percentile" in m.label for m in at.metric)             # baseline comparison computed


@pytest.mark.parametrize("freq", ["daily", "monthly", "seasonal"])
def test_all_rainfall_groupings(freq):
    at = goto(start(), "location")
    at.selectbox(key="freq").set_value(freq)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_invalid_dates_show_a_clear_error():
    at = goto(start(), "location")
    at.radio(key="preset").set_value("Custom dates")
    at.run()
    at.date_input(key="start").set_value(date(2020, 6, 1))
    at.date_input(key="end").set_value(date(2020, 1, 1))
    at.button(key="load").click()
    at.run()
    assert any("after end" in e.value for e in at.error)


def test_changed_selection_is_flagged_until_reloaded():
    at = goto(start(), "location")
    at.selectbox(key="state").set_value("Rajasthan")
    at.run()
    assert "Your selection has changed" in md(at)


def test_pages_that_need_data_guide_the_user_when_nothing_is_loaded():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox(key="use_synth").check()
    at.run()
    goto(at, "ask")                      # nothing loaded yet because the Location page was never opened
    assert "No location loaded yet" in md(at)


def test_ask_page_answers_refuses_and_evaluates():
    at = goto(start(), "location")
    goto(at, "ask")
    at.text_input(key="question").set_value("What was the total rainfall in July 2022?")
    at.button(key="ask").click()
    at.run()
    assert not at.exception and any("total precipitation" in s.value for s in at.success)
    at.text_input(key="question").set_value("Will it rain tomorrow?")
    at.button(key="ask").click()
    at.run()
    assert any("forecast" in w.value.lower() for w in at.warning)
    at.button(key="run_eval").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["eval_out"]["metrics"]["overall_pass_rate"] == 1.0


def test_example_button_fills_the_question_box():
    at = goto(start(), "location")
    goto(at, "ask")
    at.button(key="ex_0").click()
    at.run()
    assert "total rainfall" in at.text_input(key="question").value.lower()


def test_export_page_builds_pdf():
    at = goto(start(), "location")
    goto(at, "export")
    at.button(key="make_pdf").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_saved_configuration_restores_settings_on_location_page():
    from weather_platform.reproduce import config_to_state
    cfg = {"location": "Delhi, Delhi", "start": "2016-01-01", "end": "2019-12-31", "variables": ["precipitation_sum"], "aggregation": "seasonal",
           "wet_day_threshold_mm": 2.5, "data_source": "SYNTHETIC demo data (NOT real weather)", "baseline_years": [1995, 2015]}
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_pending_cfg_state"] = config_to_state(cfg)
    at.session_state["_auto_load"] = True
    at.run()
    goto(at, "location")
    d = at.session_state["data"]
    assert d["city"]["name"] == "Delhi" and d["variables"] == ["precipitation_sum"] and str(d["df"].index.min().date()) == "2016-01-01"
    assert at.session_state["settings"]["wet_mm"] == 2.5 and at.session_state["settings"]["baseline"] == (1995, 2015)


def test_india_pages_render_in_demo_mode():
    at = start()
    goto(at, "india")
    assert len(at.tabs) == 3 and "Key findings" in md(at)
    goto(at, "compare")
    assert any("Average temperature" in s.value for s in at.subheader)
    goto(at, "regimes")
    assert any("Groups" in m.label for m in at.metric)


def test_india_page_without_data_explains_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr("weather_platform.india.CACHE_DIR", str(tmp_path / "nothing_here"))
    at = start(synthetic=False)
    goto(at, "india")
    assert "No weather data downloaded yet" in md(at)


def test_methods_page_has_all_sections():
    at = goto(start(synthetic=False), "methods")
    assert len(at.tabs) == 5 and "Reanalysis is not a weather station" in md(at)


def test_statistics_table_is_rounded_to_two_decimals():
    at = goto(start(), "location")
    tab = next(d.value for d in at.dataframe if {"Average", "Valid days"} <= set(d.value.columns))
    for c in ["Average", "Median", "Lowest", "Highest", "Std. deviation"]:
        assert (tab[c] == tab[c].round(2)).all(), c


def test_pressing_enter_asks_the_question():
    at = goto(start(), "location")
    goto(at, "ask")
    at.text_input(key="question").set_value("Tell me a joke").run()      # no button click: Enter key behaviour
    assert any("could not tell" in w.value.lower() for w in at.warning)


def test_trend_refusal_points_to_the_correct_tab():
    at = goto(start(), "location")
    goto(at, "ask")
    at.text_input(key="question").set_value("Is the temperature getting warmer?").run()
    assert any("Unusual days and trends" in w.value for w in at.warning)


def test_trend_result_metric_is_short_enough_to_display():
    at = goto(start(), "location")
    res = [m.value for m in at.metric if m.label == "Result"]
    assert res and all(len(v) <= 15 for v in res)


def test_rainfall_trend_tab_warns_about_reanalysis_artifacts():
    at = start()
    goto(at, "india")
    at.radio(key="ind_tvar").set_value("Yearly rainfall").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Treat rainfall trends with extra caution" in md(at) and "Largest changes per decade" in md(at)


def test_regimes_explains_a_manual_k_choice():
    at = start()
    goto(at, "regimes")
    at.checkbox(key="ind_auto").uncheck().run()
    at.number_input(key="ind_k").set_value(5).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Why not the best silhouette" not in md(at)        # the hill-station explanation only applies to the automatic k = 2 case
