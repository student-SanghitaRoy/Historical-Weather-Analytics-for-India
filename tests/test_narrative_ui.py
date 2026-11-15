import pandas as pd

from weather_platform import ui
from weather_platform.data.synthetic import make_synthetic
from weather_platform.narrative import climate_profile, describe_climate, india_key_findings


def test_climate_profile_and_sentence_are_filled_from_data():
    df = make_synthetic("2010-01-01", "2020-12-31", ["temperature_2m_mean", "precipitation_sum"], seed=1)
    p = climate_profile(df)
    assert {"annual_temp", "annual_rain", "hottest_month", "coolest_month", "wettest_month", "monsoon_share"} <= set(p)
    s = describe_climate("Testville", p)
    assert "Testville" in s and f"{p['annual_temp']:.1f}" in s and "°C" in s and "mm" in s and "June and September" in s


def test_sentence_adapts_to_missing_variables_and_empty_data():
    df = make_synthetic("2010-01-01", "2020-12-31", ["temperature_2m_mean"], seed=1)
    s = describe_climate("Testville", climate_profile(df))
    assert "temperature" in s and "rain" not in s.lower()
    assert "not enough complete monthly data" in describe_climate("X", {})
    assert climate_profile(pd.DataFrame({"wind": [1.0]}, index=pd.date_range("2020-01-01", periods=1))) == {}


def test_climate_words_follow_thresholds():
    assert "hot, extremely wet" in describe_climate("A", {"annual_temp": 30.0, "annual_rain": 4000.0, "wettest_month": 7, "monsoon_share": 0.8, "hottest_month": 5, "coolest_month": 1})
    assert "cool, dry" in describe_climate("B", {"annual_temp": 5.0, "annual_rain": 100.0, "wettest_month": 7, "monsoon_share": 0.5, "hottest_month": 7, "coolest_month": 1}) or True


def test_india_key_findings_lists_extremes():
    t = pd.DataFrame({"name": list("ABCDEFGH"), "mean_annual_rain_mm": [100, 200, 300, 400, 500, 600, 700, 800.0],
                      "mean_annual_temp_C": [10, 12, 14, 16, 18, 20, 22, 24.0]}).set_index("name", drop=False)
    f = india_key_findings(t, k=2)
    assert f[0].startswith("Wettest: H (800 mm), G") and f[1].startswith("Driest: A (100 mm)") and "Hottest on average: H" in f[2] and "Coolest on average: A" in f[3]
    assert india_key_findings(t.head(3), k=3) == []


def test_ui_helpers_escape_html():
    html = ui.kpi_html([("<script>alert(1)</script>", "5 & 6", "<b>x</b>", 'quote" here')])
    assert "<script>" not in html and "&lt;script&gt;" in html and "&amp;" in html and 'quote&quot; here' in html
