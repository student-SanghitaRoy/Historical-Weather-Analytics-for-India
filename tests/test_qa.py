"""Q&A tests. Expected values are computed INDEPENDENTLY with plain pandas, not with the module under test."""
import numpy as np
import pytest

from weather_platform.ai.llm import explain_with_llm
from weather_platform.ai.parse import parse_question
from weather_platform.ai.qa import answer_question, verification_context
from weather_platform.ai.verify import unsupported_numbers
from weather_platform.data.synthetic import make_synthetic

VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
META = {"source": "SYNTHETIC test data", "location": "Testville", "start": "2015-01-01", "end": "2023-12-31"}
CITIES = ["Guwahati", "Kolkata"]


@pytest.fixture(scope="module")
def df():
    return make_synthetic("2015-01-01", "2023-12-31", VARS, seed=7)


@pytest.fixture(scope="module")
def df_long():
    return make_synthetic("1991-01-01", "2023-12-31", ["temperature_2m_mean", "precipitation_sum"], seed=3)


def ask(q, df, variables=VARS, **kw):
    return answer_question(q, df, META, variables, city_names=CITIES, loaded_city="Guwahati", **kw)


def fig(a, name):
    return next(f["value"] for f in a.figures if f["name"] == name)


# ---------------- correct numbers (independent calculation) ----------------
def test_total_rainfall_matches_pandas(df):
    a = ask("What was the total rainfall in July 2022?", df)
    assert a.status == "answered" and a.intent == "total"
    assert fig(a, "total precipitation") == pytest.approx(round(df.loc["2022-07", "precipitation_sum"].sum(), 1))
    assert a.date_range == ["2022-07-01", "2022-07-31"] and "ERA5" in " ".join(a.warnings) and a.source


def test_average_temperature_year(df):
    a = ask("What was the average temperature in 2021?", df)
    assert fig(a, "average mean temperature") == pytest.approx(round(df.loc["2021", "temperature_2m_mean"].mean(), 2))
    assert a.chart["kind"] == "temperature"


def test_highest_temperature_uses_max_variable_and_reports_date(df):
    a = ask("What was the highest temperature in 2022?", df)
    s = df.loc["2022", "temperature_2m_max"]
    assert a.variable == "temperature_2m_max"
    assert fig(a, "highest maximum temperature") == pytest.approx(round(s.max(), 2))
    assert fig(a, "date") == str(s.idxmax().date())


def test_wet_days_in_monsoon(df):
    a = ask("How many wet days were there in the monsoon of 2022?", df)
    s = df.loc["2022-06-01":"2022-09-30", "precipitation_sum"]
    assert fig(a, "wet days") == int((s >= 1.0).sum())
    assert fig(a, "zero-rain days") == int((s == 0).sum())


def test_longest_dry_spell_independent(df):
    s = df.loc["2021", "precipitation_sum"]
    best = run = 0
    for v in s:
        run = run + 1 if (not np.isnan(v) and v < 1.0) else 0
        best = max(best, run)
    a = ask("What was the longest dry spell in 2021?", df)
    assert fig(a, "longest dry spell") == best


def test_month_without_year_aggregates_all_years(df):
    a = ask("What was the average rainfall in July?", df)
    sel = df[df.index.month == 7]["precipitation_sum"]
    assert fig(a, "average precipitation (daily total)") == pytest.approx(round(sel.mean(), 2))
    assert "all years" in a.text


def test_year_range(df):
    a = ask("What was the total rainfall from 2018 to 2020?", df)
    assert fig(a, "total precipitation") == pytest.approx(round(df.loc["2018":"2020", "precipitation_sum"].sum(), 1))


# ---------------- missing vs zero ----------------
def test_missing_question_separates_zero_rain():
    d = make_synthetic("2020-01-01", "2020-12-31", VARS, seed=1)
    d.loc["2020-03-01":"2020-03-10", "precipitation_sum"] = np.nan
    a = ask("How much data is missing for precipitation in 2020?", d)
    assert fig(a, "missing days") == 10 and "separate" in a.text
    assert fig(a, "zero-rain days") == int((d.loc["2020", "precipitation_sum"] == 0).sum())


def test_total_abstains_when_coverage_low():
    d = make_synthetic("2020-01-01", "2020-12-31", VARS, seed=1)
    d.loc["2020-07-01":"2020-07-20", "precipitation_sum"] = np.nan  # 20 of 31 days missing
    a = ask("What was the total rainfall in July 2020?", d)
    assert a.status == "abstained" and "valid" in a.text


def test_mean_with_partial_coverage_warns():
    d = make_synthetic("2020-01-01", "2020-12-31", VARS, seed=1)
    d.loc["2020-07-01":"2020-07-10", "temperature_2m_mean"] = np.nan  # 68% coverage
    a = ask("What was the average temperature in July 2020?", d)
    assert a.status == "answered" and any("caution" in w for w in a.warnings)


# ---------------- abstention ----------------
@pytest.mark.parametrize("q,frag", [
    ("Will it rain tomorrow?", "forecast"),
    ("Predict the temperature next week", "forecast"),
    ("Why was July 2022 so wet?", "causes"),
    ("Is the temperature getting warmer over the years?", "trend"),
    ("Compare cities for rainfall", "location"),
    ("What was the average temperature in Kolkata in 2021?", "Guwahati"),
    ("What was the total rainfall in 2030?", "outside"),
    ("Tell me a joke", "could not tell"),
    ("What was the average wind speed in 2021?", "not loaded"),
    ("How many wet days in 2021 for temperature?", "only defined for precipitation"),
])
def test_abstains(df, q, frag):
    a = ask(q, df)
    assert a.status == "abstained" and frag.lower() in a.text.lower(), a.text
    assert not a.figures


def test_may_the_verb_is_not_a_month(df):
    assert parse_question("What may happen to rainfall?").months == []
    assert parse_question("rainfall in May 2022").months == [5]


# ---------------- baseline ----------------
def test_baseline_question_answered_with_uncertainty(df_long):
    a = ask("Was July 2023 unusual for temperature compared to normal?", df_long, ["temperature_2m_mean", "precipitation_sum"])
    assert a.status == "answered" and a.intent == "baseline"
    names = {f["name"] for f in a.figures}
    assert {"z-score", "percentile in baseline", "interval low", "interval high", "baseline mean"} <= names
    assert "95%" in a.text and any("assumes" in w.lower() for w in a.warnings)


def test_baseline_abstains_with_short_data(df):
    a = ask("Was July 2023 unusual for rainfall compared to normal?", df, baseline_years=(2018, 2020))
    assert a.status == "abstained" and "baseline" in a.text.lower()


# ---------------- structure guarantees ----------------
def test_every_answered_has_source_period_and_verified_numbers(df):
    qs = ["What was the total rainfall in July 2022?", "What was the median temperature in 2020?", "What was the lowest temperature in 2019?",
          "What was the standard deviation of temperature in 2021?", "How many wet days in 2022?"]
    for q in qs:
        a = ask(q, df)
        assert a.status == "answered", (q, a.text)
        assert a.source and a.date_range and a.period_label and a.figures
        assert unsupported_numbers(a.text, a.figures, verification_context(a)) == [], q


# ---------------- verification + LLM guard ----------------
def test_verify_numbers_rounding_and_fabrication():
    figs = [{"name": "x", "value": 12.345, "unit": "mm"}]
    assert unsupported_numbers("It was 12.3 mm.", figs, []) == []
    assert unsupported_numbers("It was 12.35 mm.", figs, []) == []
    assert unsupported_numbers("It was 99.0 mm.", figs, []) == ["99.0"]
    assert unsupported_numbers("Rain fell in 2022.", figs, ["July 2022"]) == []


def test_llm_output_with_fabricated_number_is_rejected(df):
    a = ask("What was the total rainfall in July 2022?", df)
    total = fig(a, "total precipitation")
    good = lambda prompt: f"In July 2022 the total precipitation was {total} mm (ERA5 reanalysis)."
    bad = lambda prompt: f"In July 2022 the total precipitation was {total} mm, 40% above normal."
    boom = lambda prompt: (_ for _ in ()).throw(RuntimeError("network down"))
    t1, used1, p1 = explain_with_llm(a, good)
    assert used1 and not p1 and str(total) in t1
    t2, used2, p2 = explain_with_llm(a, bad)
    assert not used2 and "40" in p2 and t2 == a.text
    t3, used3, _ = explain_with_llm(a, boom)
    assert not used3 and t3 == a.text


def test_llm_never_called_for_abstained(df):
    called = []
    a = ask("Will it rain tomorrow?", df)
    t, used, _ = explain_with_llm(a, lambda p: called.append(p) or "x")
    assert not used and not called and t == a.text


def test_six_year_baseline_answers_but_warns_short(df):
    a = ask("Was July 2023 unusual for rainfall compared to normal?", df)  # 2015-2020 = 6 usable baseline years
    assert a.status == "answered" and any("short baseline" in w.lower() for w in a.warnings)
