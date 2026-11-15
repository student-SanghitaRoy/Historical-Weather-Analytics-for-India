import numpy as np
import pytest

from weather_platform.ai.evaluate import build_eval_set, evaluate, make_answer_fn, report_markdown
from weather_platform.ai.qa import Answer
from weather_platform.data.synthetic import make_synthetic

VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
META = {"source": "SYNTHETIC test data", "location": "Testville", "start": "1991-01-01", "end": "2023-12-31"}
CITIES = ["Guwahati", "Kolkata"]


@pytest.fixture(scope="module")
def setup():
    df = make_synthetic("1991-01-01", "2023-12-31", VARS, seed=11)
    items, gappy = build_eval_set(df, VARS, "Guwahati", "Kolkata")
    return df, items, gappy


def real_fn():
    return make_answer_fn(META, VARS, "Guwahati", CITIES)


def test_eval_set_is_broad(setup):
    df, items, gappy = setup
    cats = {i.category for i in items}
    assert len(items) >= 40 and "total rainfall" in cats and "baseline comparison" in cats and "insufficient data" in cats
    assert sum(i.expect_status == "abstained" for i in items) >= 15
    assert gappy.loc["1991-07-01":"1991-07-20", "precipitation_sum"].isna().all() and df.loc["1991-07-01", "precipitation_sum"] == df.loc["1991-07-01", "precipitation_sum"]


def test_system_scores_perfectly_against_independent_key(setup):
    df, items, gappy = setup
    out = evaluate(real_fn(), items, df, gappy)
    bad = out["results"][~out["results"]["passed"]]
    assert bad.empty, bad[["id", "question", "detail", "answer_text"]].to_string()
    m = out["metrics"]
    assert m["numerical_accuracy"] == 1.0 and m["correct_abstention_rate"] == 1.0 and m["unsupported_claim_rate"] == 0.0 and m["source_consistency"] == 1.0


def test_evaluator_catches_numerical_errors(setup):
    df, items, gappy = setup
    real = real_fn()

    def sloppy(q, d):
        a = real(q, d)
        for f in a.figures:
            if isinstance(f["value"], (int, float)) and f["name"] not in ("threshold",):
                f["value"] = f["value"] * 1.1 + 0.5
        return a
    out = evaluate(sloppy, items, df, gappy)
    assert out["metrics"]["numerical_accuracy"] < 0.2


def test_evaluator_catches_hallucination_on_must_refuse(setup):
    df, items, gappy = setup

    def reckless(q, d):
        return Answer("answered", q, "It will be 31.5 degrees tomorrow.", figures=[{"name": "temperature", "value": 31.5, "unit": "C"}],
                      period_label="tomorrow", date_range=["2030-01-01", "2030-01-01"], source="made up")
    out = evaluate(reckless, items, df, gappy)
    m = out["metrics"]
    assert m["false_answer_rate"] == 1.0 and m["correct_abstention_rate"] == 0.0 and m["overall_pass_rate"] < 0.1


def test_evaluator_catches_over_abstention_and_bad_source(setup):
    df, items, gappy = setup
    never = lambda q, d: Answer("abstained", q, "no")  # noqa: E731
    m = evaluate(never, items, df, gappy)["metrics"]
    assert m["answerable_answered_rate"] == 0.0 and m["correct_abstention_rate"] == 1.0
    real = real_fn()

    def wrong_dates(q, d):
        a = real(q, d)
        if a.status == "answered":
            a.date_range = ["1900-01-01", "1900-12-31"]
        return a
    m2 = evaluate(wrong_dates, items, df, gappy)["metrics"]
    assert m2["source_consistency"] < 0.1   # only the all-years question has no fixed date range to check


def test_evaluator_catches_unsupported_numbers_in_text(setup):
    df, items, gappy = setup
    real = real_fn()

    def padded(q, d):
        a = real(q, d)
        if a.status == "answered":
            a.text += " This is 37% higher than last decade."
        return a
    assert evaluate(padded, items, df, gappy)["metrics"]["unsupported_claim_rate"] > 0.4


def test_short_data_set_expects_abstention_for_baseline():
    df = make_synthetic("2018-01-01", "2023-12-31", VARS, seed=2)
    items, gappy = build_eval_set(df, VARS, "Guwahati", "Kolkata")
    b = [i for i in items if i.category == "baseline comparison"][0]
    assert b.expect_status == "abstained"                 # only 2018-2020 baseline years -> fewer than 5
    out = evaluate(make_answer_fn(META, VARS, "Guwahati", CITIES), items, df, gappy)
    assert out["results"]["passed"].all()


def test_data_with_gaps_in_the_real_dataset_is_handled():
    df = make_synthetic("1991-01-01", "2023-12-31", VARS, seed=5)
    df.loc["2005-01-01":"2005-12-31", "precipitation_sum"] = np.nan        # a whole year missing
    items, gappy = build_eval_set(df, VARS, "Guwahati", "Kolkata")
    out = evaluate(make_answer_fn(META, VARS, "Guwahati", CITIES), items, df, gappy)
    assert out["results"]["passed"].all(), out["results"][~out["results"]["passed"]][["question", "detail"]].to_string()


def test_report_markdown_contains_metrics(setup):
    df, items, gappy = setup
    out = evaluate(real_fn(), items, df, gappy)
    md = report_markdown(out["metrics"], out["by_category"], "Testville", "synthetic")
    assert "Numerical accuracy" in md and "independently" in md and "| total rainfall |" in md
