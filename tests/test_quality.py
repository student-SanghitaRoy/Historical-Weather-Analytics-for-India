import numpy as np
import pandas as pd
import pytest

from weather_platform import config
from weather_platform.data.synthetic import make_synthetic
from weather_platform.quality.checks import clean_daily, longest_missing_run, validate_request

TODAY = pd.Timestamp("2026-10-26")
PRECIP = ["precipitation_sum"]


def _df(values, start="2020-01-01", var="precipitation_sum", units=None):
    idx = pd.date_range(start, periods=len(values), freq="D", name="date")
    df = pd.DataFrame({var: values}, index=idx)
    df.attrs["units"] = units or {var: config.VARIABLES[var]["unit"]}
    return df


# ---------- missing vs zero ----------
def test_zero_rain_is_preserved_and_counted_separately():
    df = _df([0.0, 0.0, 5.0, np.nan, 0.0])
    out, rep = clean_daily(df, PRECIP, "2020-01-01", "2020-01-05")
    assert out["precipitation_sum"].tolist()[:3] == [0.0, 0.0, 5.0]
    assert np.isnan(out["precipitation_sum"].iloc[3])
    st = rep.per_variable["precipitation_sum"]
    assert st["n_zero_days"] == 3
    assert st["n_missing"] == 1
    assert st["pct_missing"] == 20.0


def test_missing_dates_are_nan_not_zero():
    df = _df([1.0, 2.0, 3.0]).drop(pd.Timestamp("2020-01-02"))
    out, rep = clean_daily(df, PRECIP, "2020-01-01", "2020-01-03")
    assert rep.missing_dates_added == 1
    assert np.isnan(out.loc["2020-01-02", "precipitation_sum"])
    assert rep.per_variable["precipitation_sum"]["n_zero_days"] == 0


# ---------- duplicates ----------
def test_duplicates_removed_and_conflicts_counted():
    df = _df([1.0, 2.0, 3.0])
    dup = pd.DataFrame({"precipitation_sum": [9.0]}, index=[pd.Timestamp("2020-01-02")])
    df2 = pd.concat([df, dup]); df2.attrs = df.attrs
    out, rep = clean_daily(df2, PRECIP, "2020-01-01", "2020-01-03")
    assert len(out) == 3
    assert rep.duplicate_rows_removed == 1
    assert rep.conflicting_duplicate_dates == 1
    assert out.loc["2020-01-02", "precipitation_sum"] == 2.0  # first kept


# ---------- impossible values ----------
def test_negative_rain_becomes_missing_not_zero():
    df = _df([1.0, -3.0, 2.0])
    out, rep = clean_daily(df, PRECIP, "2020-01-01", "2020-01-03")
    assert np.isnan(out["precipitation_sum"].iloc[1])
    assert rep.impossible_values == {"precipitation_sum": 1}


def test_impossible_temperature_flagged():
    df = _df([20.0, 150.0, -200.0], var="temperature_2m_mean")
    out, rep = clean_daily(df, ["temperature_2m_mean"], "2020-01-01", "2020-01-03")
    assert out["temperature_2m_mean"].isna().tolist() == [False, True, True]
    assert rep.impossible_values["temperature_2m_mean"] == 2


def test_min_greater_than_max_flagged():
    idx = pd.date_range("2020-01-01", periods=2, name="date")
    df = pd.DataFrame({"temperature_2m_min": [10.0, 30.0], "temperature_2m_max": [20.0, 25.0],
                       "temperature_2m_mean": [15.0, 27.0]}, index=idx)
    df.attrs["units"] = {v: "°C" for v in df.columns}
    out, rep = clean_daily(df, list(df.columns), "2020-01-01", "2020-01-02")
    assert rep.inconsistent_temperature_rows == 1
    assert out.iloc[1].isna().all() and out.iloc[0].notna().all()


# ---------- units ----------
def test_unit_mismatch_flagged_not_converted():
    df = _df([1.0, 2.0], units={"precipitation_sum": "inch"})
    out, rep = clean_daily(df, PRECIP, "2020-01-01", "2020-01-02")
    assert not rep.units_ok
    assert "inch" in rep.unit_issues[0]
    assert out["precipitation_sum"].tolist() == [1.0, 2.0]


def test_units_ok_when_matching():
    _, rep = clean_daily(_df([1.0]), PRECIP, "2020-01-01", "2020-01-01")
    assert rep.units_ok


# ---------- misc ----------
def test_rows_outside_range_dropped():
    df = _df([1, 2, 3, 4, 5])
    out, rep = clean_daily(df, PRECIP, "2020-01-02", "2020-01-04")
    assert len(out) == 3 and rep.out_of_range_rows_dropped == 2


def test_longest_missing_run():
    s = pd.Series([1, np.nan, np.nan, np.nan, 2, np.nan])
    assert longest_missing_run(s) == 3


def test_steps_are_logged():
    _, rep = clean_daily(_df([1.0, 2.0]), PRECIP, "2020-01-01", "2020-01-02")
    assert len(rep.steps) >= 7
    assert any("not zero" in s.lower() for s in rep.steps)


def test_synthetic_defects_are_caught_end_to_end():
    vars_ = ["temperature_2m_mean", "precipitation_sum"]
    df = make_synthetic("2020-01-01", "2020-03-31", vars_, inject_defects=True)
    out, rep = clean_daily(df, vars_, "2020-01-01", "2020-03-31")
    assert len(out) == 91
    assert rep.duplicate_rows_removed == 1
    assert rep.missing_dates_added == 1
    assert rep.impossible_values.get("precipitation_sum") == 1
    assert rep.per_variable["precipitation_sum"]["n_missing"] >= 4


# ---------- request validation ----------
@pytest.mark.parametrize("start,end,vars_,msg", [
    ("2020-05-01", "2020-04-01", ["precipitation_sum"], "after end"),
    ("1900-01-01", "1950-01-01", ["precipitation_sum"], "before the dataset"),
    ("2026-01-01", "2026-12-31", ["precipitation_sum"], "latest available"),
    ("2020-01-01", "2020-02-01", [], "at least one"),
    ("2020-01-01", "2020-02-01", ["humidity"], "Unsupported"),
])
def test_validate_request_rejects(start, end, vars_, msg):
    with pytest.raises(ValueError, match=msg):
        validate_request(start, end, vars_, today=TODAY)


def test_validate_request_accepts_good_input():
    s, e = validate_request("2020-01-01", "2020-12-31", ["precipitation_sum"], today=TODAY)
    assert (s, e) == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
