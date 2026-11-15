import numpy as np
import pandas as pd
import pytest
from scipy import stats as sst

from weather_platform.stats.baseline import compare_to_baseline, window_value_by_year
from weather_platform.stats.descriptive import describe
from weather_platform.stats.rainfall import dry_spells, longest_dry_spell, rainfall_total, wet_dry_counts
from weather_platform.stats.summary import summarize


def daily(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def yearly_const(year_values: dict, name="v") -> pd.Series:
    """Daily series where every day of year y equals year_values[y]."""
    parts = []
    for y, v in year_values.items():
        idx = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        parts.append(pd.Series(v, index=idx, dtype=float))
    return pd.concat(parts)


# ---------------- descriptive ----------------
def test_describe_known_values_and_ignores_nan():
    d = describe(daily([2, 4, 4, 4, 5, 5, 7, 9, np.nan]))
    assert d["mean"] == 5.0 and d["median"] == 4.5 and d["min"] == 2 and d["max"] == 9
    assert d["std"] == pytest.approx(np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1))
    assert d["n_valid"] == 8 and d["n_missing"] == 1
    assert d["max_date"] == "2020-01-08"


def test_describe_all_missing():
    d = describe(daily([np.nan, np.nan]))
    assert d["mean"] is None and d["n_valid"] == 0


# ---------------- rainfall ----------------
def test_rainfall_total_flags_lower_bound_when_missing():
    t = rainfall_total(daily([0, 5, np.nan, 10]))
    assert t["total_mm"] == 15 and t["is_lower_bound"] and t["n_missing_days"] == 1
    assert not rainfall_total(daily([0, 5, 10]))["is_lower_bound"]


def test_wet_day_threshold_is_inclusive_and_missing_not_counted():
    c = wet_dry_counts(daily([0.0, 0.99, 1.0, 5.0, np.nan]), wet_mm=1.0)
    assert (c["wet_days"], c["dry_days"], c["zero_rain_days"], c["missing_days"]) == (2, 2, 1, 1)
    assert c["wet_day_fraction"] == 0.5


def test_dry_spells_missing_breaks_run():
    s = daily([0, 0, 0, 5, 0, 0, np.nan, 0, 0, 0, 0])
    sp = dry_spells(s)
    assert sp["length_days"].tolist() == [3, 2, 4]
    best = longest_dry_spell(s)
    assert best["length_days"] == 4 and best["start"] == "2020-01-08" and best["end"] == "2020-01-11"


def test_dry_spell_all_wet():
    assert longest_dry_spell(daily([5, 6, 7]))["length_days"] == 0


def test_dry_spell_runs_to_end():
    assert longest_dry_spell(daily([5, 0, 0, 0]))["length_days"] == 3


# ---------------- window helper ----------------
def test_window_value_by_year_mean_and_sum():
    s = yearly_const({2019: 2.0, 2020: 3.0})
    m = window_value_by_year(s, "06-01", "08-31", [2019, 2020], "mean")
    assert m["value"].tolist() == [2.0, 3.0]
    t = window_value_by_year(s, "06-01", "08-31", [2019], "sum")
    assert t.loc[2019, "value"] == 2.0 * 92


def test_window_excludes_years_with_too_much_missing():
    s = yearly_const({2019: 2.0, 2020: 3.0})
    s.loc["2020-06-01":"2020-07-15"] = np.nan  # 46 of 92 days missing
    w = window_value_by_year(s, "06-01", "08-31", [2019, 2020], "mean")
    assert w["used"].tolist() == [True, False] and np.isnan(w.loc[2020, "value"])


def test_feb29_window_does_not_crash():
    s = yearly_const({2019: 1.0, 2020: 1.0})
    w = window_value_by_year(s, "02-01", "02-29", [2019, 2020], "mean")
    assert w["used"].all()


# ---------------- baseline comparison ----------------
def test_single_year_vs_baseline_numbers():
    s = yearly_const({y: float(y - 1990) for y in range(1991, 2021)})  # 1991->1 ... 2020->30
    r = compare_to_baseline(s, "2020-01-01", "2020-12-31", (1991, 2020))
    base = np.arange(1, 30.0)
    assert r["status"] == "ok" and r["baseline_years_excluded_overlap"] == [2020]
    assert r["n_baseline_years"] == 29
    assert r["baseline_mean"] == pytest.approx(15.0) and r["baseline_sd"] == pytest.approx(base.std(ddof=1))
    assert r["difference"] == pytest.approx(15.0)
    assert r["z_score"] == pytest.approx(15 / base.std(ddof=1))
    assert r["percentile_in_baseline"] == 100.0
    half = sst.t.ppf(0.975, 28) * base.std(ddof=1) * np.sqrt(1 + 1 / 29)
    assert r["difference_interval"][1] == pytest.approx(15 + half)
    assert r["outside_normal_variability"] is False  # 15 < 17.7
    assert r["caveats"]


def test_unusual_year_flagged():
    vals = {y: 20.0 + 0.1 * (y % 3) for y in range(1991, 2020)}
    vals[2020] = 27.0
    r = compare_to_baseline(yearly_const(vals), "2020-01-01", "2020-12-31", (1991, 2019))
    assert r["outside_normal_variability"] is True and r["difference"] > 6


def test_relative_difference_for_rainfall_style_stat():
    vals = {y: 10.0 for y in range(1991, 2020)}
    vals[2020] = 15.0
    vals[1995] = 11.0  # a little spread so sd > 0
    r = compare_to_baseline(yearly_const(vals), "2020-01-01", "2020-12-31", (1991, 2019), stat="sum", relative=True)
    assert r["percent_difference"] == pytest.approx(100 * r["difference"] / r["baseline_mean"])


def test_insufficient_baseline_abstains():
    s = yearly_const({2018: 1.0, 2019: 2.0, 2020: 3.0})
    r = compare_to_baseline(s, "2020-01-01", "2020-12-31", (2018, 2019))
    assert r["status"] == "insufficient_data" and "baseline" in r["reason"].lower()


def test_selected_period_all_missing_abstains():
    s = yearly_const({y: float(y) for y in range(1991, 2021)})
    s.loc["2020"] = np.nan
    r = compare_to_baseline(s, "2020-01-01", "2020-12-31", (1991, 2019))
    assert r["status"] == "insufficient_data"


def test_baseline_year_with_missing_data_is_dropped_and_reported():
    s = yearly_const({y: float(y % 7) for y in range(1991, 2021)})
    s.loc["1995"] = np.nan
    r = compare_to_baseline(s, "2020-01-01", "2020-12-31", (1991, 2019))
    assert 1995 in r["baseline_years_dropped_missing_data"] and r["n_baseline_years"] == 28


def test_multi_year_uses_welch_and_matches_scipy():
    rng = np.random.default_rng(1)
    vals = {y: float(rng.normal(20, 1)) for y in range(1991, 2016)}
    vals.update({y: float(rng.normal(21.5, 1)) for y in range(2016, 2021)})
    s = yearly_const(vals)
    r = compare_to_baseline(s, "2016-01-01", "2020-12-31", (1991, 2020))
    sel = np.array([vals[y] for y in range(2016, 2021)]); base = np.array([vals[y] for y in range(1991, 2016)])
    ref = sst.ttest_ind(sel, base, equal_var=False)
    assert r["p_value"] == pytest.approx(ref.pvalue)
    assert r["difference_ci"][0] < r["difference"] < r["difference_ci"][1]
    assert r["bootstrap_ci"][0] < r["difference"] < r["bootstrap_ci"][1]
    assert r["n_selected_years"] == 5 and r["method"].startswith("Welch")


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(2)
    s = yearly_const({y: float(rng.normal(20, 1)) for y in range(1991, 2021)})
    a = compare_to_baseline(s, "2018-01-01", "2020-12-31", (1991, 2020), seed=5)
    b = compare_to_baseline(s, "2018-01-01", "2020-12-31", (1991, 2020), seed=5)
    assert a["bootstrap_ci"] == b["bootstrap_ci"]


# ---------------- summary ----------------
def test_summarize_contains_everything():
    df = pd.DataFrame({"temperature_2m_mean": [20.0, 22.0, np.nan], "precipitation_sum": [0.0, 3.0, np.nan]},
                      index=pd.date_range("2020-01-01", periods=3))
    out = summarize(df, ["temperature_2m_mean", "precipitation_sum"])
    assert out["n_days"] == 3 and out["descriptive"]["temperature_2m_mean"]["mean"] == 21.0
    assert out["rainfall"]["total_mm"] == 3.0 and out["rainfall"]["is_lower_bound"] is True
