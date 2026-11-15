import numpy as np
import pandas as pd
import pytest

from weather_platform.trend import (annual_series, benjamini_hochberg, mann_kendall, sens_slope, trend_table, trend_test)

mk_ref = pytest.importorskip("pymannkendall")


def test_perfect_increase_known_values():
    x = np.arange(1, 11, dtype=float)
    m = mann_kendall(x)
    assert m["S"] == 45 and m["tau"] == 1.0 and m["p_value"] < 0.001
    s = sens_slope(x)
    assert s["slope"] == 1.0 and s["ci_low"] == pytest.approx(1.0) and s["ci_high"] == pytest.approx(1.0)


def test_no_trend_series():
    rng = np.random.default_rng(0)
    m = mann_kendall(rng.normal(size=30))
    assert m["p_value"] > 0.05


def test_matches_pymannkendall_with_and_without_ties():
    rng = np.random.default_rng(3)
    for x in [rng.normal(size=25) + 0.05 * np.arange(25), np.round(rng.normal(size=40) * 2) + 0.1 * np.arange(40)]:
        ref = mk_ref.original_test(x)
        mine, sen = mann_kendall(x), sens_slope(x)
        assert mine["S"] == ref.s and mine["Z"] == pytest.approx(ref.z) and mine["p_value"] == pytest.approx(ref.p)
        assert sen["slope"] == pytest.approx(ref.slope)


def test_sen_slope_ci_coverage_simulation():
    rng = np.random.default_rng(1)
    hits, trials, true = 0, 300, 0.3
    t = np.arange(30, dtype=float)
    for _ in range(trials):
        s = sens_slope(true * t + rng.normal(0, 2.0, 30), t)
        hits += s["ci_low"] <= true <= s["ci_high"]
    assert hits / trials > 0.90


def test_trend_test_detects_trend_and_reports_per_decade():
    rng = np.random.default_rng(2)
    years = np.arange(1991, 2026)
    r = trend_test(0.04 * (years - 1991) + rng.normal(0, 0.3, len(years)), years)
    assert r["status"] == "ok" and r["significant"] and r["direction"] == "increasing"
    assert r["slope_per_decade"] == pytest.approx(0.4, abs=0.15) and r["slope_ci_low"] < r["slope_per_year"] < r["slope_ci_high"]


def test_insufficient_data_abstains():
    r = trend_test([1, 2, 3, np.nan, 5], [2000, 2001, 2002, 2003, 2004])
    assert r["status"] == "insufficient_data" and "at least" in r["reason"]


def test_hamed_rao_matches_reference_implementation():
    rng = np.random.default_rng(8)
    for _ in range(5):
        e = np.zeros(35)
        for i in range(1, 35):
            e[i] = 0.5 * e[i - 1] + rng.normal()
        x = e + 0.03 * np.arange(35)
        ref = mk_ref.hamed_rao_modification_test(x)
        mine = trend_test(x, np.arange(35) + 1990)
        # Sen's slope makes one pair of detrended values mathematically tied; the reference ranks that pair by floating-point noise,
        # this implementation treats it as a tie, so tiny differences (<0.5%) are expected on some series.
        assert mine["Z"] == pytest.approx(ref.z, rel=5e-3, abs=1e-6) and mine["p_value"] == pytest.approx(ref.p, rel=2e-2, abs=1e-6)
        assert mine["slope_per_year"] == pytest.approx(ref.slope)


def test_autocorrelation_correction_reduces_false_positives():
    rng = np.random.default_rng(5)
    n, trials = 40, 400
    plain = corrected = 0
    for _ in range(trials):
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.6 * e[i - 1] + rng.normal()
        yrs = np.arange(n)
        plain += trend_test(e, yrs, correct_autocorrelation=False)["p_value"] < 0.05
        corrected += trend_test(e, yrs, correct_autocorrelation=True)["p_value"] < 0.05
    assert plain / trials > 0.15                   # naive MK is badly over-confident on red noise
    assert corrected / trials < plain / trials     # the correction helps (it is not perfect for short series)


def test_benjamini_hochberg_known_example_and_nan():
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])
    adj2 = benjamini_hochberg([0.01, np.nan, 0.5])
    assert np.isnan(adj2[1]) and adj2[0] == pytest.approx(0.02)


def test_annual_series_skips_incomplete_years():
    idx = pd.date_range("2000-01-01", "2004-12-31")
    s = pd.Series(1.0, index=idx)
    s.loc["2002-01-01":"2002-09-30"] = np.nan
    a = annual_series(s, "mean")
    assert 2002 not in a.index and len(a) == 4


def test_trend_table_marks_planted_trend_and_applies_fdr():
    from weather_platform.data.cities import load_cities
    cs = load_cities()[:6]
    rng = np.random.default_rng(0)
    idx = pd.date_range("1991-01-01", "2025-12-31")
    data = {}
    for i, c in enumerate(cs):
        trend = 0.06 / 365.25 * np.arange(len(idx)) if i == 0 else 0.0          # only the first location warms
        t = 25 + 5 * np.sin(2 * np.pi * idx.day_of_year / 365.25) + trend + rng.normal(0, 1.5, len(idx))
        data[c["name"]] = pd.DataFrame({"temperature_2m_mean": t, "precipitation_sum": np.abs(rng.normal(3, 3, len(idx)))}, index=idx)
    tt = trend_table(data, cs, "temperature_2m_mean").set_index("name")
    assert tt.loc[cs[0]["name"], "significant_fdr"] and tt.loc[cs[0]["name"], "slope_per_decade"] == pytest.approx(0.6, abs=0.15)
    assert tt["p_fdr"].ge(tt["p_value"]).all() and tt["n_years"].eq(35).all()
    assert tt["significant_fdr"].sum() <= tt["significant_raw"].sum()
