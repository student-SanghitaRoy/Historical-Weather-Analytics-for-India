import numpy as np
import pandas as pd

from weather_platform.anomaly import (compare_detectors, doy_aligned, flag_zscore, group_events, isolation_forest_flags,
                                      precision_recall, rainfall_extremes, robust_zscores)

BASE = (1991, 2010)


def make(seed=0, years=("1991-01-01", "2015-12-31")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(*years, freq="D")
    doy = idx.day_of_year.to_numpy()
    temp = 25 + 6 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 1.0, len(idx))
    wet = rng.random(len(idx)) < 0.35
    rain = np.where(wet, rng.gamma(0.8, 9.0, len(idx)), 0.0)
    return pd.DataFrame({"temperature_2m_mean": temp, "precipitation_sum": rain}, index=idx)


def test_doy_alignment_across_leap_years():
    a = doy_aligned(pd.DatetimeIndex(["2019-03-01", "2020-03-01", "2020-02-29", "2020-12-31", "2019-12-31"]))
    assert a[0] == a[1] == 60 and a[2] == 59 and a[3] == a[4] == 365


def test_robust_z_removes_seasonality_and_is_roughly_standard_normal():
    df = make()
    z = robust_zscores(df["temperature_2m_mean"], BASE)["z"].dropna()
    assert abs(z.median()) < 0.1 and 0.8 < z.std() < 1.3
    assert (z.abs() >= 3.5).mean() < 0.005          # few false alarms on normal data


def test_injected_temperature_spikes_are_found_with_few_false_alarms():
    df = make(seed=1)
    truth = pd.Series(False, index=df.index)
    spikes = df.index[[4000, 5200, 6100, 7000, 7600, 8000, 8300, 8600]]
    df.loc[spikes, "temperature_2m_mean"] += 9.0
    truth.loc[spikes] = True
    flags = flag_zscore(robust_zscores(df["temperature_2m_mean"], BASE))
    pr = precision_recall(flags, truth)
    assert pr["recall"] == 1.0 and pr["precision"] > 0.6
    iso = isolation_forest_flags(df, BASE, contamination=0.01)
    pri = precision_recall(iso, truth)
    assert pri["recall"] >= 0.75                      # comparison detector also finds big spikes
    cmp = compare_detectors(flags, iso)
    assert cmp["both"] >= 6 and 0 < cmp["jaccard"] <= 1


def test_missing_values_never_flagged_and_zero_scale_gives_nan():
    df = make()
    s = df["temperature_2m_mean"].copy()
    s.iloc[8000] = np.nan
    z = robust_zscores(s, BASE)
    assert np.isnan(z["z"].iloc[8000]) and not flag_zscore(z).iloc[8000]
    const = pd.Series(20.0, index=df.index)                       # MAD = 0 -> z undefined, not inf
    zc = robust_zscores(const, BASE)
    assert zc["z"].isna().all() and not flag_zscore(zc).any()


def test_baseline_too_small_returns_nan():
    df = make()
    z = robust_zscores(df["temperature_2m_mean"], (2010, 2010), min_n=60)
    assert z["z"].isna().all()                                    # one year gives only 15 values per window


def test_rainfall_extremes_flag_injected_storms_and_respect_wet_only():
    df = make(seed=2)
    storms = df.index[[3500, 6000, 8200]]
    df.loc[storms, "precipitation_sum"] = 400.0
    r = rainfall_extremes(df["precipitation_sum"], BASE)
    assert r.loc[storms, "flag"].all()
    assert 0.0 < r["flag"].mean() < 0.01 * 0.4 * 3 + 0.01        # roughly 1% of wet days (35% wet) plus slack
    assert not r.loc[df["precipitation_sum"] < 1.0, "flag"].any()


def test_rainfall_extremes_missing_not_flagged():
    df = make(seed=2)
    df.iloc[8000, 1] = np.nan
    assert not rainfall_extremes(df["precipitation_sum"], BASE)["flag"].iloc[8000]


def test_group_events_merges_consecutive_days():
    idx = pd.date_range("2020-01-01", periods=10)
    flag = pd.Series([0, 1, 1, 1, 0, 0, 1, 0, 0, 1], index=idx).astype(bool)
    score = pd.Series([0, 4, 5.5, 4.2, 0, 0, -6, 0, 0, 3.6], index=idx)
    ev = group_events(flag, score)
    assert ev["length_days"].tolist() == [3, 1, 1] and ev.loc[0, "peak_score"] == 5.5 and ev.loc[1, "peak_score"] == -6
    assert str(ev.loc[0, "peak_date"]) == "2020-01-03"
    assert len(group_events(flag, score, min_len=2)) == 1
    assert group_events(pd.Series(False, index=idx), score).empty


def test_precision_recall_math():
    f = pd.Series([1, 1, 0, 0], dtype=bool)
    t = pd.Series([1, 0, 1, 0], dtype=bool)
    assert precision_recall(f, t) == {"precision": 0.5, "recall": 0.5, "tp": 1, "fp": 1, "fn": 1}
