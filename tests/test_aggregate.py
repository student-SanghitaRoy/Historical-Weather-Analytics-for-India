import io

import numpy as np
import pandas as pd
import pytest

from weather_platform.aggregate import aggregate, monthly_climatology, pivot_by_year, season_bounds
from weather_platform.data.synthetic import make_synthetic
from weather_platform.export import aggregated_csv_bytes, daily_csv_bytes
from weather_platform.viz import charts


def frame(start, end, rain=1.0, temp=20.0):
    idx = pd.date_range(start, end, freq="D", name="date")
    return pd.DataFrame({"precipitation_sum": rain, "temperature_2m_mean": temp}, index=idx, dtype=float)


def test_monthly_sum_and_mean():
    df = frame("2020-01-01", "2020-02-29")
    a = aggregate(df, ["precipitation_sum", "temperature_2m_mean"], "monthly")
    p = a[a.variable == "precipitation_sum"].set_index("period_label")
    assert p.loc["2020-01", "value"] == 31 and p.loc["2020-02", "value"] == 29  # leap Feb
    t = a[a.variable == "temperature_2m_mean"].set_index("period_label")
    assert t.loc["2020-01", "value"] == 20.0 and a["stat"].nunique() == 2


def test_incomplete_month_is_nan_not_partial_total():
    df = frame("2020-01-01", "2020-01-31")
    df.loc["2020-01-10":"2020-01-20", "precipitation_sum"] = np.nan  # 11 of 31 days missing
    a = aggregate(df, ["precipitation_sum"], "monthly")
    assert np.isnan(a.loc[0, "value"]) and not a.loc[0, "complete"] and a.loc[0, "n_valid"] == 20


def test_partial_month_at_range_edge_flagged_incomplete():
    a = aggregate(frame("2020-01-15", "2020-02-29"), ["precipitation_sum"], "monthly")
    assert a["complete"].tolist() == [False, True]


def test_zero_rain_month_is_zero_not_nan():
    a = aggregate(frame("2020-06-01", "2020-06-30", rain=0.0), ["precipitation_sum"], "monthly")
    assert a.loc[0, "value"] == 0.0 and a.loc[0, "complete"]


def test_seasonal_djf_uses_previous_december():
    df = frame("2019-12-01", "2020-02-29")
    a = aggregate(df, ["precipitation_sum"], "seasonal")
    row = a.iloc[0]
    assert row["period_label"] == "2020 DJF" and row["value"] == 31 + 31 + 29 and row["complete"]
    assert season_bounds(2020, "DJF") == (pd.Timestamp("2019-12-01"), pd.Timestamp("2020-02-29"))


def test_seasonal_first_december_alone_is_incomplete():
    a = aggregate(frame("2019-12-01", "2019-12-31"), ["precipitation_sum"], "seasonal")
    assert a.loc[0, "period_label"] == "2020 DJF" and not a.loc[0, "complete"]


def test_daily_passthrough_keeps_missing():
    df = frame("2020-01-01", "2020-01-03"); df.loc["2020-01-02", "precipitation_sum"] = np.nan
    a = aggregate(df, ["precipitation_sum"], "daily")
    assert a["value"].isna().tolist() == [False, True, False]


def test_bad_freq():
    with pytest.raises(ValueError):
        aggregate(frame("2020-01-01", "2020-01-05"), ["precipitation_sum"], "weekly")


def test_climatology_and_pivot():
    df = make_synthetic("2018-01-01", "2020-12-31", ["precipitation_sum", "temperature_2m_mean"], seed=3)
    m = aggregate(df, ["precipitation_sum", "temperature_2m_mean"], "monthly")
    c = monthly_climatology(m, "precipitation_sum")
    assert len(c) == 12 and (c["n_years"] == 3).all()
    jan = m[(m.variable == "precipitation_sum") & (m["sub"] == 1)]["value"]
    assert c.loc[1, "mean"] == pytest.approx(jan.mean())
    p = pivot_by_year(m, "precipitation_sum")
    assert p.shape == (12, 3) and list(p.columns) == [2018, 2019, 2020]
    s = pivot_by_year(aggregate(df, ["precipitation_sum"], "seasonal"), "precipitation_sum")
    assert list(s.index) == ["DJF", "MAM", "JJA", "SON"]


def test_csv_missing_is_empty_not_zero_and_header_present():
    df = frame("2020-01-01", "2020-01-03"); df.loc["2020-01-02", "precipitation_sum"] = np.nan
    meta = {"location": "Testville", "latitude": 1.0, "longitude": 2.0, "start": "2020-01-01", "end": "2020-01-03"}
    raw = daily_csv_bytes(df, meta).decode("utf-8-sig")
    assert raw.startswith("# Source:") and "Testville" in raw and "NOT zero" in raw
    back = pd.read_csv(io.StringIO(raw), comment="#", index_col="date")
    assert "precipitation_sum (mm)" in back.columns and "temperature_2m_mean (°C)" in back.columns
    assert back["precipitation_sum (mm)"].isna().tolist() == [False, True, False]


def test_aggregated_csv_has_completeness_columns():
    a = aggregate(frame("2020-01-01", "2020-03-31"), ["precipitation_sum"], "monthly")
    raw = aggregated_csv_bytes(a, {"start": "2020-01-01", "end": "2020-03-31"}, "monthly", 0.9).decode("utf-8-sig")
    back = pd.read_csv(io.StringIO(raw), comment="#")
    assert {"n_valid", "n_expected", "complete", "unit"} <= set(back.columns) and len(back) == 3


def test_charts_build_with_units_and_gaps():
    df = make_synthetic("2018-01-01", "2020-12-31", ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max"], seed=1)
    m = aggregate(df, ["precipitation_sum"], "monthly")
    assert "°C" in charts.temperature_trend(df).layout.yaxis.title.text
    assert "mm" in charts.rainfall_chart(m, "monthly").layout.yaxis.title.text
    assert len(charts.year_comparison(pivot_by_year(m, "precipitation_sum"), "t", "mm").data) == 3
    assert len(charts.climatology_chart(monthly_climatology(m, "precipitation_sum"), "t", "mm").data) == 1


def test_aggregated_csv_values_are_rounded_without_float_noise():
    df = frame("2020-01-01", "2020-03-31", rain=0.1)            # 31 * 0.1 = 3.1000000000000005 in floating point
    a = aggregate(df, ["precipitation_sum"], "monthly")
    raw = aggregated_csv_bytes(a, {"start": "2020-01-01", "end": "2020-03-31"}, "monthly", 0.9).decode("utf-8-sig")
    assert "3.1000000000000005" not in raw and "3.1," in raw
