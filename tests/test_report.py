import io

import pytest

pypdf = pytest.importorskip("pypdf")

from weather_platform.data.synthetic import make_synthetic
from weather_platform.quality.checks import clean_daily
from weather_platform.report.pdf import analysis_config, build_report
from weather_platform.stats.baseline import compare_to_baseline

VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
META = {"source": "Open-Meteo Historical Weather API (ERA5 reanalysis)", "source_url": "https://open-meteo.com/en/docs/historical-weather-api",
        "data_type": "Gridded reanalysis (model output), NOT direct station measurements", "location": "Guwahati, Assam",
        "latitude": 26.1445, "longitude": 91.7362, "start": "1995-01-01", "end": "2023-12-31", "fetched_at_utc": "2026-10-26T10:00:00+00:00"}


@pytest.fixture(scope="module")
def prepared():
    raw = make_synthetic("1995-01-01", "2023-12-31", VARS, seed=2, inject_defects=True)
    return clean_daily(raw, VARS, "1995-01-01", "2023-12-31")


def text_of(pdf: bytes) -> str:
    r = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join(p.extract_text() for p in r.pages)


def test_pdf_has_required_sections_and_disclaimers(prepared):
    df, rep = prepared
    pdf = build_report(df, rep, META, VARS, "monthly")
    assert pdf[:5] == b"%PDF-"
    t = text_of(pdf)
    for must in ["Guwahati, Assam", "1995-01-01 to 2023-12-31", "NOT a weather forecast", "Open-Meteo", "ERA5", "Copernicus",
                 "Methodology", "Limitations", "Data quality", "Missing versus zero", "26.1445", "mm", "Reproducibility"]:
        assert must in t, must
    assert "°C" in t or "C" in t


def test_every_page_has_forecast_disclaimer_footer(prepared):
    df, rep = prepared
    r = pypdf.PdfReader(io.BytesIO(build_report(df, rep, META, VARS)))
    assert len(r.pages) >= 3
    assert all("Not a weather forecast" in p.extract_text() for p in r.pages)


def test_missing_and_zero_reported_separately(prepared):
    df, rep = prepared
    t = text_of(build_report(df, rep, META, VARS))
    assert "counted separately from missing days" in t and "Missing days (not counted)" in t


def test_synthetic_data_is_loudly_labelled(prepared):
    df, rep = prepared
    t = text_of(build_report(df, rep, {**META, "source": "SYNTHETIC demo data (NOT real weather)"}, VARS))
    assert "SYNTHETIC demonstration data" in t


def test_baseline_section_with_results_and_without(prepared):
    df, rep = prepared
    res = compare_to_baseline(df["temperature_2m_mean"], "2023-01-01", "2023-12-31", (1995, 2020))
    res.update(label="Mean temperature", unit="°C")
    t = text_of(build_report(df, rep, META, VARS, baseline_results=[res]))
    assert "Percentile within baseline years" in t and "95% range of normal" in t
    t2 = text_of(build_report(df, rep, META, VARS, baseline_note="Baseline years are not covered by the loaded data."))
    assert "not covered by the loaded data" in t2


def test_temperature_only_and_rain_only_reports(prepared):
    df, rep = prepared
    raw = make_synthetic("2020-01-01", "2021-12-31", ["precipitation_sum"], seed=1)
    d2, r2 = clean_daily(raw, ["precipitation_sum"], "2020-01-01", "2021-12-31")
    assert build_report(d2, r2, {**META, "start": "2020-01-01", "end": "2021-12-31"}, ["precipitation_sum"])[:5] == b"%PDF-"
    raw = make_synthetic("2020-01-01", "2021-12-31", ["temperature_2m_mean"], seed=1)
    d3, r3 = clean_daily(raw, ["temperature_2m_mean"], "2020-01-01", "2021-12-31")
    assert build_report(d3, r3, {**META, "start": "2020-01-01", "end": "2021-12-31"}, ["temperature_2m_mean"])[:5] == b"%PDF-"


def test_special_characters_do_not_break_the_report(prepared):
    df, rep = prepared
    assert build_report(df, rep, {**META, "location": "A & B <test>, State"}, VARS)[:5] == b"%PDF-"


def test_analysis_config_is_complete_and_serialisable():
    import json
    c = analysis_config(META, VARS, "monthly", 1.0, 0.9, (1991, 2020), ("2023-01-01", "2023-12-31"))
    json.dumps(c)
    assert c["wet_day_threshold_mm"] == 1.0 and c["baseline_years"] == [1991, 2020] and c["cache_key"] and "pandas" in c["software"]


def test_trend_section_with_results_and_without(prepared):
    from weather_platform.trend import annual_series, trend_test
    df, rep = prepared
    ann = annual_series(df["temperature_2m_mean"], "mean")
    tr = trend_test(ann.values, ann.index.values)
    tr.update(label="Mean temperature", unit="°C")
    t = text_of(build_report(df, rep, META, VARS, trend_results=[tr]))
    assert "Long-term trend" in t and "Change per decade" in t and "Hamed-Rao" in t
    t2 = text_of(build_report(df, rep, META, VARS))
    assert "No trend test is included" in t2 and "Methodology" in t2 and "Reproducibility" in t2
