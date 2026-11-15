import json

import numpy as np
import pytest
from scipy import stats as sst

from weather_platform.compare import MIN_COMMON_YEARS, annual_matrix, compare_locations
from weather_platform.data.cities import get_city, load_cities
from weather_platform.data.synthetic import make_synthetic
from weather_platform.india import synthetic_datasets
from weather_platform.reproduce import compare_fingerprints, config_to_state, results_fingerprint, validate_config
from weather_platform.report.pdf import analysis_config

CS = load_cities()


@pytest.fixture(scope="module")
def data():
    return synthetic_datasets([get_city(n) for n in ("Guwahati", "Jaisalmer", "Shillong", "Cherrapunji")], "1991-01-01", "2020-12-31", seed=3)


# ---------------- comparison ----------------
def test_matching_years_paired_difference_matches_scipy(data):
    r = compare_locations(data, CS, ["Guwahati", "Jaisalmer"], "1991-01-01", "2020-12-31")
    v = r["variables"]["precipitation_sum"]
    assert r["status"] == "ok" and v["status"] == "ok" and v["unit"] == "mm" and len(v["years_used"]) == 30
    a, b = v["annual"]["Guwahati"].to_numpy(), v["annual"]["Jaisalmer"].to_numpy()
    p = v["pairs"]["Guwahati minus Jaisalmer"]
    assert p["mean_difference"] == pytest.approx((a - b).mean())
    assert p["p_value_ttest"] == pytest.approx(sst.ttest_rel(a, b).pvalue)
    d = a - b
    half = sst.t.ppf(0.975, 29) * d.std(ddof=1) / np.sqrt(30)
    assert p["ci_low"] == pytest.approx(d.mean() - half) and p["ci_high"] == pytest.approx(d.mean() + half)
    assert p["mean_difference"] > 0 and p["p_value_wilcoxon"] is not None       # Guwahati is far wetter than Jaisalmer
    assert r["notes"] and "Guwahati - Jaisalmer" in r["distances_km"]


def test_only_years_complete_everywhere_are_used(data):
    d = {k: v.copy() for k, v in data.items()}
    d["Jaisalmer"].loc["2000-01-01":"2000-12-31", "precipitation_sum"] = np.nan
    r = compare_locations(d, CS, ["Guwahati", "Jaisalmer"], "1991-01-01", "2020-12-31")
    v = r["variables"]["precipitation_sum"]
    assert 2000 not in v["years_used"] and len(v["years_used"]) == 29
    assert 2000 in r["variables"]["temperature_2m_mean"]["years_used"]       # temperature is still complete


def test_insufficient_common_years_and_too_few_locations(data):
    r = compare_locations(data, CS, ["Guwahati", "Jaisalmer"], "2018-01-01", "2020-12-31")
    assert r["variables"]["precipitation_sum"]["status"] == "insufficient_data" and str(MIN_COMMON_YEARS) in r["variables"]["precipitation_sum"]["reason"]
    assert compare_locations(data, CS, ["Guwahati"], "1991-01-01", "2020-12-31")["status"] == "insufficient_data"
    assert compare_locations(data, CS, ["Guwahati", "Nowhere"], "1991-01-01", "2020-12-31")["status"] == "insufficient_data"


def test_inconsistent_units_are_refused(data):
    d = {k: v.copy() for k, v in data.items()}
    d["Jaisalmer"].attrs = {"units": {"temperature_2m_mean": "°F", "precipitation_sum": "mm"}}
    r = compare_locations(d, CS, ["Guwahati", "Jaisalmer"], "1991-01-01", "2020-12-31")
    assert r["status"] == "refused" and "consistent" in r["reason"]


def test_nearby_locations_get_shared_grid_warning(data):
    r = compare_locations(data, CS, ["Shillong", "Cherrapunji"], "1991-01-01", "2020-12-31")
    assert r["distances_km"]["Shillong - Cherrapunji"] < 50
    r2 = compare_locations(data, CS, ["Guwahati", "Jaisalmer"], "1991-01-01", "2020-12-31")
    assert r2["shared_grid_warnings"] == []


def test_three_locations_make_three_pairs(data):
    r = compare_locations(data, CS, ["Guwahati", "Jaisalmer", "Shillong"], "1991-01-01", "2020-12-31")
    assert len(r["variables"]["temperature_2m_mean"]["pairs"]) == 3


def test_annual_matrix_marks_incomplete_years_nan(data):
    d = {k: v.copy() for k, v in data.items()}
    d["Guwahati"].loc["1995-03-01":"1995-12-31", "temperature_2m_mean"] = np.nan
    m = annual_matrix(d, ["Guwahati"], "temperature_2m_mean", "1991-01-01", "2020-12-31")
    assert np.isnan(m.loc[1995, "Guwahati"]) and m["Guwahati"].notna().sum() == 29


# ---------------- reproducibility ----------------
META = {"source": "Open-Meteo (ERA5)", "location": "Guwahati, Assam", "latitude": 26.1445, "longitude": 91.7362, "start": "2015-01-01", "end": "2020-12-31"}


def good_cfg(df=None):
    df = df if df is not None else make_synthetic("2015-01-01", "2020-12-31", ["temperature_2m_mean", "precipitation_sum"], seed=1)
    return analysis_config(META, ["temperature_2m_mean", "precipitation_sum"], "monthly", 1.0, 0.9, (1991, 2020), fingerprint=results_fingerprint(df, ["temperature_2m_mean", "precipitation_sum"]))


def test_config_roundtrip_through_json_restores_every_setting():
    cfg = json.loads(json.dumps(good_cfg()))
    assert validate_config(cfg) == []
    st = config_to_state(cfg)
    assert st["city"]["name"] == "Guwahati" and str(st["start"]) == "2015-01-01" and str(st["end"]) == "2020-12-31"
    assert st["vars"] == ["temperature_2m_mean", "precipitation_sum"] and st["freq"] == "monthly" and st["wet"] == 1.0 and (st["b0"], st["b1"]) == (1991, 2020)
    assert st["use_synth"] is False


@pytest.mark.parametrize("mutate,frag", [
    (lambda c: c.pop("start"), "Missing field"), (lambda c: c.update(start="2021-01-01", end="2020-01-01"), "after end"),
    (lambda c: c.update(variables=["humidity"]), "Unsupported"), (lambda c: c.update(variables=[]), "Unsupported"),
    (lambda c: c.update(aggregation="weekly"), "aggregation"), (lambda c: c.update(location="Atlantis, Nowhere"), "not supported"),
    (lambda c: c.update(wet_day_threshold_mm=-1), "wet-day"), (lambda c: c.update(start="garbage"), "cannot be read")])
def test_invalid_configs_are_rejected(mutate, frag):
    cfg = json.loads(json.dumps(good_cfg()))
    mutate(cfg)
    assert any(frag.lower() in p.lower() for p in validate_config(cfg)), validate_config(cfg)


def test_non_dict_config_rejected():
    assert validate_config([1, 2]) == ["Not a valid analysis configuration file."]


def test_fingerprint_detects_changes_and_is_stable():
    v = ["temperature_2m_mean", "precipitation_sum"]
    df = make_synthetic("2015-01-01", "2020-12-31", v, seed=1)
    fp = results_fingerprint(df, v)
    assert fp == results_fingerprint(df.copy(), v)
    changed = df.copy()
    changed.iloc[100, 1] += 0.5
    assert results_fingerprint(changed, v) != fp
    gap = df.copy()
    gap.iloc[100, 1] = np.nan
    assert results_fingerprint(gap, v) != fp


def test_rerun_from_saved_config_reproduces_identical_results():
    v = ["temperature_2m_mean", "precipitation_sum"]
    df1 = make_synthetic("2015-01-01", "2020-12-31", v, seed=1)
    cfg = json.loads(json.dumps(good_cfg(df1)))
    df2 = make_synthetic("2015-01-01", "2020-12-31", v, seed=1)          # re-running the same configuration
    ok, msg = compare_fingerprints(cfg["results_fingerprint"], results_fingerprint(df2, v))
    assert ok is True and "identical" in msg
    ok2, msg2 = compare_fingerprints(cfg["results_fingerprint"], results_fingerprint(make_synthetic("2015-01-01", "2020-12-31", v, seed=2), v))
    assert ok2 is False and "differ" in msg2
    assert compare_fingerprints(None, "x")[0] is None


def test_synthetic_source_in_config_switches_synthetic_mode_on():
    cfg = good_cfg()
    cfg["data_source"] = "SYNTHETIC demo data (NOT real weather)"
    assert config_to_state(cfg)["use_synth"] is True
