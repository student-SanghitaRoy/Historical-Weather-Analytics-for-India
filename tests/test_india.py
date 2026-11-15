import numpy as np
import pandas as pd
import pytest

from weather_platform.data.cities import get_city, load_cities
from weather_platform.data.geonames_check import check_cities, find_match, haversine_km, parse_geonames
from weather_platform.india import (PREFETCH_END, PREFETCH_START, anomaly_table, city_table, load_datasets, prefetch,
                                    prefetch_until_done, synthetic_datasets)
from weather_platform.mining.climate_regimes import build_feature_matrix, city_profile, cluster_regimes, describe_cluster
from weather_platform.viz import maps

TODAY = pd.Timestamp("2026-10-26")


@pytest.fixture(scope="module")
def cities():
    return load_cities()


@pytest.fixture(scope="module")
def data(cities):
    return synthetic_datasets(cities, "1991-01-01", "2023-12-31", seed=0)


# ---------------- synthetic demo data + tables ----------------
def test_synthetic_shapes_and_climates(data, cities):
    assert set(data) == {c["name"] for c in cities}
    t = city_table(data, cities).set_index("name")
    assert t.loc["Leh", "mean_annual_temp_C"] < t.loc["Chennai", "mean_annual_temp_C"] - 10     # hills cooler than the coast
    assert t.loc["Cherrapunji", "mean_annual_rain_mm"] > 5 * t.loc["Jaisalmer", "mean_annual_rain_mm"]
    assert (t["n_years_rain"] == 33).all()


def test_anomaly_table_has_one_row_per_city(data):
    a = anomaly_table(data, 2023, "precipitation_sum")
    assert len(a) == len(data) and (a["status"] == "ok").all()
    assert np.isfinite(a["percent_difference"]).all()


def test_anomaly_table_marks_insufficient_data(data):
    short = {k: v.loc["2015":] for k, v in list(data.items())[:3]}
    assert (anomaly_table(short, 2023, "temperature_2m_mean", baseline=(2018, 2020))["status"] == "insufficient_data").all()


# ---------------- clustering ----------------
def _profile(temp, rain_year, rng):
    t = temp + 6 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12) + rng.normal(0, 0.4, 12)
    r = rain_year * np.exp(-0.5 * ((np.arange(12) - 6.5) / 1.6) ** 2) / 4.0 * (1 + rng.normal(0, 0.05, 12))
    return list(t) + list(np.clip(r, 0, None))


def test_clustering_recovers_three_planted_regimes():
    rng = np.random.default_rng(0)
    rows, truth = {}, {}
    for lab, (temp, rain) in enumerate([(12, 900), (28, 150), (27, 2600)]):   # cool-moderate / hot-dry / hot-very-wet
        for i in range(10):
            name = f"c{lab}_{i}"
            rows[name] = _profile(temp, rain, rng)
            truth[name] = lab
    from weather_platform.mining.climate_regimes import RCOLS, TCOLS
    X = pd.DataFrame(rows, index=TCOLS + RCOLS).T
    res = cluster_regimes(X)
    from sklearn.metrics import adjusted_rand_score
    assert res["best_k_by_silhouette"] == 3 and res["k"] == 3
    assert adjusted_rand_score([truth[n] for n in X.index], [res["labels"][n] for n in X.index]) > 0.95
    assert res["stability_ari"] > 0.9 and res["ward_agreement_ari"] > 0.9
    assert set(res["silhouette_by_k"]["k"]) == set(range(2, 9)) and res["caveats"]
    assert res["summary"]["n_locations"].sum() == 30


def test_clustering_on_synthetic_india_separates_extremes(data):
    X, skipped = build_feature_matrix(data)
    assert not skipped and X.shape == (len(data), 24)
    res = cluster_regimes(X)
    lab = res["labels"]
    assert lab["Jaisalmer"] != lab["Cherrapunji"] and lab["Leh"] != lab["Chennai"]
    assert 2 <= res["k"] <= 8 and len(res["summary"]) == res["k"]
    assert res["pca_coords"].shape == (len(data), 2)


def test_user_can_override_k(data):
    X, _ = build_feature_matrix(data)
    assert cluster_regimes(X, k=4)["k"] == 4


def test_profile_requires_enough_years(data):
    assert city_profile(data["Delhi"].loc["2020":"2021"]) is None
    assert len(city_profile(data["Delhi"])) == 24


def test_too_few_locations_raises():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 24)))
    with pytest.raises(ValueError):
        cluster_regimes(X)


def test_describe_cluster_thresholds():
    from weather_platform.mining.climate_regimes import RCOLS, TCOLS
    s = pd.Series([30.0] * 12 + [300.0] * 12, index=TCOLS + RCOLS)   # 3600 mm / year, 30 C
    assert describe_cluster(s) == "hot, extremely wet"


# ---------------- prefetch / cache ----------------
class FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        days = pd.date_range(params["start_date"], params["end_date"], freq="D")
        return FakeResp({"daily_units": {"temperature_2m_mean": "°C", "precipitation_sum": "mm"},
                         "daily": {"time": [d.strftime("%Y-%m-%d") for d in days],
                                   "temperature_2m_mean": [25.0] * len(days), "precipitation_sum": [0.0, 2.0] * (len(days) // 2) + [0.0] * (len(days) % 2)}})


def test_prefetch_is_resumable_and_loadable(tmp_path):
    cs = [get_city(n) for n in ("Guwahati", "Delhi", "Chennai")]
    sess = FakeSession()
    r1 = prefetch(cs, str(tmp_path), delay=0, max_new=2, session=sess, today=TODAY, sleep=lambda *_: None, log=lambda *_: None)
    assert len(r1["fetched"]) == 2 and r1["skipped"] == ["Chennai"] and sess.calls == 2
    r2 = prefetch(cs, str(tmp_path), delay=0, session=sess, today=TODAY, sleep=lambda *_: None, log=lambda *_: None)
    assert r2["cached"] == ["Guwahati", "Delhi"] and r2["fetched"] == ["Chennai"] and sess.calls == 3
    ds, missing = load_datasets(cs, str(tmp_path))
    assert not missing and len(ds["Delhi"]) == len(pd.date_range(PREFETCH_START, PREFETCH_END))
    assert ds["Delhi"]["temperature_2m_mean"].notna().all()


def test_prefetch_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class Bad:
        def get(self, *a, **k):
            return type("R", (), {"status_code": 429, "text": "rate limited", "json": lambda s: {}})()
    r = prefetch([get_city("Delhi")], str(tmp_path), delay=0, session=Bad(), today=TODAY, sleep=lambda *_: None, log=lambda *_: None)
    assert "Delhi" in r["failed"] and "429" in r["failed"]["Delhi"]


def test_load_datasets_reports_uncached(tmp_path):
    ds, missing = load_datasets([get_city("Delhi")], str(tmp_path))
    assert ds == {} and missing == ["Delhi"]


# ---------------- geonames check ----------------
GN = ["1\tGuwahati\tGuwahati\tGauhati,Gawahati\t26.1844\t91.7458\tP\tPPLA\tIN\t\t03\t\t\t\t957352\t\t55\tAsia/Kolkata\t2020-01-01",
      "2\tGuwahati Hill\tGuwahati Hill\t\t26.0\t91.5\tH\tHLL\tIN\t\t03\t\t\t\t0\t\t55\tAsia/Kolkata\t2020-01-01",
      "bad line"]


def test_geonames_parse_match_and_distance():
    recs = parse_geonames(GN)
    assert len(recs) == 2 and recs[0]["population"] == 957352
    m = find_match({"name": "Guwahati"}, recs)
    assert m["fclass"] == "P"
    assert find_match({"name": "Gauhati"}, recs) is not None          # alternate name
    out = check_cities([{"name": "Guwahati", "lat": 26.1445, "lon": 91.7362}, {"name": "Nowhere", "lat": 1, "lon": 1},
                        {"name": "Guwahati", "lat": 20.0, "lon": 80.0}], recs)
    assert [o["status"] for o in out] == ["ok", "no match", "CHECK"]
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)


# ---------------- maps ----------------
def test_maps_build_and_never_colour_missing_as_zero(cities):
    f1 = maps.location_map(cities, selected="Delhi")
    assert len(f1.data) == 2 and len(f1.data[0].lat) == len(cities) - 1
    vals = {c["name"]: 10.0 for c in cities[:5]}
    f2 = maps.value_map(cities, vals, "t", "mm")
    assert {t.name for t in f2.data if t.name} == {"No data", "Value"}
    assert len(next(t for t in f2.data if t.name == "Value").lat) == 5 and len(next(t for t in f2.data if t.name == "No data").lat) == len(cities) - 5
    f3 = maps.category_map(cities, {c["name"]: i % 3 for i, c in enumerate(cities)}, "t", {0: "a", 1: "b", 2: "c"})
    assert len(f3.data) == 3


def test_prefetch_stops_cleanly_on_hourly_limit(tmp_path):
    class Limited:
        calls = 0

        def get(self, *a, **k):
            Limited.calls += 1
            return type("R", (), {"status_code": 429, "text": '{"reason":"Hourly API request limit exceeded"}', "json": lambda s: {}})()
    cs = [get_city(n) for n in ("Delhi", "Patna", "Agra")]
    r = prefetch(cs, str(tmp_path), delay=0, session=Limited(), today=TODAY, sleep=lambda *_: None, log=lambda *_: None)
    assert Limited.calls == 1                      # no pointless retries or further cities
    assert r["skipped"] == ["Delhi", "Patna", "Agra"] and not r["failed"]


def test_auto_mode_waits_out_hourly_limit_then_finishes(tmp_path):
    ok = FakeSession()

    class LimitThenOk:
        """First 2 requests succeed, then the 'hourly limit' blocks until we have 'slept' once."""
        def __init__(self):
            self.n, self.slept = 0, 0

        def get(self, url, params=None, timeout=None):
            self.n += 1
            if self.n > 2 and self.slept == 0:
                return type("R", (), {"status_code": 429, "text": '{"reason":"Hourly API request limit exceeded"}', "json": lambda s: {}})()
            return ok.get(url, params, timeout)
    sess = LimitThenOk()
    waits = []

    def fake_sleep(sec):
        if sec >= 3600:
            sess.slept += 1
            waits.append(sec)
    cs = [get_city(n) for n in ("Delhi", "Patna", "Agra", "Kanpur")]
    out = prefetch_until_done(cs, str(tmp_path), delay=0, wait_minutes=65, session=sess, today=TODAY, sleep=fake_sleep, log=lambda *_: None)
    assert out["complete"] and out["rounds"] == 2 and waits == [65 * 60]
    ds, missing = load_datasets(cs, str(tmp_path))
    assert not missing


def test_auto_mode_gives_up_after_max_rounds(tmp_path):
    class Always429:
        def get(self, *a, **k):
            return type("R", (), {"status_code": 429, "text": "Hourly API request limit exceeded", "json": lambda s: {}})()
    out = prefetch_until_done([get_city("Delhi")], str(tmp_path), delay=0, max_rounds=2, session=Always429(), today=TODAY,
                              sleep=lambda *_: None, log=lambda *_: None)
    assert not out["complete"] and out["rounds"] == 2 and out["remaining"] == ["Delhi"]


def test_value_map_temperature_uses_warm_is_red_and_dots_are_visible(cities):
    vals = {c["name"]: (1.0 if i % 2 else -1.0) for i, c in enumerate(cities)}
    blue_pos = maps.value_map(cities, vals, "t", "x", diverging=True)
    red_pos = maps.value_map(cities, vals, "t", "x", diverging=True, warm_is_red=True)
    assert blue_pos.data[-1].marker.colorscale[0][1] != red_pos.data[-1].marker.colorscale[0][1]
    assert blue_pos.layout.map.zoom <= 3.4 and abs(blue_pos.layout.map.center.lat - 21.5) < 3     # fixed view that shows all of India


def test_regimes_log_transform_and_k_selection_rules(data):
    X, _ = build_feature_matrix(data)
    res = cluster_regimes(X)
    assert res["log_rain"] and res["best_k_by_silhouette"] >= 3 and res["best_k_overall"] >= 2
    assert (res["silhouette_by_k"]["k"] == list(range(2, 9))).all()                       # every k is still reported
    assert "rain_range_mm" in res["summary"].columns
    raw = cluster_regimes(X, log_rain=False)
    assert not raw["log_rain"]


def test_mixed_group_is_flagged():
    from weather_platform.mining.climate_regimes import RCOLS, TCOLS
    rng = np.random.default_rng(0)
    rows = {f"wet{i}": [10.0] * 12 + [300.0 + rng.normal(0, 5)] * 12 for i in range(4)}
    rows.update({f"dry{i}": [10.0] * 12 + [5.0 + rng.normal(0, 1)] * 12 for i in range(4)})
    rows.update({f"mid{i}": [25.0] * 12 + [60.0 + rng.normal(0, 3)] * 12 for i in range(4)})
    X = pd.DataFrame(rows, index=TCOLS + RCOLS).T
    res = cluster_regimes(X, k=2)
    assert any("mixed group" in d for d in res["summary"]["description"])


def test_summary_features_option_works_and_is_interpretable(data):
    from weather_platform.mining.climate_regimes import summary_features
    X, _ = build_feature_matrix(data)
    F = summary_features(X)
    assert list(F.columns) == ["mean_temp", "temp_range", "log_annual_rain", "monsoon_share"] and F.notna().all().all()
    assert F.loc["Leh", "mean_temp"] < F.loc["Chennai", "mean_temp"] and F.loc["Cherrapunji", "log_annual_rain"] > F.loc["Jaisalmer", "log_annual_rain"]
    assert (F["monsoon_share"].between(0, 1)).all()
    res = cluster_regimes(X, features="summary")
    assert res["features"] == "summary" and res["labels"]["Jaisalmer"] != res["labels"]["Cherrapunji"]
    with pytest.raises(ValueError):
        cluster_regimes(X, features="nonsense")


def test_cluster_colours_are_identical_on_map_and_helper_and_colour_blind_safe(cities):
    labels = {c["name"]: i % 4 for i, c in enumerate(cities)}
    cmap = maps.cluster_color_map(labels.values())
    fig = maps.category_map(cities, labels, "t")
    for trace in fig.data:
        lab = trace.name.replace("Cluster ", "")
        assert trace.marker.color.lower() == cmap[lab].lower()
    assert len(set(cmap.values())) == len(cmap)
    assert "#0072B2" in maps.PALETTE and "#D55E00" in maps.PALETTE            # Okabe-Ito blue and vermillion


def test_value_map_accepts_a_sequential_scale_and_draws_outlines(cities):
    vals = {c["name"]: float(i) for i, c in enumerate(cities)}
    f = maps.value_map(cities, vals, "t", "mm", scale="YlGnBu")
    assert len(f.data) == 2 and f.data[0].hoverinfo == "skip"                    # outline ring + the coloured dots
    assert f.data[1].marker.colorscale[0][1] != maps.value_map(cities, vals, "t", "mm", scale="YlOrRd").data[1].marker.colorscale[0][1]


def test_outlier_does_not_wash_out_the_diverging_colour_scale(cities):
    vals = {c["name"]: 0.3 * ((i % 5) - 2) for i, c in enumerate(cities)}
    vals["Leh"] = 5.6                                           # one extreme value
    f = maps.value_map(cities, vals, "Temperature anomaly", "°C", diverging=True, warm_is_red=True)
    dots = next(t for t in f.data if t.name == "Value")
    assert dots.marker.cmax < 5.6 and dots.marker.cmin == -dots.marker.cmax and "capped" in f.layout.title.text and "Leh" in f.layout.title.text
    assert not dots.showlegend
    plain = {c["name"]: 0.3 * ((i % 5) - 2) for i, c in enumerate(cities)}
    assert "capped" not in maps.value_map(cities, plain, "t", "°C", diverging=True).layout.title.text      # no cap without an outlier
