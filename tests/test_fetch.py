import numpy as np
import pandas as pd
import pytest

from weather_platform.data.cities import get_city, search_cities
from weather_platform.data.fetch import FetchError, build_params, fetch_daily, parse_response

TODAY = pd.Timestamp("2026-10-26")
VARS = ["temperature_2m_mean", "precipitation_sum"]

PAYLOAD = {
    "daily_units": {"time": "iso8601", "temperature_2m_mean": "°C", "precipitation_sum": "mm"},
    "daily": {
        "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "temperature_2m_mean": [20.1, None, 21.0],
        "precipitation_sum": [0.0, None, 4.2],
    },
}


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, "err"

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.resp


def test_parse_response_nulls_become_nan_and_zero_stays_zero():
    df = parse_response(PAYLOAD, VARS)
    assert df["precipitation_sum"].iloc[0] == 0.0
    assert np.isnan(df["precipitation_sum"].iloc[1])
    assert df.attrs["units"]["precipitation_sum"] == "mm"


def test_parse_response_error_payload():
    with pytest.raises(FetchError, match="bad thing"):
        parse_response({"error": True, "reason": "bad thing"}, VARS)


def test_build_params():
    p = build_params(26.1, 91.7, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"), VARS)
    assert p["daily"] == "temperature_2m_mean,precipitation_sum" and p["start_date"] == "2020-01-01"


def test_fetch_then_cache(tmp_path):
    sess = FakeSession(FakeResp(PAYLOAD))
    df1, meta1 = fetch_daily(26.1, 91.7, "2020-01-01", "2020-01-03", VARS, tmp_path, session=sess, today=TODAY)
    assert sess.calls == 1 and meta1["from_cache"] is False
    df2, meta2 = fetch_daily(26.1, 91.7, "2020-01-01", "2020-01-03", VARS, tmp_path, session=sess, today=TODAY)
    assert sess.calls == 1 and meta2["from_cache"] is True  # no second network call
    assert df2["precipitation_sum"].iloc[0] == 0.0 and np.isnan(df2["precipitation_sum"].iloc[1])
    assert df2.attrs["units"]["temperature_2m_mean"] == "°C"
    assert "Open-Meteo" in meta2["source"]


def test_invalid_request_never_hits_network(tmp_path):
    sess = FakeSession(FakeResp(PAYLOAD))
    with pytest.raises(ValueError):
        fetch_daily(26.1, 91.7, "2020-02-01", "2020-01-01", VARS, tmp_path, session=sess, today=TODAY)
    assert sess.calls == 0


def test_http_error_raises_after_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    sess = FakeSession(FakeResp({}, status=500))
    with pytest.raises(FetchError, match="3 attempts"):
        fetch_daily(26.1, 91.7, "2020-01-01", "2020-01-03", VARS, tmp_path, session=sess, today=TODAY)
    assert sess.calls == 3


def test_city_search_and_lookup():
    assert [c["name"] for c in search_cities("guwa")] == ["Guwahati"]
    assert get_city("kolkata")["country"] == "India"
    with pytest.raises(KeyError):
        get_city("Atlantis")
