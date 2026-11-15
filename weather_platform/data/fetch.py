"""Fetch daily historical weather from Open-Meteo (ERA5) with on-disk caching.

The request is validated BEFORE any network call. Responses are parsed into a
date-indexed DataFrame; JSON nulls become NaN (missing), never 0.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

from weather_platform import config
from weather_platform.quality.checks import validate_request


class FetchError(RuntimeError):
    pass


def build_params(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, variables: list[str]) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ",".join(variables),
        "timezone": "auto",
    }


def parse_response(payload: dict, variables: list[str]) -> pd.DataFrame:
    """Turn the API JSON into a date-indexed DataFrame; units stored in df.attrs['units']."""
    if "daily" not in payload or "time" not in payload["daily"]:
        reason = payload.get("reason", "response had no 'daily' block")
        raise FetchError(f"Unexpected API response: {reason}")
    daily = payload["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for v in variables:
        # None -> NaN. A missing measurement must never silently become 0.
        df[v] = pd.to_numeric(pd.Series(daily.get(v, [None] * len(df))), errors="coerce")
    df = df.set_index("date")
    df.attrs["units"] = dict(payload.get("daily_units", {}))
    return df


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.csv", cache_dir / f"{key}.meta.json"


def make_cache_key(lat, lon, start, end, variables) -> str:
    raw = json.dumps([round(lat, 4), round(lon, 4), str(start.date()), str(end.date()), sorted(variables)])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def fetch_daily(
    lat: float,
    lon: float,
    start,
    end,
    variables: list[str],
    cache_dir: str | Path = ".cache",
    use_cache: bool = True,
    session=None,
    retries: int = 3,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (raw_df, meta). meta records source, URL, fetch time and cache status."""
    s, e = validate_request(start, end, variables, today=today)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = make_cache_key(lat, lon, s, e, variables)
    csv_path, meta_path = _cache_paths(cache_dir, key)

    if use_cache and csv_path.exists() and meta_path.exists():
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        meta = json.loads(meta_path.read_text())
        df.attrs["units"] = meta["units"]
        meta["from_cache"] = True
        return df, meta

    session = session or requests
    params = build_params(lat, lon, s, e, variables)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(config.DATA_SOURCE["api_url"], params=params, timeout=60)
            if resp.status_code != 200:
                raise FetchError(f"HTTP {resp.status_code}: {getattr(resp, 'text', '')[:200]}")
            df = parse_response(resp.json(), variables)
            break
        except (requests.RequestException, FetchError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 8) if retries > 1 else 0)
    else:
        raise FetchError(f"Could not fetch data after {retries} attempts: {last_err}")

    meta = {
        "source": config.DATA_SOURCE["name"],
        "source_url": config.DATA_SOURCE["url"],
        "data_type": config.DATA_SOURCE["type"],
        "latitude": lat,
        "longitude": lon,
        "start": str(s.date()),
        "end": str(e.date()),
        "variables": variables,
        "units": df.attrs["units"],
        "fetched_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "from_cache": False,
    }
    df.to_csv(csv_path, index_label="date")
    meta_path.write_text(json.dumps(meta, indent=2))
    return df, meta


def load_from_cache(lat: float, lon: float, start, end, variables: list[str], cache_dir: str | Path = ".cache"):
    """Return (df, meta) if this exact request is already cached, else None. Never touches the network."""
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    csv_path, meta_path = _cache_paths(Path(cache_dir), make_cache_key(lat, lon, s, e, variables))
    if not (csv_path.exists() and meta_path.exists()):
        return None
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    meta = json.loads(meta_path.read_text())
    df.attrs["units"] = meta["units"]
    meta["from_cache"] = True
    return df, meta
