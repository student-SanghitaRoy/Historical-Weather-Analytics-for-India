"""CSV export. Missing values are written as EMPTY cells (never 0). A '#' header records the context."""
from __future__ import annotations

import io

import pandas as pd

from weather_platform import config


def _header(meta: dict, extra: list[str]) -> list[str]:
    lines = [
        f"Source: {meta.get('source', config.DATA_SOURCE['name'])}",
        f"Source URL: {meta.get('source_url', config.DATA_SOURCE['url'])}",
        f"Data type: {meta.get('data_type', config.DATA_SOURCE['type'])}",
        f"Location: {meta.get('location', 'n/a')} (lat {meta.get('latitude')}, lon {meta.get('longitude')})",
        f"Period: {meta.get('start')} to {meta.get('end')}",
        f"Fetched (UTC): {meta.get('fetched_at_utc', 'n/a')}",
        "Missing values are empty cells, NOT zero. Historical reanalysis; not a forecast.",
    ] + extra
    return [f"# {l}" for l in lines]


def daily_csv_bytes(df: pd.DataFrame, meta: dict) -> bytes:
    out = df.copy()
    out.columns = [f"{c} ({config.VARIABLES[c]['unit']})" for c in out.columns]
    buf = io.StringIO()
    buf.write("\n".join(_header(meta, ["Resolution: daily"])) + "\n")
    out.to_csv(buf, index_label="date", na_rep="")
    return buf.getvalue().encode("utf-8-sig")


def aggregated_csv_bytes(agg: pd.DataFrame, meta: dict, freq: str, min_valid_frac: float) -> bytes:
    out = agg[["period_label", "variable", "value", "stat", "unit", "n_valid", "n_expected", "complete"]].copy()
    out["value"] = out["value"].round(2)          # no floating-point noise such as 15.899999999999999
    buf = io.StringIO()
    note = [f"Resolution: {freq}", f"Periods with < {min_valid_frac:.0%} valid days are incomplete and have an empty value."]
    buf.write("\n".join(_header(meta, note)) + "\n")
    out.to_csv(buf, index=False, na_rep="")
    return buf.getvalue().encode("utf-8-sig")
