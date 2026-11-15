"""Data-quality checks and cleaning for daily weather data.

Design rules
------------
1. Missing is not zero. Missing values are NaN everywhere. We never fill with 0.
   A day with 0.0 mm of rain is a real measurement; a NaN is "no valid value".
2. Impossible values become NaN (missing), never clipped or zeroed.
3. Every action is recorded in `QualityReport.steps` so the UI and the PDF can show
   the processing steps verbatim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from weather_platform import config


# --------------------------------------------------------------------------- #
# Request validation (runs BEFORE any network call)
# --------------------------------------------------------------------------- #
def validate_request(
    start,
    end,
    variables: list[str],
    today: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate the user's date range and variable choice. Returns normalised dates.

    Raises ValueError with ALL problems listed, so the UI can show them together.
    """
    today = (today or pd.Timestamp.today()).normalize()
    latest = today - pd.Timedelta(days=config.DATA_LATENCY_DAYS)
    problems: list[str] = []

    try:
        s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    except (ValueError, TypeError):
        raise ValueError(f"Could not parse dates: start={start!r}, end={end!r}")

    if s > e:
        problems.append(f"Start date {s.date()} is after end date {e.date()}.")
    if s < config.DATA_START:
        problems.append(f"Start date {s.date()} is before the dataset begins ({config.DATA_START.date()}).")
    if e > latest:
        problems.append(
            f"End date {e.date()} is later than the latest available data (~{latest.date()}); "
            "recent days are not yet published."
        )
    if not variables:
        problems.append("Select at least one weather variable.")
    unknown = [v for v in variables if v not in config.VARIABLES]
    if unknown:
        problems.append(f"Unsupported variable(s): {unknown}. Supported: {list(config.VARIABLES)}.")

    if problems:
        raise ValueError(" ".join(problems))
    return s, e


# --------------------------------------------------------------------------- #
# Report object
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    start: str = ""
    end: str = ""
    n_expected_days: int = 0
    n_rows_raw: int = 0
    out_of_range_rows_dropped: int = 0
    duplicate_rows_removed: int = 0
    conflicting_duplicate_dates: int = 0
    missing_dates_added: int = 0
    impossible_values: dict = field(default_factory=dict)  # var -> count set to missing
    inconsistent_temperature_rows: int = 0
    unit_issues: list = field(default_factory=list)
    per_variable: dict = field(default_factory=dict)  # var -> stats dict
    steps: list = field(default_factory=list)

    @property
    def units_ok(self) -> bool:
        return not self.unit_issues

    def to_dict(self) -> dict:
        d = asdict(self)
        d["units_ok"] = self.units_ok
        return d

    def summary_table(self) -> pd.DataFrame:
        """One row per variable: what the dashboard / PDF shows as 'data completeness'."""
        rows = []
        for var, st in self.per_variable.items():
            rows.append({"variable": var, **st})
        return pd.DataFrame(rows)


def longest_missing_run(s: pd.Series) -> int:
    """Longest consecutive run of NaN values (in days, since the index is daily)."""
    isna = s.isna().to_numpy()
    best = cur = 0
    for v in isna:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_daily(
    df: pd.DataFrame,
    variables: list[str],
    start,
    end,
    units: dict | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate and clean a daily DataFrame indexed by date.

    Returns (clean_df, report). `clean_df` has exactly one row per calendar day in
    [start, end]; days with no valid value are NaN (never 0).
    """
    rep = QualityReport()
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    rep.start, rep.end = str(s.date()), str(e.date())
    full_index = pd.date_range(s, e, freq="D", name="date")
    rep.n_expected_days = len(full_index)
    rep.n_rows_raw = len(df)

    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out.index.name = "date"
    rep.steps.append(f"1. Loaded {rep.n_rows_raw} raw rows; expected {rep.n_expected_days} daily rows for {s.date()} to {e.date()}.")

    # --- units -------------------------------------------------------------
    units = units or dict(getattr(df, "attrs", {}).get("units", {}))
    for var in variables:
        expected = config.VARIABLES[var]["unit"]
        got = units.get(var)
        if got is None:
            rep.unit_issues.append(f"{var}: no unit information supplied (expected {expected}).")
        elif got != expected:
            rep.unit_issues.append(f"{var}: unit is {got!r}, expected {expected!r}. Values were NOT converted.")
    if rep.unit_issues:
        rep.steps.append("2. Unit validation FAILED: " + " | ".join(rep.unit_issues))
    else:
        rep.steps.append("2. Unit validation passed: " + ", ".join(f"{v} in {config.VARIABLES[v]['unit']}" for v in variables) + ".")

    # --- keep only requested columns that exist -----------------------------
    absent = [v for v in variables if v not in out.columns]
    for v in absent:
        out[v] = np.nan
    if absent:
        rep.steps.append(f"   Columns absent from source and treated as fully missing: {absent}.")
    out = out[variables]

    # --- rows outside the requested range ------------------------------------
    outside = ~out.index.isin(full_index) & ~out.index.isna()
    rep.out_of_range_rows_dropped = int(outside.sum())
    out = out[~outside]

    # --- duplicates ----------------------------------------------------------
    dup_mask = out.index.duplicated(keep=False)
    if dup_mask.any():
        grp = out[dup_mask].groupby(level=0).nunique(dropna=False)
        rep.conflicting_duplicate_dates = int((grp > 1).any(axis=1).sum())
    n_before = len(out)
    out = out[~out.index.duplicated(keep="first")]
    rep.duplicate_rows_removed = n_before - len(out)
    rep.steps.append(
        f"3. Duplicates: removed {rep.duplicate_rows_removed} duplicate-date row(s) (kept first); "
        f"{rep.conflicting_duplicate_dates} duplicated date(s) had conflicting values."
    )

    # --- gaps in the date index (reindex => NaN, never 0) --------------------
    out = out.sort_index()
    rep.missing_dates_added = int((~full_index.isin(out.index)).sum())
    out = out.reindex(full_index)
    rep.steps.append(f"4. Date continuity: {rep.missing_dates_added} calendar day(s) had no record; added as missing (NaN), not zero.")

    # --- impossible values -> NaN ----------------------------------------------
    for var in variables:
        lo, hi = config.VARIABLES[var]["min"], config.VARIABLES[var]["max"]
        bad = out[var].notna() & ((out[var] < lo) | (out[var] > hi))
        n_bad = int(bad.sum())
        if n_bad:
            rep.impossible_values[var] = n_bad
            out.loc[bad, var] = np.nan
    if rep.impossible_values:
        rep.steps.append(f"5. Plausibility: set impossible values to missing (outside physical bounds): {rep.impossible_values}.")
    else:
        rep.steps.append("5. Plausibility: no values outside physical bounds.")

    # --- temperature consistency (min <= mean <= max) ------------------------
    tmin, tmean, tmax = "temperature_2m_min", "temperature_2m_mean", "temperature_2m_max"
    bad_rows = pd.Series(False, index=out.index)
    if tmin in out and tmax in out:
        bad_rows |= out[tmin].notna() & out[tmax].notna() & (out[tmin] > out[tmax])
    if tmean in out and tmin in out:
        bad_rows |= out[tmean].notna() & out[tmin].notna() & (out[tmean] < out[tmin] - 1e-6)
    if tmean in out and tmax in out:
        bad_rows |= out[tmean].notna() & out[tmax].notna() & (out[tmean] > out[tmax] + 1e-6)
    rep.inconsistent_temperature_rows = int(bad_rows.sum())
    if rep.inconsistent_temperature_rows:
        for c in (tmin, tmean, tmax):
            if c in out:
                out.loc[bad_rows, c] = np.nan
    rep.steps.append(
        f"6. Temperature consistency (min <= mean <= max): {rep.inconsistent_temperature_rows} inconsistent row(s) set to missing."
    )

    # --- completeness per variable ---------------------------------------------
    for var in variables:
        n_valid = int(out[var].notna().sum())
        n_missing = rep.n_expected_days - n_valid
        st = {
            "label": config.VARIABLES[var]["label"],
            "unit": config.VARIABLES[var]["unit"],
            "n_expected": rep.n_expected_days,
            "n_valid": n_valid,
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / rep.n_expected_days, 2) if rep.n_expected_days else 0.0,
            "longest_missing_run_days": longest_missing_run(out[var]),
        }
        if config.VARIABLES[var]["kind"] == "precipitation":
            st["n_zero_days"] = int((out[var] == 0).sum())  # REAL zero rainfall
        rep.per_variable[var] = st
    rep.steps.append("7. Completeness computed per variable. Zero-rain days are reported separately from missing days.")

    out.attrs["units"] = {v: config.VARIABLES[v]["unit"] for v in variables}
    return out, rep
