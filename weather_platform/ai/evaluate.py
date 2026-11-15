"""Scored evaluation of the Q&A system against an INDEPENDENT answer key.

The answer key is computed here with plain pandas/numpy loops, never with the statistics code used by the Q&A engine,
so a bug in the engine cannot also be a bug in the key. For every question we check:
  * numerical correctness (within a stated tolerance),
  * source consistency (data source, period label and exact date range are present and correct),
  * unsupported claims (numbers in the text that are not in the computed evidence; figures attached to an abstention),
  * correct abstention (forecasts, causes, trends, other locations, unloaded variables, missing/insufficient data, nonsense).
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from weather_platform.ai.qa import answer_question, verification_context
from weather_platform.ai.verify import unsupported_numbers

WET_MM = 1.0
MIN_BASELINE_YEARS = 5   # the specification used by the system under test


@dataclass
class EvalItem:
    id: str
    category: str
    question: str
    expect_status: str                       # 'answered' | 'abstained'
    variant: str = "base"                    # 'base' | 'gappy'
    figures: list = field(default_factory=list)   # [{'name': exact figure name, 'value': x, 'tol': t}]
    date_range: list | None = None
    note: str = ""


# ------------------------------------------------------------------ independent ground truth
def _expected_days(a, b) -> int:
    return (b - a).days + 1


def _valid_frac(s: pd.Series, a, b) -> float:
    return float(s.loc[a:b].notna().sum()) / _expected_days(a, b)


def _month(y: int, m: int):
    a = pd.Timestamp(y, m, 1)
    return a, a + pd.offsets.MonthEnd(0)


def _longest_dry_run(s: pd.Series) -> int:
    best = run = 0
    for v in s.to_numpy(float):
        run = run + 1 if (not np.isnan(v) and v < WET_MM) else 0
        best = max(best, run)
    return best


def _full_years(df: pd.DataFrame) -> list[int]:
    return [y for y in sorted(set(df.index.year)) if pd.Timestamp(y, 1, 1) >= df.index.min() and pd.Timestamp(y, 12, 31) <= df.index.max()]


def _pick(seq: list, k: int) -> list:
    if len(seq) <= k:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, k).round().astype(int)
    return [seq[i] for i in dict.fromkeys(idx)]


def _baseline_truth(s: pd.Series, y: int, m: int, base: tuple, stat: str):
    """Independent baseline z-score and percentile for month m of year y vs the same month in baseline years (excluding y)."""
    vals = []
    for by in range(base[0], base[1] + 1):
        if by == y:
            continue
        a, b = _month(by, m)
        if a > s.index.max() or b < s.index.min() or _valid_frac(s, a, b) < 0.9:
            continue
        seg = s.loc[a:b]
        vals.append(seg.mean() if stat == "mean" else seg.sum())
    arr = np.array(vals, float)
    a, b = _month(y, m)
    if len(arr) < MIN_BASELINE_YEARS or _valid_frac(s, a, b) < 0.9:
        return None
    seg = s.loc[a:b]
    sel = seg.mean() if stat == "mean" else seg.sum()
    z = (sel - arr.mean()) / arr.std(ddof=1)
    pct = 100 * ((arr < sel).sum() + 0.5 * (arr == sel).sum()) / len(arr)
    return {"z": float(z), "pct": float(pct)}


def build_eval_set(df: pd.DataFrame, variables: list[str], loaded_city: str, other_city: str = "Kolkata",
                   baseline_years: tuple[int, int] = (1991, 2020)) -> tuple[list[EvalItem], pd.DataFrame]:
    """Return (items, gappy_df). Expected answers are computed independently from `df`."""
    items: list[EvalItem] = []
    P, T, TX = "precipitation_sum", "temperature_2m_mean", "temperature_2m_max"
    years = _pick(_full_years(df), 4)
    months = [7, 1, 10, 4]
    has_p, has_t, has_tx = P in variables, T in variables, TX in variables
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        items.append(EvalItem(id=f"Q{n:03d}", **kw))

    for i, y in enumerate(years):
        m = months[i % len(months)]
        mname = calendar.month_name[m]
        ya, yb = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        ma, mb = _month(y, m)
        if has_p:
            p = df[P]
            ok = _valid_frac(p, ma, mb) >= 0.9
            add(category="total rainfall", question=f"What was the total rainfall in {mname} {y}?", expect_status="answered" if ok else "abstained",
                figures=[{"name": "total precipitation", "value": round(float(p.loc[ma:mb].sum()), 1), "tol": 0.06}] if ok else [],
                date_range=[str(ma.date()), str(mb.date())] if ok else None)
            ok = _valid_frac(p, ya, yb) >= 0.9
            wet = int((p.loc[ya:yb].dropna() >= WET_MM).sum())
            add(category="wet days", question=f"How many wet days were there in {y}?", expect_status="answered" if ok else "abstained",
                figures=[{"name": "wet days", "value": wet, "tol": 0}] if ok else [], date_range=[str(ya.date()), str(yb.date())] if ok else None)
            add(category="dry spell", question=f"What was the longest dry spell in {y}?", expect_status="answered" if ok else "abstained",
                figures=[{"name": "longest dry spell", "value": _longest_dry_run(p.loc[ya:yb]), "tol": 0}] if ok else [],
                date_range=[str(ya.date()), str(yb.date())] if ok else None)
            ja, jb = pd.Timestamp(y, 6, 1), pd.Timestamp(y, 9, 30)
            ok = _valid_frac(p, ja, jb) >= 0.9
            add(category="wet days (monsoon)", question=f"How many wet days were there in the monsoon of {y}?", expect_status="answered" if ok else "abstained",
                figures=[{"name": "wet days", "value": int((p.loc[ja:jb].dropna() >= WET_MM).sum()), "tol": 0}] if ok else [],
                date_range=[str(ja.date()), str(jb.date())] if ok else None)
            miss = 100 * (_expected_days(ya, yb) - int(p.loc[ya:yb].notna().sum())) / _expected_days(ya, yb)
            add(category="missing data", question=f"How much data is missing for precipitation in {y}?", expect_status="answered",
                figures=[{"name": "missing", "value": round(miss, 2), "tol": 0.006}], date_range=[str(ya.date()), str(yb.date())])
        if has_t:
            t = df[T]
            ok = _valid_frac(t, ya, yb) >= 0.5
            add(category="average temperature", question=f"What was the average temperature in {y}?", expect_status="answered" if ok else "abstained",
                figures=[{"name": "average mean temperature", "value": round(float(t.loc[ya:yb].mean()), 2), "tol": 0.006}] if ok else [],
                date_range=[str(ya.date()), str(yb.date())] if ok else None)
        if has_tx:
            tx = df[TX].loc[ya:yb]
            if _valid_frac(df[TX], ya, yb) >= 0.5 and tx.notna().any():
                add(category="highest temperature", question=f"What was the highest temperature in {y}?", expect_status="answered",
                    figures=[{"name": "highest maximum temperature", "value": round(float(tx.max()), 2), "tol": 0.006}, {"name": "date", "value": str(tx.idxmax().date()), "tol": 0}],
                    date_range=[str(ya.date()), str(yb.date())])
    # all-years month (only when the data covers whole calendar years)
    if has_p and _full_years(df) and df.index.min() == pd.Timestamp(df.index.min().year, 1, 1) and df.index.max() == pd.Timestamp(df.index.max().year, 12, 31):
        sel = df[P][df.index.month == 7].dropna()
        add(category="month across years", question="What was the average rainfall in July?", expect_status="answered",
            figures=[{"name": "average precipitation (daily total)", "value": round(float(sel.mean()), 2), "tol": 0.006}])
    # baseline comparison
    if has_t and years:
        y = years[-1]
        bt = _baseline_truth(df[T], y, 7, baseline_years, "mean")
        a, b = _month(y, 7)
        add(category="baseline comparison", question=f"Was July {y} unusual for temperature compared to normal?",
            expect_status="answered" if bt else "abstained",
            figures=[{"name": "z-score", "value": round(bt["z"], 2), "tol": 0.011}, {"name": "percentile in baseline", "value": round(bt["pct"], 0), "tol": 0.51}] if bt else [],
            date_range=[str(a.date()), str(b.date())] if bt else None)

    # ----- questions that MUST be refused
    yy = years[-1] if years else int(df.index.max().year)
    far = int(df.index.max().year) + 5
    refuse = [("forecast", f"Will it rain tomorrow in {loaded_city}?"), ("forecast", "What will the temperature be next week?"),
              ("forecast", "Predict the rainfall for next month."),
              ("cause", f"Why was July {yy} so wet?"), ("cause", f"What caused the high temperatures in {yy}?"),
              ("trend", "Is the temperature getting warmer over the years?"), ("trend", "Is rainfall increasing over time?"),
              ("other location", f"What was the average temperature in {other_city} in {yy}?"),
              ("outside period", f"What was the total rainfall in {far}?"),
              ("outside period", f"What was the average temperature in {far + 3}?"),
              ("not loaded", f"What was the average wind speed in {yy}?") if "wind_speed_10m_max" not in variables else None,
              ("conflicting request", f"How many wet days in {yy} for temperature?"),
              ("compare locations", "Compare cities for rainfall"),
              ("nonsense", "Tell me a joke"), ("nonsense", "What is the meaning of life?"), ("nonsense", "asdfgh qwerty")]
    for cat, q in [r for r in refuse if r]:
        add(category=f"refuse: {cat}", question=q, expect_status="abstained")

    # ----- missing-data variant (copy of the data with deliberate gaps)
    gappy = df.copy()
    if years and has_p and has_t:
        y0 = years[0]
        a, b = _month(y0, 7)
        gappy.loc[a: a + pd.Timedelta(days=19), P] = np.nan                        # 20 of 31 days missing
        a10, b10 = _month(y0, 10)
        gappy.loc[a10: a10 + pd.Timedelta(days=11), T] = np.nan                    # 12 of 31 days missing -> 61% valid
        seg = gappy[P].loc[a:b]
        add(category="insufficient data", variant="gappy", question=f"What was the total rainfall in July {y0}?", expect_status="abstained")
        add(category="missing data (gappy)", variant="gappy", question=f"How much data is missing for precipitation in July {y0}?", expect_status="answered",
            figures=[{"name": "missing", "value": round(100 * (31 - int(seg.notna().sum())) / 31, 2), "tol": 0.006},
                     {"name": "zero-rain days", "value": int((seg == 0).sum()), "tol": 0}], date_range=[str(a.date()), str(b.date())])
        add(category="partial coverage", variant="gappy", question=f"What was the average temperature in October {y0}?", expect_status="answered",
            figures=[{"name": "average mean temperature", "value": round(float(gappy[T].loc[a10:b10].mean()), 2), "tol": 0.006}], date_range=[str(a10.date()), str(b10.date())],
            note="61% valid: answered with a caution")
    return items, gappy


# ------------------------------------------------------------------ scoring
def make_answer_fn(meta: dict, variables: list[str], loaded_city: str, city_names: list[str], baseline_years=(1991, 2020), wet_mm: float = WET_MM):
    return lambda q, d: answer_question(q, d, meta, variables, wet_mm=wet_mm, baseline_years=baseline_years, city_names=city_names, loaded_city=loaded_city)


def _find_fig(ans, name: str):
    for f in ans.figures:
        if f["name"] == name:
            return f["value"]
    return None


def evaluate(answer_fn, items: list[EvalItem], df_base: pd.DataFrame, df_gappy: pd.DataFrame) -> dict:
    rows = []
    for it in items:
        ans = answer_fn(it.question, df_gappy if it.variant == "gappy" else df_base)
        status_ok = ans.status == it.expect_status
        numeric_ok = source_ok = None
        detail = []
        unsupported = []
        if it.expect_status == "answered":
            if ans.status == "answered":
                numeric_ok = True
                for f in it.figures:
                    got = _find_fig(ans, f["name"])
                    if got is None:
                        numeric_ok = False
                        detail.append(f"missing figure '{f['name']}'")
                    elif isinstance(f["value"], str):
                        if got != f["value"]:
                            numeric_ok = False
                            detail.append(f"'{f['name']}': got {got}, expected {f['value']}")
                    elif abs(float(got) - f["value"]) > f["tol"] + 1e-9:
                        numeric_ok = False
                        detail.append(f"'{f['name']}': got {got}, expected {f['value']} (tol {f['tol']})")
                source_ok = bool(ans.source) and bool(ans.period_label) and (it.date_range is None or list(ans.date_range) == it.date_range)
                if not source_ok:
                    detail.append(f"source/date range wrong: {ans.date_range} vs {it.date_range}")
                unsupported = unsupported_numbers(ans.text, ans.figures, verification_context(ans))
            else:
                detail.append("abstained but the answer key says it is answerable")
        else:
            if ans.status == "answered":
                detail.append("answered a question that must be refused")
            if ans.figures:
                unsupported = ["figures attached to an abstention"]
        passed = status_ok and (numeric_ok is not False) and (source_ok is not False) and not unsupported
        rows.append({"id": it.id, "category": it.category, "question": it.question, "expected": it.expect_status, "got": ans.status,
                     "status_ok": status_ok, "numeric_ok": numeric_ok, "source_ok": source_ok, "unsupported_numbers": len(unsupported),
                     "passed": passed, "detail": "; ".join(detail), "answer_text": ans.text})
    res = pd.DataFrame(rows)
    ans_exp = res[res["expected"] == "answered"]
    ref_exp = res[res["expected"] == "abstained"]
    got_ans = ans_exp[ans_exp["got"] == "answered"]
    metrics = {
        "n_questions": int(len(res)), "n_answerable": int(len(ans_exp)), "n_must_refuse": int(len(ref_exp)),
        "answerable_answered_rate": float(len(got_ans) / len(ans_exp)) if len(ans_exp) else None,
        "numerical_accuracy": float(got_ans["numeric_ok"].mean()) if len(got_ans) else None,
        "source_consistency": float(got_ans["source_ok"].mean()) if len(got_ans) else None,
        "correct_abstention_rate": float((ref_exp["got"] == "abstained").mean()) if len(ref_exp) else None,
        "false_answer_rate": float((ref_exp["got"] == "answered").mean()) if len(ref_exp) else None,
        "unsupported_claim_rate": float((res["unsupported_numbers"] > 0).mean()),
        "overall_pass_rate": float(res["passed"].mean()),
    }
    return {"results": res, "metrics": metrics, "by_category": res.groupby("category")["passed"].agg(["sum", "count"]).rename(columns={"sum": "passed", "count": "total"})}


def report_markdown(metrics: dict, by_category: pd.DataFrame, title: str, source_note: str) -> str:
    pc = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"# Q&A evaluation: {title}", "", f"Data: {source_note}", "",
             "The answer key is computed independently with plain pandas (not with the engine's own statistics code).", "",
             "| Metric | Value |", "|---|---|",
             f"| Questions | {metrics['n_questions']} ({metrics['n_answerable']} answerable, {metrics['n_must_refuse']} that must be refused) |",
             f"| Answerable questions answered | {pc(metrics['answerable_answered_rate'])} |", f"| Numerical accuracy (of those answered) | {pc(metrics['numerical_accuracy'])} |",
             f"| Source and date-range consistency | {pc(metrics['source_consistency'])} |", f"| Correct abstention on must-refuse questions | {pc(metrics['correct_abstention_rate'])} |",
             f"| False answers (should have refused) | {pc(metrics['false_answer_rate'])} |", f"| Unsupported-claim rate | {pc(metrics['unsupported_claim_rate'])} |",
             f"| Overall pass rate | {pc(metrics['overall_pass_rate'])} |", "", "## By category", "", "| Category | Passed | Total |", "|---|---|---|"]
    lines += [f"| {c} | {int(r.passed)} | {int(r.total)} |" for c, r in by_category.iterrows()]
    lines += ["", "Limits: questions are generated from templates and the parser is rule-based, so this measures correctness on the supported question forms, "
              "not general language understanding."]
    return "\n".join(lines)
