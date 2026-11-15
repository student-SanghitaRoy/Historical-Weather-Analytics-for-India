"""Score the Q&A system against an independent answer key, on REAL downloaded data.

  python scripts/evaluate_qa.py                       # a default set of diverse, already-downloaded cities
  python scripts/evaluate_qa.py --cities Guwahati,Delhi
  python scripts/evaluate_qa.py --synthetic           # offline demo on synthetic data (NOT real)

Writes evaluation/qa_results.csv and evaluation/qa_report.md next to the project.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from weather_platform.ai.evaluate import build_eval_set, evaluate, make_answer_fn, report_markdown  # noqa: E402
from weather_platform.data.cities import get_city, load_cities  # noqa: E402
from weather_platform.india import CACHE_DIR, PREFETCH_END, PREFETCH_START, load_datasets, synthetic_datasets  # noqa: E402

DEFAULT = ["Guwahati", "Delhi", "Jaisalmer", "Cherrapunji", "Chennai", "Leh"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=",".join(DEFAULT))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "evaluation"))
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.cities.split(",") if n.strip()]
    cities = [get_city(n) for n in names]
    if a.synthetic:
        data, missing = synthetic_datasets(cities), []
        src = "SYNTHETIC data (not real weather)"
    else:
        data, missing = load_datasets(cities)
        src = f"ERA5 reanalysis via Open-Meteo, cached in {CACHE_DIR} ({PREFETCH_START} to {PREFETCH_END})"
    if missing:
        print("Not downloaded (skipped):", ", ".join(missing))
    all_rows, summaries = [], []
    all_names = [c["name"] for c in load_cities()]
    for c in cities:
        if c["name"] not in data:
            continue
        df, variables = data[c["name"]], ["temperature_2m_mean", "precipitation_sum"]
        meta = {"source": src.split(" (")[0], "location": f"{c['name']}, {c['state']}", "start": PREFETCH_START, "end": PREFETCH_END}
        other = "Kolkata" if c["name"] != "Kolkata" else "Delhi"
        items, gappy = build_eval_set(df, variables, c["name"], other)
        out = evaluate(make_answer_fn(meta, variables, c["name"], all_names), items, df, gappy)
        r = out["results"].copy()
        r.insert(0, "city", c["name"])
        all_rows.append(r)
        summaries.append({"city": c["name"], **out["metrics"]})
        m = out["metrics"]
        print(f"{c['name']:<14} questions {m['n_questions']:>3}  numeric accuracy {m['numerical_accuracy']:.1%}  correct abstention {m['correct_abstention_rate']:.1%}  "
              f"unsupported claims {m['unsupported_claim_rate']:.1%}  overall {m['overall_pass_rate']:.1%}")
        for _, bad in r[~r["passed"]].iterrows():
            print(f"    FAIL {bad['id']}: {bad['question']} -> {bad['detail']}")
    if not all_rows:
        print("No data available to evaluate. Download data first (download_data.bat) or use --synthetic.")
        return 1
    res = pd.concat(all_rows, ignore_index=True)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "qa_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "qa_summary_by_city.csv", index=False)
    total_metrics = {"n_questions": len(res), "n_answerable": int((res.expected == "answered").sum()), "n_must_refuse": int((res.expected == "abstained").sum())}
    ans, ref = res[res.expected == "answered"], res[res.expected == "abstained"]
    got = ans[ans.got == "answered"]
    total_metrics.update(answerable_answered_rate=len(got) / len(ans) if len(ans) else None, numerical_accuracy=got["numeric_ok"].mean() if len(got) else None,
                         source_consistency=got["source_ok"].mean() if len(got) else None, correct_abstention_rate=(ref.got == "abstained").mean() if len(ref) else None,
                         false_answer_rate=(ref.got == "answered").mean() if len(ref) else None, unsupported_claim_rate=(res.unsupported_numbers > 0).mean(),
                         overall_pass_rate=res.passed.mean())
    by_cat = res.groupby("category")["passed"].agg(["sum", "count"]).rename(columns={"sum": "passed", "count": "total"})
    md = report_markdown(total_metrics, by_cat, ", ".join(sorted(res.city.unique())), src)
    (out_dir / "qa_report.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {out_dir / 'qa_results.csv'}, qa_summary_by_city.csv and qa_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
