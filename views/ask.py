import pandas as pd
import streamlit as st

from weather_platform import config, state, ui
from weather_platform.aggregate import aggregate
from weather_platform.ai.evaluate import build_eval_set, evaluate, make_answer_fn
from weather_platform.ai.qa import EXAMPLES, answer_question
from weather_platform.data.cities import load_cities
from weather_platform.viz import charts

ui.page_header("Ask the data", "Ask a question in plain English. The answer is calculated from the data, shown with its evidence, and refused when the data cannot support it.")
D = state.require_loaded()
df, meta, VARS, name = D["df"], D["meta"], D["variables"], D["city"]["name"]
S = state.settings()
b = state.baseline()
cities = load_cities()
ui.chips([(f"Asking about {meta['location']}", ""), (f"{meta['start']} to {meta['end']}", ""), ("Answers are computed by code, not guessed", "good")])

ui.explainer("A question box connected to the data you loaded on the Location analysis page.",
             "Name a statistic (total, average, highest, wet days, dry spell, missing data, unusual compared with normal) and a period (a year, a month, the monsoon). Try the examples below.")
st.markdown("**Try an example**")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES[:6]):
    cols[i % 3].button(ex, key=f"ex_{i}", on_click=lambda e=ex: st.session_state.update(question=e), width="stretch")
question = st.text_input("Your question", key="question", placeholder="For example: What was the total rainfall in July 2022?  (press Enter or click Ask)",
                         on_change=lambda: st.session_state.update(_ask_trigger=True))
_clicked = st.button("Ask", key="ask", type="primary")
_enter = st.session_state.pop("_ask_trigger", False)
if (_clicked or _enter) and question.strip():
    ans = answer_question(question, df, meta, VARS, wet_mm=S["wet_mm"], baseline_years=b, city_names=[c["name"] for c in cities], loaded_city=name)
    if ans.status == "answered":
        st.success(ans.text)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Evidence**")
            st.dataframe(pd.DataFrame(ans.figures).rename(columns={"name": "Figure", "value": "Value", "unit": "Unit"}), width="stretch", hide_index=True)
            st.markdown(f"**Period:** {ans.period_label}  \n**Date range used:** {ans.date_range[0]} to {ans.date_range[1]}  \n"
                        f"**Measurement:** {config.VARIABLES[ans.variable]['label']} ({config.VARIABLES[ans.variable]['unit']})  \n**Data source:** {ans.source}")
            for w in ans.warnings:
                st.caption("Note: " + w)
        with c2:
            if ans.chart:
                sl = df.loc[ans.chart["start"]:ans.chart["end"]]
                if ans.chart["kind"] == "rainfall":
                    fq = "daily" if (pd.Timestamp(ans.chart["end"]) - pd.Timestamp(ans.chart["start"])).days <= 400 else "monthly"
                    st.plotly_chart(charts.rainfall_chart(aggregate(sl, ["precipitation_sum"], fq), fq), width="stretch", key="ask_chart")
                else:
                    st.plotly_chart(charts.temperature_trend(sl[[ans.chart["variable"]]]), width="stretch", key="ask_chart")
                st.caption("The chart shows the full span of the question; the figures use only the period named on the left.")
    else:
        st.warning(ans.text)
        st.caption("Declining is intentional: a wrong but confident answer is worse than no answer.")

with st.expander("What the assistant will and will not answer"):
    c1, c2 = st.columns(2)
    c1.markdown("**It answers**\n\n- Totals, averages, highest and lowest values\n- Wet days and dry spells\n- How much data is missing\n- Whether a month or year was unusual compared with the normal")
    c2.markdown("**It declines**\n\n- Forecasts (it only knows the past)\n- Causes (\"why\")\n- Trend claims (use the trend tool, which tests them properly)\n- Other places than the one loaded, or periods outside the data\n- Periods with too much missing data")
with st.expander("How answers are produced and checked"):
    st.markdown("1. The question is parsed by fixed rules into *what statistic, which measurement, which period*.\n"
                "2. The number is computed from the loaded data with the same tested code used everywhere in this platform.\n"
                "3. The wording is filled from a template and every number in it is re-checked against the computed evidence.\n"
                "4. A language model can optionally be added only to reword the answer; its text is rejected if it contains any number that is not in the evidence.")
with st.expander("Evaluate this assistant on the loaded data (independent answer key)"):
    st.caption("Generates questions from the loaded data, computes the correct answers independently with plain pandas, and scores the assistant on numerical accuracy, "
               "source consistency, unsupported claims and correct refusals.")
    if st.button("Run evaluation", key="run_eval"):
        items, gappy = build_eval_set(df, VARS, name, "Kolkata" if name != "Kolkata" else "Delhi", b)
        st.session_state["eval_out"] = evaluate(make_answer_fn(meta, VARS, name, [c["name"] for c in cities], b, S["wet_mm"]), items, df, gappy)
    o = st.session_state.get("eval_out")
    if o:
        m = o["metrics"]
        pc = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Numerical accuracy", pc(m["numerical_accuracy"]))
        e2.metric("Correct refusals", pc(m["correct_abstention_rate"]))
        e3.metric("Unsupported claims", pc(m["unsupported_claim_rate"]))
        e4.metric("Overall pass", pc(m["overall_pass_rate"]))
        st.dataframe(o["by_category"], width="stretch")
        bad = o["results"][~o["results"]["passed"]]
        if len(bad):
            st.warning(f"{len(bad)} question(s) failed:")
            st.dataframe(bad[["id", "question", "detail"]], width="stretch", hide_index=True)
        else:
            st.success(f"All {m['n_questions']} questions passed. This covers the supported question forms only; it is not a general language test.")
