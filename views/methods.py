
import pandas as pd
import streamlit as st

from weather_platform import config, state, ui

ui.page_header("Methods and trust", "How every number is produced, what the platform cannot tell you, and how it was checked. Written for reviewers as much as users.")

t_data, t_stats, t_ai, t_limits, t_gloss = st.tabs(["Data and quality", "Statistics", "The AI assistant", "Limitations and ethics", "Glossary"])

with t_data:
    st.subheader("Where the data comes from")
    st.markdown(f"**Source:** {config.DATA_SOURCE['name']}  \n**URL:** {config.DATA_SOURCE['url']}  \n**Type:** {config.DATA_SOURCE['type']}  \n**Licence note:** {config.DATA_SOURCE['licence_note']}")
    ui.callout("warn", "Reanalysis is not a weather station",
               "ERA5 blends a weather model with observations on a grid of about 25 km. Each location here is the grid cell around the city centre, so hills, coasts and city effects are smoothed. "
               "Rainfall in particular can differ from station records, and reanalysis tends to produce many light-rain days.")
    st.subheader("What happens to the data before any number is calculated")
    st.markdown("1. The requested dates and variables are validated before anything is downloaded.  \n2. Units are checked against the expected unit.  \n"
                "3. Duplicate dates are removed and missing calendar days are added as missing.  \n4. Values outside physical limits are set to missing.  \n"
                "5. Temperature minimum, mean and maximum must be consistent.  \n6. Completeness is reported per variable, and real zero-rain days are counted separately from missing days.")
    ui.callout("good", "The key rule", "Missing is never turned into zero. A period with too little valid data is left blank instead of being guessed.")

with t_stats:
    st.subheader("Definitions used throughout")
    st.dataframe(pd.DataFrame([
        ("Wet day", f"A day with at least {config.WET_DAY_MM:g} mm of precipitation (adjustable)."),
        ("Dry spell", "Consecutive valid days below the wet-day threshold. A missing day ends a spell."),
        ("Complete period", "A month, season or year with at least 90% valid days. Otherwise it is left blank."),
        ("Seasons", "Meteorological: DJF, MAM, JJA, SON. December belongs to the next year's DJF."),
        ("Baseline", "The 'normal' years used for comparison (default 1991 to 2020). One value per year over the same calendar window."),
        ("Comparison with the baseline", "One year: z-score, percentile and a t-based 95% range. Several years: Welch t-test, Cohen's d and a bootstrap interval."),
        ("Unusual temperature day", "Robust z-score (median and MAD for the same time of year) of at least 3.5 in size."),
        ("Extreme rain day", "More than the 99th percentile of baseline wet-day amounts for that time of year."),
        ("Trend", "Mann-Kendall test on annual values with Hamed-Rao autocorrelation correction; Sen's slope with a confidence interval."),
        ("Many locations", "p-values are adjusted with Benjamini-Hochberg so that testing 71 places does not inflate 'significant'."),
        ("Climate regimes", "KMeans on 24 monthly values per location, k chosen by silhouette score, checked with Ward clustering and random restarts."),
    ], columns=["Term", "Definition"]), width="stretch", hide_index=True)
    st.subheader("Why annual values")
    st.write("Daily weather values are strongly related to the days before them, so treating them as independent would overstate certainty. Tests and comparisons therefore use one number per year.")
    ui.callout("info", "A design decision worth knowing", "Trend-free pre-whitening was tried first for autocorrelation. A simulation test showed it made false positives worse, so it was replaced by the Hamed-Rao correction, which is checked against an independent implementation.")

with t_ai:
    st.subheader("An assistant that is allowed to say no")
    st.markdown("The assistant never lets a language model produce numbers. The question is read by fixed rules; the answer is computed from the loaded data with the same tested code as the rest of the platform; "
                "the wording is filled from a template; and every number in the text is checked against the computed evidence. It declines forecasts, causes, unsupported trend claims, other locations, "
                "unloaded measurements, periods outside the data, and periods with too much missing data.")
    st.subheader("How it was evaluated")
    st.write("Questions are generated from the data and their correct answers are computed independently with plain pandas, not with the assistant's own code. Each answer is scored for numerical correctness, "
             "source and date-range consistency, unsupported numbers, and correct refusal. The evaluator itself is tested by feeding it deliberately wrong, hallucinated and padded answers.")
    summ = state.PROJECT_ROOT / "evaluation" / "qa_summary_by_city.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        show = s[["city", "n_questions", "numerical_accuracy", "correct_abstention_rate", "unsupported_claim_rate", "overall_pass_rate"]].copy()
        for c in show.columns[2:]:
            show[c] = (100 * show[c]).round(1).astype(str) + "%"
        st.markdown("**Latest results on the downloaded data**")
        st.dataframe(show.rename(columns={"city": "City", "n_questions": "Questions", "numerical_accuracy": "Numerical accuracy", "correct_abstention_rate": "Correct refusals",
                                          "unsupported_claim_rate": "Unsupported claims", "overall_pass_rate": "Overall"}), width="stretch", hide_index=True)
    else:
        st.caption("Run `python scripts/evaluate_qa.py` to produce the evaluation results; they will appear here.")
    ui.callout("warn", "What the score does and does not mean", "Questions come from templates and the parser is rule-based, so the score covers the supported question forms only. It shows the code applies its definitions correctly, "
               "not that the definitions are the only reasonable ones and not that the assistant understands free language.")

with t_limits:
    st.subheader("Limitations")
    for t in ["The data are reanalysis for a ~25 km grid cell, not station measurements.", "Rainfall totals and wet-day counts depend on the threshold and on reanalysis biases.",
              "Baseline comparisons assume roughly independent, approximately normal years; rainfall is skewed, so the percentile is more reliable than the z-score.",
              "Trends in reanalysis can partly reflect changes in the observations assimilated over time; a significant trend is not proof of a cause.",
              "A short record gives wide intervals. Only about 35 years are available here.", "Results depend on analyst choices (baseline years, threshold, completeness rule); they are stated so they can be changed and challenged.",
              "The platform analyses the past. It does not forecast."]:
        st.write("- " + t)
    st.subheader("Responsible use")
    for t in ["Not for safety-critical decisions. For warnings and forecasts use the India Meteorological Department.", "No personal data is collected or stored; location queries are not logged.",
              "Data attribution and licences are shown (Open-Meteo CC BY 4.0, Copernicus/ECMWF ERA5, GeoNames CC BY 4.0).",
              "Uncertainty is shown wherever a difference or trend is reported, and the assistant declines rather than guesses.", "Analyses can be repeated exactly from a saved settings file."]:
        st.write("- " + t)

with t_gloss:
    gl = {"ERA5": "A global reanalysis from Copernicus/ECMWF that combines a weather model with observations on a regular grid.",
          "Reanalysis": "A consistent re-creation of past weather from a fixed model and the observations available.", "Baseline": "The 'normal' years that other periods are compared with.",
          "Anomaly": "How far a value is from what is normal for that time of year.", "Robust z-score": "A z-score using the median and MAD, so extreme values do not distort it.",
          "Percentile": "The share of baseline years that were lower. 50 is typical.", "Sen's slope": "The median of all pairwise slopes; a robust estimate of change per year.",
          "Mann-Kendall test": "A rank-based test for a monotonic trend that does not assume a straight line or normal data.",
          "False discovery rate": "The expected share of 'significant' results that are false alarms when many tests are run.", "Silhouette score": "How well each location fits its own group compared with the nearest other group.",
          "ARI (adjusted Rand index)": "How similar two groupings are; 1 means identical.", "Dry spell": "A run of consecutive days with little or no rain."}
    for k, v in gl.items():
        st.markdown(f"**{k}**: {v}")
