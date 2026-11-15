# Historical Weather Analytics Platform

## Step 1 (done): data layer + quality checks
- `weather_platform/config.py`   variables, units, plausibility bounds, data-source attribution
- `weather_platform/data/`       city list/search, Open-Meteo fetcher with on-disk cache, synthetic data (tests only)
- `weather_platform/quality/`    request validation, duplicate/gap/range/unit checks, processing-step log
- `tests/`                       25 tests (`python -m pytest -q`)

Core rule: **missing is NaN, never 0.** Real zero-rain days are counted separately from missing days.

## Run
    pip install -r requirements.txt
    python -m pytest -q
    python demo_step1.py            # offline, synthetic data
    python demo_step1.py --live     # real data (internet needed)

Data is ERA5 reanalysis via Open-Meteo: gridded model output, not station measurements.

## Step 2 (done): statistics
- `stats/descriptive.py`  mean, median, min, max, sample std (ignores NaN, reports counts)
- `stats/rainfall.py`     totals (flagged as lower bound if days missing), wet/dry days, dry spells (missing days break a spell)
- `stats/baseline.py`     period vs baseline using annual values over the same calendar window; z-score, percentile,
                          t-based interval (single year) or Welch t-test + Cohen's d + bootstrap CI (multi-year)
- `stats/summary.py`      one call returning all statistics (evidence for the AI step)
Run: `python demo_step2.py` (offline) or `python demo_step2.py --live`.

## Step 3 (done): aggregation, dashboard, CSV export
- `aggregate.py`   daily / monthly / seasonal values (temperature averaged, rainfall summed); a period needs >= 90 % valid
                   days, otherwise it is left blank (never a partial total); monthly climatology; year-to-year tables
- `export.py`      CSV with a source/period/units header; missing values are empty cells, never 0
- `viz/charts.py`  temperature trend, rainfall bars, year-to-year lines, monthly climatology (units on every axis)
- `app.py`         Streamlit dashboard: Overview, Charts, Comparisons, Statistics (incl. baseline comparison),
                   Data quality (processing steps), Data and download

Run the dashboard:  `streamlit run app.py`   (tick "Use SYNTHETIC demo data" to try it offline)
Seasons are meteorological (DJF, MAM, JJA, SON); December belongs to the next year's DJF.

## Step 4 (done): evidence-grounded Q&A ("Ask the data" tab)
- `ai/parse.py`   rule-based question parser (intent, variable, period); never produces numbers
- `ai/qa.py`      computes answers from the loaded data with the tested stats code; abstains when it cannot answer reliably
- `ai/verify.py`  every number in an answer must trace to the computed evidence
- `ai/llm.py`     OPTIONAL wording layer: pass any `prompt -> str` function; output is rejected if it contains an unsupported number

Answers always show: period, date range, variable and units, data source, figures, warnings.
Abstains for: forecasts, causal "why" questions, trend claims, other locations, unloaded variables,
periods outside the data, and periods with too much missing data (<90% for totals/counts/baselines, <50% for averages).
Tests compute expected answers independently with plain pandas.

## India-wide version (done)
- `data/india_cities.csv`  71 locations: all state/UT capitals plus major cities and climate extremes (see docs/CITIES_DATA.md)
- `viz/maps.py`            India maps: selected location, value map, category map (missing data is grey, never zero)
- `india.py`               cached multi-city datasets, resumable `prefetch`, per-city tables, annual anomaly table, SYNTHETIC demo data
- `mining/climate_regimes.py`  KMeans + Ward + PCA on 24-value monthly climate profiles (data mining), silhouette reported for every k
- `pages/1_India_overview.py`  maps, annual anomalies, climate-regime clustering
- `scripts/prefetch_india.py`  download all locations politely and resumably; `scripts/verify_cities_geonames.py` checks coordinates

Workflow:
    python scripts/verify_cities_geonames.py          # once, needs internet
    python scripts/prefetch_india.py --dry-run
    python scripts/prefetch_india.py --max-new 15     # repeat until all are cached
    python -m streamlit run app.py                    # then open "India overview" in the sidebar

## PDF report (done: completes Level 1)
`report/pdf.py` builds the report from the loaded data: scope, source and attribution, data-quality table and processing steps,
summary statistics, rainfall metrics, charts, monthly averages, baseline comparison, methodology, limitations, and the analysis
configuration (JSON) for reproducibility. Every page footer says "Not a weather forecast". In the app: Data and download > Generate PDF report.

## Quick start on Windows (no commands needed)
Double-click `setup.bat` once, then `download_data.bat` (leave it running), then `run_app.bat`. See `START_HERE.txt`.
Downloaded data is saved in `data_cache\india` next to these files, wherever you run commands from.

## Level 2: anomalies and trends (done)
- `anomaly.py`  robust z-scores (median/MAD, same time of year, explicit baseline) for temperature; extreme-rain days above the baseline
                99th percentile of wet-day amounts; event grouping; Isolation Forest as a clearly-labelled comparison only
- `trend.py`    Mann-Kendall + Sen's slope with confidence interval, Hamed-Rao autocorrelation correction (checked against the
                `pymannkendall` package), Benjamini-Hochberg FDR across locations. Pre-whitening was tried first and DROPPED because a
                simulation test showed it increased false positives.
- App: "Anomalies and trends" tab per location; India overview page: "Trends" tab with significance after FDR correction.

## Level 2: comparison, reproducibility, AI evaluation (done)
- `compare.py`          several locations over MATCHING years (complete at every location), identical units enforced, paired differences with 95% CI,
                        paired t-test and Wilcoxon, distances and shared-grid-cell warnings (India overview > Compare locations)
- `reproduce.py`        save settings + a results fingerprint; load them back (sidebar > Re-run a saved analysis) and the app confirms whether
                        the results are identical
- `ai/evaluate.py`      question set generated from the data with an INDEPENDENT answer key; scores numerical accuracy, source/date-range
                        consistency, unsupported claims and correct refusals. App: Ask the data > Evaluate; script: `python scripts/evaluate_qa.py`
                        (writes evaluation/qa_results.csv and qa_report.md). Mutation tests prove the evaluator catches wrong, hallucinated and padded answers.

## Redesigned interface (multi-page)
`app.py` is a router; each page is a file in `views/`:
Home (what the platform is, 3-step guide) | India overview (climate map, yearly anomalies, long-term trends) | Location analysis (control panel, Summary,
Charts, Statistics, Unusual days and trends, Data quality) | Compare locations | Climate regimes | Ask the data | Reports and export | Methods and trust.
Design system in `weather_platform/ui.py` (theme in `.streamlit/config.toml`); plain-language text is generated from the computed numbers in `narrative.py`.
Run: `python -m streamlit run app.py` (or run_app.bat). Demo mode (sidebar) uses synthetic data.

## Verification status of this version
- 203 automated tests pass, including a full run through the REAL-DATA code path (cache built by the downloader, partial downloads, missing data, offline fallback).
- Passes in a brand-new virtual environment created from requirements.txt only (pandas 3.1, numpy 2.5, streamlit 1.66, reportlab 5.1, scikit-learn 1.9).
- The Streamlit server boots with the theme configuration; static analysis (pyflakes) is clean.
- See TROUBLESHOOTING.txt for common Windows problems.
