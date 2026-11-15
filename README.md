# 🌦️ Historical Weather Analytics Platform for India

*An interactive, evidence-grounded platform for analysing 35 years of daily weather data across 71 Indian locations.*

> **Describes the past only.** This platform is **not** a weather forecast and is not for safety-critical use. For warnings, use the India Meteorological Department (IMD).

---

## 📌 Project Overview

This project is an **interactive platform for historical weather analysis and visualisation over India**. A user picks a location and a period, and the platform shows charts and statistics, checks data quality, compares the period with a stated baseline (with uncertainty), detects unusual days and long-term trends, and answers plain-language questions.

The central design rule is **trust**:

- Every number is computed by tested code. The question-answering layer never generates numbers.
- Missing days are stored as missing (`NaN`), never as zero rainfall.
- The assistant **refuses** when the data cannot answer (forecasts, causes, periods outside the data, too many missing days).
- Every analysis can be **reproduced** from a saved settings file.

The scope covers all state and union-territory capitals plus major cities and climate extremes (**71 locations**), from **1991 to 2025** (12,784 daily values per location).

---

## 🗂️ Dataset

| Item | Detail |
|---|---|
| Source | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) |
| Underlying data | **ERA5 reanalysis** (Copernicus Climate Change Service / ECMWF) |
| Variables | Daily mean temperature (°C), daily precipitation (mm) |
| Period | 1991-01-01 to 2025-12-31 (12,784 days per location) |
| Locations | 71 (listed in `weather_platform/data/india_cities.csv`) |
| Grid | About 0.25° (~25 km) cells around each city centre |
| Missing values | 0% in the downloaded ERA5 data |

> ⚠️ ERA5 is **gridded model output, not station measurements**. Hills, coasts and city effects are smoothed, and rainfall can differ strongly from station records.

---

## 🧠 Methodology

### 1️⃣ Data Cleaning (`quality/checks.py`)

Seven logged steps: load, unit check, duplicate dates, **missing calendar days added as `NaN`**, physical-bounds check (impossible values → `NaN`), temperature consistency, and per-variable completeness. Zero-rain days are counted separately from missing days. A month, season or year is used only if **at least 90% of its days are valid**.

### 2️⃣ Statistics and Baseline Comparison (`stats/`)

Mean, median, extremes, rainfall totals, wet days (≥ 1 mm) and dry spells. A selected period is compared with a **1991–2020 baseline**: difference, z-score, percentile and a 95% interval (t-based for one year; Welch t-test, Cohen's d and bootstrap for several years).

### 3️⃣ Anomaly Detection (`anomaly.py`)

- **Temperature:** robust z-score using the median and MAD of baseline values within ±7 days of the same calendar day; flagged at |z| ≥ 3.5.
- **Rainfall:** extreme if above the 99th percentile of baseline wet-day amounts (rain is mostly zero, so a z-score does not work).
- Isolation Forest is shown only as a labelled cross-check.

### 4️⃣ Trend Analysis (`trend.py`)

Mann-Kendall test and Sen's slope on annual values, with the **Hamed-Rao correction** for autocorrelation and **Benjamini-Hochberg** false-discovery-rate control across all 71 locations.

### 5️⃣ Climate Regimes (`mining/climate_regimes.py`)

KMeans on standardised summary features (silhouette reported for every k), checked with Ward clustering, restart stability and a second feature set (adjusted Rand index). A warning appears when the structure is weak.

### 6️⃣ Evidence-Grounded Question Answering (`ai/`)

`parse.py` reads the question → `qa.py` computes the answer with the same tested code → `verify.py` checks that **every number in the text matches a computed figure**. The assistant abstains for forecasts, causes, trend claims, other cities, unloaded variables and periods with too few valid days. `evaluate.py` scores the assistant against an **independent pandas answer key**.

### 7️⃣ Reproducibility (`reproduce.py`)

Settings are saved as JSON together with a **results fingerprint** (a hash of valid-day counts, means and sums). Uploading the file restores the analysis and shows whether the results are reproduced.

---

## 🛠 Technologies Used

- Python 3.12
- pandas, NumPy, SciPy
- scikit-learn (KMeans, Ward, PCA, Isolation Forest)
- Streamlit (multipage app) and Plotly (charts and maps)
- ReportLab and Matplotlib (PDF report)
- requests (Open-Meteo API, with on-disk cache)
- pytest, Streamlit AppTest and pymannkendall (testing)

---

## ✨ Features

- **India overview:** climate maps, yearly anomalies for any year, long-term trend maps with raw and corrected significance counts
- **Location analysis:** summary, charts, statistics, baseline comparison, unusual days, trend test and data-quality tabs
- **Compare locations:** 2 to 6 places over matching complete years
- **Climate regimes:** data-mining view of India's climate groups
- **Ask the data:** natural-language questions with an evidence table and honest refusals, plus a built-in evaluation panel
- **Reports and export:** PDF report, daily and aggregated CSV, settings JSON, one-click re-run
- **Methods and trust:** definitions, limitations and glossary
- **Demo mode:** synthetic data for trying the app (never real results)

---

## 📂 Project Structure

```
Historical_Weather_Analytics_Platform/
│
├── app.py                      # Streamlit router (st.navigation)
├── views/                      # The eight pages
│   ├── home.py  india.py  location.py  compare.py
│   └── regimes.py  ask.py  export.py  methods.py
│
├── weather_platform/
│   ├── data/                   # fetch.py (API + cache), cities, india_cities.csv, synthetic data
│   ├── quality/checks.py       # cleaning and quality report
│   ├── stats/                  # descriptive, rainfall, baseline, summary
│   ├── anomaly.py  trend.py  compare.py  aggregate.py
│   ├── mining/climate_regimes.py
│   ├── ai/                     # parse, qa, verify, evaluate, llm (optional hook)
│   ├── report/pdf.py           # PDF report
│   ├── viz/                    # charts and maps
│   └── reproduce.py  export.py  india.py  narrative.py  state.py  ui.py  config.py
│
├── scripts/
│   ├── prefetch_india.py       # download all locations (resumable)
│   ├── evaluate_qa.py          # scored Q&A evaluation
│   └── verify_cities_geonames.py
│
├── tests/                      # 214 automated tests
├── evaluation/                 # Q&A evaluation outputs
├── evidence/                   # test and data-check logs
├── docs/CITIES_DATA.md
├── data_cache/india/           # cached data, one CSV + metadata file per location
│
├── setup.bat  download_data.bat  run_app.bat
├── START_HERE.txt  TROUBLESHOOTING.txt
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started (Windows)

```powershell
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>

.\setup.bat            # creates venv, installs requirements, runs the tests
.\download_data.bat    # only if data_cache/india is empty (downloads all 71 locations)
.\run_app.bat          # starts the app
```

Or manually:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prefetch_india.py --auto        # optional: download the data
python -m streamlit run app.py
```

Open **http://localhost:8501**. Data and analysis work offline from the cache; map background tiles need internet. The message `ConnectionResetError [WinError 10054]` in the terminal is harmless. See `TROUBLESHOOTING.txt` for more.

**Run the tests:**

```powershell
python -m pytest
```

---

## 📊 System Workflow

Open-Meteo / ERA5 → `prefetch_india.py` → local cache → cleaning and quality checks → statistics, anomalies, trends, comparison, climate regimes → Streamlit app, question answering and PDF report

---

## 📈 Key Results

| Check | Result |
|---|---|
| Guwahati, July 2022 rainfall | **402.4 mm**, identical in pandas, the app and a hand sum of API values |
| Q&A evaluation (6 cities × 45 questions = 270) | **100%** numerical accuracy, **100%** correct abstention, **0%** unsupported claims |
| Automated tests | **214 passed** |
| Temperature trends (71 locations) | 37 significant; **31 after** false-discovery-rate correction |
| Rainfall trends (71 locations) | 49 significant; **42 after** correction (flagged as suspicious, see Limitations) |
| Climate regimes (4 features, k = 4) | Silhouette **0.37** (weak structure) |

---

## ⚠️ Limitations

- ERA5 is a ~25 km reanalysis grid, not station data, and has **not yet been validated against IMD data**.
- 42 of 71 significant rainfall trends is far more than station studies usually find, so rainfall trends are treated as a possible **reanalysis artefact, not climate change**.
- Some results, such as Leh's trend, look suspicious and are not yet verified.
- Climate groups are a partition of a continuum, not natural classes (silhouette about 0.3–0.4).
- The assistant is rule-based; its 100% score covers **supported question forms only**.
- Results depend on analyst choices (baseline years, wet-day threshold, completeness rule).

---

## 🔮 Future Improvements

- Validate rainfall and temperature against IMD station or gridded data
- Detect abrupt shifts per location (for example a Pettitt test)
- Score the assistant on questions written in a user's own wording
- Forecast post-processing judged by skill scores
- Optional language-model wording layer (numbers still verified)
- Public hosting of the app

---

## 📚 Learning Outcomes

- End-to-end data pipeline: acquisition, caching, cleaning, analysis and presentation
- Trend statistics for climate data: autocorrelation correction and multiple-testing control
- Unsupervised learning with honest quality checks
- Building a question-answering system that verifies its own numbers and abstains safely
- Reproducible analysis with saved settings and results fingerprints
- Software testing: independent expected values, simulations and headless UI tests

---

## 👩‍💻 Project Information

- **Project Title:** Historical Weather Analytics Platform for India
- **Author Name:** Sanghita Roy
- **Roll Number:** 23035010421
- **Program:** B.Sc. (Hons.) Data Science & Artificial Intelligence
- **Institute:** IIT Guwahati

---

## 🙏 Data Attribution

- Weather data: [Open-Meteo](https://open-meteo.com) Historical Weather API (CC BY 4.0)
- Source dataset: ERA5 reanalysis, Copernicus Climate Change Service / ECMWF
- City coordinates checked with [GeoNames](https://www.geonames.org) (CC BY 4.0)

---

## ⭐ Academic Purpose

This project was developed as part of academic learning to demonstrate understanding of:

- Data acquisition, cleaning and quality control
- Statistical analysis of climate data
- Data mining and clustering
- Trustworthy, evidence-grounded question answering
- Reproducible, tested analytics software

> ⚠️ **Disclaimer:** This platform is for research and educational purposes only. It describes historical, reanalysis-based data, is **not** a weather forecast, and must not be used for safety-critical decisions.
