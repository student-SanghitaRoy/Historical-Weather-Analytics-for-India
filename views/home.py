import streamlit as st

from weather_platform import india, state, ui
from weather_platform.data.cities import load_cities

cities = load_cities()
n_cached = state.cache_count()
ui.hero("Historical Weather Analytics for India",
        "Explore 35 years of daily weather across India with statistics you can trust, clear explanations, and an assistant that says "
        "\"I cannot answer that\" when the data cannot.", badge="ERA5 reanalysis  |  Historical analysis, not a forecast")
ui.kpis([("Locations", str(len(cities)), "all state and UT capitals plus major cities"),
         ("Daily record", f"{india.PREFETCH_START[:4]} to {india.PREFETCH_END[:4]}", "temperature and rainfall"),
         ("Downloaded", f"{n_cached} of {len(cities)}", "ready to use offline"),
         ("Data source", "ERA5", "via Open-Meteo, ~25 km grid")])

st.subheader("Start here: three steps")
ui.cards([("Explore India", "See how temperature and rainfall differ across the country, which years were unusual, and where trends are statistically significant."),
          ("Analyse one location", "Pick a city and a period. Get a plain-language summary, charts, statistics, unusual days and trends."),
          ("Ask questions and export", "Ask in plain English, get answers computed from the data, then export a PDF report or CSV files.")], numbered=True)
c1, c2, c3 = st.columns(3)
c1.page_link("views/india.py", label="Open India overview", icon=":material/map:")
c2.page_link("views/location.py", label="Open Location analysis", icon=":material/location_on:")
c3.page_link("views/ask.py", label="Open Ask the data", icon=":material/chat:")

st.subheader("What you can do")
ui.cards([("Maps and anomalies", "Colour-coded India maps of climate, yearly anomalies against a stated baseline, and long-term trends."),
          ("Statistics with uncertainty", "Averages, rainfall totals, wet days and dry spells, plus comparisons with a baseline that show how large and how certain a difference is."),
          ("Unusual days and trends", "Robust statistical detection of unusual temperatures and extreme rain, and Mann-Kendall trend tests with Sen's slope."),
          ("Compare locations", "Compare places over exactly the same years, with units checked and the limits of the data stated."),
          ("Climate regimes", "Data mining groups all locations by the shape of their year, such as desert, monsoon coast or cool hills."),
          ("Reports", "A PDF with sources, methods, charts and limitations, and a saved settings file to repeat any analysis exactly.")])

st.subheader("Why you can trust the results")
ui.cards([("Honest about gaps", "Missing data is never turned into zero. A period with too little data is left blank, not guessed."),
          ("Transparent methods", "Every definition (what counts as a wet day, a dry spell, a baseline) is stated, and the code is covered by automated tests."),
          ("An assistant that can refuse", "Answers are computed by code and every number is checked. It declines forecasts, causes, and questions the data cannot support."),
          ("Repeatable", "Settings are saved with a fingerprint of the results, so an analysis can be re-run and confirmed identical.")], cols=2)

ui.callout("warn", "Please keep in mind",
           "This platform analyses the past. It is not a weather forecast and must not be used for safety decisions: for warnings use the India Meteorological Department. "
           "The data are ERA5 reanalysis values for a roughly 25 km grid cell around each city, not station measurements, so rainfall in particular can differ from local records.")
if n_cached == 0 and not state.synthetic_mode():
    ui.callout("info", "No data downloaded yet",
               "Run download_data.bat to fetch the real data (it can take a few hours because the free service has hourly limits), "
               "or switch on Demo mode in the sidebar to explore the platform with invented data.")
st.caption("Weather data by Open-Meteo.com (CC BY 4.0), based on ERA5 from Copernicus Climate Change Service / ECMWF. Location coordinates verified against GeoNames (CC BY 4.0).")
