"""Historical Weather Analytics for India: router and shared layout.

Run:  python -m streamlit run app.py
"""
import streamlit as st

from weather_platform import state, ui

st.set_page_config(page_title="Historical Weather Analytics | India", page_icon=":material/partly_cloudy_day:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

# A re-run from a saved configuration may ask for demo mode: apply it before the sidebar widget exists.
_pending = st.session_state.get("_pending_cfg_state")
if _pending and "use_synth" in _pending:
    st.session_state["use_synth"] = _pending.pop("use_synth")

pages = {
    "Start": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True)],
    "Explore": [st.Page("views/india.py", title="India overview", icon=":material/map:"),
                st.Page("views/location.py", title="Location analysis", icon=":material/location_on:"),
                st.Page("views/compare.py", title="Compare locations", icon=":material/compare_arrows:"),
                st.Page("views/regimes.py", title="Climate regimes", icon=":material/hub:")],
    "Tools": [st.Page("views/ask.py", title="Ask the data", icon=":material/chat:"),
              st.Page("views/export.py", title="Reports and export", icon=":material/download:")],
    "About": [st.Page("views/methods.py", title="Methods and trust", icon=":material/verified:")],
}
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.checkbox("Demo mode (synthetic data)", key="use_synth", help="Uses invented data so you can try the platform without downloaded weather data. Never use it for real results.")
    D = state.loaded()
    if D:
        st.caption(f"Loaded: **{D['meta']['location']}**, {D['meta']['start'][:4]} to {D['meta']['end'][:4]}")
    st.markdown('<div class="wx-foot">Historical analysis of ERA5 reanalysis data (Open-Meteo, Copernicus/ECMWF). Not a weather forecast.</div>', unsafe_allow_html=True)

nav.run()
