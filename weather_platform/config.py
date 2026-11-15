"""Central configuration: supported variables, plausibility bounds, data-source metadata.

Every threshold lives here so the quality checks, the UI and the PDF report all
read the same definitions (and so they can be cited in the methodology section).
"""
from __future__ import annotations

import pandas as pd

DATA_SOURCE = {
    "name": "Open-Meteo Historical Weather API (ERA5 reanalysis)",
    "url": "https://open-meteo.com/en/docs/historical-weather-api",
    "api_url": "https://archive-api.open-meteo.com/v1/archive",
    "type": "Gridded reanalysis (model output), NOT direct station measurements",
    "licence_note": "Open-Meteo data is CC BY 4.0; ERA5 is produced by Copernicus/ECMWF. Check current terms before publishing.",
}

# Earliest date ERA5 coverage starts, and the typical delay before recent days appear.
DATA_START = pd.Timestamp("1940-01-01")
DATA_LATENCY_DAYS = 6  # conservative; the API is typically ~5 days behind real time

# key = Open-Meteo daily variable name.
# unit  = unit we REQUEST and expect back (validated on every response).
# min/max = hard physical plausibility bounds (values outside become missing, not zero).
VARIABLES: dict[str, dict] = {
    "temperature_2m_mean": {"label": "Mean temperature", "unit": "°C", "min": -90.0, "max": 60.0, "kind": "temperature"},
    "temperature_2m_max": {"label": "Maximum temperature", "unit": "°C", "min": -90.0, "max": 60.0, "kind": "temperature"},
    "temperature_2m_min": {"label": "Minimum temperature", "unit": "°C", "min": -90.0, "max": 60.0, "kind": "temperature"},
    "precipitation_sum": {"label": "Precipitation (daily total)", "unit": "mm", "min": 0.0, "max": 2000.0, "kind": "precipitation"},
    "wind_speed_10m_max": {"label": "Maximum wind speed (10 m)", "unit": "km/h", "min": 0.0, "max": 500.0, "kind": "wind"},
}

# A day with precipitation >= this many mm is a "wet day" (used from Step 2 onward).
WET_DAY_MM = 1.0
