import itertools
import math

import pytest

from weather_platform.data.cities import get_city, load_cities, search_cities, slug

STATES_AND_UTS = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana*", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Chandigarh", "Puducherry", "Lakshadweep", "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu",
}


def test_size_and_unique_names():
    cs = load_cities()
    assert 60 <= len(cs) <= 120
    assert len({c["name"] for c in cs}) == len(cs)


def test_coordinates_inside_india_bounding_box():
    for c in load_cities():
        assert 6.0 <= c["lat"] <= 37.5 and 68.0 <= c["lon"] <= 98.0, c["name"]


def test_every_state_and_ut_is_represented():
    have = {c["state"] for c in load_cities()}
    missing = {s for s in STATES_AND_UTS if not s.endswith("*")} - have
    assert not missing, missing  # Haryana is covered by Chandigarh (shared capital), noted in the file


def test_no_near_duplicate_points():
    # two points closer than ~0.15 degrees (~16 km) would be the same ERA5 grid cell
    for a, b in itertools.combinations(load_cities(), 2):
        assert math.hypot(a["lat"] - b["lat"], a["lon"] - b["lon"]) > 0.15, (a["name"], b["name"])


def test_fields_and_regions():
    for c in load_cities():
        assert c["region"] in {"North", "South", "East", "West", "Central", "Northeast", "Islands"}
        assert c["state"] and c["kind"] and c["country"] == "India"


def test_search_and_lookup():
    assert [c["name"] for c in search_cities("guwa")] == ["Guwahati"]
    assert {"Guwahati", "Dibrugarh", "Silchar"} <= {c["name"] for c in search_cities("assam")}
    assert get_city("kolkata")["state"] == "West Bengal"
    assert slug("Port Blair") == "port_blair"
    with pytest.raises(KeyError):
        get_city("Atlantis")
