"""Rule-based question understanding (no model needed, fully deterministic and testable).

The parser only extracts WHAT is being asked (intent, variable, period). It never produces numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
          "oct": 10, "nov": 11, "dec": 12}
SEASONS = {"djf", "mam", "jja", "son"}

FORECAST_WORDS = ["forecast", "predict", "tomorrow", "next week", "next month", "next year", "will it", "going to rain",
                  "will be", "future", "upcoming", "tonight", "this weekend"]
CAUSAL_WORDS = ["why ", "cause", "because", "reason for", "due to", "explain why"]
TREND_WORDS = ["trend", "getting warmer", "getting hotter", "getting wetter", "getting drier", "increasing", "decreasing",
               "climate change", "global warming", "over the years", "over time", "changing"]
COMPARE_LOC_WORDS = ["compare cities", "compare locations", "which city", "which place", "vs "]

INTENT_PATTERNS = [
    ("dry_spell", ["dry spell", "longest dry", "consecutive dry", "days without rain", "rain-free", "rainless"]),
    ("wet_days", ["wet day", "rainy day", "days of rain", "days with rain", "how many days it rained", "days it rained",
                  "number of days with rain"]),
    ("missing", ["missing", "data quality", "completeness", "how complete", "gaps", "reliable is the data"]),
    ("baseline", ["unusual", "anomal", "abnormal", "compared to normal", "compared with normal", "than normal", "than average",
                  "than usual", "compared to average", "compared with average", "compared to the baseline",
                  "against the baseline", "above normal", "below normal", "unusually", "typical year", "versus normal"]),
    ("total", ["total rain", "how much rain", "rainfall total", "total precipitation", "accumulated", "total rainfall",
               "rain in total", "how much did it rain", "how much precipitation"]),
    ("median", ["median"]),
    ("std", ["standard deviation", "variability", "spread", "std "]),
    ("max", ["maximum", "highest", "hottest", "warmest", "wettest", "peak", "max ", "largest", "most rain"]),
    ("min", ["minimum", "lowest", "coldest", "coolest", "min ", "smallest"]),
    ("mean", ["average", "mean", "typical", "on average"]),
]


@dataclass
class Parsed:
    raw: str
    intent: str | None = None
    variable_hint: str | None = None  # precip | temp | temp_max | temp_min | wind
    years: list = field(default_factory=list)
    range_years: bool = False
    months: list = field(default_factory=list)
    season: str | None = None  # DJF/MAM/JJA/SON or 'monsoon'
    cities: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)  # forecast, causal, trend, compare_locations


def _month_tokens(text: str) -> list[int]:
    out = []
    for m in re.finditer(r"\b([a-z]+)\b", text):
        w = m.group(1)
        if w not in MONTHS:
            continue
        if w == "may":  # ambiguous with the verb: accept only in a date-like context
            before = text[: m.start()].split()[-1:] or [""]
            after = text[m.end():].split()[:1] or [""]
            if not (re.fullmatch(r"(19|20)\d\d", after[0]) or before[0] in {"in", "during", "of", "for", "from", "to", "and", "every", "between"}):
                continue
        out.append(MONTHS[w])
    return out


def parse_question(question: str, city_names: list[str] | None = None) -> Parsed:
    q = " " + re.sub(r"\s+", " ", question.lower().strip()) + " "
    p = Parsed(raw=question)

    p.flags = {
        "forecast": any(w in q for w in FORECAST_WORDS),
        "causal": any(w in q for w in CAUSAL_WORDS),
        "trend": any(w in q for w in TREND_WORDS),
        "compare_locations": any(w in q for w in COMPARE_LOC_WORDS),
    }

    p.years = [int(y) for y in re.findall(r"\b(19[4-9]\d|20\d\d)\b", q)]
    p.range_years = len(p.years) == 2 and bool(re.search(r"\b(from|between)\b", q))
    months = _month_tokens(q)
    if len(months) == 2 and re.search(r"\b(from|between)\b", q) and re.search(r"\b(to|through|and|till|until)\b", q) and months[0] <= months[1]:
        months = list(range(months[0], months[1] + 1))
    p.months = sorted(set(months))
    for s in SEASONS:
        if re.search(rf"\b{s}\b", q):
            p.season = s.upper()
    if "monsoon" in q:
        p.season = "monsoon"

    for name in city_names or []:
        if name.lower() in q:
            p.cities.append(name)

    for intent, words in INTENT_PATTERNS:
        if any(w in q for w in words):
            p.intent = intent
            break

    # variable hint
    rain = any(w in q for w in ["rain", "precip", "wettest", "drizzle", "monsoon rainfall", "shower"])
    wind = "wind" in q
    temp_words = any(w in q for w in ["temperature", "temp ", "hot", "cold", "warm", "cool", "heat"])
    temp = temp_words and not rain
    rain_only_intent = p.intent in ("wet_days", "dry_spell", "total")
    if rain or (rain_only_intent and not temp_words):
        p.variable_hint = "precip"
    elif rain_only_intent and temp_words:
        p.variable_hint = "temp"  # conflicting request: the engine will abstain explicitly
    elif wind:
        p.variable_hint = "wind"
    elif temp or p.intent in ("max", "min") and any(w in q for w in ["hottest", "coldest", "warmest", "coolest"]):
        explicit_max = any(w in q for w in ["maximum temp", "max temp", "daytime", "high temp", "daily high"])
        explicit_min = any(w in q for w in ["minimum temp", "min temp", "nighttime", "night temp", "low temp", "daily low", "overnight"])
        if explicit_max:
            p.variable_hint = "temp_max"
        elif explicit_min:
            p.variable_hint = "temp_min"
        elif p.intent == "max":
            p.variable_hint = "temp_max"
        elif p.intent == "min":
            p.variable_hint = "temp_min"
        else:
            p.variable_hint = "temp"
    if p.intent is None and p.variable_hint and (p.years or p.months or p.season):
        p.intent = "mean"
    return p
