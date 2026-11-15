"""Optional language-model wording layer (OFF by default; the system is complete without it).

`llm_fn` is any callable `prompt:str -> str` (OpenAI, Anthropic, Gemini, a local Ollama model...).
The model is only allowed to REWORD an already-computed answer. Its output is rejected if it contains any
number that is not in the evidence, in which case the safe template text is used instead.
"""
from __future__ import annotations

import json

from weather_platform.ai.qa import Answer, verification_context
from weather_platform.ai.verify import unsupported_numbers

SYSTEM_RULES = (
    "You explain weather statistics. Use ONLY the facts in EVIDENCE. Do not add any number, date, cause, "
    "forecast or claim that is not in EVIDENCE. Keep it under 120 words. Mention the period, units and that the "
    "data is reanalysis, not a forecast."
)


def build_prompt(answer: Answer) -> str:
    ev = {"question": answer.question, "period": answer.period_label, "date_range": answer.date_range,
          "source": answer.source, "figures": answer.figures, "warnings": answer.warnings}
    return f"{SYSTEM_RULES}\n\nEVIDENCE:\n{json.dumps(ev, indent=2, default=str)}\n\nWrite the explanation."


def explain_with_llm(answer: Answer, llm_fn, baseline_years=(1991, 2020)) -> tuple[str, bool, list[str]]:
    """Returns (text, used_llm, unsupported_numbers). Abstained answers are never sent to a model."""
    if answer.status != "answered":
        return answer.text, False, []
    try:
        candidate = llm_fn(build_prompt(answer))
    except Exception as exc:  # network/API failure -> fall back safely
        return answer.text, False, [f"llm error: {exc}"]
    bad = unsupported_numbers(candidate, answer.figures, verification_context(answer, baseline_years))
    if bad:
        return answer.text, False, bad
    return candidate, True, []
