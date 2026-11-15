"""Number verification: every number in a generated explanation must come from the computed evidence.

This is the guard that stops an LLM (or a template bug) from inventing figures.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*")


def _tokens(text: str) -> list[str]:
    return [t.replace(",", "") for t in _NUM.findall(text)]


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def allowed_numbers(figures: list[dict], context_strings: list[str]) -> list[float]:
    nums: list[float] = []
    for f in figures:
        v = f.get("value")
        if isinstance(v, (int, float)):
            nums.append(float(v))
            nums.append(abs(float(v)))
    for s in context_strings:
        for tok in _tokens(str(s)):
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    return nums


def unsupported_numbers(text: str, figures: list[dict], context_strings: list[str]) -> list[str]:
    """Return the numbers in `text` that cannot be traced to the evidence (empty list = verified)."""
    allowed = allowed_numbers(figures, context_strings)
    bad = []
    for tok in _tokens(text):
        try:
            x = float(tok)
        except ValueError:
            continue
        d = _decimals(tok)
        tol = 0.5 * 10 ** (-d) + 1e-9
        ok = any(abs(a - x) <= tol or abs(abs(a) - abs(x)) <= tol for a in allowed)
        # percentages/rounding of whole numbers: allow x == round(a)
        if not ok:
            bad.append(tok)
    return bad
