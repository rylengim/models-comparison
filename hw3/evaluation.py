"""Deterministic HW1 metric plus a conservative explicit-negation guard.

The guard intentionally rejects ambiguous short answers rather than treating a
mentioned gold token as proof. This is still a heuristic: audit full answers and
source evidence, then document any confirmed dataset aliases before rerendering.
"""
import re

from freshqa.dataset import final_text, score_row

_NEGATION = re.compile(r"\b(?:not|never|neither|nor|не|нет)\b|n['’]t\b", re.IGNORECASE)


def score(row: dict, answer: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    actual = final_text(answer)
    if _NEGATION.search(actual):
        return False
    return score_row(row, actual)
