"""Guardrail helpers for explanation output safety and format."""

from __future__ import annotations

import re

BANNED_MEDICAL_TERMS = {
    "diagnose",
    "diagnosis",
    "cure",
    "treat",
    "treatment",
    "medication",
    "prescription",
    "heal",
    "therapy",
    "disease",
    "clinical",
    "medical",
}
FORBIDDEN_PHRASES = {"guaranteed results", "will cure"}
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
_WORD_RE = re.compile(r"\b[a-z]+\b")


def count_sentences(text: str) -> int:
    """Count non-empty sentence chunks ending with ., !, or ?."""
    return len([chunk for chunk in _SENTENCE_RE.findall(text.strip()) if chunk.strip()])


def guard_two_sentences(explanation_text: str) -> bool:
    return count_sentences(explanation_text) == 2


def guard_no_banned_medical_terms(explanation_text: str) -> bool:
    lowered = explanation_text.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_PHRASES):
        return False
    words = set(_WORD_RE.findall(lowered))
    return not bool(words.intersection(BANNED_MEDICAL_TERMS))


def guard_sources_subset(final_sources: list[str], research_sources: list[str]) -> bool:
    final = {item.strip() for item in final_sources if isinstance(item, str) and item.strip()}
    research = {
        item.strip() for item in research_sources if isinstance(item, str) and item.strip()
    }
    if not final:
        return False
    return final.issubset(research)


def filter_to_research_subset(
    final_sources: list[str],
    research_sources: list[str],
) -> list[str]:
    """Return only final sources that appear in the research set.

    Comparison is whitespace-normalized but case-sensitive to preserve
    deterministic source identifiers (e.g. ``ingredients_dictionary.md#p1``).
    """
    research_set: set[str] = {
        s.strip() for s in research_sources if isinstance(s, str) and s.strip()
    }
    seen: set[str] = set()
    filtered: list[str] = []
    for src in final_sources:
        if not isinstance(src, str):
            continue
        key = src.strip()
        if key and key in research_set and key not in seen:
            seen.add(key)
            filtered.append(key)
    return filtered
