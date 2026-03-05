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
