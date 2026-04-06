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

# Light markdown / emphasis characters (draft normalization only).
_MD_NOISE_RE = re.compile(r"[*_`]+")

# Common abbreviations whose periods inflate naive sentence counts (draft check only).
_ABBREV_FOR_COUNT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\be\.g\.", re.IGNORECASE), "eg"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "ie"),
    (re.compile(r"\bet\s+al\.", re.IGNORECASE), "et al"),
    (re.compile(r"\betc\.", re.IGNORECASE), "etc"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "vs"),
    (re.compile(r"\bDr\.", re.IGNORECASE), "Dr"),
    (re.compile(r"\bMr\.", re.IGNORECASE), "Mr"),
    (re.compile(r"\bMrs\.", re.IGNORECASE), "Mrs"),
    (re.compile(r"\bMs\.", re.IGNORECASE), "Ms"),
]


def count_sentences(text: str) -> int:
    """Count non-empty sentence chunks ending with ., !, or ?."""
    return len([chunk for chunk in _SENTENCE_RE.findall(text.strip()) if chunk.strip()])


def preview_for_log(text: str, max_len: int = 400) -> str:
    """Single-line preview for logging (no huge blobs)."""
    one_line = " ".join(text.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def normalize_draft_whitespace(text: str) -> str:
    """Collapse whitespace and line breaks; strip ends."""
    t = text.strip()
    t = _MD_NOISE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def shield_abbreviations_for_sentence_count(text: str) -> str:
    """Remove spurious sentence-ending periods from common abbreviations."""
    t = text
    for pattern, repl in _ABBREV_FOR_COUNT_REPLACEMENTS:
        t = pattern.sub(repl, t)
    return t


def draft_text_for_sentence_check(text: str) -> str:
    """Normalize draft text only for sentence counting (no semantic rewriting)."""
    return shield_abbreviations_for_sentence_count(normalize_draft_whitespace(text))


def sentence_count_draft(text: str) -> int:
    """Sentence count for draft guardrail (whitespace + abbreviation shielding)."""
    return count_sentences(draft_text_for_sentence_check(text))


def guard_two_sentences(explanation_text: str) -> bool:
    return count_sentences(explanation_text) == 2


def guard_two_sentences_draft(raw_draft: str) -> bool:
    """Exactly two sentences for draft_task; uses robust counting only here."""
    return sentence_count_draft(raw_draft) == 2


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
