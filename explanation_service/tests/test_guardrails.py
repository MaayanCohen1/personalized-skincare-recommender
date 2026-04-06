"""Tests for explanation_service.guardrails — source filtering and validation."""

from __future__ import annotations

from explanation_service.guardrails import (
    draft_text_for_sentence_check,
    filter_to_research_subset,
    guard_sources_subset,
    guard_two_sentences,
    guard_two_sentences_draft,
    normalize_draft_whitespace,
    preview_for_log,
    sentence_count_draft,
)


# ---------------------------------------------------------------------------
# filter_to_research_subset
# ---------------------------------------------------------------------------


class TestFilterToResearchSubset:
    def test_keeps_matching_sources(self) -> None:
        result = filter_to_research_subset(
            ["dict.md#p1", "safety.md#p0"],
            ["dict.md#p1", "safety.md#p0", "guide.md#p2"],
        )
        assert result == ["dict.md#p1", "safety.md#p0"]

    def test_removes_non_matching_sources(self) -> None:
        result = filter_to_research_subset(
            ["dict.md#p1", "unknown_source", "safety.md#p0"],
            ["dict.md#p1", "safety.md#p0"],
        )
        assert result == ["dict.md#p1", "safety.md#p0"]

    def test_returns_empty_when_no_match(self) -> None:
        result = filter_to_research_subset(
            ["completely_unknown"],
            ["dict.md#p1"],
        )
        assert result == []

    def test_deduplicates_results(self) -> None:
        result = filter_to_research_subset(
            ["dict.md#p1", "dict.md#p1", "dict.md#p1"],
            ["dict.md#p1"],
        )
        assert result == ["dict.md#p1"]

    def test_strips_whitespace(self) -> None:
        result = filter_to_research_subset(
            ["  dict.md#p1  "],
            ["dict.md#p1"],
        )
        assert result == ["dict.md#p1"]

    def test_empty_final_sources(self) -> None:
        assert filter_to_research_subset([], ["dict.md#p1"]) == []

    def test_empty_research_sources(self) -> None:
        assert filter_to_research_subset(["dict.md#p1"], []) == []

    def test_preserves_order(self) -> None:
        result = filter_to_research_subset(
            ["b.md#p0", "a.md#p1", "c.md#p2"],
            ["c.md#p2", "a.md#p1", "b.md#p0"],
        )
        assert result == ["b.md#p0", "a.md#p1", "c.md#p2"]


# ---------------------------------------------------------------------------
# guard_sources_subset
# ---------------------------------------------------------------------------


class TestGuardSourcesSubset:
    def test_valid_subset(self) -> None:
        assert guard_sources_subset(
            ["dict.md#p1"],
            ["dict.md#p1", "safety.md#p0"],
        ) is True

    def test_not_a_subset(self) -> None:
        assert guard_sources_subset(
            ["unknown"],
            ["dict.md#p1"],
        ) is False

    def test_empty_final_sources_returns_false(self) -> None:
        assert guard_sources_subset([], ["dict.md#p1"]) is False


# ---------------------------------------------------------------------------
# Draft sentence helpers (draft_task guardrail only)
# ---------------------------------------------------------------------------


class TestDraftSentenceNormalization:
    def test_preview_for_log_truncates(self) -> None:
        long = "word " * 200
        p = preview_for_log(long, max_len=50)
        assert len(p) <= 50
        assert p.endswith("...")

    def test_normalize_draft_whitespace_collapses_newlines(self) -> None:
        t = normalize_draft_whitespace("  One  two.\n\n  Three.  ")
        assert t == "One two. Three."

    def test_normalize_strips_light_markdown_noise(self) -> None:
        t = normalize_draft_whitespace("**Bold** start. Second `here`.")
        assert "`" not in t
        assert "**" not in t

    def test_abbreviation_shield_reduces_spurious_sentence_count(self) -> None:
        raw = "This fits routine needs (e.g. barrier comfort). It uses gentle hydration daily."
        assert guard_two_sentences(raw) is False
        assert sentence_count_draft(raw) == 2
        assert guard_two_sentences_draft(raw) is True

    def test_three_real_sentences_still_fails_draft_guard(self) -> None:
        raw = "One thing here. Two things here. Three things here."
        assert sentence_count_draft(raw) == 3
        assert guard_two_sentences_draft(raw) is False

    def test_draft_check_does_not_change_audit_sentence_counter(self) -> None:
        """Audit/final path still uses strict count_sentences / guard_two_sentences."""
        raw = "This fits routine needs (e.g. barrier comfort). It uses gentle hydration daily."
        assert guard_two_sentences(raw) is False
        checked = draft_text_for_sentence_check(raw)
        assert guard_two_sentences(checked) is True
