"""Tests for explanation_service.guardrails — source filtering and validation."""

from __future__ import annotations

from explanation_service.guardrails import (
    filter_to_research_subset,
    guard_sources_subset,
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
