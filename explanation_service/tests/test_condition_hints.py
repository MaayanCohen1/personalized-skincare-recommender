"""Unit tests for explanation_service.condition_hints — locally runnable (no crewai)."""

from __future__ import annotations

from explanation_service.condition_hints import build_condition_guidance


def test_oily_only_mentions_lightweight_and_no_dry() -> None:
    result = build_condition_guidance(["oily"])
    assert "lightweight" in result.lower()
    assert "oil control" in result.lower()
    assert "NOT" in result


def test_oily_acne_mentions_pore_care() -> None:
    result = build_condition_guidance(["oily", "acne"])
    assert "acne" in result.lower()
    assert "pore" in result.lower()


def test_dry_only_mentions_hydration() -> None:
    result = build_condition_guidance(["dry"])
    assert "hydration" in result.lower()
    assert "barrier" in result.lower()


def test_unknown_conditions_returns_empty() -> None:
    assert build_condition_guidance(["combination"]) == ""


def test_empty_conditions_returns_empty() -> None:
    assert build_condition_guidance([]) == ""


def test_oily_only_case_insensitive() -> None:
    result = build_condition_guidance(["Oily"])
    assert "lightweight" in result.lower()


def test_oily_with_extra_condition_returns_empty() -> None:
    """Oily + dry is not oily-only, should not get oily-only guidance."""
    result = build_condition_guidance(["oily", "dry"])
    assert result == ""
