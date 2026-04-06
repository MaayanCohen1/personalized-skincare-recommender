"""Unit tests for explanation_service.condition_hints — locally runnable (no crewai)."""

from __future__ import annotations

from explanation_service.condition_hints import build_condition_guidance


def test_oily_only_mentions_lightweight_and_no_dry() -> None:
    result = build_condition_guidance(["oily"])
    assert "lightweight" in result.lower()
    assert "oil" in result.lower() or "oil balance" in result.lower()
    assert "NOT" in result


def test_oily_acne_mentions_pore_care() -> None:
    result = build_condition_guidance(["oily", "acne"])
    assert "acne" in result.lower()
    assert "pore" in result.lower()


def test_dry_only_mentions_hydration() -> None:
    result = build_condition_guidance(["dry"])
    assert "hydration" in result.lower() or "dry skin" in result.lower()
    assert "mattifying" in result.lower() or "stripping" in result.lower()


def test_combination_returns_balanced_framing() -> None:
    result = build_condition_guidance(["combination"])
    assert "combination" in result.lower()
    assert "balanced" in result.lower() or "zone" in result.lower()


def test_sensitive_only_prefers_gentle() -> None:
    result = build_condition_guidance(["sensitive"])
    assert "sensitive" in result.lower()
    assert "gentle" in result.lower() or "soothing" in result.lower()


def test_dry_sensitive_prefers_barrier_not_stripping() -> None:
    result = build_condition_guidance(["dry", "sensitive"])
    assert "dry" in result.lower() and "sensitive" in result.lower()
    assert "barrier" in result.lower() or "soothing" in result.lower()
    assert "stripping" in result.lower() or "mattifying" in result.lower()


def test_oily_dehydration_lightweight_humectant() -> None:
    result = build_condition_guidance(["oily", "dehydration"])
    assert "dehydrat" in result.lower()
    assert "lightweight" in result.lower() or "humectant" in result.lower()


def test_acne_sensitive_gentle_barrier() -> None:
    result = build_condition_guidance(["acne", "sensitive"])
    assert "acne" in result.lower()
    assert "sensitive" in result.lower()
    assert "gentle" in result.lower() or "barrier" in result.lower()


def test_acne_dry_hydration_not_mattifying() -> None:
    result = build_condition_guidance(["acne", "dry"])
    assert "acne" in result.lower()
    assert "dry" in result.lower()
    assert "mattifying" in result.lower() or "oil-control" in result.lower()


def test_normal_moderate_framing() -> None:
    result = build_condition_guidance(["normal"])
    assert "normal" in result.lower() or "balanced" in result.lower()


def test_empty_conditions_returns_empty() -> None:
    assert build_condition_guidance([]) == ""


def test_oily_only_case_insensitive() -> None:
    result = build_condition_guidance(["Oily"])
    assert "lightweight" in result.lower()


def test_oily_dry_gets_combination_style_not_oily_only() -> None:
    """Oily + dry should not receive oily-only (no dryness) guidance."""
    result = build_condition_guidance(["oily", "dry"])
    assert "combination" in result.lower() or "balanced" in result.lower()
    assert result != build_condition_guidance(["oily"])


def test_unknown_combo_still_returns_guidance() -> None:
    result = build_condition_guidance(["rosacea"])
    assert "rosacea" in result.lower()
    assert "skin profile" in result.lower()
