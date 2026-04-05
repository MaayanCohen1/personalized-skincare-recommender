"""Tests for explanation_service.ingredients — extract_key_ingredients."""

from __future__ import annotations

from explanation_service.ingredients import extract_key_ingredients


def test_selects_known_actives() -> None:
    ingredients = ["water", "niacinamide", "glycolic acid", "carbomer"]
    result = extract_key_ingredients(ingredients)
    assert "niacinamide" in result
    assert "glycolic acid" in result


def test_skips_filler_ingredients() -> None:
    ingredients = ["water", "aqua", "xanthan gum", "sodium chloride", "carbomer"]
    result = extract_key_ingredients(ingredients)
    assert result == []


def test_includes_fragrance_as_notable() -> None:
    ingredients = ["water", "glycerin", "fragrance"]
    result = extract_key_ingredients(ingredients)
    assert "fragrance" in result


def test_includes_drying_alcohol_as_notable() -> None:
    ingredients = ["water", "alcohol denat", "retinol"]
    result = extract_key_ingredients(ingredients)
    assert "alcohol denat" in result
    assert "retinol" in result


def test_notables_ordered_before_actives() -> None:
    ingredients = ["retinol", "fragrance", "niacinamide"]
    result = extract_key_ingredients(ingredients)
    assert result.index("fragrance") < result.index("retinol")


def test_max_key_limit() -> None:
    actives = [f"peptide-{i}" for i in range(15)]
    result = extract_key_ingredients(actives)
    assert len(result) <= 10


def test_fallback_returns_non_filler_when_no_actives() -> None:
    ingredients = ["water", "dimethicone", "butylene glycol", "caprylic triglyceride", "sodium chloride"]
    result = extract_key_ingredients(ingredients)
    assert len(result) <= 3
    assert "water" not in result
    assert "sodium chloride" not in result


def test_fallback_empty_when_all_filler() -> None:
    ingredients = ["water", "aqua", "carbomer"]
    result = extract_key_ingredients(ingredients)
    assert result == []


def test_empty_input() -> None:
    assert extract_key_ingredients([]) == []


def test_case_insensitive_matching() -> None:
    ingredients = ["Niacinamide", "RETINOL", "Water"]
    result = extract_key_ingredients(ingredients)
    assert "Niacinamide" in result
    assert "RETINOL" in result


def test_preserves_original_casing() -> None:
    ingredients = ["Hyaluronic Acid", "Salicylic Acid"]
    result = extract_key_ingredients(ingredients)
    assert result == ["Hyaluronic Acid", "Salicylic Acid"]


def test_substring_matching_for_ceramides() -> None:
    ingredients = ["ceramide np", "ceramide ap", "water"]
    result = extract_key_ingredients(ingredients)
    assert len(result) == 2
    assert all("ceramide" in r.lower() for r in result)


def test_glycerin_treated_as_active_not_filler() -> None:
    """glycerin appears in _KEY_ACTIVES, so it should be selected."""
    ingredients = ["water", "glycerin", "carbomer"]
    result = extract_key_ingredients(ingredients)
    assert "glycerin" in result
