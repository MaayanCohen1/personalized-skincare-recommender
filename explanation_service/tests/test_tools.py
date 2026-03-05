"""Unit tests for explanation_service.tools — IngredientEvidenceTool.

Tests the pure lookup_ingredient function directly, avoiding any LLM or
CrewAI runtime dependency.
"""

from __future__ import annotations

import pytest

from explanation_service.tools import IngredientEvidenceTool, _EVIDENCE_DB, lookup_ingredient


class TestLookupIngredient:
    """Tests for the local evidence lookup."""

    @pytest.mark.parametrize("ingredient", list(_EVIDENCE_DB.keys()))
    def test_known_ingredient_returns_fact_and_source(self, ingredient: str) -> None:
        result = lookup_ingredient(ingredient)
        assert "fact" in result
        assert "source" in result
        assert result["source"] != "generic"

    def test_unknown_ingredient_returns_generic(self) -> None:
        result = lookup_ingredient("unicorn_extract")
        assert result["source"] == "generic"
        assert len(result["fact"]) > 0

    def test_lookup_is_case_insensitive(self) -> None:
        lower = lookup_ingredient("niacinamide")
        upper = lookup_ingredient("NIACINAMIDE")
        mixed = lookup_ingredient("Niacinamide")
        assert lower == upper == mixed

    def test_lookup_strips_whitespace(self) -> None:
        result = lookup_ingredient("  glycerin  ")
        assert result["source"] != "generic"

    @pytest.mark.parametrize("ingredient", list(_EVIDENCE_DB.keys()))
    def test_no_medical_claims_in_evidence(self, ingredient: str) -> None:
        banned = {
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
        banned_phrases = {"guaranteed results", "will cure"}
        fact = lookup_ingredient(ingredient)["fact"].lower()
        violations = banned.intersection(fact.split())
        phrase_violations = {p for p in banned_phrases if p in fact}
        assert not violations, f"Banned word(s) {violations} in fact for {ingredient}"
        assert not phrase_violations, (
            f"Banned phrase(s) {phrase_violations} in fact for {ingredient}"
        )


class TestIngredientEvidenceTool:
    """Tests for the @tool wrapper output contract."""

    def test_tool_returns_dict_contract(self) -> None:
        result = IngredientEvidenceTool.run(ingredient_name="glycerin")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"fact", "source"}
        assert isinstance(result["fact"], str)
        assert isinstance(result["source"], str)

    def test_tool_unknown_ingredient_returns_generic_source(self) -> None:
        result = IngredientEvidenceTool.run(ingredient_name="unknown_ingredient")
        assert result["source"] == "generic"
