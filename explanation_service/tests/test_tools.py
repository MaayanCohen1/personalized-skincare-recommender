"""Unit tests for explanation_service.tools — IngredientEvidenceTool (Naive RAG).

All tests are fast and deterministic: SentenceTransformer and the pickle KB
are mocked so no model download or disk I/O is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import explanation_service.tools as tools_module
from explanation_service.tools import (
    IngredientEvidenceTool,
    _GENERIC_ENTRY,
    _cosine_similarity,
    lookup_ingredient,
)


_FAKE_KB: list[dict[str, Any]] = [
    {
        "text": "Niacinamide supports the skin barrier and helps maintain an even tone.",
        "source": "ingredients_dictionary.md#p1",
        "vector": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    },
    {
        "text": "Hyaluronic acid is a humectant that helps the skin retain moisture.",
        "source": "ingredients_dictionary.md#p2",
        "vector": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    },
    {
        "text": "Sunscreen with SPF 30 or higher is recommended for daily use.",
        "source": "safety_guidelines.md#p1",
        "vector": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    },
]


class _FakeModel:
    """Stub for SentenceTransformer that returns controllable dummy vectors."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        vecs = []
        for text in texts:
            lower = text.lower()
            if "niacinamide" in lower:
                vecs.append(np.array([0.95, 0.05, 0.0], dtype=np.float32))
            elif "hyaluronic" in lower:
                vecs.append(np.array([0.05, 0.95, 0.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.33, 0.33, 0.34], dtype=np.float32))
        return np.array(vecs)


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Ensure singleton caches are cleared between tests."""
    tools_module._kb_cache = None
    tools_module._model_cache = None
    yield
    tools_module._kb_cache = None
    tools_module._model_cache = None


@pytest.fixture()
def _mock_rag() -> None:
    """Inject fake KB + fake model so tests never touch disk or network."""
    tools_module._kb_cache = list(_FAKE_KB)
    tools_module._model_cache = _FakeModel()


class TestCosineUtility:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestLookupIngredient:
    """Tests for the RAG-based lookup function."""

    @pytest.mark.usefixtures("_mock_rag")
    def test_known_ingredient_returns_fact_and_source(self) -> None:
        result = lookup_ingredient("niacinamide")
        assert "fact" in result
        assert "source" in result
        assert result["source"] != "generic"
        assert len(result["fact"]) > 0

    @pytest.mark.usefixtures("_mock_rag")
    def test_result_dict_has_correct_keys(self) -> None:
        result = lookup_ingredient("hyaluronic acid")
        assert set(result.keys()) == {"fact", "source"}

    def test_returns_generic_when_pickle_missing(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            result = lookup_ingredient("niacinamide")
        assert result["source"] == "generic"
        assert result["fact"] == _GENERIC_ENTRY["fact"]

    def test_returns_generic_when_pickle_corrupted(self) -> None:
        with patch("builtins.open", side_effect=Exception("bad pickle")):
            result = lookup_ingredient("anything")
        assert result["source"] == "generic"

    @pytest.mark.usefixtures("_mock_rag")
    def test_returns_generic_for_no_match_above_threshold(self) -> None:
        """A query that is far from every KB vector still gets generic."""
        tools_module._model_cache = MagicMock()
        tools_module._model_cache.encode.return_value = np.array(
            [[0.0, 0.0, 0.0]], dtype=np.float32
        )
        result = lookup_ingredient("totally_unrelated_xyz")
        assert result["source"] == "generic"

    @pytest.mark.usefixtures("_mock_rag")
    def test_fact_truncated_to_max_chars(self) -> None:
        long_text = "A" * 500
        tools_module._kb_cache = [
            {"text": long_text, "source": "long.md#p0", "vector": np.array([1, 0, 0], dtype=np.float32)},
        ]
        result = lookup_ingredient("niacinamide")
        assert len(result["fact"]) <= 400

    @pytest.mark.usefixtures("_mock_rag")
    def test_no_medical_claims_in_generic_entry(self) -> None:
        banned = {
            "diagnose", "diagnosis", "cure", "treat", "treatment",
            "medication", "prescription", "heal", "therapy",
            "disease", "clinical", "medical",
        }
        banned_phrases = {"guaranteed results", "will cure"}
        fact_lower = _GENERIC_ENTRY["fact"].lower()
        violations = banned.intersection(fact_lower.split())
        phrase_violations = {p for p in banned_phrases if p in fact_lower}
        assert not violations
        assert not phrase_violations


class TestLazyLoading:
    """Verify that importing the module does NOT eagerly load model or KB."""

    def test_kb_cache_is_none_at_import(self) -> None:
        assert tools_module._kb_cache is None

    def test_model_cache_is_none_at_import(self) -> None:
        assert tools_module._model_cache is None

    def test_sentence_transformer_not_imported_at_module_level(self) -> None:
        import sys
        tools_module._kb_cache = None
        tools_module._model_cache = None
        if "sentence_transformers" in sys.modules:
            del sys.modules["sentence_transformers"]
        import importlib
        importlib.reload(tools_module)
        assert tools_module._model_cache is None
        assert tools_module._kb_cache is None


class TestIngredientEvidenceTool:
    """Tests for the @tool wrapper output contract."""

    @pytest.mark.usefixtures("_mock_rag")
    def test_tool_returns_dict_contract(self) -> None:
        result = IngredientEvidenceTool.run(ingredient_name="niacinamide")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"fact", "source"}
        assert isinstance(result["fact"], str)
        assert isinstance(result["source"], str)

    def test_tool_returns_generic_when_kb_unavailable(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError("gone")):
            result = IngredientEvidenceTool.run(ingredient_name="unknown")
        assert result["source"] == "generic"
