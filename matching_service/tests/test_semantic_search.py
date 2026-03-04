"""TDD test suite for matching_service.core.semantic_search — RED phase.

All tests FAIL until SemanticMatcher.rank() is implemented.

sentence_transformers is injected via sys.modules so the real model is never
downloaded. Individual tests control encode() return values with side_effect,
giving deterministic cosine-similarity results without network or GPU.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Block the real sentence_transformers package before any project import fires.
# This lets SemanticMatcher.__init__ run without downloading any model.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)

from shared.models import Product, ProductCategory  # noqa: E402
from matching_service.core.semantic_search import SemanticMatcher  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def products() -> list[Product]:
    """Three products whose descriptions map to basis vectors in tests."""
    return [
        Product(
            id="cleanser-a",
            name="Cleanser A",
            category=ProductCategory.CLEANSER,
            ingredients=["water"],
            description="gentle cleanser for sensitive skin",
        ),
        Product(
            id="moisturizer-b",
            name="Moisturizer B",
            category=ProductCategory.MOISTURIZER,
            ingredients=["glycerin"],
            description="hydrating moisturizer for dry skin",
        ),
        Product(
            id="spf-c",
            name="SPF C",
            category=ProductCategory.SPF,
            ingredients=["zinc oxide"],
            description="daily sunscreen for sun protection",
        ),
    ]


@pytest.fixture
def matcher() -> SemanticMatcher:
    """SemanticMatcher whose internal model is already a MagicMock.

    sentence_transformers is patched via sys.modules above, so __init__
    assigns a MagicMock to self._model — no real model is loaded.
    Call history is reset each time so call_args_list[0] always refers to
    the first encode() call made within the current test.
    """
    m = SemanticMatcher()
    m._model.encode.reset_mock()
    return m


# ---------------------------------------------------------------------------
# Ranking correctness
# ---------------------------------------------------------------------------


def test_products_ranked_highest_similarity_first(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """Product with the highest cosine similarity should come first."""
    # cleanser-a: sim=1.0, moisturizer-b: sim=0.0, spf-c: sim=-1.0
    matcher._model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),                                    # query
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),  # products
    ]
    result = matcher.rank(["sensitive skin"], products)
    assert [p.id for p in result] == ["cleanser-a", "moisturizer-b", "spf-c"]


def test_products_ranked_when_similarity_order_is_reversed(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """When similarities are in reverse order the ranking must also reverse."""
    # spf-c: sim=1.0, moisturizer-b: sim=0.0, cleanser-a: sim=-1.0
    matcher._model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ]
    result = matcher.rank(["sun damage"], products)
    assert [p.id for p in result] == ["spf-c", "moisturizer-b", "cleanser-a"]


def test_partial_similarity_ordering(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """Fractional cosine scores are ordered correctly."""
    # moisturizer-b ~0.894, cleanser-a ~0.707, spf-c ~0.0
    query_vec = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    product_vecs = np.array([
        [1.0, 0.0, 0.0],   # cleanser-a
        [1.0, 1.0, 0.0],   # moisturizer-b — closest
        [0.0, 0.0, 1.0],   # spf-c — orthogonal
    ])
    matcher._model.encode.side_effect = [query_vec, product_vecs]
    result = matcher.rank(["dryness"], products)
    assert result[0].id == "moisturizer-b"
    assert result[-1].id == "spf-c"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_product_list_returns_empty(matcher: SemanticMatcher) -> None:
    result = matcher.rank(["oily skin"], [])
    assert result == []


def test_single_product_is_returned(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    single = [products[0]]
    matcher._model.encode.side_effect = [
        np.array([1.0, 0.0]),
        np.array([[0.8, 0.6]]),
    ]
    result = matcher.rank(["redness"], single)
    assert len(result) == 1
    assert result[0].id == "cleanser-a"


# ---------------------------------------------------------------------------
# encode() call contract
# ---------------------------------------------------------------------------


def test_encode_receives_joined_skin_conditions(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """rank() must join skin_conditions into one string for the first encode call."""
    matcher._model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    matcher.rank(["dryness", "redness"], products)

    first_arg = matcher._model.encode.call_args_list[0][0][0]
    assert isinstance(first_arg, str)
    assert "dryness" in first_arg
    assert "redness" in first_arg


def test_encode_receives_product_descriptions(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """rank() must pass a list of description strings as the second encode call."""
    matcher._model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    matcher.rank(["oily"], products)

    second_arg = matcher._model.encode.call_args_list[1][0][0]
    assert second_arg == [p.description for p in products]
