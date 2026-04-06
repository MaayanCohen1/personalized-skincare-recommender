"""Tests for matching_service.core.semantic_search.

sentence_transformers is injected via sys.modules so the real model is never
downloaded.  Precomputed embeddings and catalog JSON are provided via temp
files created by pytest fixtures.
"""

import json
import logging
import pickle
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Block the real sentence_transformers package before any project import.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
_original_st = sys.modules.get("sentence_transformers")
sys.modules["sentence_transformers"] = _st_mock

from shared.models import Product, ProductCategory  # noqa: E402
from matching_service.core.semantic_search import SemanticMatcher, build_query_text  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Precomputed embedding vectors shared across most tests.
# cleanser-a  -> points along +x
# moisturizer-b -> points along +y
# spf-c       -> points along -x
_EMBEDDINGS: dict[str, np.ndarray] = {
    "cleanser-a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "moisturizer-b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "spf-c": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
}

_CATALOG_JSON: dict = {
    "products": [
        {"id": "cleanser-a"},
        {"id": "moisturizer-b"},
        {"id": "spf-c"},
    ]
}


@pytest.fixture(autouse=True)
def _patch_st_module():
    """Keep sentence_transformers mocked for every test, then restore."""
    sys.modules["sentence_transformers"] = _st_mock
    yield
    if _original_st is not None:
        sys.modules["sentence_transformers"] = _original_st
    else:
        sys.modules.pop("sentence_transformers", None)


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "products.json"
    path.write_text(json.dumps(_CATALOG_JSON), encoding="utf-8")
    return path


@pytest.fixture
def embeddings_path(tmp_path):
    path = tmp_path / "product_embeddings.pkl"
    with open(path, "wb") as f:
        pickle.dump(_EMBEDDINGS, f)
    return path


@pytest.fixture
def products() -> list[Product]:
    """Three products whose IDs match the precomputed embeddings."""
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
def matcher(catalog_path, embeddings_path) -> SemanticMatcher:
    """SemanticMatcher backed by temp test data files."""
    m = SemanticMatcher(
        catalog_path=catalog_path,
        embeddings_path=embeddings_path,
    )
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
    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    result = matcher.rank(["sensitive skin"], products)
    assert [p.id for p in result] == ["cleanser-a", "moisturizer-b", "spf-c"]


def test_products_ranked_when_similarity_order_is_reversed(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """When query is opposite, ranking reverses."""
    matcher._model.encode.return_value = np.array([-1.0, 0.0, 0.0])
    result = matcher.rank(["sun damage"], products)
    assert [p.id for p in result] == ["spf-c", "moisturizer-b", "cleanser-a"]


def test_partial_similarity_ordering(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """Fractional cosine scores are ordered correctly."""
    matcher._model.encode.return_value = np.array([0.3, 0.95, 0.0])
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
    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    result = matcher.rank(["redness"], single)
    assert len(result) == 1
    assert result[0].id == "cleanser-a"


# ---------------------------------------------------------------------------
# encode() call contract — only the query is encoded at runtime
# ---------------------------------------------------------------------------


def test_encode_called_once_for_query_only(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """rank() should encode the query only — products use precomputed vectors."""
    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    matcher.rank(["dryness", "redness"], products)

    assert matcher._model.encode.call_count == 1

    first_arg = matcher._model.encode.call_args[0][0]
    assert isinstance(first_arg, str)
    assert "dryness" in first_arg
    assert "redness" in first_arg


def test_encode_receives_rich_query_text(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """rank() should encode the output of build_query_text, not a raw join."""
    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    matcher.rank(["oily"], products)

    query = matcher._model.encode.call_args[0][0]
    assert query == build_query_text(["oily"])
    assert "Skin Types:" in query
    assert "oil control" in query


def test_encode_receives_rich_query_for_unknown_conditions(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """Unknown condition combos should still produce structured query text."""
    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    matcher.rank(["rosacea", "dehydrated"], products)

    query = matcher._model.encode.call_args[0][0]
    assert "Skin Types:" in query
    assert "rosacea" in query
    assert "dehydrated" in query


# ---------------------------------------------------------------------------
# Embedding loading and catalog validation
# ---------------------------------------------------------------------------


def test_embeddings_loaded_from_pickle(
    catalog_path,
    embeddings_path,
) -> None:
    """Constructor should load all embeddings from the pickle file."""
    m = SemanticMatcher(catalog_path=catalog_path, embeddings_path=embeddings_path)
    assert set(m._embeddings.keys()) == {"cleanser-a", "moisturizer-b", "spf-c"}


def test_warns_product_missing_embedding(tmp_path, caplog) -> None:
    """A product in the catalog without an embedding triggers a warning."""
    catalog = {"products": [{"id": "cleanser-a"}, {"id": "extra-product"}]}
    cat_path = tmp_path / "products.json"
    cat_path.write_text(json.dumps(catalog), encoding="utf-8")

    emb_path = tmp_path / "product_embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump({"cleanser-a": np.array([1.0], dtype=np.float32)}, f)

    with caplog.at_level(logging.WARNING):
        SemanticMatcher(catalog_path=cat_path, embeddings_path=emb_path)

    assert "extra-product" in caplog.text
    assert "no embedding" in caplog.text


def test_warns_unknown_embedding_id(tmp_path, caplog) -> None:
    """An embedding for an unknown product id triggers a warning."""
    catalog = {"products": [{"id": "cleanser-a"}]}
    cat_path = tmp_path / "products.json"
    cat_path.write_text(json.dumps(catalog), encoding="utf-8")

    emb_path = tmp_path / "product_embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump({
            "cleanser-a": np.array([1.0], dtype=np.float32),
            "ghost-product": np.array([0.0], dtype=np.float32),
        }, f)

    with caplog.at_level(logging.WARNING):
        SemanticMatcher(catalog_path=cat_path, embeddings_path=emb_path)

    assert "ghost-product" in caplog.text
    assert "unknown product" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Products without embeddings
# ---------------------------------------------------------------------------


def test_product_without_embedding_gets_zero_score(
    matcher: SemanticMatcher,
    products: list[Product],
) -> None:
    """A product with no precomputed embedding should receive score 0.0."""
    unknown = Product(
        id="unknown-x",
        name="Unknown X",
        category=ProductCategory.OTHER,
        ingredients=["water"],
        description="mystery product",
    )
    all_products = products + [unknown]

    matcher._model.encode.return_value = np.array([1.0, 0.0, 0.0])
    scores = matcher.score(["test"], all_products)

    assert scores["unknown-x"] == 0.0
    assert scores["cleanser-a"] > 0

    result = matcher.rank(["test"], all_products)
    assert len(result) == 4


def test_all_products_without_embeddings_preserves_order(
    catalog_path,
    embeddings_path,
) -> None:
    """When no products have embeddings, original order is preserved."""
    m = SemanticMatcher(catalog_path=catalog_path, embeddings_path=embeddings_path)
    m._model.encode.reset_mock()
    m._model.encode.return_value = np.array([1.0, 0.0, 0.0])

    unknown_products = [
        Product(id="x1", name="X1", category=ProductCategory.OTHER,
                ingredients=["water"], description="x1"),
        Product(id="x2", name="X2", category=ProductCategory.OTHER,
                ingredients=["water"], description="x2"),
    ]

    result = m.rank(["anything"], unknown_products)
    assert [p.id for p in result] == ["x1", "x2"]


# ---------------------------------------------------------------------------
# build_query_text — deterministic query expansion
# ---------------------------------------------------------------------------


def test_build_query_text_dry_only() -> None:
    q = build_query_text(["dry"])
    assert "Skin Types: dry" in q
    assert "barrier support" in q
    assert "Avoid:" in q


def test_build_query_text_oily_only() -> None:
    q = build_query_text(["oily"])
    assert "Skin Types: oily" in q
    assert "oil control" in q
    assert "lightweight" in q


def test_build_query_text_oily_acne() -> None:
    q = build_query_text(["oily", "acne"])
    assert "acne" in q
    assert "pore care" in q


def test_build_query_text_oily_acne_order_independent() -> None:
    assert build_query_text(["acne", "oily"]) == build_query_text(["oily", "acne"])


def test_build_query_text_sensitive() -> None:
    q = build_query_text(["sensitive"])
    assert "Skin Types: sensitive" in q
    assert "soothing" in q
    assert "Fragrance-free" in q


def test_build_query_text_dry_sensitive() -> None:
    q = build_query_text(["dry", "sensitive"])
    assert "barrier repair" in q
    assert "Fragrance-free" in q


def test_build_query_text_unknown_conditions_fallback() -> None:
    q = build_query_text(["rosacea"])
    assert "Skin Types: rosacea" in q
    assert "Concerns: rosacea" in q


def test_build_query_text_empty() -> None:
    q = build_query_text([])
    assert "general skincare" in q


def test_build_query_text_case_insensitive() -> None:
    assert build_query_text(["Oily"]) == build_query_text(["oily"])
