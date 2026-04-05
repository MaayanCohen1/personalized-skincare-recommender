"""Soft-filter semantic search — ranks safe products by cosine similarity.

Uses precomputed product embeddings from the offline pipeline and
sentence-transformers (all-MiniLM-L6-v2) to embed user queries at runtime.
Product vectors are never recomputed — only the query is encoded per call.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from shared.models import Product

logger = logging.getLogger(__name__)

_DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_CATALOG_PATH: Path = _DATA_DIR / "products.json"
_DEFAULT_EMBEDDINGS_PATH: Path = _DATA_DIR / "product_embeddings.pkl"


_CONDITION_QUERIES: dict[frozenset[str], str] = {
    frozenset({"dry"}): (
        "Skin Types: dry. "
        "Concerns: dryness, dehydration, flaking. "
        "Benefits: barrier support, soothing, deep hydration, nourishing. "
        "Avoid: matte finish, oil control, stripping cleansers."
    ),
    frozenset({"oily"}): (
        "Skin Types: oily. "
        "Concerns: excess oil, shine, large pores. "
        "Benefits: lightweight hydration, oil control, mattifying, balanced moisture. "
        "Avoid: heavy rich creams, dry skin focused products."
    ),
    frozenset({"oily", "acne"}): (
        "Skin Types: oily, acne prone. "
        "Concerns: acne, breakouts, clogged pores, excess oil. "
        "Benefits: gentle cleansing, pore care, lightweight hydration, non-comedogenic. "
        "Avoid: heavy rich products, comedogenic ingredients."
    ),
    frozenset({"sensitive"}): (
        "Skin Types: sensitive. "
        "Concerns: irritation, redness, reactive skin. "
        "Benefits: soothing, gentle cleansing, barrier support, calming. "
        "Fragrance-free preference."
    ),
    frozenset({"dry", "sensitive"}): (
        "Skin Types: dry, sensitive. "
        "Concerns: dryness, irritation, redness, compromised barrier. "
        "Benefits: deep hydration, soothing, barrier repair, gentle cleansing. "
        "Fragrance-free preference. Avoid: stripping, matte, oil control."
    ),
}


def build_query_text(skin_conditions: list[str]) -> str:
    """Build a keyword-rich, structured query from detected skin conditions.

    Maps known condition combinations to dense tagged text that mirrors the
    product embedding format.  Falls back to a reasonable generic query for
    unknown combinations.
    """
    if not skin_conditions:
        return "general skincare routine"

    key = frozenset(c.lower().strip() for c in skin_conditions)
    if key in _CONDITION_QUERIES:
        return _CONDITION_QUERIES[key]

    humanized = [c.replace("_", " ") for c in skin_conditions]
    skin_types_tag = f"Skin Types: {', '.join(humanized)}."
    concerns_tag = f"Concerns: {', '.join(humanized)}."
    return f"{skin_types_tag} {concerns_tag} Benefits: suitable skincare routine."


class SemanticMatcher:
    """Ranks products by semantic similarity using precomputed embeddings.

    At init time the matcher loads a ``{product_id: vector}`` mapping
    produced by the offline embedding pipeline and cross-validates it
    against the product catalog.  At rank time only the user query is
    embedded; product vectors are looked up by id.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        catalog_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

        self._catalog_path = catalog_path or _DEFAULT_CATALOG_PATH
        self._embeddings_path = embeddings_path or _DEFAULT_EMBEDDINGS_PATH

        self._embeddings: dict[str, Any] = self._load_embeddings()
        self._validate_catalog_consistency()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_embeddings(self) -> dict[str, Any]:
        """Load the precomputed ``{product_id: vector}`` mapping from disk."""
        logger.info("Loading precomputed embeddings from %s", self._embeddings_path)
        with open(self._embeddings_path, "rb") as f:
            embeddings: dict[str, Any] = pickle.load(f)  # noqa: S301
        logger.info("Loaded %d product embeddings", len(embeddings))
        return embeddings

    def _validate_catalog_consistency(self) -> None:
        """Log warnings for products/embeddings that don't match up."""
        with open(self._catalog_path, encoding="utf-8") as f:
            raw = json.load(f)

        catalog_ids = {p["id"] for p in raw.get("products", [])}
        embedding_ids = set(self._embeddings.keys())

        for pid in sorted(catalog_ids - embedding_ids):
            logger.warning("Product %s exists in catalog but has no embedding", pid)
        for pid in sorted(embedding_ids - catalog_ids):
            logger.warning("Embedding exists for unknown product id: %s", pid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        skin_conditions: list[str],
        products: list[Product],
    ) -> dict[str, float]:
        """Return ``{product.id: cosine_similarity}`` for each product.

        Products without a precomputed embedding receive a score of ``0.0``.
        """
        import numpy as np

        if not products:
            return {}

        query_text: str = build_query_text(skin_conditions)
        query_vec = self._model.encode(query_text)

        result: dict[str, float] = {}
        indexed: list[tuple[str, Any]] = []

        for product in products:
            emb = self._embeddings.get(product.id)
            if emb is not None:
                indexed.append((product.id, emb))
            else:
                logger.warning(
                    "No precomputed embedding for product %s; score=0.0",
                    product.id,
                )
                result[product.id] = 0.0

        if indexed:
            ids, vecs = zip(*indexed)
            product_vecs = np.stack(vecs)
            query_norm = np.linalg.norm(query_vec)
            product_norms = np.linalg.norm(product_vecs, axis=1)
            denominator = query_norm * product_norms
            denominator = np.where(denominator == 0, 1e-10, denominator)
            scores: np.ndarray = product_vecs.dot(query_vec) / denominator
            for pid, s in zip(ids, scores):
                result[pid] = float(s)

        return result

    def rank(
        self,
        skin_conditions: list[str],
        products: list[Product],
    ) -> list[Product]:
        """Return *products* sorted by cosine similarity to *skin_conditions*.

        Convenience wrapper around :meth:`score`.
        """
        if not products:
            return []

        scores = self.score(skin_conditions, products)
        ranked = sorted(products, key=lambda p: scores.get(p.id, 0.0), reverse=True)

        top_score = scores.get(ranked[0].id, 0.0) if ranked else 0.0
        logger.debug(
            "SemanticMatcher ranked %d products; top score=%.4f",
            len(ranked),
            top_score,
        )

        return ranked
