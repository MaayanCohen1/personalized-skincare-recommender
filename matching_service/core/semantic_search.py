"""Soft-filter semantic search — ranks safe products by cosine similarity.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed the user's skin
conditions and each product description, then sorts by cosine similarity.

Intentionally pure: no I/O, no RabbitMQ, no HTTP dependencies.
"""

from __future__ import annotations

import logging

from shared.models import Product

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Ranks products by semantic similarity to the user's skin conditions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def rank(
        self,
        skin_conditions: list[str],
        products: list[Product],
    ) -> list[Product]:
        """Return products sorted by cosine similarity to skin_conditions (highest first).

        Steps:
        1. Join skin conditions into one query string.
        2. Encode the query and all product descriptions.
        3. Compute cosine similarity: dot(q, p) / (‖q‖ · ‖p‖).
        4. Return products sorted by score descending.
        """
        import numpy as np  # noqa: PLC0415

        if not products:
            return []

        query_text: str = " ".join(skin_conditions)
        descriptions: list[str] = [p.description for p in products]

        query_vec: np.ndarray = self._model.encode(query_text)
        product_vecs: np.ndarray = self._model.encode(descriptions)

        # Cosine similarity: dot product of each product vector with the query,
        # divided by the product of their L2 norms.
        query_norm = np.linalg.norm(query_vec)
        product_norms = np.linalg.norm(product_vecs, axis=1)

        # Avoid division by zero for zero-magnitude vectors.
        denominator = query_norm * product_norms
        denominator = np.where(denominator == 0, 1e-10, denominator)

        scores: np.ndarray = product_vecs.dot(query_vec) / denominator

        ranked_indices = np.argsort(scores)[::-1]

        logger.debug(
            "SemanticMatcher ranked %d products; top score=%.4f",
            len(products),
            float(scores[ranked_indices[0]]),
        )
        return [products[i] for i in ranked_indices]
