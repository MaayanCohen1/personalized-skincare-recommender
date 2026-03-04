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
        """Return products sorted by cosine similarity to skin_conditions (highest first)."""
        # TODO: implement semantic ranking
        raise NotImplementedError
