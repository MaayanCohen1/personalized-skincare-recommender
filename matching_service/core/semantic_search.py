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

    def rank(
        self,
        skin_conditions: list[str],
        products: list[Product],
    ) -> list[Product]:
        """Return *products* sorted by cosine similarity to *skin_conditions*.

        Only the user query is encoded at runtime; product vectors come from
        the precomputed embeddings loaded at init.  Products without an
        embedding are appended at the end of the ranked list.
        """
        import numpy as np

        if not products:
            return []

        query_text: str = " ".join(skin_conditions)
        query_vec = self._model.encode(query_text)

        indexed: list[tuple[int, Any]] = []
        unranked: list[Product] = []

        for i, product in enumerate(products):
            emb = self._embeddings.get(product.id)
            if emb is not None:
                indexed.append((i, emb))
            else:
                logger.warning(
                    "No precomputed embedding for product %s; appended to end",
                    product.id,
                )
                unranked.append(product)

        if not indexed:
            return list(products)

        indices, vecs = zip(*indexed)
        product_vecs = np.stack(vecs)

        query_norm = np.linalg.norm(query_vec)
        product_norms = np.linalg.norm(product_vecs, axis=1)
        denominator = query_norm * product_norms
        denominator = np.where(denominator == 0, 1e-10, denominator)

        scores: np.ndarray = product_vecs.dot(query_vec) / denominator
        ranked_order = np.argsort(scores)[::-1]

        ranked = [products[indices[i]] for i in ranked_order]

        logger.debug(
            "SemanticMatcher ranked %d products; top score=%.4f",
            len(ranked),
            float(scores[ranked_order[0]]),
        )

        return ranked + unranked
