"""Offline embedding generation for the product catalog.

Loads the raw catalog JSON, validates it via the existing parsing layer,
builds a rich text representation per product, encodes embeddings with
sentence-transformers, and saves the result as a pickle file.

Usage:
    python -m matching_service.scripts.build_product_embeddings
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from matching_service.core.models import (
    CatalogProduct,
    CatalogSchema,
    ProductFlags,
    parse_catalog,
)

logger = logging.getLogger(__name__)

_CATALOG_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "products.json"
_OUTPUT_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "product_embeddings.pkl"
_INPUTS_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "product_embedding_inputs.json"
_MODEL_NAME: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_raw_json(path: Path) -> dict[str, Any]:
    """Read and parse the catalog JSON file."""
    logger.info("Loading catalog JSON from %s", path)
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


# Substring keywords used to identify semantically important ingredients.
# Matched ingredients keep their original INCI order (descending concentration).
_ACTIVE_KEYWORDS: tuple[str, ...] = (
    # Retinoids
    "retinol", "retinal", "retinoic", "retinyl",
    # Exfoliating acids
    "azelaic acid", "salicylic acid", "glycolic acid", "lactic acid",
    "mandelic acid",
    # Vitamin C derivatives
    "ascorbic", "ascorbyl",
    # Brightening / anti-aging
    "niacinamide", "bakuchiol", "arbutin",
    # Hydration / barrier
    "hyaluronic", "sodium hyaluronate", "ceramide", "panthenol",
    "squalane", "squalene", "phytosphingosine", "cholesterol",
    # Soothing / botanical actives
    "centella", "madecassoside", "allantoin", "bisabolol", "aloe",
    # Peptides / proteins
    "peptide", "collagen", "elastin",
    # Mineral UV filters
    "zinc oxide", "titanium dioxide",
    # Chemical UV filters
    "ethylhexyl triazone", "bis-ethylhexyloxyphenol",
    "methylene bis-benzotriazolyl", "butyl methoxydibenzoylmethane",
    "homosalate", "octinoxate", "octocrylene",
    "phenylbenzimidazole sulfonic acid", "diethylhexyl butamido triazone",
    # Antioxidants
    "tocopherol", "tocopheryl",
)

_FILLER_INGREDIENTS: frozenset[str] = frozenset({
    "aqua", "water", "parfum", "fragrance", "perfume",
})


def _humanize(token: str) -> str:
    """Convert underscore-separated tokens to natural English phrases."""
    return token.replace("_", " ")


def _join_natural(items: list[str]) -> str:
    """Join items with commas and 'and' before the last item."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _select_key_ingredients(
    ingredients: list[str],
    *,
    max_count: int = 10,
) -> list[str]:
    """Select the most informative ingredients, prioritizing known actives.

    Matched active ingredients come first (in their original INCI order),
    followed by non-filler ingredients until *max_count* is reached.
    """
    priority: list[str] = []
    rest: list[str] = []

    for ing in ingredients:
        lower = ing.lower()
        if lower in _FILLER_INGREDIENTS:
            continue
        if any(kw in lower for kw in _ACTIVE_KEYWORDS):
            priority.append(ing)
        else:
            rest.append(ing)

    selected = priority[:max_count]
    remaining_slots = max_count - len(selected)
    if remaining_slots > 0:
        selected.extend(rest[:remaining_slots])
    return selected


def _flag_sentences(flags: ProductFlags) -> list[str]:
    """Build natural-language sentences from product flags."""
    sentences: list[str] = []
    if flags.contains_fragrance:
        sentences.append("Contains fragrance.")
    if flags.contains_alcohol:
        sentences.append("Contains alcohol.")
    if flags.pregnancy_safe:
        sentences.append("Pregnancy-safe.")
    if flags.nursing_safe:
        sentences.append("Nursing-safe.")
    if flags.sensitive_skin_tested:
        sentences.append("Tested for sensitive skin.")
    return sentences


def _build_product_text(product: CatalogProduct) -> str:
    """Build a compact, semantically rich text string for embedding."""
    parts: list[str] = [
        f"Brand: {product.brand}.",
        f"Category: {product.category}.",
        f"Name: {product.name}.",
        product.description.rstrip(".") + ".",
    ]

    if product.concerns:
        humanized = [_humanize(c) for c in product.concerns]
        parts.append(f"Helps with {_join_natural(humanized)}.")

    if product.benefits:
        humanized = [_humanize(b) for b in product.benefits]
        parts.append(f"Benefits include {_join_natural(humanized)}.")

    key_ings = _select_key_ingredients(product.ingredients)
    if key_ings:
        parts.append(f"Key ingredients: {', '.join(key_ings)}.")

    parts.extend(_flag_sentences(product.flags))

    return " ".join(parts)


def _generate_embeddings(
    input_texts: dict[str, str],
) -> dict[str, np.ndarray]:
    """Encode each product and return a mapping of id -> embedding vector."""
    from sentence_transformers import SentenceTransformer

    ids = list(input_texts.keys())
    texts = list(input_texts.values())

    for sample_id, sample_text in list(input_texts.items())[:3]:
        logger.info("Sample embedding input [%s]: %s", sample_id, sample_text)

    logger.info("Loading embedding model: %s", _MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)

    logger.info("Generating embeddings for %d products", len(texts))
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False)

    return {pid: np.array(vec, dtype=np.float32) for pid, vec in zip(ids, vectors)}


def _build_input_texts(catalog: CatalogSchema) -> dict[str, str]:
    """Build a mapping of product id -> concatenated embedding input text."""
    return {p.id: _build_product_text(p) for p in catalog.products}


def _save_input_texts(texts: dict[str, str], path: Path) -> None:
    """Save the embedding input texts as pretty-printed JSON for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    logger.info("Saved embedding input texts to %s", path)


def _save_pickle(embeddings: dict[str, np.ndarray], path: Path) -> None:
    """Persist the embeddings dictionary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Saved embeddings to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not _CATALOG_PATH.is_file():
        logger.error("Catalog file not found: %s", _CATALOG_PATH)
        sys.exit(1)

    try:
        raw = _load_raw_json(_CATALOG_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read catalog JSON: %s", exc)
        sys.exit(1)

    try:
        logger.info("Validating catalog schema")
        catalog = parse_catalog(raw)
    except Exception as exc:
        logger.error("Catalog validation failed: %s", exc)
        sys.exit(1)

    logger.info("Catalog contains %d products", len(catalog.products))

    input_texts = _build_input_texts(catalog)
    _save_input_texts(input_texts, _INPUTS_PATH)

    try:
        embeddings = _generate_embeddings(input_texts)
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        sys.exit(1)

    _save_pickle(embeddings, _OUTPUT_PATH)
    logger.info(
        "Done — %d product embeddings saved to %s",
        len(embeddings),
        _OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
