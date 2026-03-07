"""Custom CrewAI tools for the explanation service.

Provides ingredient evidence lookup backed by a Naive RAG retriever over a
vectorised knowledge base (sentence-transformers + numpy, no external DB).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from crewai.tools import tool

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH = os.path.join(_THIS_DIR, "knowledge_base.pkl")
_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.15
_TOP_K = 2
_MAX_FACT_CHARS = 400

_GENERIC_ENTRY: dict[str, str] = {
    "fact": (
        "This ingredient is commonly found in skincare formulations and is "
        "generally considered suitable for topical use."
    ),
    "source": "generic",
}

_kb_cache: list[dict[str, Any]] | None = None
_model_cache: Any | None = None


def _load_kb() -> list[dict[str, Any]]:
    """Lazy-load the knowledge base pickle. Returns empty list on failure."""
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    logger.info("Attempting to load KB from %s (exists=%s)", _PKL_PATH, os.path.isfile(_PKL_PATH))
    try:
        with open(_PKL_PATH, "rb") as f:
            _kb_cache = pickle.load(f)
        logger.info("Loaded KB with %d chunks from %s", len(_kb_cache), _PKL_PATH)
    except Exception:
        logger.warning(
            "Could not load knowledge_base.pkl at %s — falling back to generic.",
            _PKL_PATH,
        )
        _kb_cache = []
    return _kb_cache


def _load_model() -> Any:
    """Lazy-load the SentenceTransformer model. Never loaded at import time."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    from sentence_transformers import SentenceTransformer

    _model_cache = SentenceTransformer(_MODEL_NAME)
    logger.info("Loaded SentenceTransformer model %s", _MODEL_NAME)
    return _model_cache


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors using pure numpy."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def lookup_ingredient(ingredient_name: str) -> dict[str, str]:
    """RAG-based lookup: encode query, rank KB chunks by cosine similarity."""
    kb = _load_kb()
    if not kb:
        return dict(_GENERIC_ENTRY)

    model = _load_model()
    query = f"Ingredient: {ingredient_name.strip()}. Benefits and safety in skincare. General guidance."
    query_vec: np.ndarray = model.encode([query], show_progress_bar=False)[0]

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in kb:
        sim = _cosine_similarity(query_vec, record["vector"])
        scored.append((sim, record))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    max_score = scored[0][0] if scored else 0.0
    top = [(sim, rec) for sim, rec in scored[:_TOP_K] if sim >= _SIMILARITY_THRESHOLD]

    if not top:
        logger.info(
            "lookup_ingredient(%r) -> GENERIC fallback (max_score=%.4f, threshold=%.2f)",
            ingredient_name, max_score, _SIMILARITY_THRESHOLD,
        )
        return dict(_GENERIC_ENTRY)

    combined_text = " ".join(rec["text"] for _, rec in top)
    if len(combined_text) > _MAX_FACT_CHARS:
        combined_text = combined_text[:_MAX_FACT_CHARS].rsplit(" ", 1)[0]

    sources = "; ".join(rec["source"] for _, rec in top)

    logger.info(
        "lookup_ingredient(%r) -> %d chunk(s) matched (max_score=%.4f, sources=%s)",
        ingredient_name, len(top), max_score, sources,
    )
    return {"fact": combined_text, "source": sources}


@tool("IngredientEvidenceTool")
def IngredientEvidenceTool(ingredient_name: str) -> dict[str, str]:
    """Look up skincare evidence for a cosmetic ingredient.

    Returns a dictionary with 'fact' and 'source' keys describing the
    ingredient's role in skincare, backed by a vectorised knowledge base
    using semantic similarity search.
    """
    return lookup_ingredient(ingredient_name)
