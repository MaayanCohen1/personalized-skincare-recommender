"""Custom CrewAI tools and RAG helpers for the explanation service.

Provides:
  - Ingredient evidence lookup (``IngredientEvidenceTool``) over ``kind=ingredient`` rows.
  - Contextual guidance retrieval over ``kind=guidance`` rows (markdown KB chunks).
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
from crewai.tools import tool

from explanation_service.kb_paths import KNOWLEDGE_BASE_PKL

logger = logging.getLogger(__name__)

_PKL_PATH = KNOWLEDGE_BASE_PKL
_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.15
_TOP_K = 2
_MAX_FACT_CHARS = 400

_CONTEXT_TOP_K = 5
_CONTEXT_SIM_THRESHOLD = 0.10
_CONTEXT_MAX_SNIPPET = 520

_CONTEXT_FALLBACK_TEXT = (
    "No strong knowledge-base matches were retrieved for this query; rely on the "
    "product profile, matcher context, and ingredient research JSON only."
)

_GENERIC_FACT = (
    "This ingredient is commonly found in skincare formulations and is "
    "generally considered suitable for topical use."
)

_kb_cache: list[dict[str, Any]] | None = None
_model_cache: Any | None = None


def _generic_lookup_result() -> dict[str, Any]:
    return {"fact": _GENERIC_FACT, "sources": ["generic"]}


def _load_kb() -> list[dict[str, Any]]:
    """Lazy-load the knowledge base pickle. Returns empty list on failure."""
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    logger.info("Attempting to load KB from %s (exists=%s)", _PKL_PATH, _PKL_PATH.is_file())
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


def _is_ingredient_record(rec: dict[str, Any]) -> bool:
    """Rows used only for ingredient tool RAG."""
    k = rec.get("kind")
    if k == "ingredient":
        return True
    if k == "guidance":
        return False
    # Legacy pickles (no ``kind``): keep previous behavior — search entire index.
    return True


def _is_guidance_record(rec: dict[str, Any]) -> bool:
    """Markdown / framing chunks for contextual product-fit retrieval."""
    k = rec.get("kind")
    if k == "guidance":
        return True
    if k == "ingredient":
        return False
    # Legacy pickles: allow contextual search over the same corpus.
    return True


def _ingredient_kb_subset(kb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in kb if _is_ingredient_record(r)]


def _guidance_kb_subset(kb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in kb if _is_guidance_record(r)]


def build_contextual_query(
    *,
    skin_conditions: list[str],
    product_category: str | None = None,
    product_description: str | None = None,
    product_skin_types: list[str] | None = None,
    product_concerns: list[str] | None = None,
    product_benefits: list[str] | None = None,
    contains_fragrance: bool | None = None,
    contains_alcohol: bool | None = None,
    product_rationale: dict[str, Any] | None = None,
    image_analysis: dict[str, Any] | None = None,
    condition_guidance: str = "",
) -> str:
    """Compose a retrieval query for guidance chunks (conditions, fit, routine role)."""
    parts: list[str] = [
        "Skincare routine explanation. Product fit, category role, formulation texture, "
        "safe wording, what to emphasize and avoid for this user.",
        f"User skin conditions: {', '.join(skin_conditions) if skin_conditions else 'not specified'}.",
    ]
    if condition_guidance.strip():
        parts.append(f"Condition framing notes: {condition_guidance.strip()}")
    cat = (product_category or "").strip()
    if cat:
        parts.append(f"Product category: {cat}.")
    desc = (product_description or "").strip()
    if desc:
        parts.append(f"Product description: {desc[:400]}")
    if product_skin_types:
        parts.append("Product skin types: " + ", ".join(product_skin_types[:8]) + ".")
    if product_concerns:
        parts.append("Product concerns: " + ", ".join(product_concerns[:8]) + ".")
    if product_benefits:
        parts.append("Product benefits: " + ", ".join(product_benefits[:8]) + ".")
    if contains_fragrance is True:
        parts.append("Formulation contains fragrance — note caution for sensitive skin.")
    elif contains_fragrance is False:
        parts.append("Fragrance-free positioning.")
    if contains_alcohol is True:
        parts.append("Contains alcohol — drying alcohol caution where relevant.")
    elif contains_alcohol is False:
        parts.append("No alcohol flagged in catalog metadata.")
    if isinstance(product_rationale, dict) and product_rationale:
        rbits: list[str] = []
        for key in ("role", "category", "fit_score", "strong_actives"):
            val = product_rationale.get(key)
            if val is None or val == []:
                continue
            rbits.append(f"{key}={val!r}")
        if rbits:
            parts.append("Matcher context: " + "; ".join(rbits) + ".")
    if isinstance(image_analysis, dict) and image_analysis:
        vs = image_analysis.get("visual_signals")
        if isinstance(vs, list) and vs:
            parts.append("Visual signals: " + ", ".join(str(x) for x in vs[:6]) + ".")
        isc = image_analysis.get("skin_conditions")
        if isinstance(isc, list) and isc:
            parts.append("Image-linked conditions: " + ", ".join(str(x) for x in isc[:6]) + ".")
    return " ".join(parts)


def retrieve_contextual_guidance(
    *,
    skin_conditions: list[str],
    product_category: str | None = None,
    product_description: str | None = None,
    product_skin_types: list[str] | None = None,
    product_concerns: list[str] | None = None,
    product_benefits: list[str] | None = None,
    contains_fragrance: bool | None = None,
    contains_alcohol: bool | None = None,
    product_rationale: dict[str, Any] | None = None,
    image_analysis: dict[str, Any] | None = None,
    condition_guidance: str = "",
) -> dict[str, Any]:
    """RAG over guidance chunks; returns formatted text + citable source ids.

    Contract: ``{"guidance_text": str, "sources": list[str]}`` — sources are plain
    chunk ids (e.g. ``skin_conditions_guide.md#p2``), same as ingredient sources.
    """
    kb = _load_kb()
    if not kb:
        return {"guidance_text": _CONTEXT_FALLBACK_TEXT, "sources": []}

    pool = _guidance_kb_subset(kb)
    if not pool:
        return {"guidance_text": _CONTEXT_FALLBACK_TEXT, "sources": []}

    model = _load_model()
    query = build_contextual_query(
        skin_conditions=skin_conditions,
        product_category=product_category,
        product_description=product_description,
        product_skin_types=product_skin_types,
        product_concerns=product_concerns,
        product_benefits=product_benefits,
        contains_fragrance=contains_fragrance,
        contains_alcohol=contains_alcohol,
        product_rationale=product_rationale,
        image_analysis=image_analysis,
        condition_guidance=condition_guidance,
    )
    query_vec: np.ndarray = model.encode([query], show_progress_bar=False)[0]

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in pool:
        sim = _cosine_similarity(query_vec, record["vector"])
        scored.append((sim, record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    max_score = scored[0][0] if scored else 0.0
    top = [
        (sim, rec)
        for sim, rec in scored[:_CONTEXT_TOP_K]
        if sim >= _CONTEXT_SIM_THRESHOLD
    ]

    if not top:
        logger.info(
            "retrieve_contextual_guidance -> no hits (max_score=%.4f, threshold=%.2f)",
            max_score,
            _CONTEXT_SIM_THRESHOLD,
        )
        return {"guidance_text": _CONTEXT_FALLBACK_TEXT, "sources": []}

    lines: list[str] = []
    sources_ordered: list[str] = []
    seen: set[str] = set()
    for sim, rec in top:
        sid = str(rec["source"])
        body = str(rec["text"]).strip()
        if len(body) > _CONTEXT_MAX_SNIPPET:
            body = body[: _CONTEXT_MAX_SNIPPET - 3].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{sid}] {body}")
        if sid not in seen:
            seen.add(sid)
            sources_ordered.append(sid)

    logger.info(
        "retrieve_contextual_guidance -> %d chunk(s) (max_score=%.4f)",
        len(top),
        max_score,
    )
    return {
        "guidance_text": "\n\n".join(lines),
        "sources": sources_ordered,
    }


def lookup_ingredient(ingredient_name: str) -> dict[str, Any]:
    """RAG-based lookup: encode query, rank KB chunks by cosine similarity.

    Returns ``{"fact": str, "sources": list[str]}`` — chunk source ids only
    (no similarity scores in the tool contract).
    """
    kb = _load_kb()
    if not kb:
        return _generic_lookup_result()

    pool = _ingredient_kb_subset(kb)
    if not pool:
        return _generic_lookup_result()

    model = _load_model()
    query = f"Ingredient: {ingredient_name.strip()}. Benefits and safety in skincare. General guidance."
    query_vec: np.ndarray = model.encode([query], show_progress_bar=False)[0]

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in pool:
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
        return _generic_lookup_result()

    combined_text = " ".join(rec["text"] for _, rec in top)
    if len(combined_text) > _MAX_FACT_CHARS:
        combined_text = combined_text[:_MAX_FACT_CHARS].rsplit(" ", 1)[0]

    source_ids = [str(rec["source"]) for _, rec in top]
    logger.info(
        "lookup_ingredient(%r) -> %d chunk(s) matched (max_score=%.4f, sources=%s)",
        ingredient_name, len(top), max_score, source_ids,
    )
    return {"fact": combined_text, "sources": source_ids}


@tool("IngredientEvidenceTool")
def IngredientEvidenceTool(ingredient_name: str) -> dict[str, Any]:
    """Look up skincare evidence for a cosmetic ingredient.

    Returns ``fact`` and ``sources`` (list of knowledge-base chunk ids) from
    the vectorised knowledge base.
    """
    return lookup_ingredient(ingredient_name)
