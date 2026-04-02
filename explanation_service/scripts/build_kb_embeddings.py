"""Offline embedding generation for the ingredient knowledge base.

Reads raw ingredient records from JSON, encodes each entry with
sentence-transformers, and saves a pickle compatible with the
``explanation_service.tools`` RAG loader.

Usage:
    python -m explanation_service.scripts.build_kb_embeddings
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
_RAW_JSON_PATH: Path = _DATA_DIR / "ingredients_raw_data.json"
_PKL_PATH: Path = _DATA_DIR / "knowledge_base.pkl"
_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _load_raw_records(json_path: Path) -> list[dict[str, Any]]:
    """Read and validate the raw ingredient JSON file.

    Returns an empty list and logs an error when the file is missing
    or contains invalid JSON — callers should treat that as a clean abort.
    """
    if not json_path.is_file():
        logger.error("Raw data file not found: %s", json_path)
        return []

    try:
        raw = json_path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", json_path, exc)
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "ingredients" in data:
        return data["ingredients"]

    logger.error("Unexpected JSON structure in %s (expected list or {ingredients: [...]})", json_path)
    return []


def _build_text(record: dict[str, Any]) -> str:
    """Build the embedding input text for a single ingredient record."""
    ingredient: str = record.get("ingredient", "").strip()
    fact: str = record.get("fact", "").strip()
    return f"Ingredient: {ingredient}. Fact: {fact}"


def _generate_embeddings(
    records: list[dict[str, Any]],
    model_name: str = _MODEL_NAME,
) -> list[dict[str, Any]]:
    """Encode each record and return KB entries ready for pickling."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading SentenceTransformer model: %s", model_name)
    model = SentenceTransformer(model_name)

    texts: list[str] = [_build_text(r) for r in records]

    logger.info("Encoding %d records ...", len(texts))
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False)

    kb_entries: list[dict[str, Any]] = []
    for record, text, vec in zip(records, texts, vectors):
        kb_entries.append(
            {
                "ingredient": record.get("ingredient", ""),
                "fact": record.get("fact", ""),
                "source": record.get("source", "generic"),
                "text": text,
                "vector": np.array(vec, dtype=np.float32),
            }
        )

    # Log a few samples for verification.
    for entry in kb_entries[:3]:
        logger.info("  sample text: %s", entry["text"][:120])

    return kb_entries


def _save_pickle(entries: list[dict[str, Any]], pkl_path: Path) -> None:
    """Persist KB entries as a pickle file."""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as fh:
        pickle.dump(entries, fh)
    logger.info("Saved %d KB entries to %s", len(entries), pkl_path)


def main(
    json_path: Path = _RAW_JSON_PATH,
    pkl_path: Path = _PKL_PATH,
    model_name: str = _MODEL_NAME,
) -> None:
    """End-to-end pipeline: load JSON -> embed -> save pickle."""
    records = _load_raw_records(json_path)
    if not records:
        logger.warning("No records to process — exiting.")
        return

    logger.info("Loaded %d raw ingredient records from %s", len(records), json_path)
    entries = _generate_embeddings(records, model_name=model_name)
    _save_pickle(entries, pkl_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
