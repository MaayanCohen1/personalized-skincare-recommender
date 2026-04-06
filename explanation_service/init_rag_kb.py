"""Build the canonical vectorised knowledge-base pickle for explanation_service.

Merges:
  - ``knowledge_base/*.md`` → guidance chunks (skin conditions, formulation fit,
    routine category fit, framing, safety, ingredient dictionary prose, etc.)
  - ``data/ingredients_raw_data.json`` → structured ingredient evidence rows

Writes to ``explanation_service/data/knowledge_base.pkl`` (same path as the runtime
loader in ``tools``).

Usage:
    python -m explanation_service.init_rag_kb
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from explanation_service.kb_paths import KNOWLEDGE_BASE_PKL

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_KB_DIR = _PACKAGE_DIR / "knowledge_base"
_DATA_DIR = _PACKAGE_DIR / "data"
_INGREDIENTS_JSON = _DATA_DIR / "ingredients_raw_data.json"
_PKL_PATH = KNOWLEDGE_BASE_PKL
_MODEL_NAME = "all-MiniLM-L6-v2"


def _slug_ingredient(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def _read_guidance_chunks(kb_dir: Path) -> list[dict[str, Any]]:
    """Markdown files → guidance records (product-fit, conditions, safety, etc.)."""
    chunks: list[dict[str, Any]] = []
    if not kb_dir.is_dir():
        logger.warning("Knowledge base directory missing: %s", kb_dir)
        return chunks
    for md_file in sorted(kb_dir.glob("*.md")):
        paragraphs = md_file.read_text(encoding="utf-8").split("\n\n")
        for idx, para in enumerate(paragraphs):
            text = para.strip()
            if not text:
                continue
            source = f"{md_file.name}#p{idx}"
            chunks.append({"text": text, "source": source, "kind": "guidance"})
    return chunks


def _read_ingredient_records(json_path: Path) -> list[dict[str, Any]]:
    """JSON ingredient rows → embedded ingredient evidence records."""
    if not json_path.is_file():
        logger.warning("Ingredient JSON not found: %s", json_path)
        return []
    try:
        raw: Any = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", json_path, exc)
        return []

    if isinstance(raw, list):
        records_list = raw
    elif isinstance(raw, dict) and "ingredients" in raw:
        records_list = raw["ingredients"]
    else:
        logger.error("Unexpected JSON structure in %s", json_path)
        return []

    out: list[dict[str, Any]] = []
    for row in records_list:
        if not isinstance(row, dict):
            continue
        name = str(row.get("ingredient", "")).strip()
        fact = str(row.get("fact", "")).strip()
        if not name or not fact:
            continue
        text = f"Ingredient: {name}. Fact: {fact}"
        source = f"ingredient:{_slug_ingredient(name)}"
        out.append({"text": text, "source": source, "kind": "ingredient"})
    logger.info("Loaded %d ingredient records from %s", len(out), json_path)
    return out


def build_kb(
    kb_dir: Path = _KB_DIR,
    ingredients_json: Path = _INGREDIENTS_JSON,
    pkl_path: Path | None = None,
    model_name: str = _MODEL_NAME,
) -> list[dict[str, Any]]:
    """Read guidance + ingredient chunks, encode with SentenceTransformer, save pickle."""
    out_path = pkl_path if pkl_path is not None else _PKL_PATH
    guidance = _read_guidance_chunks(kb_dir)
    ingredients = _read_ingredient_records(ingredients_json)
    chunks: list[dict[str, Any]] = guidance + ingredients

    if not chunks:
        logger.warning("No chunks to encode — nothing written.")
        return []

    logger.info(
        "Encoding %d records (%d guidance + %d ingredient) with %s ...",
        len(chunks),
        len(guidance),
        len(ingredients),
        model_name,
    )
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False)

    records: list[dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        records.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "kind": chunk["kind"],
                "vector": np.array(vec, dtype=np.float32),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(records, f)

    logger.info("Saved %d records to %s", len(records), out_path)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    build_kb()
