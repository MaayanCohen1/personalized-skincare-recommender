"""Build a vectorised knowledge-base pickle from the markdown files.

Usage:
    python -m explanation_service.init_rag_kb
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
_PKL_PATH = Path(__file__).resolve().parent / "knowledge_base.pkl"
_MODEL_NAME = "all-MiniLM-L6-v2"


def _read_chunks(kb_dir: Path) -> list[dict[str, str]]:
    """Read all .md files and split into paragraph chunks."""
    chunks: list[dict[str, str]] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        paragraphs = md_file.read_text(encoding="utf-8").split("\n\n")
        for idx, para in enumerate(paragraphs):
            text = para.strip()
            if not text:
                continue
            source = f"{md_file.name}#p{idx}"
            chunks.append({"text": text, "source": source})
    return chunks


def build_kb(
    kb_dir: Path = _KB_DIR,
    pkl_path: Path = _PKL_PATH,
    model_name: str = _MODEL_NAME,
) -> list[dict[str, Any]]:
    """Read markdown chunks, encode with SentenceTransformer, save pickle."""
    chunks = _read_chunks(kb_dir)
    if not chunks:
        logger.warning("No chunks found in %s — nothing to encode.", kb_dir)
        return []

    logger.info("Encoding %d chunks with %s ...", len(chunks), model_name)
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False)

    records: list[dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        records.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "vector": np.array(vec, dtype=np.float32),
            }
        )

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)

    logger.info("Saved %d records to %s", len(records), pkl_path)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    build_kb()
