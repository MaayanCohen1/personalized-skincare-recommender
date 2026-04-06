"""Offline embedding generation — **deprecated path**.

The canonical runtime KB is built by ``explanation_service.init_rag_kb``, which merges
markdown guidance chunks and ``ingredients_raw_data.json`` into one pickle.

This module now delegates to ``init_rag_kb.build_kb`` so Docker and legacy scripts keep working.

Usage:
    python -m explanation_service.scripts.build_kb_embeddings
"""

from __future__ import annotations

import logging

from explanation_service.init_rag_kb import build_kb

logger = logging.getLogger(__name__)


def main() -> None:
    """Build the unified guidance + ingredient KB pickle."""
    logger.warning(
        "build_kb_embeddings is deprecated — invoking init_rag_kb.build_kb (markdown + JSON).",
    )
    build_kb()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
