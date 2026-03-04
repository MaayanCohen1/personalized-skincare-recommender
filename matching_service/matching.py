"""Soft-filter matching — semantic similarity via sentence-transformers.

Future step: ranks safe products against the user's skin profile description.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def rank_by_relevance() -> None:
    # TODO: embed product descriptions + skin_conditions with sentence-transformers,
    #       return ranked list by cosine similarity
    raise NotImplementedError
