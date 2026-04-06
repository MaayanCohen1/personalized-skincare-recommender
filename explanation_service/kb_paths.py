"""Canonical on-disk paths for explanation_service assets (single source of truth)."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# Must match tools RAG loader and offline KB build scripts.
KNOWLEDGE_BASE_PKL = _PACKAGE_DIR / "data" / "knowledge_base.pkl"
