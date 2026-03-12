"""Vision Service — skin image analysis using local HuggingFace models.

Loads two image-classification pipelines once at startup:
  1. Skin-type detection  (dima806/skin_types_image_detection)
  2. Acne severity grading (imfarzanansari/skintelligent-acne)

Public entry point: analyze_skin_image(image_path) -> {"visual_signals": [...]}
"""

from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

_SKIN_TYPE_MODEL = "dima806/skin_types_image_detection"
_ACNE_MODEL = "imfarzanansari/skintelligent-acne"

# ---------------------------------------------------------------------------
# Lazy singletons — models are loaded once on first use, not at import time
# in case the module is imported by tests that don't need GPU/model weights.
# ---------------------------------------------------------------------------
_skin_type_pipe: Pipeline | None = None
_acne_pipe: Pipeline | None = None


def _get_skin_type_pipeline() -> Pipeline:
    global _skin_type_pipe
    if _skin_type_pipe is None:
        logger.info("Loading skin-type model: %s", _SKIN_TYPE_MODEL)
        _skin_type_pipe = pipeline("image-classification", model=_SKIN_TYPE_MODEL)
    return _skin_type_pipe


def _get_acne_pipeline() -> Pipeline:
    global _acne_pipe
    if _acne_pipe is None:
        logger.info("Loading acne model: %s", _ACNE_MODEL)
        _acne_pipe = pipeline("image-classification", model=_ACNE_MODEL)
    return _acne_pipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pipeline(pipe: Pipeline, image: Image.Image) -> list[dict[str, Any]]:
    """Run a HuggingFace image-classification pipeline on a PIL image."""
    results: list[dict[str, Any]] = pipe(image)
    if not isinstance(results, list) or len(results) == 0:
        raise RuntimeError(
            f"Pipeline returned unexpected output (expected non-empty list, got {type(results).__name__})"
        )
    return results


def _parse_skin_type(results: list[dict[str, Any]]) -> str:
    """Extract the top-1 skin-type label, normalised to lowercase."""
    if not results:
        logger.warning("Skin-type model returned no results")
        return "unknown"
    label: str = results[0].get("label", "unknown")
    normalised = label.strip().lower()
    logger.info("Skin-type top-1: %s (score=%.4f)", normalised, results[0].get("score", 0.0))
    return normalised


def _parse_acne_label(raw_label: str) -> str | None:
    """Map the acne model's raw label to a condition string or None.

    Level -1 / Level 0  →  None   (no acne detected)
    Level 1–4           →  "acne"
    """
    normalised = raw_label.strip().lower()
    for absent in ("level -1", "level 0"):
        if absent in normalised:
            return None
    for present in ("level 1", "level 2", "level 3", "level 4"):
        if present in normalised:
            return "acne"
    logger.warning("Unexpected acne label format: %r — treating as no acne", raw_label)
    return None


def _parse_acne(results: list[dict[str, Any]]) -> str | None:
    """Extract the top-1 acne label and map it to a condition or None."""
    if not results:
        logger.warning("Acne model returned no results")
        return None
    label: str = results[0].get("label", "")
    score: float = results[0].get("score", 0.0)
    condition = _parse_acne_label(label)
    logger.info("Acne top-1: %r (score=%.4f) -> %s", label, score, condition or "none")
    return condition


def _open_image(image_path: str) -> Image.Image:
    """Open and validate an image file, raising on invalid/missing paths."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as exc:
        raise ValueError(f"Unreadable image at {image_path}: {exc}") from exc
    # Re-open after verify() because verify() can leave the file in a bad state
    return Image.open(image_path).convert("RGB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_skin_image(image_path: str) -> dict[str, list[str]]:
    """Analyse a skin image and return fused visual signals.

    Returns:
        {"visual_signals": ["oily", "acne"]}   (example)
    """
    image = _open_image(image_path)

    skin_type_results = _run_pipeline(_get_skin_type_pipeline(), image)
    acne_results = _run_pipeline(_get_acne_pipeline(), image)

    signals: list[str] = []

    skin_type = _parse_skin_type(skin_type_results)
    if skin_type and skin_type != "unknown":
        signals.append(skin_type)

    acne_condition = _parse_acne(acne_results)
    if acne_condition:
        signals.append(acne_condition)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_signals: list[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique_signals.append(s)

    logger.info("analyze_skin_image(%s) -> %s", image_path, unique_signals)
    return {"visual_signals": unique_signals}


# ---------------------------------------------------------------------------
# RabbitMQ consumer (placeholder — wired in a future phase)
# ---------------------------------------------------------------------------

def main() -> None:
    # TODO: connect to RabbitMQ, consume routine.requested,
    #       run analyze_skin_image, publish ProfileAnalyzedEvent
    raise NotImplementedError


if __name__ == "__main__":
    main()
