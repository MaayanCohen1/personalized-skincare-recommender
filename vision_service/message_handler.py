"""Pure message-handling layer for image.uploaded event payload."""

from __future__ import annotations

import json
import logging
from typing import Any

from vision_service.main import analyze_skin_image

logger = logging.getLogger(__name__)


def handle_image_uploaded_message(body: bytes) -> dict[str, Any]:
    """Parse image.uploaded payload, run vision analysis, return structured response.

    Args:
        body: Raw JSON bytes from the message queue.

    Returns:
        Dictionary with keys: request_id, image_path, user_preferences,
        visual_signals.  All values are JSON-serializable.

    Raises:
        json.JSONDecodeError: If body is not valid JSON.
        ValueError: If required fields are missing.
    """
    try:
        payload: dict[str, Any] = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from message body: %s", exc)
        raise

    if "request_id" not in payload:
        logger.error("Missing required field: request_id")
        raise ValueError("Missing required field: request_id")

    request_id: str = payload["request_id"]

    if "image_path" not in payload:
        logger.error("Missing required field: image_path (request_id=%s)", request_id)
        raise ValueError("Missing required field: image_path")

    image_path: str = payload["image_path"]
    user_preferences: dict[str, Any] | None = payload.get("user_preferences")

    logger.debug(
        "Handling image.uploaded message: request_id=%s, image_path=%s",
        request_id,
        image_path,
    )

    analysis_result = analyze_skin_image(image_path)
    visual_signals: list[str] = analysis_result["visual_signals"]

    return {
        "request_id": request_id,
        "image_path": image_path,
        "user_preferences": user_preferences,
        "visual_signals": visual_signals,
    }
