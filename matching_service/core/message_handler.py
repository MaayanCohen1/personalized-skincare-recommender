"""Pure message-handling layer for signals.detected event payload."""

from __future__ import annotations

import json
import logging
from typing import Any

from matching_service.core.models import UserPreferences
from matching_service.core.orchestrator import match_for_user
from shared.models import Product

logger = logging.getLogger(__name__)


def _serialize_products(products: list[Product]) -> list[dict[str, Any]]:
    """Convert Product models to JSON-serializable dictionaries."""
    return [p.model_dump(mode="json") for p in products]


def handle_signals_detected_message(body: bytes, catalog: list[Product]) -> dict[str, Any]:
    """Parse signals.detected payload, call match_for_user, return structured response.

    Args:
        body: Raw JSON bytes from the message queue.
        catalog: Available products to match against.

    Returns:
        Dictionary with keys: request_id, visual_signals, matched_products.
        All values are JSON-serializable (matched_products is a list of dicts).

    Raises:
        json.JSONDecodeError: If body is not valid JSON.
        ValueError: If required fields are missing or invalid.
        pydantic.ValidationError: If user_preferences fails validation.
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
    visual_signals: list[str] | None = payload.get("visual_signals")
    user_preferences_raw: dict[str, Any] | None = payload.get("user_preferences")

    if user_preferences_raw is None:
        logger.error("Missing required field: user_preferences (request_id=%s)", request_id)
        raise ValueError("Missing required field: user_preferences")

    try:
        user_preferences = UserPreferences.model_validate(user_preferences_raw)
    except Exception as exc:
        logger.error(
            "Invalid user_preferences payload (request_id=%s): %s",
            request_id,
            exc,
        )
        raise

    logger.debug(
        "Handling signals.detected message: request_id=%s, visual_signals=%s",
        request_id,
        visual_signals,
    )

    matched_products = match_for_user(
        catalog=catalog,
        user_preferences=user_preferences,
        visual_signals=visual_signals,
    )

    return {
        "request_id": request_id,
        "visual_signals": visual_signals,
        "matched_products": _serialize_products(matched_products),
    }
