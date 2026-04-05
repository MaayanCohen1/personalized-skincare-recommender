"""Pure message-handling layer for signals.detected event payload.

Parses an incoming signals.detected message, runs matching, and returns
a ``routine.matched`` envelope ready for publishing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from matching_service.core.models import UserPreferences
from matching_service.core.orchestrator import build_matching_context, match_for_user
from matching_service.rules_engine import build_routine_rationale, filter_safe_products
from shared.models import Product, RoutineMatchedEvent, SkinProfile, UserConstraints

logger = logging.getLogger(__name__)


def handle_signals_detected_message(
    body: bytes,
    catalog: list[Product],
    scorer: Callable[[list[str], list[Product]], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Parse signals.detected payload, match products, return routine.matched envelope.

    Args:
        body: Raw JSON bytes from the message queue.
        catalog: Available products to match against.
        scorer: Optional semantic scoring callable (product_id -> similarity).

    Returns:
        Dictionary with ``request_id`` and ``event`` (a JSON-serializable
        ``RoutineMatchedEvent``).

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
        scorer=scorer,
    )

    skin_conditions, constraints = build_matching_context(
        user_preferences,
        visual_signals=visual_signals,
        request_id=request_id,
    )

    event = RoutineMatchedEvent(
        matched_products=matched_products,
        profile=SkinProfile(
            request_id=request_id,
            skin_conditions=skin_conditions,
        ),
        constraints=UserConstraints(
            request_id=request_id,
            sensitivities=constraints.sensitivities,
        ),
    )

    safe_products = filter_safe_products(catalog, constraints)
    used_semantic = scorer is not None

    image_analysis: dict[str, Any] = {
        "visual_signals": visual_signals or [],
        "skin_conditions": skin_conditions,
        "source": (
            "vision_service + questionnaire"
            if visual_signals
            else "questionnaire"
        ),
    }

    routine_rationale = build_routine_rationale(
        catalog=catalog,
        safe_products=safe_products,
        selected=matched_products,
        skin_conditions=skin_conditions,
        used_semantic_ranking=used_semantic,
    )

    return {
        "request_id": request_id,
        "event": event.model_dump(mode="json"),
        "image_analysis": image_analysis,
        "routine_rationale": routine_rationale,
    }
