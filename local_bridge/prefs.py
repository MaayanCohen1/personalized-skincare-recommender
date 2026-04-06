"""Parse demo form fields into ``UserPreferences`` (testable, no FastAPI)."""

from __future__ import annotations

import json
from typing import Any

from matching_service.core.models import Sensitivity, SkinType, UserPreferences


def parse_has_breakouts(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes", "on")


def parse_demo_preferences(
    skin_type: str,
    has_breakouts: str,
    sensitivities_json: str,
) -> UserPreferences:
    """Build preferences from bridge multipart fields."""
    raw = json.loads(sensitivities_json)
    if not isinstance(raw, list):
        raise ValueError("sensitivities must be a JSON array of strings")
    sens_items: list[Any] = list(raw)
    sensitivities = [Sensitivity(str(x)) for x in sens_items]
    return UserPreferences(
        skin_type=SkinType(skin_type.strip()),
        has_breakouts=parse_has_breakouts(has_breakouts),
        sensitivities=sensitivities,
    )
