"""Pure orchestration for preference-driven product matching."""

from __future__ import annotations

from matching_service.core.models import Sensitivity, SkinType, UserPreferences
from matching_service.rules_engine import match_products
from shared.models import Product, UserConstraints

_VALID_SIGNALS: frozenset[str] = frozenset(
    {
        SkinType.DRY.value,
        SkinType.OILY.value,
        SkinType.COMBINATION.value,
        "acne",
    }
)


def _build_constraints(user_preferences: UserPreferences) -> UserConstraints:
    # "none" means no blocked ingredients.
    sensitivities: list[str]
    if Sensitivity.NONE in user_preferences.sensitivities:
        sensitivities = []
    else:
        sensitivities = [s.value for s in user_preferences.sensitivities]

    return UserConstraints(
        request_id="orchestrator",
        sensitivities=sensitivities,
    )


def _questionnaire_signals(user_preferences: UserPreferences) -> list[str]:
    signals: list[str] = []

    if user_preferences.skin_type != SkinType.NOT_SURE:
        signals.append(user_preferences.skin_type.value)
    if user_preferences.has_breakouts:
        signals.append("acne")

    return signals


def _normalize_visual_signals(visual_signals: list[str] | None) -> list[str]:
    if visual_signals is None:
        return []

    normalized: list[str] = []
    for raw_signal in visual_signals:
        signal = raw_signal.strip().lower()
        if signal not in _VALID_SIGNALS:
            raise ValueError(f"Invalid visual signal: {raw_signal!r}")
        normalized.append(signal)

    return normalized


def _merge_signals(base: list[str], extra: list[str]) -> list[str]:
    # Keep first-seen order and remove duplicates.
    seen: set[str] = set()
    merged: list[str] = []
    for signal in [*base, *extra]:
        if signal not in seen:
            seen.add(signal)
            merged.append(signal)
    return merged


def match_for_user(
    catalog: list[Product],
    user_preferences: UserPreferences,
    visual_signals: list[str] | None = None,
) -> list[Product]:
    """Translate user inputs into matcher inputs and return matched products."""
    normalized_visual_signals = _normalize_visual_signals(visual_signals)
    if not catalog:
        return []

    questionnaire_signals = _questionnaire_signals(user_preferences)
    merged_signals = _merge_signals(questionnaire_signals, normalized_visual_signals)
    constraints = _build_constraints(user_preferences)

    return match_products(
        catalog=catalog,
        constraints=constraints,
        skin_conditions=merged_signals or None,
    )
