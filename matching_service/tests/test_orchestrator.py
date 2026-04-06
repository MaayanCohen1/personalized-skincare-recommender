"""Focused tests for matching_service.core.orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from matching_service.core.models import Sensitivity, SkinType, UserPreferences
from matching_service.core.orchestrator import build_matching_context, match_for_user
from shared.models import Product, ProductCategory, UserConstraints


@pytest.fixture
def catalog() -> list[Product]:
    return [
        Product(
            id="cleanser-1",
            name="Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["water"],
            description="gentle cleanser",
        )
    ]


def _prefs(
    *,
    skin_type: SkinType,
    has_breakouts: bool,
    sensitivities: list[Sensitivity] | None = None,
) -> UserPreferences:
    return UserPreferences(
        skin_type=skin_type,
        has_breakouts=has_breakouts,
        sensitivities=sensitivities or [],
    )


def _install_match_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _spy(
        catalog: list[Product],
        constraints: UserConstraints,
        skin_conditions: list[str] | None = None,
        scorer: Callable[[list[str], list[Product]], dict[str, float]] | None = None,
    ) -> list[Product]:
        captured["catalog"] = catalog
        captured["constraints"] = constraints
        captured["skin_conditions"] = skin_conditions
        captured["scorer"] = scorer
        return list(catalog)

    monkeypatch.setattr("matching_service.core.orchestrator.match_products", _spy)
    return captured


def test_questionnaire_only_input(monkeypatch: pytest.MonkeyPatch, catalog: list[Product]) -> None:
    captured = _install_match_spy(monkeypatch)

    result = match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.DRY, has_breakouts=False),
    )

    assert result == catalog
    assert captured["skin_conditions"] == ["dry"]


def test_vision_only_input(monkeypatch: pytest.MonkeyPatch, catalog: list[Product]) -> None:
    captured = _install_match_spy(monkeypatch)

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.NOT_SURE, has_breakouts=False),
        visual_signals=["oily"],
    )

    assert captured["skin_conditions"] == ["oily"]


def test_merged_input(monkeypatch: pytest.MonkeyPatch, catalog: list[Product]) -> None:
    captured = _install_match_spy(monkeypatch)

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.DRY, has_breakouts=True),
        visual_signals=["oily"],
    )

    # Questionnaire order first, then visual signals.
    assert captured["skin_conditions"] == ["dry", "acne", "oily"]


def test_breakouts_adds_acne(monkeypatch: pytest.MonkeyPatch, catalog: list[Product]) -> None:
    captured = _install_match_spy(monkeypatch)

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.NOT_SURE, has_breakouts=True),
    )

    assert captured["skin_conditions"] == ["acne"]


def test_not_sure_does_not_add_skin_type_signal(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    captured = _install_match_spy(monkeypatch)

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.NOT_SURE, has_breakouts=False),
    )

    assert captured["skin_conditions"] is None


def test_duplicate_signals_are_removed(monkeypatch: pytest.MonkeyPatch, catalog: list[Product]) -> None:
    captured = _install_match_spy(monkeypatch)

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.OILY, has_breakouts=True),
        visual_signals=["oily", "acne", "oily"],
    )

    assert captured["skin_conditions"] == ["oily", "acne"]


def test_invalid_visual_signals_raises_value_error(catalog: list[Product]) -> None:
    with pytest.raises(ValueError, match="Invalid visual signal"):
        match_for_user(
            catalog=catalog,
            user_preferences=_prefs(skin_type=SkinType.DRY, has_breakouts=False),
            visual_signals=["redness"],
        )


def test_empty_catalog_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "matching_service.core.orchestrator.match_products",
        lambda *args, **kwargs: pytest.fail("match_products should not be called"),
    )

    result = match_for_user(
        catalog=[],
        user_preferences=_prefs(
            skin_type=SkinType.DRY,
            has_breakouts=False,
            sensitivities=[Sensitivity.NONE],
        ),
    )

    assert result == []


def test_scorer_is_forwarded_to_rules_engine(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    captured = _install_match_spy(monkeypatch)
    scorer = lambda skin_conditions, products: {p.id: 0.5 for p in products}

    match_for_user(
        catalog=catalog,
        user_preferences=_prefs(skin_type=SkinType.DRY, has_breakouts=False),
        scorer=scorer,
    )

    assert captured["scorer"] is scorer


# ---------------------------------------------------------------------------
# build_matching_context
# ---------------------------------------------------------------------------


def test_build_matching_context_merges_signals_and_builds_constraints() -> None:
    """build_matching_context should return merged skin conditions and UserConstraints."""
    prefs = _prefs(skin_type=SkinType.DRY, has_breakouts=True, sensitivities=[Sensitivity.FRAGRANCE])

    skin_conditions, constraints = build_matching_context(
        prefs, visual_signals=["oily"], request_id="ctx-1"
    )

    assert skin_conditions == ["dry", "acne", "oily"]
    assert constraints.request_id == "ctx-1"
    assert constraints.sensitivities == ["fragrance"]


def test_build_matching_context_none_sensitivity_yields_empty_list() -> None:
    prefs = _prefs(skin_type=SkinType.OILY, has_breakouts=False, sensitivities=[Sensitivity.NONE])

    _, constraints = build_matching_context(prefs, request_id="ctx-2")

    assert constraints.sensitivities == []
