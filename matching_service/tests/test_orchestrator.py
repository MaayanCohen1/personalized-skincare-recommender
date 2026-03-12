"""Focused tests for matching_service.core.orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from matching_service.core.models import Sensitivity, SkinType, UserPreferences
from matching_service.core.orchestrator import match_for_user
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
        is_cruelty_free_required=False,
    )


def _install_match_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _spy(
        catalog: list[Product],
        constraints: UserConstraints,
        skin_conditions: list[str] | None = None,
        ranker: Callable[[list[str], list[Product]], list[Product]] | None = None,
    ) -> list[Product]:
        captured["catalog"] = catalog
        captured["constraints"] = constraints
        captured["skin_conditions"] = skin_conditions
        captured["ranker"] = ranker
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
