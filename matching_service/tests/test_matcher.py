"""Tests for matching_service.core.models — UserPreferences Pydantic model.

Validates enum constraints, optional fields, max-length enforcement,
and round-trip serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from matching_service.core.models import Sensitivity, SkinType, UserPreferences


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oily_prefs() -> UserPreferences:
    """Typical oily-skin user with fragrance sensitivity."""
    return UserPreferences(
        skin_type=SkinType.OILY,
        has_breakouts=True,
        sensitivities=[Sensitivity.FRAGRANCE],
        is_cruelty_free_required=True,
    )


@pytest.fixture
def minimal_prefs() -> UserPreferences:
    """Minimal valid preferences — no sensitivities, no notes."""
    return UserPreferences(
        skin_type=SkinType.NOT_SURE,
        has_breakouts=False,
        sensitivities=[],
        is_cruelty_free_required=False,
    )


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_valid_construction(oily_prefs: UserPreferences) -> None:
    assert oily_prefs.skin_type == SkinType.OILY
    assert oily_prefs.has_breakouts is True
    assert oily_prefs.sensitivities == [Sensitivity.FRAGRANCE]
    assert oily_prefs.is_cruelty_free_required is True
    assert oily_prefs.additional_notes is None


def test_additional_notes_defaults_to_none(minimal_prefs: UserPreferences) -> None:
    assert minimal_prefs.additional_notes is None


def test_additional_notes_accepts_string() -> None:
    prefs = UserPreferences(
        skin_type=SkinType.DRY,
        has_breakouts=False,
        sensitivities=[Sensitivity.NONE],
        is_cruelty_free_required=False,
        additional_notes="I prefer fragrance-free products.",
    )
    assert prefs.additional_notes == "I prefer fragrance-free products."


def test_empty_sensitivities_list_is_valid(minimal_prefs: UserPreferences) -> None:
    assert minimal_prefs.sensitivities == []


def test_multiple_sensitivities() -> None:
    prefs = UserPreferences(
        skin_type=SkinType.COMBINATION,
        has_breakouts=True,
        sensitivities=[Sensitivity.FRAGRANCE, Sensitivity.ALCOHOL],
        is_cruelty_free_required=True,
    )
    assert len(prefs.sensitivities) == 2
    assert Sensitivity.FRAGRANCE in prefs.sensitivities
    assert Sensitivity.ALCOHOL in prefs.sensitivities


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_skin_type_enum_values() -> None:
    assert set(SkinType) == {
        SkinType.DRY,
        SkinType.OILY,
        SkinType.COMBINATION,
        SkinType.NOT_SURE,
    }


def test_sensitivity_enum_values() -> None:
    assert set(Sensitivity) == {
        Sensitivity.FRAGRANCE,
        Sensitivity.ALCOHOL,
        Sensitivity.NONE,
    }


def test_skin_type_string_values() -> None:
    assert SkinType.DRY.value == "dry"
    assert SkinType.OILY.value == "oily"
    assert SkinType.COMBINATION.value == "combination"
    assert SkinType.NOT_SURE.value == "not_sure"


def test_sensitivity_string_values() -> None:
    assert Sensitivity.FRAGRANCE.value == "fragrance"
    assert Sensitivity.ALCOHOL.value == "alcohol"
    assert Sensitivity.NONE.value == "none"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_skin_type_rejected() -> None:
    with pytest.raises(ValidationError):
        UserPreferences(
            skin_type="greasy",  # type: ignore[arg-type]
            has_breakouts=False,
            sensitivities=[],
            is_cruelty_free_required=False,
        )


def test_invalid_sensitivity_rejected() -> None:
    with pytest.raises(ValidationError):
        UserPreferences(
            skin_type=SkinType.DRY,
            has_breakouts=False,
            sensitivities=["sulfate"],  # type: ignore[list-item]
            is_cruelty_free_required=False,
        )


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        UserPreferences(  # type: ignore[call-arg]
            has_breakouts=True,
            sensitivities=[],
            is_cruelty_free_required=False,
        )


def test_none_sensitivity_alone_is_valid() -> None:
    prefs = UserPreferences(
        skin_type=SkinType.DRY,
        has_breakouts=False,
        sensitivities=[Sensitivity.NONE],
        is_cruelty_free_required=False,
    )
    assert prefs.sensitivities == [Sensitivity.NONE]


def test_none_combined_with_other_sensitivity_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        UserPreferences(
            skin_type=SkinType.OILY,
            has_breakouts=False,
            sensitivities=[Sensitivity.NONE, Sensitivity.FRAGRANCE],
            is_cruelty_free_required=False,
        )


def test_none_combined_with_alcohol_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        UserPreferences(
            skin_type=SkinType.OILY,
            has_breakouts=False,
            sensitivities=[Sensitivity.ALCOHOL, Sensitivity.NONE],
            is_cruelty_free_required=False,
        )


def test_additional_notes_exceeding_max_length_rejected() -> None:
    with pytest.raises(ValidationError):
        UserPreferences(
            skin_type=SkinType.OILY,
            has_breakouts=False,
            sensitivities=[],
            is_cruelty_free_required=False,
            additional_notes="x" * 501,
        )


def test_additional_notes_at_max_length_accepted() -> None:
    prefs = UserPreferences(
        skin_type=SkinType.OILY,
        has_breakouts=False,
        sensitivities=[],
        is_cruelty_free_required=False,
        additional_notes="x" * 500,
    )
    assert len(prefs.additional_notes) == 500  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_model_dump_produces_plain_dict(oily_prefs: UserPreferences) -> None:
    data = oily_prefs.model_dump()
    assert isinstance(data, dict)
    assert data["skin_type"] == "oily"
    assert data["sensitivities"] == ["fragrance"]


def test_model_validate_from_dict() -> None:
    raw = {
        "skin_type": "combination",
        "has_breakouts": True,
        "sensitivities": ["alcohol"],
        "is_cruelty_free_required": False,
        "additional_notes": "Prefer lightweight textures.",
    }
    prefs = UserPreferences.model_validate(raw)
    assert prefs.skin_type == SkinType.COMBINATION
    assert prefs.sensitivities == [Sensitivity.ALCOHOL]
    assert prefs.additional_notes == "Prefer lightweight textures."


def test_json_round_trip(oily_prefs: UserPreferences) -> None:
    json_str = oily_prefs.model_dump_json()
    restored = UserPreferences.model_validate_json(json_str)
    assert restored == oily_prefs
