"""Pydantic v2 models for user-facing form input.

These models capture the raw user preferences submitted via the UI.
They are intentionally separate from the internal ``UserConstraints``
model used by the rules engine so the two layers can evolve independently.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SkinType(str, Enum):
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    NOT_SURE = "not_sure"


class Sensitivity(str, Enum):
    FRAGRANCE = "fragrance"
    ALCOHOL = "alcohol"
    NONE = "none"


class UserPreferences(BaseModel):
    """Structured representation of the skincare questionnaire form."""

    skin_type: SkinType
    has_breakouts: bool
    sensitivities: list[Sensitivity]
    is_cruelty_free_required: bool
    additional_notes: str | None = Field(default=None, max_length=500)

    @field_validator("sensitivities")
    @classmethod
    def none_must_be_exclusive(cls, v: list[Sensitivity]) -> list[Sensitivity]:
        """``Sensitivity.NONE`` cannot appear alongside other values."""
        if Sensitivity.NONE in v and len(v) > 1:
            raise ValueError(
                "'none' sensitivity cannot be combined with other sensitivities"
            )
        return v
