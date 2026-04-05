"""Pydantic v2 models for user-facing form input and catalog validation.

These models capture the raw user preferences submitted via the UI.
They are intentionally separate from the internal ``UserConstraints``
model used by the rules engine so the two layers can evolve independently.

``CatalogProduct`` / ``CatalogSchema`` / ``parse_catalog`` provide
validated access to the full product catalog JSON used by the
offline embedding pipeline and matching service.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Product catalog validation (full JSON shape)
# ---------------------------------------------------------------------------


class ProductFlags(BaseModel):
    contains_fragrance: bool = False
    contains_alcohol: bool = False
    pregnancy_safe: bool | None = None
    nursing_safe: bool | None = None
    sensitive_skin_tested: bool | None = None


class CatalogProduct(BaseModel):
    """Single product entry from the raw catalog JSON."""

    id: str
    name: str
    brand: str
    category: str
    description: str
    skin_types: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    ingredients: list[str] = Field(default_factory=list)
    flags: ProductFlags = Field(default_factory=ProductFlags)


class CatalogSchema(BaseModel):
    """Top-level wrapper matching the ``{"products": [...]}`` JSON shape."""

    products: list[CatalogProduct]


def parse_catalog(raw_json: str | bytes | dict[str, Any]) -> CatalogSchema:
    """Validate raw catalog data and return a typed ``CatalogSchema``.

    Accepts a JSON string, bytes, or an already-parsed dict.
    """
    if isinstance(raw_json, (str, bytes)):
        data = json.loads(raw_json)
    else:
        data = raw_json
    return CatalogSchema.model_validate(data)
