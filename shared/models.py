"""Shared Pydantic domain models and RabbitMQ event payloads for SafeGlow AI."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ProductCategory(str, Enum):
    CLEANSER = "CLEANSER"
    MOISTURIZER = "MOISTURIZER"
    SPF = "SPF"
    SERUM = "SERUM"
    TONER = "TONER"
    OTHER = "OTHER"

    # Priority order used by the rules engine to cap routine size.
    PRIORITY_ORDER: list[str] = [
        "CLEANSER",
        "MOISTURIZER",
        "SPF",
        "SERUM",
        "TONER",
        "OTHER",
    ]


class Product(BaseModel):
    id: str
    name: str
    category: ProductCategory
    ingredients: list[str]
    description: str

    @field_validator("ingredients", mode="before")
    @classmethod
    def normalize_ingredients(cls, v: list[str]) -> list[str]:
        return [ingredient.lower() for ingredient in v]


class UserConstraints(BaseModel):
    request_id: str
    sensitivities: list[str] = Field(default_factory=list)
    max_products: int = Field(default=5, ge=1, le=10)
    image_path: Optional[str] = None

    @field_validator("sensitivities", mode="before")
    @classmethod
    def normalize_sensitivities(cls, v: list[str]) -> list[str]:
        return [s.lower() for s in v]


class SkinProfile(BaseModel):
    request_id: str
    skin_conditions: list[str]


# ---------------------------------------------------------------------------
# RabbitMQ event payloads
# ---------------------------------------------------------------------------


class RoutineRequestedEvent(BaseModel):
    """Published by api_service, consumed by vision_service."""

    constraints: UserConstraints
    catalog_ref: str  # identifier / path for the product catalog


class ProfileAnalyzedEvent(BaseModel):
    """Published by vision_service, consumed by matching_service."""

    profile: SkinProfile
    constraints: UserConstraints


class RoutineMatchedEvent(BaseModel):
    """Published by matching_service, consumed by explanation_service."""

    matched_products: list[Product]
    profile: SkinProfile
    constraints: UserConstraints


class RoutineCompletedEvent(BaseModel):
    """Published by explanation_service, consumed by api_service."""

    matched_products: list[Product]
    explanations: dict[str, str]  # product id → explanation text
