"""Tests for the _build_product_text embedding input formatter."""

from __future__ import annotations

from matching_service.core.models import CatalogProduct, ProductFlags
from matching_service.scripts.build_product_embeddings import _build_product_text


def _sample_product(**overrides) -> CatalogProduct:
    defaults = dict(
        id="p1",
        name="Gentle Gel Cleanser",
        brand="CleanCo",
        category="CLEANSER",
        description="A lightweight gel cleanser for daily use",
        skin_types=["oily", "combination"],
        concerns=["excess_oil", "large_pores"],
        benefits=["oil_control", "gentle_cleansing"],
        ingredients=["water", "niacinamide", "salicylic acid", "glycerin"],
        flags=ProductFlags(),
    )
    defaults.update(overrides)
    return CatalogProduct(**defaults)


def test_field_order() -> None:
    """Fields must appear in the specified tag order."""
    text = _build_product_text(_sample_product())
    brand_pos = text.index("Brand:")
    cat_pos = text.index("Category:")
    name_pos = text.index("Name:")
    skin_pos = text.index("Skin Types:")
    concerns_pos = text.index("Concerns:")
    benefits_pos = text.index("Benefits:")
    desc_pos = text.index("Description:")
    key_pos = text.index("Key Ingredients:")
    assert brand_pos < cat_pos < name_pos < skin_pos
    assert skin_pos < concerns_pos < benefits_pos < desc_pos < key_pos


def test_skin_types_included() -> None:
    """skin_types must appear explicitly in the output."""
    text = _build_product_text(_sample_product())
    assert "Skin Types: oily, combination." in text


def test_concerns_humanized() -> None:
    """Underscore-separated concerns should be humanized."""
    text = _build_product_text(_sample_product())
    assert "excess oil" in text
    assert "large pores" in text


def test_benefits_humanized() -> None:
    """Underscore-separated benefits should be humanized."""
    text = _build_product_text(_sample_product())
    assert "oil control" in text
    assert "gentle cleansing" in text


def test_description_present() -> None:
    """Description should appear after structured fields."""
    text = _build_product_text(_sample_product())
    assert "Description: A lightweight gel cleanser for daily use." in text


def test_key_ingredients_uses_select() -> None:
    """Key Ingredients should include actives and skip fillers."""
    text = _build_product_text(_sample_product())
    assert "niacinamide" in text
    assert "salicylic acid" in text
    # water is a filler and should be excluded
    key_section = text[text.index("Key Ingredients:"):]
    assert "water" not in key_section.split(".")[0]


def test_flags_appended_when_true() -> None:
    """True flags should appear at the end."""
    flags = ProductFlags(
        contains_fragrance=True,
        pregnancy_safe=True,
        sensitive_skin_tested=True,
    )
    text = _build_product_text(_sample_product(flags=flags))
    assert "Contains fragrance." in text
    assert "Pregnancy-safe." in text
    assert "Tested for sensitive skin." in text


def test_false_flags_omitted() -> None:
    """False/None flags must not appear."""
    text = _build_product_text(_sample_product())
    assert "Contains fragrance" not in text
    assert "Pregnancy-safe" not in text
    assert "Contains alcohol" not in text


def test_empty_skin_types_omitted() -> None:
    """When skin_types is empty, the tag should not appear."""
    text = _build_product_text(_sample_product(skin_types=[]))
    assert "Skin Types:" not in text


def test_empty_concerns_omitted() -> None:
    """When concerns is empty, the tag should not appear."""
    text = _build_product_text(_sample_product(concerns=[]))
    assert "Concerns:" not in text


def test_category_humanized() -> None:
    """Underscore-separated category values should be humanized."""
    text = _build_product_text(_sample_product(category="SUN_PROTECTION"))
    assert "Category: SUN PROTECTION." in text
