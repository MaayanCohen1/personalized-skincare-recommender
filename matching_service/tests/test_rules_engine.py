"""TDD test suite for matching_service.rules_engine.

All tests are RED until rules_engine.py is implemented.
"""

import pytest

from shared.models import Product, ProductCategory, UserConstraints
from matching_service.rules_engine import build_routine, filter_safe_products


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_product(
    id: str,
    category: ProductCategory,
    ingredients: list[str],
    price: float = 10.0,
) -> Product:
    return Product(
        id=id,
        name=id.replace("-", " ").title(),
        category=category,
        ingredients=ingredients,
        description="",
        price=price,
    )


def make_constraints(
    sensitivities: list[str] = None,
    max_products: int = 5,
    budget: float | None = None,
) -> UserConstraints:
    return UserConstraints(
        request_id="test-request-id",
        sensitivities=sensitivities or [],
        max_products=max_products,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# filter_safe_products — allergen filtering
# ---------------------------------------------------------------------------


def test_product_with_sensitizing_ingredient_is_removed() -> None:
    products = [
        make_product("bad-cream", ProductCategory.MOISTURIZER, ["fragrance", "water"])
    ]
    constraints = make_constraints(sensitivities=["fragrance"])
    result = filter_safe_products(products, constraints)
    assert result == []


def test_product_without_sensitizing_ingredients_passes() -> None:
    products = [
        make_product("good-cream", ProductCategory.MOISTURIZER, ["water", "glycerin"])
    ]
    constraints = make_constraints(sensitivities=["fragrance"])
    result = filter_safe_products(products, constraints)
    assert len(result) == 1
    assert result[0].id == "good-cream"


def test_filtering_is_case_insensitive() -> None:
    # Model validators normalize both sides to lowercase; this confirms end-to-end.
    products = [
        make_product("scented", ProductCategory.SERUM, ["Fragrance", "niacinamide"])
    ]
    constraints = make_constraints(sensitivities=["Fragrance"])
    result = filter_safe_products(products, constraints)
    assert result == []


def test_multiple_products_only_unsafe_removed() -> None:
    products = [
        make_product("p1", ProductCategory.CLEANSER, ["water"]),
        make_product("p2", ProductCategory.SERUM, ["alcohol", "fragrance"]),
        make_product("p3", ProductCategory.MOISTURIZER, ["glycerin"]),
        make_product("p4", ProductCategory.TONER, ["alcohol"]),
        make_product("p5", ProductCategory.SPF, ["zinc oxide"]),
    ]
    constraints = make_constraints(sensitivities=["fragrance", "alcohol"])
    result = filter_safe_products(products, constraints)
    assert len(result) == 3
    safe_ids = {p.id for p in result}
    assert safe_ids == {"p1", "p3", "p5"}


def test_empty_sensitivities_returns_all_products() -> None:
    products = [
        make_product("p1", ProductCategory.CLEANSER, ["fragrance"]),
        make_product("p2", ProductCategory.MOISTURIZER, ["alcohol"]),
    ]
    constraints = make_constraints(sensitivities=[])
    result = filter_safe_products(products, constraints)
    assert len(result) == 2


def test_product_flagged_if_any_single_ingredient_matches() -> None:
    ingredients = [f"ingredient_{i}" for i in range(9)] + ["bad-stuff"]
    products = [make_product("ten-ing", ProductCategory.SERUM, ingredients)]
    constraints = make_constraints(sensitivities=["bad-stuff"])
    result = filter_safe_products(products, constraints)
    assert result == []


# ---------------------------------------------------------------------------
# build_routine — constraint & priority
# ---------------------------------------------------------------------------


def test_routine_respects_max_products() -> None:
    products = [
        make_product(f"p{i}", ProductCategory.OTHER, ["water"]) for i in range(10)
    ]
    constraints = make_constraints(max_products=3)
    result = build_routine(products, constraints)
    assert len(result) == 3


def test_priority_order_cleanser_moisturizer_spf_first() -> None:
    products = [
        make_product("serum-1", ProductCategory.SERUM, ["water"]),
        make_product("spf-1", ProductCategory.SPF, ["water"]),
        make_product("cleanser-1", ProductCategory.CLEANSER, ["water"]),
        make_product("other-1", ProductCategory.OTHER, ["water"]),
        make_product("moisturizer-1", ProductCategory.MOISTURIZER, ["water"]),
    ]
    constraints = make_constraints(max_products=3)
    result = build_routine(products, constraints)
    categories = [p.category for p in result]
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories
    assert ProductCategory.OTHER not in categories
    assert ProductCategory.SERUM not in categories


def test_routine_never_exceeds_available_products() -> None:
    products = [
        make_product("p1", ProductCategory.CLEANSER, ["water"]),
        make_product("p2", ProductCategory.MOISTURIZER, ["water"]),
    ]
    constraints = make_constraints(max_products=5)
    result = build_routine(products, constraints)
    assert len(result) == 2


def test_empty_safe_products_returns_empty_list() -> None:
    constraints = make_constraints(max_products=5)
    result = build_routine([], constraints)
    assert result == []


def test_budget_constraint_excludes_expensive_products() -> None:
    products = [
        make_product("cheap-cleanser", ProductCategory.CLEANSER, ["water"], price=10.0),
        make_product("mid-moisturizer", ProductCategory.MOISTURIZER, ["water"], price=15.0),
        make_product("pricey-spf", ProductCategory.SPF, ["water"], price=25.0),
    ]
    # Total of all three = 50.0; budget of 30 should keep only cheapest two (10 + 15 = 25)
    constraints = make_constraints(max_products=5, budget=30.0)
    result = build_routine(products, constraints)
    total_cost = sum(p.price for p in result)
    assert total_cost <= 30.0
    assert any(p.id == "cheap-cleanser" for p in result)
    assert any(p.id == "mid-moisturizer" for p in result)
    assert all(p.id != "pricey-spf" for p in result)
