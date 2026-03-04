"""TDD test suite for matching_service.rules_engine — RED phase.

All tests FAIL until rules_engine.py is implemented.
Fixtures provide reusable UserConstraints and Product objects.
"""

import pytest

from shared.models import Product, ProductCategory, UserConstraints
from matching_service.rules_engine import build_routine, filter_safe_products

@pytest.fixture
def standard_constraints() -> UserConstraints:
    """Default constraints: no sensitivities, max 5 products."""
    return UserConstraints(
        request_id="req-standard",
        sensitivities=[],
        max_products=5,
    )


@pytest.fixture
def fragrance_sensitive_constraints() -> UserConstraints:
    """Constraints for a user sensitive to fragrance."""
    return UserConstraints(
        request_id="req-fragrance",
        sensitivities=["fragrance"],
        max_products=5,
    )


@pytest.fixture
def multi_sensitivity_constraints() -> UserConstraints:
    """Constraints for a user sensitive to both fragrance and alcohol."""
    return UserConstraints(
        request_id="req-multi",
        sensitivities=["fragrance", "alcohol"],
        max_products=5,
    )


@pytest.fixture
def tight_max_constraints() -> UserConstraints:
    """Constraints that cap the routine at 3 products."""
    return UserConstraints(
        request_id="req-tight",
        sensitivities=[],
        max_products=3,
    )


@pytest.fixture
def essential_catalog() -> list[Product]:
    """Catalog with one product per essential category (Cleanser, Moisturizer, SPF)."""
    return [
        Product(
            id="cleanser-basic",
            name="Basic Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["water", "glycerin"],
            description="Gentle daily cleanser.",
        ),
        Product(
            id="moisturizer-basic",
            name="Basic Moisturizer",
            category=ProductCategory.MOISTURIZER,
            ingredients=["water", "shea butter"],
            description="Hydrating daily moisturizer.",
        ),
        Product(
            id="spf-basic",
            name="Basic SPF",
            category=ProductCategory.SPF,
            ingredients=["zinc oxide", "water"],
            description="Broad-spectrum SPF 50.",
        ),
    ]


@pytest.fixture
def mixed_catalog(essential_catalog: list[Product]) -> list[Product]:
    """Catalog with essentials plus optional categories (Serum, Toner, Other)."""
    optional = [
        Product(
            id="serum-vitamin-c",
            name="Vitamin C Serum",
            category=ProductCategory.SERUM,
            ingredients=["ascorbic acid", "water"],
            description="Brightening vitamin C serum.",
        ),
        Product(
            id="toner-gentle",
            name="Gentle Toner",
            category=ProductCategory.TONER,
            ingredients=["witch hazel", "water"],
            description="Alcohol-free balancing toner.",
        ),
        Product(
            id="face-oil-other",
            name="Rosehip Face Oil",
            category=ProductCategory.OTHER,
            ingredients=["rosehip oil"],
            description="Nourishing face oil.",
        ),
    ]
    return essential_catalog + optional


@pytest.fixture
def scented_catalog() -> list[Product]:
    """Catalog where some products contain fragrance."""
    return [
        Product(
            id="scented-moisturizer",
            name="Scented Moisturizer",
            category=ProductCategory.MOISTURIZER,
            ingredients=["fragrance", "water", "glycerin"],
            description="Floral-scented moisturizer.",
        ),
        Product(
            id="unscented-cleanser",
            name="Unscented Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["water", "glycerin"],
            description="Fragrance-free cleanser.",
        ),
    ]


# ---------------------------------------------------------------------------
# filter_safe_products — allergy filtering
# ---------------------------------------------------------------------------


def test_product_with_sensitizing_ingredient_is_removed(
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    products = [
        Product(
            id="bad-cream",
            name="Bad Cream",
            category=ProductCategory.MOISTURIZER,
            ingredients=["fragrance", "water"],
            description="",
        )
    ]
    result = filter_safe_products(products, fragrance_sensitive_constraints)
    assert result == []


def test_product_without_sensitizing_ingredients_passes(
    scented_catalog: list[Product],
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    result = filter_safe_products(scented_catalog, fragrance_sensitive_constraints)
    assert len(result) == 1
    assert result[0].id == "unscented-cleanser"


def test_filtering_is_case_insensitive() -> None:
    """Model validators normalize both sides to lowercase — this confirms end-to-end."""
    constraints = UserConstraints(
        request_id="req-case",
        sensitivities=["Fragrance"],  # uppercase in input
        max_products=5,
    )
    products = [
        Product(
            id="scented",
            name="Scented Serum",
            category=ProductCategory.SERUM,
            ingredients=["Fragrance", "niacinamide"],  # uppercase in product
            description="",
        )
    ]
    result = filter_safe_products(products, constraints)
    assert result == []


def test_multiple_products_only_unsafe_removed(
    multi_sensitivity_constraints: UserConstraints,
) -> None:
    products = [
        Product(id="p1", name="P1", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="p2", name="P2", category=ProductCategory.SERUM,
                ingredients=["alcohol", "fragrance"], description=""),
        Product(id="p3", name="P3", category=ProductCategory.MOISTURIZER,
                ingredients=["glycerin"], description=""),
        Product(id="p4", name="P4", category=ProductCategory.TONER,
                ingredients=["alcohol"], description=""),
        Product(id="p5", name="P5", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = filter_safe_products(products, multi_sensitivity_constraints)
    assert len(result) == 3
    assert {p.id for p in result} == {"p1", "p3", "p5"}


def test_no_sensitivities_returns_all_products(
    mixed_catalog: list[Product],
    standard_constraints: UserConstraints,
) -> None:
    result = filter_safe_products(mixed_catalog, standard_constraints)
    assert len(result) == len(mixed_catalog)


def test_product_flagged_if_any_single_ingredient_matches(
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    """A product with 10 ingredients is removed if even one matches a sensitivity."""
    ingredients = [f"ingredient_{i}" for i in range(9)] + ["fragrance"]
    products = [
        Product(
            id="ten-ingredient-serum",
            name="Ten Ingredient Serum",
            category=ProductCategory.SERUM,
            ingredients=ingredients,
            description="",
        )
    ]
    result = filter_safe_products(products, fragrance_sensitive_constraints)
    assert result == []


# ---------------------------------------------------------------------------
# build_routine — max products, priority, and edge cases
# ---------------------------------------------------------------------------


def test_routine_respects_max_products(
    mixed_catalog: list[Product],
    tight_max_constraints: UserConstraints,
) -> None:
    result = build_routine(mixed_catalog, tight_max_constraints)
    assert len(result) == 3


def test_priority_essentials_chosen_before_optionals(
    mixed_catalog: list[Product],
    tight_max_constraints: UserConstraints,
) -> None:
    """When capped at 3, Cleanser, Moisturizer, and SPF must be selected over Serum/Toner/Other."""
    result = build_routine(mixed_catalog, tight_max_constraints)
    categories = {p.category for p in result}
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories
    assert ProductCategory.SERUM not in categories
    assert ProductCategory.TONER not in categories
    assert ProductCategory.OTHER not in categories


def test_routine_never_exceeds_available_products(
    essential_catalog: list[Product],
    standard_constraints: UserConstraints,
) -> None:
    """If fewer products are available than max_products, return all without error."""
    result = build_routine(essential_catalog, standard_constraints)
    assert len(result) == len(essential_catalog)


def test_empty_product_list_returns_empty(
    standard_constraints: UserConstraints,
) -> None:
    result = build_routine([], standard_constraints)
    assert result == []


def test_invalid_max_products_raises_value_error() -> None:
    """build_routine must raise ValueError when max_products <= 0.

    UserConstraints.model_construct bypasses Pydantic field validation so we can
    test the function's own defensive guard in isolation.
    """
    constraints = UserConstraints.model_construct(
        request_id="req-invalid",
        sensitivities=[],
        max_products=0,
    )
    with pytest.raises(ValueError):
        build_routine([], constraints)
