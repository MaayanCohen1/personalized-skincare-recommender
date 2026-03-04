"""Hard-filter rules engine — allergen filtering and routine assembly.

Intentionally pure: no I/O, no RabbitMQ, no HTTP dependencies.
"""

from __future__ import annotations

import logging

from shared.models import Product, ProductCategory, UserConstraints

logger = logging.getLogger(__name__)

# Maps each category to a numeric priority (lower = higher priority).
# Mirrors ProductCategory.PRIORITY_ORDER but kept here as a fast lookup dict.
_CATEGORY_PRIORITY: dict[ProductCategory, int] = {
    ProductCategory.CLEANSER: 0,
    ProductCategory.MOISTURIZER: 1,
    ProductCategory.SPF: 2,
    ProductCategory.SERUM: 3,
    ProductCategory.TONER: 4,
    ProductCategory.OTHER: 5,
}


def filter_safe_products(
    products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Return only products whose ingredients do not overlap with user sensitivities.

    Both sides are already normalized to lowercase by the Pydantic field validators,
    so a plain set-intersection is sufficient.
    """
    if not constraints.sensitivities:
        return list(products)

    blocked = set(constraints.sensitivities)
    safe = [p for p in products if not blocked.intersection(p.ingredients)]

    logger.debug(
        "request_id=%s filter_safe_products: %d/%d products passed",
        constraints.request_id,
        len(safe),
        len(products),
    )
    return safe


def build_routine(
    safe_products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Select and order products respecting max_products and category priority.

    Raises ValueError if max_products <= 0 (defensive guard against bypassed
    Pydantic validation, e.g. via model_construct).
    """
    if constraints.max_products <= 0:
        raise ValueError(
            f"max_products must be >= 1, got {constraints.max_products}"
        )

    if not safe_products:
        return []

    # Sort by priority so essential categories always come first.
    prioritized = sorted(
        safe_products,
        key=lambda p: _CATEGORY_PRIORITY.get(p.category, len(_CATEGORY_PRIORITY)),
    )

    routine = prioritized[: constraints.max_products]

    logger.debug(
        "request_id=%s build_routine: selected %d products (max=%d)",
        constraints.request_id,
        len(routine),
        constraints.max_products,
    )
    return routine
