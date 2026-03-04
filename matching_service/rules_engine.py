"""Hard-filter rules engine — allergen filtering and routine assembly.

Intentionally pure: no I/O, no RabbitMQ, no HTTP dependencies.
"""

from __future__ import annotations

import logging

from shared.models import Product, ProductCategory, UserConstraints

logger = logging.getLogger(__name__)


def filter_safe_products(
    products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Return only products whose ingredients do not overlap with user sensitivities."""
    # TODO: implement allergen filtering
    raise NotImplementedError


def build_routine(
    safe_products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Select and order products respecting max_products, budget, and category priority."""
    # TODO: implement priority ordering and constraint enforcement
    raise NotImplementedError
