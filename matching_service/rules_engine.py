"""Hard-filter rules engine — allergen filtering, ranking, and routine assembly.

Intentionally pure: no I/O, no RabbitMQ, no HTTP dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from shared.models import Product, ProductCategory, UserConstraints

logger = logging.getLogger(__name__)

# Maps each category to a numeric priority (lower = higher priority).
_CATEGORY_PRIORITY: dict[ProductCategory, int] = {
    ProductCategory.CLEANSER: 0,
    ProductCategory.MOISTURIZER: 1,
    ProductCategory.SPF: 2,
    ProductCategory.SERUM: 3,
    ProductCategory.TONER: 4,
    ProductCategory.OTHER: 5,
}

# Categories that should receive a slot before optional ones.
_ESSENTIAL_CATEGORIES: tuple[ProductCategory, ...] = (
    ProductCategory.CLEANSER,
    ProductCategory.MOISTURIZER,
    ProductCategory.SPF,
)

# ---------------------------------------------------------------------------
# Active-ingredient families and conflict groups
# ---------------------------------------------------------------------------

# (substring to match in lowercased ingredient, family tag)
_ACTIVE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("azelaic acid", "azelaic_acid"),
    ("salicylic acid", "salicylic_acid"),
    ("glycolic acid", "glycolic_acid"),
    ("lactic acid", "lactic_acid"),
    ("retinol", "retinoid"),
    ("retinal", "retinoid"),
    ("retinaldehyde", "retinoid"),
    ("adapalene", "retinoid"),
    ("benzoyl peroxide", "benzoyl_peroxide"),
)

# At most one family from this set per routine to prevent exfoliant stacking.
_EXFOLIANT_FAMILIES: frozenset[str] = frozenset({
    "azelaic_acid",
    "salicylic_acid",
    "glycolic_acid",
    "lactic_acid",
})


def active_families(product: Product) -> frozenset[str]:
    """Return the set of active-family tags present in a product's ingredients."""
    families: set[str] = set()
    for ingredient in product.ingredients:
        for keyword, family in _ACTIVE_KEYWORDS:
            if keyword in ingredient:
                families.add(family)
    return frozenset(families)


def _has_active_conflict(
    candidate_families: frozenset[str],
    routine_families: set[str],
) -> bool:
    """True when *candidate_families* would clash with the current routine.

    Two rules:
    1. Same active family already present  (redundant active).
    2. Routine already has an exfoliant and candidate adds another
       (exfoliant stacking).
    """
    if candidate_families & routine_families:
        return True

    candidate_exfoliants = candidate_families & _EXFOLIANT_FAMILIES
    routine_exfoliants = routine_families & _EXFOLIANT_FAMILIES
    if candidate_exfoliants and routine_exfoliants:
        return True

    return False


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


def select_balanced_routine(
    ranked_products: list[Product],
    max_products: int = 5,
) -> list[Product]:
    """Pick at most one product per category with active-conflict avoidance.

    Essential categories (cleanser, moisturizer, SPF) are guaranteed
    slots when available — if every candidate in an essential category
    conflicts, the top-ranked candidate is used anyway (category coverage
    outweighs active overlap).  Optional categories are skipped when
    all candidates conflict.

    Output is sorted by category priority for a natural routine order.
    """
    if not ranked_products:
        return []

    # Group candidates by category, preserving rank order within each.
    candidates_by_cat: dict[ProductCategory, list[Product]] = {}
    for product in ranked_products:
        candidates_by_cat.setdefault(product.category, []).append(product)

    selected: dict[ProductCategory, Product] = {}
    used_actives: set[str] = set()

    def _try_pick(cat: ProductCategory, force: bool) -> bool:
        """Pick best non-conflicting product for *cat*.

        When *force* is True (essentials), the top-ranked candidate is
        used as fallback if every candidate conflicts.
        """
        for candidate in candidates_by_cat.get(cat, []):
            families = active_families(candidate)
            if not _has_active_conflict(families, used_actives):
                selected[cat] = candidate
                used_actives.update(families)
                return True

        if force and cat in candidates_by_cat:
            best = candidates_by_cat[cat][0]
            selected[cat] = best
            used_actives.update(active_families(best))
            logger.debug(
                "Active conflict forced for essential %s (product=%s)",
                cat.value,
                best.id,
            )
            return True

        return False

    # Phase 1: reserve slots for essential categories.
    for cat in _ESSENTIAL_CATEGORIES:
        if cat in candidates_by_cat and len(selected) < max_products:
            _try_pick(cat, force=True)

    # Phase 2: fill remaining slots from ranked order (1 per category).
    for product in ranked_products:
        if len(selected) >= max_products:
            break
        cat = product.category
        if cat in selected:
            continue
        _try_pick(cat, force=False)

    routine = sorted(
        selected.values(),
        key=lambda p: _CATEGORY_PRIORITY.get(p.category, len(_CATEGORY_PRIORITY)),
    )
    return routine


def build_routine(
    safe_products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Select and order products into a category-balanced routine.

    Raises ValueError if max_products <= 0 (defensive guard against bypassed
    Pydantic validation, e.g. via model_construct).
    """
    if constraints.max_products <= 0:
        raise ValueError(
            f"max_products must be >= 1, got {constraints.max_products}"
        )

    if not safe_products:
        return []

    prioritized = sorted(
        safe_products,
        key=lambda p: _CATEGORY_PRIORITY.get(p.category, len(_CATEGORY_PRIORITY)),
    )

    routine = select_balanced_routine(prioritized, constraints.max_products)

    logger.debug(
        "request_id=%s build_routine: selected %d products (max=%d)",
        constraints.request_id,
        len(routine),
        constraints.max_products,
    )
    return routine


# Signature expected by the *ranker* parameter of match_products.
RankerFn = Callable[[list[str], list[Product]], list[Product]]


def match_products(
    catalog: list[Product],
    constraints: UserConstraints,
    skin_conditions: list[str] | None = None,
    ranker: RankerFn | None = None,
) -> list[Product]:
    """Full matching pipeline: filter -> rank -> balanced routine.

    When *ranker* and *skin_conditions* are both provided the semantic
    ranking determines which product per category is selected.
    Otherwise the result falls back to ``build_routine`` which uses
    deterministic category-priority ordering.  Both paths enforce
    at-most-one-per-category via ``select_balanced_routine``.
    """
    if constraints.max_products <= 0:
        raise ValueError(
            f"max_products must be >= 1, got {constraints.max_products}"
        )

    safe = filter_safe_products(catalog, constraints)

    if ranker is not None and skin_conditions:
        ranked = ranker(skin_conditions, safe)
        selected = select_balanced_routine(ranked, constraints.max_products)
        logger.debug(
            "request_id=%s match_products (semantic+balanced): %d/%d selected",
            constraints.request_id,
            len(selected),
            len(safe),
        )
        return selected

    return build_routine(safe, constraints)


# ---------------------------------------------------------------------------
# Routine rationale builder — deterministic explanation of selection logic
# ---------------------------------------------------------------------------


def build_routine_rationale(
    catalog: list[Product],
    safe_products: list[Product],
    selected: list[Product],
    skin_conditions: list[str] | None = None,
    used_semantic_ranking: bool = False,
) -> dict[str, Any]:
    """Build a deterministic rationale explaining why the routine was selected.

    Re-derives pipeline metadata from the same inputs used during matching.
    Returns a JSON-serializable dict with pipeline steps, per-product
    rationales, and active-avoidance explanations.
    """
    selected_ids = {p.id for p in selected}

    # Collect active families present in the final routine.
    routine_actives: set[str] = set()
    for p in selected:
        routine_actives.update(active_families(p))

    # Group safe products by category to count candidates.
    candidates_by_cat: dict[ProductCategory, int] = {}
    for p in safe_products:
        candidates_by_cat[p.category] = candidates_by_cat.get(p.category, 0) + 1

    # --- Pipeline steps ---
    steps: list[dict[str, Any]] = []

    filtered_count = len(catalog) - len(safe_products)
    steps.append({
        "step": "safety_filter",
        "catalog_size": len(catalog),
        "passed": len(safe_products),
        "removed": filtered_count,
    })

    if used_semantic_ranking and skin_conditions:
        steps.append({
            "step": "semantic_ranking",
            "description": (
                f"Ranked {len(safe_products)} safe products by relevance "
                f"to skin conditions: {', '.join(skin_conditions)}"
            ),
        })
    else:
        steps.append({
            "step": "category_priority",
            "description": (
                f"Ordered {len(safe_products)} safe products "
                f"by category priority (no semantic model)"
            ),
        })

    steps.append({
        "step": "routine_assembly",
        "selected_count": len(selected),
        "categories": [p.category.value for p in selected],
        "description": (
            f"Selected {len(selected)} products — 1 per category, "
            "essentials (cleanser, moisturizer, SPF) first, "
            "with active-conflict avoidance"
        ),
    })

    # --- Per-product rationale ---
    product_rationales: dict[str, dict[str, Any]] = {}
    for p in selected:
        families = active_families(p)
        is_essential = p.category in _ESSENTIAL_CATEGORIES
        entry: dict[str, Any] = {
            "name": p.name,
            "category": p.category.value,
            "role": "essential" if is_essential else "optional",
            "candidates_in_category": candidates_by_cat.get(p.category, 0),
        }
        if families:
            entry["strong_actives"] = sorted(families)
        product_rationales[p.id] = entry

    # --- Active avoidances ---
    avoided: list[str] = []
    for p in safe_products:
        if p.id in selected_ids:
            continue
        families = active_families(p)
        if not families:
            continue
        overlap = families & routine_actives
        if overlap:
            avoided.append(
                f"'{p.name}' ({p.category.value}) skipped — "
                f"active already in routine: {', '.join(sorted(overlap))}"
            )
        elif families & _EXFOLIANT_FAMILIES and routine_actives & _EXFOLIANT_FAMILIES:
            avoided.append(
                f"'{p.name}' ({p.category.value}) skipped — "
                "exfoliant stacking prevention"
            )

    rationale: dict[str, Any] = {
        "pipeline_steps": steps,
        "product_rationales": product_rationales,
    }
    if avoided:
        rationale["active_avoidances"] = avoided

    return rationale
