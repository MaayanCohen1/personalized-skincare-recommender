"""Deterministic product-matching engine — safety filtering, fit scoring,
category-aware selection, and active-conflict avoidance.

Architecture:
  1. ``filter_safe_products`` — hard-block products by sensitivity / flags
  2. ``score_product_fit`` — compute a per-product fit score from structured
     catalog metadata (skin_types, concerns, benefits, ingredients, category,
     flags) plus optional semantic similarity and optional
     ``routine_treatment_families`` (cleanser SA overlap hook); body-SPF wording
     may use ``description`` only
  3. ``select_best_per_category`` — group safe products by category, pick the
     highest-fit candidate per category, category-aware active-ingredient policy
  4. ``match_products`` — public orchestrator that wires these steps together

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

_ESSENTIAL_CATEGORIES: tuple[ProductCategory, ...] = (
    ProductCategory.CLEANSER,
    ProductCategory.MOISTURIZER,
    ProductCategory.SPF,
)

# ---------------------------------------------------------------------------
# Active-ingredient families and conflict groups
# ---------------------------------------------------------------------------

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

_EXFOLIANT_FAMILIES: frozenset[str] = frozenset({
    "azelaic_acid",
    "salicylic_acid",
    "glycolic_acid",
    "lactic_acid",
})

# Strong treatment families detected via ``_ACTIVE_KEYWORDS`` (exfoliants,
# retinoids, benzoyl peroxide).  Support-style overlaps (niacinamide, HA,
# ceramides, etc.) are not tagged here and therefore never drive conflicts.
_TREATMENT_FAMILIES: frozenset[str] = frozenset({
    *_EXFOLIANT_FAMILIES,
    "retinoid",
    "benzoyl_peroxide",
})

# Fit tie-break: prefer candidates with fewer soft-penalty pairs vs routine.
_SOFT_ACTIVE_CONFLICT_PENALTY: float = 0.45


def active_families(product: Product) -> frozenset[str]:
    """Return the set of active-family tags present in a product's ingredients."""
    families: set[str] = set()
    for ingredient in product.ingredients:
        for keyword, family in _ACTIVE_KEYWORDS:
            if keyword in ingredient:
                families.add(family)
    return frozenset(families)


def _treatment_families(families: frozenset[str]) -> frozenset[str]:
    return families & _TREATMENT_FAMILIES


def _pair_active_policy(
    cat_a: ProductCategory,
    fam_a: frozenset[str],
    cat_b: ProductCategory,
    fam_b: frozenset[str],
) -> str:
    """Pairwise policy: ``block`` | ``penalty`` | ``allow`` (symmetric)."""
    t_a = _treatment_families(fam_a)
    t_b = _treatment_families(fam_b)
    if not t_a or not t_b:
        return "allow"

    rinse_a = cat_a == ProductCategory.CLEANSER
    rinse_b = cat_b == ProductCategory.CLEANSER
    spf_a = cat_a == ProductCategory.SPF
    spf_b = cat_b == ProductCategory.SPF

    if spf_a or spf_b:
        if rinse_a or rinse_b:
            return "penalty" if (t_a & t_b) else "allow"
        return "penalty" if _risky_leave_on_treatment_pair(t_a, t_b) else "allow"

    if (rinse_a and not rinse_b) or (rinse_b and not rinse_a):
        leave_cat = cat_b if rinse_a else cat_a
        ex_a = t_a & _EXFOLIANT_FAMILIES
        ex_b = t_b & _EXFOLIANT_FAMILIES
        # Stacking two different acids across wash-off + leave-on treatment steps.
        if leave_cat in (ProductCategory.TONER, ProductCategory.SERUM):
            if ex_a and ex_b and not (ex_a & ex_b):
                return "block"
        return "penalty" if _rinse_leave_treatment_interaction(t_a, t_b) else "allow"

    return _leave_on_vs_leave_on_policy(t_a, t_b)


def _rinse_leave_treatment_interaction(t_a: frozenset[str], t_b: frozenset[str]) -> bool:
    """True when a rinse-off treatment + a leave-on treatment should be softened."""
    return bool(t_a and t_b)


def _risky_leave_on_treatment_pair(t_a: frozenset[str], t_b: frozenset[str]) -> bool:
    """SPF vs leave-on: penalize overlap or aggressive stacking."""
    if t_a & t_b:
        return True
    exa, exb = t_a & _EXFOLIANT_FAMILIES, t_b & _EXFOLIANT_FAMILIES
    if exa and exb:
        return True
    if ("retinoid" in t_a and "benzoyl_peroxide" in t_b) or (
        "retinoid" in t_b and "benzoyl_peroxide" in t_a
    ):
        return True
    if ("retinoid" in t_a and exb) or ("retinoid" in t_b and exa):
        return True
    return False


def _leave_on_vs_leave_on_policy(t_a: frozenset[str], t_b: frozenset[str]) -> str:
    """Hard blocks for leave-on + leave-on strong treatment stacking."""
    if ("benzoyl_peroxide" in t_a and "retinoid" in t_b) or (
        "retinoid" in t_a and "benzoyl_peroxide" in t_b
    ):
        return "block"
    if t_a & t_b:
        return "block"
    exa, exb = t_a & _EXFOLIANT_FAMILIES, t_b & _EXFOLIANT_FAMILIES
    if exa and exb:
        return "block"
    if ("retinoid" in t_a and exb) or ("retinoid" in t_b and exa):
        return "block"
    return "allow"


def _routine_active_assessment(
    candidate: Product,
    candidate_cat: ProductCategory,
    routine_segments: list[tuple[ProductCategory, frozenset[str]]],
) -> tuple[bool, int]:
    """Return (hard_blocked, soft_penalty_count) vs already-selected slots."""
    fam_c = active_families(candidate)
    penalties = 0
    for cat_r, fam_r in routine_segments:
        # SPF: best facial fit wins; wash-off / cream actives do not block or penalize.
        if candidate_cat == ProductCategory.SPF and cat_r in (
            ProductCategory.CLEANSER,
            ProductCategory.MOISTURIZER,
        ):
            continue
        # Moisturizer: cleanser is rinse-off — shared actives (e.g. SA) do not block or penalize.
        if candidate_cat == ProductCategory.MOISTURIZER and cat_r == ProductCategory.CLEANSER:
            continue
        pol = _pair_active_policy(cat_r, fam_r, candidate_cat, fam_c)
        if pol == "block":
            return True, penalties
        if pol == "penalty":
            penalties += 1
    return False, penalties


def _pick_with_active_policy(
    category: ProductCategory,
    candidates: list[Product],
    routine_segments: list[tuple[ProductCategory, frozenset[str]]],
    product_fit: dict[str, float] | None,
    *,
    force: bool,
    skin_conditions: list[str] | None = None,
) -> Product | None:
    """Best non-blocked candidate; lower soft penalties and higher fit win.

    Cleanser: if several candidates are within a fit epsilon of the best
    conflict-adjusted score, fragrance-free is preferred inside that band, then
    (for acne + oily or barrier-sensitive) gentler benefit signals, then higher
    fit. Other categories: on exact tie of adjusted fit, prefer fragrance-free,
    then list order.
    """
    cond_set = (
        frozenset(c.lower().strip() for c in skin_conditions)
        if skin_conditions
        else frozenset()
    )
    acne_ff_gentle_cleanser = (
        category == ProductCategory.CLEANSER
        and _cleanser_acne_prefers_ff_gentle(cond_set)
    )
    cleanser_eps = (
        _CLEANSER_NEAR_TOP_FIT_EPSILON_OILY_ACNE
        if acne_ff_gentle_cleanser
        else _CLEANSER_NEAR_TOP_FIT_EPSILON
    )

    rows: list[tuple[Product, int, float, int]] = []
    for i, cand in enumerate(candidates):
        blocked, n_pen = _routine_active_assessment(cand, category, routine_segments)
        if blocked:
            continue
        fit = product_fit[cand.id] if product_fit is not None else 0.0
        tie = fit - (n_pen * _SOFT_ACTIVE_CONFLICT_PENALTY)
        rows.append((cand, n_pen, tie, i))

    if not rows:
        if force and candidates:
            logger.debug(
                "Active conflict forced for %s — picking top-listed candidate %s",
                category.value,
                candidates[0].id,
            )
            return candidates[0]
        return None

    if category == ProductCategory.CLEANSER and len(rows) > 1:
        t_max = max(r[2] for r in rows)
        floor = t_max - cleanser_eps
        near = [r for r in rows if r[2] >= floor]
        if acne_ff_gentle_cleanser:
            near.sort(
                key=lambda r: (
                    int(r[0].contains_fragrance),
                    -_cleanser_gentle_signals_rank(r[0]),
                    -r[2],
                    r[3],
                )
            )
        else:
            near.sort(key=lambda r: (int(r[0].contains_fragrance), -r[2], r[3]))
        return near[0][0]

    scored = [
        (r[1], -r[2], int(r[0].contains_fragrance), r[3], r[0]) for r in rows
    ]
    scored.sort()
    return scored[0][4]


_SENSITIVITY_SYNONYMS: dict[str, frozenset[str]] = {
    "fragrance": frozenset({"fragrance", "parfum", "perfume"}),
    "alcohol": frozenset({"alcohol", "alcohol denat", "alcohol denat."}),
}


def _expand_sensitivities(sensitivities: list[str]) -> set[str]:
    """Expand user-facing sensitivity names into all known ingredient synonyms."""
    expanded: set[str] = set()
    for s in sensitivities:
        synonyms = _SENSITIVITY_SYNONYMS.get(s)
        if synonyms:
            expanded.update(synonyms)
        else:
            expanded.add(s)
    return expanded


_FLAG_SENSITIVITY_MAP: dict[str, str] = {
    "fragrance": "contains_fragrance",
    "alcohol": "contains_alcohol",
}


def _is_blocked_by_flags(product: Product, sensitivities: list[str]) -> bool:
    """Check if a product's catalog flags conflict with user sensitivities."""
    for sens in sensitivities:
        flag_attr = _FLAG_SENSITIVITY_MAP.get(sens)
        if flag_attr and getattr(product, flag_attr, False):
            return True
    return False


def filter_safe_products(
    products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Return only products safe for the user's stated sensitivities."""
    if not constraints.sensitivities:
        return list(products)

    blocked = _expand_sensitivities(constraints.sensitivities)
    safe: list[Product] = []
    for p in products:
        if blocked.intersection(p.ingredients):
            continue
        if _is_blocked_by_flags(p, constraints.sensitivities):
            continue
        safe.append(p)

    logger.debug(
        "request_id=%s filter_safe_products: %d/%d products passed "
        "(blocked ingredients: %s)",
        constraints.request_id,
        len(safe),
        len(products),
        blocked,
    )
    return safe


# ---------------------------------------------------------------------------
# Fit scoring — the core of the new approach
# ---------------------------------------------------------------------------

# Condition -> primary skin-type matches (full weight when not overly broad).
_SKIN_TYPE_DIRECT: dict[str, frozenset[str]] = {
    "oily": frozenset({"oily"}),
    "dry": frozenset({"dry", "sensitive", "normal"}),
    "normal": frozenset({"normal"}),
    "combination": frozenset({"combination"}),
    "acne": frozenset({"oily", "combination"}),
    "sensitive": frozenset({"sensitive"}),
}
# Secondary skin-type matches (weaker than direct).
_SKIN_TYPE_COMPATIBLE: dict[str, frozenset[str]] = {
    "oily": frozenset({"combination"}),
    "dry": frozenset({"combination"}),
    "normal": frozenset({"combination", "sensitive"}),
    "combination": frozenset({"oily", "normal"}),
    "acne": frozenset({"normal"}),
    "sensitive": frozenset({"dry", "normal"}),
}

# Oily / acne: combination-only on the label is a strong (near-direct) signal.
_STRONG_COMBO_FOR_OILY_ACNE: frozenset[str] = frozenset({"oily", "acne"})


def _optional_skin_type_gate_passes(
    product: Product,
    conditions_set: frozenset[str],
) -> bool:
    """Whether a SERUM/TONER may be considered: skin_types must align with conditions.

    Uses the same direct / compatible / combo-only rules as fit scoring’s skin
    block, excluding the weak ``normal``/``all``-only fallback. Explicit
    ``all`` on the product still passes.
    """
    product_types = set(product.skin_types)
    if not product_types:
        return False
    if "all" in product_types:
        return True
    if not conditions_set:
        return False

    direct_match_types: set[str] = set()
    compat_match_types: set[str] = set()
    for cond in conditions_set:
        direct_match_types.update(_SKIN_TYPE_DIRECT.get(cond, frozenset()))
        compat_match_types.update(_SKIN_TYPE_COMPATIBLE.get(cond, frozenset()))

    if product_types & direct_match_types:
        return True
    if product_types & compat_match_types:
        return True
    if (
        "combination" in product_types
        and "oily" not in product_types
        and bool(conditions_set & _STRONG_COMBO_FOR_OILY_ACNE)
    ):
        return True
    return False


# Condition -> concerns that are a strong match.
_CONDITION_CONCERNS: dict[str, frozenset[str]] = {
    "oily": frozenset({
        "oiliness", "excess oil", "shine", "large pores", "pores",
    }),
    "dry": frozenset({
        "dryness", "dehydration", "flaking", "barrier damage",
    }),
    "normal": frozenset({
        "dehydration", "sensitivity", "irritation", "redness",
    }),
    "acne": frozenset({
        "acne", "breakouts", "clogged pores", "blackheads", "pores", "oiliness",
    }),
    "sensitive": frozenset({"sensitivity", "irritation", "redness", "reactive skin"}),
    "combination": frozenset({"oiliness", "dryness", "dehydration"}),
}

# Condition -> benefits that are a strong match.
_CONDITION_BENEFITS: dict[str, frozenset[str]] = {
    "oily": frozenset({
        "oil control", "lightweight", "lightweight hydration",
        "non greasy", "pore cleansing", "pore refining", "pore care",
        "balanced moisture",
    }),
    "dry": frozenset({
        "hydrating", "nourishing", "barrier support", "barrier repair",
        "soothing", "non stripping", "gentle cleansing", "moisture balance",
    }),
    "normal": frozenset({
        "gentle cleansing", "balanced hydration", "barrier support",
        "lightweight", "soothing",
    }),
    "acne": frozenset({
        "acne support", "pore cleansing", "oil control", "lightweight",
        "lightweight hydration", "non comedogenic", "gentle cleansing",
    }),
    "sensitive": frozenset({
        "soothing", "gentle cleansing", "barrier support", "calming",
        "fragrance free", "hypoallergenic",
    }),
    "combination": frozenset({
        "balanced moisture", "lightweight hydration", "gentle cleansing",
    }),
}

# Ingredient substrings (product.ingredients are lowercased).
_CONDITION_INGREDIENT_KEYWORDS: dict[str, frozenset[str]] = {
    "oily": frozenset({"niacinamide", "salicylic acid"}),
    "dry": frozenset({
        "hyaluronic acid", "sodium hyaluronate", "ceramide", "ceramides",
        "squalane", "panthenol", "glycerin",
    }),
    "normal": frozenset(),
    "acne": frozenset({"salicylic acid", "niacinamide", "azelaic acid"}),
    "sensitive": frozenset(),
    "combination": frozenset(),
}

# Benefit tokens that clash with a condition (structured mismatch).
_NEGATIVE_BENEFITS: dict[str, frozenset[str]] = {
    "oily": frozenset({
        "intense moisture", "deep hydration", "barrier repair", "nourishing",
    }),
    "dry": frozenset({
        "matte finish", "oil control", "mattifying", "stripping",
    }),
    "normal": frozenset({
        "matte finish", "oil control", "mattifying", "intense moisture",
    }),
    "acne": frozenset({
        "nourishing", "intense moisture", "deep hydration", "barrier repair",
    }),
    "sensitive": frozenset(),
    "combination": frozenset(),
}

# Treatment-heavy structured signals (benefits normalized like other lists).
_DRY_TREATMENT_BENEFITS: frozenset[str] = frozenset({
    "exfoliating", "pore refining", "pore cleansing", "oil control",
    "acne support", "mattifying",
})
_DRY_TREATMENT_INGREDIENTS: frozenset[str] = frozenset({
    "salicylic acid", "glycolic acid", "lactic acid", "benzoyl peroxide",
})

_NORMAL_TREATMENT_BENEFITS: frozenset[str] = frozenset({
    "acne support", "oil control", "exfoliating", "pore refining",
    "pore cleansing", "mattifying", "stripping",
})
_NORMAL_TREATMENT_INGREDIENTS: frozenset[str] = frozenset({
    "salicylic acid", "glycolic acid", "lactic acid",
    "adapalene", "benzoyl peroxide",
})

_OILY_HEAVY_MOISTURE_INGREDIENTS: frozenset[str] = frozenset({
    "shea butter", "petrolatum", "mineral oil",
})

# User conditions that justify acids / oil-control on an otherwise dry profile.
_TREATMENT_JUSTIFY_CONDITIONS: frozenset[str] = frozenset({"acne", "oily"})

# How many distinct skin labels count as “broad” (dilutes skin-type weight).
_BROAD_SKIN_TYPE_COUNT: int = 4

# Body-focused SPF signals (penalised for facial routines).
_BODY_SPF_HINTS: tuple[str, ...] = (
    "body", "body spf", "body sunscreen", "body lotion spf",
)

# Soft fit penalties for catalog flags (ranking only; not safety filtering).
_LEAVE_ON_FRAGRANCE_PENALTY_CATEGORIES: frozenset[ProductCategory] = frozenset({
    ProductCategory.MOISTURIZER,
    ProductCategory.SPF,
    ProductCategory.SERUM,
    ProductCategory.TONER,
})
# Fragranced leave-on products (longer skin contact) are penalised more than rinse-off cleanser.
_SOFT_FIT_PENALTY_FRAGRANCE_LEAVE_ON: float = -1.0
# Cleanser penalty (default); leave-on tier stays at -1.0 and is unchanged.
_SOFT_FIT_PENALTY_FRAGRANCE_CLEANSER: float = -0.82
# Stronger wash-off penalty when oily+acne: prefer fragrance-free gentle cleansers over fragranced actives.
_SOFT_FIT_PENALTY_FRAGRANCE_CLEANSER_OILY_ACNE: float = -1.12
_SOFT_FIT_PENALTY_ALCOHOL: float = -0.85

# Profile for cleanser-specific fragrance, SA wash-off, and near-top band behaviour.
_OILY_ACNE_PROFILE: frozenset[str] = frozenset({"oily", "acne"})

# Benefit hints for cleanser tie-break (oily+acne near-top band only).
_GENTLE_CLEANSER_BENEFIT_HINTS: frozenset[str] = frozenset({
    "gentle cleansing", "soothing", "non stripping", "calming",
})

# Cleanser pick: among candidates whose conflict-adjusted fit is within this gap of
# the best in the pool, prefer fragrance-free (then higher fit, then list order).
_CLEANSER_NEAR_TOP_FIT_EPSILON: float = 0.45
# Wider band for oily+acne so close fragranced SA cleansers lose to FF gentle options.
_CLEANSER_NEAR_TOP_FIT_EPSILON_OILY_ACNE: float = 0.62

# Oily+acne: salicylic in a wash-off step is partially redundant with typical leave-on acne actives.
_CLEANSER_SA_OILY_ACNE_DEDUCTION: float = 0.42
# Extra deduction when scoring with actives already present elsewhere in the routine (optional hook).
_CLEANSER_SA_ROUTINE_OVERLAP_EXTRA: float = 0.28
_ROUTINE_OVERLAP_REDUCE_CLEANSER_SA: frozenset[str] = frozenset({
    "salicylic_acid",
    "benzoyl_peroxide",
})

_ACNE_BARRIER_EXACT_CONDITIONS: frozenset[str] = frozenset({"dry", "sensitive"})
_ACNE_BARRIER_SIGNAL_PHRASES: tuple[str, ...] = (
    "dehydration",
    "dehydrat",
    "dryness",
    "sensitivity",
    "irritation",
    "irritated",
    "barrier damage",
    "barrier repair",
    "barrier dysfunction",
    "compromised barrier",
    "skin barrier",
)


def _acne_with_dryness_sensitivity_or_barrier_signals(conditions_set: frozenset[str]) -> bool:
    """True when labels suggest acne alongside dryness, sensitivity, or barrier irritation."""
    if "acne" not in conditions_set:
        return False
    for raw in conditions_set:
        if raw in _ACNE_BARRIER_EXACT_CONDITIONS:
            return True
        norm = raw.replace("_", " ").strip().lower()
        if norm in _ACNE_BARRIER_EXACT_CONDITIONS:
            return True
        if norm == "barrier":
            return True
        for phrase in _ACNE_BARRIER_SIGNAL_PHRASES:
            if phrase in norm:
                return True
    return False


def _cleanser_acne_prefers_ff_gentle(conditions_set: frozenset[str]) -> bool:
    """Oily+acne, or acne with dryness / sensitivity / barrier-dehydration signals."""
    return (
        _OILY_ACNE_PROFILE <= conditions_set
        or _acne_with_dryness_sensitivity_or_barrier_signals(conditions_set)
    )


def _fragrance_flag_fit_penalty(
    category: ProductCategory,
    conditions_set: frozenset[str] | None,
) -> float:
    if category == ProductCategory.CLEANSER:
        if conditions_set is not None and _cleanser_acne_prefers_ff_gentle(conditions_set):
            return _SOFT_FIT_PENALTY_FRAGRANCE_CLEANSER_OILY_ACNE
        return _SOFT_FIT_PENALTY_FRAGRANCE_CLEANSER
    if category in _LEAVE_ON_FRAGRANCE_PENALTY_CATEGORIES:
        return _SOFT_FIT_PENALTY_FRAGRANCE_LEAVE_ON
    return _SOFT_FIT_PENALTY_FRAGRANCE_LEAVE_ON


def _catalog_flag_fit_adjustment(
    product: Product,
    conditions_set: frozenset[str] | None = None,
) -> float:
    """Negative score deltas from catalog flags (deterministic ranking only)."""
    adj = 0.0
    if product.contains_fragrance:
        adj += _fragrance_flag_fit_penalty(product.category, conditions_set)
    if product.contains_alcohol:
        adj += _SOFT_FIT_PENALTY_ALCOHOL
    return adj


def _cleanser_gentle_signals_rank(product: Product) -> int:
    """Deterministic gentle tie-break strength (higher = prefer in acne FF-gentle cleanser band)."""
    return len(set(product.benefits) & _GENTLE_CLEANSER_BENEFIT_HINTS)

# Optional categories need a clearer structured case to score competitively.
_OPTIONAL_CATEGORY_PENALTY_SERUM: float = -0.5
_OPTIONAL_CATEGORY_PENALTY_TONER: float = -0.75


def _ingredient_keyword_hits(product: Product, keywords: frozenset[str]) -> int:
    """Count distinct keywords found as substrings in any ingredient line."""
    if not keywords:
        return 0
    found: set[str] = set()
    for line in product.ingredients:
        for kw in keywords:
            if kw in line:
                found.add(kw)
    return len(found)


def score_product_fit(
    product: Product,
    skin_conditions: list[str],
    semantic_score: float = 0.0,
    *,
    routine_treatment_families: frozenset[str] | None = None,
) -> float:
    """Compute a deterministic fit score for a product given skin conditions.

    Score components (roughly weighted):
      A. Skin-type suitability   : up to +3.0 / down to -2.0 (reduced if overly broad)
      B. Concerns match          : up to +2.5
      C. Benefits                : up to +2.2 plus structured mismatch penalties
      D. Ingredient keywords     : up to +1.8 (condition-aligned positives)
      E. Treatment-heavy mismatch: structured penalties (benefits + ingredients)
      F. Category-specific rules : body-SPF hint on ``description`` only
      G. Optional categories   : SERUM / TONER baseline penalty
      H. Semantic similarity     : up to +0.5 (secondary tie-breaker)
      I. Catalog flags           : soft penalties (tiered fragrance by category;
                                   alcohol unchanged)
      J. Cleanser acne + oily or barrier-sensitive: stronger wash-off fragrance
         penalty; salicylic wash-off deduction; optional extra overlap vs
         ``routine_treatment_families``

    ``routine_treatment_families`` is treatment-family tags already present in
    other routine steps (e.g. for diagnostics). The default matcher leaves this
    unset when building the initial fit matrix so cleanser selection stays
    profile-based only.

    ``name`` / ``description`` are not used for general fit hints.

    Higher is better.
    """
    if not skin_conditions:
        return semantic_score * 0.5 + _catalog_flag_fit_adjustment(product, None)

    conditions_set = frozenset(c.lower().strip() for c in skin_conditions)
    score = 0.0

    wanted_all_concerns: set[str] = set()
    for cond in conditions_set:
        wanted_all_concerns.update(_CONDITION_CONCERNS.get(cond, frozenset()))

    # ----- A. Skin-type suitability -----
    product_types = set(product.skin_types)
    if product_types:
        direct_match_types: set[str] = set()
        compat_match_types: set[str] = set()
        for cond in conditions_set:
            direct_match_types.update(_SKIN_TYPE_DIRECT.get(cond, frozenset()))
            compat_match_types.update(_SKIN_TYPE_COMPATIBLE.get(cond, frozenset()))

        is_broad = "all" in product_types or len(product_types) >= _BROAD_SKIN_TYPE_COUNT
        direct_hit = bool(product_types & direct_match_types)
        compat_hit = bool(product_types & compat_match_types)
        combo_only_strong = (
            "combination" in product_types
            and "oily" not in product_types
            and bool(conditions_set & _STRONG_COMBO_FOR_OILY_ACNE)
        )

        if direct_hit:
            score += 2.0 if is_broad else 3.0
        elif combo_only_strong:
            score += 2.2 if is_broad else 2.6
        elif compat_hit:
            score += 1.0 if is_broad else 1.5
        elif "all" in product_types or "normal" in product_types:
            score += 0.2
        else:
            score -= 2.0

    # ----- B. Concerns match -----
    product_concerns = set(product.concerns)
    if product_concerns:
        wanted_concerns: set[str] = set()
        for cond in conditions_set:
            wanted_concerns.update(_CONDITION_CONCERNS.get(cond, frozenset()))
        overlap = product_concerns & wanted_concerns
        if overlap:
            score += min(2.5, len(overlap) * 0.75)

    # ----- C. Benefits + structured mismatches -----
    product_benefits = set(product.benefits)
    if product_benefits:
        wanted_benefits: set[str] = set()
        neg_benefits: set[str] = set()
        for cond in conditions_set:
            wanted_benefits.update(_CONDITION_BENEFITS.get(cond, frozenset()))
            neg_benefits.update(_NEGATIVE_BENEFITS.get(cond, frozenset()))
        overlap = product_benefits & wanted_benefits
        if overlap:
            score += min(2.2, len(overlap) * 0.55)
        clash = product_benefits & neg_benefits
        if clash:
            score -= min(1.4, len(clash) * 0.55)

    # ----- D. Ingredient keyword fit -----
    wanted_kw: set[str] = set()
    for cond in conditions_set:
        wanted_kw.update(_CONDITION_INGREDIENT_KEYWORDS.get(cond, frozenset()))
    if wanted_kw:
        hits = _ingredient_keyword_hits(product, frozenset(wanted_kw))
        if hits:
            score += min(1.8, hits * 0.6)

    if (
        product.category == ProductCategory.CLEANSER
        and _cleanser_acne_prefers_ff_gentle(conditions_set)
        and _ingredient_keyword_hits(product, frozenset({"salicylic acid"})) > 0
    ):
        score -= _CLEANSER_SA_OILY_ACNE_DEDUCTION
        if routine_treatment_families and (
            routine_treatment_families & _ROUTINE_OVERLAP_REDUCE_CLEANSER_SA
        ):
            score -= _CLEANSER_SA_ROUTINE_OVERLAP_EXTRA

    # ----- E. Treatment-heavy mismatch (structured only) -----
    product_concerns_set = set(product.concerns)
    product_benefits_set = set(product.benefits)
    concern_align = bool(product_concerns_set & wanted_all_concerns)
    justify_acids_for_dry = bool(conditions_set & _TREATMENT_JUSTIFY_CONDITIONS)

    if "dry" in conditions_set and not justify_acids_for_dry:
        b_hit = len(product_benefits_set & _DRY_TREATMENT_BENEFITS)
        ing_hit = _ingredient_keyword_hits(product, _DRY_TREATMENT_INGREDIENTS)
        if b_hit or ing_hit:
            raw = min(2.6, b_hit * 0.7 + ing_hit * 0.9)
            if concern_align:
                raw *= 0.35
            score -= raw

    if "normal" in conditions_set:
        b_hit = len(product_benefits_set & _NORMAL_TREATMENT_BENEFITS)
        ing_hit = _ingredient_keyword_hits(product, _NORMAL_TREATMENT_INGREDIENTS)
        if b_hit or ing_hit:
            raw = min(2.2, b_hit * 0.6 + ing_hit * 0.75)
            if concern_align:
                raw *= 0.35
            score -= raw

    if "oily" in conditions_set:
        ing_hit = _ingredient_keyword_hits(product, _OILY_HEAVY_MOISTURE_INGREDIENTS)
        if ing_hit:
            score -= min(1.5, ing_hit * 0.55)

    # ----- F. Category-specific (description only: body SPF) -----
    desc_lower = product.description.lower()
    if product.category == ProductCategory.SPF:
        for hint in _BODY_SPF_HINTS:
            if hint in desc_lower:
                score -= 1.5
                break

    # ----- G. Optional categories -----
    if product.category == ProductCategory.TONER:
        score += _OPTIONAL_CATEGORY_PENALTY_TONER
    elif product.category == ProductCategory.SERUM:
        score += _OPTIONAL_CATEGORY_PENALTY_SERUM

    # ----- H. Semantic similarity (small bonus) -----
    score += semantic_score * 0.5

    # ----- I. Catalog flags (soft rank penalty; not a safety filter) -----
    score += _catalog_flag_fit_adjustment(product, conditions_set)

    return score


# ---------------------------------------------------------------------------
# Category-aware selection — best product per category
# ---------------------------------------------------------------------------

# Minimum fit score for SERUM/TONER: every picked optional must meet this
# (not only the category’s top candidate — avoids falling through to weak
# alternates after active-conflict skips).
_OPTIONAL_FIT_THRESHOLD: float = 4.0


def select_best_per_category(
    products: list[Product],
    skin_conditions: list[str],
    semantic_scores: dict[str, float] | None = None,
    max_products: int = 5,
) -> list[Product]:
    """Group *products* by category, pick the best-fit per category.

    Essential categories (CLEANSER, MOISTURIZER, SPF) always get a slot
    when a candidate exists.      Optional SERUM/TONER slots require fit at least ``_OPTIONAL_FIT_THRESHOLD``,
    a skin-type direct/compatible gate, and after conflict resolution only
    eligible alternates are considered (no sub-threshold fallback).

    Active-ingredient handling uses a category-aware policy (hard block vs
    soft fit penalty vs allow) so rinse-off products do not hard-block SPF or
    moisturizers, while leave-on treatment stacking stays guarded.

    Output is sorted by category priority.
    """
    if not products:
        return []

    sem = semantic_scores or {}
    conditions_set = frozenset(c.lower().strip() for c in skin_conditions)

    # Compute fit for every product.
    product_fit: dict[str, float] = {}
    for p in products:
        product_fit[p.id] = score_product_fit(
            p, skin_conditions, semantic_score=sem.get(p.id, 0.0),
        )

    # Group by category and sort within each group by fit (descending).
    by_category: dict[ProductCategory, list[Product]] = {}
    for p in products:
        by_category.setdefault(p.category, []).append(p)

    for cat in by_category:
        by_category[cat].sort(key=lambda p: product_fit[p.id], reverse=True)

    selected: dict[ProductCategory, Product] = {}
    routine_segments: list[tuple[ProductCategory, frozenset[str]]] = []

    def _try_pick(
        cat: ProductCategory,
        force: bool,
        *,
        pool: list[Product] | None = None,
    ) -> bool:
        candidates = pool if pool is not None else by_category.get(cat, [])
        chosen = _pick_with_active_policy(
            cat,
            candidates,
            routine_segments,
            product_fit,
            force=force,
            skin_conditions=skin_conditions,
        )
        if chosen is None:
            return False
        selected[cat] = chosen
        routine_segments.append((cat, active_families(chosen)))
        return True

    # Phase 1: essential categories (always included when available).
    for cat in _ESSENTIAL_CATEGORIES:
        if cat in by_category and len(selected) < max_products:
            _try_pick(cat, force=True)

    # Phase 2: optional categories — only candidates at or above the fit floor.
    optional_cats = sorted(
        (c for c in by_category if c not in _ESSENTIAL_CATEGORIES and c not in selected),
        key=lambda c: _CATEGORY_PRIORITY.get(c, 99),
    )
    for cat in optional_cats:
        if len(selected) >= max_products:
            break
        ranked = by_category[cat]
        best = ranked[0]
        best_fit = product_fit[best.id]
        eligible = []
        for p in ranked:
            if product_fit[p.id] < _OPTIONAL_FIT_THRESHOLD:
                continue
            if cat in (ProductCategory.SERUM, ProductCategory.TONER):
                if not _optional_skin_type_gate_passes(p, conditions_set):
                    continue
            eligible.append(p)
        if not eligible:
            logger.debug(
                "Skipping optional %s — no candidate passes fit + gates "
                "(best '%s' fit=%.2f, threshold=%.2f)",
                cat.value, best.name, best_fit, _OPTIONAL_FIT_THRESHOLD,
            )
            continue
        _try_pick(cat, force=False, pool=eligible)

    routine = sorted(
        selected.values(),
        key=lambda p: _CATEGORY_PRIORITY.get(p.category, len(_CATEGORY_PRIORITY)),
    )
    return routine


# ---------------------------------------------------------------------------
# Legacy deterministic fallback (no semantic model)
# ---------------------------------------------------------------------------


def build_routine(
    safe_products: list[Product],
    constraints: UserConstraints,
) -> list[Product]:
    """Select and order products into a category-balanced routine (no ranker)."""
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


def select_balanced_routine(
    ranked_products: list[Product],
    max_products: int = 5,
    *,
    gate_optionals: bool = False,
) -> list[Product]:
    """Pick at most one product per category with active-conflict avoidance.

    Used by the legacy ``build_routine`` path (no semantic model).
    When ``gate_optionals`` is True, optional categories are only included
    if their best candidate is ranked in the top half of the input list.
    """
    if not ranked_products:
        return []

    total = len(ranked_products)
    rank_position: dict[str, int] = {
        p.id: i for i, p in enumerate(ranked_products)
    }
    cutoff_idx = max(1, int(total * 0.5))

    candidates_by_cat: dict[ProductCategory, list[Product]] = {}
    for product in ranked_products:
        candidates_by_cat.setdefault(product.category, []).append(product)

    selected: dict[ProductCategory, Product] = {}
    routine_segments: list[tuple[ProductCategory, frozenset[str]]] = []

    def _try_pick(cat: ProductCategory, force: bool) -> bool:
        pool = candidates_by_cat.get(cat, [])
        chosen = _pick_with_active_policy(
            cat, pool, routine_segments, None, force=force,
        )
        if chosen is None:
            return False
        selected[cat] = chosen
        routine_segments.append((cat, active_families(chosen)))
        return True

    for cat in _ESSENTIAL_CATEGORIES:
        if cat in candidates_by_cat and len(selected) < max_products:
            _try_pick(cat, force=True)

    for product in ranked_products:
        if len(selected) >= max_products:
            break
        cat = product.category
        if cat in selected:
            continue
        if gate_optionals and cat not in _ESSENTIAL_CATEGORIES:
            best_for_cat = candidates_by_cat[cat][0]
            best_rank = rank_position[best_for_cat.id]
            if best_rank >= cutoff_idx:
                continue
        _try_pick(cat, force=False)

    routine = sorted(
        selected.values(),
        key=lambda p: _CATEGORY_PRIORITY.get(p.category, len(_CATEGORY_PRIORITY)),
    )
    return routine


# ---------------------------------------------------------------------------
# Public pipeline orchestrator
# ---------------------------------------------------------------------------

# Scorer callable: (skin_conditions, products) -> {product_id: similarity}
ScorerFn = Callable[[list[str], list[Product]], dict[str, float]]

# Legacy ranker type kept for backward-compat with message_handler signature.
RankerFn = Callable[[list[str], list[Product]], list[Product]]


def match_products(
    catalog: list[Product],
    constraints: UserConstraints,
    skin_conditions: list[str] | None = None,
    ranker: RankerFn | None = None,
    scorer: ScorerFn | None = None,
) -> list[Product]:
    """Full matching pipeline: filter -> fit-score -> best-per-category.

    When *scorer* is provided the pipeline uses category-first fit scoring
    with semantic similarity as a small bonus.  When *ranker* is provided
    but *scorer* is not, the legacy global-rank path is used.  Otherwise
    the result falls back to ``build_routine`` (category-priority ordering).
    """
    if constraints.max_products <= 0:
        raise ValueError(
            f"max_products must be >= 1, got {constraints.max_products}"
        )

    safe = filter_safe_products(catalog, constraints)
    conditions = skin_conditions or []

    # Primary path: category-first fit scoring.
    if scorer is not None and conditions:
        semantic_scores = scorer(conditions, safe)
        selected = select_best_per_category(
            safe,
            conditions,
            semantic_scores=semantic_scores,
            max_products=constraints.max_products,
        )
        logger.debug(
            "request_id=%s match_products (fit-scoring): %d/%d selected",
            constraints.request_id, len(selected), len(safe),
        )
        return selected

    # Legacy path: global semantic ranking.
    if ranker is not None and conditions:
        ranked = ranker(conditions, safe)
        selected = select_balanced_routine(
            ranked, constraints.max_products, gate_optionals=True,
        )
        logger.debug(
            "request_id=%s match_products (legacy-ranking): %d/%d selected",
            constraints.request_id, len(selected), len(safe),
        )
        return selected

    # Fallback: no semantic model available.
    return build_routine(safe, constraints)


# ---------------------------------------------------------------------------
# Routine rationale builder
# ---------------------------------------------------------------------------


def build_routine_rationale(
    catalog: list[Product],
    safe_products: list[Product],
    selected: list[Product],
    skin_conditions: list[str] | None = None,
    used_semantic_ranking: bool = False,
) -> dict[str, Any]:
    """Build a deterministic rationale explaining why the routine was selected."""
    selected_ids = {p.id for p in selected}

    routine_actives: set[str] = set()
    for p in selected:
        routine_actives.update(active_families(p))

    candidates_by_cat: dict[ProductCategory, int] = {}
    for p in safe_products:
        candidates_by_cat[p.category] = candidates_by_cat.get(p.category, 0) + 1

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
            "with category-aware active-ingredient policy; "
            "leave-on fragrance soft penalties stay at -1.0; "
            "for acne with oily skin or with dryness/sensitivity/dehydration/"
            "irritation/barrier-related labels, fragranced cleansers use a "
            "stronger wash-off fragrance penalty, salicylic wash-off fit is "
            "down-weighted (extra when overlapping acne actives are supplied), "
            "and cleanser picks use a wider near-top band with fragrance-free "
            "and gentle-benefit tie-breaks"
        ),
    })

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
        if skin_conditions:
            entry["fit_score"] = round(score_product_fit(p, skin_conditions), 2)
        if p.contains_fragrance:
            if p.category in _LEAVE_ON_FRAGRANCE_PENALTY_CATEGORIES:
                entry["fragrance_ranking_penalty_tier"] = "leave_on"
            elif p.category == ProductCategory.CLEANSER and skin_conditions:
                cs = frozenset(c.lower().strip() for c in skin_conditions)
                if _OILY_ACNE_PROFILE <= cs:
                    entry["fragrance_ranking_penalty_tier"] = "cleanser_oily_acne"
                elif _cleanser_acne_prefers_ff_gentle(cs):
                    entry["fragrance_ranking_penalty_tier"] = (
                        "cleanser_acne_barrier_sensitive"
                    )
                else:
                    entry["fragrance_ranking_penalty_tier"] = "cleanser"
            else:
                entry["fragrance_ranking_penalty_tier"] = "cleanser"
        product_rationales[p.id] = entry

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
