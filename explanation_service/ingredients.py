"""Pure deterministic helpers for ingredient filtering.

Reduces RAG tool calls by selecting only high-value ingredients before
passing them to the LLM-backed research task.
"""

from __future__ import annotations

# Substrings that mark an ingredient as a meaningful active.
_KEY_ACTIVES: tuple[str, ...] = (
    "retinol",
    "retinal",
    "retinoic",
    "niacinamide",
    "hyaluronic acid",
    "salicylic acid",
    "zinc oxide",
    "ceramide",
    "panthenol",
    "azelaic acid",
    "glycolic acid",
    "vitamin c",
    "ascorbic acid",
    "centella asiatica",
    "peptide",
    "bakuchiol",
    "squalane",
    "tocopherol",
    "allantoin",
    "arbutin",
    "kojic acid",
    "lactic acid",
    "benzoyl peroxide",
    "sulfur",
    "tea tree",
    "aloe vera",
    "colloidal oatmeal",
    "glycerin",
    "shea butter",
    "jojoba",
    "rosehip",
    "madecassoside",
    "tranexamic acid",
)

# Ingredients that are noteworthy for safety / explanation quality,
# even though they are not "active" ingredients.
_NOTABLE: tuple[str, ...] = (
    "fragrance",
    "parfum",
    "essential oil",
    "denatured alcohol",
    "alcohol denat",
    "sd alcohol",
    "isopropyl alcohol",
)

# Low-value fillers that should never consume a RAG tool call.
_FILLER: frozenset[str] = frozenset(
    {
        "water",
        "aqua",
        "xanthan gum",
        "sodium chloride",
        "carbomer",
        "disodium edta",
        "tetrasodium edta",
        "citric acid",
        "sodium hydroxide",
        "phenoxyethanol",
        "ethylhexylglycerin",
        "potassium sorbate",
        "sodium benzoate",
        "triethanolamine",
        "acrylates copolymer",
        "hydroxyethylcellulose",
        "polysorbate 20",
        "polysorbate 80",
        "peg-40 hydrogenated castor oil",
    }
)

_MAX_KEY: int = 10
_FALLBACK_MAX: int = 3


def _is_filler(ingredient: str) -> bool:
    return ingredient in _FILLER


def _matches_any(ingredient: str, terms: tuple[str, ...]) -> bool:
    return any(term in ingredient for term in terms)


def extract_key_ingredients(ingredients: list[str]) -> list[str]:
    """Select the most explanation-relevant ingredients from a full INCI list.

    Priority order:
      1. Notable safety-relevant items (fragrance, drying alcohol, etc.)
      2. Known active / functional ingredients
      3. Non-filler remainders (fallback when no actives are found)

    Returns at most ``_MAX_KEY`` items. When no actives or notables match,
    returns up to ``_FALLBACK_MAX`` non-filler ingredients.
    """
    notables: list[str] = []
    actives: list[str] = []
    others: list[str] = []

    for raw in ingredients:
        lower = raw.strip().lower()
        if not lower:
            continue

        if _is_filler(lower):
            continue

        if _matches_any(lower, _NOTABLE):
            notables.append(raw)
        elif _matches_any(lower, _KEY_ACTIVES):
            actives.append(raw)
        else:
            others.append(raw)

    key = notables + actives
    if key:
        return key[:_MAX_KEY]

    return others[:_FALLBACK_MAX]
