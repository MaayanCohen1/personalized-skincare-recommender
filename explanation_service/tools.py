"""Custom CrewAI tools for the explanation service.

Provides ingredient evidence lookup backed by a small local knowledge base.
"""

from __future__ import annotations

import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)

_EVIDENCE_DB: dict[str, dict[str, str]] = {
    "niacinamide": {
        "fact": (
            "Niacinamide (vitamin B3) is widely used in skincare for its "
            "ability to support the skin barrier and help maintain an even "
            "skin tone."
        ),
        "source": "Paula's Choice Ingredient Dictionary",
    },
    "hyaluronic acid": {
        "fact": (
            "Hyaluronic acid is a humectant that attracts and holds moisture "
            "in the skin, helping to maintain hydration levels."
        ),
        "source": "CIR (Cosmetic Ingredient Review)",
    },
    "salicylic acid": {
        "fact": (
            "Salicylic acid is a beta-hydroxy acid (BHA) commonly used to "
            "help exfoliate the skin and unclog pores."
        ),
        "source": "American Academy of Dermatology (AAD) Public Education",
    },
    "retinol": {
        "fact": (
            "Retinol (a vitamin A derivative) is used in skincare to support "
            "skin cell turnover and improve the appearance of fine lines."
        ),
        "source": "CIR (Cosmetic Ingredient Review)",
    },
    "vitamin c": {
        "fact": (
            "Vitamin C (ascorbic acid) is an antioxidant used in skincare to "
            "help protect against environmental stressors and brighten skin "
            "appearance."
        ),
        "source": "Paula's Choice Ingredient Dictionary",
    },
    "aloe vera": {
        "fact": (
            "Aloe vera is used in skincare for its soothing and moisturizing "
            "properties, often applied to help calm irritated skin."
        ),
        "source": "NCCIH (National Center for Complementary and Integrative Health)",
    },
    "glycerin": {
        "fact": (
            "Glycerin is a humectant that draws water to the skin surface, "
            "helping to keep the skin hydrated and supple."
        ),
        "source": "CIR (Cosmetic Ingredient Review)",
    },
    "zinc oxide": {
        "fact": (
            "Zinc oxide is a mineral UV filter used in sunscreens to help "
            "protect skin from UVA and UVB rays."
        ),
        "source": "FDA Sunscreen Monograph (OTC)",
    },
}

_GENERIC_ENTRY: dict[str, str] = {
    "fact": (
        "This ingredient is commonly found in skincare formulations and is "
        "generally considered suitable for topical use."
    ),
    "source": "generic",
}


def lookup_ingredient(ingredient_name: str) -> dict[str, str]:
    """Pure lookup against the local evidence database."""
    key = ingredient_name.strip().lower()
    entry = _EVIDENCE_DB.get(key, _GENERIC_ENTRY)
    logger.debug("lookup_ingredient(%r) -> source=%s", key, entry["source"])
    return entry


@tool("IngredientEvidenceTool")
def IngredientEvidenceTool(ingredient_name: str) -> dict[str, str]:
    """Look up skincare evidence for a cosmetic ingredient.

    Returns a dictionary with 'fact' and 'source' keys describing the
    ingredient's role in skincare, backed by a curated local database.
    """
    return lookup_ingredient(ingredient_name)
