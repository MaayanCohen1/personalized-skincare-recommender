"""Deterministic condition-aware framing hints for the explanation LLM.

Pure module — no ML, no CrewAI, no I/O dependencies.
"""

from __future__ import annotations


def build_condition_guidance(skin_conditions: list[str]) -> str:
    """Return a short framing hint for the LLM based on detected conditions.

    Purely deterministic — helps the explainer avoid mismatched wording
    (e.g. dry-skin framing for an oily-only user).
    """
    conditions_set = {c.lower().strip() for c in skin_conditions}

    if conditions_set == {"oily"}:
        return (
            "The user has oily skin only (no acne, no dryness). "
            "Frame benefits around lightweight hydration, oil control, "
            "and barrier support without heaviness. "
            "Do NOT emphasise dry-skin relief or intense moisture."
        )

    if "oily" in conditions_set and "acne" in conditions_set:
        return (
            "The user has oily, acne-prone skin. "
            "Frame benefits around gentle cleansing, pore care, "
            "and lightweight hydration."
        )

    if conditions_set == {"dry"}:
        return (
            "The user has dry skin. "
            "Frame benefits around deep hydration, barrier repair, "
            "and moisture retention."
        )

    return ""
