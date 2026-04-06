"""Deterministic condition-aware framing hints for the explanation LLM.

Pure module — no ML, no CrewAI, no I/O dependencies.
"""

from __future__ import annotations


def _normalize_conditions(skin_conditions: list[str]) -> set[str]:
    return {
        c.lower().strip().replace("_", " ")
        for c in skin_conditions
        if c is not None and str(c).strip()
    }


def build_condition_guidance(skin_conditions: list[str]) -> str:
    """Return a short framing hint for the LLM based on detected conditions.

    Purely deterministic — helps the explainer avoid mismatched wording
    (e.g. dry-skin framing for an oily-only user, or mattifying talk for
    dry/sensitive profiles).
    """
    s = _normalize_conditions(skin_conditions)
    if "dehydrated" in s:
        s = set(s) | {"dehydration"}

    if "acne" in s and "sensitive" in s:
        return (
            "The user has acne-prone and sensitive skin. "
            "Prefer gentle, non-stripping, barrier-aware wording; avoid harsh or "
            "aggressive treatment language. "
            "Do NOT emphasise strong oil-stripping or heavy mattifying as primary benefits."
        )

    if "acne" in s and "dry" in s:
        return (
            "The user has acne-prone skin with dryness. "
            "Frame benefits around supportive hydration and barrier comfort while "
            "keeping textures lightweight where possible. "
            "Do NOT emphasise oil-control or mattifying as the main story."
        )

    if "oily" in s and "dehydration" in s:
        return (
            "The user has oily yet dehydrated skin (water-light, not necessarily dry). "
            "Frame benefits around lightweight humectant hydration and barrier support "
            "without heavy, occlusive richness. "
            "Do NOT describe the routine as only oil-control or mattifying."
        )

    if "dry" in s and "sensitive" in s:
        return (
            "The user has dry and sensitive skin. "
            "Emphasise soothing, barrier support, moisture retention, and "
            "low-irritation / fragrance-aware fit where relevant. "
            "Do NOT emphasise strong exfoliation, oil stripping, or mattifying."
        )

    if "combination" in s:
        return (
            "The user has combination skin. "
            "Frame benefits around balanced, zone-friendly care — lightweight where "
            "needed, supportive hydration without universal heaviness. "
            "Do NOT write as if the whole face is very oily or very dry only."
        )

    if "oily" in s and "dry" in s:
        return (
            "The user reports both oily and dry areas (combination-style). "
            "Use balanced framing: lightweight hydration and barrier support without "
            "heavy occlusive language everywhere. "
            "Do NOT lean only into intense dry-skin relief or only into mattifying control."
        )

    if "oily" in s and "acne" in s:
        return (
            "The user has oily, acne-prone skin. "
            "Frame benefits around gentle cleansing, pore care, and lightweight hydration. "
            "Do NOT emphasise rich, heavy moisturisation as the primary fit."
        )

    if "oily" in s and "sensitive" in s:
        return (
            "The user has oily and sensitive skin. "
            "Prefer gentle, barrier-friendly oil balance and lightweight hydration. "
            "Do NOT emphasise aggressive oil stripping, harsh exfoliation, or "
            "heavy dry-skin barrier repair language."
        )

    if s == {"sensitive"}:
        return (
            "The user has sensitive skin. "
            "Emphasise soothing, gentle use, low-irritation fit, and barrier comfort. "
            "Do NOT emphasise strong actives, peeling, or aggressive correction language."
        )

    if s <= {"normal", "balanced"} and s:
        return (
            "The user describes generally normal or balanced skin. "
            "Keep explanations moderate — everyday support, comfort, and sensible routine fit. "
            "Do NOT over-emphasise extreme dryness, heavy oil control, or clinical correction."
        )

    if s == {"oily"}:
        return (
            "The user has oily skin only (no acne, no dryness). "
            "Frame benefits around lightweight hydration, oil balance, "
            "and barrier support without heaviness. "
            "Do NOT emphasise dry-skin relief or intense moisture."
        )

    if s == {"dry"}:
        return (
            "The user has dry skin. "
            "Frame benefits around hydration, barrier support, and moisture retention. "
            "Do NOT emphasise mattifying or oil-stripping as the main benefit."
        )

    if s:
        joined = ", ".join(sorted(s))
        return (
            f"The user's skin profile includes: {joined}. "
            "Align benefits with these stated conditions and avoid stressing priorities "
            "that clearly conflict (e.g. heavy mattifying for dry profiles)."
        )

    return ""
