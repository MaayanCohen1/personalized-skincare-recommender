"""CrewAI Project-mode crew for explanation generation with task guardrails."""

from __future__ import annotations

import os

# Suppress CrewAI interactive prompts and telemetry before library import.
os.environ.setdefault("CI", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPTOUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel as _BaseModel

from explanation_service.guardrails import (
    draft_text_for_sentence_check,
    filter_to_research_subset,
    guard_no_banned_medical_terms,
    guard_two_sentences,
    guard_two_sentences_draft,
    preview_for_log,
    sentence_count_draft,
)
from explanation_service.output_models import (
    FinalExplanation,
    IngredientEvidence,
    ResearchOutput,
)
from explanation_service.condition_hints import build_condition_guidance
from explanation_service.ingredients import extract_key_ingredients
from explanation_service.tools import (
    IngredientEvidenceTool,
    lookup_ingredient,
    retrieve_contextual_guidance,
)

logger = logging.getLogger(__name__)


@CrewBase
class ExplanationCrew:
    """Crew definition using official CrewAI project decorators."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    _last_research_sources: list[str] = []
    _contextual_citation_sources: list[str] = []
    _task_cache: dict[str, Task]
    _guardrail_failures: dict[str, int]
    _current_request_id: str = "unknown"
    _current_product_name: str = "unknown"

    @agent
    def ingredient_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["ingredient_analyst"],
            tools=[IngredientEvidenceTool],
            verbose=False,
        )

    @agent
    def user_explainer(self) -> Agent:
        return Agent(
            config=self.agents_config["user_explainer"],
            verbose=False,
        )

    @agent
    def safety_auditor(self) -> Agent:
        return Agent(
            config=self.agents_config["safety_auditor"],
            verbose=False,
        )

    @task
    def research_task(self) -> Task:
        cache = self._get_task_cache()
        if "research_task" not in cache:
            cache["research_task"] = Task(
                config=self.tasks_config["research_task"],
                tools=[IngredientEvidenceTool],
                output_pydantic=ResearchOutput,
                guardrail=self._research_guardrail,
                guardrail_max_retries=3,
            )
        return cache["research_task"]

    @task
    def draft_task(self) -> Task:
        cache = self._get_task_cache()
        if "draft_task" not in cache:
            cache["draft_task"] = Task(
                config=self.tasks_config["draft_task"],
                context=[self.research_task()],
                guardrail=self._draft_guardrail,
                guardrail_max_retries=3,
            )
        return cache["draft_task"]

    @task
    def audit_task(self) -> Task:
        cache = self._get_task_cache()
        if "audit_task" not in cache:
            cache["audit_task"] = Task(
                config=self.tasks_config["audit_task"],
                context=[self.research_task(), self.draft_task()],
                output_pydantic=FinalExplanation,
                guardrail=self._audit_guardrail,
                guardrail_max_retries=3,
            )
        return cache["audit_task"]

    @crew
    def crew(self) -> Crew:
        output_log_file = _output_log_file_path()
        output_log_file.parent.mkdir(parents=True, exist_ok=True)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
            cache=_is_cache_enabled(),
            output_log_file=str(output_log_file),
            task_callback=self._task_callback,
        )

    def _research_guardrail(self, output: Any):
        parsed = _coerce_research_output(output)
        if parsed is None:
            self._log_guardrail_failure(
                "research_task",
                "Research output must parse into ResearchOutput(items=[...]).",
            )
            return False, "Research output must parse into ResearchOutput(items=[...])."

        for item in parsed.items:
            if not item.ingredient.strip() or not item.fact.strip() or not item.sources:
                self._log_guardrail_failure(
                    "research_task",
                    "Each research item must include non-empty ingredient, fact, and sources.",
                )
                return False, "Each research item must include non-empty ingredient, fact, and sources."
            if not all(isinstance(s, str) and s.strip() for s in item.sources):
                self._log_guardrail_failure(
                    "research_task",
                    "Each research item sources entry must be a non-empty string.",
                )
                return False, "Each research item sources entry must be a non-empty string."
            if not guard_no_banned_medical_terms(item.fact):
                self._log_guardrail_failure(
                    "research_task",
                    "Research facts contain banned medical terms or forbidden phrases.",
                )
                return False, "Research facts contain banned medical terms or forbidden phrases."

        seen: set[str] = set()
        flat: list[str] = []
        for item in parsed.items:
            for s in item.sources:
                t = s.strip()
                if t and t not in seen:
                    seen.add(t)
                    flat.append(t)
        self._last_research_sources = flat
        self._reset_guardrail_failures("research_task")
        logger.info("Research guardrail passed.")
        return True, output

    def _draft_guardrail(self, output: Any):
        raw_draft = _coerce_draft_plain_text(output)
        req_id = getattr(self, "_current_request_id", "unknown") or "unknown"
        product = getattr(self, "_current_product_name", "unknown") or "unknown"

        if not raw_draft:
            logger.info(
                "draft_guardrail request_id=%s product_name=%r raw_preview=<empty> sentence_count=n/a",
                req_id,
                product,
            )
            self._log_guardrail_failure(
                "draft_task",
                "Draft output must be non-empty plain text (two sentences).",
            )
            return False, "Draft output must be non-empty plain text (two sentences)."

        logger.info(
            "draft_guardrail request_id=%s product_name=%r raw_preview=%r",
            req_id,
            product,
            preview_for_log(raw_draft, max_len=400),
        )

        checked = draft_text_for_sentence_check(raw_draft)
        n_sent = sentence_count_draft(raw_draft)
        logger.info(
            "draft_guardrail request_id=%s product_name=%r sentence_count=%d checked_preview=%r",
            req_id,
            product,
            n_sent,
            preview_for_log(checked, max_len=400),
        )

        if not guard_two_sentences_draft(raw_draft):
            logger.warning(
                "draft_guardrail FAILED request_id=%s product_name=%r sentence_count=%d (expected 2)",
                req_id,
                product,
                n_sent,
            )
            self._log_guardrail_failure(
                "draft_task",
                "Draft explanation must contain exactly 2 sentences.",
            )
            return False, "Draft explanation must contain exactly 2 sentences."
        if not guard_no_banned_medical_terms(raw_draft):
            logger.warning(
                "draft_guardrail FAILED request_id=%s product_name=%r banned_terms",
                req_id,
                product,
            )
            self._log_guardrail_failure(
                "draft_task",
                "Draft explanation contains banned medical terms or phrases.",
            )
            return False, "Draft explanation contains banned medical terms or phrases."
        self._reset_guardrail_failures("draft_task")
        logger.info(
            "draft_guardrail PASSED request_id=%s product_name=%r sentence_count=%d",
            req_id,
            product,
            n_sent,
        )
        return True, output

    def _audit_guardrail(self, output: Any):
        parsed = _coerce_final_output(output)
        if parsed is None:
            self._log_guardrail_failure(
                "audit_task",
                "Audit output must parse into FinalExplanation.",
            )
            return False, "Audit output must parse into FinalExplanation."

        text = parsed.explanation_text.strip()
        if not guard_two_sentences(text):
            self._log_guardrail_failure(
                "audit_task",
                "Final explanation must contain exactly 2 sentences.",
            )
            return False, "Final explanation must contain exactly 2 sentences."
        if not guard_no_banned_medical_terms(text):
            self._log_guardrail_failure(
                "audit_task",
                "Final explanation contains banned medical terms or phrases.",
            )
            return False, "Final explanation contains banned medical terms or phrases."

        _FALLBACK_SOURCES = {"generic", "fallback"}

        final_source_set = {s.strip().lower() for s in parsed.sources if s.strip()}
        is_fallback = (not parsed.sources) or final_source_set.issubset(_FALLBACK_SOURCES)

        if is_fallback:
            logger.info("Audit guardrail: sources are fallback-only — accepting.")
            self._reset_guardrail_failures("audit_task")
            return True, output

        ctx_extra = getattr(self, "_contextual_citation_sources", None) or []
        seen_merge: set[str] = set()
        research_sources: list[str] = []
        for s in list(self._last_research_sources) + list(ctx_extra):
            t = s.strip() if isinstance(s, str) else ""
            if t and t not in seen_merge:
                seen_merge.add(t)
                research_sources.append(t)

        if not research_sources:
            logger.warning("Audit guardrail: no research sources available — accepting output as-is.")
            self._reset_guardrail_failures("audit_task")
            return True, output

        # Filter audit sources to ingredient-research + contextual-guidance ids.
        valid_sources = filter_to_research_subset(parsed.sources, research_sources)

        if valid_sources:
            logger.info(
                "Audit guardrail: filtered %d -> %d valid sources.",
                len(parsed.sources),
                len(valid_sources),
            )
        else:
            # No exact matches; inject the research sources as a
            # deterministic fallback so the explanation is still usable.
            seen: set[str] = set()
            valid_sources = []
            for s in research_sources:
                if s not in seen:
                    seen.add(s)
                    valid_sources.append(s)
            logger.warning(
                "Audit guardrail: no valid source subset; injecting %d research source(s).",
                len(valid_sources),
            )

        corrected = FinalExplanation(
            explanation_text=parsed.explanation_text,
            sources=valid_sources,
        )
        self._reset_guardrail_failures("audit_task")
        return True, corrected

    def _get_task_cache(self) -> dict[str, Task]:
        if not hasattr(self, "_task_cache"):
            self._task_cache = {}
        return self._task_cache

    def _get_guardrail_failures(self) -> dict[str, int]:
        if not hasattr(self, "_guardrail_failures"):
            self._guardrail_failures = {}
        return self._guardrail_failures

    def _log_guardrail_failure(self, task_name: str, reason: str) -> None:
        failures = self._get_guardrail_failures()
        failures[task_name] = failures.get(task_name, 0) + 1
        logger.warning(
            "Guardrail failed for %s (retry_count=%d): %s",
            task_name,
            failures[task_name],
            reason,
        )

    def _reset_guardrail_failures(self, task_name: str) -> None:
        failures = self._get_guardrail_failures()
        if task_name in failures:
            failures[task_name] = 0

    def _task_callback(self, task_output: Any) -> None:
        task_name = _get_task_output_field(task_output, "name", "unknown_task")
        summary = _summarize_task_output(task_output)
        status = "success" if summary != "empty output" else "failure"
        logger.info(
            "Task callback request_id=%s task=%s status=%s summary=%s",
            self._current_request_id,
            task_name,
            status,
            summary,
        )


def _strip_json_fence(raw: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_research_output(output: Any) -> ResearchOutput | None:
    payload = _extract_payload(output)
    try:
        if isinstance(payload, ResearchOutput):
            return payload
        if isinstance(payload, list):
            payload = {"items": payload}
        if isinstance(payload, str):
            parsed = json.loads(_strip_json_fence(payload))
            payload = {"items": parsed} if isinstance(parsed, list) else parsed
        if isinstance(payload, dict):
            if "items" not in payload and all(k in payload for k in ("ingredient", "fact")) and (
                "sources" in payload or "source" in payload
            ):
                payload = {"items": [payload]}
            return ResearchOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _coerce_draft_plain_text(output: Any) -> str | None:
    """Extract plain-text draft from TaskOutput or string (no structured draft schema)."""
    payload = _extract_payload(output)
    if payload is None:
        return None
    if isinstance(payload, str):
        t = _strip_json_fence(payload).strip()
        return t if t else None
    # Defensive: CrewAI retry paths may still surface a Pydantic instance briefly.
    if isinstance(payload, _BaseModel):
        et = getattr(payload, "explanation_text", None)
        if isinstance(et, str) and et.strip():
            return et.strip()
    return None


def _coerce_final_output(output: Any) -> FinalExplanation | None:
    payload = _extract_payload(output)
    try:
        if isinstance(payload, FinalExplanation):
            return payload
        if isinstance(payload, _BaseModel):
            return FinalExplanation.model_validate(payload.model_dump())
        if isinstance(payload, str):
            cleaned = _strip_json_fence(payload)
            parsed = json.loads(cleaned)
            return FinalExplanation.model_validate(parsed)
        if isinstance(payload, dict):
            return FinalExplanation.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        raw_preview = repr(payload)[:300] if payload is not None else "None"
        logger.exception("_coerce_final_output failed — raw payload preview: %s", raw_preview)
        return None
    return None


def _extract_payload(output: Any) -> Any:
    """Pull the usable payload from a CrewAI TaskOutput (or plain value).

    Preference order:
      1. .pydantic  — only if it is an actual Pydantic BaseModel instance
      2. .raw       — only if it is a string (CrewAI sometimes stuffs a
         Pydantic object here which breaks json.loads downstream)
      3. .raw       — any other non-None value
      4. the output itself
    """
    pydantic_val = getattr(output, "pydantic", None)
    if isinstance(pydantic_val, _BaseModel):
        return pydantic_val

    raw_val = getattr(output, "raw", None)
    if isinstance(raw_val, str):
        return raw_val
    if isinstance(raw_val, _BaseModel):
        return raw_val
    if raw_val is not None:
        return raw_val

    return output


def _safe_generic_explanation(product_name: str) -> str:
    return (
        f"{product_name} includes commonly used skincare ingredients for everyday routine support. "
        "Introduce it gradually and monitor how your skin feels over time."
    )


def _allowed_sources_from_ingredients(ingredients: list[str]) -> set[str]:
    """Union of RAG chunk ids from ``lookup_ingredient`` (``sources`` list only)."""
    allowed: set[str] = set()
    for ingredient in ingredients:
        data = lookup_ingredient(ingredient)
        for s in data.get("sources") or []:
            if isinstance(s, str) and s.strip():
                allowed.add(s.strip())
    return allowed


def _allowed_citation_sources(
    ingredients: list[str],
    contextual_sources: list[str] | None,
) -> set[str]:
    """Ingredient tool ids plus contextual guidance chunk ids (for final filtering)."""
    allowed = _allowed_sources_from_ingredients(ingredients)
    for s in contextual_sources or []:
        if isinstance(s, str) and (t := s.strip()):
            allowed.add(t)
    return allowed


def _build_safe_fallback(product_name: str) -> FinalExplanation:
    return FinalExplanation(
        explanation_text=_safe_generic_explanation(product_name),
        sources=["generic"],
    )


def _truncate_text(text: str, max_len: int) -> str:
    t = text.strip()
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 3].rsplit(" ", 1)[0]
    return cut + "..." if cut else t[: max_len - 3] + "..."


def _format_product_profile(
    *,
    product_name: str,
    product_category: str | None,
    product_description: str | None,
    product_skin_types: list[str] | None,
    product_concerns: list[str] | None,
    product_benefits: list[str] | None,
    contains_fragrance: bool | None,
    contains_alcohol: bool | None,
) -> str:
    lines: list[str] = []
    cat = (product_category or "").strip() or "Not specified"
    lines.append(f"- Category: {cat}")
    lines.append(f"- Product name: {product_name}")
    desc = (product_description or "").strip()
    if desc:
        lines.append(f"- Description: {_truncate_text(desc, 400)}")
    if product_skin_types:
        joined = ", ".join(product_skin_types[:12])
        lines.append(f"- Target / listed skin types: {joined}")
    if product_concerns:
        lines.append(
            "- Product concerns (from catalog): "
            + ", ".join(product_concerns[:12])
        )
    if product_benefits:
        lines.append(
            "- Product benefits (from catalog): "
            + ", ".join(product_benefits[:12])
        )
    if contains_fragrance is True:
        lines.append("- Formulation note: listed as containing fragrance.")
    elif contains_fragrance is False:
        lines.append("- Formulation note: listed as fragrance-free.")
    if contains_alcohol is True:
        lines.append("- Formulation note: listed as containing alcohol.")
    elif contains_alcohol is False:
        lines.append("- Formulation note: listed without alcohol.")
    return "\n".join(lines)


def _format_selection_context(product_rationale: dict[str, Any] | None) -> str:
    if not product_rationale:
        return "No matching-layer rationale was provided for this product."
    lines: list[str] = []
    role = product_rationale.get("role")
    if isinstance(role, str) and role.strip():
        lines.append(
            "- Routine role: "
            f"{role.strip()} (essential = core step such as cleanse, moisturize, or SPF; "
            "optional = add-on such as serum or toner)."
        )
    cat = product_rationale.get("category")
    if isinstance(cat, str) and cat.strip():
        lines.append(f"- Selected category slot: {cat.strip()}.")
    fs = product_rationale.get("fit_score")
    if fs is not None:
        lines.append(f"- Condition fit score (from matcher): {fs}.")
    sa = product_rationale.get("strong_actives")
    if isinstance(sa, list) and sa:
        actives = ", ".join(str(x) for x in sa[:10])
        lines.append(f"- Notable active families: {actives}.")
    tier = product_rationale.get("fragrance_ranking_penalty_tier")
    if isinstance(tier, str) and tier.strip():
        lines.append(
            "- Fragrance policy note (internal ranking): "
            f"{tier.strip()} — use only to justify gentle vs leave-on wording, not as a medical claim."
        )
    return (
        "\n".join(lines)
        if lines
        else "No matching-layer rationale was provided for this product."
    )


def _format_visual_context(image_analysis: dict[str, Any] | None) -> str:
    if not image_analysis:
        return "None provided."
    parts: list[str] = []
    vs = image_analysis.get("visual_signals")
    if isinstance(vs, list) and vs:
        parts.append("Visual signals: " + ", ".join(str(x) for x in vs[:10]) + ".")
    sc = image_analysis.get("skin_conditions")
    if isinstance(sc, list) and sc:
        parts.append(
            "Image-linked conditions: " + ", ".join(str(x) for x in sc[:10]) + "."
        )
    return " ".join(parts) if parts else "None provided."


def _build_explanation_inputs(
    *,
    skin_conditions: list[str],
    product_name: str,
    key_ingredients: list[str],
    condition_guidance: str,
    product_category: str | None = None,
    product_description: str | None = None,
    product_skin_types: list[str] | None = None,
    product_concerns: list[str] | None = None,
    product_benefits: list[str] | None = None,
    contains_fragrance: bool | None = None,
    contains_alcohol: bool | None = None,
    product_rationale: dict[str, Any] | None = None,
    image_analysis: dict[str, Any] | None = None,
    contextual_guidance: str = "",
) -> dict[str, str]:
    profile = _format_product_profile(
        product_name=product_name,
        product_category=product_category,
        product_description=product_description,
        product_skin_types=product_skin_types,
        product_concerns=product_concerns,
        product_benefits=product_benefits,
        contains_fragrance=contains_fragrance,
        contains_alcohol=contains_alcohol,
    )
    return {
        "skin_conditions": ", ".join(skin_conditions),
        "ingredients": ", ".join(key_ingredients),
        "product_name": product_name,
        "product_category": (product_category or "").strip() or "Not specified",
        "product_profile": profile,
        "selection_context": _format_selection_context(product_rationale),
        "visual_context": _format_visual_context(image_analysis),
        "condition_guidance": condition_guidance,
        "contextual_guidance": contextual_guidance,
    }


def _has_any_llm_api_key() -> bool:
    candidate_keys = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )
    for key in candidate_keys:
        value = os.getenv(key, "").strip()
        if value:
            return True
    return False


def _is_cache_enabled() -> bool:
    raw = os.getenv("EXPLAIN_CACHE", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _output_log_file_path() -> Path:
    return Path("logs") / "explanation_service.json"


def _build_request_id(
    skin_conditions: list[str], product_name: str, ingredients: list[str]
) -> str:
    seed = "|".join(
        [
            product_name,
            ",".join(skin_conditions),
            ",".join(ingredients),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"exp-{digest}"


def _get_task_output_field(task_output: Any, field: str, default: str) -> str:
    value = getattr(task_output, field, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _summarize_task_output(task_output: Any) -> str:
    pydantic_value = getattr(task_output, "pydantic", None)
    json_value = getattr(task_output, "json_dict", None)
    raw_value = getattr(task_output, "raw", "")

    if pydantic_value is not None:
        model_name = pydantic_value.__class__.__name__
        if hasattr(pydantic_value, "items"):
            count = len(getattr(pydantic_value, "items", []))
            return f"{model_name} items={count}"
        if hasattr(pydantic_value, "sources"):
            count = len(getattr(pydantic_value, "sources", []))
            return f"{model_name} sources={count}"
        return model_name

    if isinstance(json_value, dict) and json_value:
        keys = ",".join(sorted(json_value.keys())[:4])
        return f"json keys={keys}"

    raw_text = str(raw_value).strip()
    if not raw_text:
        return "empty output"
    compact = " ".join(raw_text.split())
    return (compact[:140] + "...") if len(compact) > 140 else compact


def generate_explanation_for_product(
    skin_conditions: list[str],
    product_name: str,
    ingredients: list[str],
    request_id: str | None = None,
    *,
    product_category: str | None = None,
    product_description: str | None = None,
    product_skin_types: list[str] | None = None,
    product_concerns: list[str] | None = None,
    product_benefits: list[str] | None = None,
    contains_fragrance: bool | None = None,
    contains_alcohol: bool | None = None,
    product_rationale: dict[str, Any] | None = None,
    image_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate final explanation JSON from the audit task output."""
    resolved_request_id = request_id or _build_request_id(
        skin_conditions=skin_conditions,
        product_name=product_name,
        ingredients=ingredients,
    )

    if not _has_any_llm_api_key():
        logger.warning("API key missing -> fallback request_id=%s", resolved_request_id)
        return _build_safe_fallback(product_name).model_dump()

    key_ingredients = extract_key_ingredients(ingredients)
    logger.info(
        "Filtered %d -> %d key ingredients for research (request_id=%s)",
        len(ingredients),
        len(key_ingredients),
        resolved_request_id,
    )

    condition_guidance = build_condition_guidance(skin_conditions)
    ctx_pack = retrieve_contextual_guidance(
        skin_conditions=skin_conditions,
        product_category=product_category,
        product_description=product_description,
        product_skin_types=product_skin_types,
        product_concerns=product_concerns,
        product_benefits=product_benefits,
        contains_fragrance=contains_fragrance,
        contains_alcohol=contains_alcohol,
        product_rationale=product_rationale,
        image_analysis=image_analysis,
        condition_guidance=condition_guidance,
    )
    kickoff_inputs = _build_explanation_inputs(
        skin_conditions=skin_conditions,
        product_name=product_name,
        key_ingredients=key_ingredients,
        condition_guidance=condition_guidance,
        product_category=product_category,
        product_description=product_description,
        product_skin_types=product_skin_types,
        product_concerns=product_concerns,
        product_benefits=product_benefits,
        contains_fragrance=contains_fragrance,
        contains_alcohol=contains_alcohol,
        product_rationale=product_rationale,
        image_analysis=image_analysis,
        contextual_guidance=ctx_pack["guidance_text"],
    )

    project = ExplanationCrew()
    project._contextual_citation_sources = list(ctx_pack.get("sources") or [])
    project._current_request_id = resolved_request_id
    project._current_product_name = product_name
    try:
        result = project.crew().kickoff(inputs=kickoff_inputs)
    except Exception as exc:  # pragma: no cover - exercised in integration test
        logger.exception(
            "Crew kickoff failed, using safe generic fallback. reason=%s",
            exc,
        )
        return _build_safe_fallback(product_name).model_dump()

    final = _coerce_final_output(result)
    if final is None:
        raw_preview = repr(getattr(result, "raw", result))[:400]
        logger.warning(
            "Final output parsing failed, using safe generic fallback. raw_preview=%s",
            raw_preview,
        )
        return _build_safe_fallback(product_name).model_dump()

    if not guard_two_sentences(final.explanation_text) or not guard_no_banned_medical_terms(
        final.explanation_text
    ):
        logger.warning(
            "Final output failed validation checks, using safe generic fallback."
        )
        final = _build_safe_fallback(product_name)

    allowed_sources = _allowed_citation_sources(
        ingredients,
        ctx_pack.get("sources"),
    )
    filtered_sources = [source for source in final.sources if source in allowed_sources]

    if allowed_sources == {"generic"}:
        logger.info(
            "Only generic sources available; returning generic-safe explanation."
        )
        final = _build_safe_fallback(product_name)
    elif not filtered_sources:
        logger.warning(
            "No valid source subset found in final output; assigning deterministic fallback source."
        )
        fallback_source = sorted(allowed_sources)[0] if allowed_sources else "generic"
        final = FinalExplanation(
            explanation_text=final.explanation_text,
            sources=[fallback_source],
        )
    else:
        final = FinalExplanation(
            explanation_text=final.explanation_text,
            sources=filtered_sources,
        )

    if not final.sources:
        logger.warning(
            "Final sources empty after validation; using safe generic fallback."
        )
        final = _build_safe_fallback(product_name)

    return final.model_dump()
