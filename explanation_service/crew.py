"""CrewAI Project-mode crew for explanation generation with task guardrails."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from explanation_service.guardrails import (
    guard_no_banned_medical_terms,
    guard_sources_subset,
    guard_two_sentences,
)
from explanation_service.output_models import (
    DraftExplanation,
    FinalExplanation,
    IngredientEvidence,
    ResearchOutput,
)
from explanation_service.tools import IngredientEvidenceTool, lookup_ingredient

logger = logging.getLogger(__name__)


@CrewBase
class ExplanationCrew:
    """Crew definition using official CrewAI project decorators."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    _last_research_sources: list[str] = []
    _task_cache: dict[str, Task]
    _guardrail_failures: dict[str, int]
    _current_request_id: str = "unknown"

    @agent
    def ingredient_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["ingredient_analyst"],
            tools=[IngredientEvidenceTool],
            verbose=True,
        )

    @agent
    def user_explainer(self) -> Agent:
        return Agent(
            config=self.agents_config["user_explainer"],
            verbose=True,
        )

    @agent
    def safety_auditor(self) -> Agent:
        return Agent(
            config=self.agents_config["safety_auditor"],
            verbose=True,
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
                output_pydantic=DraftExplanation,
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
            verbose=True,
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
            if not item.ingredient.strip() or not item.fact.strip() or not item.source.strip():
                self._log_guardrail_failure(
                    "research_task",
                    "Each research item must include non-empty ingredient/fact/source.",
                )
                return False, "Each research item must include non-empty ingredient/fact/source."
            if not guard_no_banned_medical_terms(item.fact):
                self._log_guardrail_failure(
                    "research_task",
                    "Research facts contain banned medical terms or forbidden phrases.",
                )
                return False, "Research facts contain banned medical terms or forbidden phrases."

        self._last_research_sources = [item.source for item in parsed.items]
        self._reset_guardrail_failures("research_task")
        return True, "Research output passed guardrail checks."

    def _draft_guardrail(self, output: Any):
        parsed = _coerce_draft_output(output)
        if parsed is None:
            self._log_guardrail_failure(
                "draft_task",
                "Draft output must parse into DraftExplanation.",
            )
            return False, "Draft output must parse into DraftExplanation."

        text = parsed.explanation_text.strip()
        if not guard_two_sentences(text):
            self._log_guardrail_failure(
                "draft_task",
                "Draft explanation must contain exactly 2 sentences.",
            )
            return False, "Draft explanation must contain exactly 2 sentences."
        if not guard_no_banned_medical_terms(text):
            self._log_guardrail_failure(
                "draft_task",
                "Draft explanation contains banned medical terms or phrases.",
            )
            return False, "Draft explanation contains banned medical terms or phrases."
        self._reset_guardrail_failures("draft_task")
        return True, "Draft output passed guardrail checks."

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

        research_sources = self._last_research_sources
        if not research_sources:
            self._log_guardrail_failure(
                "audit_task",
                "Research sources were unavailable for subset validation.",
            )
            return False, "Research sources were unavailable for subset validation."

        # If research produced only generic sources, final output must also use generic.
        if set(research_sources) == {"generic"}:
            if parsed.sources != ["generic"]:
                self._log_guardrail_failure(
                    "audit_task",
                    "When research sources are generic-only, final sources must be ['generic'].",
                )
                return False, "When research sources are generic-only, final sources must be ['generic']."
            self._reset_guardrail_failures("audit_task")
            return True, "Audit output passed guardrail checks."

        if not guard_sources_subset(parsed.sources, research_sources):
            self._log_guardrail_failure(
                "audit_task",
                "Final sources must be a subset of research sources.",
            )
            return False, "Final sources must be a subset of research sources."
        self._reset_guardrail_failures("audit_task")
        return True, "Audit output passed guardrail checks."

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
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


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
            if "items" not in payload and all(
                key in payload for key in ("ingredient", "fact", "source")
            ):
                payload = {"items": [payload]}
            return ResearchOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _coerce_draft_output(output: Any) -> DraftExplanation | None:
    payload = _extract_payload(output)
    try:
        if isinstance(payload, DraftExplanation):
            return payload
        if isinstance(payload, FinalExplanation):
            return DraftExplanation(explanation_text=payload.explanation_text)
        if isinstance(payload, str):
            candidate = _strip_json_fence(payload)
            if candidate.startswith("{"):
                parsed = json.loads(candidate)
                return DraftExplanation.model_validate(parsed)
            return DraftExplanation(explanation_text=candidate)
        if isinstance(payload, dict):
            return DraftExplanation.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _coerce_final_output(output: Any) -> FinalExplanation | None:
    payload = _extract_payload(output)
    try:
        if isinstance(payload, FinalExplanation):
            return payload
        if isinstance(payload, str):
            parsed = json.loads(_strip_json_fence(payload))
            return FinalExplanation.model_validate(parsed)
        if isinstance(payload, dict):
            return FinalExplanation.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _extract_payload(output: Any) -> Any:
    if hasattr(output, "raw"):
        return getattr(output, "raw")
    if hasattr(output, "pydantic"):
        return getattr(output, "pydantic")
    return output


def _safe_generic_explanation(product_name: str) -> str:
    return (
        f"{product_name} includes commonly used skincare ingredients for everyday routine support. "
        "Introduce it gradually and monitor how your skin feels over time."
    )


def _allowed_sources_from_ingredients(ingredients: list[str]) -> set[str]:
    return {lookup_ingredient(ingredient)["source"] for ingredient in ingredients}


def _build_safe_fallback(product_name: str) -> FinalExplanation:
    return FinalExplanation(
        explanation_text=_safe_generic_explanation(product_name),
        sources=["generic"],
    )


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

    project = ExplanationCrew()
    project._current_request_id = resolved_request_id
    try:
        result = project.crew().kickoff(
            inputs={
                "skin_conditions": ", ".join(skin_conditions),
                "ingredients": ", ".join(ingredients),
                "product_name": product_name,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in integration test
        logger.exception(
            "Crew kickoff failed, using safe generic fallback. reason=%s",
            exc,
        )
        return _build_safe_fallback(product_name).model_dump()

    final = _coerce_final_output(result)
    if final is None:
        logger.warning(
            "Final output parsing failed, using safe generic fallback."
        )
        return _build_safe_fallback(product_name).model_dump()

    if not guard_two_sentences(final.explanation_text) or not guard_no_banned_medical_terms(
        final.explanation_text
    ):
        logger.warning(
            "Final output failed validation checks, using safe generic fallback."
        )
        final = _build_safe_fallback(product_name)

    allowed_sources = _allowed_sources_from_ingredients(ingredients)
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
