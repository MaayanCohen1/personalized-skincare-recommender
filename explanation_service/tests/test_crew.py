"""Unit tests for explanation_service.crew with mocked crew execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai import Task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from explanation_service.guardrails import BANNED_MEDICAL_TERMS, count_sentences
from explanation_service.crew import ExplanationCrew, generate_explanation_for_product
from explanation_service.output_models import FinalExplanation


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")


def _mock_project_with_raw(raw_output: str) -> MagicMock:
    kickoff_result = MagicMock()
    kickoff_result.raw = raw_output

    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = kickoff_result

    project = MagicMock()
    project.crew.return_value = crew_instance
    return project


def test_final_output_is_valid_pydantic_final_explanation() -> None:
    raw = (
        '{"explanation_text":"This moisturizer supports hydration comfort. '
        'It can fit a gentle daily routine.","sources":["CIR (Cosmetic Ingredient Review)"]}'
    )
    with patch(
        "explanation_service.crew.ExplanationCrew",
        return_value=_mock_project_with_raw(raw),
    ):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin"],
        )
    validated = FinalExplanation.model_validate(result)
    assert isinstance(validated, FinalExplanation)


def test_two_sentences_enforced() -> None:
    raw = (
        '{"explanation_text":"This moisturizer supports hydration comfort. '
        'It can fit a gentle daily routine. Use daily.","sources":["CIR (Cosmetic Ingredient Review)"]}'
    )
    with patch(
        "explanation_service.crew.ExplanationCrew",
        return_value=_mock_project_with_raw(raw),
    ):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin"],
        )
    assert count_sentences(result["explanation_text"]) == 2


def test_no_banned_terms() -> None:
    raw = (
        '{"explanation_text":"This serum can treat dryness quickly. '
        'It can fit a gentle daily routine.","sources":["CIR (Cosmetic Ingredient Review)"]}'
    )
    with patch(
        "explanation_service.crew.ExplanationCrew",
        return_value=_mock_project_with_raw(raw),
    ):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Serum",
            ingredients=["glycerin"],
        )

    lowered = result["explanation_text"].lower()
    assert all(word not in lowered for word in BANNED_MEDICAL_TERMS)


def test_sources_non_empty() -> None:
    raw = (
        '{"explanation_text":"This moisturizer supports hydration comfort. '
        'It can fit a gentle daily routine.","sources":[]}'
    )
    with patch(
        "explanation_service.crew.ExplanationCrew",
        return_value=_mock_project_with_raw(raw),
    ):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin"],
        )
    assert len(result["sources"]) > 0


def test_sources_subset_of_research_sources() -> None:
    raw = (
        '{"explanation_text":"This moisturizer supports hydration comfort. '
        'It can fit a gentle daily routine.","sources":["unknown_source"]}'
    )
    with (
        patch(
            "explanation_service.crew.ExplanationCrew",
            return_value=_mock_project_with_raw(raw),
        ),
        patch(
            "explanation_service.crew._allowed_sources_from_ingredients",
            return_value={
                "CIR (Cosmetic Ingredient Review)",
                "Paula\'s Choice Ingredient Dictionary",
            },
        ),
    ):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin", "niacinamide"],
        )

    allowed = {
        "CIR (Cosmetic Ingredient Review)",
        "Paula's Choice Ingredient Dictionary",
    }
    assert set(result["sources"]).issubset(allowed)


def test_crewai_guardrail_retries_up_to_max_attempts() -> None:
    retry_calls = {"count": 0}

    class FakeAgent:
        role = "fake-agent"
        last_messages: list[Any] = []
        verbose = False

        def execute_task(self, task: Task, context: str | None, tools: list[Any]) -> str:
            retry_calls["count"] += 1
            return "still invalid"

    def always_fail_guardrail(output: TaskOutput):
        return False, "forced failure"

    task = Task(
        description="Retry verification task",
        expected_output="Anything",
        guardrail=always_fail_guardrail,
        guardrail_max_retries=3,
    )
    initial_output = TaskOutput(
        name="Retry verification task",
        description="Retry verification task",
        expected_output="Anything",
        raw="invalid output",
        agent="fake-agent",
        output_format=OutputFormat.RAW,
        messages=[],
    )
    agent = FakeAgent()

    with pytest.raises(Exception, match="after 3 retries"):
        task._invoke_guardrail_function(
            task_output=initial_output,
            agent=agent,  # type: ignore[arg-type]
            tools=[],
            guardrail=always_fail_guardrail,
        )

    assert retry_calls["count"] == 3


def test_guardrail_exhaustion_uses_safe_fallback(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    project = MagicMock()
    crew_instance = MagicMock()
    crew_instance.kickoff.side_effect = Exception(
        "Task failed guardrail validation after 3 retries. Last error: forced"
    )
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin"],
        )

    assert result["sources"] == ["generic"]
    assert count_sentences(result["explanation_text"]) == 2
    assert "Crew kickoff failed, using safe generic fallback" in caplog.text


def test_task_identity_is_shared_across_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    project = ExplanationCrew()
    research = project.research_task()
    draft = project.draft_task()
    audit = project.audit_task()

    assert project.research_task() is research
    assert project.draft_task() is draft
    assert project.audit_task() is audit
    assert draft.context[0] is research
    assert audit.context[0] is research
    assert audit.context[1] is draft


def test_crew_configures_output_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("EXPLAIN_CACHE", "true")

    with patch("explanation_service.crew.Crew") as mock_crew_cls:
        project = ExplanationCrew()
        _ = project.crew()

    kwargs = mock_crew_cls.call_args.kwargs
    assert kwargs["output_log_file"] == "logs/explanation_service.json"
    assert kwargs["cache"] is True


def test_missing_api_key_uses_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with patch("explanation_service.crew.ExplanationCrew") as mock_project:
        result = generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="HydraGlow Moisturizer",
            ingredients=["glycerin"],
            request_id="req-no-key",
        )

    assert result["sources"] == ["generic"]
    assert count_sentences(result["explanation_text"]) == 2
    assert not mock_project.called
