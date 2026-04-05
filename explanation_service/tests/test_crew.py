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


@pytest.fixture(autouse=True)
def _stub_lookup_ingredient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent crew-level tests from reaching the real RAG/vector pipeline."""
    monkeypatch.setattr(
        "explanation_service.crew.lookup_ingredient",
        lambda ingredient_name: {"fact": "stub", "source": f"stub:{ingredient_name}"},
    )


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


def test_audit_json_string_output_parses_without_exception() -> None:
    """Regression: audit_task returns raw JSON string (not Pydantic). The
    pipeline must parse it into a valid FinalExplanation dict without
    TypeError or double-encoding."""
    raw_json = (
        '{"explanation_text":"Niacinamide supports the skin barrier and helps maintain '
        'an even tone. It is widely used in daily skincare routines.",'
        '"sources":["ingredients_dictionary.md#p1"]}'
    )

    mock_result = MagicMock()
    mock_result.pydantic = None
    mock_result.raw = raw_json

    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = mock_result

    project = MagicMock()
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        result = generate_explanation_for_product(
            skin_conditions=["oily"],
            product_name="Barrier Cream",
            ingredients=["niacinamide"],
        )

    assert isinstance(result, dict)
    assert "explanation_text" in result
    assert "sources" in result
    assert isinstance(result["explanation_text"], str)
    assert isinstance(result["sources"], list)
    validated = FinalExplanation.model_validate(result)
    assert count_sentences(validated.explanation_text) == 2


def test_audit_pydantic_object_in_raw_parses_without_exception() -> None:
    """Regression: CrewAI sometimes puts a Pydantic object in .raw instead of
    a string. The pipeline must still succeed."""
    pydantic_obj = FinalExplanation(
        explanation_text=(
            "Niacinamide supports the skin barrier and helps maintain an even tone. "
            "It is widely used in daily skincare routines."
        ),
        sources=["ingredients_dictionary.md#p1"],
    )

    mock_result = MagicMock()
    mock_result.pydantic = pydantic_obj
    mock_result.raw = pydantic_obj

    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = mock_result

    project = MagicMock()
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        result = generate_explanation_for_product(
            skin_conditions=["oily"],
            product_name="Barrier Cream",
            ingredients=["niacinamide"],
        )

    assert isinstance(result, dict)
    assert isinstance(result["explanation_text"], str)
    assert isinstance(result["sources"], list)


def test_key_ingredients_filter_applied_before_kickoff() -> None:
    """Only key ingredients should reach the crew, not the full INCI list."""
    captured_inputs: dict[str, Any] = {}

    kickoff_result = MagicMock()
    kickoff_result.raw = (
        '{"explanation_text":"Niacinamide supports skin barrier comfort. '
        'It is widely used in daily routines.","sources":["generic"]}'
    )

    crew_instance = MagicMock()
    crew_instance.kickoff.side_effect = lambda inputs: (
        captured_inputs.update(inputs) or kickoff_result
    )

    project = MagicMock()
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        generate_explanation_for_product(
            skin_conditions=["dryness"],
            product_name="Test Product",
            ingredients=[
                "water",
                "niacinamide",
                "carbomer",
                "xanthan gum",
                "retinol",
                "sodium chloride",
            ],
        )

    ingredients_sent = captured_inputs["ingredients"]
    assert "niacinamide" in ingredients_sent
    assert "retinol" in ingredients_sent
    assert "water" not in ingredients_sent
    assert "carbomer" not in ingredients_sent
    assert "xanthan gum" not in ingredients_sent


def test_audit_guardrail_filters_sources_to_research_subset() -> None:
    """Audit guardrail should keep only sources that match research output."""
    project = ExplanationCrew()
    project._last_research_sources = ["dict.md#p1", "safety.md#p0"]

    final = FinalExplanation(
        explanation_text=(
            "This product supports skin barrier comfort. "
            "It is suitable for daily use."
        ),
        sources=["dict.md#p1", "completely_unknown"],
    )
    passed, result = project._audit_guardrail(final)

    assert passed is True
    assert result.sources == ["dict.md#p1"]


def test_audit_guardrail_injects_research_sources_when_none_match() -> None:
    """When no audit sources match research, inject research sources as fallback."""
    project = ExplanationCrew()
    project._last_research_sources = ["dict.md#p1", "guide.md#p2"]

    final = FinalExplanation(
        explanation_text=(
            "This product supports skin barrier comfort. "
            "It is suitable for daily use."
        ),
        sources=["unknown_a", "unknown_b"],
    )
    passed, result = project._audit_guardrail(final)

    assert passed is True
    assert set(result.sources) == {"dict.md#p1", "guide.md#p2"}


def test_audit_guardrail_accepts_fallback_sources_unchanged() -> None:
    """Generic/fallback sources should be accepted without filtering."""
    project = ExplanationCrew()
    project._last_research_sources = ["dict.md#p1"]

    final = FinalExplanation(
        explanation_text=(
            "This product includes common skincare ingredients. "
            "Introduce it gradually."
        ),
        sources=["generic"],
    )
    passed, _ = project._audit_guardrail(final)
    assert passed is True


def test_tracing_env_vars_set_on_import() -> None:
    """Importing crew.py should set CI and telemetry env vars."""
    import os
    assert os.environ.get("CI") == "true"
    assert os.environ.get("CREWAI_TELEMETRY_OPTOUT") == "true"
    assert os.environ.get("OTEL_SDK_DISABLED") == "true"


def test_oily_only_condition_guidance_passed_to_kickoff() -> None:
    """For oily-only conditions, condition_guidance in kickoff inputs
    should instruct the LLM to avoid dry-skin framing."""
    captured_inputs: dict[str, Any] = {}

    kickoff_result = MagicMock()
    kickoff_result.raw = (
        '{"explanation_text":"This lightweight formula helps control oil. '
        'It supports barrier comfort without heaviness.","sources":["generic"]}'
    )

    crew_instance = MagicMock()
    crew_instance.kickoff.side_effect = lambda inputs: (
        captured_inputs.update(inputs) or kickoff_result
    )

    project = MagicMock()
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        generate_explanation_for_product(
            skin_conditions=["oily"],
            product_name="Oil Control Cleanser",
            ingredients=["niacinamide"],
        )

    assert "condition_guidance" in captured_inputs
    guidance = captured_inputs["condition_guidance"]
    assert "lightweight" in guidance.lower()
    assert "NOT" in guidance


def test_acne_condition_guidance_passed_to_kickoff() -> None:
    """For oily+acne, guidance should mention pore care."""
    captured_inputs: dict[str, Any] = {}

    kickoff_result = MagicMock()
    kickoff_result.raw = (
        '{"explanation_text":"This formula supports pore care. '
        'It helps balance oily skin.","sources":["generic"]}'
    )

    crew_instance = MagicMock()
    crew_instance.kickoff.side_effect = lambda inputs: (
        captured_inputs.update(inputs) or kickoff_result
    )

    project = MagicMock()
    project.crew.return_value = crew_instance

    with patch("explanation_service.crew.ExplanationCrew", return_value=project):
        generate_explanation_for_product(
            skin_conditions=["oily", "acne"],
            product_name="Acne Cleanser",
            ingredients=["salicylic acid"],
        )

    guidance = captured_inputs["condition_guidance"]
    assert "pore" in guidance.lower()


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
