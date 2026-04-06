"""Unit tests for explanation consumer publish contract."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import explanation_service.consumer as consumer_mod
from explanation_service.consumer import _handle_matched_payload, publish_routine_completed
from shared.models import Product, ProductCategory, RoutineCompletedEvent


@pytest.fixture(autouse=True)
def _stub_explain_product(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing CrewAI when tests exercise the RabbitMQ handler only."""

    def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "explanation_text": "Stub sentence one. Stub sentence two.",
            "sources": ["generic"],
        }

    monkeypatch.setattr(consumer_mod, "_explain_product", _fake)


def test_publish_routine_completed_routing_and_correlation() -> None:
    channel = MagicMock()
    request_id = "req-2"
    event = RoutineCompletedEvent(
        matched_products=[
            Product(
                id="p1",
                name="Test Product",
                category=ProductCategory.SPF,
                ingredients=["zinc oxide"],
                description="desc",
            )
        ],
        explanations={"p1": "Two safe sentences."},
    )

    publish_routine_completed(channel, event=event, request_id=request_id)

    channel.basic_publish.assert_called_once()
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == "routine.events"
    assert kwargs["routing_key"] == "routine.completed"
    assert kwargs["properties"].correlation_id == request_id

    payload = json.loads(kwargs["body"])
    assert payload["request_id"] == request_id
    assert "event" in payload
    assert payload["event"]["explanations"]["p1"] == "Two safe sentences."


def test_publish_routine_completed_includes_analysis_fields() -> None:
    """image_analysis and routine_rationale should appear in the published body."""
    channel = MagicMock()
    event = RoutineCompletedEvent(
        matched_products=[],
        explanations={},
    )
    image_analysis = {"visual_signals": ["oily"], "skin_conditions": ["oily"]}
    routine_rationale = {"pipeline_steps": [], "product_rationales": {}}

    publish_routine_completed(
        channel,
        event=event,
        request_id="req-3",
        image_analysis=image_analysis,
        routine_rationale=routine_rationale,
    )

    payload = json.loads(channel.basic_publish.call_args.kwargs["body"])
    assert payload["image_analysis"] == image_analysis
    assert payload["routine_rationale"] == routine_rationale


def test_publish_routine_completed_omits_none_analysis_fields() -> None:
    """When analysis fields are None, they should not appear in the body."""
    channel = MagicMock()
    event = RoutineCompletedEvent(matched_products=[], explanations={})

    publish_routine_completed(channel, event=event, request_id="req-4")

    payload = json.loads(channel.basic_publish.call_args.kwargs["body"])
    assert "image_analysis" not in payload
    assert "routine_rationale" not in payload


def test_matched_payload_missing_event_is_rejected() -> None:
    payload = {"request_id": "req-2"}
    with pytest.raises(ValueError, match="Missing 'event'"):
        _handle_matched_payload(payload=payload, correlation_id="req-2")


def test_matched_payload_missing_request_id_is_rejected() -> None:
    payload = {
        "event": {
            "matched_products": [],
            "profile": {"request_id": "inner", "skin_conditions": []},
            "constraints": {
                "request_id": "inner",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
        }
    }
    with pytest.raises(ValueError, match="Missing request_id"):
        _handle_matched_payload(payload=payload, correlation_id=None)


def test_matched_payload_uses_correlation_id_fallback() -> None:
    payload = {
        "event": {
            "matched_products": [],
            "profile": {"request_id": "inner", "skin_conditions": []},
            "constraints": {
                "request_id": "inner",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
        }
    }
    result = _handle_matched_payload(
        payload=payload,
        correlation_id="req-from-correlation",
    )
    assert result.request_id == "req-from-correlation"


def test_matched_payload_passes_through_analysis_fields() -> None:
    """image_analysis and routine_rationale from the incoming envelope
    should be preserved in the handler result."""
    image_analysis = {"visual_signals": ["acne"], "skin_conditions": ["oily", "acne"]}
    routine_rationale = {"pipeline_steps": [{"step": "safety_filter"}]}
    payload = {
        "request_id": "req-passthrough",
        "event": {
            "matched_products": [],
            "profile": {"request_id": "req-passthrough", "skin_conditions": ["oily"]},
            "constraints": {
                "request_id": "req-passthrough",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
        },
        "image_analysis": image_analysis,
        "routine_rationale": routine_rationale,
    }
    result = _handle_matched_payload(payload=payload, correlation_id=None)
    assert result.image_analysis == image_analysis
    assert result.routine_rationale == routine_rationale


def test_generate_explanations_forwards_product_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2: per-product catalog fields and matcher rationale reach explain API."""
    captured: list[dict[str, Any]] = []

    def fake_explain(*args: Any, **kwargs: Any) -> dict[str, str]:
        captured.append(kwargs)
        return {"explanation_text": "One. Two.", "sources": ["generic"]}

    monkeypatch.setattr(consumer_mod, "_explain_product", fake_explain)

    product = Product(
        id="p-spf",
        name="Sun Spray",
        category=ProductCategory.SPF,
        ingredients=["homosalate"],
        description="Lightweight spray.",
        skin_types=["oily"],
        concerns=["uv"],
        benefits=["protection"],
        contains_fragrance=False,
        contains_alcohol=False,
    )
    routine_rationale = {
        "product_rationales": {
            "p-spf": {"role": "essential", "fit_score": 0.71},
        }
    }
    image_analysis = {"visual_signals": ["shine"]}

    out = consumer_mod._generate_explanations(
        [product],
        skin_conditions=["oily"],
        request_id="req-stage2",
        image_analysis=image_analysis,
        routine_rationale=routine_rationale,
    )

    assert out == {"p-spf": "One. Two."}
    assert len(captured) == 1
    kw = captured[0]
    assert kw["product_category"] == "SPF"
    assert kw["product_description"] == "Lightweight spray."
    assert kw["product_skin_types"] == ["oily"]
    assert kw["product_concerns"] == ["uv"]
    assert kw["product_benefits"] == ["protection"]
    assert kw["contains_fragrance"] is False
    assert kw["contains_alcohol"] is False
    assert kw["product_rationale"] == {"role": "essential", "fit_score": 0.71}
    assert kw["image_analysis"] == image_analysis


def test_matched_payload_missing_analysis_fields_returns_none() -> None:
    """When envelope has no analysis fields, result should have None."""
    payload = {
        "request_id": "req-no-analysis",
        "event": {
            "matched_products": [],
            "profile": {"request_id": "req-no-analysis", "skin_conditions": []},
            "constraints": {
                "request_id": "req-no-analysis",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
        },
    }
    result = _handle_matched_payload(payload=payload, correlation_id=None)
    assert result.image_analysis is None
    assert result.routine_rationale is None
