"""Unit tests for explanation consumer publish contract."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from explanation_service.consumer import _handle_matched_payload, publish_routine_completed
from shared.models import Product, ProductCategory, RoutineCompletedEvent


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
