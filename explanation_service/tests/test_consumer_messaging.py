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
    _, request_id = _handle_matched_payload(
        payload=payload,
        correlation_id="req-from-correlation",
    )
    assert request_id == "req-from-correlation"
