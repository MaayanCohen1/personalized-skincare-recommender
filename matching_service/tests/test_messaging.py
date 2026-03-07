"""Unit tests for matching service messaging contracts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from matching_service.matching import _handle_requested_payload, publish_routine_matched
from shared.models import Product, ProductCategory, RoutineMatchedEvent, SkinProfile, UserConstraints


def test_publish_routine_matched_routing_and_correlation() -> None:
    channel = MagicMock()
    request_id = "req-1"
    event = RoutineMatchedEvent(
        matched_products=[
            Product(
                id="p1",
                name="Test Product",
                category=ProductCategory.MOISTURIZER,
                ingredients=["glycerin"],
                description="desc",
            )
        ],
        profile=SkinProfile(request_id=request_id, skin_conditions=[]),
        constraints=UserConstraints(request_id=request_id, sensitivities=[], max_products=5),
    )

    publish_routine_matched(channel, event=event, request_id=request_id)

    channel.basic_publish.assert_called_once()
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == "routine.events"
    assert kwargs["routing_key"] == "routine.matched"
    assert kwargs["properties"].correlation_id == request_id

    payload = json.loads(kwargs["body"])
    assert payload["request_id"] == request_id
    assert "event" in payload
    assert payload["event"]["constraints"]["request_id"] == request_id


def test_requested_payload_missing_event_is_rejected() -> None:
    payload = {"request_id": "req-1"}
    with pytest.raises(ValueError, match="Missing 'event'"):
        _handle_requested_payload(payload=payload, correlation_id="req-1")


def test_requested_payload_missing_request_id_is_rejected() -> None:
    payload = {
        "event": {
            "constraints": {
                "request_id": "inside-event-only",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
            "catalog_ref": "default",
        }
    }
    with pytest.raises(ValueError, match="Missing request_id"):
        _handle_requested_payload(payload=payload, correlation_id=None)


def test_requested_payload_uses_correlation_id_fallback() -> None:
    request_id = "req-from-correlation"
    payload = {
        "event": {
            "constraints": {
                "request_id": "inside-event-only",
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
            "catalog_ref": "default",
        }
    }
    _, derived_request_id = _handle_requested_payload(
        payload=payload,
        correlation_id=request_id,
    )
    assert derived_request_id == request_id
