"""Unit tests for API async messaging and Redis result store."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from api_service.main import (
    RESULT_TTL_SECONDS,
    extract_completed_envelope,
    publish_routine_requested,
    redis_result_key,
    store_completed_result,
)
from shared.models import RoutineRequestedEvent, UserConstraints


def test_redis_key_format_and_ttl_usage() -> None:
    redis_client = MagicMock()
    request_id = "req-123"
    result = {"status": "completed"}

    store_completed_result(redis_client, request_id=request_id, result=result)

    redis_client.setex.assert_called_once()
    called_key, called_ttl, called_payload = redis_client.setex.call_args.args
    assert called_key == "result:req-123"
    assert called_ttl == RESULT_TTL_SECONDS
    assert json.loads(called_payload) == result


def test_redis_result_key_helper() -> None:
    assert redis_result_key("abc") == "result:abc"


def test_publish_routine_requested_routing_and_correlation() -> None:
    channel = MagicMock()
    request_id = "req-xyz"
    event = RoutineRequestedEvent(
        constraints=UserConstraints(
            request_id=request_id,
            sensitivities=["fragrance"],
            max_products=3,
            image_path=None,
        ),
        catalog_ref="default_catalog",
    )

    publish_routine_requested(channel, event=event, request_id=request_id)

    channel.basic_publish.assert_called_once()
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == "routine.events"
    assert kwargs["routing_key"] == "routine.requested"
    assert kwargs["properties"].correlation_id == request_id

    payload = json.loads(kwargs["body"])
    assert payload["request_id"] == request_id
    assert payload["event"]["constraints"]["request_id"] == request_id
    assert payload["event"]["catalog_ref"] == "default_catalog"


def test_extract_completed_envelope_rejects_missing_event() -> None:
    with pytest.raises(ValueError, match="Missing 'event'"):
        extract_completed_envelope(payload={"request_id": "req-1"}, correlation_id=None)


def test_extract_completed_envelope_rejects_missing_request_id() -> None:
    with pytest.raises(ValueError, match="Missing request_id"):
        extract_completed_envelope(payload={"event": {"matched_products": [], "explanations": {}}}, correlation_id=None)


def test_extract_completed_envelope_uses_correlation_id_fallback() -> None:
    request_id, event = extract_completed_envelope(
        payload={"event": {"matched_products": [], "explanations": {}}},
        correlation_id="req-from-correlation",
    )
    assert request_id == "req-from-correlation"
    assert event.explanations == {}
