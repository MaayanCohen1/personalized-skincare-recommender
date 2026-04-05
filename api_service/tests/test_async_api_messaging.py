"""Unit tests for API async messaging and Redis result store."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from api_service.main import (
    EXCHANGE_ROUTINE_EVENTS,
    QUEUE_COMPLETED,
    QUEUE_IMAGE_UPLOADED,
    RESULT_TTL_SECONDS,
    ROUTING_KEY_COMPLETED,
    extract_request_id,
    publish_image_uploaded,
    redis_result_key,
    store_completed_result,
)
from matching_service.core.models import Sensitivity, SkinType, UserPreferences


def _sample_prefs() -> UserPreferences:
    return UserPreferences(
        skin_type=SkinType.OILY,
        has_breakouts=True,
        sensitivities=[Sensitivity.FRAGRANCE],
    )


# ------------------------------------------------------------------
# Redis helpers
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# publish_image_uploaded
# ------------------------------------------------------------------

def test_publish_image_uploaded_sends_correct_payload() -> None:
    channel = MagicMock()
    request_id = "req-xyz"
    prefs = _sample_prefs()

    publish_image_uploaded(
        channel,
        request_id=request_id,
        image_path="/uploads/face.jpg",
        user_preferences=prefs,
    )

    channel.basic_publish.assert_called_once()
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == ""
    assert kwargs["routing_key"] == QUEUE_IMAGE_UPLOADED
    assert kwargs["properties"].delivery_mode == 2

    payload = json.loads(kwargs["body"])
    assert payload["request_id"] == request_id
    assert payload["image_path"] == "/uploads/face.jpg"
    assert payload["user_preferences"]["skin_type"] == "oily"
    assert payload["user_preferences"]["has_breakouts"] is True
    assert payload["user_preferences"]["sensitivities"] == ["fragrance"]
    assert "visual_signals" not in payload


def test_publish_image_uploaded_body_is_bytes() -> None:
    channel = MagicMock()
    publish_image_uploaded(
        channel,
        request_id="req-bytes",
        image_path="/uploads/face.jpg",
        user_preferences=_sample_prefs(),
    )
    body = channel.basic_publish.call_args.kwargs["body"]
    assert isinstance(body, bytes)


# ------------------------------------------------------------------
# extract_request_id
# ------------------------------------------------------------------


def test_extract_request_id_returns_id() -> None:
    payload = {"request_id": "req-abc", "matched_products": []}
    assert extract_request_id(payload) == "req-abc"


def test_extract_request_id_rejects_missing_id() -> None:
    with pytest.raises(ValueError, match="Missing request_id"):
        extract_request_id({"matched_products": []})


def test_extract_request_id_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Missing request_id"):
        extract_request_id({"request_id": ""})


# ------------------------------------------------------------------
# routing constants
# ------------------------------------------------------------------


def test_completed_queue_constants() -> None:
    """Verify the API service binds to the correct exchange/queue/routing_key."""
    assert EXCHANGE_ROUTINE_EVENTS == "routine.events"
    assert ROUTING_KEY_COMPLETED == "routine.completed"
    assert QUEUE_COMPLETED == "api.routine.completed.q"
