"""Focused tests for matching_service.worker message orchestration."""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import Mock

import pytest

# Provide a lightweight pika stub before importing the worker module.
_pika_stub = types.ModuleType("pika")
_pika_stub.BasicProperties = lambda **kwargs: kwargs
_pika_stub.BlockingConnection = object
_pika_stub.URLParameters = object
_pika_stub.adapters = types.SimpleNamespace(
    blocking_connection=types.SimpleNamespace(BlockingChannel=object)
)
_pika_stub.exceptions = types.SimpleNamespace(AMQPConnectionError=Exception)
_pika_stub.spec = types.SimpleNamespace(Basic=types.SimpleNamespace(Deliver=object))
sys.modules.setdefault("pika", _pika_stub)
sys.modules.setdefault("pika.adapters", _pika_stub.adapters)
sys.modules.setdefault("pika.adapters.blocking_connection", _pika_stub.adapters.blocking_connection)
sys.modules.setdefault("pika.exceptions", _pika_stub.exceptions)
sys.modules.setdefault("pika.spec", _pika_stub.spec)

from matching_service.worker import _on_message
from shared.models import Product, ProductCategory


@pytest.fixture
def catalog() -> list[Product]:
    return [
        Product(
            id="cleanser-1",
            name="Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["water"],
            description="gentle cleanser",
        )
    ]


def _make_mock_envelope(request_id: str = "req-123") -> dict[str, Any]:
    """Return a handler-shaped envelope with RoutineMatchedEvent fields."""
    return {
        "request_id": request_id,
        "event": {
            "matched_products": [],
            "profile": {
                "request_id": request_id,
                "skin_conditions": ["oily"],
            },
            "constraints": {
                "request_id": request_id,
                "sensitivities": [],
                "max_products": 5,
                "image_path": None,
            },
        },
    }


def test_on_message_passes_scorer_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """Worker callback should pass scorer to handler and publish the envelope."""
    captured: dict[str, object] = {}

    def _mock_handle(body: bytes, cat: list[Product], scorer: Any = None) -> dict[str, Any]:
        captured["body"] = body
        captured["catalog"] = cat
        captured["scorer"] = scorer
        return _make_mock_envelope()

    monkeypatch.setattr("matching_service.worker.handle_signals_detected_message", _mock_handle)

    channel = Mock()
    method = Mock()
    method.delivery_tag = 10
    properties = Mock()
    body = b'{"request_id":"req-123"}'
    scorer = lambda skin_conditions, products: {p.id: 0.5 for p in products}

    _on_message(channel, method, properties, body, catalog, scorer)

    assert captured["body"] == body
    assert captured["catalog"] == catalog
    assert captured["scorer"] is scorer

    channel.basic_publish.assert_called_once()
    publish_kwargs = channel.basic_publish.call_args.kwargs
    assert publish_kwargs["exchange"] == "routine.events"
    assert publish_kwargs["routing_key"] == "routine.matched"

    payload = json.loads(publish_kwargs["body"].decode("utf-8"))
    assert payload["request_id"] == "req-123"
    assert "event" in payload
    assert payload["event"]["matched_products"] == []
    assert payload["event"]["profile"]["request_id"] == "req-123"
    assert payload["event"]["profile"]["skin_conditions"] == ["oily"]
    assert payload["event"]["constraints"]["request_id"] == "req-123"
    assert payload["event"]["constraints"]["sensitivities"] == []

    channel.basic_ack.assert_called_once_with(delivery_tag=10)
    channel.basic_nack.assert_not_called()


def test_on_message_nacks_on_handler_error(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """When the handler raises, the message should be nacked without requeue."""

    def _failing_handle(body: bytes, cat: list[Product], scorer: Any = None) -> dict[str, Any]:
        raise ValueError("bad payload")

    monkeypatch.setattr("matching_service.worker.handle_signals_detected_message", _failing_handle)

    channel = Mock()
    method = Mock()
    method.delivery_tag = 42
    properties = Mock()

    _on_message(channel, method, properties, b'{}', catalog)

    channel.basic_nack.assert_called_once_with(delivery_tag=42, requeue=False)
    channel.basic_ack.assert_not_called()
    channel.basic_publish.assert_not_called()
