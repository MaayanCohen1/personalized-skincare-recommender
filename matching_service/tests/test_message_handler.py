"""Focused tests for matching_service.core.message_handler."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from matching_service.core.message_handler import handle_signals_detected_message
from matching_service.core.models import UserPreferences
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
        ),
        Product(
            id="moisturizer-1",
            name="Moisturizer",
            category=ProductCategory.MOISTURIZER,
            ingredients=["glycerin"],
            description="hydrating moisturizer",
        ),
    ]


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "request_id": "req-123",
        "visual_signals": ["oily", "acne"],
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": True,
            "sensitivities": ["fragrance"],
            "is_cruelty_free_required": False,
        },
    }


def test_valid_payload_returns_expected_dictionary_shape(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """Valid payload should return dict with request_id, visual_signals, matched_products."""
    
    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert "request_id" in result
    assert "visual_signals" in result
    assert "matched_products" in result
    assert result["request_id"] == "req-123"
    assert result["visual_signals"] == ["oily", "acne"]
    assert len(result["matched_products"]) == 1
    # Verify matched_products contains plain dicts, not Product objects
    assert isinstance(result["matched_products"][0], dict)
    assert result["matched_products"][0]["id"] == "cleanser-1"


def test_invalid_json_raises(catalog: list[Product]) -> None:
    """Non-JSON bytes should raise JSONDecodeError."""
    invalid_body = b"not valid json {"
    
    with pytest.raises(json.JSONDecodeError):
        handle_signals_detected_message(invalid_body, catalog)


def test_invalid_user_preferences_payload_raises(catalog: list[Product]) -> None:
    """Invalid user_preferences structure should raise ValidationError."""
    payload = {
        "request_id": "req-456",
        "visual_signals": [],
        "user_preferences": {
            "skin_type": "invalid_type",
            "has_breakouts": True,
            "sensitivities": [],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValidationError):
        handle_signals_detected_message(body, catalog)


def test_missing_user_preferences_raises(catalog: list[Product]) -> None:
    """Payload without user_preferences should raise ValueError."""
    payload = {
        "request_id": "req-789",
        "visual_signals": ["oily"],
    }
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValueError, match="Missing required field: user_preferences"):
        handle_signals_detected_message(body, catalog)


def test_missing_request_id_raises(catalog: list[Product]) -> None:
    """Payload without request_id should raise ValueError."""
    payload = {
        "visual_signals": ["oily"],
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": False,
            "sensitivities": [],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValueError, match="Missing required field: request_id"):
        handle_signals_detected_message(body, catalog)


def test_original_request_id_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """The request_id from the payload should be preserved in the response."""
    
    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "unique-req-id-999",
        "user_preferences": {
            "skin_type": "oily",
            "has_breakouts": False,
            "sensitivities": [],
            "is_cruelty_free_required": True,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert result["request_id"] == "unique-req-id-999"


def test_visual_signals_are_passed_correctly_into_match_for_user(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """visual_signals from payload should be passed to match_for_user."""
    captured_args: dict[str, Any] = {}

    def _spy_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        captured_args["catalog"] = catalog
        captured_args["user_preferences"] = user_preferences
        captured_args["visual_signals"] = visual_signals
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _spy_match_for_user,
    )

    payload = {
        "request_id": "req-signals-test",
        "visual_signals": ["combination", "acne"],
        "user_preferences": {
            "skin_type": "not_sure",
            "has_breakouts": False,
            "sensitivities": ["alcohol"],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    handle_signals_detected_message(body, catalog)

    assert captured_args["visual_signals"] == ["combination", "acne"]
    assert captured_args["user_preferences"].skin_type.value == "not_sure"
    assert captured_args["catalog"] == catalog


def test_visual_signals_none_when_not_in_payload(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """When visual_signals is absent from payload, None should be passed to match_for_user."""
    captured_args: dict[str, Any] = {}

    def _spy_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        captured_args["visual_signals"] = visual_signals
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _spy_match_for_user,
    )

    payload = {
        "request_id": "req-no-signals",
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": True,
            "sensitivities": [],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    handle_signals_detected_message(body, catalog)

    assert captured_args["visual_signals"] is None


def test_matched_products_are_plain_dictionaries(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """matched_products should contain plain dicts, not Product objects."""
    
    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        return catalog

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-serialization-test",
        "user_preferences": {
            "skin_type": "combination",
            "has_breakouts": False,
            "sensitivities": [],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    # Verify matched_products is a list of dicts
    assert isinstance(result["matched_products"], list)
    assert len(result["matched_products"]) == 2
    
    for product_dict in result["matched_products"]:
        assert isinstance(product_dict, dict)
        assert not isinstance(product_dict, Product)
        assert "id" in product_dict
        assert "name" in product_dict
        assert "category" in product_dict
        assert "ingredients" in product_dict
        assert "description" in product_dict


def test_result_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """The entire result dictionary should be JSON-serializable."""
    
    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-json-test",
        "visual_signals": ["oily"],
        "user_preferences": {
            "skin_type": "oily",
            "has_breakouts": True,
            "sensitivities": ["fragrance"],
            "is_cruelty_free_required": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    # Should not raise any exception
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    
    # Round-trip verification
    parsed = json.loads(json_str)
    assert parsed["request_id"] == "req-json-test"
    assert parsed["visual_signals"] == ["oily"]
    assert len(parsed["matched_products"]) == 1
    assert parsed["matched_products"][0]["id"] == "cleanser-1"
