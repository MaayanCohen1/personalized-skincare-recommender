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
        },
    }


def test_valid_payload_returns_expected_envelope_shape(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """Result should be a routine.matched envelope with request_id and event."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert "request_id" in result
    assert "event" in result
    assert "image_analysis" in result
    assert "routine_rationale" in result
    assert result["request_id"] == "req-123"

    event = result["event"]
    assert "matched_products" in event
    assert "profile" in event
    assert "constraints" in event
    assert len(event["matched_products"]) == 1
    assert event["matched_products"][0]["id"] == "cleanser-1"
    assert event["profile"]["request_id"] == "req-123"
    assert event["constraints"]["request_id"] == "req-123"


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
        },
    }
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValueError, match="Missing required field: request_id"):
        handle_signals_detected_message(body, catalog)


def test_original_request_id_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """The request_id from the payload should appear in both the envelope and the event."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
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
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert result["request_id"] == "unique-req-id-999"
    assert result["event"]["profile"]["request_id"] == "unique-req-id-999"
    assert result["event"]["constraints"]["request_id"] == "unique-req-id-999"


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
        scorer: Any = None,
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
        scorer: Any = None,
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
        },
    }
    body = json.dumps(payload).encode("utf-8")
    handle_signals_detected_message(body, catalog)

    assert captured_args["visual_signals"] is None


def test_matched_products_are_plain_dictionaries(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """matched_products inside the event should be plain dicts."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
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
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    products = result["event"]["matched_products"]
    assert isinstance(products, list)
    assert len(products) == 2

    for product_dict in products:
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
        scorer: Any = None,
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
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    json_str = json.dumps(result)
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert parsed["request_id"] == "req-json-test"
    assert len(parsed["event"]["matched_products"]) == 1
    assert parsed["event"]["matched_products"][0]["id"] == "cleanser-1"


def test_scorer_is_passed_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """Provided scorer should be forwarded to match_for_user."""
    captured: dict[str, Any] = {}

    def _spy_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        captured["scorer"] = scorer
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _spy_match_for_user,
    )

    payload = {
        "request_id": "req-scorer-test",
        "user_preferences": {
            "skin_type": "oily",
            "has_breakouts": True,
            "sensitivities": [],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    scorer = lambda skin_conditions, products: {p.id: 0.5 for p in products}
    handle_signals_detected_message(body, catalog, scorer=scorer)

    assert captured["scorer"] is scorer


def test_profile_contains_merged_skin_conditions(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """profile.skin_conditions should contain merged questionnaire + visual signals."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-profile-test",
        "visual_signals": ["oily"],
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": True,
            "sensitivities": [],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    skin_conditions = result["event"]["profile"]["skin_conditions"]
    assert skin_conditions == ["dry", "acne", "oily"]


def test_constraints_contain_sensitivities(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """constraints.sensitivities should reflect user preference sensitivities."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-constraints-test",
        "user_preferences": {
            "skin_type": "oily",
            "has_breakouts": False,
            "sensitivities": ["fragrance", "alcohol"],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert result["event"]["constraints"]["sensitivities"] == ["fragrance", "alcohol"]


def test_constraints_empty_when_none_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """When user selects 'none' sensitivity, constraints.sensitivities should be []."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-none-sensitivity",
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": False,
            "sensitivities": ["none"],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    assert result["event"]["constraints"]["sensitivities"] == []


def test_integration_hard_filter_then_fit_scoring() -> None:
    """Integration path: parse/validate -> hard filter -> fit scoring."""
    catalog = [
        Product(
            id="unsafe-fragrance",
            name="Unsafe",
            category=ProductCategory.MOISTURIZER,
            ingredients=["fragrance", "water"],
            description="unsafe product",
        ),
        Product(
            id="safe-cleanser",
            name="Safe Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["water"],
            description="cleanser",
        ),
        Product(
            id="safe-spf",
            name="Safe SPF",
            category=ProductCategory.SPF,
            ingredients=["zinc oxide"],
            description="spf",
        ),
    ]

    seen_by_scorer: dict[str, Any] = {}

    def scorer(skin_conditions: list[str], products: list[Product]) -> dict[str, float]:
        seen_by_scorer["skin_conditions"] = skin_conditions
        seen_by_scorer["product_ids"] = [p.id for p in products]
        return {p.id: 0.5 for p in products}

    payload = {
        "request_id": "req-integration",
        "visual_signals": ["oily"],
        "user_preferences": {
            "skin_type": "not_sure",
            "has_breakouts": False,
            "sensitivities": ["fragrance"],
        },
    }

    result = handle_signals_detected_message(
        body=json.dumps(payload).encode("utf-8"),
        catalog=catalog,
        scorer=scorer,
    )

    assert seen_by_scorer["product_ids"] == ["safe-cleanser", "safe-spf"]
    assert seen_by_scorer["skin_conditions"] == ["oily"]

    matched_ids = [p["id"] for p in result["event"]["matched_products"]]
    assert "unsafe-fragrance" not in matched_ids
    assert set(matched_ids) == {"safe-cleanser", "safe-spf"}

    assert result["event"]["constraints"]["sensitivities"] == ["fragrance"]
    assert result["event"]["profile"]["skin_conditions"] == ["oily"]


# ---------------------------------------------------------------------------
# image_analysis and routine_rationale fields
# ---------------------------------------------------------------------------


def test_image_analysis_contains_visual_signals(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """image_analysis should surface the visual_signals and skin_conditions."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    ia = result["image_analysis"]
    assert ia["visual_signals"] == ["oily", "acne"]
    assert "skin_conditions" in ia
    assert isinstance(ia["skin_conditions"], list)
    assert ia["source"] == "vision_service + questionnaire"


def test_image_analysis_questionnaire_only_when_no_visual_signals(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
) -> None:
    """When visual_signals are absent, source should be 'questionnaire'."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return []

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    payload = {
        "request_id": "req-no-vis",
        "user_preferences": {
            "skin_type": "dry",
            "has_breakouts": False,
            "sensitivities": [],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    ia = result["image_analysis"]
    assert ia["visual_signals"] == []
    assert ia["source"] == "questionnaire"


def test_routine_rationale_has_pipeline_steps(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """routine_rationale must include pipeline_steps and product_rationales."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    rr = result["routine_rationale"]
    assert "pipeline_steps" in rr
    assert "product_rationales" in rr
    step_names = [s["step"] for s in rr["pipeline_steps"]]
    assert "safety_filter" in step_names
    assert "routine_assembly" in step_names


def test_routine_rationale_reports_semantic_when_scorer_provided(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """When a scorer is provided, rationale should report semantic_ranking step."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return [catalog[0]]

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    dummy_scorer = lambda sc, prods: {p.id: 0.5 for p in prods}
    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog, scorer=dummy_scorer)

    step_names = [s["step"] for s in result["routine_rationale"]["pipeline_steps"]]
    assert "semantic_ranking" in step_names


def test_new_fields_are_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[Product],
    valid_payload: dict[str, Any],
) -> None:
    """image_analysis and routine_rationale must be JSON-serializable."""

    def _mock_match_for_user(
        catalog: list[Product],
        user_preferences: UserPreferences,
        visual_signals: list[str] | None = None,
        scorer: Any = None,
    ) -> list[Product]:
        return catalog

    monkeypatch.setattr(
        "matching_service.core.message_handler.match_for_user",
        _mock_match_for_user,
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_signals_detected_message(body, catalog)

    serialized = json.dumps(result)
    parsed = json.loads(serialized)
    assert "image_analysis" in parsed
    assert "routine_rationale" in parsed
