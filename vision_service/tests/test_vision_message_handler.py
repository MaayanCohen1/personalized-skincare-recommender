"""Focused tests for vision_service.message_handler."""

from __future__ import annotations

import json
from typing import Any

import pytest

from vision_service.message_handler import handle_image_uploaded_message


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "request_id": "req-img-001",
        "image_path": "/images/face.jpg",
        "user_preferences": {
            "skin_type": "oily",
            "has_breakouts": True,
            "sensitivities": ["fragrance"],
            "is_cruelty_free_required": False,
        },
    }


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------


def test_valid_payload_returns_expected_dictionary_shape(
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, Any],
) -> None:
    """Valid payload should return dict with all expected keys."""
    monkeypatch.setattr(
        "vision_service.message_handler.analyze_skin_image",
        lambda path: {"visual_signals": ["oily", "acne"]},
    )

    body = json.dumps(valid_payload).encode("utf-8")
    result = handle_image_uploaded_message(body)

    assert set(result.keys()) == {
        "request_id",
        "image_path",
        "user_preferences",
        "visual_signals",
    }
    assert result["request_id"] == "req-img-001"
    assert result["image_path"] == "/images/face.jpg"
    assert result["visual_signals"] == ["oily", "acne"]
    assert result["user_preferences"] == valid_payload["user_preferences"]


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


def test_invalid_json_raises() -> None:
    """Non-JSON bytes should raise JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        handle_image_uploaded_message(b"not json {")


def test_missing_request_id_raises() -> None:
    """Payload without request_id should raise ValueError."""
    payload = {"image_path": "/images/face.jpg"}
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValueError, match="Missing required field: request_id"):
        handle_image_uploaded_message(body)


def test_missing_image_path_raises() -> None:
    """Payload without image_path should raise ValueError."""
    payload = {"request_id": "req-no-path"}
    body = json.dumps(payload).encode("utf-8")

    with pytest.raises(ValueError, match="Missing required field: image_path"):
        handle_image_uploaded_message(body)


# ------------------------------------------------------------------
# Passthrough & delegation
# ------------------------------------------------------------------


def test_preserves_original_user_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """user_preferences from the payload should be passed through unchanged."""
    monkeypatch.setattr(
        "vision_service.message_handler.analyze_skin_image",
        lambda path: {"visual_signals": []},
    )

    prefs = {
        "skin_type": "dry",
        "has_breakouts": False,
        "sensitivities": ["alcohol"],
        "is_cruelty_free_required": True,
        "additional_notes": "Prefer lightweight textures.",
    }
    payload = {
        "request_id": "req-prefs-test",
        "image_path": "/images/face.jpg",
        "user_preferences": prefs,
    }
    body = json.dumps(payload).encode("utf-8")
    result = handle_image_uploaded_message(body)

    assert result["user_preferences"] == prefs


def test_user_preferences_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When user_preferences is not in the payload, result should contain None."""
    monkeypatch.setattr(
        "vision_service.message_handler.analyze_skin_image",
        lambda path: {"visual_signals": ["combination"]},
    )

    payload = {"request_id": "req-no-prefs", "image_path": "/images/face.jpg"}
    body = json.dumps(payload).encode("utf-8")
    result = handle_image_uploaded_message(body)

    assert result["user_preferences"] is None


def test_passes_image_path_correctly_into_analyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """image_path from the payload should be forwarded to analyze_skin_image."""
    captured: dict[str, str] = {}

    def _spy(path: str) -> dict[str, list[str]]:
        captured["path"] = path
        return {"visual_signals": ["dry"]}

    monkeypatch.setattr(
        "vision_service.message_handler.analyze_skin_image",
        _spy,
    )

    payload = {
        "request_id": "req-path-spy",
        "image_path": "/data/uploads/selfie.png",
    }
    body = json.dumps(payload).encode("utf-8")
    handle_image_uploaded_message(body)

    assert captured["path"] == "/data/uploads/selfie.png"
