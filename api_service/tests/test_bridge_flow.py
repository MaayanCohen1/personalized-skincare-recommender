"""Tests for Azure + local-bridge demo endpoints and request store."""

from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api_service.request_store import BridgeRequestStore, bridge_request_store


@pytest.fixture(autouse=True)
def _clean_bridge_store() -> None:
    bridge_request_store.clear_for_tests()
    yield
    bridge_request_store.clear_for_tests()


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_BRIDGE_URL", "http://bridge.test")
    monkeypatch.setenv("RESULT_CALLBACK_URL", "https://azure.example/internal/result-callback")
    monkeypatch.setenv("CALLBACK_TOKEN", "test-callback-secret")

    fake_thread = MagicMock(start=MagicMock())

    with patch("api_service.main.threading.Thread", return_value=fake_thread):
        with patch("api_service.main.redis.Redis.from_url", return_value=MagicMock()):
            from api_service.main import app

            with TestClient(app) as client:
                yield client


def test_bridge_request_store_lifecycle() -> None:
    store = BridgeRequestStore()
    store.create("a1", "queued")
    assert store.get("a1")["status"] == "queued"
    assert store.set_status("a1", "processing") is True
    assert store.complete("a1", {"request_id": "a1", "event": {}}) is True
    row = store.get("a1")
    assert row["status"] == "completed"
    assert row["result"]["request_id"] == "a1"
    assert store.fail("missing", "x") is False


def test_bridge_request_store_fail() -> None:
    bridge_request_store.create("f1", "processing")
    assert bridge_request_store.fail("f1", "boom") is True
    row = bridge_request_store.get("f1")
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_submit_returns_request_id_and_forwards_to_bridge(api_client: TestClient) -> None:
    with patch("api_service.main.post_to_local_bridge") as mock_forward:
        mock_forward.return_value = {"status": "accepted", "request_id": "ignored"}
        files = {"image": ("face.jpg", io.BytesIO(b"fake-bytes"), "image/jpeg")}
        data = {
            "skin_type": "oily",
            "has_breakouts": "true",
            "sensitivities": '["fragrance"]',
        }
        r = api_client.post("/submit", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert "request_id" in body
    assert body["status"] == "queued"
    rid = body["request_id"]
    mock_forward.assert_called_once()
    kwargs = mock_forward.call_args.kwargs
    assert kwargs["request_id"] == rid
    assert kwargs["skin_type"] == "oily"
    assert kwargs["has_breakouts"] is True
    assert kwargs["sensitivities_json"] == '["fragrance"]'
    assert kwargs["callback_url"] == "https://azure.example/internal/result-callback"
    assert kwargs["callback_token"] == "test-callback-secret"

    res = api_client.get(f"/result/{rid}")
    assert res.status_code == 200
    assert res.json()["status"] in ("queued", "processing")


def test_result_callback_requires_token(api_client: TestClient) -> None:
    rid = uuid.uuid4().hex
    bridge_request_store.create(rid, "processing")
    r = api_client.post(
        "/internal/result-callback",
        json={"request_id": rid, "status": "completed", "result": {"request_id": rid, "event": {}}},
    )
    assert r.status_code == 401


def test_result_callback_updates_completed(api_client: TestClient) -> None:
    rid = uuid.uuid4().hex
    bridge_request_store.create(rid, "processing")
    payload = {"request_id": rid, "event": {"matched_products": []}}
    r = api_client.post(
        "/internal/result-callback",
        headers={"X-Callback-Token": "test-callback-secret"},
        json={"request_id": rid, "status": "completed", "result": payload},
    )
    assert r.status_code == 200
    poll = api_client.get(f"/result/{rid}").json()
    assert poll["status"] == "completed"
    assert poll["result"] == payload


def test_result_callback_failed(api_client: TestClient) -> None:
    rid = uuid.uuid4().hex
    bridge_request_store.create(rid, "processing")
    r = api_client.post(
        "/internal/result-callback",
        headers={"X-Callback-Token": "test-callback-secret"},
        json={"request_id": rid, "status": "failed", "error": "timeout"},
    )
    assert r.status_code == 200
    poll = api_client.get(f"/result/{rid}").json()
    assert poll["status"] == "failed"
    assert poll["error"] == "timeout"
