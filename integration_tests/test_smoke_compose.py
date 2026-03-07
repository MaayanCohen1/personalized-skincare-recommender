"""Optional docker-compose smoke test for the async vertical slice.

This test is skipped by default. Run with:
    RUN_SMOKE_E2E=1 pytest integration_tests/test_smoke_compose.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_SMOKE_E2E") != "1",
    reason="Set RUN_SMOKE_E2E=1 to run docker-compose smoke test",
)
def test_post_then_poll_result_completed() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compose_env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ):
        compose_env.pop(key, None)

    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=repo_root,
        env=compose_env,
        check=True,
    )

    try:
        post_body = json.dumps(
            {
                "sensitivities": ["fragrance"],
                "max_products": 3,
                "catalog_ref": "default_catalog",
            }
        ).encode("utf-8")
        post_req = urllib.request.Request(
            "http://localhost:8000/recommend",
            data=post_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(post_req, timeout=10) as resp:
            post_response = json.loads(resp.read().decode("utf-8"))

        request_id = post_response["request_id"]

        deadline = time.time() + 45
        last_payload: dict[str, str] | dict[str, object] = {}
        while time.time() < deadline:
            with urllib.request.urlopen(
                f"http://localhost:8000/result/{request_id}", timeout=10
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            last_payload = payload

            if payload.get("status") == "completed":
                assert payload["request_id"] == request_id
                assert "result" in payload
                return
            time.sleep(2)

        raise AssertionError(f"Result did not complete in time: {last_payload}")
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=repo_root, check=False)
