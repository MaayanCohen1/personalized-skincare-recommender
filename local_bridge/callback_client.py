"""POST result back to Azure ``api_service`` (demo)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def post_result_callback(
    callback_url: str,
    callback_token: str,
    *,
    request_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    timeout_seconds: float = 60.0,
) -> None:
    body: dict[str, Any] = {"request_id": request_id, "status": status}
    if status == "completed" and result is not None:
        body["result"] = result
    if status == "failed":
        body["error"] = error or "Unknown error"

    headers = {"X-Callback-Token": callback_token}
    logger.info("Posting callback request_id=%s status=%s url=%s", request_id, status, callback_url)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(callback_url, json=body, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "Callback HTTP error request_id=%s status_code=%s body=%r",
            request_id,
            response.status_code,
            response.text[:500],
        )
        raise
    logger.info("Callback POST succeeded request_id=%s", request_id)
