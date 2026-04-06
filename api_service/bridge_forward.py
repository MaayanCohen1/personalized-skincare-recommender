"""HTTP forward from API service to local bridge (demo)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def post_to_local_bridge(
    bridge_base_url: str,
    *,
    file_content: bytes,
    filename: str,
    request_id: str,
    skin_type: str,
    has_breakouts: bool,
    sensitivities_json: str,
    callback_url: str,
    callback_token: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """POST multipart to ``{bridge_base_url}/demo-submit``. Returns bridge JSON response."""
    import httpx

    url = f"{bridge_base_url.rstrip('/')}/demo-submit"
    files = {"image": (filename or "upload.jpg", file_content, "application/octet-stream")}
    data = {
        "request_id": request_id,
        "skin_type": skin_type,
        "has_breakouts": "true" if has_breakouts else "false",
        "sensitivities": sensitivities_json,
        "callback_url": callback_url,
        "callback_token": callback_token,
    }
    logger.info(
        "Forwarding request_id=%s to local bridge url=%s filename=%r",
        request_id,
        url,
        filename,
    )
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(url, files=files, data=data)
    response.raise_for_status()
    body = response.json()
    logger.info(
        "Bridge accepted request_id=%s bridge_status=%r",
        request_id,
        body.get("status"),
    )
    return body
