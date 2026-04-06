"""In-memory request lifecycle for Azure + local-bridge demo flow.

Swappable later for Redis or DB; thread-safe for FastAPI background tasks.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

Status = Literal["queued", "processing", "completed", "failed"]


class BridgeRequestStore:
    """Tracks bridge-submitted jobs until callback or failure."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}

    def create(self, request_id: str, status: Status = "queued") -> None:
        with self._lock:
            self._data[request_id] = {
                "status": status,
                "result": None,
                "error": None,
            }

    def set_status(self, request_id: str, status: Status) -> bool:
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return False
            row["status"] = status
            return True

    def complete(self, request_id: str, result: dict[str, Any]) -> bool:
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return False
            row["status"] = "completed"
            row["result"] = result
            row["error"] = None
            return True

    def fail(self, request_id: str, error: str) -> bool:
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return False
            row["status"] = "failed"
            row["error"] = error
            row["result"] = None
            return True

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return None
            return {
                "status": row["status"],
                "result": row["result"],
                "error": row["error"],
            }

    def clear_for_tests(self) -> None:
        """Remove all rows (unit tests only)."""
        with self._lock:
            self._data.clear()


# Process-wide singleton for demo
bridge_request_store = BridgeRequestStore()
