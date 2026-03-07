"""Static safety checks for forbidden price usage in project code."""

from __future__ import annotations

from pathlib import Path
import re


def test_no_product_price_usage_in_codebase() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_dirs = [
        repo_root / "api_service",
        repo_root / "matching_service",
        repo_root / "explanation_service",
        repo_root / "shared",
    ]

    forbidden_patterns = [
        re.compile(r"Product\.price\b"),
        re.compile(r"\.price\b"),
    ]

    violations: list[str] = []
    for target_dir in target_dirs:
        for path in target_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append(f"{path}: {pattern.pattern}")

    assert not violations, "Forbidden price usage found:\n" + "\n".join(violations)
