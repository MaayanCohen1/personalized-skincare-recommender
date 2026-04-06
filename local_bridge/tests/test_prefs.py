"""Tests for local_bridge preference parsing."""

from __future__ import annotations

import pytest

from local_bridge.prefs import parse_demo_preferences, parse_has_breakouts


def test_parse_has_breakouts() -> None:
    assert parse_has_breakouts("true") is True
    assert parse_has_breakouts("FALSE") is False
    assert parse_has_breakouts("1") is True


def test_parse_demo_preferences() -> None:
    p = parse_demo_preferences("dry", "no", '["none"]')
    assert p.skin_type.value == "dry"
    assert p.has_breakouts is False
    assert [s.value for s in p.sensitivities] == ["none"]


def test_parse_demo_preferences_invalid_skin() -> None:
    with pytest.raises(ValueError):
        parse_demo_preferences("invalid", "false", '["none"]')
