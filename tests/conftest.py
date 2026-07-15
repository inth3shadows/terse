"""Shared test guards."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_xdg_state(monkeypatch, tmp_path_factory):
    """Point $XDG_STATE_HOME at a per-session temp dir so no test — present or future —
    can write the proxy's default-on savings ledger (stats.py) into the real
    ~/.local/state. Tests that care about the path set their own value on top."""
    monkeypatch.setenv("XDG_STATE_HOME",
                       str(tmp_path_factory.getbasetemp() / "xdg-state"))
