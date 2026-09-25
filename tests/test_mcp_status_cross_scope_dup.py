"""#447 — `mcp-status` names a live duplicate in ANOTHER scope, as `terse stats` does.

Before, the "runs TWICE" warning fired only for a same-scope `folded-and-live` row. #424's
layout — a project router folding `kb` beside a user-scope `kb` proxy — printed `folded` and
`wrapped` with no warning while `terse stats` reported the label contested (#396).
Driven through `main(["mcp-status"])`, with `scan_scopes` pinned to fixed rows.
"""
from __future__ import annotations

import pytest

from terse.cli import main


def _run(monkeypatch, capsys, rows):
    import terse.install_mcp as install_mcp
    monkeypatch.setattr(install_mcp, "scan_scopes", lambda *a, **k: rows)
    capsys.readouterr()
    assert main(["mcp-status"]) == 0
    return capsys.readouterr().out


def _row(scope, server, state, wraps=None, identity=None):
    r = {"scope": scope, "server": server, "state": state, "wraps": wraps, "policy": None,
         "config": f"/{scope}.json", "router": "terse" if state.startswith("folded") else None}
    if identity is not None:
        r["ledger_identity"], r["ledger_identity_explicit"] = identity, True
    return r


@pytest.mark.parametrize("absent", ["folded", "folded-unstashed"])
def test_a_cross_scope_live_duplicate_is_named_on_both_writers(monkeypatch, capsys, absent):
    out = _run(monkeypatch, capsys, [
        _row("user", "kb", "wrapped", "kb-server --stdio", identity="kb"),
        _row("project", "terse", "router", "gh, kb"),
        _row("project", "kb", absent)])
    lines = out.splitlines()
    kb_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("kb ")
                and "wrapped" in ln)
    terse_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("terse "))
    assert any("`kb` is ALSO run by terse" in ln and "TWICE" in ln
               for ln in lines[kb_i:kb_i + 4])
    assert any("`kb` is ALSO run by kb" in ln for ln in lines[terse_i:terse_i + 4])


def test_a_plain_folded_peer_behind_its_router_is_not_a_duplicate(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [_row("user", "terse", "router", "gh, kb"),
                                     _row("user", "kb", "folded")])
    assert "TWICE" not in out


def test_the_same_scope_folded_and_live_row_keeps_its_own_line_only(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [
        _row("user", "terse", "router", "gh, kb"),
        _row("user", "kb", "folded-and-live", "kb-server --stdio", identity="kb")])
    assert out.count("ALSO live as its own entry") == 1
    assert "`kb` is ALSO run by terse" not in out          # not said twice on that row
    assert "`kb` is ALSO run by kb" in out                 # the router side is named


def test_the_warning_sits_on_the_row_that_runs_not_the_first_listed(monkeypatch, capsys):
    """Scan order is user -> project. Here the user `kb` is folded (absent, #424) and the
    project `kb` proxy is the one launched — the warning belongs under that row."""
    out = _run(monkeypatch, capsys, [
        _row("user", "terse", "router", "gh, kb"),
        _row("user", "kb", "folded"),
        _row("project", "kb", "wrapped", "kb-server --stdio", identity="kb")])
    lines = out.splitlines()
    proj = lines.index(next(ln for ln in lines if ln.startswith("[project]")))
    user_block, proj_block = lines[:proj], lines[proj:]
    assert any("`kb` is ALSO run by terse" in ln for ln in proj_block)
    assert not any("`kb` is ALSO run by terse" in ln for ln in user_block)
