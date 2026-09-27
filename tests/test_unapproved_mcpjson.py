"""#448 — a project `.mcp.json` entry the user REJECTED is in the file but never launched by
Claude Code, so it must not take a name's precedence slot from the user- or local-scope
definition that actually runs. A merely PENDING entry is different: `claude -p`, Agent SDK
and cloud sessions load it without asking, so it keeps its real state and stays a writer.

Every fixture is synthetic: a temp `claude.json`, a temp `.mcp.json`, and temp settings files.
Nothing here reads the real `~/.claude.json`.
"""
from __future__ import annotations

import json

import pytest

from terse import install_mcp as im
from terse.stats import (
    _ABSENT_FROM_SCOPE,
    _PAYS_PRIMER,
    _WRITES_LEDGER_ROWS,
    _contested_labels,
    _precedence_winner,
)


@pytest.fixture(autouse=True)
def _no_real_config_dir(monkeypatch):
    # The real $CLAUDE_CONFIG_DIR would make the user settings file the operator's own.
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def fleet(tmp_path):
    """User scope defines `kb`; the project `.mcp.json` defines `kb` and `gh`."""
    cfg = tmp_path / "home" / "claude.json"
    _write(cfg, {"mcpServers": {"kb": {"command": "kb-user"}}})
    proj = tmp_path / "proj"
    mcp = proj / ".mcp.json"
    _write(mcp, {"mcpServers": {"kb": {"command": "kb-proj"}, "gh": {"command": "gh-proj"}}})
    return cfg, proj, mcp


def _scan(cfg, mcp, proj):
    # `repo_path` pins local scope to a key the fixture controls: never the real repo.
    return im.scan_scopes(cfg=cfg, file=str(mcp), repo_path=str(proj))


def _project(rows):
    return {r["server"]: r for r in rows if r["scope"] == "project"}


def test_a_rejected_project_entry_is_unapproved_and_loses_the_slot(fleet):
    """The issue's reproduction: the project `kb` is rejected, so the user `kb` is the one
    running, and it must be the row that speaks for the name."""
    cfg, proj, mcp = fleet
    _write(proj / ".claude" / "settings.local.json", {"disabledMcpjsonServers": ["kb"]})
    rows = _scan(cfg, mcp, proj)
    project = _project(rows)
    assert (project["kb"]["state"], project["kb"]["approval"]) == ("unapproved", "rejected")
    assert rows[_precedence_winner(rows)["kb"]]["scope"] == "user"


def test_a_pending_project_entry_keeps_its_state_and_the_slot(fleet):
    """Review of PR #477: `claude -p` loads a pending project server without asking, so it
    is still the running definition in those sessions and keeps project precedence."""
    cfg, proj, mcp = fleet
    rows = _scan(cfg, mcp, proj)
    project = _project(rows)
    assert (project["kb"]["state"], project["kb"]["approval"]) == ("unwrapped", "pending")
    assert rows[_precedence_winner(rows)["kb"]]["scope"] == "project"


def _terse_entry(server_name):
    return {"command": "/abs/python",
            "args": ["-m", "terse", "proxy", "--server-name", server_name,
                     "--", f"{server_name}-server"]}


def test_a_pending_wrapped_row_still_pays_its_primer_and_writes_the_ledger(tmp_path):
    cfg = tmp_path / "home" / "claude.json"
    _write(cfg, {"mcpServers": {}})
    proj = tmp_path / "proj"
    mcp = proj / ".mcp.json"
    _write(mcp, {"mcpServers": {"kb": _terse_entry("kb")}})
    row = _project(_scan(cfg, mcp, proj))["kb"]
    assert row["approval"] == "pending"
    assert row["state"] in _PAYS_PRIMER and row["state"] in _WRITES_LEDGER_ROWS
    assert row["ledger_identity"] == "kb"


@pytest.mark.parametrize("approval, contested", [(None, True), ("reject", False)])
def test_a_pending_duplicate_of_a_router_peer_is_contested_a_rejected_one_is_not(
        tmp_path, approval, contested):
    """A user-scope router folds `kb`; the project `.mcp.json` wraps its own `kb` proxy.
    Pending: `claude -p` runs both, so the `kb` label has two writers. Rejected: only the
    router writes it."""
    cfg = tmp_path / "home" / "claude.json"
    _write(cfg, {"mcpServers": {"kb": {"command": "kb-mcp"}, "gh": {"command": "gh-mcp"}}})
    pol = tmp_path / "p.json"
    _write(pol, {"version": 1, "defaults": {"tiers": ["minify"]}})
    im.do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    proj = tmp_path / "proj"
    mcp = proj / ".mcp.json"
    _write(mcp, {"mcpServers": {"kb": _terse_entry("kb")}})
    if approval == "reject":
        _write(proj / ".claude" / "settings.local.json", {"disabledMcpjsonServers": ["kb"]})
    rows = _scan(cfg, mcp, proj)
    assert ("kb" in _contested_labels(rows)) is contested


def test_approved_in_the_claude_json_project_block(fleet):
    cfg, proj, mcp = fleet
    doc = json.loads(cfg.read_text())
    doc["projects"] = {str(proj): {"enabledMcpjsonServers": ["kb"],
                                   "disabledMcpjsonServers": ["gh"]}}
    _write(cfg, doc)
    rows = _scan(cfg, mcp, proj)
    project = _project(rows)
    assert (project["kb"]["state"], project["kb"]["approval"]) == ("unwrapped", "approved")
    assert (project["gh"]["state"], project["gh"]["approval"]) == ("unapproved", "rejected")
    assert rows[_precedence_winner(rows)["kb"]]["scope"] == "project"


@pytest.mark.parametrize("settings", ["settings.json", "settings.local.json"])
def test_approved_in_project_settings_files(fleet, settings):
    """Current Claude Code writes the approval dialog's answer to `.claude/settings.local.json`,
    not to `~/.claude.json` — both must count."""
    cfg, proj, mcp = fleet
    _write(proj / ".claude" / settings, {"enabledMcpjsonServers": ["kb"]})
    project = _project(_scan(cfg, mcp, proj))
    assert project["kb"]["approval"] == "approved"
    assert project["gh"]["approval"] == "pending"


def test_approved_in_user_settings_beside_the_config(fleet):
    cfg, proj, mcp = fleet
    _write(cfg.parent / ".claude" / "settings.json", {"enabledMcpjsonServers": ["gh"]})
    assert _project(_scan(cfg, mcp, proj))["gh"]["approval"] == "approved"


def test_user_settings_follow_CLAUDE_CONFIG_DIR(fleet, tmp_path, monkeypatch):
    cfg, proj, mcp = fleet
    config_dir = tmp_path / "ccdir"
    _write(config_dir / "settings.json", {"disabledMcpjsonServers": ["gh"]})
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    assert _project(_scan(cfg, mcp, proj))["gh"]["approval"] == "rejected"


def test_enable_all_approves_everything_but_a_rejection_still_wins(fleet):
    cfg, proj, mcp = fleet
    _write(proj / ".claude" / "settings.local.json",
           {"enableAllProjectMcpServers": True, "disabledMcpjsonServers": ["gh"]})
    project = _project(_scan(cfg, mcp, proj))
    assert project["kb"]["approval"] == "approved"
    assert project["gh"]["approval"] == "rejected"
    assert project["gh"]["state"] == "unapproved"


def test_non_project_rows_carry_no_approval(fleet):
    cfg, proj, mcp = fleet
    rows = _scan(cfg, mcp, proj)
    assert all(r["approval"] is None for r in rows if r["scope"] != "project")


def test_unreadable_settings_are_ignored_not_raised(fleet):
    cfg, proj, mcp = fleet
    (proj / ".claude").mkdir()
    (proj / ".claude" / "settings.local.json").write_text("{not json", encoding="utf-8")
    _write(proj / ".claude" / "settings.json", ["not", "a", "dict"])
    project = _project(_scan(cfg, mcp, proj))
    assert project["kb"]["approval"] == "pending"


def test_unapproved_is_an_absent_state():
    assert "unapproved" in _ABSENT_FROM_SCOPE
