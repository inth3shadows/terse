"""#448 — a project `.mcp.json` entry the user never approved (or rejected) is in the file but
never launched by Claude Code, so it must not take a name's precedence slot from the user- or
local-scope definition that actually runs.

Every fixture is synthetic: a temp `claude.json`, a temp `.mcp.json`, and temp settings files.
Nothing here reads the real `~/.claude.json`.
"""
from __future__ import annotations

import json

import pytest

from terse import install_mcp as im
from terse.stats import _ABSENT_FROM_SCOPE, _precedence_winner


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


def test_a_never_approved_project_entry_is_unapproved_and_loses_the_slot(fleet):
    """The issue's reproduction: nothing approves the project `kb`, so the user `kb` is the
    one running, and it must be the row that speaks for the name."""
    cfg, proj, mcp = fleet
    rows = _scan(cfg, mcp, proj)
    project = _project(rows)
    assert project["kb"]["state"] == "unapproved"
    assert project["kb"]["approval"] == "pending"
    winner = rows[_precedence_winner(rows)["kb"]]
    assert winner["scope"] == "user"


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
