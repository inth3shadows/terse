"""Tests for the Claude Code MCP installer (terse install-mcp / uninstall-mcp)."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from terse import install_mcp as im

TERSE_CMD = ["/abs/python", "-m", "terse"]


def _cfg(**servers):
    return {"mcpServers": servers, "otherTopLevel": {"keep": True}}


def test_wrap_then_unwrap_roundtrips_exactly():
    original = {"command": "uvx", "args": ["runecho-mcp", "--flag"],
                "env": {"X": "1"}, "cwd": "/some/dir"}
    config = _cfg(runecho=dict(original))
    stash: dict = {}

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD)
    entry = config["mcpServers"]["runecho"]
    assert entry["command"] == "/abs/python"
    assert entry["args"] == ["-m", "terse", "proxy", "--policy", "/p/policy.json",
                             "--", "uvx", "runecho-mcp", "--flag"]
    # non-command/args keys preserved
    assert entry["env"] == {"X": "1"} and entry["cwd"] == "/some/dir"
    assert stash["runecho"] == original

    im.unwrap(config, stash, "runecho")
    assert config["mcpServers"]["runecho"] == original
    assert "runecho" not in stash


def test_wrap_is_idempotent_no_double_nesting():
    config = _cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})
    stash: dict = {}
    im.wrap(config, stash, "runecho", "/p/a.json", TERSE_CMD)
    once = json.loads(json.dumps(config["mcpServers"]["runecho"]))
    # re-wrap with a NEW policy: must re-wrap from the stashed original, not nest
    im.wrap(config, stash, "runecho", "/p/b.json", TERSE_CMD)
    twice = config["mcpServers"]["runecho"]
    assert twice["args"].count("proxy") == 1
    assert "/p/b.json" in twice["args"] and "/p/a.json" not in twice["args"]
    # and it still restores to the true original
    im.unwrap(config, stash, "runecho")
    assert config["mcpServers"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}
    assert once != twice  # policy actually changed between wraps


def test_wrap_http_sse_server_fails_fast_with_clear_message():
    # an HTTP/SSE server has a 'url', no 'command' — terse can't proxy it (#19/#5)
    config = _cfg(remote={"type": "sse", "url": "https://example.com/mcp"})
    with pytest.raises(ValueError) as exc:
        im.wrap(config, {}, "remote", "/p/policy.json", TERSE_CMD)
    msg = str(exc.value)
    assert "url" in msg and "#5" in msg


def test_unwrap_unmanaged_raises():
    with pytest.raises(KeyError):
        im.unwrap(_cfg(x={"command": "c"}), {}, "x")


def test_do_install_writes_config_stash_and_backup(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    res = im.do_install(["runecho"], str(policy), cfg=cfg)
    assert res["backup"] and (tmp_path / res["backup"].split("/")[-1]).exists()
    written = json.loads(cfg.read_text())
    assert written["mcpServers"]["runecho"]["command"] == "/abs/python"
    assert written["otherTopLevel"] == {"keep": True}  # untouched
    stash = json.loads(im.stash_path(cfg).read_text())
    assert stash["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}

    # config, stash, and backup can all carry secrets (MCP server `env` blocks) — every
    # file this operation writes must be owner-only, never world/group-readable.
    for written_path in (cfg, im.stash_path(cfg), Path(res["backup"])):
        assert stat.S_IMODE(written_path.stat().st_mode) == 0o600

    # full round-trip: uninstall restores the original mcpServers entry
    im.do_uninstall(["runecho"], cfg=cfg)
    back = json.loads(cfg.read_text())
    assert back["mcpServers"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}


def test_do_install_capture_dir_adds_proxy_flag(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    cap = tmp_path / "session-corpus"
    res = im.do_install(["runecho"], str(policy), cfg=cfg, capture_dir=str(cap))
    args = json.loads(cfg.read_text())["mcpServers"]["runecho"]["args"]
    # the proxy carries --capture-dir <abs> BEFORE the `--` downstream separator
    assert "--capture-dir" in args
    ci = args.index("--capture-dir")
    assert args[ci + 1] == str(cap.resolve())          # absolute, cwd-independent
    assert ci < args.index("--")                        # an opt, not a downstream arg
    assert res["capture_dir"] == str(cap.resolve())
    # uninstall still restores the true original (capture flag was terse's, not theirs)
    im.do_uninstall(["runecho"], cfg=cfg)
    assert json.loads(cfg.read_text())["mcpServers"]["runecho"] == {
        "command": "uvx", "args": ["runecho-mcp"]}


def test_roundtrip_byte_identical_with_non_ascii(tmp_path, monkeypatch):
    # The real ~/.claude.json holds non-ASCII (em-dashes, emoji, arrows) and is written
    # by Claude Code with indent=2, ensure_ascii=False, and NO trailing newline. #27's
    # acceptance is that install -> uninstall restores the file byte-for-byte. A naive
    # json.dumps (ensure_ascii=True, +"\n") silently fails this on any non-ASCII config.
    cfg = tmp_path / ".claude.json"
    original_obj = {
        "note": "onboarding — em-dash, emoji 🚨, and an arrow →",
        "mcpServers": {"runecho": {"command": "uvx", "args": ["runecho-mcp"]}},
        "otherTopLevel": {"keep": True},
    }
    original_text = json.dumps(original_obj, indent=2, ensure_ascii=False)  # no trailing nl
    cfg.write_text(original_text, encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    im.do_install(["runecho"], str(policy), cfg=cfg)
    assert "🚨" in cfg.read_text(encoding="utf-8")  # literal, not \uXXXX-escaped

    im.do_uninstall(["runecho"], cfg=cfg)
    assert cfg.read_text(encoding="utf-8") == original_text  # byte-identical to backup


def test_do_install_unknown_server_raises_with_available(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx"})))
    policy = tmp_path / "p.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)
    with pytest.raises(ValueError) as e:
        im.do_install(["nope"], str(policy), cfg=cfg)
    assert "runecho" in str(e.value)  # lists available


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    before = json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]}))
    cfg.write_text(before)
    policy = tmp_path / "p.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)
    res = im.do_install(["runecho"], str(policy), dry_run=True, cfg=cfg)
    assert res["dry_run"] and res["backup"] is None
    assert cfg.read_text() == before  # unchanged
    assert not im.stash_path(cfg).exists()
