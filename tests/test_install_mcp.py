"""Tests for the Claude Code MCP installer (terse install-mcp / uninstall-mcp)."""
from __future__ import annotations

import json
import os
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
                             "--server-name", "runecho",
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


def test_wrap_url_server_proxies_the_url_with_headers():
    # An HTTP/SSE server has a 'url' (+ optional 'headers'), no 'command' — #5: terse
    # now wraps it by pointing the proxy's downstream at that url.
    original = {"type": "sse", "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer secret-token"}}
    config = _cfg(remote=dict(original))
    stash: dict = {}

    im.wrap(config, stash, "remote", "/p/policy.json", TERSE_CMD)
    entry = config["mcpServers"]["remote"]
    assert entry["command"] == "/abs/python"
    assert entry["args"] == [
        "-m", "terse", "proxy", "--policy", "/p/policy.json", "--server-name", "remote",
        "--header", "Authorization=Bearer secret-token",
        "--", "https://example.com/mcp",
    ]
    assert "url" not in entry and "headers" not in entry     # folded into args
    assert entry["type"] == "sse"                             # other keys preserved
    assert stash["remote"] == original

    im.unwrap(config, stash, "remote")
    assert config["mcpServers"]["remote"] == original
    assert "remote" not in stash


def test_wrap_url_server_without_headers_omits_header_flags():
    config = _cfg(remote={"url": "https://example.com/mcp"})
    im.wrap(config, {}, "remote", "/p/policy.json", TERSE_CMD)
    args = config["mcpServers"]["remote"]["args"]
    assert "--header" not in args
    assert args[-1] == "https://example.com/mcp"


def test_wrap_malformed_entry_without_command_or_url_raises():
    # Neither 'command' nor 'url' — not a valid MCP server entry at all (#19).
    config = _cfg(broken={"type": "mystery"})
    with pytest.raises(ValueError) as exc:
        im.wrap(config, {}, "broken", "/p/policy.json", TERSE_CMD)
    msg = str(exc.value)
    assert "command" in msg and "url" in msg


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
    assert stash["user"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}

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


def test_wrap_diff_adds_proxy_flags_and_rewrap_drops_them():
    config = _cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})
    stash: dict = {}

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD,
            diff=True, diff_keyframe_interval=3)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--diff" in args
    ki = args.index("--diff-keyframe-interval")
    assert args[ki + 1] == "3"
    assert args.index("--diff") < args.index("--")      # opts, not downstream args

    # tri-state: None (default) writes no diff flags at all — the entry inherits the
    # proxy default; a re-wrap from an explicit state drops the old flags.
    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD,
            diff=None, diff_keyframe_interval=3)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--diff" not in args and "--no-diff" not in args
    assert "--diff-keyframe-interval" in args           # keyframe is diff-independent now

    # False bakes an explicit opt-out (and a keyframe interval would be dead weight)
    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD,
            diff=False, diff_keyframe_interval=3)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--no-diff" in args and "--diff" not in args
    assert "--diff-keyframe-interval" not in args

    # and the original is still restored untouched
    im.unwrap(config, stash, "runecho")
    assert config["mcpServers"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}


def test_wrap_bakes_the_config_server_name(tmp_path):
    # #83: the config's own name is the one server identity terse can state rather than
    # guess from the launch command — it makes a server-scoped policy rule match and
    # labels the stats ledger truthfully.
    config = _cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})
    stash: dict = {}
    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD)
    args = config["mcpServers"]["runecho"]["args"]
    assert args[args.index("--server-name") + 1] == "runecho"
    assert args.index("--server-name") < args.index("--")   # a proxy opt, not a downstream arg
    im.unwrap(config, stash, "runecho")
    assert config["mcpServers"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}


def test_wrap_no_stats_bakes_opt_out_and_rewrap_drops_it():
    # The ledger is the proxy default, so only the opt-out is bakeable — and like the
    # diff flags it reflects the LATEST invocation, never accumulating.
    config = _cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})
    stash: dict = {}

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD, no_stats=True)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--no-stats" in args and args.index("--no-stats") < args.index("--")

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--no-stats" not in args                     # default: inherit the proxy's ON

    im.unwrap(config, stash, "runecho")
    assert config["mcpServers"]["runecho"] == {"command": "uvx", "args": ["runecho-mcp"]}


def test_do_install_diff_adds_flag_and_reinstall_without_it_drops_it(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    res = im.do_install(["runecho"], str(policy), cfg=cfg, diff=True)
    args = json.loads(cfg.read_text())["mcpServers"]["runecho"]["args"]
    assert "--diff" in args and args.index("--diff") < args.index("--")
    assert "--diff-keyframe-interval" not in args       # default left to the proxy
    assert res["diff"] is True

    # flags reflect the latest install: a plain re-install (tri-state None) removes
    # the explicit flag and the entry inherits the proxy default again
    res = im.do_install(["runecho"], str(policy), cfg=cfg)
    args = json.loads(cfg.read_text())["mcpServers"]["runecho"]["args"]
    assert "--diff" not in args and "--no-diff" not in args
    assert res["diff"] is None

    # an explicit opt-out bakes --no-diff
    res = im.do_install(["runecho"], str(policy), cfg=cfg, diff=False)
    args = json.loads(cfg.read_text())["mcpServers"]["runecho"]["args"]
    assert "--no-diff" in args and args.index("--no-diff") < args.index("--")
    assert res["diff"] is False

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


def test_prune_backups_keeps_only_most_recent(tmp_path):
    # Config backups hold copies of the config's secrets (MCP `env` blocks), so they must
    # not accumulate without bound — keep a short rollback window, delete the rest.
    cfg = tmp_path / ".claude.json"
    cfg.write_text("{}")
    made = []
    for i in range(im._MAX_BACKUPS + 3):
        b = cfg.with_name(f"{cfg.name}.bak-{1000 + i}")
        b.write_text(f"backup {i}")
        os.utime(b, (1000 + i, 1000 + i))  # deterministic oldest->newest mtimes
        made.append(b)

    im._prune_backups(cfg)

    remaining = sorted(cfg.parent.glob(f"{cfg.name}.bak-*"))
    assert len(remaining) == im._MAX_BACKUPS          # window enforced
    assert not made[0].exists() and not made[2].exists()  # 3 oldest pruned
    assert made[-1].exists() and made[3].exists()     # newest _MAX_BACKUPS survive
    assert cfg.read_text() == "{}"                    # the config itself is never touched


def test_prune_backups_disabled_when_keep_zero(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text("{}")
    for i in range(4):
        cfg.with_name(f"{cfg.name}.bak-{2000 + i}").write_text("x")
    im._prune_backups(cfg, keep=0)  # 0 = pruning off
    assert len(list(cfg.parent.glob(f"{cfg.name}.bak-*"))) == 4


def test_do_install_prunes_old_backups(tmp_path, monkeypatch):
    # Integration: a real install triggers _backup, which prunes down to the window even
    # when a pile of stale backups already exists.
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)
    for i in range(im._MAX_BACKUPS + 2):  # more stale backups than the window allows
        b = cfg.with_name(f"{cfg.name}.bak-{500 + i}")
        b.write_text("stale")
        os.utime(b, (500 + i, 500 + i))  # all older than the one do_install will make

    im.do_install(["runecho"], str(policy), cfg=cfg)

    assert len(list(cfg.parent.glob(f"{cfg.name}.bak-*"))) == im._MAX_BACKUPS


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


# --------------------------------------------------------------------------- #
# --scope support (#58): user (default), project (.mcp.json), local (nested
# projects."<repo-path>".mcpServers)
# --------------------------------------------------------------------------- #
def test_resolve_target_user_scope_defaults_to_config_path(monkeypatch, tmp_path):
    fake_home_cfg = tmp_path / ".claude.json"
    monkeypatch.setattr(im, "config_path", lambda: fake_home_cfg)
    target = im.resolve_target("user")
    assert target.cfg == fake_home_cfg
    assert target.server_path == ()
    assert target.stash_prefix == "user"


def test_resolve_target_project_scope_defaults_to_cwd_mcp_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = im.resolve_target("project")
    assert target.cfg == (tmp_path / ".mcp.json").resolve()
    assert target.server_path == ()
    assert target.stash_prefix == "project"


def test_resolve_target_project_scope_honors_file_override(tmp_path):
    custom = tmp_path / "sub" / "custom.mcp.json"
    target = im.resolve_target("project", file=str(custom))
    assert target.cfg == custom.resolve()


def test_resolve_target_local_scope_nests_under_projects(monkeypatch, tmp_path):
    fake_home_cfg = tmp_path / ".claude.json"
    monkeypatch.setattr(im, "config_path", lambda: fake_home_cfg)
    target = im.resolve_target("local", repo_path="/repo/root")
    assert target.cfg == fake_home_cfg
    assert target.server_path == ("projects", "/repo/root")
    assert target.stash_prefix == "local:/repo/root"


def test_resolve_target_unknown_scope_raises():
    with pytest.raises(ValueError):
        im.resolve_target("bogus")


def test_default_repo_path_resolves_to_worktree_bare_root(tmp_path):
    # A claudew/codexw-style bare-worktree layout: <repo>/.bare is the actual git
    # dir, and a worktree checkout under <repo>/wt has its own .git FILE pointing
    # into .bare's worktrees/ subdir. `git rev-parse --git-common-dir` from inside
    # the worktree must resolve to the .bare dir itself, not the worktree cwd —
    # this is the exact acceptance criterion from #58 ("worktree repos resolve
    # local scope to the bare root, not cwd"), reproduced with a real git repo
    # rather than mocked.
    import os as _os
    import subprocess

    def run(*args, cwd):
        subprocess.run([str(a) for a in args], cwd=cwd, check=True, capture_output=True)

    repo = tmp_path / "myrepo"
    src = tmp_path / "_src"
    src.mkdir()
    run("git", "init", cwd=src)
    run("git", "config", "user.email", "t@example.com", cwd=src)
    run("git", "config", "user.name", "t", cwd=src)
    (src / "f.txt").write_text("x")
    run("git", "add", "f.txt", cwd=src)
    run("git", "commit", "-m", "init", cwd=src)

    bare = repo / ".bare"
    run("git", "clone", "--bare", str(src), str(bare), cwd=tmp_path)
    branch = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], cwd=bare,
                            capture_output=True, text=True, check=True).stdout.strip()

    worktree = repo / "wt"
    run("git", "worktree", "add", str(worktree), branch, cwd=bare)

    old_cwd = Path.cwd()
    try:
        _os.chdir(worktree)
        repo_path = im.default_repo_path()
    finally:
        _os.chdir(old_cwd)
    assert repo_path == str(bare.resolve())  # bare root, not `worktree`


def test_default_repo_path_not_a_git_repo_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # tmp_path is not inside any git repo
    with pytest.raises(ValueError, match="not a git repo|--repo-path"):
        im.default_repo_path()


def test_install_uninstall_roundtrip_project_scope(tmp_path, monkeypatch):
    mcp_json = tmp_path / ".mcp.json"
    original = {"command": "uvx", "args": ["runecho-mcp"]}
    mcp_json.write_text(json.dumps({"mcpServers": {"runecho": original}}))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    res = im.do_install(["runecho"], str(policy), scope="project", file=str(mcp_json))
    assert res["scope"] == "project"
    written = json.loads(mcp_json.read_text())
    assert written["mcpServers"]["runecho"]["command"] == "/abs/python"
    stash = json.loads(im.stash_path(mcp_json).read_text())
    assert stash["project"]["runecho"] == original

    im.do_uninstall(["runecho"], scope="project", file=str(mcp_json))
    assert json.loads(mcp_json.read_text())["mcpServers"]["runecho"] == original


def test_install_uninstall_roundtrip_local_scope(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    original = {"command": "uvx", "args": ["runecho-mcp"]}
    cfg.write_text(json.dumps({
        "mcpServers": {},
        "projects": {"/repo/root": {"mcpServers": {"runecho": original}, "otherKey": 1}},
    }))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    res = im.do_install(["runecho"], str(policy), scope="local", cfg=cfg,
                        repo_path="/repo/root")
    assert res["scope"] == "local"
    written = json.loads(cfg.read_text())
    proj = written["projects"]["/repo/root"]
    assert proj["mcpServers"]["runecho"]["command"] == "/abs/python"
    assert proj["otherKey"] == 1  # untouched sibling key
    assert written["mcpServers"] == {}  # user-scope block untouched
    stash = json.loads(im.stash_path(cfg).read_text())
    assert stash["local:/repo/root"]["runecho"] == original

    im.do_uninstall(["runecho"], scope="local", cfg=cfg, repo_path="/repo/root")
    restored = json.loads(cfg.read_text())["projects"]["/repo/root"]["mcpServers"]["runecho"]
    assert restored == original


def test_same_server_independently_managed_in_user_and_local_scope(tmp_path, monkeypatch):
    # user and local scope share the same physical ~/.claude.json — a server wrapped
    # in BOTH must not collide in the stash (#58's "stash needs a scope-qualified
    # key" requirement).
    cfg = tmp_path / ".claude.json"
    user_original = {"command": "uvx", "args": ["runecho-mcp", "--user"]}
    local_original = {"command": "uvx", "args": ["runecho-mcp", "--local"]}
    cfg.write_text(json.dumps({
        "mcpServers": {"runecho": user_original},
        "projects": {"/repo/root": {"mcpServers": {"runecho": local_original}}},
    }))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    im.do_install(["runecho"], str(policy), scope="user", cfg=cfg)
    im.do_install(["runecho"], str(policy), scope="local", cfg=cfg, repo_path="/repo/root")

    written = json.loads(cfg.read_text())
    assert written["mcpServers"]["runecho"]["args"][-1] == "--user"
    assert written["projects"]["/repo/root"]["mcpServers"]["runecho"]["args"][-1] == "--local"

    im.do_uninstall(["runecho"], scope="user", cfg=cfg)
    im.do_uninstall(["runecho"], scope="local", cfg=cfg, repo_path="/repo/root")
    written = json.loads(cfg.read_text())
    assert written["mcpServers"]["runecho"] == user_original
    assert written["projects"]["/repo/root"]["mcpServers"]["runecho"] == local_original


def test_legacy_flat_stash_migrates_to_user_scope(tmp_path, monkeypatch):
    # Pre-#58 stash files are flat ({server: original_entry}) with no scope
    # namespacing at all — every real installed stash predates this change, so
    # uninstall must keep working on them without any manual migration step.
    cfg = tmp_path / ".claude.json"
    wrapped = {"command": "/abs/python", "args": ["-m", "terse", "proxy", "--policy",
                                                   "/p.json", "--", "uvx", "runecho-mcp"]}
    original = {"command": "uvx", "args": ["runecho-mcp"]}
    cfg.write_text(json.dumps({"mcpServers": {"runecho": wrapped}}))
    im.stash_path(cfg).write_text(json.dumps({"runecho": original}))

    res = im.do_uninstall(["runecho"], cfg=cfg)  # default scope="user"
    assert res["changes"] == [{"server": "runecho", "restored": True}]
    assert json.loads(cfg.read_text())["mcpServers"]["runecho"] == original
    # migrated on write: stash is now namespaced, not flat
    stash = json.loads(im.stash_path(cfg).read_text())
    assert stash == {"user": {}}


# --------------------------------------------------------------------------- #
# scan_scopes / mcp-status: read-only enumeration across all three scopes
# --------------------------------------------------------------------------- #
def test_scan_scopes_reports_wrapped_unwrapped_and_orphaned(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    wrapped = {"command": "/abs/python", "args": ["-m", "terse", "proxy", "--policy",
                                                   "/p.json", "--", "uvx", "runecho-mcp"]}
    cfg.write_text(json.dumps({
        "mcpServers": {
            "runecho": wrapped,           # managed + present -> wrapped
            "plain": {"command": "uvx", "args": ["plain-mcp"]},  # unmanaged -> unwrapped
        },
    }))
    # a stash entry with NO matching mcpServers entry -> orphaned-stash
    im.stash_path(cfg).write_text(json.dumps(
        {"user": {"runecho": {"command": "uvx", "args": ["runecho-mcp"]},
                  "ghost": {"command": "uvx", "args": ["ghost-mcp"]}}}))
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)  # no .mcp.json here -> project scope contributes nothing

    rows = im.scan_scopes()
    by_name = {r["server"]: r for r in rows if r["scope"] == "user"}
    assert by_name["runecho"]["state"] == "wrapped"
    assert by_name["runecho"]["policy"] == "/p.json"
    assert by_name["plain"]["state"] == "unwrapped"
    assert by_name["plain"]["policy"] is None
    assert by_name["ghost"]["state"] == "orphaned-stash"
    assert by_name["ghost"]["policy"] is None
    assert not any(r["scope"] == "project" for r in rows)


def test_scan_scopes_includes_project_and_local_when_present(tmp_path, monkeypatch):
    # cfg (user+local, ~/.claude.json) and .mcp.json (project) live in DIFFERENT
    # directories -- each has its own sidecar stash (STASH_NAME is a fixed filename
    # next to its config), so sharing one dir would collide the two stashes.
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()

    cfg = home_dir / ".claude.json"
    cfg.write_text(json.dumps({
        "mcpServers": {},
        "projects": {"/repo/root": {"mcpServers": {
            "demo": {"command": "/abs/python", "args": ["-m", "terse", "proxy",
                                                         "--policy", "/local.json",
                                                         "--", "demo-mcp"]}}}},
    }))
    im.stash_path(cfg).write_text(json.dumps(
        {"local:/repo/root": {"demo": {"command": "demo-mcp", "args": []}}}))
    monkeypatch.setattr(im, "config_path", lambda: cfg)

    mcp_json = proj_dir / ".mcp.json"
    mcp_json.write_text(json.dumps({"mcpServers": {
        "proj-demo": {"command": "/abs/python", "args": ["-m", "terse", "proxy",
                                                          "--policy", "/proj.json",
                                                          "--", "proj-mcp"]}}}))
    im.stash_path(mcp_json).write_text(json.dumps(
        {"project": {"proj-demo": {"command": "proj-mcp", "args": []}}}))
    monkeypatch.chdir(proj_dir)

    rows = im.scan_scopes(repo_path="/repo/root")
    by_scope = {(r["scope"], r["server"]): r for r in rows}
    assert by_scope[("local", "demo")]["state"] == "wrapped"
    assert by_scope[("local", "demo")]["policy"] == "/local.json"
    assert by_scope[("project", "proj-demo")]["state"] == "wrapped"
    assert by_scope[("project", "proj-demo")]["policy"] == "/proj.json"


def test_scan_scopes_never_raises_when_local_scope_unresolvable(tmp_path, monkeypatch):
    # Not inside a git repo, no --repo-path given -> local scope is silently omitted,
    # not an error (this is the common case: most invocations aren't in a repo at all).
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)  # tmp_path is not inside any git repo
    rows = im.scan_scopes()
    assert not any(r["scope"] == "local" for r in rows)


def test_scan_scopes_missing_files_return_empty_not_error(tmp_path, monkeypatch):
    monkeypatch.setattr(im, "config_path", lambda: tmp_path / "nonexistent.json")
    monkeypatch.chdir(tmp_path)
    assert im.scan_scopes() == []


def test_scan_scopes_is_read_only(tmp_path, monkeypatch):
    # A scan must never write the config, the stash, or fabricate a backup — same
    # write-nothing contract as do_uninstall(dry_run=True).
    cfg = tmp_path / ".claude.json"
    before = json.dumps({"mcpServers": {"demo": {"command": "uvx", "args": []}}})
    cfg.write_text(before)
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)

    im.scan_scopes()
    assert cfg.read_text() == before
    assert not im.stash_path(cfg).exists()
    assert list(tmp_path.glob("*.bak-*")) == []


def test_rewrap_preserves_hand_edits_on_wrapped_entry():
    # The 2026-07-13 production incident: a scoped env.PATH pin hand-added to the
    # WRAPPED entry was silently reverted by a re-install, because wrap() rebuilt
    # purely from the stashed (pre-pin) original. The drift guard keeps live
    # non-terse-owned keys on a re-wrap.
    config = _cfg(codegraph={"command": "/usr/local/bin/codegraph",
                             "args": ["serve", "--mcp"], "type": "stdio"})
    stash: dict = {}
    im.wrap(config, stash, "codegraph", "/p/policy.json", TERSE_CMD)
    # operator pins node@22 on the wrapped entry by hand
    config["mcpServers"]["codegraph"]["env"] = {"PATH": "/opt/node22/bin:/usr/bin"}

    im.wrap(config, stash, "codegraph", "/p/policy.json", TERSE_CMD, diff=False)
    entry = config["mcpServers"]["codegraph"]
    assert entry["env"] == {"PATH": "/opt/node22/bin:/usr/bin"}   # pin survived
    assert "--no-diff" in entry["args"]                            # flags still rebuilt
    assert entry["command"] == "/abs/python"                       # command still terse's

    # a live hand-edit also WINS over the stashed original's value for the same key
    config["mcpServers"]["codegraph"]["type"] = "http"             # hand-changed
    im.wrap(config, stash, "codegraph", "/p/policy.json", TERSE_CMD)
    assert config["mcpServers"]["codegraph"]["type"] == "http"

    # the guard never leaks the hand-edit into the stash: uninstall restores pristine
    im.unwrap(config, stash, "codegraph")
    assert config["mcpServers"]["codegraph"] == {
        "command": "/usr/local/bin/codegraph", "args": ["serve", "--mcp"], "type": "stdio"}


def test_rewrap_never_resurrects_url_headers_from_a_drifted_live_entry():
    # If someone hand-replaces a managed server's live entry with a raw url entry,
    # a re-wrap must not copy url/headers onto the wrapped shape (an entry with both
    # args and url is broken) — those keys are always folded into args from the stash.
    original = {"url": "https://example.com/mcp", "headers": {"X": "1"}}
    config = _cfg(remote=dict(original))
    stash: dict = {}
    im.wrap(config, stash, "remote", "/p/policy.json", TERSE_CMD)
    config["mcpServers"]["remote"] = dict(original)                # hand-reverted
    im.wrap(config, stash, "remote", "/p/policy.json", TERSE_CMD)
    entry = config["mcpServers"]["remote"]
    assert "url" not in entry and "headers" not in entry
    assert entry["args"][-1] == "https://example.com/mcp"


def test_do_install_reports_preserved_hand_edits(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    im.do_install(["runecho"], str(policy), cfg=cfg)
    written = json.loads(cfg.read_text())
    written["mcpServers"]["runecho"]["env"] = {"PATH": "/pin"}     # hand-edit
    cfg.write_text(json.dumps(written))

    res = im.do_install(["runecho"], str(policy), cfg=cfg)
    change = res["changes"][0]
    assert change["preserved"] == ["env"]
    assert json.loads(cfg.read_text())["mcpServers"]["runecho"]["env"] == {"PATH": "/pin"}
    # the edit stays live-only (never leaks into the stash), so EVERY later re-wrap
    # keeps carrying — and keeps reporting — it; that persistence is the guard working
    res = im.do_install(["runecho"], str(policy), cfg=cfg)
    assert res["changes"][0]["preserved"] == ["env"]
    assert json.loads(cfg.read_text())["mcpServers"]["runecho"]["env"] == {"PATH": "/pin"}


def test_classify_server_sensitivity():
    from terse.install_mcp import classify_server_sensitivity
    # obvious by name
    assert classify_server_sensitivity("secret-broker")
    assert classify_server_sensitivity("acme-vault")
    assert classify_server_sensitivity("my-authgw")
    # caught via the launch command even when the name is innocuous
    assert classify_server_sensitivity("store", ["python", "-m", "credential_daemon"])
    # not flagged — operator must add these to never_lossy_servers by hand (kb, sb-run)
    assert not classify_server_sensitivity("runecho")
    assert not classify_server_sensitivity("kb")
    assert not classify_server_sensitivity("sb-run")


def test_add_never_lossy_server_pure():
    doc: dict = {}
    assert im.add_never_lossy_server(doc, "kb") is True
    assert doc["never_lossy_servers"] == ["kb"]
    assert im.add_never_lossy_server(doc, "kb") is False          # dedup -> no change
    assert im.add_never_lossy_server(doc, "sb-run") is True
    assert doc["never_lossy_servers"] == ["kb", "sb-run"]          # sorted


def test_do_install_never_lossy_bakes_into_policy(tmp_path, monkeypatch):
    from terse.policy import load_policy
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "policies": []}))
    monkeypatch.setattr(im, "terse_invocation", lambda: TERSE_CMD)

    res = im.do_install(["runecho"], str(policy), cfg=cfg, never_lossy=True)
    assert res["never_lossy_added"] == ["runecho"]
    # runecho's name is NOT secret-shaped, so this proves the BAKED list did the work:
    assert load_policy(policy).server_never_lossy("runecho") is True

    # dry-run reports what it would add but does NOT write the policy file
    policy.write_text(json.dumps({"version": 1, "policies": []}))
    res2 = im.do_install(["runecho"], str(policy), cfg=cfg, never_lossy=True, dry_run=True)
    assert res2["never_lossy_added"] == ["runecho"]
    assert load_policy(policy).server_never_lossy("runecho") is False
