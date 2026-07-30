"""Tests for the Claude Code MCP installer (terse install-mcp / uninstall-mcp)."""
from __future__ import annotations

import json
import os
import stat
import sys
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


def test_wrap_no_join_blocks_bakes_opt_out_and_rewrap_drops_it():
    # Joining is the proxy default (#116), so only the opt-out is bakeable — and like the
    # diff/stats flags it reflects the LATEST invocation, never accumulating.
    config = _cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})
    stash: dict = {}

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD, no_join_blocks=True)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--no-join-blocks" in args and args.index("--no-join-blocks") < args.index("--")

    im.wrap(config, stash, "runecho", "/p/policy.json", TERSE_CMD)
    args = config["mcpServers"]["runecho"]["args"]
    assert "--no-join-blocks" not in args               # default: inherit the proxy's ON

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


def test_scan_scopes_surfaces_wraps_diff_stats_and_missing_policy(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    missing = tmp_path / "gone.json"          # never created -> policy_missing
    present = tmp_path / "here.json"
    present.write_text("{}")
    cfg.write_text(json.dumps({"mcpServers": {
        # diff off, stats off, absolute policy that does not exist
        "codegraph": {"command": "/abs/python", "args": [
            "-m", "terse", "proxy", "--policy", str(missing), "--server-name",
            "codegraph", "--no-diff", "--no-stats", "--", "codegraph-mcp", "serve"]},
        # default diff (no flag), stats on, policy present
        "runecho": {"command": "/abs/python", "args": [
            "-m", "terse", "proxy", "--policy", str(present), "--", "runecho-mcp"]},
    }}))
    im.stash_path(cfg).write_text(json.dumps({"user": {
        "codegraph": {"command": "codegraph-mcp", "args": ["serve"]},
        "runecho": {"command": "runecho-mcp", "args": []}}}))
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)

    by = {r["server"]: r for r in im.scan_scopes() if r["scope"] == "user"}
    cg = by["codegraph"]
    assert cg["wraps"] == "codegraph-mcp serve"
    assert cg["diff"] == "off" and cg["stats"] is False
    assert cg["policy_missing"] is True
    ru = by["runecho"]
    assert ru["wraps"] == "runecho-mcp"
    # Resolved, not merely named: a bare "default" read as "on" and convinced a reader
    # diffing was unimplemented (#181). The label states the value it actually inherits.
    assert ru["diff"] == "default (off)" and ru["stats"] is True
    assert ru["policy_missing"] is False


def test_scan_scopes_never_flags_a_relative_policy_as_missing(tmp_path, monkeypatch):
    # A relative policy resolves against the MCP launcher's cwd, which a status scan
    # can't know — so it must never be reported MISSING even when absent here.
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "s": {"command": "/abs/python", "args": [
            "-m", "terse", "proxy", "--policy", "rel/policy.json", "--", "s-mcp"]}}}))
    im.stash_path(cfg).write_text(json.dumps({"user": {"s": {"command": "s-mcp", "args": []}}}))
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    row = next(r for r in im.scan_scopes() if r["server"] == "s")
    assert row["policy"] == "rel/policy.json" and row["policy_missing"] is False


def test_scan_scopes_wrapped_only_fields_are_none_for_non_wrapped(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "uvx", "args": ["plain-mcp"]}}}))  # unwrapped
    monkeypatch.setattr(im, "config_path", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    row = next(r for r in im.scan_scopes() if r["server"] == "plain")
    assert row["state"] == "unwrapped"
    assert row["wraps"] is None and row["diff"] is None and row["stats"] is None
    assert row["policy_missing"] is False


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


# --------------------------------------------------- $TERSE_MCP_CMD (the launcher)
# The override existed since the installer landed and had no test at all, which is
# how the tilde bug below survived: a wrapped entry is spawned from JSON via execve
# with no shell, so an unexpanded `~` writes a command that can never resolve — and
# the failure is silent (the client just can't start the server).
def test_terse_invocation_defaults_to_the_running_interpreter(monkeypatch):
    monkeypatch.delenv("TERSE_MCP_CMD", raising=False)
    assert im.terse_invocation() == [sys.executable, "-m", "terse"]


def test_terse_mcp_cmd_override_is_whitespace_split(tmp_path, monkeypatch):
    launcher = tmp_path / "terse"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TERSE_MCP_CMD", f"{launcher} --flag")
    assert im.terse_invocation() == [str(launcher), "--flag"]


def test_terse_mcp_cmd_override_expands_a_leading_tilde(tmp_path, monkeypatch):
    # Quoting the value (or setting it in a script) means the shell never expands `~`.
    # Without expanduser here the literal tilde lands in the config verbatim.
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    launcher = home / ".local" / "bin" / "terse"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # expanduser's key on Windows
    monkeypatch.setenv("TERSE_MCP_CMD", "~/.local/bin/terse")

    assert im.terse_invocation() == [str(launcher)]


def test_terse_mcp_cmd_override_rejects_a_path_that_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("TERSE_MCP_CMD", str(tmp_path / "nope" / "terse"))
    with pytest.raises(FileNotFoundError, match="TERSE_MCP_CMD"):
        im.terse_invocation()


def test_terse_mcp_cmd_override_allows_a_bare_name(monkeypatch):
    # A bare `terse` resolves against the launcher's PATH, which we can't know from
    # here — so it is passed through rather than false-flagged as missing.
    monkeypatch.setenv("TERSE_MCP_CMD", "terse")
    assert im.terse_invocation() == ["terse"]


def test_do_install_refuses_a_bad_override_before_touching_the_config(tmp_path, monkeypatch):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    before = cfg.read_text(encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "policies": []}))
    monkeypatch.setenv("TERSE_MCP_CMD", str(tmp_path / "gone" / "terse"))

    with pytest.raises(FileNotFoundError):
        im.do_install(["runecho"], str(policy), cfg=cfg)
    assert cfg.read_text(encoding="utf-8") == before  # config untouched


def test_scan_flags_a_wrapped_entry_whose_launcher_vanished(tmp_path, monkeypatch):
    # The upgrade case: a versioned uv-tool/pipx venv moves and every wrapped entry is
    # left pointing at an interpreter that no longer exists.
    launcher = tmp_path / "python"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(runecho={"command": "uvx", "args": ["runecho-mcp"]})))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "policies": []}))
    monkeypatch.setattr(im, "terse_invocation", lambda: [str(launcher), "-m", "terse"])
    im.do_install(["runecho"], str(policy), cfg=cfg)

    row = next(r for r in im.scan_scopes(cfg=cfg) if r["server"] == "runecho")
    assert row["state"] == "wrapped" and row["launcher_missing"] is False

    launcher.unlink()  # the upgrade moves the venv out from under the entry
    row = next(r for r in im.scan_scopes(cfg=cfg) if r["server"] == "runecho")
    assert row["launcher_missing"] is True and row["launcher"] == str(launcher)


def test_scan_never_flags_an_unwrapped_entry_or_a_bare_command(tmp_path):
    # `uvx` is a bare name on PATH, and the row isn't terse-managed anyway — neither
    # should acquire a launcher flag.
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps(_cfg(plain={"command": "uvx", "args": ["plain-mcp"]})))
    row = next(r for r in im.scan_scopes(cfg=cfg) if r["server"] == "plain")
    assert row["state"] == "unwrapped" and row["launcher_missing"] is False


# --- reading the baked proxy opts back out of a wrapped entry (#136) ---

def test_parse_proxy_opts_reads_policy_capture_and_server_between_proxy_and_sep():
    entry = {"command": "/usr/bin/python",
             "args": ["-m", "terse", "proxy", "--policy", "/p/pol.json",
                      "--server-name", "runecho", "--capture-dir", "/c/corpus",
                      "--", "runecho-mcp"]}
    assert im.parse_proxy_opts(entry) == {
        "policy": "/p/pol.json", "server_name": "runecho", "capture_dir": "/c/corpus"}


def test_parse_proxy_opts_recognizes_console_script_launcher():
    entry = {"command": "/home/u/.local/bin/terse",
             "args": ["proxy", "--policy", "/p.json", "--server-name", "kb", "--", "sb-run"]}
    assert im.parse_proxy_opts(entry) == {"policy": "/p.json", "server_name": "kb"}


def test_parse_proxy_opts_ignores_a_downstream_policy_flag():
    # A wrapped server whose OWN downstream args carry --policy must not have that value
    # misread as terse's — only the segment before `--` is terse's.
    entry = {"command": "/usr/bin/python",
             "args": ["-m", "terse", "proxy", "--policy", "/terse/pol.json",
                      "--", "weird-server", "--policy", "/downstream/other.json"]}
    assert im.parse_proxy_opts(entry) == {"policy": "/terse/pol.json"}


def test_parse_proxy_opts_returns_none_for_non_terse_entries():
    assert im.parse_proxy_opts(
        {"command": "node", "args": ["s.js", "proxy", "--policy", "x"]}) is None  # not terse
    assert im.parse_proxy_opts({"command": "/bin/foo"}) is None                    # no args
    assert im.parse_proxy_opts(
        {"command": "/usr/bin/python", "args": ["-m", "terse", "capture"]}) is None  # no proxy


def test_discover_wrapped_opts_collects_only_wrapped_in_order():
    config = {"mcpServers": {
        "a": {"command": "/usr/bin/python",
              "args": ["-m", "terse", "proxy", "--policy", "/p.json",
                       "--capture-dir", "/c", "--", "srv-a"]},
        "b": {"command": "node", "args": ["b.js"]},                 # unrelated server
        "c": {"command": "/home/u/.local/bin/terse",
              "args": ["proxy", "--policy", "/p.json", "--", "srv-c"]},
    }}
    assert im.discover_wrapped_opts(config) == [
        {"server": "a", "policy": "/p.json", "capture_dir": "/c"},
        {"server": "c", "policy": "/p.json"},
    ]


def test_discover_wrapped_opts_empty_without_mcpservers():
    assert im.discover_wrapped_opts({}) == []
    assert im.discover_wrapped_opts({"mcpServers": "not-a-dict"}) == []


def test_parse_proxy_opts_detects_uvx_and_uv_tool_run_launchers():
    # $TERSE_MCP_CMD='uvx terse' / 'uv tool run terse' bake `terse` as a bare arg token,
    # not `-m terse` — these must still be recognized or their policy silently drops out
    # of the ambiguity set (#136 review Finding 1).
    uvx = {"command": "uvx",
           "args": ["terse", "proxy", "--policy", "/p.json", "--", "gh-mcp"]}
    assert im.parse_proxy_opts(uvx) == {"policy": "/p.json"}
    uv_run = {"command": "uv",
              "args": ["tool", "run", "terse", "proxy", "--policy", "/q.json", "--", "kb"]}
    assert im.parse_proxy_opts(uv_run) == {"policy": "/q.json"}


# ---------------------------------------------------------------------------
# #172: classify from the config, not from stash membership. A wrapped entry whose
# stash record is missing used to read "unwrapped" AND be skipped by uninstall --all,
# leaving it terse-wrapped forever and invisible to terse's own tooling.
# ---------------------------------------------------------------------------

def _wrapped_entry(downstream, *, extra_args=()):
    return {"type": "stdio", "command": "/home/u/.local/bin/terse",
            "args": ["proxy", "--policy", "/home/u/.config/terse/policy.json",
                     "--", downstream, *extra_args]}


def _cfg_with(tmp_path, servers, stash):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": servers}))
    (tmp_path / im.STASH_NAME).write_text(json.dumps({"user": stash}))
    return cfg


def test_wrapped_but_unstashed_entry_is_not_reported_as_unwrapped(tmp_path):
    """The live repro: two servers wrapped identically, only one in the stash."""
    cfg = _cfg_with(
        tmp_path,
        {"runecho": _wrapped_entry("/home/u/.local/bin/runecho-mcp"),
         "codegraph": _wrapped_entry("/opt/homebrew/bin/codegraph",
                                     extra_args=("serve", "--mcp"))},
        {"runecho": {"type": "stdio", "command": "/home/u/.local/bin/runecho-mcp",
                     "args": [], "env": {}}})
    rows = {r["server"]: r for r in im.scan_scopes(cfg=cfg)}
    assert rows["runecho"]["state"] == "wrapped"
    assert rows["codegraph"]["state"] == "wrapped-unstashed"
    # the wrapped-only detail fields must be populated for BOTH, or status hides the
    # downstream of exactly the entry the operator most needs to fix by hand
    assert rows["codegraph"]["wraps"] == "/opt/homebrew/bin/codegraph serve --mcp"
    assert rows["codegraph"]["policy"] == "/home/u/.config/terse/policy.json"


def test_a_genuinely_unwrapped_entry_is_still_unwrapped(tmp_path):
    cfg = _cfg_with(tmp_path, {"plain": {"type": "stdio", "command": "/usr/bin/plain",
                                         "args": []}}, {})
    rows = {r["server"]: r for r in im.scan_scopes(cfg=cfg)}
    assert rows["plain"]["state"] == "unwrapped"
    assert rows["plain"]["wraps"] is None


def test_orphaned_stash_still_detected(tmp_path):
    """The opposite drift (#58) must keep working — stash entry, no config entry."""
    cfg = _cfg_with(tmp_path, {}, {"gone": {"type": "stdio", "command": "/usr/bin/gone"}})
    rows = {r["server"]: r for r in im.scan_scopes(cfg=cfg)}
    assert rows["gone"]["state"] == "orphaned-stash"


def test_uninstall_all_refuses_an_unstashed_entry_with_a_reason(tmp_path):
    """It must NOT silently skip: `do_uninstall` used to iterate stash keys only, so this
    server could never be unwrapped by terse at all."""
    cfg = _cfg_with(
        tmp_path,
        {"codegraph": _wrapped_entry("/opt/homebrew/bin/codegraph",
                                     extra_args=("serve", "--mcp"))},
        {})
    out = im.do_uninstall(None, all_=True, cfg=cfg, dry_run=True)
    (change,) = out["changes"]
    assert change["server"] == "codegraph" and change["restored"] is False
    assert "stash entry is missing" in change["reason"]
    assert "not managed by terse" not in change["reason"]


def test_uninstall_all_still_restores_a_properly_stashed_entry(tmp_path):
    original = {"type": "stdio", "command": "/home/u/.local/bin/runecho-mcp",
                "args": [], "env": {}}
    cfg = _cfg_with(tmp_path, {"runecho": _wrapped_entry("/home/u/.local/bin/runecho-mcp")},
                    {"runecho": original})
    out = im.do_uninstall(None, all_=True, cfg=cfg)
    assert out["changes"] == [{"server": "runecho", "restored": True}]
    assert json.loads(cfg.read_text())["mcpServers"]["runecho"] == original


# --------------------------------------------------------------- #179 --multiproxy

def _multi_cfg(tmp_path):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "kb": {"type": "stdio", "command": "kb-mcp", "args": ["--x"], "env": {"K": "1"}},
        "gh": {"url": "https://gh.example/mcp", "headers": {"Authorization": "Bearer t"}},
        "other": {"command": "other-mcp"},
    }}), encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}), encoding="utf-8")
    return cfg, pol


def test_multiproxy_folds_servers_into_one_router_entry_and_a_peers_file(tmp_path):
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    res = do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    live = json.loads(cfg.read_text())["mcpServers"]
    # the two peers are GONE from the client's view; one router entry fronts them
    assert set(live) == {"other", "terse"}
    assert "--config" in live["terse"]["args"]
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    assert [d["name"] for d in peers] == ["kb", "gh"]
    # stdio peer keeps command+args+env; http peer keeps url+headers
    assert peers[0]["command"] == ["kb-mcp", "--x"] and peers[0]["env"] == {"K": "1"}
    assert peers[1]["url"] == "https://gh.example/mcp"
    assert peers[1]["headers"] == {"Authorization": "Bearer t"}


def test_multiproxy_uninstall_all_restores_every_original_byte_for_byte(tmp_path):
    # The load-bearing case: the stash is 1:1 but a multiproxy install collapses N live
    # entries into 1, so `uninstall-mcp --all` has to restore N from a config that shows
    # only the router. Tested BEFORE the happy path, per the issue.
    from terse.install_mcp import do_install, do_uninstall
    cfg, pol = _multi_cfg(tmp_path)
    original = json.loads(cfg.read_text())
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    res = do_uninstall(None, all_=True, cfg=cfg)
    assert json.loads(cfg.read_text()) == original          # byte-for-byte, router gone
    assert {c["server"] for c in res["changes"] if c["restored"]} == {"kb", "gh"}
    assert not Path(res["peers_file"]).exists()     # no zero-peer leftover


def test_uninstalling_one_peer_detaches_it_and_leaves_the_router_serving_the_rest(tmp_path):
    from terse.install_mcp import do_install, do_uninstall
    cfg, pol = _multi_cfg(tmp_path)
    res = do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    out = do_uninstall(["kb"], cfg=cfg)
    live = json.loads(cfg.read_text())["mcpServers"]
    assert "kb" in live and "terse" in live                  # kb back, router still there
    assert out["changes"][0]["detached_from"] == "terse"
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    assert [d["name"] for d in peers] == ["gh"]              # pruned from the peers file


def test_multiproxy_print_is_a_dry_run_and_reports_the_allowlist_rewrite(tmp_path):
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    before = cfg.read_text()
    res = do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True, dry_run=True)
    assert cfg.read_text() == before                          # writes nothing
    assert not Path(res["peers_file"]).exists()
    # both real permission forms, and the widening called out — not a `mcp__kb__*` glob,
    # which is not a shape a Claude Code settings file ever holds
    assert res["allowlist"] == [
        {"server": "kb", "from": "mcp__kb", "to": "mcp__terse",
         "from_tool": "mcp__kb__<tool>", "to_tool": "mcp__terse__<tool>", "widens": True},
        {"server": "gh", "from": "mcp__gh", "to": "mcp__terse",
         "from_tool": "mcp__gh__<tool>", "to_tool": "mcp__terse__<tool>", "widens": True}]


def test_multiproxy_is_idempotent_and_never_nests_a_proxy(tmp_path):
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    res = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    # re-described from the STASHED original, so the peer command is kb-mcp, not terse
    assert peers[0]["command"] == ["kb-mcp", "--x"]


def test_multiproxy_refuses_a_router_name_that_collides_with_a_peer(tmp_path):
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    with pytest.raises(ValueError, match="also a server being wrapped"):
        do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True, router="kb")


def test_multiproxy_never_folds_a_terse_wrapped_entry_in_verbatim(tmp_path):
    # A server can be wrapped with its stash under a DIFFERENT scope, or missing (#172).
    # Folding that entry in as a peer would nest `terse proxy ... -- kb-mcp` inside the
    # router — the primer charged twice, defeating the point of consolidating.
    from terse.install_mcp import do_install
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "kb": {"type": "stdio", "command": "/usr/bin/terse", "env": {"K": "1"},
               "args": ["proxy", "--policy", "/old.json", "--server-name", "kb",
                        "--", "kb-mcp", "--flag"]},
        "gh": {"command": "/usr/bin/terse",
               "args": ["proxy", "--policy", "/old.json", "--header",
                        "Authorization=Bearer t", "--", "https://gh.example/mcp"]},
    }}), encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}), encoding="utf-8")
    res = do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    assert peers[0]["command"] == ["kb-mcp", "--flag"]     # the DOWNSTREAM, not terse
    assert peers[0]["env"] == {"K": "1"}                   # non-terse keys survive
    assert peers[1]["url"] == "https://gh.example/mcp"
    assert peers[1]["headers"] == {"Authorization": "Bearer t"}


def test_multiproxy_adds_to_the_fleet_instead_of_replacing_it(tmp_path):
    """The second invocation is almost always "fold one more server in". Overwriting the
    peers file from the argument list made that evict the earlier peers — which stay
    stashed, so their live entries are gone too: the client loses them entirely."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    res = do_install(["other"], str(pol), cfg=cfg, multiproxy=True)
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    assert [d["name"] for d in peers] == ["kb", "gh", "other"]
    assert res["fleet"] == ["kb", "gh", "other"]
    # and the allowlist rewrite covers the whole fleet, not just this run's argument
    assert [a["server"] for a in res["allowlist"]] == ["kb", "gh", "other"]
    assert set(json.loads(cfg.read_text())["mcpServers"]) == {"terse"}


def test_multiproxy_drops_a_stale_peer_that_was_already_uninstalled(tmp_path):
    """A peers entry whose server is LIVE again is stale bookkeeping — re-running must not
    resurrect it behind the router. `do_uninstall` also prunes the peers file, so the
    peers file is hand-seeded here to exercise the retention filter itself rather than
    a state `_prune_peer` already cleaned up."""
    from terse.install_mcp import do_install, do_uninstall, peers_path
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    stale = json.loads(peers_path(cfg).read_text())
    do_uninstall(["kb"], cfg=cfg)
    peers_path(cfg).write_text(json.dumps(stale), encoding="utf-8")  # kb back in the file
    res = do_install(["other"], str(pol), cfg=cfg, multiproxy=True)
    assert res["fleet"] == ["gh", "other"]
    assert "kb" in json.loads(cfg.read_text())["mcpServers"]   # still standalone


def test_multiproxy_keeps_a_peer_whose_stash_entry_drifted_away(tmp_path):
    """Staleness is LIVENESS, not stash membership. A peer whose stash entry went missing
    (#172) is the peer that needs the peers file MOST — it is the last record of how to
    launch it. Evicting it deleted the server from the stash, the live config, the peers
    file and status at once, from a run that never named it."""
    from terse.install_mcp import do_install, peers_path, stash_path
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    full = json.loads(stash_path(cfg).read_text())
    del full["user"]["kb"]                                     # stash drift
    stash_path(cfg).write_text(json.dumps(full), encoding="utf-8")
    res = do_install(["other"], str(pol), cfg=cfg, multiproxy=True)
    assert res["fleet"] == ["kb", "gh", "other"]
    kb = [d for d in json.loads(peers_path(cfg).read_text())["downstreams"]
          if d["name"] == "kb"][0]
    assert kb["command"] == ["kb-mcp", "--x"]                  # still launchable


def test_multiproxy_bakes_the_runtime_flags_onto_the_router_entry(tmp_path):
    """--capture-dir/--no-stats/--no-diff/--no-join-blocks are accepted by `terse proxy`
    with --config too, so the switch to a router must not silently drop them."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    res = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True,
                     capture_dir=str(tmp_path / "corpus"), no_stats=True, diff=False,
                     no_join_blocks=True)
    args = json.loads(cfg.read_text())["mcpServers"]["terse"]["args"]
    assert "--capture-dir" in args and args[args.index("--capture-dir") + 1] == \
        str((tmp_path / "corpus").resolve())
    assert "--no-stats" in args and "--no-diff" in args and "--no-join-blocks" in args
    assert args.index("--config") > args.index("proxy")
    assert res["capture_dir"] == str((tmp_path / "corpus").resolve())
    assert res["no_stats"] is True and res["diff"] is False


def test_multiproxy_never_lossy_still_reaches_the_policy_file(tmp_path):
    """`--never-lossy` bakes server names into the policy's never_lossy_servers. The
    runtime matches on --server-name, which multiproxy sets per PEER, so a folded
    credential server needs exactly the same protection it had standalone."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    res = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True, never_lossy=True)
    assert res["never_lossy_added"] == ["kb"]
    assert json.loads(pol.read_text())["never_lossy_servers"] == ["kb"]


def test_multiproxy_peers_file_is_scope_namespaced(tmp_path):
    """User and local scope share one ~/.claude.json. One peers file beside it would let
    a local-scope fleet silently overwrite the user-scope one — the collision
    Target.stash_prefix already prevents for the stash."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    config = json.loads(cfg.read_text())
    config["projects"] = {"/repo": {"mcpServers": {"loc": {"command": "loc-mcp"}}}}
    cfg.write_text(json.dumps(config), encoding="utf-8")
    user = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    local = do_install(["loc"], str(pol), cfg=cfg, multiproxy=True, scope="local",
                       repo_path="/repo")
    assert user["peers_file"] != local["peers_file"]
    assert [d["name"] for d in json.loads(Path(user["peers_file"]).read_text())
            ["downstreams"]] == ["kb"]
    assert [d["name"] for d in json.loads(Path(local["peers_file"]).read_text())
            ["downstreams"]] == ["loc"]


def test_uninstall_does_not_mistake_an_unrelated_config_taking_server_for_a_router(tmp_path):
    """`"--config" in args` alone matched any server whose own CLI takes a --config flag.
    Detaching the router's last peer deletes the router entry — so a false positive
    deletes a third party's server."""
    from terse.install_mcp import do_install, do_uninstall
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    live = json.loads(cfg.read_text())
    # Pointed at the REAL peers file, so only the launcher check can reject it — a decoy
    # with some other --config value is rejected by the path check alone and would leave
    # the launcher guard untested.
    peers_file = live["mcpServers"]["terse"]["args"][-1]
    live["mcpServers"]["decoy"] = {"command": "some-mcp", "args": ["--config", peers_file]}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    do_uninstall(["kb"], cfg=cfg)
    after = json.loads(cfg.read_text())["mcpServers"]
    assert "decoy" in after            # untouched
    assert "terse" not in after        # the real router went, its last peer having left


def test_status_reports_a_healthy_fleet_as_router_plus_folded_peers(tmp_path):
    """Both halves of a multiproxy install read as drift to the stash/live classifier:
    a folded peer is stashed-but-absent (orphaned-stash), the router is terse-launched
    with no stash (wrapped-unstashed, "original unrecoverable"). Both are healthy."""
    from terse.install_mcp import do_install, scan_scopes
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    rows = {r["server"]: r for r in scan_scopes(cfg=cfg) if r["scope"] == "user"}
    assert rows["terse"]["state"] == "router"
    assert rows["terse"]["wraps"] == "gh, kb"
    assert rows["terse"]["policy"] == str(pol.resolve())
    assert rows["kb"]["state"] == "folded" and rows["kb"]["router"] == "terse"
    assert rows["gh"]["state"] == "folded"
    assert rows["other"]["state"] == "unwrapped"


def test_multiproxy_refuses_a_router_name_held_by_an_unrelated_live_server(tmp_path):
    """The router entry is written over, not stashed — so a same-named live entry would be
    destroyed with nothing to restore from. `terse` is the DEFAULT name."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["terse"] = {"command": "some-other-mcp", "args": ["--serve"]}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    with pytest.raises(ValueError, match="Pick another name"):
        do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    assert json.loads(cfg.read_text())["mcpServers"]["terse"]["command"] == "some-other-mcp"


def test_multiproxy_rename_moves_the_router_instead_of_leaving_two(tmp_path):
    """Two entries running the same `proxy --config` launch every peer twice and export
    every tool twice — and `_detect_router` then sees two, returns None, and neither
    status nor `uninstall-mcp --all` can clean the config up at all."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    do_install(["gh"], str(pol), cfg=cfg, multiproxy=True, router="router2")
    live = json.loads(cfg.read_text())["mcpServers"]
    assert "terse" not in live and "router2" in live
    assert set(live) == {"other", "router2"}


def test_multiproxy_additive_run_keeps_the_routers_runtime_flags(tmp_path):
    """An additive run names the new server, not the flags the fleet was installed with.
    Rebuilding the args from this invocation alone cleared --capture-dir/--no-stats for
    every peer at once."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True,
               capture_dir=str(tmp_path / "corpus"), no_stats=True, diff=False)
    do_install(["gh"], str(pol), cfg=cfg, multiproxy=True)
    args = json.loads(cfg.read_text())["mcpServers"]["terse"]["args"]
    assert "--capture-dir" in args and "--no-stats" in args and "--no-diff" in args


def test_multiproxy_stashes_the_downstream_not_the_wrapper_it_unnested(tmp_path):
    """Folding a wrapped-but-unstashed entry used to stash the PROXY as its original, so
    `uninstall` reported restored:True while writing the wrapper back into the config."""
    from terse.install_mcp import do_install, do_uninstall
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "kb": {"command": "/usr/bin/terse", "env": {"K": "1"},
               "args": ["proxy", "--policy", "/old.json", "--server-name", "kb",
                        "--", "kb-mcp", "--flag"]}}}), encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}),
                   encoding="utf-8")
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    do_uninstall(["kb"], cfg=cfg)
    restored = json.loads(cfg.read_text())["mcpServers"]["kb"]
    assert restored["command"] == "kb-mcp" and restored["args"] == ["--flag"]
    assert restored["env"] == {"K": "1"}


def test_uninstall_all_does_not_report_the_router_as_unrecoverable(tmp_path):
    """The router is terse-launched with no stash, so `--all` filed it under #172's
    "edit the config by hand" — on the documented undo path, about an entry `unwrap`
    deletes for the operator one line later."""
    from terse.install_mcp import do_install, do_uninstall
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    res = do_uninstall(None, all_=True, cfg=cfg)
    assert [c["server"] for c in res["changes"]] == ["gh", "kb"]   # no `terse` row
    assert all(c["restored"] for c in res["changes"])


def test_plain_install_refuses_a_server_already_folded_behind_a_router(tmp_path):
    """Without the guard it comes back as a standalone wrapped entry WHILE the router
    keeps launching it: the same downstream twice, every tool exported twice."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    with pytest.raises(ValueError, match="already folded"):
        do_install(["kb"], str(pol), cfg=cfg)
    assert set(json.loads(cfg.read_text())["mcpServers"]) == {"other", "terse"}


def test_allowlist_does_not_cry_widening_for_a_single_peer_router(tmp_path):
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    one = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True, dry_run=True)
    assert [m["widens"] for m in one["allowlist"]] == [False]
    two = do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True, dry_run=True)
    assert [m["widens"] for m in two["allowlist"]] == [True, True]


def test_multiproxy_refuses_a_router_name_that_belongs_to_a_folded_peer(tmp_path):
    """A folded peer has no live entry by construction, so a `router in live` check could
    not see it. Naming the router after one made a later `unwrap` write that peer's
    original OVER the router entry — stranding every other peer (stashed, no live entry,
    no router) while reporting success."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    with pytest.raises(ValueError, match="Pick another name"):
        do_install(["other"], str(pol), cfg=cfg, multiproxy=True, router="kb")
    live = json.loads(cfg.read_text())["mcpServers"]
    assert "terse" in live and set(live) == {"other", "terse"}


def test_multiproxy_refuses_to_fold_the_router_into_its_own_peers_file(tmp_path):
    """`terse proxy --config <this very file>` as a PEER spawns a router that spawns a
    router, unbounded, at the next client restart. `_unnest` can't catch it — a router
    entry has no `--`."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    with pytest.raises(ValueError, match="cannot also be one of its own peers"):
        do_install(["terse"], str(pol), cfg=cfg, multiproxy=True, router="gw")


def test_multiproxy_rename_carries_the_routers_hand_edited_keys(tmp_path):
    """The router's `env` is the base environment every peer inherits. The rename path
    looked the base entry up under the NEW name, so it silently dropped a PATH pin — the
    same drift loss `wrap`'s guard exists to prevent."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["terse"]["env"] = {"PATH": "/opt/node22/bin:/usr/bin"}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    do_install(["gh"], str(pol), cfg=cfg, multiproxy=True, router="gateway")
    after = json.loads(cfg.read_text())["mcpServers"]
    assert "terse" not in after
    assert after["gateway"]["env"] == {"PATH": "/opt/node22/bin:/usr/bin"}


def test_multiproxy_runtime_flags_can_be_cleared_by_a_run_that_names_others(tmp_path):
    """Inheritance is all-or-nothing. `--no-stats`/`--capture-dir` have no inverse flag,
    so a per-flag `or`-merge could only ever SET them: once a measurement run wired a
    capture dir, every later install kept teeing raw payloads there with no way back
    except hand-editing the config."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True,
               capture_dir=str(tmp_path / "corpus"), no_stats=True)
    res = do_install(["gh"], str(pol), cfg=cfg, multiproxy=True, diff=True)
    args = json.loads(cfg.read_text())["mcpServers"]["terse"]["args"]
    assert "--diff" in args
    assert "--capture-dir" not in args and "--no-stats" not in args
    assert res["capture_dir"] is None and res["no_stats"] is False


def test_multiproxy_additive_run_reports_the_flags_it_actually_baked(tmp_path):
    """An additive run that INHERITED --capture-dir used to report capture_dir=None, so
    --print showed no capture line and cli's autotune follow-up hint never fired."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True,
               capture_dir=str(tmp_path / "corpus"), no_stats=True)
    res = do_install(["gh"], str(pol), cfg=cfg, multiproxy=True)
    assert res["capture_dir"] == str((tmp_path / "corpus").resolve())
    assert res["no_stats"] is True


def test_plain_install_refuses_to_wrap_the_router_itself(tmp_path):
    """`wrap` would nest `terse proxy ... -- terse proxy --config ...`, charging the
    primer twice — the exact cost --multiproxy removes — and `_detect_router` still
    matches it, so status would call the nested entry a healthy router."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    with pytest.raises(ValueError, match="nest a proxy inside a proxy"):
        do_install(["terse"], str(pol), cfg=cfg)


def test_multiproxy_coerces_a_numeric_env_value_the_client_would_have_accepted(tmp_path):
    """A client's own spawn coerces, so `{"PORT": 3000}` is a working entry today and a
    plain wrap preserves it. The router reads the peers file with load_multi_config —
    a non-string value there kills the WHOLE fleet at launch, on an install that
    reported success."""
    from terse.install_mcp import do_install
    from terse.multiproxy import load_multi_config
    cfg, pol = _multi_cfg(tmp_path)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["kb"]["env"] = {"PORT": 3000, "DEBUG": True}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    res = do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    peers = json.loads(Path(res["peers_file"]).read_text())["downstreams"]
    assert peers[0]["env"] == {"PORT": "3000", "DEBUG": "True"}
    # and the router can actually load what the installer wrote
    specs = load_multi_config(res["peers_file"])
    assert specs[0].env == {"PORT": "3000", "DEBUG": "True"}


# --------------------------------------- #179 round-4: recovery from every bad state

def test_uninstall_all_restores_a_folded_peer_whose_stash_entry_vanished(tmp_path):
    """A folded peer has no live entry by construction, so it can never be in `detected`;
    with its stash entry gone it was in neither set — silently skipped by a run that
    reported success, and absent from status too. The peers file still holds enough to
    launch it, so it is restorable."""
    from terse.install_mcp import do_install, do_uninstall, scan_scopes, stash_path
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    full = json.loads(stash_path(cfg).read_text())
    del full["user"]["kb"]
    stash_path(cfg).write_text(json.dumps(full), encoding="utf-8")
    rows = {r["server"]: r for r in scan_scopes(cfg=cfg) if r["scope"] == "user"}
    assert rows["kb"]["state"] == "folded-unstashed"        # visible at all
    res = do_uninstall(None, all_=True, cfg=cfg)
    kb_change = [c for c in res["changes"] if c["server"] == "kb"][0]
    assert kb_change["restored"] is True and kb_change["partial"] is True
    live = json.loads(cfg.read_text())["mcpServers"]
    assert live["kb"] == {"command": "kb-mcp", "args": ["--x"], "env": {"K": "1"}}
    assert "terse" not in live                              # router swept too


def test_uninstall_all_removes_the_router_when_the_peers_file_was_deleted(tmp_path):
    """The one bad state reachable with no JSON hand-edit at all. Router removal used to
    be gated on a successful prune, so with peers_doc None every original was restored,
    a clean uninstall reported, and an entry left running `terse proxy --config <missing>`
    that exits 2 on every client start."""
    from terse.install_mcp import do_install, do_uninstall, peers_path
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    peers_path(cfg).unlink()
    res = do_uninstall(None, all_=True, cfg=cfg)
    live = json.loads(cfg.read_text())["mcpServers"]
    assert "terse" not in live and {"kb", "gh"} <= set(live)
    assert [c for c in res["changes"] if c.get("router")][0]["server"] == "terse"
    assert res["peers_file"] is not None       # not "no multiproxy involved"


def test_uninstall_all_removes_the_router_when_downstreams_went_empty_or_malformed(tmp_path):
    """`downstreams: []`, a non-list, or one nameless leftover all made `_prune_peer`
    return False or never empty the list, so the router outlived its fleet. `_prune_peer`
    and `wrap_multi` also disagreed about whether a nameless entry counts."""
    from terse.install_mcp import do_install, do_uninstall, peers_path
    for i, downs in enumerate(([], "nope", [{"policy": "/p.json"}])):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        cfg, pol = _multi_cfg(sub)
        do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
        peers_path(cfg).write_text(json.dumps({"downstreams": downs}), encoding="utf-8")
        do_uninstall(None, all_=True, cfg=cfg)
        live = json.loads(cfg.read_text())["mcpServers"]
        assert "terse" not in live, downs
        assert {"kb", "gh"} <= set(live), downs


def test_a_corrupt_peers_file_never_tracebacks_and_always_names_the_path(tmp_path):
    """It used to raise JSONDecodeError out of `mcp-status` (whose contract says it never
    raises) and to block install-mcp, --multiproxy, uninstall-mcp and --all with a message
    naming no file — every route out of the state closed, nothing saying which file."""
    from terse.install_mcp import do_install, do_uninstall, peers_path, scan_scopes
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    peers_path(cfg).write_text("{ not json", encoding="utf-8")
    rows = scan_scopes(cfg=cfg)                              # must not raise
    assert any("unreadable peers file" in (r.get("peers_error") or "") for r in rows)
    err = "unreadable peers file"
    with pytest.raises(ValueError, match=err):
        do_uninstall(None, all_=True, cfg=cfg)
    with pytest.raises(ValueError, match=err):
        do_uninstall(["kb"], cfg=cfg)
    with pytest.raises(ValueError, match=err):
        do_install(["other"], str(pol), cfg=cfg)
    with pytest.raises(ValueError, match=err):
        do_install(["other"], str(pol), cfg=cfg, multiproxy=True)


def test_peers_path_does_not_collide_for_repo_paths_that_slugify_alike(tmp_path):
    """The disambiguating hash was applied only past 40 chars, but the collision comes
    from slugification: `/home/e/a/b` and `/home/e/a-b` both become `local-home-e-a-b`.
    Two repos then shared one peers file — repo 1's router launching repo 2's servers."""
    from terse.install_mcp import peers_path
    cfg = tmp_path / "claude.json"
    a = peers_path(cfg, "local:/home/e/a/b")
    b = peers_path(cfg, "local:/home/e/a-b")
    c = peers_path(cfg, "local:/home/e/my project")
    d = peers_path(cfg, "local:/home/e/my-project")
    assert len({a, b, c, d}) == 4
    assert peers_path(cfg, "user") == peers_path(cfg, "user")     # still deterministic


def test_two_routers_on_one_peers_file_are_named_not_guessed_at(tmp_path):
    """Uninstall used to prune every peer, fail to remove either router (detection returns
    None), then DELETE the peers file both entries depend on — and the error message's own
    `--router-name` advice added a third router, poisoning detection permanently."""
    from terse.install_mcp import do_install, do_uninstall, peers_path, scan_scopes
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["terse2"] = dict(live["mcpServers"]["terse"])
    cfg.write_text(json.dumps(live), encoding="utf-8")
    states = {r["server"]: r["state"] for r in scan_scopes(cfg=cfg) if r["scope"] == "user"}
    assert states["terse"] == states["terse2"] == "router-ambiguous"
    res = do_uninstall(None, all_=True, cfg=cfg)
    assert sorted(res["router_ambiguous"]) == ["terse", "terse2"]
    assert peers_path(cfg).exists()                    # NOT deleted out from under them
    with pytest.raises(ValueError, match="all front"):
        do_install(["other"], str(pol), cfg=cfg, multiproxy=True, router="terse3")


def test_status_flags_a_peer_that_is_both_folded_and_live(tmp_path):
    """`claude mcp add <name>` (or a hand-edit) can re-add an entry the operator folded.
    The same downstream then runs twice, every tool exported twice at double cost, and
    the old classification called it a plain `wrapped` entry."""
    from terse.install_mcp import do_install, scan_scopes
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["kb"] = {"command": "kb-mcp", "args": ["--x"]}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    rows = {r["server"]: r for r in scan_scopes(cfg=cfg) if r["scope"] == "user"}
    assert rows["kb"]["state"] == "folded-and-live" and rows["kb"]["router"] == "terse"


def test_detaching_the_last_peer_removes_the_router_despite_a_malformed_leftover(tmp_path):
    """The single-peer path has no `--all` sweep, so `unwrap`'s own gate is what removes
    the router. Gating it on "this server's prune fired AND downstreams is falsy" left the
    router alive whenever one nameless entry kept the list non-empty — an entry running
    `terse proxy --config <no usable peers>`, which exits 2 on every client start."""
    from terse.install_mcp import do_install, do_uninstall, peers_path
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    doc = json.loads(peers_path(cfg).read_text())
    doc["downstreams"].append({"policy": "/p.json"})          # nameless leftover
    peers_path(cfg).write_text(json.dumps(doc), encoding="utf-8")
    do_uninstall(["kb"], cfg=cfg)
    live = json.loads(cfg.read_text())["mcpServers"]
    assert "terse" not in live and "kb" in live
    assert not peers_path(cfg).exists()


def test_folding_a_server_with_a_malformed_env_fails_with_a_clear_message(tmp_path):
    """A hand-edited `env` of the wrong shape used to reach `.items()` and crash with a
    bare AttributeError naming no server and no file; a container value was `str()`-ed
    into a garbage variable like `K="['x']"` that the peer read as meaningful."""
    from terse.install_mcp import do_install
    for i, (bad, msg) in enumerate((("string-env", "non-object 'env'"),
                                    (["a"], "non-object 'env'"),
                                    ({"K": ["x"]}, "are not scalars"),
                                    ({"K": {"n": 1}}, "are not scalars"))):
        sub = tmp_path / f"env{i}"
        sub.mkdir()
        cfg, pol = _multi_cfg(sub)
        live = json.loads(cfg.read_text())
        live["mcpServers"]["kb"]["env"] = bad
        cfg.write_text(json.dumps(live), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)


def test_a_peers_record_that_cannot_launch_anything_is_not_restored_as_a_broken_entry(
        tmp_path):
    """`peers_downstreams` only checks for a `name`, so a hand-edited record with no
    `command`/`url` reached the rebuild path and wrote `{"url": null}` into the live
    config — an entry no client can launch, reported as a successful restore."""
    from terse.install_mcp import (
        _entry_from_peer_spec,
        do_install,
        do_uninstall,
        peers_path,
        stash_path,
    )
    assert _entry_from_peer_spec({"name": "x"}) is None
    assert _entry_from_peer_spec({"name": "x", "command": []}) is None
    assert _entry_from_peer_spec({"name": "x", "url": None}) is None
    cfg, pol = _multi_cfg(tmp_path)
    do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    peers_path(cfg).write_text(json.dumps(
        {"downstreams": [{"name": "kb", "policy": str(pol)}]}), encoding="utf-8")
    full = json.loads(stash_path(cfg).read_text())
    del full["user"]["kb"]
    stash_path(cfg).write_text(json.dumps(full), encoding="utf-8")
    res = do_uninstall(None, all_=True, cfg=cfg)
    kb = [c for c in res["changes"] if c["server"] == "kb"][0]
    assert kb["restored"] is False
    assert "kb" not in json.loads(cfg.read_text())["mcpServers"]


# ------------------------------------------- #179 round-5: install-path review findings

def test_multiproxy_writes_recovery_data_before_deleting_the_live_entries(tmp_path,
                                                                         monkeypatch):
    """`wrap_multi` DELETES a folded peer's live entry rather than rewriting it the way
    `wrap` does, and the three writes are not atomic together. Config-first left a window
    where the live entry was already gone while the stash still described the old state:
    the original existed NOWHERE terse looks, so status reported nothing missing and
    `uninstall --all` never mentioned the server. Recovery data must land first."""
    from terse import install_mcp as im
    cfg, pol = _multi_cfg(tmp_path)
    original = json.loads(cfg.read_text())
    real = im._write_json
    calls: list[str] = []

    def failing(path, obj, **kw):
        calls.append(Path(path).name)
        if len(calls) == 2:                       # die mid-sequence
            raise OSError("disk full")
        real(path, obj, **kw)

    monkeypatch.setattr(im, "_write_json", failing)
    with pytest.raises(OSError):
        im.do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    # the client config is the LAST write, so it is untouched and both servers still work
    assert json.loads(cfg.read_text()) == original
    assert calls[0] == ".terse-mcp-stash.json"


def test_multiproxy_does_not_fold_the_same_server_twice_from_a_duplicated_argument(
        tmp_path):
    """`servers` comes from a `nargs="+"` positional, so `install-mcp kb kb` is one typo
    away — and it appended the same peer to `downstreams` twice, launching the downstream
    twice with every tool exported twice."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    res = do_install(["kb", "kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    assert res["fleet"] == ["kb", "gh"]


def test_multiproxy_refuses_to_fold_a_router_belonging_to_a_DIFFERENT_peers_file(tmp_path):
    """`_unnest` recovers a downstream from the `--` a wrapped entry carries; a router has
    `--config` and no `--`, so it came through VERBATIM and became a peer whose command is
    `terse proxy --config <other fleet>` — a proxy nested in a proxy. The prior guard only
    covered the router for THIS peers file."""
    from terse.install_mcp import do_install
    cfg, pol = _multi_cfg(tmp_path)
    live = json.loads(cfg.read_text())
    live["mcpServers"]["othermux"] = {
        "command": "/usr/bin/terse",
        "args": ["proxy", "--config", str(tmp_path / "other-fleet.json")]}
    cfg.write_text(json.dumps(live), encoding="utf-8")
    with pytest.raises(ValueError, match="nest a proxy inside a proxy"):
        do_install(["othermux", "kb"], str(pol), cfg=cfg, multiproxy=True)
    assert "othermux" in json.loads(cfg.read_text())["mcpServers"]


def test_prune_peer_normalizes_away_malformed_entries(tmp_path):
    """Pinned directly, not through a caller: every caller re-normalizes via
    `peers_downstreams`, so reverting `_prune_peer` alone regressed silently with the
    suite green. `downstreams` must not keep a nameless leftover that would strand the
    router entry forever."""
    from terse.install_mcp import _prune_peer
    doc = {"downstreams": [{"name": "kb", "command": ["kb-mcp"]}, {"policy": "/p.json"},
                           "junk", {"name": "gh", "url": "https://x"}]}
    assert _prune_peer(doc, "kb") is True
    assert doc["downstreams"] == [{"name": "gh", "url": "https://x"}]
    assert _prune_peer(doc, "nope") is False
    assert doc["downstreams"] == [{"name": "gh", "url": "https://x"}]   # still normalized


# --------------------------------------------------------------------------- #
# #181: `mcp-status` must resolve the diff setting, not just name it
# --------------------------------------------------------------------------- #
def test_default_diff_label_resolves_against_the_builtin_default():
    """A bare "default" reads as "the feature's normal state, i.e. on". #170 made it off, so
    that label actively misled a reader into concluding diffing was never implemented."""
    from terse.install_mcp import _default_diff_label
    assert _default_diff_label(None) == "default (off)"


def test_default_diff_label_tracks_the_dataclass_instead_of_a_copied_constant():
    """Derived from `Policy.diff`, so a future flip of the default cannot leave this label
    asserting the opposite of what the proxy does (the #144 failure mode)."""
    from dataclasses import fields

    from terse import policy as P
    from terse.install_mcp import _default_diff_label
    on = next(f.default for f in fields(P.Policy) if f.name == "diff")
    assert _default_diff_label(None) == f"default ({'on' if on else 'off'})"


@pytest.mark.parametrize("value, expected", [(True, "policy (on)"), (False, "policy (off)")])
def test_default_diff_label_prefers_the_entrys_own_policy_file(tmp_path, value, expected):
    from terse.install_mcp import _default_diff_label
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "diff": value}), encoding="utf-8")
    assert _default_diff_label(str(pol)) == expected


@pytest.mark.parametrize("body", ["{ not json", json.dumps({"version": 1})])
def test_default_diff_label_falls_back_when_the_policy_cannot_answer(tmp_path, body):
    """Malformed, or simply silent on `diff` — either way the built-in default is the truth,
    and an unreadable policy must not crash `mcp-status`."""
    from terse.install_mcp import _default_diff_label
    pol = tmp_path / "p.json"
    pol.write_text(body, encoding="utf-8")
    assert _default_diff_label(str(pol)) == "default (off)"
    assert _default_diff_label("/nonexistent/absent.json") == "default (off)"


def _agg_with(diff_reasons):
    from terse.stats import aggregate, build_record
    recs = []
    for reason, n in diff_reasons.items():
        for _ in range(n):
            r = build_record("s", "t", '{"a":1}', '{"a":1}', passthrough=False)
            r["diff_reason"] = reason
            recs.append(r)
    return aggregate(recs)


def test_stats_explains_diff_off_when_it_is_the_only_reason():
    """The reader's question ("why did my repeat call not diff?") is asked at this line, so
    the answer belongs here rather than only in the policy dataclass (#181)."""
    from terse.stats import build_stats_report
    out = build_stats_report(_agg_with({"diff_off": 3}), log_path="/x/s.jsonl")
    assert "OFF by default since #170" in out
    assert "--diff" in out


def test_stats_does_not_explain_diff_off_when_diffing_is_actually_working():
    """A session with real diff activity does not need the explainer, and printing it there
    would imply diffing is disabled when it plainly is not."""
    from terse.stats import build_stats_report
    out = build_stats_report(_agg_with({"diff_off": 3, "emitted": 1}), log_path="/x/s.jsonl")
    assert "OFF by default since #170" not in out
