"""`install-mcp --print` states what it is CHANGING, not just what it will write.

#277 (split out of #275, whose second ask this is). `--print` rendered a `before:`/`after:`
pair per server, but three shapes of change were invisible in it:

  * **An already-folded peer showed `before: (absent)`.** Its entry legitimately no longer
    exists standalone — it lives in the peers file — so a run that rewrites its launcher
    looked like a first-time install rather than a change.
  * **The router's own command had no `before:` at all.** `router: <name> -> <command>`
    printed the new value only, and the router is the entry every folded peer is reached
    through.
  * **`_short_cmd` truncated at 100 chars** with no marker, so a long policy path pushed
    the launcher off the end and a reader diffing two lines by eye could not tell a short
    command from a cut one.

Why it matters: an `install-mcp` run intended to refresh a policy can silently rewrite
`command` on every managed entry, and that failure is silent by construction — the MCP
client cannot spawn a bad entry, so the server appears with no tools and nothing says why,
days later. The distance between the config change and the symptom is the whole reason
#275 was hard to diagnose. Disclosure is the cheap half of that fix.
"""

import json
import os
from pathlib import Path

import pytest

from terse import cli


@pytest.fixture(autouse=True)
def _never_touch_the_real_config(tmp_path, monkeypatch):
    """Isolation, then a canary that proves the isolation held.

    An earlier draft of this file drove the CLI with `--file <tmp>` at the DEFAULT scope.
    `--file` is honoured only for `--scope project` (#366), so a non-dry-run setup call
    reached the developer's live `~/.claude.json` and rewrote the router's `command` to a
    pytest temp binary that pytest then deleted — killing the real router and every peer
    behind it. That is the exact failure #277 exists to disclose, caused while testing the
    disclosure.

    So: `HOME` is redirected (the peers file and stash are derived from the config's
    parent directory, which for user scope is `$HOME`), and every test asserts afterwards
    that nothing under the real home moved.

    The real home is resolved BEFORE the redirect. A first version resolved it after, so
    `Path("~").expanduser()` read the already-patched `$HOME` and the canary watched a
    tmp path that never existed — `None == None` on every run, silent about the one
    failure it was written for (#277 review).

    It watches a SET, not one file. A user-scope run writes `~/.terse-mcp-stash.json`, a
    `~/.terse-peers-user-<hash>.json`, and a timestamped `~/.claude.json.bak-<ts>`; that
    last one changes no mtime on `~/.claude.json` at all, so a backup-only clobber would
    pass a single-file check.

    `~/.claude.json` is watched by CONTENT of its `mcpServers` subtree, not by mtime: a
    live Claude Code session rewrites that file continuously for its own bookkeeping
    (`promptQueueUseCount`, `pluginUsage`), so an mtime check there fails for reasons that
    have nothing to do with these tests. The subtree is exactly what `install-mcp`
    writes."""
    real_home = Path(os.environ.get("HOME") or Path.home())

    def snapshot() -> set:
        out = set()
        for p in real_home.glob(".terse-*"):
            try:
                st = p.stat()
                out.add((p.name, st.st_mtime_ns, st.st_size))
            except OSError:
                out.add((p.name, None, None))
        cfg = real_home / ".claude.json"
        try:
            servers = json.loads(cfg.read_text()).get("mcpServers")
            out.add(("<mcpServers>", json.dumps(servers, sort_keys=True)))
        except (OSError, ValueError):
            out.add(("<mcpServers>", None))
        return out

    before = snapshot()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG", raising=False)
    yield
    after = snapshot()
    assert after == before, (
        f"a test touched the real home {real_home}: "
        f"added/changed {sorted(after - before)}, removed {sorted(before - after)}")


def _cfg(tmp_path: Path, servers: dict) -> tuple[Path, Path]:
    """A PROJECT-scope target. `peers_path` and `stash_path` are both derived from the
    config's parent, so putting the config under `tmp_path` keeps all three artifacts —
    config, peers file, stash — inside the sandbox."""
    cfg = tmp_path / "proj" / ".mcp.json"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}),
                   encoding="utf-8")
    return cfg, pol


@pytest.fixture
def launcher(tmp_path: Path):
    """Two real, executable launchers, both BASENAMED `terse`, in different directories.

    The basename matters: `_looks_like_terse_launcher` requires exactly `terse`, so a
    fixture named `terse-old` is not recognised as a router at all and a re-run refuses
    with "already a server terse manages". Differing only in directory is also the shape
    the real change takes — a pyenv/asdf shim, a uv tool venv, a `~/.local/bin` install —
    which is the case #275/#277 are about."""
    made = []
    for name in ("old", "new"):
        d = tmp_path / name
        d.mkdir()
        p = d / "terse"
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
        made.append(p)
    return made


def _real_install(cfg, pol, servers, *, multiproxy=False) -> None:
    """A non-dry-run install, always PROJECT scope so every artifact stays in the
    sandbox."""
    args = ["install-mcp", *servers, "--policy", str(pol),
            "--scope", "project", "--file", str(cfg)]
    if multiproxy:
        args.append("--multiproxy")
    assert cli.main(args) == 0


def _print_install(capsys, monkeypatch, cfg, pol, servers, *, cmd, **kw) -> str:
    monkeypatch.setenv("TERSE_MCP_CMD", str(cmd))
    args = ["install-mcp", *servers, "--policy", str(pol),
            "--scope", "project", "--file", str(cfg), "--print"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}"] + ([str(v)] if v is not True else [])
    cli.main(args)
    return capsys.readouterr().out


def test_a_launcher_rewrite_on_a_wrapped_entry_is_called_out(tmp_path, capsys, monkeypatch,
                                                             launcher):
    """The core ask: run --print with a different launcher over entries terse already
    manages, and the output must state the OLD command for every entry whose `command`
    changes — not leave the reader to diff two truncated strings by eye."""
    old, new = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    # Set TERSE_MCP_CMD explicitly rather than leaning on _print_install's side effect —
    # reordering these two lines otherwise broke the test for a reason nobody would find.
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb"])
    capsys.readouterr()

    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=new)
    assert "command CHANGED" in out, out
    assert f"from: {old}" in out, out
    assert f"to:   {new}" in out, out


def test_an_unchanged_launcher_is_not_announced_as_a_change(tmp_path, capsys, monkeypatch,
                                                            launcher):
    """The other direction. A disclosure that fires on every run teaches the reader to
    ignore it, which is the same as not having it."""
    old, _ = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb"])
    capsys.readouterr()
    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=old)
    assert "command CHANGED" not in out, out


def test_an_already_folded_peer_shows_its_real_before_not_absent(tmp_path, capsys,
                                                                 monkeypatch, launcher):
    """A peer folded by an earlier run has no live entry by construction. Reading only
    `mcpServers` reported `(absent)`, so a launcher rewrite on it was invisible."""
    old, new = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]},
                               "gh": {"command": "gh-mcp", "args": []}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb", "gh"], multiproxy=True)
    capsys.readouterr()

    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb", "gh"], cmd=new,
                         multiproxy=True)
    assert "before: (absent)" not in out, out
    assert "from peers file" in out, out
    assert "kb-mcp --x" in out, "the peer's real prior command is not shown"


def test_the_router_entry_gets_a_before_and_after_pair(tmp_path, capsys, monkeypatch,
                                                       launcher):
    """`router: <name> -> <command>` printed the new value with no counterpart. The
    router is the entry every folded peer is reached through, so a rewrite of its
    launcher breaks the whole fleet at once."""
    old, new = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb"], multiproxy=True)
    capsys.readouterr()

    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=new, multiproxy=True)
    router = out.split("router:")[1]
    assert "before:" in router, router
    assert "after:" in router, router
    assert str(old) in router, "the router's prior launcher is not stated"
    assert "command CHANGED" in router, router


def test_a_first_time_router_still_reports_absent_rather_than_inventing_a_before(
        tmp_path, capsys, monkeypatch, launcher):
    """`(absent)` is the CORRECT reading when there genuinely was no prior entry. The fix
    must not manufacture a before-state for a first install."""
    old, _ = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=old, multiproxy=True)
    router = out.split("router:")[1]
    assert "before: (absent)" in router, router
    assert "command CHANGED" not in router, router


def test_a_truncated_command_is_marked_as_truncated(tmp_path, capsys, monkeypatch):
    """A bare 100-char slice ended mid-token and read as the whole value, so a reader
    could not tell a short command from a cut one — and the differing part may be the
    half that was cut."""
    entry = {"command": "/x/terse", "args": ["proxy", "--policy", "/" + "d" * 200]}
    rendered = cli._short_cmd(entry)
    assert len(rendered) == 100
    assert rendered.endswith("…"), rendered
    assert cli._short_cmd({"command": "short", "args": []}) == "short"
    assert "…" not in cli._short_cmd({"command": "short", "args": []})


def test_the_change_lines_print_the_raw_field_not_the_truncated_render(tmp_path):
    """`command CHANGED` prints the raw field, not the truncated render — otherwise the
    line that exists to make the change legible would itself cut it off."""
    long_old = "/" + "a" * 300 + "/terse"
    long_new = "/" + "b" * 300 + "/terse"
    lines = cli._entry_change_lines({"command": long_old, "args": []},
                                    {"command": long_new, "args": []})
    assert any(long_old in ln for ln in lines), lines
    assert any(long_new in ln for ln in lines), lines


def test_a_url_entry_change_is_called_out_too(tmp_path):
    """The url/headers shape (#5 HTTP downstream) has no `command`; its endpoint moving
    is the same class of silent, fatal change."""
    lines = cli._entry_change_lines({"url": "https://a.example/mcp"},
                                    {"url": "https://b.example/mcp"})
    assert any("url CHANGED" in ln for ln in lines), lines
    assert cli._entry_change_lines({"url": "https://a.example/mcp"},
                                   {"url": "https://a.example/mcp"}) == []


def test_no_change_lines_when_either_side_is_missing():
    """A first install has no before, an unfold has no after. Neither is a `CHANGED`."""
    assert cli._entry_change_lines(None, {"command": "x"}) == []
    assert cli._entry_change_lines({"command": "x"}, None) == []


def test_a_moved_policy_path_is_called_out_even_when_the_launcher_is_the_same(
        tmp_path, capsys, monkeypatch, launcher):
    """The issue's framing — `command` is "the one field whose change is both invisible
    and fatal" — is wrong (#277 review). A moved `--policy` path exits 2 at spawn
    (`proxy: [Errno 2] No such file or directory`), which reaches the operator as the
    identical symptom: a server with no tools. It is MORE likely to be hidden, because the
    launcher sits at the head of the rendered line and the policy path sits past the
    100-char cut, so the before/after pair can be byte-identical on screen."""
    old, _ = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb"])
    capsys.readouterr()

    pol2 = tmp_path / ("policy-" + "d" * 90 + ".json")
    pol2.write_text(pol.read_text(), encoding="utf-8")
    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=old, policy=str(pol2))

    before_line = next(ln for ln in out.splitlines() if "before:" in ln)
    after_line = next(ln for ln in out.splitlines() if "after:" in ln)
    assert before_line.split("before:")[1].strip() == after_line.split("after:")[1].strip(), (
        "fixture no longer reproduces the truncation collision", before_line, after_line)
    assert "args CHANGED" in out, out
    assert str(pol) in out and str(pol2) in out, "the moved paths are not both stated"


def test_the_router_before_follows_a_rename(tmp_path, capsys, monkeypatch, launcher):
    """`router_before` reads `current_router or router`, and reading plain `router`
    survived the whole suite (#277 review). Renaming the router is the documented remedy
    for several install-time errors, so a rename plus a launcher rewrite is a real
    combination — and it is the one entry every folded peer is reached through."""
    old, new = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb"], multiproxy=True)
    capsys.readouterr()

    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb"], cmd=new,
                         multiproxy=True, router_name="gateway")
    router = out.split("router:")[1]
    assert "gateway" in router, router
    assert "before: (absent)" not in router, (
        "the renamed router lost its before-state — reading `router` instead of "
        "`current_router or router`")
    assert str(old) in router, router
    assert "command CHANGED" in router, router


def test_a_hand_edited_peers_file_does_not_abort_the_disclosure(tmp_path, capsys,
                                                                monkeypatch, launcher):
    """`peers_downstreams` validates only `name` and documents malformed entries as
    expected input. A non-string `command` raised TypeError out of `" ".join`, aborting
    --print after one line — destroying the whole disclosure on exactly the state where
    the operator most needs it (#277 review)."""
    from terse.install_mcp import peers_path
    old, new = launcher
    cfg, pol = _cfg(tmp_path, {"kb": {"command": "kb-mcp", "args": ["--x"]},
                               "gh": {"command": "gh-mcp", "args": []}})
    monkeypatch.setenv("TERSE_MCP_CMD", str(old))
    _real_install(cfg, pol, ["kb", "gh"], multiproxy=True)
    capsys.readouterr()

    pf = peers_path(cfg, "project")
    doc = json.loads(pf.read_text())
    for d in doc["downstreams"]:
        if d["name"] == "kb":
            d["command"] = [["/opt/kb/bin", "kb-mcp"], "--x"]   # hand-edit, nested list
    pf.write_text(json.dumps(doc), encoding="utf-8")

    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb", "gh"], cmd=new,
                         multiproxy=True)
    assert "router:" in out, "the disclosure aborted before the router block"
    assert "peers:" in out, out
    # And it must not claim a provenance for a value it never recovered: the malformed
    # record yields no entry, so this peer is honestly `(absent)`, not "(from peers file)
    # (absent)".
    kb_block = out.split("would wrap kb:")[1].split("would wrap")[0]
    assert "(absent)" in kb_block, kb_block
    assert "from peers file" not in kb_block, kb_block


def test_a_hand_edited_live_entry_does_not_abort_the_disclosure(tmp_path, capsys,
                                                                monkeypatch, launcher):
    """The same crash reachable from the LIVE config rather than the peers file — this one
    predates #277 (`" ".join` over a non-string `command`), and `--print` is exactly the
    command an operator reaches for when a config looks wrong."""
    old, _ = launcher
    # Both halves malformed: a non-string `command` reaches `" ".join`, and a non-string
    # ARG reaches `_redact_args`, which splits each token on "=". Guarding only one moves
    # the crash a frame rather than removing it — which is what happened on the first
    # attempt, twice.
    cfg, pol = _cfg(tmp_path, {"kb": {"command": ["/opt/kb/bin", "kb-mcp"],
                                      "args": [["--x"], 7]},
                               "gh": {"command": "gh-mcp", "args": []}})
    out = _print_install(capsys, monkeypatch, cfg, pol, ["kb", "gh"], cmd=old)
    assert "would wrap gh" in out, "the disclosure aborted on the malformed entry"
    assert "config:" in out, out


def test_a_malformed_arg_survives_the_change_lines_too(tmp_path):
    """`_entry_change_lines` runs its own `_redact_args`, so it needs the same coercion —
    a mutation removing it survived a suite that only had a malformed COMMAND."""
    lines = cli._entry_change_lines({"command": "t", "args": [["nested"], 7]},
                                    {"command": "t", "args": ["proxy", "--policy", "/a"]})
    assert any("args CHANGED" in ln for ln in lines), lines


def test_an_added_or_removed_arg_is_reported_not_just_a_substitution(tmp_path):
    """Zipping the two lists pairwise reports only substitutions. A flag appearing or
    disappearing — `--no-stats`, `--capture-dir` — changes what the server does and would
    otherwise be announced as `args CHANGED` with nothing under it."""
    lines = cli._entry_change_lines({"command": "t", "args": ["proxy", "--policy", "/a"]},
                                    {"command": "t", "args": ["proxy", "--policy", "/a",
                                                              "--no-stats"]})
    assert any("args CHANGED" in ln for ln in lines), lines
    assert any("3 arg(s) -> 4" in ln for ln in lines), lines
