"""A scope flag that does not apply to the active scope is refused, not ignored.

#366. `resolve_target` read `--file` only for project scope and `--repo-path` only for
local, and silently dropped either one elsewhere. So `install-mcp kb --file /tmp/x.json`
— at the DEFAULT scope, which is `user` — named one file, created no `/tmp/x.json`, said
nothing about it, and rewrote `~/.claude.json`.

That is documented behaviour (`--file` is helped as "--scope project: ..."), which is
what made it dangerous rather than merely surprising: a flag that accepts a path, ignores
it, and then writes somewhere else reads as correct right up until it isn't.

Not theoretical. It fired during #277's own test development, rewriting a live router's
`command` to a pytest temp binary that pytest then deleted — killing that router and all
three peers behind it until it was noticed and repaired by hand. The symptom is a server
with no tools, days later, which is exactly the class of failure #277 exists to disclose.
"""

import json
import subprocess
from pathlib import Path

import pytest

from terse import cli
from terse.install_mcp import resolve_target


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A sandboxed home holding a config that must survive every test in this file."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("CLAUDE_CONFIG", raising=False)
    cfg = h / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"kb": {"command": "kb-mcp"}}}),
                   encoding="utf-8")
    return h


@pytest.fixture
def policy(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}),
                 encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# resolve_target — the one place scope and its flags meet
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scope", ["user", "local"])
def test_file_outside_project_scope_is_refused(scope):
    with pytest.raises(ValueError, match=r"--file applies only to --scope project"):
        resolve_target(scope, file="/tmp/x.json")


@pytest.mark.parametrize("scope", ["user", "project"])
def test_repo_path_outside_local_scope_is_refused(scope):
    with pytest.raises(ValueError, match=r"--repo-path applies only to --scope local"):
        resolve_target(scope, repo_path="/some/repo")


def test_the_refusal_names_the_file_it_would_otherwise_have_written(tmp_path):
    """The point of the message. "This flag is ignored" does not convey the danger;
    "ignoring it would have written THAT file instead" does."""
    other = tmp_path / "elsewhere.json"
    with pytest.raises(ValueError) as e:
        resolve_target("user", cfg=other, file="/tmp/x.json")
    assert str(other) in str(e.value), str(e.value)


def test_the_message_never_prints_a_placeholder_for_the_target(tmp_path):
    """The `else "?"` arm of a single shared message template reached the user as a
    literal question mark at project scope, and survived the mutation sweep because only
    the `--file`/user pair was ever exercised (#366 review)."""
    for scope, kw in (("user", {"file": "/tmp/x.json"}),
                      ("local", {"file": "/tmp/x.json"}),
                      ("project", {"repo_path": "/some/repo"}),
                      ("user", {"repo_path": "/some/repo"})):
        with pytest.raises(ValueError) as e:
            resolve_target(scope, cfg=tmp_path / "c.json", **kw)
        assert "?" not in str(e.value), (scope, kw, str(e.value))


def test_the_repo_path_message_names_a_key_not_a_different_file(tmp_path):
    """`--repo-path` does not name a write target — user and local scope write the SAME
    physical file, differing only in whether the entry lands at the top level or under
    `projects.<key>`. The shared template asserted a difference between two identical
    paths."""
    cfg = tmp_path / "c.json"
    assert resolve_target("user", cfg=cfg).cfg == \
        resolve_target("local", cfg=cfg, repo_path="/r").cfg, "premise changed"
    with pytest.raises(ValueError) as e:
        resolve_target("user", cfg=cfg, repo_path="/r")
    msg = str(e.value)
    assert "top level" in msg and "projects." in msg, msg
    assert "instead of the file" not in msg, "still claims a different file"


def test_the_project_scope_repo_path_message_names_the_file_it_would_write(tmp_path):
    """At project scope the would-be target IS knowable — it is the `.mcp.json` — so the
    message states it rather than a placeholder, and says plainly that the key
    `--repo-path` names lives in a different file altogether."""
    with pytest.raises(ValueError) as e:
        resolve_target("project", cfg=tmp_path / "c.json", file=None, repo_path="/r")
    msg = str(e.value)
    assert ".mcp.json" in msg, msg
    assert "different file entirely" in msg, msg


def test_each_flag_is_accepted_by_the_scope_it_belongs_to(tmp_path):
    """The guard must not break correct usage."""
    assert resolve_target("project", file=str(tmp_path / "a.mcp.json")).cfg == \
        (tmp_path / "a.mcp.json").resolve()
    t = resolve_target("local", cfg=tmp_path / "c.json", repo_path="/r")
    assert t.server_path == ("projects", "/r")


def test_neither_flag_is_required(tmp_path):
    """Every scope still resolves with no scope flag at all.

    `local` is given an explicit `repo_path` here on purpose. Omitting it falls through to
    `default_repo_path()`, which shells out to `git rev-parse` against the PROCESS cwd —
    not `tmp_path` — so this became the only test in the suite that required the checkout
    to be a git repo, and it failed on an sdist/tarball or a `.git`-less Docker build
    (#366 review). The git-backed default has its own test below, skipped off a repo the
    way the two pre-existing tests of that resolver already are."""
    for scope, kw in (("user", {}), ("project", {}), ("local", {"repo_path": "/r"})):
        assert resolve_target(scope, cfg=tmp_path / "c.json", **kw).cfg is not None


@pytest.mark.skipif(subprocess.run(["git", "rev-parse", "--git-common-dir"],
                                   capture_output=True).returncode != 0,
                    reason="local scope resolves its default --repo-path from git")
def test_local_scope_without_repo_path_resolves_it_from_git(tmp_path):
    """The branch the test above deliberately stopped exercising, kept but isolated so a
    non-git checkout skips it rather than failing."""
    t = resolve_target("local", cfg=tmp_path / "c.json")
    assert t.server_path[0] == "projects" and t.server_path[1]


def test_an_unknown_scope_is_still_rejected_first(tmp_path):
    """The scope check runs before the flag check, so a bad scope reports the bad scope
    rather than complaining about a flag that could never have applied."""
    with pytest.raises(ValueError, match="unknown scope"):
        resolve_target("nonsense", file="/tmp/x.json")


# --------------------------------------------------------------------------- #
# through the CLI, on the exact command that caused the incident
# --------------------------------------------------------------------------- #

def test_install_mcp_file_at_the_default_scope_exits_2_and_writes_nothing(
        home, policy, tmp_path, capsys):
    """The reported bug, end to end. Before: rc 0, `~/.claude.json` rewritten, no
    `/tmp/x.json`, nothing said."""
    cfg = home / ".claude.json"
    before = cfg.read_bytes()
    named = tmp_path / "x.json"

    rc = cli.main(["install-mcp", "kb", "--policy", str(policy), "--file", str(named)])

    assert rc == 2
    assert cfg.read_bytes() == before, "the real config was written despite the refusal"
    assert not named.exists(), "the named file was not created either"
    err = capsys.readouterr().err
    assert "--file applies only to --scope project" in err, err
    assert str(cfg) in err, "the refusal does not say what it would have written"


def test_uninstall_mcp_refuses_the_same_way(home, policy, capsys):
    """`uninstall-mcp` shares `resolve_target`, and restoring the wrong config is the
    more destructive direction of the same mistake."""
    rc = cli.main(["uninstall-mcp", "kb", "--file", "/tmp/x.json"])
    assert rc == 2
    assert "--file applies only to --scope project" in capsys.readouterr().err


def test_install_mcp_repo_path_at_the_default_scope_exits_2(home, policy, capsys):
    rc = cli.main(["install-mcp", "kb", "--policy", str(policy),
                   "--repo-path", "/some/repo", "--print"])
    assert rc == 2
    assert "--repo-path applies only to --scope local" in capsys.readouterr().err


def test_correct_project_scope_usage_still_installs(home, policy, tmp_path, capsys):
    """The guard is worthless if it also blocks the supported invocation."""
    proj = tmp_path / "proj"
    proj.mkdir()
    mcp = proj / ".mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"kb": {"command": "kb-mcp"}}}),
                   encoding="utf-8")
    rc = cli.main(["install-mcp", "kb", "--policy", str(policy),
                   "--scope", "project", "--file", str(mcp), "--print"])
    assert rc == 0
    assert "would wrap kb" in capsys.readouterr().out


def test_mcp_status_still_accepts_both_flags_at_once(home, tmp_path, capsys):
    """`mcp-status` has no `--scope`: it scans EVERY scope, passing each scope its own
    flag. A guard that fired on the flag alone rather than on the scope/flag pair would
    have broken it — which is why the check lives in `resolve_target`, where both are
    known, and not in argument parsing."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".mcp.json").write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    rc = cli.main(["mcp-status", "--file", str(proj / ".mcp.json"),
                   "--repo-path", "/some/repo"])
    assert rc == 0, capsys.readouterr().err


def test_the_sandbox_home_is_the_one_being_written(home):
    """Guards this file itself: if `HOME` redirection ever stops working, these tests
    would be exercising the developer's real config — which is how #366 was found."""
    assert resolve_target("user").cfg == home / ".claude.json"
    assert Path("~").expanduser() == home


def test_an_empty_flag_value_is_still_a_flag_that_was_passed(tmp_path):
    """`--file ""` is what `--file "$CFG"` becomes when `$CFG` is unset — the shell
    foot-gun that produces this mistake in the first place. Testing truthiness instead of
    `is not None` let it through to the default scope silently, which is the original bug
    with an extra step."""
    with pytest.raises(ValueError, match=r"--file applies only to --scope project"):
        resolve_target("user", file="")
    with pytest.raises(ValueError, match=r"--repo-path applies only to --scope local"):
        resolve_target("user", repo_path="")
