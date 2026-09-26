"""Offline tests for arms.py. Every test points `cfg`/stash at temp fixture
files -- never the operator's real ~/.claude.json or ~/.terse-mcp-stash.json
-- so these run identically on any machine and touch nothing live.

Requires `terse` importable (`import terse.install_mcp`); run via
`uv run pytest` from the repo so the editable install resolves.
"""
from __future__ import annotations

import json
from pathlib import Path

import arms


def _write(path, doc):
    path.write_text(json.dumps(doc))


def test_arm_a_is_empty():
    assert arms.arm_a_config() == {"mcpServers": {}}


def test_sanitize_router_args_strips_capture_dir_and_adds_no_stats():
    args = ["--policy", "/p.json", "--capture-dir", "/corpus", "--config", "/peers.json"]
    out = arms._sanitize_router_args(args)
    assert "--capture-dir" not in out
    assert "/corpus" not in out
    assert "--no-stats" in out
    assert out[:2] == ["--policy", "/p.json"]
    assert "--config" in out and out[out.index("--config") + 1] == "/peers.json"


def test_sanitize_router_args_does_not_duplicate_no_stats():
    args = ["--policy", "/p.json", "--no-stats"]
    out = arms._sanitize_router_args(args)
    assert out.count("--no-stats") == 1


def _write_router_and_peers(tmp_path, downstreams, *, extra_router_args=None):
    cfg = tmp_path / "claude.json"
    peers = tmp_path / "peers.json"
    _write(peers, {"downstreams": downstreams})
    args = ["proxy", "--config", str(peers)] + (extra_router_args or [])
    router = {"command": "/bin/terse", "args": args}
    _write(cfg, {"mcpServers": {"terse": router}})
    return cfg, peers


def test_arm_b_and_arm_c_launch_the_same_peer_commands(tmp_path):
    # Plan blocker #4's own regression test: the set of B's server commands
    # must equal the set of commands C's router launches for its peers --
    # built from the SAME peers-file downstreams, so the router is the only
    # difference between a B run and a C run.
    downstreams = [
        {"name": "kb", "policy": "p.json", "command": ["/bin/kb-mcp", "--flag"]},
        {"name": "codegraph", "policy": "p.json", "command": ["/bin/codegraph", "serve", "--mcp"]},
        {"name": "runecho", "policy": "p.json", "command": ["/bin/runecho-mcp"],
         "env": {"FOO": "bar"}},
    ]
    cfg, peers = _write_router_and_peers(tmp_path, downstreams)

    b_doc = arms.arm_b_config(cfg)
    b_commands = {name: [entry["command"], *entry.get("args", [])]
                  for name, entry in b_doc["mcpServers"].items()}
    c_commands = {d["name"]: d["command"] for d in downstreams}
    assert b_commands == c_commands
    assert b_doc["mcpServers"]["runecho"]["env"] == {"FOO": "bar"}


def test_arm_b_reflects_live_peers_file_not_a_stale_install_time_snapshot(tmp_path):
    # If a stash-of-originals-at-install-time approach were used instead
    # (the previous implementation), a peers file hand-edited or
    # regenerated AFTER install would silently diverge from what arm B
    # launches -- turning a B-vs-C delta partly into configuration noise
    # rather than purely the router's effect. Arm B must always reflect
    # what the peers file says TODAY.
    downstreams = [
        {"name": "kb", "policy": "p.json", "command": ["/bin/kb-mcp", "--NEW-FLAG-ADDED-LATER"]},
        {"name": "codegraph", "policy": "p.json", "command": ["/bin/codegraph"]},
        {"name": "runecho", "policy": "p.json", "command": ["/bin/runecho-mcp"]},
    ]
    cfg, peers = _write_router_and_peers(tmp_path, downstreams)

    b_doc = arms.arm_b_config(cfg)
    assert b_doc["mcpServers"]["kb"]["args"] == ["--NEW-FLAG-ADDED-LATER"]


def test_arm_b_raises_when_no_live_router(tmp_path):
    cfg = tmp_path / "claude.json"
    _write(cfg, {"mcpServers": {}})
    try:
        arms.arm_b_config(cfg)
    except KeyError as e:
        assert "terse" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_arm_b_raises_when_peers_file_missing_a_downstream(tmp_path):
    cfg, peers = _write_router_and_peers(tmp_path, [
        {"name": "kb", "policy": "p.json", "command": ["/bin/kb-mcp"]},
    ])  # codegraph/runecho absent
    try:
        arms.arm_b_config(cfg)
    except KeyError as e:
        assert "codegraph" in str(e) and "runecho" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_arm_c_copies_live_router_and_repoints_peers(tmp_path):
    cfg = tmp_path / "claude.json"
    peers = tmp_path / "peers.json"
    _write(peers, {"downstreams": [{"name": "kb"}]})
    router = {
        "command": "/bin/terse",
        "args": ["proxy", "--policy", "/p.json", "--capture-dir", "/corpus",
                 "--config", str(peers)],
    }
    _write(cfg, {"mcpServers": {"terse": router}})

    new_peers = tmp_path / "peers-copy.json"
    doc = arms.arm_c_config(cfg, peers_copy_path=new_peers)
    entry = doc["mcpServers"]["terse"]
    assert "--capture-dir" not in entry["args"]
    assert "--no-stats" in entry["args"]
    assert str(new_peers) in entry["args"]
    assert str(peers) not in entry["args"]
    # original live config on disk must be untouched
    assert json.loads(cfg.read_text())["mcpServers"]["terse"] == router


def test_write_arm_config_arm_c_pins_policy_via_peers_snapshot(tmp_path):
    # Plan Partial #10: the LIVE router carries no top-level `--policy` flag
    # at all (real shape confirmed against ~/.terse-peers-*.json) -- each
    # downstream in the peers file names its OWN `policy` path instead. A
    # fabricated `extra_router_args=["--policy", ...]` (the previous version
    # of this test) tested a flag arm C never actually has. The real
    # mechanism to pin is the peers file's per-downstream `policy` field.
    policy_src = tmp_path / "policy.json"
    _write(policy_src, {"default": {"redact": []}})
    downstreams = [{"name": name, "policy": "policy.json", "command": [f"/bin/{name}"]}
                   for name in arms.SERVERS]
    cfg, peers = _write_router_and_peers(tmp_path, downstreams)  # no fake --policy arg

    out_dir = tmp_path / "out"
    path = arms.write_arm_config("C", out_dir, cfg=cfg)
    doc = json.loads(path.read_text())
    entry = doc["mcpServers"]["terse"]
    args = entry["args"]
    assert "--policy" not in args  # real router shape has no top-level --policy to pin

    peers_copy = Path(args[args.index("--config") + 1])
    assert peers_copy.parent == out_dir
    peers_doc = json.loads(peers_copy.read_text())
    assert len(peers_doc["downstreams"]) == len(arms.SERVERS)
    for d in peers_doc["downstreams"]:
        pinned = Path(d["policy"])
        assert pinned != policy_src           # repointed away from the live file
        assert pinned.parent == out_dir
        assert json.loads(pinned.read_text()) == {"default": {"redact": []}}
        # the pinned copy is itself mode 600
        assert (pinned.stat().st_mode & 0o777) == 0o600
    # every downstream shared the SAME live policy.json -- the snapshot must
    # not make three separate copies of an identical source file.
    assert len({d["policy"] for d in peers_doc["downstreams"]}) == 1


def test_write_arm_config_arm_c_overrides_policy_and_terse_binary(tmp_path):
    # A C-prime variant (e.g. a policy with keep_first, served by an unreleased terse)
    # measured with everything else identical to arm C: the pinned policy copy holds the
    # OVERRIDE's content, and the router launches the override binary. Arg list unchanged.
    live_policy = tmp_path / "policy.json"
    _write(live_policy, {"tag": "live"})
    override = tmp_path / "variant.json"
    _write(override, {"tag": "variant"})
    downstreams = [{"name": name, "policy": "policy.json", "command": [f"/bin/{name}"]}
                   for name in arms.SERVERS]
    cfg, _ = _write_router_and_peers(tmp_path, downstreams)
    base = json.loads(arms.write_arm_config("C", tmp_path / "base", cfg=cfg).read_text())
    path = arms.write_arm_config("C", tmp_path / "out", cfg=cfg,
                                 c_policy=override, c_terse="/opt/terse-dev/bin/terse")
    entry = json.loads(path.read_text())["mcpServers"]["terse"]
    assert entry["command"] == "/opt/terse-dev/bin/terse"
    args = entry["args"]
    base_args = base["mcpServers"]["terse"]["args"]
    assert [a for a in args if "peers-" not in a] == [a for a in base_args if "peers-" not in a]
    peers_doc = json.loads(Path(args[args.index("--config") + 1]).read_text())
    for d in peers_doc["downstreams"]:
        assert json.loads(Path(d["policy"]).read_text()) == {"tag": "variant"}
    assert json.loads(live_policy.read_text()) == {"tag": "live"}  # live file untouched


def test_c_overrides_are_rejected_for_other_arms(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        arms.write_arm_config("B", tmp_path, c_terse="/x")
    with pytest.raises(ValueError):
        arms.write_arm_config("A", tmp_path, c_policy=tmp_path / "p.json")


def test_snapshot_peers_with_pinned_policy_leaves_policy_less_downstreams_alone(tmp_path):
    peers_doc = {"downstreams": [{"name": "kb", "command": ["/bin/kb"]}]}  # no 'policy' key
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    snapshot = arms._snapshot_peers_with_pinned_policy(peers_doc, tmp_path, out_dir, "C")
    assert "policy" not in snapshot["downstreams"][0]


def test_snapshot_peers_with_pinned_policy_resolves_relative_to_peers_dir(tmp_path):
    # multiproxy.py resolves a downstream's relative 'policy' against the
    # peers FILE's own directory, not the process cwd -- the snapshot must
    # match that resolution rule or it will copy the wrong file (or none).
    sub = tmp_path / "peers-dir"
    sub.mkdir()
    _write(sub / "policy.json", {"tag": "real-one"})
    peers_doc = {"downstreams": [{"name": "kb", "policy": "policy.json",
                                   "command": ["/bin/kb"]}]}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    snapshot = arms._snapshot_peers_with_pinned_policy(peers_doc, sub, out_dir, "C")
    pinned = Path(snapshot["downstreams"][0]["policy"])
    assert json.loads(pinned.read_text()) == {"tag": "real-one"}


def test_arm_c_raises_without_live_router(tmp_path):
    cfg = tmp_path / "claude.json"
    _write(cfg, {"mcpServers": {}})
    try:
        arms.arm_c_config(cfg)
    except KeyError as e:
        assert "terse" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_write_restricted_is_mode_600(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = arms._write_restricted(out_dir / "x.json", {"a": 1})
    assert (path.stat().st_mode & 0o777) == 0o600
    assert json.loads(path.read_text()) == {"a": 1}


def test_write_arm_config_arm_a_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    path = arms.write_arm_config("A", out_dir)
    assert json.loads(path.read_text()) == {"mcpServers": {}}
    assert (path.stat().st_mode & 0o777) == 0o600


def test_write_arm_config_arm_b_end_to_end(tmp_path):
    downstreams = [{"name": name, "policy": "p.json", "command": [f"/bin/{name}"]}
                   for name in arms.SERVERS]
    cfg, peers = _write_router_and_peers(tmp_path, downstreams)

    out_dir = tmp_path / "out"
    path = arms.write_arm_config("B", out_dir, cfg=cfg)
    doc = json.loads(path.read_text())
    assert set(doc["mcpServers"]) == set(arms.SERVERS)
    for name in arms.SERVERS:
        assert doc["mcpServers"][name]["command"] == f"/bin/{name}"


def test_write_arm_config_rejects_unknown_arm(tmp_path):
    try:
        arms.write_arm_config("Z", tmp_path)
    except ValueError as e:
        assert "unknown arm" in str(e)
    else:
        raise AssertionError("expected ValueError")
