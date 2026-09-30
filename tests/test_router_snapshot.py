"""The router answers `initialize` and `tools/list` from a persisted snapshot (#270).

`initialize` used to be a broadcast that blocked on the slowest peer (4.2s measured on the
live fleet, 2026-09-26). A headless client (`claude --print`) sends its first inference
request before that, so terse's tools arrived on turn 2 and the API reported
`cache_miss_reason: tools_changed` -- a whole prefix re-written.

These drive the real `run_multi_proxy` over a PIPE, not a StringIO: the claims here are
about WHEN replies arrive, so the client has to be able to wait between lines. The slow
peer is `fake_mcp_server.py` with `FAKE_INIT_DELAY`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pathlib
import sys
import threading
import time
from threading import Lock

import pytest

from terse.multiproxy import (
    _SNAPSHOT_MAX_AGE,
    Peer,
    Router,
    RouterSnapshot,
    load_multi_config,
    peers_fingerprint,
    router_snapshot_path,
    run_multi_proxy,
    snapshot_scope,
)
from terse.policy import Policy, Rule
from terse.proxy import SWALLOW, Interceptor
from terse.stats import load_stats

FAKE = pathlib.Path(__file__).parent / "fake_mcp_server.py"
POLICY = Policy(rules=[Rule("*", ("minify",))])
DELAY = 1.5          # the slow peer's initialize
FAST = 0.5           # "answered without waiting on it" -- a third of DELAY, generous for CI


class _Out:
    """A thread-safe client-side stdout that timestamps each line as it lands."""

    def __init__(self):
        self._buf = ""
        self.msgs: list[tuple[float, dict]] = []
        self._cv = threading.Condition()

    def write(self, s: str) -> int:
        with self._cv:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self.msgs.append((time.monotonic(), json.loads(line)))
            self._cv.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def wait(self, pred, timeout: float = 10.0) -> tuple[float, dict]:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                for t, m in self.msgs:
                    if pred(m):
                        return t, m
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"no matching message; got {self.msgs!r}")
                self._cv.wait(left)

    def notes(self, method: str) -> list[dict]:
        with self._cv:
            return [m for _, m in self.msgs if m.get("method") == method]


class _Client:
    """`run_multi_proxy` on a thread, fed over a real pipe."""

    def __init__(self, cfg: pathlib.Path, broadcast_timeout: float = 10.0,
                 stats_log: pathlib.Path | None = None):
        self._stats_log = stats_log
        r, w = os.pipe()
        self._cin = os.fdopen(r, "r", encoding="utf-8")
        self._w = os.fdopen(w, "w", encoding="utf-8")
        self.out = _Out()
        self.rc: int | None = None
        self._t = threading.Thread(target=self._run, args=(cfg, broadcast_timeout),
                                   daemon=True)
        self._t.start()

    def _run(self, cfg, broadcast_timeout):
        self.rc = run_multi_proxy(str(cfg), POLICY, stdin=self._cin, stdout=self.out,
                                  broadcast_timeout=broadcast_timeout,
                                  stats_log=(str(self._stats_log) if self._stats_log
                                             else None))

    def send(self, msg: dict) -> float:
        self._w.write(json.dumps(msg) + "\n")
        self._w.flush()
        return time.monotonic()

    def request(self, mid, method, params=None, timeout=10.0) -> tuple[float, dict]:
        """Send and wait; returns (seconds until the reply, reply)."""
        t0 = self.send({"jsonrpc": "2.0", "id": mid, "method": method,
                        "params": params or {}})
        t, m = self.out.wait(lambda m: m.get("id") == mid, timeout)
        return t - t0, m

    def handshake(self, protocol: str = "2025-06-18") -> tuple[float, dict, float, dict]:
        init_s, init = self.request(1, "initialize",
                                    {"protocolVersion": protocol, "capabilities": {},
                                     "clientInfo": {"name": "t", "version": "0"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        list_s, listed = self.request(2, "tools/list")
        return init_s, init, list_s, listed

    def close(self) -> int | None:
        self._w.close()
        self._t.join(timeout=60)
        return self.rc


def _config(tmp_path, *, slow_env=None, fast_env=None) -> pathlib.Path:
    cfg = tmp_path / "peers.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "fast", "command": [sys.executable, str(FAKE)],
         "env": {"FAKE_TOOLS": "fast.tool", **(fast_env or {})}},
        {"name": "slow", "command": [sys.executable, str(FAKE)],
         "env": {"FAKE_TOOLS": "slow.tool", "FAKE_INIT_DELAY": str(DELAY),
                 **(slow_env or {})}},
    ]}), encoding="utf-8")
    return cfg


def _path(cfg) -> pathlib.Path:
    specs = load_multi_config(str(cfg))
    return router_snapshot_path(str(cfg), snapshot_scope(specs))


def _fp(cfg) -> str:
    specs = load_multi_config(str(cfg))
    return peers_fingerprint(specs, snapshot_scope(specs))


def _names(listed: dict) -> list[str]:
    return [t["name"] for t in listed["result"]["tools"]]


def _seed(cfg) -> dict:
    """One ordinary (blocking) session, which persists the snapshot; returns it."""
    c = _Client(cfg)
    c.handshake()
    assert c.close() == 0
    return json.loads(_path(cfg).read_text(encoding="utf-8"))


# --- 1 ---

def test_with_a_snapshot_initialize_and_tools_list_answer_before_the_slow_peer(tmp_path):
    cfg = _config(tmp_path)
    _seed(cfg)
    c = _Client(cfg)
    try:
        init_s, init, list_s, listed = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s < FAST and list_s < FAST, (init_s, list_s)
    assert init["result"]["serverInfo"]["name"] == "terse"
    # The router may now emit list_changed, so it says so.
    assert init["result"]["capabilities"]["tools"]["listChanged"] is True
    assert _names(listed) == ["fast.tool", "slow.tool"]


# --- 2 ---

def test_a_call_to_a_peer_that_is_not_ready_yet_is_delivered_not_errored(tmp_path):
    cfg = _config(tmp_path)
    _seed(cfg)
    c = _Client(cfg)
    try:
        c.handshake()
        call_s, reply = c.request(3, "tools/call", {"name": "slow.tool", "arguments": {}})
    finally:
        assert c.close() == 0
    assert "result" in reply and "error" not in reply
    # It waited for the peer rather than failing fast.
    assert call_s > DELAY - FAST


# --- 3 ---

def test_a_snapshot_that_differs_from_live_emits_one_list_changed_and_is_updated(tmp_path):
    cfg = _config(tmp_path)
    snap = _seed(cfg)
    snap["parts"]["tools/list"]["slow"]["result"]["tools"] = [{"name": "stale.tool"}]
    path = _path(cfg)
    path.write_text(json.dumps(snap), encoding="utf-8")

    c = _Client(cfg)
    try:
        _, _, _, listed = c.handshake()
        assert _names(listed) == ["fast.tool", "stale.tool"]    # served from the snapshot
        c.out.wait(lambda m: m.get("method") == "notifications/tools/list_changed",
                   timeout=DELAY + 5)
        time.sleep(0.5)
        assert len(c.out.notes("notifications/tools/list_changed")) == 1
        _, relisted = c.request(4, "tools/list")
        assert _names(relisted) == ["fast.tool", "slow.tool"]   # the live list now
        # ...and it routes: the stale name is gone, the live one is there.
        _, gone = c.request(5, "tools/call", {"name": "stale.tool"})
        assert gone["error"]["code"] == -32601
    finally:
        assert c.close() == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["parts"]["tools/list"]["slow"]["result"]["tools"] == [{"name": "slow.tool"}]


def test_a_snapshot_identical_to_live_emits_nothing_and_is_not_rewritten(tmp_path):
    cfg = _config(tmp_path)
    _seed(cfg)
    path = _path(cfg)
    # By inode, not mtime: `load` touches the file it serves (so `_prune` sees it in use),
    # while a rewrite is an atomic replace, which always lands a new inode.
    before = (path.read_bytes(), path.stat().st_ino)
    c = _Client(cfg)
    try:
        c.handshake()
        c.request(3, "tools/call", {"name": "slow.tool"})   # the slow peer is up now
        time.sleep(1.0)                                      # room for the live refresh
        assert c.out.notes("notifications/tools/list_changed") == []
    finally:
        assert c.close() == 0
    assert (path.read_bytes(), path.stat().st_ino) == before


# --- 4 ---

def test_without_a_snapshot_initialize_blocks_as_today_and_one_is_written(tmp_path):
    cfg = _config(tmp_path)
    path = _path(cfg)
    assert not path.exists()
    c = _Client(cfg)
    try:
        init_s, init, _, listed = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s > DELAY - FAST                        # waited on the slow peer, as today
    assert "tools" not in init["result"]["capabilities"]  # no listChanged claim on this path
    assert _names(listed) == ["fast.tool", "slow.tool"]
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["fingerprint"] == _fp(cfg)
    assert set(snap["parts"]["initialize"]) == set(snap["parts"]["tools/list"]) \
        == {"fast", "slow"}


# --- 5 ---

def test_an_edited_peers_config_ignores_the_old_snapshot(tmp_path):
    cfg = _config(tmp_path)
    old = _seed(cfg)
    cfg = _config(tmp_path, fast_env={"EDITED": "1"})     # same path, different peers
    c = _Client(cfg)
    try:
        init_s, _, _, _ = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s > DELAY - FAST
    new = json.loads(_path(cfg).read_text(encoding="utf-8"))
    assert new["fingerprint"] != old["fingerprint"]
    assert new["fingerprint"] == _fp(cfg)


# --- 6 ---

@pytest.mark.parametrize("damage", ["garbage", "wrong-shape", "directory"])
def test_a_corrupt_or_unreadable_snapshot_falls_back_to_blocking(tmp_path, damage):
    cfg = _config(tmp_path)
    _seed(cfg)
    path = _path(cfg)
    if damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    elif damage == "wrong-shape":
        path.write_text(json.dumps({"version": 2, "protocol": "2025-06-18", "fingerprint": _fp(cfg), "parts": {"initialize": []}}), encoding="utf-8")
    else:
        path.unlink()
        path.mkdir()
    c = _Client(cfg)
    try:
        init_s, init, _, listed = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s > DELAY - FAST
    assert "result" in init and _names(listed) == ["fast.tool", "slow.tool"]
    if damage != "directory":   # rewritten whole; a directory is left for the operator
        json.loads(path.read_text(encoding="utf-8"))


def test_snapshot_load_rejects_every_shape_it_cannot_serve(tmp_path):
    path = tmp_path / "s.json"
    snap = RouterSnapshot(path, "fp")
    good = {"version": 2, "fingerprint": "fp", "protocol": "2025-06-18",
            "parts": {"initialize": {"a": {"result": {}}},
                      "tools/list": {"a": {"result": {"tools": []}}}}}
    assert snap.load() is None                           # missing
    path.write_text(json.dumps(good), encoding="utf-8")
    assert snap.load() == (good["parts"], "2025-06-18")
    no_protocol = {k: v for k, v in good.items() if k != "protocol"}
    for bad in ([], {**good, "version": 1}, {**good, "fingerprint": "other"}, no_protocol,
                {**good, "parts": {"initialize": good["parts"]["initialize"]}},
                {**good, "parts": {**good["parts"], "tools/list": {"a": "x"}}},
                {**good, "parts": {**good["parts"], "tools/list": {"b": {"result": {}}}}}):
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert snap.load() is None, bad


# --- 7 ---

def test_a_peer_that_fails_to_start_still_errors_its_calls(tmp_path):
    flag = tmp_path / "die"
    cfg = _config(tmp_path, slow_env={"FAKE_DIE_IF": str(flag), "FAKE_INIT_DELAY": "0"})
    seeded = _seed(cfg)
    flag.touch()
    c = _Client(cfg, broadcast_timeout=1.0)
    try:
        init_s, _, _, listed = c.handshake()
        assert init_s < FAST
        assert _names(listed) == ["fast.tool", "slow.tool"]   # the snapshot's view
        _, dead = c.request(3, "tools/call", {"name": "slow.tool"})
        _, live = c.request(4, "tools/call", {"name": "fast.tool"})
        # Once the router sees the peer never came up, the client is told the list changed.
        c.out.wait(lambda m: m.get("method") == "notifications/tools/list_changed",
                   timeout=5)
    finally:
        assert c.close() == 0
    assert dead["error"]["code"] == -32001 and "timed out" in dead["error"]["message"]
    assert "result" in live
    # A degraded live listing is never persisted over a complete snapshot.
    path = _path(cfg)
    assert json.loads(path.read_text(encoding="utf-8")) == seeded


# --- review fixes: cwd, initialize drift, protocolVersion, advertised list_changed ---

def test_two_launch_directories_keep_separate_snapshots(tmp_path, monkeypatch):
    # A peer with no `cwd` inherits the router's, and its replies can depend on it (the
    # live codegraph peer has zero tools outside an indexed repo). One directory's snapshot
    # must never answer another's.
    cfg = _config(tmp_path)
    (tmp_path / "repo_a").mkdir()
    (tmp_path / "repo_b").mkdir()
    monkeypatch.chdir(tmp_path / "repo_a")
    _seed(cfg)
    path_a, fp_a = _path(cfg), _fp(cfg)
    monkeypatch.chdir(tmp_path / "repo_b")
    assert _path(cfg) != path_a and _fp(cfg) != fp_a
    c = _Client(cfg)
    try:
        init_s, _, _, _ = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s > DELAY - FAST          # repo_a's snapshot was not served here
    assert path_a.exists() and _path(cfg).exists()


def _marked_config(tmp_path, markers) -> pathlib.Path:
    """`_config` with `snapshot_markers` declared on both (cwd-inheriting) peers."""
    cfg = _config(tmp_path)
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    for d in doc["downstreams"]:
        d["snapshot_markers"] = markers
    cfg.write_text(json.dumps(doc), encoding="utf-8")
    return cfg


def test_directories_sharing_a_declared_marker_share_a_snapshot_and_start_warm(
        tmp_path, monkeypatch):
    # #479: with the marker declared, a new worktree whose `.codegraph` links to the same
    # index is served the snapshot another worktree saved -- and so is a subdirectory.
    cfg = _marked_config(tmp_path, [".idx"])
    (tmp_path / "main" / ".idx").mkdir(parents=True)
    (tmp_path / "main" / "sub").mkdir()
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".idx").symlink_to("../main/.idx")
    monkeypatch.chdir(tmp_path / "main")
    _seed(cfg)
    path, fp = _path(cfg), _fp(cfg)
    for d in (tmp_path / "wt", tmp_path / "main" / "sub"):
        monkeypatch.chdir(d)
        assert (_path(cfg), _fp(cfg)) == (path, fp), d
    c = _Client(cfg)
    try:
        init_s, _, list_s, _ = c.handshake()
    finally:
        assert c.close() == 0
    assert init_s < FAST and list_s < FAST, (init_s, list_s)


def test_a_different_or_missing_marker_keeps_a_separate_snapshot(tmp_path, monkeypatch):
    # The reviewed failure of keying by git repo: an unindexed worktree of an indexed repo
    # must not be served the indexed one's snapshot, nor it theirs.
    cfg = _marked_config(tmp_path, [".idx"])
    for d in ("indexed/.idx", "other/.idx", "bare"):
        (tmp_path / "repo" / d).mkdir(parents=True)
    paths = []
    for d in ("indexed", "other", "bare"):
        monkeypatch.chdir(tmp_path / "repo" / d)
        paths.append(_path(cfg))
    assert len(set(paths)) == 3


def test_directories_with_no_marker_anywhere_share_one_snapshot(tmp_path, monkeypatch):
    cfg = _marked_config(tmp_path, [".no-such-marker-479"])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    path_a = _path(cfg)
    monkeypatch.chdir(tmp_path / "b")
    assert _path(cfg) == path_a


def test_one_undeclared_inheriting_peer_keeps_the_snapshot_per_directory(tmp_path, monkeypatch):
    cfg = _marked_config(tmp_path, [])
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    del doc["downstreams"][1]["snapshot_markers"]
    cfg.write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    assert snapshot_scope(load_multi_config(str(cfg))) == str(tmp_path / "a")
    path_a = _path(cfg)
    monkeypatch.chdir(tmp_path / "b")
    assert _path(cfg) != path_a


def test_an_undeclared_config_keys_exactly_as_before_the_field_existed(tmp_path, monkeypatch):
    # Adding the field must not invalidate every saved snapshot on upgrade: with no peer
    # declaring markers, the scope is the cwd and the fingerprint hashes the same entries.
    cfg = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    specs = load_multi_config(str(cfg))
    assert snapshot_scope(specs) == os.getcwd()
    legacy = [{k: v for k, v in dataclasses.asdict(s).items() if k != "snapshot_markers"}
              for s in specs]
    doc = json.dumps({"peers": legacy, "cwd": os.getcwd()}, sort_keys=True,
                     separators=(",", ":"))
    assert _fp(cfg) == hashlib.sha256(doc.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("bad", ["x", [""], ["/abs"], ["../up"], [1]])
def test_malformed_snapshot_markers_are_a_config_error(tmp_path, bad):
    cfg = _marked_config(tmp_path, bad)
    with pytest.raises(ValueError, match="snapshot_markers"):
        load_multi_config(str(cfg))


def test_snapshot_markers_on_a_peer_with_its_own_cwd_is_a_config_error(tmp_path):
    cfg = _marked_config(tmp_path, [".idx"])
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    doc["downstreams"][0]["cwd"] = str(tmp_path)
    cfg.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="inherits the router's"):
        load_multi_config(str(cfg))


def test_each_session_row_records_whether_it_started_warm(tmp_path):
    # The ledger is how the live fleet shows a new worktree actually starting warm (#479).
    cfg, log = _config(tmp_path), tmp_path / "stats.jsonl"
    for _ in range(2):
        c = _Client(cfg, stats_log=log)
        c.handshake()
        assert c.close() == 0
    rows = [r for r in load_stats(log) if r.get("event") == "router_session"]
    assert [r["snapshot"] for r in rows] == ["cold", "warm"]


def _stale(*files: pathlib.Path) -> None:
    t = time.time() - _SNAPSHOT_MAX_AGE - 60
    for f in files:
        os.utime(f, (t, t))


def test_prune_removes_only_stale_snapshots_and_never_its_own(tmp_path):
    snap = RouterSnapshot(tmp_path / "live.json", "fp")
    old, young, other = tmp_path / "old.json", tmp_path / "young.json", tmp_path / "old.txt"
    for f in (old, young, other, snap.path):
        f.write_text("{}", encoding="utf-8")
    _stale(old, other, snap.path)          # its own file stale too: the guard must hold
    snap._prune()
    assert not old.exists()
    assert young.exists() and other.exists() and snap.path.exists()


def test_save_prunes_its_stale_siblings(tmp_path):
    snap = RouterSnapshot(tmp_path / "live.json", "fp")
    old = tmp_path / "old.json"
    old.write_text("{}", encoding="utf-8")
    _stale(old)
    snap.save({"initialize": {}, "tools/list": {}}, "2025-06-18")
    assert not old.exists() and snap.path.exists()


def test_a_served_snapshot_is_not_pruned_as_abandoned(tmp_path):
    # Replies that never change are never rewritten; being served must still count as use,
    # or a stable repo's snapshot would be deleted by another repo's save after 30 days.
    live = RouterSnapshot(tmp_path / "live.json", "fp")
    live.save({"initialize": {"p": {}}, "tools/list": {"p": {}}}, "2025-06-18")
    _stale(live.path)
    assert live.load() is not None
    RouterSnapshot(tmp_path / "other.json", "fp").save(
        {"initialize": {}, "tools/list": {}}, "2025-06-18")
    assert live.path.exists()


def test_an_initialize_only_difference_is_reported_and_persisted(tmp_path, capsys):
    cfg = _config(tmp_path)
    snap = _seed(cfg)
    snap["parts"]["initialize"]["fast"]["result"]["instructions"] = "STALE NOTES."
    path = _path(cfg)
    path.write_text(json.dumps(snap), encoding="utf-8")
    c = _Client(cfg)
    try:
        _, init, _, _ = c.handshake()
        assert "STALE NOTES." in init["result"]["instructions"]   # served; cannot be undone
        c.request(3, "tools/call", {"name": "slow.tool"})          # the slow peer is up
        time.sleep(1.0)
        # The tools did not change, so the client is not told anything.
        assert c.out.notes("notifications/tools/list_changed") == []
    finally:
        assert c.close() == 0
    assert "live initialize differs" in capsys.readouterr().err
    after = json.loads(path.read_text(encoding="utf-8"))
    assert "instructions" not in after["parts"]["initialize"]["fast"]["result"]


def test_a_different_client_protocol_version_takes_the_blocking_path(tmp_path):
    cfg = _config(tmp_path)
    assert _seed(cfg)["protocol"] == "2025-06-18"
    c = _Client(cfg)
    try:
        init_s, init, _, _ = c.handshake(protocol="2024-11-05")
    finally:
        assert c.close() == 0
    assert init_s > DELAY - FAST
    assert "tools" not in init["result"]["capabilities"]
    assert json.loads(_path(cfg).read_text(encoding="utf-8"))["protocol"] == "2024-11-05"


class _FakeTransport:
    def __init__(self):
        self.out = io.StringIO()

    def inbound(self):
        return iter([])

    def outbound(self):
        return self.out

    def close(self):
        pass


def _await_sent(t: _FakeTransport, method: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        for ln in t.out.getvalue().splitlines():
            m = json.loads(ln)
            if m.get("method") == method:
                return m
        time.sleep(0.01)
    raise AssertionError(f"{method} never reached the peer")


def test_list_changed_is_sent_only_for_a_capability_that_advertised_it(tmp_path):
    # The merged capabilities declare tools and prompts but not resources: both get
    # listChanged, and a changed resources list is NOT announced (it was never declared).
    init = {"result": {"protocolVersion": "2025-06-18",
                       "capabilities": {"tools": {}, "prompts": {}}}}
    parts = {"initialize": {"a": init},
             "tools/list": {"a": {"result": {"tools": [{"name": "t1"}]}}},
             "prompts/list": {"a": {"result": {"prompts": [{"name": "p1"}]}}},
             "resources/list": {"a": {"result": {"resources": []}}}}
    path = tmp_path / "snap.json"
    RouterSnapshot(path, "fp").save(parts, "2025-06-18")
    t, out = _FakeTransport(), io.StringIO()
    router = Router([Peer("a", t, Interceptor(POLICY))], out, Lock(), broadcast_timeout=1000,
                    snapshot=RouterSnapshot(path, "fp"))
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18"}}))
        caps = json.loads(out.getvalue())["result"]["capabilities"]
        assert caps["tools"]["listChanged"] is True
        assert caps["prompts"]["listChanged"] is True
        assert "resources" not in caps
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        feed = router.from_peer(0)
        sent = _await_sent(t, "initialize")
        assert feed(json.dumps({"jsonrpc": "2.0", "id": sent["id"], **init})) is SWALLOW
        replies = {"tools/list": {"tools": [{"name": "t1"}]},             # unchanged
                   "prompts/list": {"prompts": [{"name": "p2"}]},         # changed
                   "resources/list": {"resources": [{"uri": "a://x"}]}}   # changed
        for method, result in replies.items():
            sent = _await_sent(t, method)
            assert feed(json.dumps({"jsonrpc": "2.0", "id": sent["id"],
                                    "result": result})) is SWALLOW
    finally:
        router.close_senders()
    notes = [m["method"] for m in (json.loads(ln) for ln in out.getvalue().splitlines())
             if "method" in m]
    assert notes == ["notifications/prompts/list_changed"]


def test_peers_swapping_initialize_arrival_order_is_not_reported_as_a_change(tmp_path, capsys):
    # Instructions are joined in ARRIVAL order by the real merge. Two instruction-bearing
    # peers answering in the opposite order to the snapshot's is not a change.
    def init(text):
        return {"result": {"protocolVersion": "2025-06-18",
                           "capabilities": {"tools": {}}, "instructions": text}}
    parts = {"initialize": {"a": init("A NOTES."), "b": init("B NOTES.")},   # a first
             "tools/list": {"a": {"result": {"tools": [{"name": "ta"}]}},
                            "b": {"result": {"tools": [{"name": "tb"}]}}}}
    path = tmp_path / "snap.json"
    RouterSnapshot(path, "fp").save(parts, "2025-06-18")
    before = path.read_bytes()
    ts, out = [_FakeTransport(), _FakeTransport()], io.StringIO()
    router = Router([Peer("a", ts[0], Interceptor(POLICY)),
                     Peer("b", ts[1], Interceptor(POLICY))],
                    out, Lock(), broadcast_timeout=1000, snapshot=RouterSnapshot(path, "fp"))
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18"}}))
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        for i in (1, 0):                                                       # b first
            sent = _await_sent(ts[i], "initialize")
            router.from_peer(i)(json.dumps({"jsonrpc": "2.0", "id": sent["id"],
                                            **parts["initialize"]["ab"[i]]}))
        for i in (0, 1):
            sent = _await_sent(ts[i], "tools/list")
            router.from_peer(i)(json.dumps({"jsonrpc": "2.0", "id": sent["id"],
                                            **parts["tools/list"]["ab"[i]]}))
    finally:
        router.close_senders()
    assert "live initialize differs" not in capsys.readouterr().err
    assert not [ln for ln in out.getvalue().splitlines() if "list_changed" in ln]
    assert path.read_bytes() == before
