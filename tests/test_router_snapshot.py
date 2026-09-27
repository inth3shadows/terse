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
    Peer,
    Router,
    RouterSnapshot,
    load_multi_config,
    peers_fingerprint,
    router_snapshot_path,
    run_multi_proxy,
    snapshot_cwd,
)
from terse.policy import Policy, Rule
from terse.proxy import SWALLOW, Interceptor

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

    def __init__(self, cfg: pathlib.Path, broadcast_timeout: float = 10.0):
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
                                  broadcast_timeout=broadcast_timeout)

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
    return router_snapshot_path(str(cfg), snapshot_cwd(specs))


def _fp(cfg) -> str:
    specs = load_multi_config(str(cfg))
    return peers_fingerprint(specs, snapshot_cwd(specs))


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
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    c = _Client(cfg)
    try:
        c.handshake()
        c.request(3, "tools/call", {"name": "slow.tool"})   # the slow peer is up now
        time.sleep(1.0)                                      # room for the live refresh
        assert c.out.notes("notifications/tools/list_changed") == []
    finally:
        assert c.close() == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


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
