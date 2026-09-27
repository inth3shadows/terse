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

import json
import os
import pathlib
import sys
import threading
import time

import pytest

from terse.multiproxy import (
    RouterSnapshot,
    load_multi_config,
    peers_fingerprint,
    router_snapshot_path,
    run_multi_proxy,
)
from terse.policy import Policy, Rule

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

    def handshake(self) -> tuple[float, dict, float, dict]:
        init_s, init = self.request(1, "initialize",
                                    {"protocolVersion": "2025-06-18", "capabilities": {},
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


def _names(listed: dict) -> list[str]:
    return [t["name"] for t in listed["result"]["tools"]]


def _seed(cfg) -> dict:
    """One ordinary (blocking) session, which persists the snapshot; returns it."""
    c = _Client(cfg)
    c.handshake()
    assert c.close() == 0
    return json.loads(router_snapshot_path(str(cfg)).read_text(encoding="utf-8"))


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
    path = router_snapshot_path(str(cfg))
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
    path = router_snapshot_path(str(cfg))
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
    path = router_snapshot_path(str(cfg))
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
    assert snap["fingerprint"] == peers_fingerprint(load_multi_config(str(cfg)))
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
    new = json.loads(router_snapshot_path(str(cfg)).read_text(encoding="utf-8"))
    assert new["fingerprint"] != old["fingerprint"]
    assert new["fingerprint"] == peers_fingerprint(load_multi_config(str(cfg)))


# --- 6 ---

@pytest.mark.parametrize("damage", ["garbage", "wrong-shape", "directory"])
def test_a_corrupt_or_unreadable_snapshot_falls_back_to_blocking(tmp_path, damage):
    cfg = _config(tmp_path)
    _seed(cfg)
    path = router_snapshot_path(str(cfg))
    if damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    elif damage == "wrong-shape":
        path.write_text(json.dumps({"version": 1, "fingerprint": peers_fingerprint(
            load_multi_config(str(cfg))), "parts": {"initialize": []}}), encoding="utf-8")
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
    good = {"version": 1, "fingerprint": "fp",
            "parts": {"initialize": {"a": {"result": {}}},
                      "tools/list": {"a": {"result": {"tools": []}}}}}
    assert snap.load() is None                           # missing
    path.write_text(json.dumps(good), encoding="utf-8")
    assert snap.load() == good["parts"]
    for bad in ([], {**good, "version": 2}, {**good, "fingerprint": "other"},
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
    path = router_snapshot_path(str(cfg))
    assert json.loads(path.read_text(encoding="utf-8")) == seeded
