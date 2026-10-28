"""The wire log (`proxy --wire-log DIR`, wirelog.py): every line at all four hops, in both
proxy modes, recorded once, in order, with capture:false traffic reduced to metadata.

Every end-to-end test runs `_check`, the completeness invariant: seq contiguous from 0,
start first and stop last, every client line read appears as a client_in record, and
every line the proxy wrote to the client's stdout appears exactly once, in order, as a
client_out record (the bytes themselves, or their length when excluded)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import signal
import sys
import threading
import time
import uuid

import pytest

from terse import multiproxy as mp
from terse import wirelog
from terse.lossy import _handle, _serialize
from terse.multiproxy import _PeerSender, run_multi_proxy
from terse.policy import Policy, Rule
from terse.proxy import run_proxy
from terse.transport import HttpTransport

FAKE = pathlib.Path(__file__).parent / "fake_mcp_server.py"
TIERS = ("minify", "tabularize", "dictionary")
FULL = Policy(rules=[Rule("gh.*", TIERS)])
MODES = ("single", "multi")


@pytest.fixture(autouse=True)
def _no_session(monkeypatch):
    # These tests may themselves run under Claude Code, which sets the variable.
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)


def _req(mid, method, params=None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def _call(mode, mid, tool, args=None) -> str:
    # multiproxy without a listing routes by the `<peer>__` prefix fallback
    name = tool if mode == "single" else f"gh__{tool}"
    return _req(mid, "tools/call", {"name": name, "arguments": args or {}})


INIT = _req(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "t", "version": "0"}})
INITIALIZED = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})


class _Paced:
    """Client stdin that holds each line until the replies it depends on are on stdout."""

    def __init__(self, cout: io.StringIO, steps: list[tuple[str, tuple]]):
        self.cout, self.steps = cout, steps

    def _ids(self) -> set:
        ids = set()
        for ln in self.cout.getvalue().splitlines():
            try:
                ids.add(json.loads(ln).get("id"))
            except (ValueError, AttributeError):
                pass
        return ids

    def __iter__(self):
        for line, wait in self.steps:
            deadline = time.monotonic() + 20
            while not set(wait) <= self._ids():
                assert time.monotonic() < deadline, f"no reply to {wait}"
                time.sleep(0.01)
            yield line + "\n"


def _peers_cfg(tmp_path, peers=None) -> pathlib.Path:
    cfg = tmp_path / "peers.json"
    cfg.write_text(json.dumps({"downstreams": peers or [
        {"name": "gh", "command": [sys.executable, str(FAKE)]}]}), encoding="utf-8")
    return cfg


def _run(mode, tmp_path, lines, pol=FULL, *, wire=True, peers=None, steps=None, **kw):
    """Run one proxy session; returns (rc, stdout text, wire records or None, client lines)."""
    cout = io.StringIO()
    if steps is not None:
        cin, sent = _Paced(cout, steps), [s for s, _ in steps]
    else:
        cin, sent = io.StringIO("".join(ln + "\n" for ln in lines)), list(lines)
    wdir = tmp_path / f"wire-{uuid.uuid4().hex[:6]}"
    wl = str(wdir) if wire else None
    if mode == "single":
        rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout,
                       wire_log=wl, **kw)
    else:
        rc = run_multi_proxy(str(_peers_cfg(tmp_path, peers)), pol, stdin=cin, stdout=cout,
                             wire_log=wl, **kw)
    return rc, cout.getvalue(), (_records(wdir) if wire else None), sent


def _records(wdir: pathlib.Path) -> list[dict]:
    files = list(wdir.glob("*/*.jsonl"))
    assert len(files) == 1, files
    return [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines()]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _check(recs: list[dict], out: str, sent: list[str]) -> None:
    assert [r["seq"] for r in recs] == list(range(len(recs)))       # no holes
    assert len({r["conn"] for r in recs}) == 1
    assert recs[0]["event"] == "start" and recs[-1]["event"] == "stop"
    for r in recs:
        if r["hop"] != "event" and r["line"] is not None:
            assert r["sha256"] == _sha(r["line"]) and r["len"] == len(r["line"])
    assert [r["line"] for r in recs if r["hop"] == "client_in"
            and r["line"] is not None] == [s for s in sent if not _excluded_in(recs, s)]
    written = out.split("\n")[:-1]
    outs = [r for r in recs if r["hop"] == "client_out"]
    assert len(outs) == len(written)
    for r, ln in zip(outs, written, strict=True):
        if r["capture_status"] == "excluded_by_policy":
            assert r["line"] is None and r["sha256"] is None and r["len"] == len(ln)
        else:
            assert r["line"] == ln


def _excluded_in(recs, sent_line) -> bool:
    return any(r["hop"] == "client_in" and r["line"] is None and r["len"] == len(sent_line)
               for r in recs)


def _hops(recs, mid) -> list[tuple[str, str]]:
    return [(r["hop"], r["capture_status"]) for r in recs if r.get("jsonrpc_id") == mid]


def _events(recs, name) -> list[dict]:
    return [r for r in recs if r["hop"] == "event" and r["event"] == name]


# --- normal traffic -------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_normal_and_repeated_calls_each_recorded_and_output_unchanged(mode, tmp_path):
    lines = [INIT, INITIALIZED] + [_call(mode, i, "gh.api.items") for i in (2, 3, 4)]
    rc, out, recs, sent = _run(mode, tmp_path, lines)
    rc_off, out_off, _, _ = _run(mode, tmp_path, lines, wire=False)
    assert rc == rc_off == 0
    assert out == out_off                      # the tap never changes what the client gets
    _check(recs, out, sent)
    # no dedup: three identical results, three server_in records carrying them
    results = [json.dumps(json.loads(r["line"])["result"]) for r in recs
               if r["hop"] == "server_in" and r["jsonrpc_id"] in (2, 3, 4)]
    assert len(results) == 3 and len(set(results)) == 1
    for mid in (2, 3, 4):
        assert [h for h, _ in _hops(recs, mid)] == [
            "client_in", "server_out", "server_in", "client_out"]
    note = [r for r in recs if r["method"] == "notifications/initialized"]
    assert note[0]["hop"] == "client_in" and any(r["hop"] == "server_out" for r in note)
    assert all(r["claude_session"] == "no-session" and r["v"] == 1 for r in recs)
    start = recs[0]["detail"]
    assert start["mode"] == mode and start["peers"]
    assert _events(recs, "peer_start")[0]["detail"]["pid"]


@pytest.mark.parametrize("mode", MODES)
def test_large_payload_recorded_intact(mode, tmp_path):
    n = 1_200_000
    rc, out, recs, sent = _run(mode, tmp_path, [_call(mode, 2, "fake.big", {"n": n})])
    assert rc == 0
    _check(recs, out, sent)
    (big,) = [r for r in recs if r["hop"] == "server_in" and r["jsonrpc_id"] == 2]
    assert big["len"] > 1_000_000 and big["capture_status"] == "full"
    assert big["sha256"] == _sha(big["line"])
    assert len(json.loads(big["line"])["result"]["content"][0]["text"]) == n


@pytest.mark.parametrize("mode", MODES)
def test_server_error_reply_recorded(mode, tmp_path):
    rc, out, recs, sent = _run(mode, tmp_path, [_call(mode, 2, "fake.error")])
    _check(recs, out, sent)
    (reply,) = [r for r in recs if r["hop"] == "client_out" and r["jsonrpc_id"] == 2]
    assert json.loads(reply["line"])["error"]["code"] == -32050


@pytest.mark.parametrize("mode", MODES)
def test_malformed_and_blank_lines_recorded_as_parse_error(mode, tmp_path):
    lines = ["", "   ", "not json", _call(mode, 2, "fake.noise")]
    rc, out, recs, sent = _run(mode, tmp_path, lines)
    _check(recs, out, sent)
    bad_in = [r["line"] for r in recs if r["hop"] == "client_in"
              and r["capture_status"] == "parse_error"]
    assert bad_in == ["", "   ", "not json"]
    bad_srv = [r["line"] for r in recs if r["hop"] == "server_in"
               and r["capture_status"] == "parse_error"]
    assert bad_srv == ["", "   ", "not json {"]


@pytest.mark.parametrize("mode", MODES)
def test_reinitialize_on_same_stream_recorded_twice(mode, tmp_path):
    second = INIT.replace('"id": 1', '"id": 9')
    rc, out, recs, sent = _run(mode, tmp_path, [INIT, _call(mode, 2, "gh.api.items"), second])
    _check(recs, out, sent)
    inits = [r for r in recs if r["hop"] == "client_in" and r["method"] == "initialize"]
    assert [r["jsonrpc_id"] for r in inits] == [1, 9]
    assert {1, 9} <= {r["jsonrpc_id"] for r in recs if r["hop"] == "client_out"}


# --- capture:false exclusion ----------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_capture_false_call_is_metadata_only(mode, tmp_path):
    secret = f"S3CRET-{uuid.uuid4().hex}"
    pol = Policy(rules=[Rule("*fake.echo", (), capture=False), Rule("gh.*", TIERS)])
    lines = [INIT, _call(mode, 5, "fake.echo", {"token": secret}),
             _call(mode, 6, "gh.api.items")]
    rc, out, recs, sent = _run(mode, tmp_path, lines, pol)
    _check(recs, out, sent)
    assert secret in out                                 # forwarding is untouched
    wdir_text = json.dumps(recs)
    assert secret not in wdir_text
    excl = [r for r in recs if r["capture_status"] == "excluded_by_policy"]
    assert sorted(r["hop"] for r in excl) == sorted(
        ["client_in", "server_out", "server_in", "client_out"])
    for r in excl:
        assert r["line"] is None and r["sha256"] is None and r["len"] > 0
        assert r["rule"] == "*fake.echo" and r["jsonrpc_id"] == 5
    assert {s for h, s in _hops(recs, 6)} == {"full"}


@pytest.mark.parametrize("mode", MODES)
def test_capture_false_server_excludes_every_line_for_it(mode, tmp_path):
    secret = f"S3CRET-{uuid.uuid4().hex}"
    if mode == "single":
        pol = Policy(rules=[Rule("secret-broker.*", (), capture=False,
                                 require_server_name=True)])
        rc, out, recs, sent = _run(mode, tmp_path, [
            INIT, _req(5, "tools/call", {"name": "fake.echo",
                                         "arguments": {"token": secret}})],
            pol, server_name="secret-broker")
        assert {r["capture_status"] for r in recs if r["hop"] != "event"} == {
            "excluded_by_policy"}
    else:
        env = {"FAKE_TOOLS": "fake.echo,gh.api.items"}   # same names: both get qualified
        peers = [{"name": "gh", "command": [sys.executable, str(FAKE)], "env": env},
                 {"name": "secret-broker", "command": [sys.executable, str(FAKE)],
                  "env": env}]
        pol = Policy(rules=[Rule("secret-broker.*", (), capture=False), Rule("gh.*", TIERS)])
        steps = [(INIT, ()), (_req(2, "tools/list"), (1,)),
                 (_req(5, "tools/call", {"name": "secret-broker__fake.echo",
                                         "arguments": {"token": secret}}), (2,)),
                 (_req(6, "tools/call", {"name": "gh__gh.api.items"}), (5,))]
        rc, out, recs, sent = _run(mode, tmp_path, None, pol, peers=peers, steps=steps)
        listed = [t["name"] for t in json.loads(out.splitlines()[1])["result"]["tools"]]
        assert "secret-broker__fake.echo" in listed      # resolved through the table
        sb = [r for r in recs if r["peer"] == "secret-broker" and r["hop"] != "event"]
        assert sb and {r["capture_status"] for r in sb} == {"excluded_by_policy"}
        assert {s for _, s in _hops(recs, 5)} == {"excluded_by_policy"}
        assert {s for _, s in _hops(recs, 6)} == {"full"}
    _check(recs, out, sent)
    assert secret in out and secret not in json.dumps(recs)


# --- locally answered retrieve ----------------------------------------------------------

DROP = {"result[].status": {"lossy": "drop-to-retrieve", "min": 1}}


@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("mode", MODES)
def test_retrieve_answered_locally(mode, capture, escape, tmp_path):
    pol = Policy(rules=[Rule("gh.*", TIERS, fields=DROP, capture=capture)])
    cout_probe: dict = {}

    def retrieve():
        # the handle is content-addressed; which tool name it is keyed by differs by mode
        for tool in ("gh.api.items", "gh__gh.api.items"):
            h = _handle(tool, "result[].status", _serialize("active"))
            if h in cout_probe["out"].getvalue():
                return _req(3, "tools/call", {"name": "terse.retrieve",
                                              "arguments": {"handle": h}})
        raise AssertionError("no handle in the compressed result")

    # Paced with a late-bound third line: built once the result (and its handle) is out.
    cout = io.StringIO()
    cout_probe["out"] = cout
    steps_gen = _Paced(cout, [(INIT, ()), (_call(mode, 2, "gh.api.items"), (1,))])
    sent: list[str] = []

    def cin():
        for ln in steps_gen:
            sent.append(ln.rstrip("\n"))
            yield ln
        deadline = time.monotonic() + 20
        while 2 not in steps_gen._ids():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        last = retrieve()
        if escape:                    # W1: the same call with its method JSON-escaped
            last = last.replace("tools/call", "tools\\/call")
        sent.append(last)
        yield last + "\n"
        while 3 not in steps_gen._ids():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    wdir = tmp_path / "wire"
    if mode == "single":
        rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin(), stdout=cout,
                       wire_log=str(wdir))
    else:
        rc = run_multi_proxy(str(_peers_cfg(tmp_path)), pol, stdin=cin(), stdout=cout,
                             wire_log=str(wdir))
    assert rc == 0
    recs, out = _records(wdir), cout.getvalue()
    _check(recs, out, sent)
    hops = _hops(recs, 3)
    assert [h for h, _ in hops] == ["client_in", "client_out"]      # never forwarded
    (reply,) = [r for r in recs if r["hop"] == "client_out" and r["jsonrpc_id"] == 3]
    assert reply["origin"] == "retrieve"
    if capture:
        assert reply["capture_status"] == "local_answer"
        assert json.loads(reply["line"])["result"]["content"][0]["text"] == "active"
    else:
        assert {s for _, s in hops} == {"excluded_by_policy"}
        assert {s for _, s in _hops(recs, 2)} == {"excluded_by_policy"}


# --- lifecycle --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_server_crash_mid_session_records_peer_exit_and_stop(mode, tmp_path):
    kw = {} if mode == "single" else {"broadcast_timeout": 0.5}
    rc, out, recs, sent = _run(mode, tmp_path, [INIT, _call(mode, 2, "fake.crash")], **kw)
    _check(recs, out, sent)
    (exit_ev,) = _events(recs, "peer_exit")
    assert exit_ev["detail"]["exit_code"] == 3 and exit_ev["detail"]["pid"]
    if mode == "single":
        assert rc == 3
    else:
        (timeout,) = [r for r in recs if r["hop"] == "client_out" and r["jsonrpc_id"] == 2]
        assert timeout["origin"] == "timeout" and timeout["capture_status"] == "synthesized"
        assert json.loads(timeout["line"])["error"]["code"] == -32001


@pytest.mark.parametrize("mode", MODES)
def test_stop_is_written_before_reaping_a_peer_that_ignores_sigterm(mode, tmp_path):
    """#528: the client ends the session with SIGTERM while a peer ignores it, as kb's
    sb-run wrapper does. The reaper then waits ~2 s before SIGKILL; `stop` must not wait
    behind it, or a client that kills the proxy inside that window loses it."""
    cmd = [sys.executable, "-c",
           "import runpy, signal, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
           f"sys.argv = [{str(FAKE)!r}]; runpy.run_path(sys.argv[0], run_name='__main__')"]
    cout = io.StringIO()
    raised, done = [], threading.Event()

    def cin():
        yield INIT + "\n"
        deadline = time.monotonic() + 20
        while not any(json.loads(ln).get("id") == 1 for ln in cout.getvalue().splitlines()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raised.append(time.time_ns())
        os.kill(os.getpid(), signal.SIGTERM)    # the proxy's handler: SystemExit(143)
        done.wait(30)       # single mode reads stdin on a worker thread; hold it open

    wdir = tmp_path / "wire"
    t0 = time.monotonic()
    try:
        with pytest.raises(SystemExit):
            if mode == "single":
                run_proxy(cmd, FULL, stdin=cin(), stdout=cout, wire_log=str(wdir))
            else:
                run_multi_proxy(str(_peers_cfg(tmp_path, [{"name": "gh", "command": cmd}])),
                                FULL, stdin=cin(), stdout=cout, wire_log=str(wdir))
    finally:
        done.set()
    assert time.monotonic() - t0 >= 2.0          # the reaper did wait for SIGKILL
    recs = _records(wdir)
    _check(recs, cout.getvalue(), [INIT])
    (teardown,) = _events(recs, "teardown")
    assert teardown["detail"]["reason"] == "SystemExit"
    assert len(teardown["detail"]["live_peers"]) == 1
    assert recs[-2] is teardown
    assert (recs[-1]["ts_ns"] - raised[0]) / 1e9 < 1.0       # stop did not wait


def test_session_id_names_the_directory_and_field(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "8c1e-session")
    wdir = tmp_path / "w"
    cout = io.StringIO()
    run_proxy([sys.executable, str(FAKE)], FULL, stdin=io.StringIO(INIT + "\n"),
              stdout=cout, wire_log=str(wdir))
    (f,) = list(wdir.glob("*/*.jsonl"))
    assert f.parent.name == "8c1e-session"
    recs = _records(wdir)
    assert {r["claude_session"] for r in recs} == {"8c1e-session"}
    assert f.name == f"{recs[0]['conn']}.jsonl"
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    assert oct(f.parent.stat().st_mode & 0o777) == "0o700"


# --- multiproxy-only paths --------------------------------------------------------------

def _two_peers():
    return [{"name": "gh", "command": [sys.executable, str(FAKE)]},
            {"name": "gh2", "command": [sys.executable, str(FAKE)]}]


def test_broadcast_copies_and_merged_reply(tmp_path):
    rc, out, recs, sent = _run("multi", tmp_path, [INIT], peers=_two_peers())
    _check(recs, out, sent)
    copies = [r for r in recs if r["hop"] == "server_out" and r["method"] == "initialize"]
    assert sorted((r["peer"], r["jsonrpc_id"]) for r in copies) == [
        ("gh", "terse-b0-0"), ("gh2", "terse-b0-1")]
    assert {r["origin"] for r in copies} == {"router_broadcast"}
    assert len([r for r in recs if r["hop"] == "server_in"]) == 2
    (merged,) = [r for r in recs if r["hop"] == "client_out"]
    assert merged["origin"] == "router_broadcast" and merged["capture_status"] == "synthesized"


def test_snapshot_served_initialize_and_tools_list(tmp_path):
    lines = [INIT, INITIALIZED, _req(2, "tools/list")]
    peers = _two_peers()
    _run("multi", tmp_path, lines, peers=peers)              # cold: seeds the snapshot
    rc, out, recs, sent = _run("multi", tmp_path, lines, peers=peers)
    _check(recs, out, sent)
    served = {r["jsonrpc_id"]: r for r in recs if r["hop"] == "client_out"}
    assert served[1]["origin"] == served[2]["origin"] == "snapshot"
    internal = [r for r in recs if r["hop"] == "server_out" and r["method"] == "initialize"]
    assert {r["origin"] for r in internal} == {"router_internal"}


def test_unknown_tool_and_invalid_request_are_synthesized(tmp_path):
    lines = [_req(2, "tools/call", {"name": "nope__x"}), '{"jsonrpc":"2.0","id":[1],'
             '"method":"x"}', _req(4, "tools/call", [1])]
    rc, out, recs, sent = _run("multi", tmp_path, lines)
    _check(recs, out, sent)
    outs = [r for r in recs if r["hop"] == "client_out"]
    codes = [json.loads(r["line"])["error"]["code"] for r in outs]
    assert codes == [-32601, -32600, -32602]
    assert [r["origin"] for r in outs] == ["router", "error_writer", "error_writer"]


class _BlockingOut:
    def __init__(self, gate: threading.Event):
        self.gate, self.lines = gate, []

    def write(self, s):
        self.gate.wait(10)
        self.lines.append(s)

    def flush(self):
        pass


class _BlockingTransport:
    def __init__(self, gate):
        self.out = _BlockingOut(gate)

    def outbound(self):
        return self.out


def test_peer_queue_full_and_eviction_recorded_as_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_PEER_QUEUE_MAX", 2)
    tap = wirelog.open_wire_tap(str(tmp_path / "w"), "q")
    assert tap is not None
    gate = threading.Event()
    sender = _PeerSender(_BlockingTransport(gate), wire=tap, peer="slow")
    sender.send(_req(1, "ping"))
    time.sleep(0.2)            # the worker takes #1 and blocks writing it
    for i in (2, 3, 4, 5):
        sender.send(_req(i, "ping"))
    threading.Timer(0.3, gate.set).start()
    sender.close()
    tap.close()
    recs = _records(tmp_path / "w")
    got = {r["jsonrpc_id"]: (r["capture_status"], r["origin"])
           for r in recs if r["hop"] == "server_out"}
    assert len([r for r in recs if r["hop"] != "event"]) == 5          # once per line
    assert got == {1: ("full", None), 2: ("dropped", "evicted"), 3: ("full", None),
                   4: ("dropped", "queue_full"), 5: ("dropped", "queue_full")}


# --- the sink failing -------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_sink_write_error_warns_once_and_proxy_keeps_forwarding(mode, tmp_path, monkeypatch,
                                                                 capsys):
    class _Broken(io.StringIO):
        def write(self, s):
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(wirelog, "open_append_restricted", lambda p: _Broken())
    lines = [INIT] + [_call(mode, i, "gh.api.items") for i in (2, 3, 4)]
    cout = io.StringIO()
    wl = str(tmp_path / "w")
    if mode == "single":
        rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=io.StringIO(
            "".join(ln + "\n" for ln in lines)), stdout=cout, wire_log=wl)
    else:
        rc = run_multi_proxy(str(_peers_cfg(tmp_path)), FULL, stdin=io.StringIO(
            "".join(ln + "\n" for ln in lines)), stdout=cout, wire_log=wl)
    _, out_off, _, _ = _run(mode, tmp_path, lines, wire=False)
    assert rc == 0 and cout.getvalue() == out_off
    assert capsys.readouterr().err.count("wire log write") == 1


def test_unopenable_wire_dir_warns_and_proxy_runs(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cout = io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=io.StringIO(INIT + "\n"),
                   stdout=cout, wire_log=str(blocker))
    assert rc == 0 and json.loads(cout.getvalue())["id"] == 1
    assert "wire log disabled" in capsys.readouterr().err


def test_full_sink_queue_leaves_a_hole_explained_by_sink_drop(tmp_path, monkeypatch):
    release = threading.Event()
    real = wirelog.open_append_restricted

    class _Slow:
        def __init__(self, fh):
            self.fh = fh

        def write(self, s):
            release.wait(10)
            self.fh.write(s)

        def flush(self):
            self.fh.flush()

        def close(self):
            self.fh.close()

    monkeypatch.setattr(wirelog, "open_append_restricted", lambda p: _Slow(real(p)))
    tap = wirelog.open_wire_tap(str(tmp_path / "w"), "d", queue_max=3)
    assert tap is not None
    tap.tap("client_in", "first")
    time.sleep(0.2)                # the writer holds `first` and blocks
    for i in range(10):
        tap.tap("client_in", f"l{i}")
    release.set()
    time.sleep(0.2)
    tap.tap("client_in", "after")
    tap.close()
    recs = _records(tmp_path / "w")
    seqs = [r["seq"] for r in recs]
    holes = sorted(set(range(seqs[-1] + 1)) - set(seqs))
    drops = sum(r["detail"]["count"] for r in _events(recs, "sink_drop"))
    assert holes and drops == len(holes)
    assert recs[-2]["line"] == "after" and recs[-1]["event"] == "stop"


def test_http_synthesized_error_is_labelled(tmp_path):
    # The failed-POST path, without depending on how fast a dead port refuses.
    t = HttpTransport("http://127.0.0.1:9/mcp")
    t._maybe_enqueue_error(_req(7, "tools/call", {"name": "x"}), "terse: downstream down")
    raw = next(iter(t.inbound()))
    t.close()
    tap = wirelog.open_wire_tap(str(tmp_path / "w"), "h")
    assert tap is not None
    tap.tap("server_in", raw, peer="remote")
    tap.close()
    (rec,) = [r for r in _records(tmp_path / "w") if r["hop"] == "server_in"]
    assert rec["origin"] == "http_synth" and rec["capture_status"] == "synthesized"
    assert json.loads(rec["line"])["error"]["code"] == -32000


@pytest.mark.parametrize("extra", [[], ["--config", "peers.json"]])
def test_cli_threads_wire_log_to_both_modes(extra, monkeypatch):
    from terse.cli import main

    seen: dict = {}

    def fake(*_a, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr("terse.proxy.run_proxy", fake)
    monkeypatch.setattr("terse.multiproxy.run_multi_proxy", fake)
    tail = [] if extra else ["--", "some-server"]
    assert main(["proxy", "--no-stats", "--wire-log", "/w", *extra, *tail]) == 0
    assert seen["wire_log"] == "/w" and "policy_path" in seen
    seen.clear()
    assert main(["proxy", "--no-stats", *extra, *tail]) == 0
    assert "wire_log" not in seen


# --- the exclusion gate: nothing from a capture:false call or peer reaches the file -----
#
# Every test below sends a fresh token through one path and greps the RAW BYTES of the wire
# directory for it, so a leak through any field (line, a nested escape, an event detail)
# fails, not only one through `line`.

ASK_RULES = [Rule("*fake.echo", (), capture=False), Rule("*fake.ask", (), capture=False),
             Rule("*fake.progress", (), capture=False), Rule("gh.*", TIERS)]
SB = "secret-broker"


def _wire_bytes(wdir: pathlib.Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(wdir.rglob("*")) if p.is_file())


def _complete(recs: list[dict], out: str, sent: list[str]) -> None:
    """The completeness invariant, holes allowed only as counted sink drops: one client_in
    record per stdin line and one client_out record per stdout line, in order, each either
    the exact line or (excluded) its length alone."""
    seqs = [r["seq"] for r in recs]
    assert seqs == sorted(set(seqs))
    holes = set(range(seqs[-1] + 1)) - set(seqs)
    assert len(holes) == sum(r["detail"]["count"] for r in _events(recs, "sink_drop"))
    assert recs[0]["event"] == "start" and recs[-1]["event"] == "stop"
    assert recs[-1]["detail"]["records"] == seqs[-1] + 1
    for r in recs:
        if r["capture_status"] == "excluded_by_policy":
            assert r["line"] is None and r["sha256"] is None and r["rule"]
        elif r["hop"] != "event":
            assert r["sha256"] == _sha(r["line"]) and r["len"] == len(r["line"])
    cin = [r for r in recs if r["hop"] == "client_in"]
    assert [r["len"] for r in cin] == [len(s) for s in sent]
    for r, s in zip(cin, sent, strict=True):
        assert r["line"] in (None, s)
    written = out.split("\n")[:-1]
    couts = [r for r in recs if r["hop"] == "client_out"]
    assert [r["len"] for r in couts] == [len(ln) for ln in written]
    for r, ln in zip(couts, written, strict=True):
        assert r["line"] in (None, ln)


def _leak_run(mode, tmp_path, steps, pol, *, peers=None, **kw):
    """One paced session; returns (stdout, records, raw wire bytes, client lines)."""
    cout = io.StringIO()
    cin, sent = _Paced(cout, steps), [s for s, _ in steps]
    wdir = tmp_path / f"wire-{uuid.uuid4().hex[:6]}"
    if mode == "single":
        rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout,
                       wire_log=str(wdir), **kw)
    else:
        rc = run_multi_proxy(str(_peers_cfg(tmp_path, peers)), pol, stdin=cin, stdout=cout,
                             wire_log=str(wdir), broadcast_timeout=5, **kw)
    assert rc == 0
    out = cout.getvalue()
    recs = _records(wdir)
    _complete(recs, out, sent)
    return out, recs, _wire_bytes(wdir), sent


def _target_setup(mode, target, token, monkeypatch):
    """(policy, peers, run kwargs, excluded tool-name prefix) for an excluded TOOL on an
    otherwise-captured server, or an excluded whole PEER that leaks `token` everywhere."""
    if target == "tool":
        return Policy(rules=ASK_RULES), None, {}, ("" if mode == "single" else "gh__")
    if mode == "single":
        monkeypatch.setenv("FAKE_SECRET", token)
        pol = Policy(rules=[Rule(f"{SB}.*", (), capture=False, require_server_name=True)])
        return pol, None, {"server_name": SB}, ""
    env = {"FAKE_SECRET": token, "FAKE_TOOLS": f"fake.echo,fake.ask,tool-{token}"}
    peers = [{"name": SB, "command": [sys.executable, str(FAKE)], "env": env},
             {"name": "gh", "command": [sys.executable, str(FAKE)]}]
    pol = Policy(rules=[Rule(f"{SB}.*", (), capture=False), Rule("gh.*", TIERS)])
    return pol, peers, {}, f"{SB}__"


def _esc(s: str, rng) -> str:
    """`s` as a JSON string literal with a random subset of its characters \\u-escaped."""
    body = "".join(f"\\u{ord(c):04x}" if rng.random() < 0.5 else
                   json.dumps(c)[1:-1] for c in s)
    return f'"{body}"'


def _fuzz_call(mid, name, args, rng, extra_params="") -> str:
    """A tools/call whose method, keys and tool name are randomly escaped and spaced; the
    argument VALUES (which carry the token) are never escaped, so a leak stays greppable."""
    sp = " " * rng.randint(0, 2)
    keys = {k: _esc(k, rng) for k in ("jsonrpc", "id", "method", "params", "name",
                                      "arguments")}
    parts = [f"{keys['jsonrpc']}:{sp}\"2.0\"", f"{keys['id']}:{sp}{json.dumps(mid)}",
             f"{keys['method']}:{sp}{_esc('tools/call', rng)}",
             f"{keys['params']}:{sp}{{{keys['name']}:{sp}{_esc(name, rng)},{sp}"
             f"{keys['arguments']}:{sp}{json.dumps(args)}{extra_params}}}"]
    rng.shuffle(parts)
    return "{" + f",{sp}".join(parts) + "}"


def _vector_steps(vector, mode, target, token, pre):
    """Client steps for one leak vector: (line, ids that must be on stdout first)."""
    import random
    rng = random.Random(f"{vector}-{mode}-{target}")
    echo = f"{pre}fake.echo"
    ask_wait = "srv-1" if mode == "single" else "terse-s0"
    plain = _req(5, "tools/call", {"name": echo, "arguments": {"token": token}})
    if vector == "plain":
        return [(INIT, ()), (plain, (1,))]
    if vector == "escaped_method":       # W1: "tools\/call"
        return [(INIT, ()), (plain.replace("tools/call", "tools\\/call"), (1,))]
    if vector == "escaped_method_key":   # W1: "\u006dethod", no "tools/call" substring
        line = plain.replace('"method"', '"\\u006dethod"').replace(
            "tools/call", "tools\\u002fcall")
        return [(INIT, ()), (line, (1,))]
    if vector == "fuzzed_encodings":     # W1: 25 random encodings of the same call
        return [(INIT, ())] + [(_fuzz_call(10 + i, echo, {"token": token, "i": i}, rng),
                                (1,)) for i in range(25)]
    if vector in ("batch_first", "batch_last"):   # W1: a batch with one excluded call
        call = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": echo, "arguments": {"token": token}}}
        ping = {"jsonrpc": "2.0", "id": 8, "method": "ping"}
        batch = [call, ping] if vector == "batch_first" else [ping, call]
        return [(INIT, ()), (json.dumps(batch), (1,))]
    if vector == "float_and_string_ids":
        return [(INIT, ()), (plain.replace('"id": 5', '"id": 5.0'), (1,)),
                (plain.replace('"id": 5', '"id": "five"'), (1,))]
    if vector == "progress":             # a progress notification for an excluded call
        line = _req(5, "tools/call", {"name": f"{pre}fake.echo", "arguments": {
            "token": token}, "_meta": {"progressToken": f"pt-{uuid.uuid4().hex[:4]}"}})
        return [(INIT, ()), (line, (1,))]
    if vector.startswith("ask_"):         # W3: a server request during the excluded call
        method = {"ask_elicit": "elicitation/create", "ask_sampling":
                  "sampling/createMessage", "ask_roots": "roots/list"}[vector]
        call = _req(5, "tools/call", {"name": f"{pre}fake.ask", "arguments": {
            "method": method}})
        reply = json.dumps({"jsonrpc": "2.0", "id": ask_wait,
                            "result": {"action": "accept", "content": {"pw": token}}})
        return [(INIT, ()), (call, (1,)), (reply, (ask_wait,))]
    if vector == "cancel":               # a cancellation naming the excluded call
        cancel = json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                             "params": {"requestId": 5, "reason": token}})
        return [(INIT, ()), (plain, (1,)), (cancel, (5,))]
    # W4: any method to a whole-server-excluded peer, carrying the token
    method = vector.removeprefix("method_")
    params = {"argument": {"name": "x", "value": token}, "uri": f"x://{token}",
              "level": "debug", "name": f"{pre}p", "ref": {"type": "ref/prompt",
                                                           "name": token}}
    return [(INIT, ()), (_req(9, method, params), (1,))]


TOOL_VECTORS = ["plain", "escaped_method", "escaped_method_key", "fuzzed_encodings",
                "batch_first", "batch_last", "float_and_string_ids", "progress",
                "ask_elicit", "ask_sampling", "ask_roots", "cancel"]
PEER_VECTORS = TOOL_VECTORS + [f"method_{m}" for m in (
    "completion/complete", "resources/read", "resources/subscribe", "resources/unsubscribe",
    "resources/list", "resources/templates/list", "prompts/list", "tools/list", "ping",
    "logging/setLevel", "prompts/get")]
LEAK_CASES = ([(m, "tool", v) for m in MODES for v in TOOL_VECTORS]
              + [(m, "peer", v) for m in MODES for v in PEER_VECTORS])


@pytest.mark.parametrize(("mode", "target", "vector"), LEAK_CASES)
def test_excluded_content_never_reaches_the_wire_file(mode, target, vector, tmp_path,
                                                      monkeypatch):
    token = f"TOK{uuid.uuid4().hex}"
    pol, peers, kw, pre = _target_setup(mode, target, token, monkeypatch)
    steps = _vector_steps(vector, mode, target, token, pre)
    out, recs, raw, _ = _leak_run(mode, tmp_path, steps, pol, peers=peers, **kw)
    assert token.encode() not in raw
    assert any(r["capture_status"] == "excluded_by_policy" for r in recs)
    if vector in ("plain", "escaped_method", "progress") and (
            target == "tool" or mode == "single"):
        assert token in out                  # the path really carried it; forwarding intact


def test_merged_lists_and_resources_read_from_an_excluded_peer(tmp_path):
    """W2: a whole-server-excluded peer's resources/read answer and its share of every
    merged list are excluded at every hop, cold (broadcast) and warm (snapshot-served)."""
    token = f"TOK{uuid.uuid4().hex}"
    env = {"FAKE_SECRET": token, "FAKE_TOOLS": f"fake.echo,tool-{token}"}
    peers = [{"name": "gh", "command": [sys.executable, str(FAKE)]},
             {"name": SB, "command": [sys.executable, str(FAKE)], "env": env}]
    pol = Policy(rules=[Rule(f"{SB}.*", (), capture=False), Rule("gh.*", TIERS)])
    lists = ["tools/list", "prompts/list", "resources/list", "resources/templates/list"]
    steps = [(INIT, ()), (INITIALIZED, (1,))] + [
        (_req(10 + i, m), (1,)) for i, m in enumerate(lists)] + [
        (_req(20, "resources/read", {"uri": "x://1"}), tuple(range(10, 14)))]
    for _run_no in ("cold", "warm"):
        out, recs, raw, _ = _leak_run("multi", tmp_path, steps, pol, peers=peers)
        assert token in out and token.encode() not in raw
        merged = [r for r in recs if r["hop"] == "client_out" and r["jsonrpc_id"] in
                  (1, 10, 11, 12, 13, 20)]
        assert {r["capture_status"] for r in merged} == {"excluded_by_policy"}


def _gate_tap(tmp_path, resolve, **kw):
    tap = wirelog.open_wire_tap(str(tmp_path / "w"), "u", **kw)
    assert tap is not None
    tap.resolve = resolve
    return tap


def _echo_resolve(msg):
    params = msg.get("params")
    if msg.get("method") == "tools/call" and isinstance(params, dict):
        name = params.get("name")
        return ("*fake.echo", True) if name == "fake.echo" else ("gh.*", False)
    return None


def test_in_flight_excluded_id_is_never_evicted(tmp_path):
    """W5: an excluded call still awaiting its reply survives 10k later excluded calls;
    its late reply (on both hops) is excluded."""
    token = f"TOK{uuid.uuid4().hex}"
    tap = _gate_tap(tmp_path, _echo_resolve, queue_max=40000)
    for i in range(10_001):
        line = _req(i, "tools/call", {"name": "fake.echo", "arguments": {}})
        tap.tap("client_in", line)
        tap.tap("server_out", line, peer="gh")
    late = json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"t": token}})
    tap.tap("server_in", late, peer="gh")
    tap.tap("client_out", late)
    tap.close()
    raw = _wire_bytes(tmp_path / "w")
    assert raw and token.encode() not in raw
    recs = _records(tmp_path / "w")
    assert [r["capture_status"] for r in recs if r["jsonrpc_id"] == 0] == [
        "excluded_by_policy"] * 4


def test_progress_token_of_an_excluded_call_is_excluded(tmp_path):
    """notifications/progress carries no id: it is matched to the excluded call through
    the request's params._meta.progressToken, on the client side as on the server side,
    including after the call completed."""
    token = f"TOK{uuid.uuid4().hex}"
    tap = _gate_tap(tmp_path, _echo_resolve)
    call = _req(3, "tools/call", {"name": "fake.echo", "arguments": {},
                                  "_meta": {"progressToken": 77}})
    prog = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                       "params": {"progressToken": 77, "message": token}})
    other = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": 78, "message": "fine"}})
    tap.tap("client_in", call)
    # inherit=None: what the server pump thread writing these would carry (this test
    # thread just READ the excluded call, which client_out here would otherwise inherit)
    tap.tap("client_out", prog, inherit=None)
    tap.tap("client_out", json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}),
            inherit=None)
    tap.tap("client_out", prog, inherit=None)         # late: after the reply
    tap.tap("client_out", other, inherit=None)
    tap.close()
    raw = _wire_bytes(tmp_path / "w")
    assert token.encode() not in raw and b"fine" in raw


def test_writer_exception_warns_once_and_close_is_bounded(tmp_path, monkeypatch, capsys):
    """W6: an unexpected exception in the writer thread warns once on stderr, turns the
    log off, never reaches forwarding, and close() does not wait out its timeout."""
    def boom(*_a, **_k):
        raise RuntimeError("render bug")

    monkeypatch.setattr(wirelog.WireTap, "_render", boom)
    lines = [INIT] + [_call("single", i, "gh.api.items") for i in (2, 3)]
    cout = io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=io.StringIO(
        "".join(ln + "\n" for ln in lines)), stdout=cout, wire_log=str(tmp_path / "p"))
    _, out_off, _, _ = _run("single", tmp_path, lines, wire=False)
    assert rc == 0 and cout.getvalue() == out_off
    assert capsys.readouterr().err.count("[terse] wire log") == 1
    tap = wirelog.open_wire_tap(str(tmp_path / "w"), "x", queue_max=8)
    assert tap is not None
    tap.tap("client_in", "first")
    time.sleep(0.2)                                   # the writer dies on `first`
    for i in range(100):
        tap.tap("client_in", f"l{i}")
    t0 = time.monotonic()
    tap.close()
    assert time.monotonic() - t0 < 1.5
    assert tap.dead_drops >= 100
    assert capsys.readouterr().err.count("[terse] wire log") == 1


def test_tap_racing_close_never_lands_after_stop(tmp_path):
    """W7: taps racing close() are either written before `stop` (numbered below it) or
    counted as late; never a record after stop, and stop's count is exact."""
    workers, per = 4, 200
    for n in range(60):
        tap = wirelog.open_wire_tap(str(tmp_path / f"w{n}"), "r")
        assert tap is not None
        barrier = threading.Barrier(workers + 1)

        def spam(k, tap=tap, barrier=barrier):
            barrier.wait()
            for i in range(per):
                tap.tap("client_in", f"t{k}-{i}")

        threads = [threading.Thread(target=spam, args=(k,)) for k in range(workers)]
        for t in threads:
            t.start()
        barrier.wait()
        tap.close()
        for t in threads:
            t.join()
        recs = _records(tmp_path / f"w{n}")
        stop = recs[-1]
        assert stop["event"] == "stop"
        assert max(r["seq"] for r in recs) == stop["seq"]
        assert stop["detail"]["records"] == stop["seq"] + 1
        assert [r["seq"] for r in recs] == list(range(len(recs)))
        lines = [r for r in recs if r["hop"] == "client_in"]
        assert len(lines) + tap.late_drops == workers * per


def test_http_synthesized_reply_keeps_its_label_at_client_out(tmp_path):
    """W9: the -32000 the HTTP transport synthesizes is labelled http_synth on server_in
    AND client_out (the pump used to strip the label along with the newline)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                     # closed again: connection refused
    cout = io.StringIO()
    wdir = tmp_path / "w"
    rc = run_proxy([f"http://127.0.0.1:{port}/mcp"], FULL, stdin=_Paced(cout, [(INIT, ())]),
                   stdout=cout, wire_log=str(wdir))
    assert rc == 0
    recs = _records(wdir)
    got = [(r["hop"], r["origin"], r["capture_status"]) for r in recs
           if r["jsonrpc_id"] == 1 and r["hop"] in ("server_in", "client_out")]
    assert got == [("server_in", "http_synth", "synthesized"),
                   ("client_out", "http_synth", "synthesized")]


# --- an excluded call whose reply can't be matched by id fails closed (#524) ---------

def _echo(mid_json: str, args: str = "{}") -> str:
    return ('{"jsonrpc":"2.0","id":' + mid_json + ',"method":"tools/call","params":'
            '{"name":"fake.echo","arguments":' + args + '}}')


UNMATCHABLE = {
    # (request line, the id the server answers with)
    "big_int": (_echo("5", '{"n":' + "7" * 4400 + "}"), 5),
    "deep_nesting": (_echo("5", "[" * 150_000 + "]" * 150_000), 5),
    "duplicate_id_first_key_wins": (_echo("5,\"id\":6"), 5),
    "null_id": (_echo("null"), None),
    "string_id_coerced": (_echo('"5"'), 5),
}


@pytest.mark.parametrize("case", sorted(UNMATCHABLE))
def test_an_unmatchable_excluded_call_never_leaks_its_reply(case, tmp_path):
    token = f"TOK{uuid.uuid4().hex}"
    tap = _gate_tap(tmp_path, _echo_resolve)
    line, answered = UNMATCHABLE[case]
    reply = json.dumps({"jsonrpc": "2.0", "id": answered, "result": {"t": token}})
    tap.tap("client_in", line)
    tap.tap("server_out", line, peer="gh")
    tap.tap("server_in", reply, peer="gh")
    tap.tap("client_out", reply)
    tap.close()
    raw = _wire_bytes(tmp_path / "w")
    assert raw and token.encode() not in raw
    recs = [r for r in _records(tmp_path / "w") if r["hop"] != "event"]
    assert [r["capture_status"] for r in recs] == ["excluded_by_policy"] * 4


def test_an_int_past_the_digit_limit_is_undecidable_not_a_parse_error(tmp_path):
    # #524: the int-limit ValueError was read as "not JSON", so an excluded request
    # carrying one was logged in full as a parse_error.
    assert wirelog._parse(_echo("5", '{"n":' + "7" * 4400 + "}"), strict=True)[0] == \
        "undecidable"
    assert wirelog._parse("{not json", strict=True)[0] == "bad"


def test_fail_closed_ends_when_the_peer_exits(tmp_path):
    tap = _gate_tap(tmp_path, _echo_resolve)
    line = _echo("null")
    tap.tap("client_in", line)
    tap.tap("server_out", line, peer="gh")
    later = json.dumps({"jsonrpc": "2.0", "method": "notifications/message",
                        "params": {"data": "fine"}})
    tap.tap("server_in", later, peer="gh")
    tap.event("peer_exit", peer="gh")
    tap.tap("server_in", later, peer="gh")
    tap.close()
    recs = [r for r in _records(tmp_path / "w") if r["hop"] == "server_in"]
    assert [r["rule"] for r in recs] == [wirelog.RULE_UNMATCHED, None]
    assert recs[1]["line"] == later


def test_without_any_capture_false_rule_an_unparseable_call_is_logged_in_full(tmp_path):
    # A2: with nothing excludable, a 4400-digit int in an ordinary call must not hide
    # the server's later lines from a fleet that excludes nothing.
    tap = _gate_tap(tmp_path, lambda msg: None, may_exclude=False)
    line, _ = UNMATCHABLE["big_int"]
    reply = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"t": "fine"}})
    tap.tap("client_in", line)
    tap.tap("server_out", line, peer="gh")
    tap.tap("server_in", reply, peer="gh")
    tap.close()
    recs = [r for r in _records(tmp_path / "w") if r["hop"] != "event"]
    assert [r["line"] for r in recs] == [line, line, reply]


def test_may_exclude_reads_every_policy():
    assert not wirelog.may_exclude([FULL])
    assert wirelog.may_exclude([FULL, Policy(rules=ASK_RULES)])


# --- parsing stops once nothing excluded is in flight (#525) ---------------------------

def test_an_answered_excluded_call_does_not_keep_parsing_every_line(tmp_path, monkeypatch):
    token = f"TOK{uuid.uuid4().hex}"
    tap = _gate_tap(tmp_path, _echo_resolve)
    call = _echo("3")
    reply = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}})
    tap.tap("client_in", call)
    tap.tap("server_out", call, peer="gh")
    tap.tap("server_in", reply, peer="gh")
    tap.tap("client_out", reply)
    parsed: list[str] = []
    real = wirelog._parse
    monkeypatch.setattr(wirelog, "_parse",
                        lambda line, strict: parsed.append(line[:20]) or real(line, strict))
    big = json.dumps({"jsonrpc": "2.0", "id": 9, "result": {"x": "y" * 1_000_000,
                                                            "id": 4}})
    tap.tap("server_in", big, peer="gh")
    tap.tap("client_out", big)
    assert parsed == []                       # nothing on either side names id 3
    late = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"t": token}})
    tap.tap("server_in", late, peer="gh")     # a late duplicate: still excluded
    tap.tap("client_out", late)
    assert len(parsed) == 2
    tap.close()
    assert token.encode() not in _wire_bytes(tmp_path / "w")


def test_resolve_runs_outside_the_tap_lock(tmp_path):
    held: list[bool] = []
    tap = _gate_tap(tmp_path, None)
    tap.resolve = lambda msg: held.append(tap._lock.locked()) or _echo_resolve(msg)
    tap.tap("client_in", json.dumps([json.loads(_echo("1")), json.loads(_echo("2"))]))
    tap.close()
    assert held == [False, False]


@pytest.mark.parametrize(("line", "parse"), [
    ('{"id":3,"result":{}}', True),
    ('{"id":"3","result":{}}', True),         # coerced spelling
    ('{"id": 3.0,"result":{}}', True),        # odd spelling: decoded, then matched
    ('{"id":3e0,"result":{}}', True),
    ('{"method":"notifications/progress","params":{"progressToken":3}}', True),
    ('{"id":4,"result":{"id":5}}', False),
    (json.dumps({"id": 4, "result": [{"id": i} for i in range(100, 400)]}), True),  # dense
])
def test_the_pre_scan_parses_exactly_the_lines_that_may_name_an_answered_key(line, parse):
    scope = wirelog._Scope(None, {})
    scope._register(("p:gh", "c", "3"), "r", [])
    scope._complete(("p:gh", "c", "3"))
    assert scope.needs_parse("server_in", "gh", None, line) is parse


def test_the_pre_scan_sees_keys_answered_after_its_last_scan():
    scope = wirelog._Scope(None, {})
    for mid in ("3", "7"):
        scope._register(("p:gh", "c", mid), "r", [])
        scope._complete(("p:gh", "c", mid))
        assert scope.needs_parse("server_in", "gh", None, f'{{"id":{mid},"result":{{}}}}')
