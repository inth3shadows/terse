"""Multi-downstream fan-out proxy (#5 Half B): merge, prefix-routing, shared drop
store, broadcast timeout.

Reuses `tests/fake_mcp_server.py` (stdio) exactly like test_proxy.py, and an
in-process `http.server` fake mirroring test_transport.py's `_Handler`/`_fake_server`
pattern for the second (HTTP) peer — so a config can front one of each, matching the
plan's "mixed stdio+HTTP peers" scenario.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import pathlib
import sys
import threading
import time
from collections import OrderedDict
from threading import Lock

from terse import __version__, transforms
from terse.lossy import _handle, _serialize
from terse.multiproxy import (
    DownstreamSpec,
    Peer,
    Router,
    _build_peers,
    load_multi_config,
    run_multi_proxy,
)
from terse.policy import Policy, Rule
from terse.proxy import SWALLOW, Interceptor
from terse.transport import build_transport

FAKE = pathlib.Path(__file__).parent / "fake_mcp_server.py"
TIERS = ("minify", "tabularize", "dictionary")

RECORDS = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]


# --- in-process HTTP fake, mirroring test_transport.py's _Handler/_fake_server ---

class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # silence test output
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            msg = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            msg = {}
        self.server.requests.append(msg)  # type: ignore[attr-defined]
        method, mid = msg.get("method"), msg.get("id")

        if method == "initialize":
            self._send_json({"jsonrpc": "2.0", "id": mid,
                             "result": {"protocolVersion": "2024-11-05",
                                        "capabilities": {"http_peer": True},
                                        "serverInfo": {"name": "fake-http", "version": "0"},
                                        "instructions": "HTTP PEER NOTES."}})
            return
        if method == "tools/call":
            name = (msg.get("params") or {}).get("name")
            if name == "items.body":
                text = json.dumps({"result": [{"id": 1, "body": "B" * 400}]})
            else:
                text = json.dumps({"result": RECORDS})
            self._send_json({"jsonrpc": "2.0", "id": mid,
                             "result": {"content": [{"type": "text", "text": text}],
                                        "isError": False}})
            return
        if method == "tools/list":
            self._send_json({"jsonrpc": "2.0", "id": mid,
                             "result": {"tools": [{"name": "items.get"}, {"name": "items.body"}]}})
            return
        # notification or anything unrecognized: 202 Accepted, empty body.
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, obj: dict) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def _fake_http():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        thread.join(timeout=2)
        srv.server_close()


def _url(srv) -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}/mcp"


def _write_config(tmp_path, downstreams: list[dict]) -> pathlib.Path:
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": downstreams}), encoding="utf-8")
    return cfg


def _lines(cout: io.StringIO) -> list[dict]:
    return [json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()]


class _FakePeerTransport:
    """A minimal `Transport` for unit-testing `Router` without a real subprocess/HTTP
    peer: `outbound()` always returns the SAME `io.StringIO` (so a `_PeerSender`'s
    writes accumulate and stay inspectable after the fact), `inbound()` yields nothing
    (these tests drive `from_peer(i)`'s transform directly instead)."""

    def __init__(self):
        self.out = io.StringIO()

    def inbound(self):
        return iter([])

    def outbound(self):
        return self.out

    def close(self):
        pass


DROP_POLICY = Policy(rules=[
    Rule("gh.*", TIERS, fields={"result[].status": {"lossy": "drop-to-retrieve", "min": 1}}),
    Rule("items.*", TIERS, fields={"result[].body": {"lossy": "drop-to-retrieve"}}),
])

PLAIN_POLICY = Policy(rules=[Rule("gh.*", TIERS), Rule("items.*", TIERS)])


# --- 1: tools/list merges + prefixes + single retrieve ---

def test_tools_list_merges_prefixes_and_single_retrieve(tmp_path):
    with _fake_http() as srv:
        cfg = _write_config(tmp_path, [
            {"name": "gh", "command": [sys.executable, str(FAKE)]},
            {"name": "http", "url": _url(srv)},
        ])
        cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                                      "params": {}}) + "\n")
        cout = io.StringIO()
        rc = run_multi_proxy(str(cfg), DROP_POLICY, stdin=cin, stdout=cout)
    assert rc == 0
    msgs = _lines(cout)
    assert len(msgs) == 1
    names = [t["name"] for t in msgs[0]["result"]["tools"]]
    assert "gh__gh.api.items" in names and "gh__fs.read" in names
    assert "http__items.get" in names and "http__items.body" in names
    assert names.count("terse.retrieve") == 1  # advertised exactly once, not per-peer


# --- 2: tools/call routes by prefix and rewrites the name ---

def _log_text(n, changed_line=None):
    lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}" for i in range(n)]
    if changed_line is not None:
        lines[changed_line] = "[ERROR] worker crashed: connection reset"
    return "\n".join(lines)


def test_stats_ledger_records_peer_name_and_qualified_tool(tmp_path):
    from terse.stats import load_stats
    cfg = _write_config(tmp_path, [{"name": "gh", "command": [sys.executable, str(FAKE)]}])
    log = tmp_path / "stats.jsonl"
    cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                  "params": {"name": "gh__gh.api.items"}}) + "\n")
    cout = io.StringIO()
    rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout,
                         stats_log=str(log))
    assert rc == 0
    recs = load_stats(log)
    assert len(recs) == 1
    # server = the peer's config name; tool = the peer-qualified name the client sees
    assert recs[0]["server"] == "gh" and recs[0]["tool"] == "gh__gh.api.items"
    assert "active" not in log.read_text(encoding="utf-8")   # payload-free


def test_broken_stats_sink_warns_once_per_peer_under_the_multiproxy_prefix(tmp_path,
                                                                          capsys):
    # #131: sink-failure reporting moved into the Interceptor, so the line must still say
    # [terse-multiproxy], not the single-proxy default. The Interceptor — and with it the
    # warn-once bookkeeping — is PER PEER here, so the guard is once-per-(peer, kind), not
    # once per kind: two peers hitting the same dead sink say so twice, each attributed to
    # itself. That attribution is the point; the flood guard still holds within a peer.
    cfg = _write_config(tmp_path, [{"name": "gh", "command": [sys.executable, str(FAKE)]},
                                   {"name": "gh2", "command": [sys.executable, str(FAKE)]}])
    cin = io.StringIO("\n".join(
        json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                    "params": {"name": f"{peer}__gh.api.items"}})
        # two calls per peer: the SECOND must be silenced, the first must not
        for i, peer in enumerate(["gh", "gh", "gh2", "gh2"], start=2)) + "\n")
    cout = io.StringIO()
    rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout,
                         stats_log=str(tmp_path))     # a DIRECTORY — every append fails
    assert rc == 0
    warnings = [ln for ln in capsys.readouterr().err.splitlines() if "stats skipped" in ln]
    assert sorted(ln.split(":")[0] for ln in warnings) == [
        "[terse-multiproxy] gh2__gh.api.items", "[terse-multiproxy] gh__gh.api.items"]
    # all four calls still answered — a dead ledger is never load-bearing
    assert len([ln for ln in cout.getvalue().splitlines() if ln.strip()]) == 4


def test_tools_call_routes_by_prefix_and_strips_it(tmp_path):
    with _fake_http() as srv:
        cfg = _write_config(tmp_path, [
            {"name": "gh", "command": [sys.executable, str(FAKE)]},
            {"name": "http", "url": _url(srv)},
        ])
        # fake_mcp_server.py's "fs.read" branch only fires on the EXACT bare name — an
        # un-stripped "gh__fs.read" would instead fall through to its default RECORDS
        # branch, so getting the log text back is proof the router actually stripped
        # the prefix before the downstream ever saw the call.
        cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                      "params": {"name": "gh__fs.read"}}) + "\n")
        cout = io.StringIO()
        rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout)
        # the http peer's fake never received ANY request -- proves the call reached
        # only the targeted (gh) peer.
        assert srv.requests == []
    assert rc == 0
    msgs = _lines(cout)
    assert len(msgs) == 1 and msgs[0]["id"] == 2  # original client id, unchanged
    assert msgs[0]["result"]["content"][0]["text"] == _log_text(200)


# --- 3: initialize broadcast merges once ---

def test_initialize_broadcast_merges_once(tmp_path):
    with _fake_http() as srv:
        cfg = _write_config(tmp_path, [
            {"name": "gh", "command": [sys.executable, str(FAKE)]},
            {"name": "http", "url": _url(srv)},
        ])
        cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {}}) + "\n")
        cout = io.StringIO()
        rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout)
        reached_http = any(m.get("method") == "initialize" for m in srv.requests)
    assert rc == 0
    msgs = _lines(cout)
    assert len(msgs) == 1                          # one merged reply, not two
    result = msgs[0]["result"]
    # a single TERSE_PRIMER, not duplicated, plus the http peer's own instructions
    assert result["instructions"].count("Some tool results are 'terse'-compressed") == 1
    assert "HTTP PEER NOTES." in result["instructions"]
    # both servers actually reached: http proven via its request log, gh via the
    # marker capability its fake sets specifically for this (see fake_mcp_server.py)
    assert reached_http
    assert result["capabilities"] == {"http_peer": True, "stdio_peer": True}
    assert result["serverInfo"] == {"name": "terse", "version": __version__}


# --- 4: shared drop store across peers ---

def test_shared_drop_store_across_peers(tmp_path):
    # The two request/response legs are driven BY HAND (not through the threaded
    # run_multi_proxy pipeline): a `tools/call` write and its reply arriving on a
    # peer's own reader thread race the client->server loop moving on to the next
    # line, so "drop it, then immediately retrieve it" can't be made deterministic
    # over the live threaded proxy in one input stream (same reasoning as
    # test_transport.py's HTTP drop-to-retrieve test). This still drives the real
    # Router/Peer/Interceptor production code, just sequenced synchronously.
    with _fake_http() as srv:
        gh_transport = build_transport([sys.executable, str(FAKE)])
        http_transport = build_transport([_url(srv)])
        try:
            store: OrderedDict[str, object] = OrderedDict()
            store_lock = Lock()
            gh_inter = Interceptor(DROP_POLICY, store=store, store_lock=store_lock)
            http_inter = Interceptor(DROP_POLICY, store=store, store_lock=store_lock)
            peers = [Peer("gh", gh_transport, gh_inter), Peer("http", http_transport, http_inter)]
            out = io.StringIO()
            router = Router(peers, out, Lock())

            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "gh__gh.api.items"}}))
            line_a = next(iter(gh_transport.inbound()))
            text_a = gh_inter.transform_response(line_a)
            assert transforms.DROPPED_MARKER in text_a

            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "http__items.body"}}))
            line_b = next(iter(http_transport.inbound()))
            text_b = http_inter.transform_response(line_b)
            assert transforms.DROPPED_MARKER in text_b

            # two DIFFERENT dropped values -> two DISTINCT handles in the ONE shared
            # store (no per-peer isolation, no collision)
            assert len(store) == 2
            handle_gh = _handle("gh.api.items", "result[].status", _serialize("active"))
            handle_http = _handle("items.body", "result[].body", _serialize("B" * 400))
            assert handle_gh != handle_http
            assert set(store) == {handle_gh, handle_http}

            # answered peer-agnostically from the client's view: retrieve routes through
            # peers[0] (gh) internally for BOTH handles, yet resolves the http-dropped
            # one correctly too, because the store is shared.
            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "terse.retrieve", "arguments": {"handle": handle_gh}}}))
            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "terse.retrieve", "arguments": {"handle": handle_http}}}))
        finally:
            gh_transport.close()
            http_transport.close()

    msgs = {m["id"]: m for m in _lines(out)}
    assert not msgs[3]["result"].get("isError")
    assert msgs[3]["result"]["content"][0]["text"] == "active"
    assert not msgs[4]["result"].get("isError")
    assert msgs[4]["result"]["content"][0]["text"] == "B" * 400


# --- 5: one dead/timing-out peer doesn't wedge the broadcast ---

def test_dead_peer_does_not_wedge_broadcast_or_live_routed_calls(tmp_path, capsys):
    # A stdio child that drains stdin but NEVER writes a reply -- the "server that
    # hangs forever" case. A short broadcast_timeout override keeps this test fast
    # instead of waiting out the real 30s default.
    hang_cmd = [sys.executable, "-c", "import sys\nfor _ in sys.stdin:\n    pass\n"]
    cfg = _write_config(tmp_path, [
        {"name": "gh", "command": [sys.executable, str(FAKE)]},
        {"name": "dead", "command": hang_cmd},
    ])
    requests_text = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "gh__gh.api.items"}}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "dead__whatever"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests_text), io.StringIO()
    rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout, broadcast_timeout=0.3)
    assert rc == 0

    err = capsys.readouterr().err
    assert "timed out" in err and "dead" in err  # the missing peer is named on stderr

    msgs = {m["id"]: m for m in _lines(cout)}
    assert 1 in msgs                                     # merged reply still went out
    assert msgs[1]["result"]["capabilities"] == {"stdio_peer": True}  # only the live peer's
    assert 2 in msgs                                     # the live peer still serves routed calls
    text = msgs[2]["result"]["content"][0]["text"]
    assert transforms.decompress(text) == {"result": RECORDS}
    # a routed call TO the dead peer must not wedge the client forever either —
    # it gets a timeout error instead of never answering.
    assert 3 in msgs
    assert "error" in msgs[3] and "timed out" in msgs[3]["error"]["message"]


# --- config loading / validation ---

def test_load_multi_config_parses_command_and_url_downstreams(tmp_path):
    cfg = _write_config(tmp_path, [
        {"name": "gh", "command": ["uvx", "gh-mcp"]},
        {"name": "kb", "url": "https://kb.example/mcp", "headers": {"Authorization": "x"},
         "policy": "kb.json"},
    ])
    (tmp_path / "kb.json").write_text("{}", encoding="utf-8")  # just needs to exist
    specs = load_multi_config(str(cfg))
    assert [s.name for s in specs] == ["gh", "kb"]
    assert specs[0].target == ["uvx", "gh-mcp"]
    assert specs[1].target == ["https://kb.example/mcp"]
    assert specs[1].headers == {"Authorization": "x"}
    # a relative policy path resolves against the CONFIG file's directory, not cwd
    assert specs[1].policy_path == str(tmp_path / "kb.json")


def test_load_multi_config_rejects_duplicate_names(tmp_path):
    cfg = _write_config(tmp_path, [
        {"name": "gh", "command": ["a"]},
        {"name": "gh", "command": ["b"]},
    ])
    try:
        load_multi_config(str(cfg))
        raise AssertionError("expected ValueError for a duplicate downstream name")
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_load_multi_config_rejects_missing_target(tmp_path):
    cfg = _write_config(tmp_path, [{"name": "gh"}])
    try:
        load_multi_config(str(cfg))
        raise AssertionError("expected ValueError for a downstream with no command/url")
    except ValueError as e:
        assert "gh" in str(e)


def test_cmd_proxy_rejects_config_and_positional_cmd_together():
    from terse.cli import main
    rc = main(["proxy", "--config", "whatever.json", "--", "uvx", "some-mcp"])
    assert rc == 2


# --- Interceptor store/store_lock injection (#5 Half B, step 1) ---

def test_interceptor_default_store_is_private_and_unaffected():
    a = Interceptor(DROP_POLICY)
    b = Interceptor(DROP_POLICY)
    a._drop_put("h", "value")
    assert "h" in a.dropped and "h" not in b.dropped  # no accidental sharing by default


def test_interceptor_injected_store_is_actually_shared():
    store: OrderedDict[str, object] = OrderedDict()
    lock = Lock()
    a = Interceptor(DROP_POLICY, store=store, store_lock=lock)
    b = Interceptor(DROP_POLICY, store=store, store_lock=lock)
    a._drop_put("h", "value")
    assert b.dropped["h"] == "value"                 # visible from the OTHER Interceptor
    reply = json.loads(b.answer_retrieve(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "terse.retrieve", "arguments": {"handle": "h"}}})))
    assert reply["result"]["content"][0]["text"] == "value"


def test_shared_dropped_bytes_evicts_over_combined_cap_across_peers():
    # Regression: each Interceptor's own byte counter used to be private even when the
    # DICT was shared (multiproxy._build_peers), so the DROPPED_MAX_BYTES cap never saw
    # the true combined size — two peers each individually under-cap could jointly blow
    # way past it. A shared `dropped_bytes` box fixes that.
    store: OrderedDict[str, object] = OrderedDict()
    lock = Lock()
    dropped_bytes: list[int] = [0]
    a = Interceptor(DROP_POLICY, store=store, store_lock=lock, dropped_bytes=dropped_bytes)
    b = Interceptor(DROP_POLICY, store=store, store_lock=lock, dropped_bytes=dropped_bytes)
    a.DROPPED_MAX_BYTES = b.DROPPED_MAX_BYTES = 25

    a._drop_put("a", "x" * 10)                        # peer a: 10 bytes, under its own cap
    b._drop_put("b", "y" * 10)                         # peer b: 10 bytes, under its own cap
    a._drop_put("c", "z" * 10)                         # combined 30 > 25 -> evict oldest ("a")

    assert "a" not in store                            # evicted despite peer `a` alone never
                                                        # exceeding 25 bytes on its own
    assert set(store) == {"b", "c"}
    assert dropped_bytes[0] == 20


def test_build_peers_closes_already_launched_peer_on_partial_failure(monkeypatch):
    # Regression: _build_peers used to let an OSError from a later spec propagate with
    # no cleanup, orphaning an earlier spec's already-launched child/connection.
    from terse import multiproxy as mp

    closed = []

    class _FakeTransport:
        def inbound(self):
            return iter([])

        def outbound(self):
            return io.StringIO()

        def close(self):
            closed.append(True)

    calls = {"n": 0}

    def fake_build_transport(target, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeTransport()
        raise OSError("boom: can't launch second peer")

    monkeypatch.setattr(mp, "build_transport", fake_build_transport)
    specs = [
        DownstreamSpec(name="a", target=["a"], headers={}, policy_path=None),
        DownstreamSpec(name="b", target=["b"], headers={}, policy_path=None),
    ]
    try:
        _build_peers(specs, PLAIN_POLICY, debug=False, capture=None, audit=None,
                     store=OrderedDict(), store_lock=Lock(), dropped_bytes=[0])
        raise AssertionError("expected OSError for the unlaunchable 2nd peer")
    except OSError:
        pass
    assert closed == [True]  # the first (already-launched) peer's transport was closed


def test_build_peers_closes_already_launched_peer_on_bad_peer_policy(monkeypatch, tmp_path):
    # Regression: a later peer's malformed policy file raises ValueError from
    # load_policy, not OSError — _build_peers used to only catch OSError, so this
    # left an earlier peer's already-launched transport orphaned.
    from terse import multiproxy as mp

    closed = []

    class _FakeTransport:
        def inbound(self):
            return iter([])

        def outbound(self):
            return io.StringIO()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(mp, "build_transport", lambda target, headers=None: _FakeTransport())
    bad_policy = tmp_path / "bad.json"
    bad_policy.write_text("not valid json", encoding="utf-8")
    specs = [
        DownstreamSpec(name="a", target=["a"], headers={}, policy_path=None),
        DownstreamSpec(name="b", target=["b"], headers={}, policy_path=str(bad_policy)),
    ]
    try:
        _build_peers(specs, PLAIN_POLICY, debug=False, capture=None, audit=None,
                     store=OrderedDict(), store_lock=Lock(), dropped_bytes=[0])
        raise AssertionError("expected ValueError for the malformed 2nd peer policy")
    except ValueError:
        pass
    assert closed == [True]  # the first (already-launched) peer's transport was closed


def test_build_peers_diff_override_reaches_peer_with_own_policy_path(monkeypatch, tmp_path):
    # Regression: --diff was applied to `default_policy` only, so a peer with its OWN
    # policy_path (a freshly-loaded Policy object) silently never got cross-call
    # diffing enabled, unlike a peer using the default policy.
    from terse import multiproxy as mp

    monkeypatch.setattr(mp, "build_transport",
                        lambda target, headers=None: _FakePeerTransport())
    own_policy = tmp_path / "own.json"
    own_policy.write_text(json.dumps({"version": 1, "policies": []}), encoding="utf-8")  # ("rules" was a schema typo the loader used to swallow — now rejected)
    specs = [
        DownstreamSpec(name="a", target=["a"], headers={}, policy_path=None),
        DownstreamSpec(name="b", target=["b"], headers={}, policy_path=str(own_policy)),
    ]
    peers = _build_peers(specs, PLAIN_POLICY, debug=False, capture=None, audit=None,
                         store=OrderedDict(), store_lock=Lock(), dropped_bytes=[0],
                         diff_override=True, diff_keyframe_override=8)
    assert peers[0].inter.policy.diff is True
    assert peers[1].inter.policy.diff is True  # peer with its own policy file
    assert peers[1].inter.policy.diff_keyframe_interval == 8


def test_load_multi_config_rejects_name_containing_prefix_sep(tmp_path):
    # Regression: a name like "gh__api" wasn't rejected, so it could shadow a shorter
    # peer name ("gh") under _route_call's first-occurrence "__" split.
    cfg = _write_config(tmp_path, [{"name": "gh__api", "command": ["a"]}])
    try:
        load_multi_config(str(cfg))
        raise AssertionError("expected ValueError for a name containing '__'")
    except ValueError as e:
        assert "__" in str(e)


def test_server_initiated_request_reply_routes_back_to_originating_peer():
    # Regression: the client's reply to a server-initiated request (sampling/
    # createMessage, roots, ...) from a peer OTHER than peer 0 used to be misdelivered
    # to peer 0 unconditionally.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    router = Router(peers, io.StringIO(), Lock())
    try:
        forwarded = router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": 42, "method": "sampling/createMessage", "params": {}}))
        fwd_msg = json.loads(forwarded)
        assert fwd_msg["id"] != 42  # rewritten to a router-local id, not forwarded verbatim

        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": fwd_msg["id"], "result": {"ok": True}}))
    finally:
        router.close_senders()

    assert t0.out.getvalue() == ""                     # never reached peer 0
    delivered = json.loads(t1.out.getvalue().strip())  # reached peer 1, its true origin
    assert delivered["id"] == 42                        # with the ORIGINAL id restored


def test_reply_for_unknown_id_is_dropped_not_misrouted():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    router = Router(peers, io.StringIO(), Lock())
    try:
        router.route_client_line(json.dumps({"jsonrpc": "2.0", "id": 999, "result": {}}))
    finally:
        router.close_senders()
    assert t0.out.getvalue() == "" and t1.out.getvalue() == ""


def test_late_broadcast_reply_after_timeout_is_swallowed_not_leaked():
    # Regression: a peer's broadcast reply arriving AFTER _timeout_broadcast already
    # merged and replied used to fall through to that peer's own transform_response and
    # get written straight to the client, unmerged and carrying an internal id.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)  # never fires on its own
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        # peer 0 answers promptly
        router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-0",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {}}}))
        # force the broadcast to finish (as if its timer had fired) before peer 1 answers
        router._timeout_broadcast(0)
        assert len(_lines(out)) == 1  # the merged reply already went out

        # peer 1's reply arrives LATE
        result = router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-1",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {}}}))
    finally:
        router.close_senders()

    assert result is SWALLOW           # swallowed, not forwarded as an unsolicited message
    assert len(_lines(out)) == 1        # still exactly one reply on the client stream


def test_late_routed_call_reply_after_timeout_is_swallowed_not_double_answered():
    # Regression: a routed tools/call had no timeout at all — a hung/dead peer left it
    # unanswered forever. Once bounded, a peer's real reply arriving AFTER
    # _timeout_routed_call already answered the client must be swallowed, not
    # double-delivered (which would confuse a client tracking one reply per id).
    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)  # never fires on its own
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "a__gh.api.items"}}))
        assert len(_lines(out)) == 0  # no reply yet — still waiting on the peer

        # force the routed call to time out (as if its timer had fired) before the
        # peer answers
        router._timeout_routed_call(7, 0)
        assert len(_lines(out)) == 1
        assert "timed out" in _lines(out)[0]["error"]["message"]

        # the peer's real reply arrives LATE
        result = router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": 7,
             "result": {"content": [{"type": "text", "text": "late"}]}}))
    finally:
        router.close_senders()

    assert result is SWALLOW           # swallowed, not forwarded as a second reply
    assert len(_lines(out)) == 1        # still exactly one reply on the client stream


def test_routed_call_registers_timeout_timer_before_writing_to_peer():
    # Regression: _route_call used to write to the peer BEFORE registering the
    # timeout timer in _routed_timers, with no lock spanning both steps. A peer fast
    # enough to reply before registration ran would have its real reply processed by
    # from_peer while the timer didn't exist yet (pop -> None, delivered normally),
    # after which _route_call still inserted the now-orphaned timer — which would
    # later fire and send the client a spurious timeout for an id already answered.
    # Fixed by registering the timer before the peer write is even enqueued.
    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    seen_registered_before_send = []
    orig_send = router._senders[0].send

    def spy_send(line):
        seen_registered_before_send.append(7 in router._routed_timers)
        return orig_send(line)

    router._senders[0].send = spy_send
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "a__gh.api.items"}}))
    finally:
        router.close_senders()

    assert seen_registered_before_send == [True]


def test_peer_initiated_request_id_does_not_collide_with_routed_call_timeout():
    # Regression: _routed_timers/_routed_timed_out were keyed only by the bare id,
    # checked BEFORE a message was recognized as a peer-initiated server request
    # (sampling/createMessage, roots). A peer's own request id is unnamespaced and
    # can coincide with an unrelated in-flight routed call's id, which used to cancel
    # that routed call's timeout (or swallow the peer's request) purely on the
    # coincidence. Fixed by checking for a server-initiated request first.
    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)  # never fires on its own
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "a__gh.api.items"}}))
        assert 1 in router._routed_timers  # the routed call's timeout is pending

        # the SAME peer sends its own server-initiated request, reusing id=1 — its own
        # id space, unrelated to the client's routed-call id
        result = router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "sampling/createMessage", "params": {}}))

        # recognized as a request (rewritten + forwarded), not swallowed as a routed reply
        assert result not in (None, SWALLOW)
        rewritten = json.loads(result)
        assert rewritten["method"] == "sampling/createMessage"
        assert rewritten["id"] != 1  # namespaced, so the client's reply can route back

        # the routed call's OWN timeout must be untouched by the id coincidence
        assert 1 in router._routed_timers

        # its real reply still arrives and resolves normally afterward
        reply = router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": 1,
             "result": {"content": [{"type": "text", "text": "ok"}]}}))
    finally:
        router.close_senders()

    assert reply not in (None, SWALLOW)
    assert 1 not in router._routed_timers


def test_routed_timeout_eviction_ages_out_stale_entries_independent_of_population(monkeypatch):
    # Regression: _routed_timed_out was evicted purely by population count (FIFO) at a
    # cap of 4096 — unlike a broadcast-local id (namespaced, so an evicted-then-late
    # reply just fails to match anything and is dropped harmlessly), a routed call's id
    # IS the client's own live id, so an evicted-then-late reply looks like a real
    # second answer and gets delivered, double-answering the client. A realistic burst
    # of thousands of concurrent timeouts during a peer stall could exceed 4096 well
    # within a plausible "very late reply" window. Fixed two ways: (1) the population
    # backstop is now sized generously (65536) so a realistic burst doesn't force
    # eviction of anything still young, and (2) eviction is now proactively AGE-based
    # (Router._routed_timed_out_ttl) so a genuinely stale entry is cleaned up even when
    # population never approaches the backstop at all.
    fake_now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    router._routed_timed_out_ttl = 10.0
    try:
        router._routed_timers[1] = threading.Timer(1000, lambda: None)
        router._timeout_routed_call(1, 0)
        assert 1 in router._routed_timed_out

        # a second, unrelated timeout fires WELL WITHIN id 1's TTL — id 1 must still
        # be there (a young entry is never evicted just because another, unrelated
        # call also timed out)
        fake_now[0] = 5.0
        router._routed_timers[2] = threading.Timer(1000, lambda: None)
        router._timeout_routed_call(2, 0)
        assert 1 in router._routed_timed_out

        # time passes past id 1's TTL (age 12 > 10) but NOT id 2's (age 12 - 5 = 7 <
        # 10) — the next timeout's proactive age-based sweep must clean up only the
        # genuinely stale one, with population nowhere near the (65536) backstop,
        # proving eviction here is driven by age, not by population pressure
        fake_now[0] = 12.0
        router._routed_timers[3] = threading.Timer(1000, lambda: None)
        router._timeout_routed_call(3, 0)
        assert 1 not in router._routed_timed_out
        assert 2 in router._routed_timed_out  # still within its own TTL window
    finally:
        router.close_senders()


def test_drain_routed_calls_waits_out_in_flight_timeout_before_shutdown():
    # Regression: run_multi_proxy drained in-flight broadcasts before shutdown but had
    # no equivalent drain for routed calls — a client disconnecting right after issuing
    # a routed call to a slow/dead peer would have its still-pending timer torn down
    # mid-wait instead of given the same bounded-timeout guarantee as broadcasts.
    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=0.05)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "a__gh.api.items"}}))
        assert len(_lines(out)) == 0

        router.drain_routed_calls()  # blocks until the 0.05s timer fires

        assert len(_lines(out)) == 1
        assert "timed out" in _lines(out)[0]["error"]["message"]
    finally:
        router.close_senders()


def test_merge_initialize_protocol_version_uses_arrival_order_not_config_index():
    # Regression: _merge_initialize iterated peers by fixed config index
    # (range(len(self.peers))), so the merged protocolVersion always came from
    # whichever peer had the LOWEST index that answered — not whichever genuinely
    # replied FIRST, contradicting the method's own documented "first-arriving
    # peer's" contract. Here peer 1 (higher config index) answers first; the merge
    # must pick peer 1's protocolVersion, not peer 0's.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        # peer 1 (index 1, config-later) answers FIRST
        router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-1",
             "result": {"protocolVersion": "FIRST-ARRIVAL", "capabilities": {}}}))
        # peer 0 (index 0, config-earlier) answers SECOND
        router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-0",
             "result": {"protocolVersion": "SECOND-ARRIVAL", "capabilities": {}}}))
    finally:
        router.close_senders()

    merged = _lines(out)[0]
    assert merged["result"]["protocolVersion"] == "FIRST-ARRIVAL"


def test_broadcast_initialize_does_not_leave_stale_init_id_on_peer():
    # Regression: note_request set each peer's Interceptor.init_id to the broadcast-
    # local id (e.g. "terse-b0-1"), but that peer's real reply is swallowed by
    # _maybe_collect before transform_response ever runs its one-time reset — so
    # init_id stayed permanently stale (see test_clear_init_id_prevents_stale_reply_
    # misidentification in test_proxy.py for what that staleness could corrupt).
    t0 = _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        assert peers[0].inter.init_id is None  # cleared immediately, not left stale

        # peer answers the broadcast normally — still cleared, not repopulated
        router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-0",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {}}}))
        assert peers[0].inter.init_id is None
    finally:
        router.close_senders()


def test_reused_client_id_during_broadcast_resolves_to_correct_broadcast():
    # Regression: a client reusing an id while its broadcast was still in flight used to
    # produce IDENTICAL peer-local id strings for both broadcasts (format depended only
    # on client_id + peer index), so a stale reply for the first could get recorded into
    # the second (wrong) broadcast's merge.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        # client illegally reuses id=1 for a second broadcast before the first resolves
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))

        # a stale reply for the FIRST (now-abandoned) broadcast arrives
        late = router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-0",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {"first": True}}}))

        # both peers answer the SECOND (active) broadcast
        router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b1-0",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {"second": True}}}))
        router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b1-1",
             "result": {"protocolVersion": "2024-11-05", "capabilities": {"second": True}}}))
    finally:
        router.close_senders()

    assert late is SWALLOW              # the stale first-broadcast reply must never leak
    msgs = _lines(out)
    assert len(msgs) == 1               # exactly one merged reply for client id=1
    assert msgs[0]["result"]["capabilities"] == {"second": True}  # from the CORRECT broadcast


def test_slow_peer_write_does_not_block_routing_to_other_peers():
    # Regression: the client->server fan-out ran on one thread and wrote to each peer
    # inline/synchronously, so a slow peer's send blocked routing to every OTHER peer
    # until it finished.
    release = threading.Event()

    class _SlowTransport(_FakePeerTransport):
        def outbound(self):
            release.wait(timeout=5)
            return self.out

    slow, fast = _SlowTransport(), _FakePeerTransport()
    peers = [Peer("slow", slow, Interceptor(PLAIN_POLICY)),
             Peer("fast", fast, Interceptor(PLAIN_POLICY))]
    router = Router(peers, io.StringIO(), Lock())
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "slow__x"}}))
        # routed while `slow`'s send is still blocked in outbound() above — proves the
        # two peers' sends aren't serialized on one thread
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "fast__y"}}))
        deadline = time.monotonic() + 2.0
        while fast.out.getvalue() == "" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fast.out.getvalue() != ""  # got through despite `slow` still blocked
    finally:
        release.set()
        router.close_senders()


def test_unknown_method_forwards_to_peer_0_and_logs_without_debug(capsys):
    # Regression: this scope fallback's explanatory stderr note was gated behind
    # --debug, so by default an operator saw N-1 peers' data silently vanish from the
    # reply with no indication anything was dropped. `completion/complete` is a real
    # MCP method with no bespoke merge (unlike resources/list, now broadcast) — so it
    # exercises the genuine peer-0-only fallback that survived Phase 2.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    router = Router(peers, io.StringIO(), Lock(), debug=False)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "completion/complete", "params": {}}))
        deadline = time.monotonic() + 2.0
        while t0.out.getvalue() == "" and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        router.close_senders()
    assert t0.out.getvalue() != "" and t1.out.getvalue() == ""  # forwarded to peer 0 only
    err = capsys.readouterr().err
    assert "completion/complete" in err and "peer 0" in err  # logged even without --debug


# --- Phase 2 (#64): broadcast/merge resources|prompts|ping + route reads ---

def _drive_broadcast(router, out, client_msg, peer_replies):
    """Send `client_msg` (a broadcast), then feed each peer's reply as a
    `terse-b0-<i>` broadcast-local id (seq 0 — the first broadcast on a fresh Router),
    and return the single merged client-facing message. Each `from_peer` reply must be
    SWALLOWed (a broadcast-local id is never forwarded as-is); the final one finishes
    the broadcast and writes the merged reply to `out`."""
    router.route_client_line(json.dumps(client_msg))
    for i, reply in enumerate(peer_replies):
        got = router.from_peer(i)(json.dumps({**reply, "id": f"terse-b0-{i}"}))
        assert got is SWALLOW
    msgs = _lines(out)
    assert len(msgs) == 1
    return msgs[0]


def test_resources_list_merges_concat_without_prefix():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
            [{"result": {"resources": [{"uri": "a://1", "name": "a1"}]}},
             {"result": {"resources": [{"uri": "b://1", "name": "b1"}]}}])
    finally:
        router.close_senders()
    assert merged["id"] == 3
    # both peers' resources concatenated; a uri is NOT peer-prefixed (unlike a tool name)
    assert [r["uri"] for r in merged["result"]["resources"]] == ["a://1", "b://1"]


def test_resource_templates_list_merges_concat():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 4, "method": "resources/templates/list", "params": {}},
            [{"result": {"resourceTemplates": [{"uriTemplate": "a://{x}"}]}},
             {"result": {"resourceTemplates": [{"uriTemplate": "b://{x}"}]}}])
    finally:
        router.close_senders()
    assert [t["uriTemplate"] for t in merged["result"]["resourceTemplates"]] == \
        ["a://{x}", "b://{x}"]


def test_prompts_list_merges_with_peer_prefix():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}},
            [{"result": {"prompts": [{"name": "greet"}]}},
             {"result": {"prompts": [{"name": "farewell"}]}}])
    finally:
        router.close_senders()
    # prompt names ARE peer-prefixed (like tool names) so prompts/get can route by prefix
    assert [p["name"] for p in merged["result"]["prompts"]] == ["a__greet", "b__farewell"]


def test_ping_broadcast_replies_empty_result():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 6, "method": "ping", "params": {}},
            [{"result": {}}, {"result": {}}])
    finally:
        router.close_senders()
    assert merged == {"jsonrpc": "2.0", "id": 6, "result": {}}


def test_prompts_get_routes_by_prefix_and_strips_it():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 9, "method": "prompts/get",
             "params": {"name": "b__greet", "arguments": {"who": "x"}}}))
        deadline = time.monotonic() + 2.0
        while t1.out.getvalue() == "" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert t0.out.getvalue() == ""            # routed to the ONE named peer, not fanned out
        sent = json.loads(t1.out.getvalue().strip())
        assert sent["params"]["name"] == "greet"  # prefix stripped before the peer sees it
        assert sent["params"]["arguments"] == {"who": "x"}  # rest of params preserved
        assert sent["id"] == 9                     # client id passes through (single-peer route)
        # the peer's reply is forwarded back to the client (passthrough, not merged)
        forwarded = router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": 9,
             "result": {"messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}]}}))
    finally:
        router.close_senders()
    fwd = json.loads(forwarded)
    assert fwd["id"] == 9 and fwd["result"]["messages"][0]["content"]["text"] == "hi"


def test_prompts_get_unknown_prefix_returns_error():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock())
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 10, "method": "prompts/get",
             "params": {"name": "nope__greet"}}))
    finally:
        router.close_senders()
    assert t0.out.getvalue() == "" and t1.out.getvalue() == ""  # never reached any peer
    msgs = _lines(out)
    assert len(msgs) == 1 and msgs[0]["id"] == 10
    assert msgs[0]["error"]["code"] == -32601 and "unknown prompt" in msgs[0]["error"]["message"]


def test_resources_read_scatter_gather_first_success_wins():
    # A resource uri isn't peer-namespaced, so resources/read is fanned out to EVERY
    # peer; the one that owns the uri returns a result, the others error, and the first
    # success is forwarded to the client.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "resources/read",
             "params": {"uri": "b://res"}}))
        for t in (t0, t1):  # fanned out to BOTH peers, not peer-0-only
            deadline = time.monotonic() + 2.0
            while t.out.getvalue() == "" and time.monotonic() < deadline:
                time.sleep(0.01)
        assert "resources/read" in t0.out.getvalue() and "resources/read" in t1.out.getvalue()
        # peer 0 doesn't own the uri (error, discarded); peer 1 owns it (result, wins)
        assert router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-0",
             "error": {"code": -32002, "message": "resource not found"}})) is SWALLOW
        assert router.from_peer(1)(json.dumps(
            {"jsonrpc": "2.0", "id": "terse-b0-1",
             "result": {"contents": [{"uri": "b://res", "text": "hello"}]}})) is SWALLOW
    finally:
        router.close_senders()
    msgs = _lines(out)
    assert len(msgs) == 1 and msgs[0]["id"] == 7
    assert "error" not in msgs[0]
    assert msgs[0]["result"]["contents"][0]["text"] == "hello"


def test_resources_read_scatter_gather_all_error_surfaces_first_error():
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 8, "method": "resources/read",
             "params": {"uri": "z://none"}},
            [{"error": {"code": -32002, "message": "not found on a"}},
             {"error": {"code": -32002, "message": "not found on b"}}])
    finally:
        router.close_senders()
    assert merged["id"] == 8 and "result" not in merged
    # the first-arriving error is surfaced, not a synthesized one
    assert merged["error"]["message"] == "not found on a"


def test_peer_initiated_request_does_not_consume_that_peers_interceptor_pending():
    # Companion to the _routed_timers test above, one layer down: pins the END-TO-END
    # invariant that a peer's own request never consumes that peer's Interceptor tracking,
    # so the real result is still compressed and recorded.
    #
    # Note this now holds at TWO layers — `from_peer` recognizes the server request first,
    # and `transform_response` forwards method-bearing messages untouched. Verified: this
    # test still passes with `from_peer`'s server-request branch disabled, because the
    # Interceptor guard catches it. That is defense in depth, not a vacuous test — but it
    # does mean this test alone will NOT catch a from_peer regression; the _routed_timers
    # test above is what pins that branch's own behavior.
    t0 = _FakePeerTransport()
    inter = Interceptor(PLAIN_POLICY)
    peers = [Peer("a", t0, inter)]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "a__gh.api.items"}}))
        assert inter.pending, "the peer's Interceptor is tracking the routed call"

        # the peer emits its own request reusing id=1 (its own id space)
        router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "roots/list"}))
        assert inter.pending, "tracking must SURVIVE the peer's own request"

        # ... and the real result still gets compressed
        payload = {"result": [{"id": i, "status": "active"} for i in range(30)]}
        line = router.from_peer(0)(json.dumps(
            {"jsonrpc": "2.0", "id": 1,
             "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}}))
        assert line not in (None, SWALLOW)
        text = json.loads(line)["result"]["content"][0]["text"]
        assert transforms.decompress(text) == payload
        assert transforms.TABLE_MARKER in text
    finally:
        router.close_senders()
