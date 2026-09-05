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

def test_tools_list_exposes_uncollided_names_verbatim_and_single_retrieve(tmp_path):
    # #168: names are qualified ONLY on a real cross-peer collision. These two peers share
    # no tool name, so every tool keeps the name its own server gave it — which is what
    # makes multiproxy a drop-in for a client allowlist instead of a migration.
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
    assert "gh.api.items" in names and "fs.read" in names
    assert "items.get" in names and "items.body" in names
    assert not [n for n in names if n.startswith(("gh__", "http__"))]
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


def test_capture_dir_writes_one_envelope_per_peer_attributed_to_that_peer(tmp_path):
    """#374: the router's corpus is the input to every drop/tune measurement, so a peer
    whose payloads never land — or land under another peer's name — silently biases the
    sample.

    What pins the behavior is the direct assertion on each envelope's `tool` and `server`.
    The byte-identical setup is a SECOND, independent witness on top of that, not the
    mechanism: both peers front the same fake server and answer the same call, and capture
    is idempotent by sha, so a bare (un-qualified) capture name would fold the two into ONE
    envelope carrying whichever peer wrote last. That makes the cardinality assertion fail
    too — but it would still be pinned by the name assertion if a later change gave the
    fake per-process output and broke the fold.
    """
    from terse.capture import load_corpus

    cfg = _write_config(tmp_path, [{"name": "gh", "command": [sys.executable, str(FAKE)]},
                                   {"name": "gh2", "command": [sys.executable, str(FAKE)]}])
    corpus = tmp_path / "corpus"
    cin = io.StringIO("\n".join(
        json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                    "params": {"name": f"{peer}__gh.api.items"}})
        for i, peer in enumerate(["gh", "gh2"], start=2)) + "\n")
    cout = io.StringIO()
    rc = run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout,
                         capture_dir=str(corpus))
    assert rc == 0
    envs = load_corpus(corpus)
    # Cardinality first, and as its own assertion: `sorted()` over a list holding None
    # raises TypeError before it ever compares to the expected list, so a lost attribution
    # would otherwise fail with a message that never mentions attribution.
    assert len(envs) == 2
    assert sorted(e.get("server") or "<none>" for e in envs) == ["gh", "gh2"]
    # the peer-qualified name is what keeps two peers' identical payloads apart on disk
    assert sorted(e["tool"] for e in envs) == ["gh2__gh.api.items", "gh__gh.api.items"]
    # and it is the RAW downstream payload that was teed, not the compressed wire form
    assert all(json.loads(e["raw"])["result"][0]["status"] == "active" for e in envs)


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

    def fake_build_transport(target, headers=None, env=None, cwd=None):
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

    monkeypatch.setattr(
        mp, "build_transport",
        lambda target, headers=None, env=None, cwd=None: _FakeTransport())
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
                        lambda target, headers=None, env=None, cwd=None: _FakePeerTransport())
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


def test_build_peers_never_attaches_a_lazy_primer(monkeypatch):
    # #168 phase 2: the router already primes eagerly once via `union_primer` at
    # `initialize` (`_merge_initialize`) — a peer going lazy too would attach its OWN
    # primer on its own first compression, on top of that. Pins the explicit
    # `lazy_primer=False` `_build_peers` passes into every peer's `Interceptor`.
    from terse import multiproxy as mp

    monkeypatch.setattr(mp, "build_transport",
                        lambda target, headers=None, env=None, cwd=None: _FakePeerTransport())
    specs = [DownstreamSpec(name="a", target=["a"], headers={}, policy_path=None)]
    peers = _build_peers(specs, PLAIN_POLICY, debug=False, capture=None, audit=None,
                         store=OrderedDict(), store_lock=Lock(), dropped_bytes=[0])
    inter = peers[0].inter
    assert inter._lazy_primer is False

    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    payload = {"result": [{"id": i, "status": "active"} for i in range(20)]}
    out = json.loads(inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 1,
         "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})))
    # still compressed (the codec is unaffected), but no leading primer block appears —
    # the router's own eager union primer already covers it
    blocks = out["result"]["content"]
    assert len(blocks) == 1
    assert transforms.TABLE_MARKER in blocks[0]["text"]


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


def test_prompts_list_qualifies_only_the_collided_name():
    # Prompts follow the SAME collision-only rule as tools (#168) — two surfaces that
    # disagreed about when a name is qualified would be a client-visible inconsistency.
    # "greet" is exported by both peers, so both copies are qualified; the two names
    # unique to one peer are exposed verbatim.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(PLAIN_POLICY)), Peer("b", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out,
            {"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}},
            [{"result": {"prompts": [{"name": "greet"}, {"name": "recap"}]}},
             {"result": {"prompts": [{"name": "greet"}, {"name": "farewell"}]}}])
    finally:
        router.close_senders()
    assert [p["name"] for p in merged["result"]["prompts"]] == \
        ["a__greet", "recap", "b__greet", "farewell"]
    assert router.prompt_route == {"a__greet": (0, "greet"), "recap": (0, "recap"),
                                   "b__greet": (1, "greet"), "farewell": (1, "farewell")}


# --- #168: collision-only tool naming, and what must survive it ---

def _two_peer_router(policy=None):
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    pol = policy or PLAIN_POLICY
    peers = [Peer("a", t0, Interceptor(pol)), Peer("b", t1, Interceptor(pol))]
    out = io.StringIO()
    return t0, t1, peers, out, Router(peers, out, Lock(), broadcast_timeout=1000)


def _peer_calls(t):
    """Only the `tools/call` lines a peer received — the broadcast `tools/list` reaches
    every peer, so a bare "did this peer get anything" check can't prove routing."""
    return [m for m in (json.loads(ln) for ln in t.out.getvalue().splitlines() if ln.strip())
            if m.get("method") == "tools/call"]


def _await_peer_call(t, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not _peer_calls(t) and time.monotonic() < deadline:
        time.sleep(0.01)
    calls = _peer_calls(t)
    assert len(calls) == 1
    return calls[0]


def _list_tools(router, out, per_peer):
    return _drive_broadcast(
        router, out, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        [{"result": {"tools": tools}} for tools in per_peer])


def test_collided_tool_name_is_qualified_on_both_sides_and_routes_to_its_own_peer():
    # The case unconditional prefixing existed to defend against: two peers both export
    # "search". Both copies are qualified so the client can still address each one, and
    # each qualified name must reach ITS peer with the bare name restored.
    t0, t1, _, out, router = _two_peer_router()
    try:
        merged = _list_tools(router, out, [[{"name": "search"}], [{"name": "search"}]])
        assert [t["name"] for t in merged["result"]["tools"]] == ["a__search", "b__search"]
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "b__search", "arguments": {"q": "x"}}}))
        sent = _await_peer_call(t1)
        assert _peer_calls(t0) == []                # routed to ONE peer, not fanned out
        assert sent["params"] == {"name": "search", "arguments": {"q": "x"}}
    finally:
        router.close_senders()


def test_uncollided_tool_routes_verbatim_but_bookkeeping_keeps_peer_attribution():
    # The invariant collision-only naming could silently break: the tool is EXPOSED bare,
    # so nothing in the wire name says which peer owns it — but `note_request` must still
    # bucket it under the peer-qualified name, or two servers' corpora merge (#158/#143).
    t0, t1, peers, out, router = _two_peer_router()
    try:
        merged = _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        assert [t["name"] for t in merged["result"]["tools"]] == ["only_a", "only_b"]
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "only_b"}}))
        sent = _await_peer_call(t1)
        assert _peer_calls(t0) == []
        assert sent["params"]["name"] == "only_b"     # peer sees its own unchanged name
        # ...but the audit/corpus bucket (pending = (wire_name, tracked_name, args_key))
        # is peer-qualified, not the bare wire name
        wire_name, tracked_name, _ = peers[1].inter.pending[2]
        assert (wire_name, tracked_name) == ("only_b", "b__only_b")
    finally:
        router.close_senders()


def test_advertised_name_containing_the_separator_is_not_re_split_as_a_peer_prefix():
    # Peer "a" exports a tool literally named "b__thing". Exposed verbatim (no collision),
    # a prefix-first router would re-read it as peer "b" + tool "thing" and misroute the
    # call to the WRONG SERVER. The advertised-name table must win.
    t0, t1, _, out, router = _two_peer_router()
    try:
        merged = _list_tools(router, out, [[{"name": "b__thing"}], [{"name": "other"}]])
        assert [t["name"] for t in merged["result"]["tools"]] == ["b__thing", "other"]
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "b__thing"}}))
        sent = _await_peer_call(t0)
        assert _peer_calls(t1) == []                      # peer "b" never saw it
        assert sent["params"]["name"] == "b__thing"       # name NOT stripped
    finally:
        router.close_senders()


def test_retrieve_tool_name_is_reserved_so_a_peer_cannot_shadow_it():
    # `terse.retrieve` is answered by the router itself. A peer exporting that name must
    # be qualified even though no OTHER peer claims it, else the client's retrieve calls
    # become ambiguous between the router and that peer.
    from terse.lossy import RETRIEVE_TOOL
    t0, t1, _, out, router = _two_peer_router()
    try:
        merged = _list_tools(router, out, [[{"name": RETRIEVE_TOOL}], [{"name": "other"}]])
        assert [t["name"] for t in merged["result"]["tools"]] == [f"a__{RETRIEVE_TOOL}",
                                                                  "other"]
        assert RETRIEVE_TOOL not in router.tool_route
    finally:
        router.close_senders()


def test_qualified_name_still_routes_before_any_tools_list():
    # The prefix split survives as a FALLBACK: a client calling before it has listed
    # (or holding a name from an earlier listing) must not get -32601.
    t0, t1, _, out, router = _two_peer_router()
    try:
        assert router.tool_route == {}                 # nothing advertised yet
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "b__search"}}))
        sent = _await_peer_call(t1)
        assert _peer_calls(t0) == []
        assert sent["params"]["name"] == "search"
    finally:
        router.close_senders()


def test_a_peer_cannot_shadow_another_peers_qualified_name():
    # Peers "a" and "b" both export "search", so both are qualified to a__search /
    # b__search. Peer "c" exports a tool named LITERALLY "a__search" — with only bare
    # names reserved it would be exposed verbatim, advertise a DUPLICATE name, and hijack
    # every a__search call. The qualified forms are reserved too, so "c" is qualified in
    # turn and no two exposed names are equal.
    t0, t1, t2 = _FakePeerTransport(), _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer(n, t, Interceptor(PLAIN_POLICY))
             for n, t in (("a", t0), ("b", t1), ("c", t2))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            [{"result": {"tools": [{"name": "search"}]}},
             {"result": {"tools": [{"name": "search"}]}},
             {"result": {"tools": [{"name": "a__search"}]}}])
        names = [t["name"] for t in merged["result"]["tools"]]
        assert names == ["a__search", "b__search", "c__a__search"]
        assert len(set(names)) == len(names)          # no duplicate on the wire
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "a__search"}}))
        sent = _await_peer_call(t0)                   # peer "a", NOT the shadower
        assert sent["params"]["name"] == "search"
        assert _peer_calls(t2) == []
    finally:
        router.close_senders()


def test_the_route_table_is_exactly_the_most_recent_listing():
    # The contract after #178 withdrew route retention: no names are carried forward from
    # peers that missed a listing. A peer absent from a listing is absent from the client's
    # tool list too, so a call to it is a clean -32601 rather than a route into the dark.
    # Whether that peer errored, timed out, or answered with an empty list is not a
    # distinction the table draws — all three simply contribute nothing.
    from terse.multiproxy import _PendingBroadcast
    for i, part in enumerate([{"result": {"tools": []}},                       # empty
                              {"error": {"code": -32601, "message": "nope"}},  # errored
                              None]):                                          # timed out
        _, _, _, out, router = _two_peer_router()
        try:
            _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
            assert set(router.tool_route) == {"only_a", "only_b"}
            parts = {0: {"result": {"tools": [{"name": "only_a"}]}}}
            if part is not None:
                parts[1] = part
            router._merge_tools_list(_PendingBroadcast(
                kind="tools/list", client_id=9, seq=1, remaining=set(), parts=parts))
            assert set(router.tool_route) == {"only_a"}, f"case {i}"
            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "only_b"}}))
        finally:
            router.close_senders()
        errs = [m for m in _lines(out) if "error" in m]
        assert len(errs) == 1 and errs[0]["error"]["code"] == -32601, f"case {i}"


def test_a_stale_listing_reply_does_not_duplicate_the_retrieve_tool(tmp_path):
    # `_tool_entries` must be a COPY: the caller appends RETRIEVE_TOOL_DEF to its own list
    # after the lock, and aliasing would accumulate that append into the stored entries, so
    # a stale-refused reply would advertise `terse.retrieve` TWICE — a duplicate tool name
    # on the wire, which is an MCP protocol violation some clients reject outright.
    from terse.multiproxy import _PendingBroadcast
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("a", t0, Interceptor(DROP_POLICY)), Peer("b", t1, Interceptor(DROP_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    assert router.has_drop
    try:
        _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=9, seq=2, remaining=set(),
            parts={0: {"result": {"tools": [{"name": "new"}]}}}))
        stale = router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=8, seq=1, remaining=set(),
            parts={0: {"result": {"tools": [{"name": "old"}]}}}))
        names = [t["name"] for t in stale["tools"]]
        assert names.count("terse.retrieve") == 1, names
    finally:
        router.close_senders()


def test_exposed_names_are_unique_across_an_exhaustive_small_configuration_sweep():
    """The uniqueness invariant, checked by exhaustion rather than by argument.

    Three separate hand-reasoned versions of `_expose_names` each shipped a duplicate
    (rounds 2, 3 and 6 of review). The claim is now: for ANY assignment of names to peers,
    no two exposed names are equal unless one peer listed the same name twice. The
    alphabet deliberately includes the `{peer}__{name}` shapes that make a qualified form
    land on another entry's bare name, plus the reserved `terse.retrieve`.
    """
    from itertools import combinations_with_replacement, product

    from terse.lossy import RETRIEVE_TOOL
    names = ["x", "a__x", "b__a__x", RETRIEVE_TOOL, f"a__{RETRIEVE_TOOL}"]
    peer_names = ("a", "b", "c")
    peers = [Peer(n, _FakePeerTransport(), Interceptor(PLAIN_POLICY)) for n in peer_names]
    router = Router(peers, io.StringIO(), Lock(), broadcast_timeout=1000)
    per_peer = [t for t in combinations_with_replacement(names, 2) if len(set(t)) == len(t)]
    try:
        checked = 0
        for combo in product(per_peer, repeat=len(peer_names)):
            owned = [(idx, {"name": t}) for idx, tools in enumerate(combo) for t in tools]
            entries, route, _ = router._expose_names(owned, reserved=(RETRIEVE_TOOL,))
            exposed = [e["name"] for e in entries]
            assert len(set(exposed)) == len(exposed), (combo, exposed)
            assert len(route) == len(exposed), (combo, exposed)
            # and every entry still routes back to its own peer and its real name
            for (idx, it), name in zip(owned, exposed, strict=True):
                assert route[name] == (idx, it["name"])
            checked += 1
        assert checked == len(per_peer) ** len(peer_names) == 1000, checked
    finally:
        router.close_senders()


def test_a_peers_emitted_qualified_form_still_contests_its_own_sibling_name():
    # The other half of the sibling rule, and the case a per-peer exclusion got wrong:
    # gh's `search` DOES collide with kb's, so gh emits `gh__search` — which now contests
    # gh's OWN tool literally named `gh__search`. Excluding the entry's own peer left both
    # on the wire under one name, so gh's `search` became unaddressable.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("gh", t0, Interceptor(PLAIN_POLICY)), Peer("kb", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            [{"result": {"tools": [{"name": "search"}, {"name": "gh__search"}]}},
             {"result": {"tools": [{"name": "search"}]}}])
        names = [t["name"] for t in merged["result"]["tools"]]
        assert len(set(names)) == len(names), names
        assert names == ["gh__search", "gh__gh__search", "kb__search"]
        assert router.tool_route["gh__search"] == (0, "search")
        assert router.tool_route["gh__gh__search"] == (0, "gh__search")
    finally:
        router.close_senders()


def test_a_reserved_name_contests_a_sibling_without_any_cross_peer_collision():
    # Same shape, reached with ONE peer: `terse.retrieve` is reserved, so gh's copy is
    # qualified to `gh__terse.retrieve` — which must then contest gh's own tool of that
    # literal name. No second peer is needed to produce the duplicate.
    from terse.lossy import RETRIEVE_TOOL
    _, _, _, out, router = _two_peer_router()
    try:
        merged = _list_tools(router, out,
                             [[{"name": RETRIEVE_TOOL}, {"name": f"a__{RETRIEVE_TOOL}"}], []])
        names = [t["name"] for t in merged["result"]["tools"]]
        assert len(set(names)) == len(names), names
        assert names == [f"a__{RETRIEVE_TOOL}", f"a__a__{RETRIEVE_TOOL}"]
    finally:
        router.close_senders()


def _merge_listing(router, seq, parts):
    """Merge a tools/list at an explicit `seq` with an explicit per-peer `parts` map, so a
    test can leave a peer OUT entirely (the shape a timed-out broadcast produces) instead of
    handing it an empty reply, which is a different silent-reason. `_drive_broadcast` can't
    do this: it hardcodes seq 0 and requires a reply from every peer."""
    from terse.multiproxy import _PendingBroadcast
    return router._merge_tools_list(_PendingBroadcast(
        kind="tools/list", client_id=1, seq=seq, remaining=set(), parts=parts))


def test_a_collision_seen_once_keeps_the_name_qualified_when_the_rival_goes_silent():
    # terse#178's naming half. `search` flipping between `a__search` (both peers answered)
    # and bare `search` (b missed the listing) hands a caching client a -32601 for whichever
    # spelling it kept. Note the direction that #226's -32601 diagnosis CANNOT explain: on
    # the listing where b RETURNS, no peer is silent, so that error names nobody.
    _, _, _, out, router = _two_peer_router()
    try:
        # 1. both answer -> a genuine collision, both qualified, `search` ratcheted
        assert [t["name"] for t in _merge_listing(
            router, 0, {0: {"result": {"tools": [{"name": "search"}]}},
                        1: {"result": {"tools": [{"name": "search"}]}}})["tools"]] \
            == ["a__search", "b__search"]
        assert router._contested_tools == {"search"}

        # 2. b MISSES this listing entirely (a timeout leaves it out of `parts`). Before the
        #    ratchet this exposed bare `search`; the whole point is that it no longer does.
        assert [t["name"] for t in _merge_listing(
            router, 1, {0: {"result": {"tools": [{"name": "search"}]}}})["tools"]] \
            == ["a__search"]
        assert router.tool_route == {"a__search": (0, "search")}
        # and b's copy is NOT resurrected — names are carried forward, routes never are
        assert "b__search" not in router.tool_route

        # 3. b returns: unchanged again, so no listing in the sequence renamed anything
        assert [t["name"] for t in _merge_listing(
            router, 2, {0: {"result": {"tools": [{"name": "search"}]}},
                        1: {"result": {"tools": [{"name": "search"}]}}})["tools"]] \
            == ["a__search", "b__search"]
    finally:
        router.close_senders()


def test_a_rival_silent_on_the_first_listing_still_flips_the_name():
    # The ratchet's REMAINING limitation, pinned so it stays honest. It can only fire once a
    # contest has been witnessed, so the reverse ordering of the test above is NOT closed:
    # nothing had contested `search` when b was silent, so that listing exposes it bare and
    # the listing where b returns re-qualifies it. Closing this needs knowledge of what a
    # peer exports before it has ever answered, which the router has no source for.
    #
    # This test passing is not a bug — it is the documented boundary. If a future change
    # closes it, this test SHOULD fail and be rewritten; that is the point of pinning it.
    _, _, _, out, router = _two_peer_router()
    try:
        assert [t["name"] for t in _merge_listing(
            router, 0, {0: {"result": {"tools": [{"name": "search"}]}}})["tools"]] \
            == ["search"]
        assert router._contested_tools == set()   # nothing witnessed yet, nothing to ratchet
        assert [t["name"] for t in _merge_listing(
            router, 1, {0: {"result": {"tools": [{"name": "search"}]}},
                        1: {"result": {"tools": [{"name": "search"}]}}})["tools"]] \
            == ["a__search", "b__search"]
        # the bare name a client may have cached from listing 0 is now unroutable
        assert "search" not in router.tool_route
    finally:
        router.close_senders()


def test_a_reserved_only_qualification_never_enters_the_ratchet():
    # The self-feeding hazard. `terse.retrieve` is qualified because it is RESERVED, not
    # because two peers contested it. Ratcheting that would put it back in `reserved` next
    # listing as if it had collided — and every reserved-qualified name after it — walking
    # the router back to the unconditional prefixing #168 removed as its one unshippable
    # defect. Only a genuine cross-peer contest may ratchet.
    from terse.lossy import RETRIEVE_TOOL
    _, _, _, out, router = _two_peer_router()
    try:
        merged = _merge_listing(
            router, 0, {0: {"result": {"tools": [{"name": RETRIEVE_TOOL}]}},
                        1: {"result": {"tools": [{"name": "other"}]}}})
        assert [t["name"] for t in merged["tools"]] == [f"a__{RETRIEVE_TOOL}", "other"]
        assert router._contested_tools == set()      # qualified, but never contested
    finally:
        router.close_senders()


def test_a_peer_that_did_not_answer_can_never_contest_a_name():
    # The R3 defect class, at its root: an `error`/`malformed`/`empty` peer contributes no
    # entries, so it cannot collide with anything and cannot be "remembered" as owning a
    # name. Checked for all three shapes at once — `a` exports `search` every time, and the
    # ratchet must stay empty, so `search` is never force-qualified on b's behalf.
    for part in ({"error": {"code": -32601, "message": "no"}},   # explicit JSON-RPC error
                 {"result": {"tools": "not-a-list"}},            # malformed
                 {"result": {"tools": []}}):                     # empty
        _, _, _, out, router = _two_peer_router()
        try:
            merged = _merge_listing(
                router, 0, {0: {"result": {"tools": [{"name": "search"}]}}, 1: part})
            assert [t["name"] for t in merged["tools"]] == ["search"], part
            assert router._contested_tools == set(), part
        finally:
            router.close_senders()


def test_one_peer_listing_a_name_twice_does_not_ratchet_it():
    # The ratchet counts DISTINCT PEERS, not occurrences. A single broken peer listing the
    # same name twice is not a cross-peer contest: qualification cannot disambiguate it
    # (both copies qualify to the same string and the second is dropped — see
    # `test_an_unresolvable_duplicate_warns_without_debug`), so ratcheting it would let one
    # malformed listing permanently re-spell that name for the peer that legitimately owns
    # it. Caught by mutation: an occurrence COUNT passes every other test in this file.
    _, _, _, out, router = _two_peer_router()
    try:
        merged = _merge_listing(router, 0, {0: {"result": {"tools": [{"name": "dup"},
                                                                    {"name": "dup"}]}}})
        assert [t["name"] for t in merged["tools"]] == ["a__dup"]   # dropped, as before
        assert router._contested_tools == set()
        # the next well-formed listing must still expose it BARE, not `a__dup`
        assert [t["name"] for t in _merge_listing(
            router, 1, {0: {"result": {"tools": [{"name": "dup"}]}}})["tools"]] == ["dup"]
    finally:
        router.close_senders()


def test_a_stale_listing_still_ratchets_even_though_its_table_is_refused():
    # The ratchet is deliberately NOT behind the seq guard. A superseded listing's TABLE is
    # rightly discarded, but "these two peers both export `search`" is not falsified by
    # arriving late — it stays true. Gating it would lose the fact for no safety gain.
    _, _, _, out, router = _two_peer_router()
    try:
        _merge_listing(router, 5, {0: {"result": {"tools": [{"name": "only"}]}}})
        assert set(router.tool_route) == {"only"}
        # seq 2 < installed seq 5: refused for install, but its collision still counts
        _merge_listing(router, 2, {0: {"result": {"tools": [{"name": "search"}]}},
                                   1: {"result": {"tools": [{"name": "search"}]}}})
        assert set(router.tool_route) == {"only"}    # table untouched by the stale listing
        assert router._contested_tools == {"search"} # but the fact was kept
    finally:
        router.close_senders()


def test_a_peer_does_not_qualify_a_tool_against_its_own_sibling_name():
    # `gh` exporting both `search` and a tool literally named `gh__search` has NO
    # cross-peer collision. Counting a peer's qualified form against its own bare names
    # renamed the second tool to `gh__gh__search`, contradicting the documented rule and
    # breaking the allowlist #168 exists to preserve.
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("gh", t0, Interceptor(PLAIN_POLICY)), Peer("kb", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        merged = _drive_broadcast(
            router, out, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            [{"result": {"tools": [{"name": "search"}, {"name": "gh__search"}]}},
             {"result": {"tools": [{"name": "other"}]}}])
        assert [t["name"] for t in merged["result"]["tools"]] == \
            ["search", "gh__search", "other"]
        assert router.tool_route["gh__search"] == (0, "gh__search")
    finally:
        router.close_senders()


def test_a_late_timing_out_listing_does_not_clobber_a_newer_one():
    # `_route_lock` serializes installs but does not ORDER them, and two listings with
    # different client ids are concurrently pending. A broadcast that times out after
    # BROADCAST_TIMEOUT would otherwise install a 30-second-old snapshot over a newer
    # complete one — resurrecting a tool its peer has since dropped ("old") and erasing
    # one it has since gained ("new").
    from terse.multiproxy import _PendingBroadcast
    _, _, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "old"}], [{"name": "only_b"}]])   # seq 0
        # the client re-lists; everyone answers and peer a has replaced old with new
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=10, seq=2, remaining=set(),
            parts={0: {"result": {"tools": [{"name": "new"}]}},
                   1: {"result": {"tools": [{"name": "only_b"}]}}}))
        assert set(router.tool_route) == {"new", "only_b"}
        # NOW the older broadcast finally times out with its stale snapshot
        stale = router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=9, seq=1, remaining={1},
            parts={0: {"result": {"tools": [{"name": "old"}]}}}))
        # its own client is still answered — but with what is ACTUALLY routable, not with
        # this listing's stale view: advertising "old" here would hand the client a name
        # that returns -32601 the moment it calls it.
        assert [t["name"] for t in stale["tools"]] == ["new", "only_b"]
        assert set(router.tool_route) == {"new", "only_b"}      # ...and nothing installed
    finally:
        router.close_senders()


def test_the_prefix_fallback_is_off_once_any_listing_has_landed_even_an_empty_one():
    # Gating the fallback on "the table is empty" rather than "a listing has ever landed"
    # leaves the wrong-server misroute reachable: an all-peers-timed-out listing installs
    # {}, and `prompts/list` is answered -32601 by most servers, so prompt_route would be
    # empty FOREVER and the fallback would never disarm.
    from terse.multiproxy import _PendingBroadcast
    _, t1, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=9, seq=1, remaining=set(), parts={}))
        assert router.tool_route == {}                       # every peer timed out
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "b__x"}}))
        assert _peer_calls(t1) == []                         # NOT split onto peer b
    finally:
        router.close_senders()
    errs = [m for m in _lines(out) if "error" in m]
    assert len(errs) == 1 and errs[0]["error"]["code"] == -32601


def test_prompts_get_does_not_fall_back_after_a_prompts_list_that_every_peer_refused():
    # prompt_route stays {} for a fleet whose peers all answer prompts/list with -32601.
    # An emptiness-gated fallback would dispatch every unknown prompt name to a peer.
    from terse.multiproxy import _PendingBroadcast
    t0, t1, _, out, router = _two_peer_router()
    try:
        router._merge_prompts_list(_PendingBroadcast(
            kind="prompts/list", client_id=1, seq=0, remaining=set(),
            parts={0: {"error": {"code": -32601, "message": "no"}},
                   1: {"error": {"code": -32601, "message": "no"}}}))
        assert router.prompt_route == {}
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 4, "method": "prompts/get", "params": {"name": "b__nope"}}))
        assert _peer_calls(t0) == [] and _peer_calls(t1) == []
    finally:
        router.close_senders()
    errs = [m for m in _lines(out) if "error" in m]
    assert len(errs) == 1 and "unknown prompt" in errs[0]["error"]["message"]


def test_the_prefix_fallback_is_off_once_a_listing_has_landed():
    # Peer "gh" exports a tool literally named "kb__thing" (exposed verbatim), then drops
    # it while "kb" gains a tool called "thing". A fallback that fires on any unadvertised
    # name would split the stale call and execute it on a DIFFERENT SERVER, with the
    # client's arguments, successfully. After a listing exists, unknown means -32601.
    from terse.multiproxy import _PendingBroadcast
    t0, t1 = _FakePeerTransport(), _FakePeerTransport()
    peers = [Peer("gh", t0, Interceptor(PLAIN_POLICY)), Peer("kb", t1, Interceptor(PLAIN_POLICY))]
    out = io.StringIO()
    router = Router(peers, out, Lock(), broadcast_timeout=1000)
    try:
        _list_tools(router, out, [[{"name": "kb__thing"}], []])
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=9, seq=1, remaining=set(),
            parts={0: {"result": {"tools": []}},
                   1: {"result": {"tools": [{"name": "thing"}]}}}))
        assert set(router.tool_route) == {"thing"}
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "kb__thing", "arguments": {"danger": 1}}}))
        assert _peer_calls(t1) == [] and _peer_calls(t0) == []   # no server ran it
    finally:
        router.close_senders()
    errs = [m for m in _lines(out) if "error" in m]
    assert len(errs) == 1 and errs[0]["error"]["code"] == -32601


def test_a_depth_two_shadow_chain_stays_unique_in_either_config_order():
    # The qualified-form reservation must not depend on config order. Reserving only the
    # already-collided entries' qualified forms made it a single forward pass, so this
    # chain produced a duplicate name in one order and not the other.
    from terse.multiproxy import Router as R
    chain = [("d", "c__a__x"), ("c", "a__x"), ("b", "x"), ("a", "x")]
    for order in (chain, list(reversed(chain))):
        transports = [_FakePeerTransport() for _ in order]
        peers = [Peer(n, t, Interceptor(PLAIN_POLICY))
                 for (n, _), t in zip(order, transports, strict=True)]
        out = io.StringIO()
        router = R(peers, out, Lock(), broadcast_timeout=1000)
        try:
            merged = _drive_broadcast(
                router, out,
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                [{"result": {"tools": [{"name": tool}]}} for _, tool in order])
            names = [t["name"] for t in merged["result"]["tools"]]
            assert len(set(names)) == len(names), f"duplicate exposed name in {order}: {names}"
            assert len(router.tool_route) == len(order)   # every peer stays addressable
        finally:
            router.close_senders()


def test_an_unresolvable_duplicate_warns_without_debug(capsys):
    # A peer listing the same name twice can't be disambiguated by qualification. The
    # shadowed copy is DROPPED rather than advertised under a duplicate name — a duplicate
    # is an MCP protocol violation, and a client that rejects the listing over it loses
    # every peer's tools, not just this one. The warning is UNCONDITIONAL: a silently
    # dropped tool is the gap this module promises not to hide (cf. the peer-0 notice).
    _, _, _, out, router = _two_peer_router()
    assert router.debug is False
    try:
        merged = _list_tools(router, out, [[{"name": "dup"}, {"name": "dup"}], []])
        names = [t["name"] for t in merged["result"]["tools"]]
        assert names == ["a__dup"]                    # advertised ONCE, not twice
        assert len(set(names)) == len(names)
        assert router.tool_route["a__dup"] == (0, "dup")
    finally:
        router.close_senders()
    err = capsys.readouterr().err
    assert "name collision on 'a__dup'" in err and "DROPPED" in err


def test_tools_list_rebuilds_the_route_table_rather_than_accumulating():
    # A peer can change its tool set and re-issue tools/list. A stale entry would keep
    # routing a tool the peer no longer has, so the table is REPLACED, not merged.
    _, _, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "old"}], []])
        assert "old" in router.tool_route
        from terse.multiproxy import _PendingBroadcast
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=1, seq=1, remaining=set(),
            parts={0: {"result": {"tools": [{"name": "new"}]}}, 1: {"result": {"tools": []}}}))
        assert set(router.tool_route) == {"new"}
    finally:
        router.close_senders()


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
    # lazy_primer=False: matches how _build_peers actually constructs a peer's Interceptor
    # in production (#168 phase 2) — this test is about request-tracking, not primer.
    inter = Interceptor(PLAIN_POLICY, lazy_primer=False)
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


def test_merge_initialize_omits_instructions_when_nothing_to_say():
    """`"instructions": ""` and an absent key are different to a client that renders an
    instructions block. When no peer can emit a terse form and none supplied its own,
    omit the key — matching what the single-proxy `_augment_initialize` does (#168)."""
    from terse import policy as P
    from terse.proxy import union_primer
    quiet = P.Policy(rules=[], default_tiers=(), diff=False)
    assert union_primer([(quiet, "a")]) == ""
    assert union_primer([(quiet, "a"), (quiet, "b")]) == ""


# --- per-peer env / cwd are applied at LAUNCH, not merely serialized (#179) ---

# A peer that answers any request by reporting the environment it was actually launched
# in. The point of this file (vs. asserting a field reached the peers JSON) is that a
# written config field is not evidence the runtime reads it: the first cut of #179 wrote
# `env`/`cwd` into the peers file, `DownstreamSpec` had no such fields, and `Popen` was
# called without them — a green test proved only that json.dumps works.
_REPORT_ENV_PEER = (
    "import json,os,sys\n"
    "for line in sys.stdin:\n"
    "    line = line.strip()\n"
    "    if not line: continue\n"
    "    msg = json.loads(line)\n"
    "    if msg.get('id') is None: continue\n"
    "    body = json.dumps({'pinned': os.environ.get('TERSE_PEER_PIN'),\n"
    "                       'inherited': os.environ.get('TERSE_ROUTER_PIN'),\n"
    "                       'cwd': os.path.realpath(os.getcwd())})\n"
    "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],\n"
    "        'result':{'content':[{'type':'text','text':body}],'isError':False}})+'\\n')\n"
    "    sys.stdout.flush()\n"
)


def _peer_report(tmp_path, cout: io.StringIO) -> dict:
    """The one `tools/call` reply's payload, decoded."""
    msgs = _lines(cout)
    assert len(msgs) == 1, msgs
    return transforms.decompress(msgs[0]["result"]["content"][0]["text"])


def _call_reporting_peer(tmp_path, entry: dict) -> dict:
    cfg = _write_config(tmp_path, [{"name": "p",
                                    "command": [sys.executable, "-c", _REPORT_ENV_PEER],
                                    **entry}])
    cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "p__report"}}) + "\n")
    cout = io.StringIO()
    assert run_multi_proxy(str(cfg), PLAIN_POLICY, stdin=cin, stdout=cout) == 0
    return _peer_report(tmp_path, cout)


def test_peer_env_reaches_the_launched_child(tmp_path, monkeypatch):
    # The router's environ holds the SAME key with a different value, so this pins the
    # merge DIRECTION, not just its presence: with the operands swapped the child would
    # see the router's value. That is the credential case the fix exists for — a peer
    # whose `env` pins an API key must not silently authenticate as the router's.
    monkeypatch.setenv("TERSE_ROUTER_PIN", "from-router")
    monkeypatch.setenv("TERSE_PEER_PIN", "from-router")
    report = _call_reporting_peer(tmp_path, {"env": {"TERSE_PEER_PIN": "from-peers-file"}})
    assert report["pinned"] == "from-peers-file"


def test_peer_env_is_merged_over_the_routers_environment_not_a_replacement(tmp_path,
                                                                           monkeypatch):
    """A bare `env` mapping handed to Popen would launch the child with ONLY those two
    keys — no PATH, no HOME. An MCP client's `env` block is additive, so the router's
    must be too, or a peer that merely pins one variable loses everything else."""
    monkeypatch.setenv("TERSE_ROUTER_PIN", "from-router")
    report = _call_reporting_peer(tmp_path, {"env": {"TERSE_PEER_PIN": "x"}})
    assert report["inherited"] == "from-router"


def test_peer_without_env_still_inherits_the_router_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TERSE_ROUTER_PIN", "from-router")
    report = _call_reporting_peer(tmp_path, {})
    assert report["inherited"] == "from-router" and report["pinned"] is None


def test_peer_cwd_reaches_the_launched_child(tmp_path):
    workdir = tmp_path / "peer-workdir"
    workdir.mkdir()
    report = _call_reporting_peer(tmp_path, {"cwd": str(workdir)})
    assert report["cwd"] == str(pathlib.Path(workdir).resolve())


def test_load_multi_config_rejects_env_or_cwd_on_a_url_peer(tmp_path):
    import pytest
    for bad in ({"env": {"K": "v"}}, {"cwd": "/tmp"}):
        cfg = _write_config(tmp_path, [{"name": "h", "url": "https://x.example/mcp", **bad}])
        with pytest.raises(ValueError, match="launches no process"):
            load_multi_config(str(cfg))


def test_load_multi_config_coerces_scalar_env_values_and_rejects_containers(tmp_path):
    """A scalar is coerced, not rejected: an MCP client's own spawn coerces, so
    `{"PORT": 3000}` is a working entry in the config the peers file is generated from,
    and rejecting it takes down every peer in the fleet — not just this one — at router
    launch. A container or null is a mistake, not a convention, so it stays an error."""
    import pytest
    cfg = _write_config(tmp_path, [{"name": "p", "command": ["true"],
                                    "env": {"PORT": 3000, "DEBUG": True, "S": "x"}}])
    assert load_multi_config(str(cfg))[0].env == {"PORT": "3000", "DEBUG": "True",
                                                  "S": "x"}
    for bad in ({"K": ["a"]}, {"K": None}, {"K": {"n": 1}}):
        cfg = _write_config(tmp_path, [{"name": "p", "command": ["true"], "env": bad}])
        with pytest.raises(ValueError, match="must be scalars"):
            load_multi_config(str(cfg))
    cfg = _write_config(tmp_path, [{"name": "p", "command": ["true"], "env": "nope"}])
    with pytest.raises(ValueError, match="'env' must be an object"):
        load_multi_config(str(cfg))


def test_load_multi_config_rejects_an_empty_cwd(tmp_path):
    """`Popen(cwd="")` fails with a bare `[Errno 2] ... : ''` that names nothing."""
    import pytest
    cfg = _write_config(tmp_path, [{"name": "p", "command": ["true"], "cwd": ""}])
    with pytest.raises(ValueError, match="must not be empty"):
        load_multi_config(str(cfg))


# --- partial-listing diagnosis (#178, the half that does not need retention) -------------
#
# #178 withdrew route RETENTION after three review rounds found 11 defects, every one in the
# retention and ordering machinery. It left two gaps open, and closed with the design advice
# these tests encode: *prefer a design where the table is derived, not accumulated*, and
# *decide the semantics of "the peer did not answer" up front — an explicit JSON-RPC error,
# an empty list, a malformed result and a true timeout are four different things.*
#
# So: no route is ever carried forward. What is carried is a DIAGNOSIS of the listing that
# produced the current table, installed with it under the same seq guard and dying with it,
# feeding error text only. It cannot resurrect a tool, because it is never consulted to
# resolve one.


def _silent_after(router, parts):
    from terse.multiproxy import _PendingBroadcast
    router._merge_tools_list(_PendingBroadcast(
        kind="tools/list", client_id=9, seq=1, remaining=set(), parts=parts))
    return router._tool_state[2]


def test_the_four_ways_a_peer_contributes_nothing_are_four_different_answers():
    """Conflating any of them is where #178's round-3 defect came from, so they stay four
    values and never collapse to a boolean. `no reply` is a peer the broadcast completed
    without; `error` is a live peer refusing the method; `empty` is a peer exercising its
    right to export nothing; `malformed` is a reply whose `result.tools` is not a list."""
    for parts, expected in (
        ({}, [("a", "no reply"), ("b", "no reply")]),
        ({0: {"error": {"code": -32601}}, 1: {"result": {"tools": [{"name": "x"}]}}},
         [("a", "error")]),
        ({0: {"result": {"tools": []}}, 1: {"result": {"tools": [{"name": "x"}]}}},
         [("a", "empty")]),
        ({0: {"result": {"tools": "nope"}}, 1: {"result": {"tools": [{"name": "x"}]}}},
         [("a", "malformed")]),
        # A non-empty list of unusable entries is `empty`, not `malformed`: from the
        # client's side it is indistinguishable from exporting nothing, and `malformed`
        # would accuse a peer that answered perfectly well.
        ({0: {"result": {"tools": [{"no_name": 1}]}}, 1: {"result": {"tools": [{"name": "x"}]}}},
         [("a", "empty")]),
    ):
        _, _, _, _, router = _two_peer_router()
        try:
            assert list(_silent_after(router, parts)) == expected, parts
        finally:
            router.close_senders()


def test_the_unknown_tool_error_names_the_peers_that_missed_the_listing():
    """#178 gap 1: a non-conformant client holding names from an earlier listing gets -32601
    for a tool whose peer is ALIVE and merely was slow, and the bare message reads as "no
    such tool ever". The peer is still not routable — nothing is carried forward — but the
    client is now told which peers were absent and to re-read `tools/list`."""
    _, _, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        # peer b now misses the listing entirely; its tool leaves the table
        _silent_after(router, {0: {"result": {"tools": [{"name": "only_a"}]}}})
        assert set(router.tool_route) == {"only_a"}          # NOT retained
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "only_b"}}))
    finally:
        router.close_senders()
    err = [m for m in _lines(out) if m.get("id") == 7][0]["error"]
    assert err["code"] == -32601
    assert "b (no reply)" in err["message"]
    assert "re-read tools/list" in err["message"]


def test_a_complete_listing_adds_no_suffix_to_the_unknown_tool_error():
    """A permanent suffix would be noise on the common case, and would imply a partial
    listing where there was none — the error would then mislead in the other direction."""
    _, _, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        assert router._tool_state[2] == ()
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "nope"}}))
    finally:
        router.close_senders()
    err = [m for m in _lines(out) if m.get("id") == 7][0]["error"]
    assert "contributed no tools" not in err["message"]
    assert "re-read tools/list" not in err["message"]


def test_a_listing_refused_as_stale_does_not_install_its_diagnosis_either():
    """The diagnosis rides inside `_tool_state` precisely so it cannot outlive or precede
    the table it describes. Installing it outside the seq guard would report peers missing
    from a listing that was never installed — the accumulated-state failure mode #178
    withdrew, reintroduced through the back door."""
    from terse.multiproxy import _PendingBroadcast
    _, _, _, out, router = _two_peer_router()
    try:
        # newer, complete listing installs first
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=10, seq=2, remaining=set(),
            parts={0: {"result": {"tools": [{"name": "a1"}]}},
                   1: {"result": {"tools": [{"name": "b1"}]}}}))
        assert router._tool_state[2] == ()
        # an older partial listing times out afterwards and is refused
        router._merge_tools_list(_PendingBroadcast(
            kind="tools/list", client_id=9, seq=1, remaining={1},
            parts={0: {"result": {"tools": [{"name": "a1"}]}}}))
        assert router._tool_state == (2, router._tool_state[1], ())
        assert set(router.tool_route) == {"a1", "b1"}
    finally:
        router.close_senders()


def test_only_the_reasons_nothing_else_reports_are_warned_about(capsys):
    """`no reply` is already named by `_timeout_broadcast` — a broadcast completes on
    arrival or on timeout, so that is the only way to be absent — and `empty` is not a
    fault. Warning on either would train a reader to ignore the line that matters."""
    _, _, _, _, router = _two_peer_router()
    try:
        _silent_after(router, {0: {"error": {"code": -32601}},
                               1: {"result": {"tools": []}}})
    finally:
        router.close_senders()
    warn = capsys.readouterr().err
    assert "'a' (error)" in warn
    assert "'b'" not in warn                       # empty is not a fault


def test_the_diagnosis_never_resolves_a_call(capsys):
    """The whole safety argument: it feeds error text and nothing else. A peer named in the
    diagnosis is exactly as unroutable as it would be without it — this is what makes the
    design derived rather than accumulated."""
    t0, t1, _, out, router = _two_peer_router()
    try:
        _list_tools(router, out, [[{"name": "only_a"}], [{"name": "only_b"}]])
        _silent_after(router, {0: {"result": {"tools": [{"name": "only_a"}]}}})
        before = len(_peer_calls(t1))
        for nm in ("only_b", "b__only_b"):
            router.route_client_line(json.dumps(
                {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                 "params": {"name": nm}}))
        assert len(_peer_calls(t1)) == before      # nothing dispatched to the absent peer
    finally:
        router.close_senders()


def test_a_peer_that_exports_nothing_does_not_taint_every_unknown_tool_error():
    """Review finding. A prompts-only or resources-only peer is `empty` on EVERY listing, so
    including it would append "re-read tools/list" to every unknown-tool error this install
    ever produces — advice that can never change anything, and which contradicts the promise
    that a complete listing says nothing extra. Only reasons a re-read might actually fix
    reach the client; `empty` still appears in the diagnosis itself, it just isn't
    actionable."""
    _, _, _, out, router = _two_peer_router()
    try:
        _silent_after(router, {0: {"result": {"tools": [{"name": "only_a"}]}},
                               1: {"result": {"tools": []}}})
        assert router._tool_state[2] == (("b", "empty"),)     # still diagnosed...
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
             "params": {"name": "nope"}}))
    finally:
        router.close_senders()
    msg = [m for m in _lines(out) if m.get("id") == 9][0]["error"]["message"]
    assert "re-read tools/list" not in msg                    # ...but not surfaced
    assert "empty" not in msg


def test_an_empty_peer_does_not_mask_a_real_one_in_the_same_listing():
    """The filter drops `empty` from the client-facing list, not the whole suffix — a
    listing where one peer exports nothing and another errored must still tell the client
    about the one a re-read could fix."""
    _, _, _, out, router = _two_peer_router()
    try:
        _silent_after(router, {0: {"result": {"tools": []}},
                               1: {"error": {"code": -32601}}})
        router.route_client_line(json.dumps(
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
             "params": {"name": "nope"}}))
    finally:
        router.close_senders()
    msg = [m for m in _lines(out) if m.get("id") == 9][0]["error"]["message"]
    assert "b (error)" in msg and "a (empty)" not in msg
    assert "re-read tools/list" in msg
