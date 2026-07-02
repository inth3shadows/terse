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
from collections import OrderedDict
from threading import Lock

from terse import __version__, transforms
from terse.lossy import _handle, _serialize
from terse.multiproxy import (
    Peer,
    Router,
    load_multi_config,
    run_multi_proxy,
)
from terse.policy import Policy, Rule
from terse.proxy import Interceptor
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
            store: "OrderedDict[str, object]" = OrderedDict()
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
    store: "OrderedDict[str, object]" = OrderedDict()
    lock = Lock()
    a = Interceptor(DROP_POLICY, store=store, store_lock=lock)
    b = Interceptor(DROP_POLICY, store=store, store_lock=lock)
    a._drop_put("h", "value")
    assert b.dropped["h"] == "value"                 # visible from the OTHER Interceptor
    reply = json.loads(b.answer_retrieve(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "terse.retrieve", "arguments": {"handle": "h"}}})))
    assert reply["result"]["content"][0]["text"] == "value"
