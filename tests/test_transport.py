"""Transport (#5): HTTP/SSE downstream, end-to-end through `run_proxy` and directly
against `HttpTransport` + `Interceptor` where determinism requires it.

Uses a real `http.server.HTTPServer` on `127.0.0.1:0` (ephemeral port) in a background
thread as the fake remote MCP server — no real network, CI stays offline.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import threading

import pytest

from terse import transforms
from terse.lossy import _handle, _serialize
from terse.policy import Policy, Rule
from terse.proxy import PRIMER_HEAD, Interceptor, run_proxy
from terse.transport import HttpTransport, build_transport

RECORDS = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]

FULL = Policy(rules=[Rule("items.*", ("minify", "tabularize", "dictionary"))])
DROP = Policy(rules=[Rule("items.*", ("minify", "tabularize", "dictionary"),
                          fields={"result[].body": {"lossy": "drop-to-retrieve"}})])


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # silence test output
        pass

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming convention
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            msg = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, RecursionError):   # a 100k-deep request line
            msg = {}
        self.server.requests.append(msg)  # type: ignore[attr-defined]
        mode = self.server.mode  # type: ignore[attr-defined]
        mid = msg.get("id") if isinstance(msg, dict) else None   # a batch: no one id

        if mode == "fail500":
            payload = b"internal error: disk full"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if mode == "fail_jsonrpc":
            # A server that follows MCP Streamable-HTTP's allowance to send a real
            # JSON-RPC error object even on a non-2xx status.
            payload = json.dumps({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32001, "message": "missing Authorization header"}}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if mode in _FRAMING_MODES:
            _FRAMING_MODES[mode](self, mid)
            return

        method = msg.get("method")
        if method == "initialize":
            self._send_json({"jsonrpc": "2.0", "id": mid,
                             "result": {"protocolVersion": "2024-11-05", "capabilities": {},
                                        "serverInfo": {"name": "fake-http", "version": "0"}}},
                            session="sess-123")
            return
        if method == "tools/call":
            name = (msg.get("params") or {}).get("name")
            if mode == "sse":
                self._send_sse(mid)
                return
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
                             "result": {"tools": [{"name": "items.get"}]}})
            return
        # notification or anything unrecognized: 202 Accepted, empty body — valid
        # Streamable HTTP, nothing to enqueue.
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, obj: dict, session: str | None = None) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, mid) -> None:
        # Two events in one POST's response: a response plus a notification — proves a
        # single SSE stream can legitimately carry multiple JSON-RPC messages.
        ev1 = json.dumps({"jsonrpc": "2.0", "id": mid,
                          "result": {"content": [{"type": "text", "text": "first"}]}})
        ev2 = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                          "params": {"pct": 50}})
        payload = f"data: {ev1}\n\ndata: {ev2}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _raw(handler, status: int, ctype: str, payload: bytes) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


# Response shapes that are legal HTTP but were framing-unsafe on the stdio side (#582):
# each is answered to ANY request, so the test drives one tools/list.
_TOOLS = {"tools": [{"name": "items.get", "description": "a b"}]}
_FRAMING_MODES = {
    "pretty": lambda h, mid: _raw(h, 200, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": mid, "result": _TOOLS}, indent=2).encode()),
    "sse_multiline": lambda h, mid: _raw(h, 200, "text/event-stream", (
        "data: " + json.dumps({"jsonrpc": "2.0", "id": mid, "result": _TOOLS},
                              indent=2).replace("\n", "\ndata: ") + "\n\n").encode()),
    "html": lambda h, mid: _raw(h, 200, "text/html", b"<html>\n<body>login</body>\n</html>"),
    "batch": lambda h, mid: _raw(h, 200, "application/json", json.dumps([
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"p": 1}},
        {"jsonrpc": "2.0", "id": mid, "result": _TOOLS}], indent=1).encode()),
    "sse_priming": lambda h, mid: _raw(h, 200, "text/event-stream", (
        "id: 1\ndata:\n\ndata: ping\n\ndata: " + json.dumps(
            {"jsonrpc": "2.0", "id": mid, "result": _TOOLS}) + "\n\n").encode()),
    "sse_junk_only": lambda h, mid: _raw(h, 200, "text/event-stream", b"data: ping\n\n"),
    "sse_blank_only": lambda h, mid: _raw(h, 200, "text/event-stream", b"data:\n\n"),
    "not_jsonrpc": lambda h, mid: _raw(h, 200, "application/json",
                                       b'{"error": "login required"}'),
    "bom": lambda h, mid: _raw(h, 200, "application/json", ("\ufeff" + json.dumps(
        {"jsonrpc": "2.0", "id": mid, "result": _TOOLS})).encode()),
    "fail_null_id": lambda h, mid: _raw(h, 404, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": None,
         "error": {"code": -32001, "message": "session expired"}}).encode()),
    "precise": lambda h, mid: _raw(h, 200, "application/json", (
        '{\n "jsonrpc": "2.0",\n "id": ' + json.dumps(mid) + ',\n "result": '
        '{"v": 0.12345678901234567890123, "a": 1, "a": 2}\n}').encode()),
    # Review round 1 of #586: response shapes the first cut of the framing fix mishandled.
    "no_jsonrpc_member": lambda h, mid: _raw(h, 200, "application/json", json.dumps(
        {"id": mid, "result": _TOOLS}, indent=2).encode()),
    "huge_int": lambda h, mid: _raw(h, 200, "application/json", (
        '{"jsonrpc": "2.0", "id": ' + json.dumps(mid) + ', "result": {"n": '
        + "7" * 5000 + '}}\n').encode()),
    "deep_result": lambda h, mid: _raw(h, 200, "application/json", (
        '{"jsonrpc": "2.0", "id": ' + json.dumps(mid) + ', "result": {"v": '
        + "[" * 60_000 + "]" * 60_000 + '}}\n').encode()),
    "deep_only": lambda h, mid: _raw(h, 200, "application/json",
                                     b"[" * 100_000 + b"]" * 100_000),
    "ndjson": lambda h, mid: _raw(h, 200, "application/json", (json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"p": 1}}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": mid, "result": _TOOLS}) + "\n").encode()),
    "ok_null_id": lambda h, mid: _raw(h, 200, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": None,
         "error": {"code": -32600, "message": "bad request"}}).encode()),
    "fail_other_id": lambda h, mid: _raw(h, 400, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": 99, "error": {"code": -32600, "message": "bad"}}).encode()),
    "fail_string_id": lambda h, mid: _raw(h, 400, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": str(mid), "error": {"code": -32600, "message": "bad"}}
    ).encode()),
    "sse_notification_then_junk": lambda h, mid: _raw(h, 200, "text/event-stream", (
        "data: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                               "params": {"p": 1}})
        + '\n\ndata: {"jsonrpc": "2.0", "id": ' + json.dumps(mid) + ', "res\n\n').encode()),
    "empty200": lambda h, mid: _raw(h, 200, "application/json", b""),
    "blank202": lambda h, mid: _raw(h, 202, "application/json", b"  \r\n"),
    "fail_sse_null_id": lambda h, mid: _raw(h, 404, "text/event-stream", (
        "event: message\ndata: " + json.dumps(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32001, "message": "session expired"}}) + "\n\n").encode()),
    "fail_bom": lambda h, mid: _raw(h, 401, "application/json", ("﻿" + json.dumps(
        {"jsonrpc": "2.0", "id": mid,
         "error": {"code": -32001, "message": "missing Authorization header"}})).encode()),
    "fail_null_id_precise": lambda h, mid: _raw(h, 404, "application/json", (
        b'{"jsonrpc": "2.0", "id": null, "error": {"code": -32001, "message": "x", '
        b'"data": {"v": 0.12345678901234567890123, "a": 1, "a": 2}}}')),
    # Review round 2 of #586.
    "server_request_only": lambda h, mid: _raw(h, 200, "application/json",
                                               b'{"jsonrpc":"2.0","id":9,"method":"ping"}'),
    "unrelated_response": lambda h, mid: _raw(h, 200, "application/json",
                                              b'{"jsonrpc":"2.0","id":77,"result":{}}'),
    "rest_401": lambda h, mid: _raw(h, 401, "application/json",
                                    b'{"error":"login required","id":null}'),
    "rest_200": lambda h, mid: _raw(h, 200, "application/json",
                                    b'{"error":"unauthorized","id":"req_abc"}'),
    "deep_raw_newline": lambda h, mid: _raw(h, 200, "application/json", (
        '{"jsonrpc":"2.0","id":3,"result":{"v":' + "[" * 60_000 + "]" * 60_000
        + ',"s":"a\nb"}}').encode()),   # the raw LF after the depth the C scanner reaches
    "huge_unterminated": lambda h, mid: _raw(h, 200, "application/json",
                                             b'{"jsonrpc":"2.0","id":3,"result":'
                                             + b"[" * 16_000_000),
}


@contextlib.contextmanager
def _fake_server(mode: str = "normal"):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    srv.mode = mode  # type: ignore[attr-defined]
    srv.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        thread.join(timeout=2)
        srv.server_close()


def _url(srv, path: str = "/mcp") -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


# --- 1: HTTP end-to-end through run_proxy ---

def test_http_end_to_end_compresses_losslessly():
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "items.get"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    with _fake_server() as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    by_id = {json.loads(ln)["id"]: json.loads(ln)
            for ln in cout.getvalue().splitlines() if ln.strip()}

    # initialize: serverInfo intact, no eager primer (#168 phase 2 default over HTTP,
    # exactly like stdio)
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake-http"
    assert "instructions" not in by_id[1]["result"]
    # tools/call result: this is the first compressible result, so the primer arrives as a
    # LEADING block ahead of the compressed data block
    blocks = by_id[2]["result"]["content"]
    assert len(blocks) == 2
    assert PRIMER_HEAD in blocks[0]["text"]
    # the compressed data block round-trips to the exact original and is smaller
    text = blocks[1]["text"]
    expected = {"result": RECORDS}
    assert transforms.decompress(text) == expected
    assert len(text) < len(json.dumps(expected))


# --- 2: SSE-framed response ---

def test_http_sse_response_carries_both_messages_in_order():
    requests = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "items.get"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    with _fake_server(mode="sse") as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2                                  # both SSE events delivered
    msgs = [json.loads(ln) for ln in lines]
    assert msgs[0]["id"] == 5
    assert msgs[0]["result"]["content"][0]["text"] == "first"
    assert msgs[1].get("method") == "notifications/progress"  # order preserved


# --- 3: drop-to-retrieve over HTTP with zero transport-specific code ---

def test_http_drop_to_retrieve_never_touches_the_downstream():
    # A run_proxy stream can't deterministically chain "drop a field" then "retrieve
    # it" (the two requests race across the client_to_server/server_to_client pump
    # threads). Drive HttpTransport + Interceptor directly instead — same production
    # code, sequenced by hand — to prove the actual architectural claim: retrieve is
    # answered PURELY from Interceptor's in-memory store and never calls `transport`
    # at all (`answer_retrieve` doesn't even take a transport argument).
    with _fake_server() as srv:
        transport = HttpTransport(_url(srv))
        inter = Interceptor(DROP)
        req_line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": "items.body"}})
        inter.note_request(req_line)
        writer = transport.outbound()
        writer.write(req_line + "\n")
        writer.flush()                                        # synchronous POST
        resp_line = next(transport.inbound())                 # response now queued
        out_line = inter.transform_response(resp_line)
        assert transforms.DROPPED_MARKER in out_line           # dropped + marked

        assert len(srv.requests) == 1                          # only the tools/call so far

        handle = _handle("items.body", "result[].body", _serialize("B" * 400))
        assert handle in inter.dropped

        retrieve_line = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                    "params": {"name": "terse.retrieve",
                                               "arguments": {"handle": handle}}})
        reply = inter.answer_retrieve(retrieve_line)
        assert reply is not None
        reply_msg = json.loads(reply)
        assert reply_msg["result"]["content"][0]["text"] == "B" * 400
        assert not reply_msg["result"].get("isError")

        # The whole point: retrieve never issued a second HTTP request.
        assert len(srv.requests) == 1
        transport.close()


def test_http_retrieve_miss_over_run_proxy_never_reaches_the_fake_server():
    # The run_proxy-level mirror of the stdio test with the same name/shape: a MISS
    # handle is enough to prove the swallow end-to-end through the real threaded proxy
    # (no race, since there's only one request in the stream) — the reply is OUR
    # synthesized error, and the fake server never saw the call at all.
    requests = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "terse.retrieve",
                                      "arguments": {"handle": "nope"}}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    with _fake_server() as srv:
        rc = run_proxy([_url(srv)], DROP, stdin=cin, stdout=cout)
        assert len(srv.requests) == 0                          # never forwarded downstream
    assert rc == 0
    resp = json.loads([ln for ln in cout.getvalue().splitlines() if ln.strip()][0])
    assert resp["id"] == 1 and resp["result"]["isError"] is True


# --- 4: HTTP failure is fail-open, not a hang ---

def test_http_failure_is_fail_open_not_a_hang():
    requests = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "items.get"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    with _fake_server(mode="fail500") as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0                                              # returns; never hangs
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    msg = json.loads(lines[0])
    assert msg["id"] == 1                                       # matched to the request
    assert "error" in msg and "terse" in msg["error"]["message"].lower()
    # Regression: the downstream's real error body used to be discarded entirely —
    # confirm it now surfaces in the synthesized error message.
    assert "internal error: disk full" in msg["error"]["message"]


def test_http_failure_forwards_real_jsonrpc_error_verbatim():
    # Regression: a downstream that follows MCP Streamable-HTTP's allowance to send a
    # real JSON-RPC error object on a non-2xx status used to have that discarded in
    # favor of terse's own generic wrapper — the client lost the actionable detail
    # (e.g. exactly which auth header is missing).
    requests = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "items.get"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    with _fake_server(mode="fail_jsonrpc") as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    msg = json.loads(lines[0])
    assert msg["id"] == 1
    # the downstream's OWN error object, not terse's generic wrapper
    assert msg["error"]["code"] == -32001
    assert msg["error"]["message"] == "missing Authorization header"


def test_http_failure_on_notification_produces_no_reply():
    # Regression: a failed POST for a notification (no "id" — never expects a reply)
    # used to synthesize an `id: null` error and forward it anyway — an unsolicited
    # message matching no request the client sent, which a strict client could reject.
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    cin, cout = io.StringIO(notification), io.StringIO()
    with _fake_server(mode="fail500") as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert lines == []  # nothing forwarded — a notification never gets a reply


def test_http_failure_on_notification_with_jsonrpc_error_body_produces_no_reply():
    # Regression: the "forward the downstream's real JSON-RPC error verbatim" branch
    # (added alongside test_http_failure_forwards_real_jsonrpc_error_verbatim above)
    # had no has-id check at all, unlike the fail500/_maybe_enqueue_error path just
    # above — a downstream returning a well-formed JSON-RPC error object in response to
    # a NOTIFICATION would still get forwarded verbatim, violating the exact
    # no-reply-to-a-notification invariant the other two tests already pin.
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    cin, cout = io.StringIO(notification), io.StringIO()
    with _fake_server(mode="fail_jsonrpc") as srv:
        rc = run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert lines == []  # nothing forwarded — a notification never gets a reply


# --- 5: stdio_transport_error no longer rejects a URL (see also test_proxy.py) ---

def test_stdio_transport_error_accepts_a_url():
    from terse.proxy import stdio_transport_error

    assert stdio_transport_error(["https://example.com/mcp"]) is None


# --- 6: Transport.half_close()/wait() — no isinstance check needed by callers ---

def test_stdio_transport_half_close_lets_child_exit_then_wait_returns_its_code():
    import sys as _sys

    from terse.transport import StdioTransport

    # A child that reads stdin to EOF then exits 0 — proves half_close() (closing
    # stdin) is enough to let it finish on its own, and wait() then returns its
    # real exit code, exactly as run_proxy relied on transport.proc.wait() doing
    # before it went through this polymorphic method.
    t = StdioTransport([_sys.executable, "-c", "import sys\nfor _ in sys.stdin: pass\nsys.exit(0)"])
    try:
        t.half_close()
        assert t.wait() == 0
    finally:
        t.close()


def test_http_transport_half_close_ends_inbound_and_wait_is_zero():
    from terse.transport import HttpTransport

    t = HttpTransport("http://127.0.0.1:1/mcp")  # never actually connected to
    t.half_close()  # no persistent connection — closes outright
    assert list(t.inbound()) == []  # sentinel already drained; iterator ends immediately
    assert t.wait() == 0  # no process — always 0


# --- URL scheme allowlist: no local-file read / SSRF via a config-supplied url ---

@pytest.mark.parametrize("bad_url", [
    "file:///etc/passwd",              # local-file read
    "ftp://example.com/x",             # urllib honors ftp too
    "data:text/plain,pwned",           # inline data
    "gopher://example.com/",           # classic SSRF smuggling scheme
])
def test_http_transport_rejects_non_http_scheme(bad_url):
    with pytest.raises(ValueError, match="not allowed"):
        HttpTransport(bad_url)


def test_build_transport_rejects_file_url_scheme():
    # build_transport routes anything containing "://" to HttpTransport, so the scheme
    # guard there is what stops a `file://` "url" from ever being opened.
    with pytest.raises(ValueError, match="not allowed"):
        build_transport(["file:///etc/passwd"])


@pytest.mark.parametrize("bad_url", [
    "http://169.254.169.254/latest/meta-data/",   # AWS/Azure/GCP/DO instance metadata
    "http://169.254.169.254/",
    "https://169.254.169.254/computeMetadata/v1/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://[fe80::1]/",                           # IPv6 link-local
])
def test_http_transport_blocks_link_local_metadata(bad_url):
    # A downstream URL can come from an untrusted, repo-committed .mcp.json; the cloud
    # instance-metadata endpoint is never a real MCP server, so it is refused outright.
    with pytest.raises(ValueError, match="metadata"):
        HttpTransport(bad_url)


def test_http_transport_still_allows_loopback_and_private():
    # The metadata guard must NOT break the first-class local/homelab MCP use case.
    assert HttpTransport("http://127.0.0.1:4000/mcp")
    assert HttpTransport("http://192.168.1.50:8080/mcp")   # ordinary private LAN host


def test_build_transport_still_allows_http_and_https():
    # HttpTransport.__init__ does no I/O, so these construct without connecting.
    assert isinstance(build_transport(["http://localhost:1/mcp"]), HttpTransport)
    assert isinstance(build_transport(["https://example.com/mcp"]), HttpTransport)


def test_run_proxy_rejects_file_url_scheme_as_config_error(capsys):
    # End-to-end: a disallowed scheme is a clean config error (exit 2), not a traceback
    # and not a silent file read into the client stream.
    rc = run_proxy(["file:///etc/passwd"], FULL, stdin=io.StringIO(""), stdout=io.StringIO())
    assert rc == 2
    assert "not allowed" in capsys.readouterr().err


def test_http_transport_refuses_credential_headers_over_remote_cleartext():
    # Parity with fluency.openai_answerer's TLS guard (audit fix #3): a Bearer/token
    # header over http to a remote host puts the credential on the wire unencrypted.
    import pytest

    for name in ("Authorization", "X-Api-Key", "Proxy-Token", "Cookie", "client-secret"):
        with pytest.raises(ValueError, match="cleartext http"):
            HttpTransport("http://api.example.com/mcp", headers={name: "v"})
    # https, loopback http, and non-sensitive headers over http all still construct
    assert HttpTransport("https://api.example.com/mcp", headers={"Authorization": "v"})
    assert HttpTransport("http://127.0.0.1:4000/mcp", headers={"Authorization": "v"})
    assert HttpTransport("http://localhost:4000/mcp", headers={"X-Api-Key": "v"})
    assert HttpTransport("http://api.example.com/mcp", headers={"X-Trace-Id": "v"})
    assert HttpTransport("http://api.example.com/mcp")


# --- redirect guards (security audit 2026-07-23) --------------------------------------
# Every guard in HttpTransport.__init__ ran ONCE, against the configured URL — but
# `urlopen` follows up to 10 redirects, and none of them were re-checked. All four tests
# below fail against the pre-fix code (verified): the redirect was followed, and CPython's
# HTTPRedirectHandler carried every request header onto the new request.

class _RedirectHandler(http.server.BaseHTTPRequestHandler):
    """A hostile downstream: answers every POST with a 302 to `self.server.target`.

    Also serves the SAME-ORIGIN hop (a relative `Location`, which arrives here as a GET),
    recording its headers into `server.seen_final` — that is what lets a test distinguish
    "credential scoped correctly" from "credential stripped unconditionally"."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming convention
        self.send_response(302)
        self.send_header("Location", self.server.target)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 — the 302 turns the POST into a GET
        self.server.seen_final.append(dict(self.headers))  # type: ignore[attr-defined]
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"same_origin": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _SinkHandler(http.server.BaseHTTPRequestHandler):
    """The redirect target — records the headers it was handed, so a leaked credential
    is observable rather than inferred."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_GET(self) -> None:  # noqa: N802 — a 302 turns the POST into a GET
        self.server.seen.append(dict(self.headers))  # type: ignore[attr-defined]
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"leaked": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "ATTACKER-CHOSEN")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def _redirect_pair(target: str | None = None):
    """(redirector, sink) — the redirector 302s to the sink unless `target` overrides."""
    sink = http.server.HTTPServer(("127.0.0.1", 0), _SinkHandler)
    sink.seen = []  # type: ignore[attr-defined]
    red = http.server.HTTPServer(("127.0.0.1", 0), _RedirectHandler)
    red.target = target or f"http://127.0.0.1:{sink.server_address[1]}/steal"  # type: ignore[attr-defined]
    red.seen_final = []  # type: ignore[attr-defined]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (sink, red)]
    for t in threads:
        t.start()
    try:
        yield red, sink
    finally:
        for s in (sink, red):
            s.shutdown()
        for t in threads:
            t.join(timeout=2)
        for s in (sink, red):
            s.server_close()


def _post_one(transport: HttpTransport, mid: int = 1) -> str:
    transport._post(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/list"}))
    return transport._q.get(timeout=10)


def test_redirect_strips_credential_headers_cross_origin():
    # The confirmed leak: `Authorization: Bearer <secret>` was re-sent verbatim to the
    # redirect target, which is a DIFFERENT host:port than the one the operator scoped
    # the credential to.
    with _redirect_pair() as (red, sink):
        t = HttpTransport(_url(red), headers={"Authorization": "Bearer SUPERSECRET"})
        _post_one(t)
        assert sink.seen, "redirect target was never reached — test is not exercising the hop"
        leaked = [h for h in sink.seen if "Authorization" in h]
        assert not leaked, f"credential survived a cross-origin redirect: {leaked}"


def test_redirect_keeps_non_credential_headers():
    # Stripping must be surgical: a legitimate redirect (CDN, renamed path) still works,
    # and headers that carry no secret are not collateral damage.
    with _redirect_pair() as (red, sink):
        t = HttpTransport(_url(red), headers={"X-Trace-Id": "abc123"})
        reply = _post_one(t)
        assert json.loads(reply)["result"] == {"leaked": True}   # the hop still succeeded
        assert sink.seen[0].get("X-Trace-Id") == "abc123"


def test_redirect_to_metadata_address_is_refused():
    # The construction-time metadata guard, re-applied per hop. Refused BEFORE any
    # connection is attempted, so this needs no network and cannot hang on a timeout.
    with _redirect_pair(target="http://169.254.169.254/latest/meta-data/") as (red, _sink):
        t = HttpTransport(_url(red), timeout=5)
        err = json.loads(_post_one(t))["error"]["message"]
        assert "metadata" in err and "169.254.169.254" in err


def test_redirect_to_disallowed_scheme_is_refused():
    # urllib's own redirect handler permits http/https/FTP, which sidesteps
    # _ALLOWED_URL_SCHEMES entirely. Re-checking the scheme per hop closes that.
    with _redirect_pair(target="ftp://ftp.example.com/pub/x") as (red, _sink):
        t = HttpTransport(_url(red), timeout=5)
        err = json.loads(_post_one(t))["error"]["message"]
        assert "scheme" in err and "ftp" in err


# --- _post's "never raises" contract ---------------------------------------------------

@pytest.mark.parametrize("bad", [
    "http://exa mple.com/mcp",     # http.client.InvalidURL (a ValueError) — passes urlsplit
    "https://ex ample.com/mcp",
])
def test_post_converts_malformed_url_to_jsonrpc_error(bad):
    # `_post` documents "Never raises", but only caught HTTPError/URLError/OSError.
    # `http.client.InvalidURL` derives from `HTTPException`, NOT from `ValueError` (the
    # audit got this backwards and this test is what caught it), so a downstream url from a
    # repo-committed .mcp.json escaped through _HttpSendWriter.flush() into pump(),
    # killed the client->server thread, and made the proxy exit 0 with the call
    # unanswered. It must become a legible JSON-RPC error like every other failure.
    t = HttpTransport(bad, timeout=3)
    err = json.loads(_post_one(t))["error"]
    assert err["code"] == -32000
    assert "downstream HTTP request failed" in err["message"]


def test_post_converts_invalid_header_value_to_jsonrpc_error():
    # Same escape, different trigger: http.client rejects a CRLF-bearing header value
    # with a bare ValueError — this one really IS a ValueError, unlike InvalidURL above,
    # which is why the except clause needs both types. (The injection itself is blocked by the stdlib — this
    # pins that the refusal is reported rather than raised.)
    t = HttpTransport("https://example.invalid/mcp",
                      headers={"X-Evil": "a\r\nX-Injected: yes"}, timeout=3)
    err = json.loads(_post_one(t))["error"]
    assert err["code"] == -32000


# --- adversarial-review follow-ups (2026-07-23) ---------------------------------------
# Six defects the first cut of the redirect fix still had. Each test below fails against
# that first cut, not merely against the pre-audit code.

@pytest.mark.parametrize("encoded", [
    "2852039166",              # decimal
    "0251.0376.0251.0376",     # octal
    "0xa9.0xfe.0xa9.0xfe",     # hex
    "169.254.43518",           # short form (a.b.c)
    "169.254.169.254.",        # fully-qualified, trailing dot
])
def test_metadata_guard_catches_alternate_ipv4_encodings(encoded):
    # `ipaddress.ip_address` parses ONLY dotted-quad, but glibc's resolver — the thing
    # that actually dials — accepts all of these, and every one is 169.254.169.254. The
    # guard classified them "not an IP, therefore allowed", which is a complete bypass of
    # the exact address it exists to refuse.
    with pytest.raises(ValueError, match="metadata"):
        HttpTransport(f"http://{encoded}/latest/meta-data/")


def test_metadata_guard_still_allows_ordinary_hosts():
    # The canonicalizer must not start refusing real servers: a DNS name is unparseable
    # as an IP and stays allowed, as do loopback and private LAN literals.
    assert HttpTransport("https://mcp.example.com/mcp")
    assert HttpTransport("http://127.0.0.1:4000/mcp")
    assert HttpTransport("http://192.168.1.50:8080/mcp")
    assert HttpTransport("http://3232235826/mcp")   # decimal 192.168.1.50 — private, fine


def test_redirect_keeps_credential_on_same_origin_hop():
    # THE MUTATION TEST. Replacing `if _origin(...) != self._origin:` with `if True:`
    # passed the whole suite: the only "control" used a header that is never stripped at
    # any origin, so an unconditional strip was undetectable. A same-origin redirect (a
    # renamed path) must KEEP the credential — that is what makes the strip a scoping
    # rule rather than a blanket removal.
    with _redirect_pair(target="/final") as (red, _sink):
        t = HttpTransport(_url(red), headers={"Authorization": "Bearer KEEPME"})
        _post_one(t)
        assert red.seen_final, "the same-origin hop was never taken"
        assert red.seen_final[0].get("Authorization") == "Bearer KEEPME"


@pytest.mark.parametrize("url_a,url_b", [
    ("https://h.example.com/mcp", "https://h.example.com:443/mcp"),
    ("http://h.example.com/mcp", "http://h.example.com:80/mcp"),
    ("http://[::1]:9/mcp", "http://[0:0:0:0:0:0:0:1]:9/mcp"),
    ("http://127.0.0.1:9/mcp", "http://2130706433:9/mcp"),
])
def test_origin_normalizes_default_ports_and_ip_spellings(url_a, url_b):
    # Two spellings of one origin compared UNEQUAL, so a legitimate same-host redirect
    # silently lost its credential and the downstream answered 401 with no explanation.
    from terse.transport import _origin

    assert _origin(url_a) == _origin(url_b)


@pytest.mark.parametrize("name", ["X-Auth", "Authentication", "Mcp-Session-Id",
                                  "X-Bearer", "Api-Credential", "X-Signature"])
def test_credential_headers_beyond_authorization_are_stripped(name):
    # `_SENSITIVE_HEADER_TOKENS` held "authorization", which matched `Authorization` and
    # nothing else — `X-Auth` and `Authentication` were observed reaching a cross-origin
    # sink intact.
    with _redirect_pair() as (red, sink):
        t = HttpTransport(_url(red), headers={name: "s3cret"})
        _post_one(t)
        assert sink.seen and name not in sink.seen[0], f"{name} survived a cross-origin hop"


def test_redirect_target_cannot_fixate_the_mcp_session():
    # A hostile downstream 302s to a host it controls, which returns its own
    # Mcp-Session-Id. Storing that meant terse presented an ATTACKER-CHOSEN session on
    # every subsequent POST to the legitimate downstream.
    with _redirect_pair() as (red, _sink):
        t = HttpTransport(_url(red))
        _post_one(t)
        assert t.session != "ATTACKER-CHOSEN"
        assert t.session is None


# --- framing: every downstream body becomes newline-free JSON-RPC lines (#582) ---
def _one_list_call(mode: str, request: dict | None = None) -> list[str]:
    request = request or {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    cin = io.StringIO(json.dumps(request) + "\n")
    cout = io.StringIO()
    with _fake_server(mode=mode) as srv:
        assert run_proxy([_url(srv)], FULL, stdin=cin, stdout=cout) == 0
    return [ln for ln in cout.getvalue().splitlines() if ln.strip()]


@pytest.mark.parametrize("mode", ["pretty", "sse_multiline"])
def test_multi_line_body_reaches_the_client_as_one_message(mode):
    lines = _one_list_call(mode)
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"jsonrpc": "2.0", "id": 3, "result": _TOOLS}


def test_batch_body_is_split_into_its_messages():
    msgs = [json.loads(ln) for ln in _one_list_call("batch")]
    assert [m.get("method") for m in msgs] == ["notifications/progress", None]
    assert msgs[1]["id"] == 3


def test_non_json_body_answers_the_request_with_an_error():
    msgs = [json.loads(ln) for ln in _one_list_call("html")]
    assert len(msgs) == 1 and msgs[0]["id"] == 3
    assert "not a JSON-RPC message" in msgs[0]["error"]["message"]
    assert "login" not in msgs[0]["error"]["message"]   # body content never echoed


def test_error_body_with_null_id_is_answered_under_the_request_id():
    msgs = [json.loads(ln) for ln in _one_list_call("fail_null_id")]
    assert len(msgs) == 1 and msgs[0]["id"] == 3
    assert msgs[0]["error"]["message"] == "session expired"


def test_compaction_keeps_literal_numbers_and_duplicate_keys():
    # Whitespace removal is lexical: the over-precise float and the duplicate key a
    # json round-trip would rewrite reach the client exactly as the server sent them.
    lines = _one_list_call("precise")
    assert lines == ['{"jsonrpc":"2.0","id":3,"result":{"v":0.12345678901234567890123,'
                     '"a":1,"a":2}}']


def test_sse_priming_and_keepalive_events_are_skipped_not_errored():
    msgs = [json.loads(ln) for ln in _one_list_call("sse_priming")]
    assert msgs == [{"jsonrpc": "2.0", "id": 3, "result": _TOOLS}]


@pytest.mark.parametrize("mode", ["sse_junk_only", "sse_blank_only"])
def test_sse_stream_with_no_message_answers_with_an_error(mode):
    msgs = [json.loads(ln) for ln in _one_list_call(mode)]
    assert len(msgs) == 1 and "no JSON-RPC message" in msgs[0]["error"]["message"]


def test_json_object_that_is_not_jsonrpc_answers_with_an_error():
    msgs = [json.loads(ln) for ln in _one_list_call("not_jsonrpc")]
    assert len(msgs) == 1 and "not a JSON-RPC message" in msgs[0]["error"]["message"]


def test_bom_prefixed_body_is_forwarded():
    msgs = [json.loads(ln) for ln in _one_list_call("bom")]
    assert msgs == [{"jsonrpc": "2.0", "id": 3, "result": _TOOLS}]


def test_lexers_handle_escapes_and_nesting():
    from terse.transport import _compact_json_text, _split_top_level_array
    text = '[ {"a": [1, 2]},\n {"b": "\\"] , {x"} ]'
    compact = _compact_json_text(text)
    assert compact == '[{"a":[1,2]},{"b":"\\"] , {x"}]'
    assert _split_top_level_array(compact) == ['{"a":[1,2]}', '{"b":"\\"] , {x"}']


def test_compaction_is_linear_on_a_truncated_string_full_of_escaped_quotes():
    import time as _time

    from terse.transport import _compact_json_text
    text = '\n{"a":"' + '\\"' * 200_000      # cut off mid-string
    start = _time.monotonic()
    assert _compact_json_text(text) == text.lstrip("\n")
    assert _time.monotonic() - start < 2.0


# --- review round 1 of #586: one rule decides what answers the request (B1-B7) ---
def _transport_reply(mode: str, line: str | None = None) -> list[str]:
    """What the transport itself enqueues for one outgoing line (by default a request with
    id 3), before the interceptor."""
    with _fake_server(mode=mode) as srv:
        t = HttpTransport(_url(srv), timeout=5)
        t._post(line or json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}))
    out = []
    while not t._q.empty():
        out.append(t._q.get_nowait())
    return out


def test_response_without_a_jsonrpc_member_is_forwarded():   # B1
    msgs = [json.loads(ln) for ln in _one_list_call("no_jsonrpc_member")]
    assert msgs == [{"id": 3, "result": _TOOLS}]


@pytest.mark.parametrize("mode", ["huge_int", "deep_result"])
def test_body_past_python_parser_limits_is_forwarded_not_replaced(mode):   # B1
    # A 5000-digit int (int max str digits) or 60k-deep nesting (recursion limit) is
    # valid JSON that json.loads refuses; the real result must still reach the client.
    lines = _transport_reply(mode)
    assert len(lines) == 1 and "\n" not in lines[0]
    assert "error" not in lines[0] and lines[0].startswith('{"jsonrpc":"2.0","id":3,')


def test_deeply_nested_non_message_answers_with_an_error_not_a_crash():
    msgs = [json.loads(ln) for ln in _transport_reply("deep_only")]
    assert len(msgs) == 1 and msgs[0]["id"] == 3 and msgs[0]["error"]["code"] == -32000


def test_newline_delimited_json_body_is_split_into_its_messages():   # B2
    msgs = [json.loads(ln) for ln in _one_list_call("ndjson")]
    assert [m.get("method") for m in msgs] == ["notifications/progress", None]
    assert msgs[1] == {"jsonrpc": "2.0", "id": 3, "result": _TOOLS}


@pytest.mark.parametrize("mode", ["ok_null_id", "fail_other_id", "fail_string_id",
                                  "fail_sse_null_id"])
def test_response_not_keyed_to_the_request_is_answered_under_its_id(mode):   # B3 B4 B7
    msgs = [json.loads(ln) for ln in _one_list_call(mode)]
    assert len(msgs) == 1 and msgs[0]["id"] == 3
    assert msgs[0]["error"]["code"] in (-32600, -32001)   # the server's own error


def test_error_status_body_gets_the_same_framing_as_a_success_body():   # B7
    msgs = [json.loads(ln) for ln in _one_list_call("fail_bom")]
    assert msgs == [{"jsonrpc": "2.0", "id": 3, "error": {
        "code": -32001, "message": "missing Authorization header"}}]


def test_rekeyed_response_keeps_its_literal_text():   # deferred re-key item, free here
    # Only the id is spliced; the one-line body is otherwise exactly what the server sent.
    lines = _one_list_call("fail_null_id_precise")
    assert lines == ['{"jsonrpc": "2.0", "id": 3, "error": {"code": -32001, "message": "x", '
                     '"data": {"v": 0.12345678901234567890123, "a": 1, "a": 2}}}']


def test_sse_stream_with_only_a_notification_and_junk_answers_the_request():   # B5
    msgs = [json.loads(ln) for ln in _one_list_call("sse_notification_then_junk")]
    assert [m.get("method") for m in msgs] == ["notifications/progress", None]
    assert msgs[1]["id"] == 3 and "no JSON-RPC message" in msgs[1]["error"]["message"]


@pytest.mark.parametrize("mode", ["empty200", "blank202"])
def test_empty_success_body_to_a_request_answers_with_an_error(mode):   # B6
    msgs = [json.loads(ln) for ln in _one_list_call(mode)]
    assert len(msgs) == 1 and msgs[0]["id"] == 3 and msgs[0]["error"]["code"] == -32000


@pytest.mark.parametrize("mode", ["empty200", "blank202", "fail_other_id"])
def test_notification_still_gets_no_synthesized_reply(mode):   # B6 B4 guard
    lines = _one_list_call(mode, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert lines == []


# --- review round 2 of #586: what expects a reply, what is a response, never raise ---
_CLIENT_RESPONSE = {"jsonrpc": "2.0", "id": 5, "result": {"roots": []}}


@pytest.mark.parametrize("mode", ["blank202", "empty200"])
def test_client_response_to_a_server_request_gets_nothing_back(mode):   # R2-1
    # The client answering a server-initiated request (roots/list) is not a request: the
    # spec's empty 202 means "accepted", and id 5 is in the SERVER's id space.
    assert _transport_reply(mode, json.dumps(_CLIENT_RESPONSE)) == []


@pytest.mark.parametrize("mode, expected", [
    ("server_request_only", '{"jsonrpc":"2.0","id":9,"method":"ping"}'),
    ("unrelated_response", '{"jsonrpc":"2.0","id":77,"result":{}}')])
def test_client_response_is_never_rekeyed_or_answered_with_an_error(mode, expected):   # R2-1
    assert _transport_reply(mode, json.dumps(_CLIENT_RESPONSE)) == [expected]


@pytest.mark.parametrize("mode", ["rest_401", "rest_200"])
def test_rest_error_body_is_not_taken_for_a_jsonrpc_response(mode):   # R2-2
    msgs = [json.loads(ln) for ln in _transport_reply(mode)]
    assert len(msgs) == 1 and msgs[0]["id"] == 3 and msgs[0]["error"]["code"] == -32000
    if mode == "rest_200":
        assert "unauthorized" not in msgs[0]["error"]["message"]   # a 2xx is never echoed


def test_deeply_nested_request_line_never_raises():   # R2-3
    line = ('{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"a":'
            + "[" * 100_000 + "]" * 100_000 + "}}")
    msgs = [json.loads(ln) for ln in _transport_reply("html", line)]
    assert len(msgs) == 1 and msgs[0]["id"] == 3 and msgs[0]["error"]["code"] == -32000


@pytest.mark.parametrize("mode", ["html", "empty200"])
def test_batch_request_gets_one_error_per_request_id(mode):   # R2-4
    batch = json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "a"},
                        {"jsonrpc": "2.0", "method": "notifications/n"},
                        {"jsonrpc": "2.0", "id": 2, "method": "b"}])
    msgs = [json.loads(ln) for ln in _transport_reply(mode, batch)]
    assert sorted(m["id"] for m in msgs) == [1, 2]
    assert all(m["error"]["code"] == -32000 for m in msgs)


def test_lexed_deep_body_with_a_raw_newline_is_not_forwarded():   # R2-5
    lines = _transport_reply("deep_raw_newline")
    assert all("\n" not in ln and "\r" not in ln for ln in lines)
    assert len(lines) == 1 and json.loads(lines[0])["error"]["code"] == -32000


def test_lexical_fallback_is_bounded_on_a_huge_hostile_body():   # R2-6
    import time as _time
    start = _time.monotonic()
    msgs = [json.loads(ln) for ln in _transport_reply("huge_unterminated")]
    assert _time.monotonic() - start < 3.0
    assert len(msgs) == 1 and msgs[0]["id"] == 3 and msgs[0]["error"]["code"] == -32000
