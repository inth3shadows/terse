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

from terse import transforms
from terse.lossy import _handle, _serialize
from terse.policy import Policy, Rule
from terse.proxy import Interceptor, run_proxy
from terse.transport import HttpTransport

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
        except json.JSONDecodeError:
            msg = {}
        self.server.requests.append(msg)  # type: ignore[attr-defined]
        mode = self.server.mode  # type: ignore[attr-defined]

        if mode == "fail500":
            payload = b"internal error"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        method, mid = msg.get("method"), msg.get("id")
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
        payload = f"data: {ev1}\n\ndata: {ev2}\n\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


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

    # initialize: serverInfo intact, format primer injected over HTTP exactly like stdio
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake-http"
    assert "__terse_table__" in by_id[1]["result"]["instructions"]
    # tools/call result compressed, smaller, and round-trips to the exact original
    text = by_id[2]["result"]["content"][0]["text"]
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


# --- 5: stdio_transport_error no longer rejects a URL (see also test_proxy.py) ---

def test_stdio_transport_error_accepts_a_url():
    from terse.proxy import stdio_transport_error

    assert stdio_transport_error(["https://example.com/mcp"]) is None
