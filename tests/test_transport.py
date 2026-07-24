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
from terse.proxy import Interceptor, run_proxy
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
        except json.JSONDecodeError:
            msg = {}
        self.server.requests.append(msg)  # type: ignore[attr-defined]
        mode = self.server.mode  # type: ignore[attr-defined]
        mid = msg.get("id")

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
    """A hostile downstream: answers every POST with a 302 to `self.server.target`."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming convention
        self.send_response(302)
        self.send_header("Location", self.server.target)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()


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
    # `http.client.InvalidURL` subclasses ValueError, so a downstream url from a
    # repo-committed .mcp.json escaped through _HttpSendWriter.flush() into pump(),
    # killed the client->server thread, and made the proxy exit 0 with the call
    # unanswered. It must become a legible JSON-RPC error like every other failure.
    t = HttpTransport(bad, timeout=3)
    err = json.loads(_post_one(t))["error"]
    assert err["code"] == -32000
    assert "downstream HTTP request failed" in err["message"]


def test_post_converts_invalid_header_value_to_jsonrpc_error():
    # Same escape, different trigger: http.client rejects a CRLF-bearing header value
    # with a bare ValueError. (The injection itself is blocked by the stdlib — this
    # pins that the refusal is reported rather than raised.)
    t = HttpTransport("https://example.invalid/mcp",
                      headers={"X-Evil": "a\r\nX-Injected: yes"}, timeout=3)
    err = json.loads(_post_one(t))["error"]
    assert err["code"] == -32000
