"""Regression tests for the 2026-10-28 red-team audit. Each test names the finding it pins;
every one fails against the pre-fix code."""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import sys
import threading
import time

import pytest

from terse import text_diff, transport
from terse.capture import classify_shape
from terse.measure import measure_payload
from terse.policy import Policy, Rule
from terse.proxy import Interceptor, pump
from terse.transport import HttpTransport, StdioTransport, _bounded_lines

FULL = Policy(rules=[Rule("items.*", ("minify", "tabularize", "dictionary"))])


def _reply(mid: int, text: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": text}]}})


# --- G1: the reply/request loops survive hostile downstream output --------------------

def test_stdio_non_utf8_byte_does_not_end_the_reply_stream():
    # A child writing one Latin-1 byte used to raise UnicodeDecodeError inside pump's
    # `for raw in src`, ending the thread: neither that reply nor the next one arrived.
    first = (b'{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text",'
             b'"text":"caf\xe9"}]}}\n')
    second = (_reply(2, "ok") + "\n").encode()
    code = ("import sys; sys.stdout.buffer.write(" + repr(first + second) + "); "
            "sys.stdout.buffer.flush()")
    t = StdioTransport([sys.executable, "-c", code])
    try:
        out = io.StringIO()
        pump(t.inbound(), out, lambda line: None)
    finally:
        t.close()
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["result"]["content"][0]["text"] == "caf�"
    assert json.loads(lines[1])["id"] == 2


def test_deeply_nested_reply_is_forwarded_not_fatal():
    inter = Interceptor(FULL)
    deep = "[" * 200_000
    assert inter.transform_response(deep) == deep


def test_pump_forwards_the_line_when_a_transform_raises(capsys):
    def boom(line):
        if line == "bad":
            raise RuntimeError("unanticipated")
        return None

    out = io.StringIO()
    pump(iter(["bad\n", "good\n"]), out, boom)
    assert out.getvalue() == "bad\ngood\n"
    assert "RuntimeError" in capsys.readouterr().err


def test_bounded_lines_drops_an_over_cap_line_and_keeps_the_rest(capsys):
    src = io.StringIO("short\n" + "x" * 50 + "\n" + "tail\n")
    assert list(_bounded_lines(src, 10)) == ["short\n", "tail\n"]
    assert "longer than 10" in capsys.readouterr().err


def test_bounded_lines_keeps_a_line_exactly_at_the_cap():
    src = io.StringIO("x" * 10 + "\n")
    assert list(_bounded_lines(src, 10)) == ["x" * 10 + "\n"]


class _Hostile(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = self.server.mode  # type: ignore[attr-defined]
        if mode == "deep4xx":
            body = b"[" * 200_000
            self.send_response(400)
        elif mode == "big4xx":
            body = b"E" * 4096
            self.send_response(500)
        else:  # drip: headers promptly, then a body that never finishes
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                for _ in range(50):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
            return
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _hostile(mode: str):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hostile)
    srv.mode = mode  # type: ignore[attr-defined]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/mcp"
    finally:
        srv.shutdown()
        th.join(timeout=2)
        srv.server_close()


def _post_one(t: HttpTransport) -> dict:
    t._post(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}))
    return json.loads(t._q.get(timeout=10))


def test_http_deeply_nested_error_body_becomes_a_jsonrpc_error():
    with _hostile("deep4xx") as url:
        msg = _post_one(HttpTransport(url, timeout=5))
    assert msg["id"] == 7 and msg["error"]["code"] == -32000


def test_http_error_body_read_is_capped(monkeypatch):
    monkeypatch.setattr(transport, "_MAX_RESPONSE_BYTES", 1024)
    monkeypatch.setattr(transport, "_MAX_ERROR_DETAIL_CHARS", 100)
    with _hostile("big4xx") as url:
        msg = _post_one(HttpTransport(url, timeout=5))
    assert msg["error"]["message"].endswith("Error: " + "E" * 100)


def test_http_dripped_body_hits_the_overall_deadline(monkeypatch):
    monkeypatch.setattr(transport, "_RESPONSE_DEADLINE_S", 0.5)
    with _hostile("drip") as url:
        started = time.monotonic()
        msg = _post_one(HttpTransport(url, timeout=5))
    assert time.monotonic() - started < 5
    assert "deadline" in msg["error"]["message"]


def test_text_diff_skips_texts_over_the_size_cap(monkeypatch):
    monkeypatch.setattr(text_diff, "_MAX_DIFF_CHARS", 1000)
    prev = "line\n" * 300
    assert text_diff.text_diff_encode(prev, prev + "x") is None
    small = "line\n" * 100
    assert text_diff.text_diff_encode(small, small + "x") is not None


def test_text_diff_tolerates_lone_surrogates():
    prev = "a" * 100 + "\ud800" + "b" * 100
    diff = text_diff.text_diff_encode(prev, prev + "c")
    assert diff is not None
    assert text_diff.text_diff_decode(prev, diff) == prev + "c"
    assert text_diff.text_diff_wire(prev, prev + "c") is not None


@pytest.mark.parametrize("fn", [measure_payload, classify_shape])
def test_huge_integer_literal_does_not_crash(fn):
    fn('{"a":1' + "0" * 5000 + "}")  # must not raise ValueError (int-digit limit)
