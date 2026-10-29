"""A lone surrogate never reaches a UTF-8 stream raw (#564).

A server may send `\\ud800` escaped, in pure ASCII. `json.loads` turns it into the code
point, and every re-serialization with `ensure_ascii=False` wrote it back raw. A UTF-8
stdout cannot encode that, so the write raised, the reply thread ended, and the call and
every later reply on the connection were lost (live repro: 0 bytes to the client).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from threading import Lock

import pytest

from terse import transforms
from terse.multiproxy import Peer, Router
from terse.policy import Policy, Rule
from terse.proxy import Interceptor, pump, run_proxy

POL = Policy(rules=[Rule("x", ("minify", "tabularize", "dictionary"))])
SURR = "a\ud800b"


def _strict():
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict",
                            newline="\n", write_through=True)


def _written(stream) -> list[dict]:
    stream.flush()
    return [json.loads(ln) for ln in stream.buffer.getvalue().decode("utf-8").splitlines()]


def _result(mid, text):
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": text}]}})


@pytest.mark.parametrize("text", [SURR, "\udfff", "x😀y", "plain é ✓ 😀"])
def test_escape_surrogates_is_lossless_and_writable(text):
    line = json.dumps([text, {"k": text}], ensure_ascii=False)
    out = transforms.escape_surrogates(line)
    assert json.loads(out) == json.loads(line)
    out.encode("utf-8")                      # strict: raises on a raw surrogate
    assert "😀" not in text or "😀" in out   # real characters are left alone


def test_ascii_is_returned_unchanged():
    line = '{"a":"\\ud800"}'
    assert transforms.escape_surrogates(line) is line


def test_minify_keeps_the_escape():
    out = transforms.minify([SURR])
    assert out == '["a\\ud800b"]' and json.loads(out) == [SURR]


def test_a_compressed_result_with_a_surrogate_reaches_a_strict_client():
    inter = Interceptor(POL, lazy_primer=False)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "x"}}))
    rows = [{"id": i, "state": "open", "s": SURR} for i in range(30)]
    dst = _strict()
    pump([_result(1, json.dumps(rows)) + "\n"], dst, inter.transform_response)
    (msg,) = _written(dst)
    text = msg["result"]["content"][0]["text"]
    assert "__terse_" in text                 # it was compressed, not passed through
    assert transforms.decompress(text) == rows


def test_an_uncompressed_text_with_a_raw_surrogate_reaches_a_strict_client():
    """The text block itself holds the code point (the server escaped it in the outer
    JSON). Passed through, it is re-serialized when the message changes."""
    inter = Interceptor(POL)                 # lazy primer: the message is rewritten
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "x"}}))
    rows = json.dumps([{"id": i, "state": "open"} for i in range(30)])
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
        {"type": "text", "text": rows}, {"type": "text", "text": "note " + SURR}]}})
    dst = _strict()
    pump([line + "\n"], dst, inter.transform_response)
    (msg,) = _written(dst)
    texts = [b["text"] for b in msg["result"]["content"]]
    assert "note " + SURR in texts


FAKE = """
import sys
sys.stdin.readline()
sys.stdout.write(%r + "\\n")
sys.stdout.flush()
"""


def test_run_proxy_answers_instead_of_losing_the_reply(tmp_path: Path):
    server = tmp_path / "srv.py"
    rows = [{"id": i, "state": "open", "s": SURR} for i in range(30)]
    server.write_text(FAKE % _result(1, json.dumps(rows)))
    cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "x"}}) + "\n")
    cout = _strict()
    assert run_proxy([sys.executable, str(server)], POL, stdin=cin, stdout=cout) == 0
    (msg,) = [m for m in _written(cout) if m.get("id") == 1]
    assert transforms.decompress(msg["result"]["content"][-1]["text"]) == rows


class _StrictPeer:
    def __init__(self):
        self.out = _strict()

    def inbound(self):
        return iter([])

    def outbound(self):
        return self.out

    def close(self):
        pass


def test_router_writes_to_client_and_peers_survive_a_raw_surrogate():
    t = _StrictPeer()
    out = _strict()
    router = Router([Peer("a", t, Interceptor(POL))], out, Lock(), broadcast_timeout=1000)
    try:
        raw = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"v": SURR}},
                         ensure_ascii=False)
        router._write_client(raw, origin="router")
        router._write_peer(0, json.dumps({"jsonrpc": "2.0", "id": 8, "method": "x",
                                          "params": {"v": SURR}}, ensure_ascii=False))
    finally:
        router.close_senders()
    assert _written(out) == [{"jsonrpc": "2.0", "id": 7, "result": {"v": SURR}}]
    assert _written(t.out)[-1]["params"]["v"] == SURR


class _Watched(io.TextIOWrapper):
    """A strict client stream that lets the test wait for the first reply."""

    def __init__(self):
        super().__init__(io.BytesIO(), encoding="utf-8", errors="strict", newline="\n",
                         write_through=True)
        import threading
        self.first = threading.Event()

    def write(self, s):
        n = super().write(s)
        if s.endswith("\n"):
            self.first.set()
        return n


def test_a_retrieve_answer_with_a_surrogate_reaches_a_strict_client(tmp_path: Path):
    """Also the drop itself: the handle hash used to raise on the surrogate, so the field
    silently stayed inline."""
    from terse.lossy import DROP_KEY, RETRIEVE_TOOL
    pol = Policy(rules=[Rule("x", ("minify",),
                             fields={"body": {"lossy": "drop-to-retrieve", "min": 1}})])
    body = SURR * 200
    server = tmp_path / "srv.py"
    server.write_text(FAKE % _result(1, json.dumps({"body": body})))
    cout = _Watched()

    def client():
        yield json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "x"}}) + "\n"
        assert cout.first.wait(10)
        (msg,) = _written(cout)
        handle = json.loads(msg["result"]["content"][-1]["text"])["body"][DROP_KEY]
        yield json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": RETRIEVE_TOOL,
                                     "arguments": {"handle": handle}}}) + "\n"

    assert run_proxy([sys.executable, str(server)], pol, stdin=client(), stdout=cout) == 0
    (answer,) = [m for m in _written(cout) if m.get("id") == 2]
    assert answer["result"]["content"][0]["text"] == body     # a string is served as-is


def test_hashes_and_sizes_of_payload_text_take_a_surrogate():
    """Each raised `UnicodeEncodeError` on a lone surrogate: the args key (diff base),
    the offload size check, the capture id and the drop handle. A string without one
    hashes exactly as before."""
    from terse import capture, lossy, proxy
    assert proxy._args_key({"q": SURR}) != proxy._args_key({"q": "ab"})
    assert proxy._args_key({"q": "é"}) == __import__("hashlib").sha1(
        json.dumps({"q": "é"}, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest()[:12]
    assert proxy.over_limit(SURR * 10, 10) is True
    assert capture._sha8(SURR) != capture._sha8("ab")
    assert lossy._handle("t", "p", SURR) != lossy._handle("t", "p", "ab")
