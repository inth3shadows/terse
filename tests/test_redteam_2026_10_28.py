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
        return

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


# --- G2: secrets are never printed or sent loosely ------------------------------------

@pytest.mark.parametrize("args", [
    ["--access-token", "S1"], ["--client-secret=S1"], ["--auth-token", "S1"],
    ["-e", "GH_TOKEN=S1"], ["API_KEY=S1"], ["https://u:S1@h.example/mcp"],
    ["https://h.example/mcp?token=S1"],
])
def test_install_print_masks_every_secret_shape(args):
    from terse.cli import _full_cmd

    rendered = _full_cmd({"command": "npx", "args": ["srv", *args]})
    assert "S1" not in rendered


def test_install_print_keeps_ordinary_args_readable():
    from terse.cli import _full_cmd

    rendered = _full_cmd({"command": "npx", "args": ["-y", "pkg@1.2.3", "--port", "8080",
                                                     "https://h.example/mcp"]})
    assert rendered == "npx -y pkg@1.2.3 --port 8080 https://h.example/mcp"


def test_install_print_masks_url_entries_and_url_changes():
    from terse.cli import _entry_change_lines, _full_cmd

    assert "S2" not in _full_cmd({"url": "https://u:S2@h.example/mcp?k=S2"})
    lines = _entry_change_lines({"url": "https://h.example/?t=S3"},
                                {"url": "https://h2.example/?t=S4"})
    joined = "\n".join(lines)
    assert "url CHANGED" in joined and "S3" not in joined and "S4" not in joined


class _BearerRedirect(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _go(self) -> None:
        self.server.seen.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]
        target = self.server.target  # type: ignore[attr-defined]
        if target:
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b'{"data": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _go  # noqa: N815


@contextlib.contextmanager
def _bearer_pair():
    sink = http.server.HTTPServer(("127.0.0.1", 0), _BearerRedirect)
    sink.seen, sink.target = [], None  # type: ignore[attr-defined]
    # "localhost" vs 127.0.0.1: a different ORIGIN on the same loopback machine.
    red = http.server.HTTPServer(("127.0.0.1", 0), _BearerRedirect)
    red.seen = []  # type: ignore[attr-defined]
    red.target = f"http://localhost:{sink.server_address[1]}/v1/models"  # type: ignore[attr-defined]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (sink, red)]
    for th in threads:
        th.start()
    try:
        yield f"http://127.0.0.1:{red.server_address[1]}", sink
    finally:
        for s in (sink, red):
            s.shutdown()
            s.server_close()


def test_model_limit_discovery_does_not_forward_the_key_cross_origin():
    from terse.cli import _discover_model_limits

    with _bearer_pair() as (base, sink):
        _discover_model_limits(base + "/v1", "SECRETKEY", ["m"])
    assert sink.seen, "the redirect was not followed at all"
    assert all(h is None for h in sink.seen)


def test_model_limit_discovery_refuses_cleartext_remote_key(capsys):
    from terse.cli import _discover_model_limits

    assert _discover_model_limits("http://gateway.example/v1", "K", ["m"]) == {}
    assert "cleartext" in capsys.readouterr().err


def test_answerers_use_the_guarded_opener():
    # dropeval's and fluency's answerers sent the bearer key through the module-global
    # urlopen, which forwards Authorization to wherever a 3xx points.
    import inspect

    from terse import dropeval
    from terse.fluency import answerers

    for mod in (dropeval, answerers):
        src = inspect.getsource(mod)
        assert "urlopen(" not in src
        assert "guarded_opener(" in src


def test_invalid_header_error_does_not_echo_the_value():
    t = HttpTransport("https://example.invalid/mcp",
                      headers={"Authorization": "Bearer SECRET123\nx"}, timeout=3)
    msg = _post_one(t)
    assert msg["error"]["code"] == -32000
    assert "SECRET123" not in msg["error"]["message"]


@pytest.mark.parametrize("sid,kept", [("abc-123", True), ("bad\x01id", False),
                                      ("sp ace", False)])
def test_session_id_is_stored_only_when_visible_ascii(sid, kept):
    assert bool(transport._VALID_SESSION_ID.fullmatch(sid)) is kept


def test_router_snapshot_creates_the_terse_state_dir_owner_only(tmp_path, monkeypatch):
    import os
    import stat

    from terse import multiproxy

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    old = os.umask(0o022)
    try:
        snap = multiproxy.RouterSnapshot(multiproxy.router_snapshot_path(
            str(tmp_path / "peers.json")), "fp")
        snap.save({}, "2024-11-05")
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(tmp_path / "terse").st_mode) == 0o700


# --- G3: control codes from tool names never reach the terminal -----------------------

EVIL = "\x1b]0;PWNED\x07\x1b[2J\rfake\x9b"


def test_scrub_control_defuses_escapes_but_keeps_colour_and_layout():
    from terse.cli import scrub_control

    out = scrub_control(f"a\t{EVIL}\n\x1b[32mgreen\x1b[0m")
    assert "\x1b]" not in out and "\x07" not in out and "\r" not in out and "\x9b" not in out
    assert "\x1b[2J" not in out and "?[2J" in out  # CSI-J is not SGR: its ESC is defused
    assert "\t" in out and "\n" in out and "\x1b[32mgreen\x1b[0m" in out


def test_stats_report_command_scrubs_tool_names(tmp_path, capsys):
    from terse import cli
    from terse.stats import append_stats, build_record

    log = tmp_path / "stats.jsonl"
    append_stats(build_record("srv" + EVIL, "tool" + EVIL, '{"a": 1}', '{"a":1}',
                              passthrough=False), log)
    assert cli.main(["stats", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "tool" in out
    assert "\x1b]" not in out and "\x07" not in out and "\x9b" not in out


def test_data_commands_are_not_scrubbed(capsys):
    from terse import cli

    assert cli._cmd_proxy in cli._STDOUT_IS_DATA and cli._cmd_compress in cli._STDOUT_IS_DATA


# --- G4: SSRF guard gaps and peer isolation -------------------------------------------

@pytest.mark.parametrize("host", ["100.100.100.200", "192.0.0.192", "fd00:ec2::254",
                                  "[fd00:ec2::254]", "::ffff:100.100.100.200"])
def test_non_link_local_metadata_addresses_are_blocked(host):
    assert transport._is_metadata_host(host)


def test_ordinary_private_and_loopback_hosts_stay_allowed():
    for host in ("127.0.0.1", "10.0.0.5", "192.168.1.10", "100.64.0.1", "localhost"):
        assert not transport._is_metadata_host(host)


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        target = self.server.target  # type: ignore[attr-defined]
        if target:
            self.send_response(307)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = _reply(7, "ok").encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _ok_server(target=None):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    srv.target = target  # type: ignore[attr-defined]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_name_that_resolves_to_a_metadata_address_is_refused_at_connect(monkeypatch):
    # Stand-in for `169.254.169.254.nip.io`: "localhost" passes the name check, and the
    # peer it connects to (127.0.0.1) is what the patched guard treats as metadata.
    real = transport._is_metadata_host
    monkeypatch.setattr(transport, "_is_metadata_host",
                        lambda h: h == "127.0.0.1" or real(h))
    with _ok_server() as port:
        msg = _post_one(HttpTransport(f"http://localhost:{port}/mcp", timeout=5))
    assert "metadata" in msg["error"]["message"]


def test_peer_check_lets_ordinary_connections_through():
    with _ok_server() as port:
        msg = _post_one(HttpTransport(f"http://localhost:{port}/mcp", timeout=5))
    assert msg["result"]["content"][0]["text"] == "ok"


def test_cross_origin_307_is_refused_and_same_origin_307_followed():
    with _ok_server() as sink_port:
        with _ok_server(target=f"http://localhost:{sink_port}/mcp") as red_port:
            msg = _post_one(HttpTransport(f"http://127.0.0.1:{red_port}/mcp", timeout=5))
        assert "cross-origin 307" in msg["error"]["message"]
        with _ok_server(target=f"http://localhost:{sink_port}/mcp") as red_port:
            msg = _post_one(HttpTransport(f"http://localhost:{red_port}/mcp", timeout=5))
        assert "cross-origin" in msg["error"]["message"]  # different port: still another origin
        with _ok_server(target="/elsewhere") as red_port:
            # A relative Location is the same origin; it lands on the redirecting server,
            # which keeps redirecting — urllib's own loop limit ends it, not our refusal.
            msg = _post_one(HttpTransport(f"http://127.0.0.1:{red_port}/mcp", timeout=5))
        assert "cross-origin" not in msg["error"]["message"]


def _router():
    from threading import Lock

    from terse.multiproxy import Peer, Router

    class _T:
        def __init__(self):
            self.out = io.StringIO()

        def outbound(self):
            return self.out

    peers = [Peer("a", _T(), Interceptor(FULL)), Peer("b", _T(), Interceptor(FULL))]
    return Router(peers, io.StringIO(), Lock(), broadcast_timeout=1000)


def test_a_peer_cannot_answer_an_id_it_was_never_sent(capsys):
    from terse.proxy import SWALLOW

    router = _router()
    try:
        forged = _reply(5, "forged")
        assert router.from_peer(0)(forged) is SWALLOW
        assert "never sent" in capsys.readouterr().err
        router._routed_timers[(1, 5)] = None   # a call really routed to peer b
        assert router.from_peer(0)(forged) is SWALLOW
        assert router.from_peer(1)(_reply(5, "real")) is not SWALLOW
    finally:
        router.close_senders()


def test_peer_zero_may_answer_its_passthrough_ids_only():
    from terse.proxy import SWALLOW

    router = _router()
    try:
        router.route_client_line(json.dumps({"jsonrpc": "2.0", "id": 6,
                                             "method": "completion/complete"}))
        assert router.from_peer(1)(_reply(6, "x")) is SWALLOW
        assert router.from_peer(0)(_reply(6, "x")) is not SWALLOW
        assert router.from_peer(0)(_reply(6, "again")) is SWALLOW  # one answer per id
    finally:
        router.close_senders()


def test_a_broadcast_reply_must_come_from_the_peer_it_was_issued_to():
    router = _router()
    try:
        router._local_id_map["terse-b0-1"] = 0
        assert router._maybe_collect(0, {"id": "terse-b0-1", "result": {}}) is True
        assert 0 not in getattr(router._pending.get(0), "parts", {})
    finally:
        router.close_senders()


# --- G5: file hygiene ---------------------------------------------------------------

def test_capture_prune_never_evicts_another_tools_samples(tmp_path):
    from terse.capture import _prune_tool_samples

    for name in ("kb__00000001.json", "kb__00000002.json", "kb__zzz__00000003.json"):
        (tmp_path / name).write_text("{}")
    _prune_tool_samples(tmp_path, "kb", keep=1)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert "kb__zzz__00000003.json" in left
    assert sum(n.startswith("kb__0000000") for n in left) == 1


def test_backup_prune_spares_user_backups_and_survives_dangling_links(tmp_path):
    from terse.install_mcp import _prune_backups

    cfg = tmp_path / ".claude.json"
    cfg.write_text("{}")
    for i in range(4):
        (tmp_path / f".claude.json.bak-{1000 + i}").write_text("{}")
    (tmp_path / ".claude.json.bak-before-upgrade").write_text("{}")
    (tmp_path / ".claude.json.bak-1").symlink_to(tmp_path / "gone")
    _prune_backups(cfg, keep=2)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert ".claude.json.bak-before-upgrade" in left
    assert sum(n.startswith(".claude.json.bak-10") for n in left) <= 2
