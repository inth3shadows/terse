"""Transport abstraction: one downstream MCP peer, over stdio or HTTP (#5).

`proxy.pump()` only needs a line-iterable `src` (server->client JSON-RPC lines,
no trailing newline) and a `.write(str)`/`.flush()` `dst` (client->server). Before
this module, `run_proxy` built those two things directly from a
`subprocess.Popen`'s stdout/stdin. This module pulls that out behind a small
`Transport` protocol so `pump` and `Interceptor` (proxy.py) never need to know
HOW the downstream is reached — a stdio subprocess (unchanged, pre-#5 behavior)
or an MCP Streamable-HTTP endpoint (new). `Interceptor.answer_retrieve` never
touches a `Transport` at all — it writes straight back to the client stream —
which is the whole point: drop-to-retrieve needed zero HTTP-specific code to
work over this new downstream (verified by test_transport.py).
"""

from __future__ import annotations

import json
import queue
import subprocess
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional, Protocol, TextIO


class Transport(Protocol):
    """One downstream MCP peer, abstracted over its wire transport.

    `inbound()` yields server->client JSON-RPC lines (no trailing newline) —
    usable directly as `proxy.pump()`'s `src` (`pump` does `for raw in src:`
    then `raw.rstrip("\\n")`, so a bare `str` iterator or a line-iterable file
    object both work). `outbound()` returns an object with `.write(str)` +
    `.flush()` for client->server lines — usable directly as `pump()`'s `dst`.
    `close()` releases whatever resource backs the peer (a child process, an
    HTTP session) — idempotent, safe to call more than once (`run_proxy` calls
    it from more than one place as a defense-in-depth cleanup)."""

    def inbound(self) -> Iterator[str]: ...

    def outbound(self) -> Any: ...

    def close(self) -> None: ...


class StdioTransport:
    """A downstream MCP server launched as a local subprocess, speaking
    newline-delimited JSON-RPC over its stdin/stdout — today's (only, pre-#5)
    proxy behavior, extracted unchanged out of `run_proxy` so it can sit behind
    `Transport` next to `HttpTransport`.

    Raises `OSError` from `__init__` on an unlaunchable command (mistyped path,
    non-executable, ...) — `run_proxy` catches that exactly as it always has,
    to report a config error (exit 127) instead of an uncaught traceback (#19)."""

    def __init__(self, cmd: list[str]):
        self.proc = subprocess.Popen(  # noqa: S603 — cmd is operator-supplied, by design
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            encoding="utf-8",
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None

    def inbound(self) -> TextIO:
        return self.proc.stdout  # type: ignore[return-value]  # already line-iterable

    def outbound(self) -> TextIO:
        return self.proc.stdin  # type: ignore[return-value]  # has .write()/.flush()

    def close(self) -> None:
        # Imported lazily: proxy.py imports `build_transport` from this module
        # at module load time, so a top-level `from .proxy import ...` here
        # would fail on the circular partial-import (this module executes
        # before proxy.py has finished defining `_terminate_child`). By the
        # time close() actually runs, both modules are fully loaded.
        from .proxy import _terminate_child

        _terminate_child(self.proc)


# Sentinel that ends `HttpTransport.inbound()`'s queue-backed iterator. A plain
# `None` would collide with a legitimate (if odd) enqueued value, so use a
# private object identity instead — mirrors `proxy.SWALLOW`'s reasoning.
_SENTINEL: Any = object()


class _HttpSendWriter:
    """The `.write(str)`/`.flush()` adapter `pump()` writes client->server
    lines through, for an `HttpTransport`. `pump()` always calls
    `dst.write(line + "\\n")` once immediately followed by `dst.flush()` for
    each JSON-RPC line, so in practice this buffers exactly one line per
    flush — but `flush()` splits on any embedded newlines defensively, so a
    differently-behaved caller still gets one POST per JSON-RPC line rather
    than one POST of concatenated lines."""

    def __init__(self, transport: "HttpTransport"):
        self._transport = transport
        self._buf = ""

    def write(self, s: str) -> None:
        self._buf += s

    def flush(self) -> None:
        if not self._buf:
            return
        buf, self._buf = self._buf, ""
        for line in buf.split("\n"):
            line = line.strip()
            if line:
                self._transport._post(line)


def _iter_sse(body: str) -> Iterator[str]:
    """Line-based Server-Sent-Events parser over an already-read response body.

    Accumulates `data:` lines and dispatches the joined payload at each event
    boundary (a blank line, per the SSE spec); `event:`/`id:`/`retry:` fields
    and `:`-prefixed comment lines are ignored — the proxy only cares about the
    JSON-RPC payload each event carries. A single POST can legitimately carry
    MULTIPLE JSON-RPC messages in one SSE stream (e.g. a tool-call response
    plus a notification), so this yields one string per event, in order."""
    data_lines: list[str] = []
    for raw_line in body.split("\n"):
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue  # comment line
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
        # event:/id:/retry: (and anything unrecognized) — ignored; not needed
        # to reconstruct the JSON-RPC payload.
    if data_lines:  # a trailing event with no final blank-line terminator
        yield "\n".join(data_lines)


class HttpTransport:
    """A downstream MCP server reached over MCP's Streamable HTTP transport
    (#5): the client POSTs one JSON-RPC line per call; the server replies with
    either a single `application/json` body or a `text/event-stream` SSE
    stream carrying one or more JSON-RPC messages. Built on stdlib
    `urllib.request` only — this repo has a hard zero-new-deps policy for
    exactly this kind of thing (mirrors `fluency.openai_answerer`'s pattern).

    v1 scope (proportionate to the real use case — front ONE remote server):
      - Synchronous POST-then-drain on the send path: the reply(ies) to line N
        are enqueued before the next client line is even sent (`_post` runs
        inline inside `_HttpSendWriter.flush()`, on the same thread `pump()`
        drives). Correct for MCP request/response and tool calls; gives up
        cross-request pipelining. Revisit only if a real workload needs
        concurrent in-flight requests.
      - No standalone GET SSE listener for unsolicited server->client
        notifications (progress, etc.) outside a request/response — the
        proxy's tool-call flows don't need one. Documented follow-up.

    Fail-open (matches `Interceptor`'s whole design philosophy — see its
    docstring in proxy.py): a network error, timeout, or bad response never
    hangs or crashes the proxy. It synthesizes a legible JSON-RPC error for the
    in-flight request's id and enqueues THAT instead, so the client always gets
    a reply rather than silence."""

    def __init__(self, url: str, headers: Optional[dict[str, str]] = None, timeout: int = 60):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._q: "queue.Queue[Any]" = queue.Queue()
        # MCP Streamable HTTP session affinity: some servers pin a client to
        # server-side state via this header, set on a prior response. Captured
        # opportunistically and echoed back on every subsequent POST — never
        # required, since plenty of servers don't use it at all.
        self.session: Optional[str] = None

    def inbound(self) -> Iterator[str]:
        return iter(self._q.get, _SENTINEL)

    def outbound(self) -> _HttpSendWriter:
        return _HttpSendWriter(self)

    def close(self) -> None:
        self._q.put(_SENTINEL)

    def _post(self, line: str) -> None:
        """POST one JSON-RPC line downstream and enqueue whatever comes back
        onto `self._q` for `inbound()` to yield. Never raises: every failure
        mode (network error, timeout, bad status) is converted to a
        synthesized JSON-RPC error enqueued in place of a real reply, so the
        client-facing pump never blocks waiting on a message that will never
        arrive (fail-open, #5)."""
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session:
            req_headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=line.encode("utf-8"),
                                     headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session = sid
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            # URLError covers HTTPError (a 4xx/5xx status) too, since it's a
            # subclass; OSError covers a bare connection-refused/timeout that
            # never got far enough to become a URLError. Either way: the
            # in-flight request gets a legible error instead of the client
            # hanging forever on a reply that's never coming.
            self._q.put(self._error_reply(line, f"terse: downstream HTTP request failed: {exc}"))
            return
        if "text/event-stream" in ctype:
            for msg in _iter_sse(body.decode("utf-8", errors="replace")):
                self._q.put(msg)
        else:
            text = body.decode("utf-8", errors="replace")
            if text.strip():
                self._q.put(text)
            # else: a 202 Accepted / empty body is valid Streamable-HTTP and
            # means nothing to enqueue (e.g. the reply to a notification).

    def _error_reply(self, line: str, message: str) -> str:
        """A JSON-RPC 2.0 error object for the request id parsed out of the
        outgoing `line`, so the client's matching in-flight call gets a
        legible failure instead of silence. Best-effort id parse — if `line`
        isn't valid JSON (shouldn't happen; it came from the client through
        `note_request`/`pump` unchanged) or carries no id (a notification),
        reply with `id: null` rather than dropping the error, so the failure
        is still visible on the wire."""
        mid = None
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                mid = parsed.get("id")
        except (json.JSONDecodeError, ValueError):
            pass
        return json.dumps(
            {"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": message}},
            separators=(",", ":"), ensure_ascii=False,
        )


def build_transport(target: list[str], *, headers: Optional[dict[str, str]] = None) -> Transport:
    """Build the right `Transport` for a proxy `cmd`/downstream target.

    A single element containing `"://"` is a URL -> `HttpTransport`; anything
    else is a stdio launch command -> `StdioTransport`. Mirrors
    `proxy.stdio_transport_error`'s own URL detection so the two can never
    disagree about what counts as a URL downstream."""
    if len(target) == 1 and "://" in target[0]:
        return HttpTransport(target[0], headers=headers)
    return StdioTransport(target)
