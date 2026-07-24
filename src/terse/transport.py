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

import http.client
import ipaddress
import json
import queue
import socket
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Protocol, TextIO
from urllib.parse import urlsplit

# Upper bound on a single downstream HTTP response body held in memory. A misbehaving or
# hostile server (the URL can come from an untrusted, repo-committed .mcp.json) could
# otherwise stream an unbounded body and exhaust memory. Generous — real tool outputs are
# large — but finite; an over-limit response becomes a legible JSON-RPC error, not an OOM.
_MAX_RESPONSE_BYTES = 128 * 1024 * 1024

# The only downstream URL schemes terse will dial. `urllib.request.urlopen` also
# honors `file://`, `ftp://`, `data:` and more — so an unrestricted scheme turns a
# config-supplied "url" into a local-file read (`file:///home/you/.ssh/id_rsa`) or an
# SSRF primitive (`http://169.254.169.254/…`) whose response is fed straight into the
# model's context. A downstream target can come from an untrusted, repo-committed
# project-scoped `.mcp.json` (see install_mcp.py), so this is not purely operator input.
_ALLOWED_URL_SCHEMES = ("http", "https")

# Hosts where cleartext http is safe (it never leaves the machine) — the same set
# fluency.openai_answerer's TLS guard exempts, for the same local-gateway use case.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Header names that carry credentials. A downstream with one of these over cleartext
# http to a REMOTE host would put the secret on the wire unencrypted — refuse at
# construction, mirroring fluency's answerer guard (parity noted in the 07-14 audit).
# The same list decides what `_GuardedRedirectHandler` STRIPS on a cross-origin hop, so a
# name missing here is a credential that leaves the machine.
#
# `auth`, not `authorization`: the narrower token matched `Authorization` but sailed past
# `X-Auth` and `Authentication`, both of which were observed reaching a cross-origin sink
# intact. `session` is here because MCP's `Mcp-Session-Id` is bearer-equivalent — a server
# that pins state to it treats presenting it as proof of identity. Deliberately
# over-inclusive: a false positive costs one header on a redirect (or an http:// config
# refusal the operator resolves with https), a false negative costs the secret.
_SENSITIVE_HEADER_TOKENS = (
    "auth", "token", "secret", "key", "cookie", "session", "bearer",
    "credential", "passwd", "password", "signature",
)


def _has_sensitive_header(headers: dict[str, str]) -> bool:
    return any(t in name.lower() for name in headers for t in _SENSITIVE_HEADER_TOKENS)


def guard_cleartext_credential(base_url: str, has_credential: bool, *, what: str) -> None:
    """Raise if `base_url` would put a credential on the wire in cleartext.

    The ONE implementation of a rule three call sites need: this module's HTTP downstream
    (`--header 'Authorization=...'`), `fluency.openai_answerer`, and
    `dropeval.openai_tool_answerer` all send a bearer credential to an operator-supplied
    URL. Each grew its own copy (or, in dropeval's case, silently grew none — the audit
    parity gap), so centralize it: a new credential-bearing caller inherits the check
    instead of having to remember it. `what` names the caller in the message."""
    if not has_credential:
        return
    parts = urlsplit(base_url)
    if parts.scheme.lower() != "http":
        return
    if (parts.hostname or "").lower() in _LOOPBACK_HOSTS:
        return  # never leaves the machine — a local LiteLLM/CCR gateway is a normal setup
    raise ValueError(
        f"{what}: refusing to send a credential over cleartext http to "
        f"{parts.hostname!r} — use https, or a loopback host "
        f"({'/'.join(sorted(_LOOPBACK_HOSTS))}) for a local gateway")


# A scheme's implicit port, so `https://h` and `https://h:443` are ONE origin. Without
# this they compared unequal and a legitimate same-host redirect silently lost its
# credential, producing an unexplained 401 the operator has no way to diagnose.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port) — the identity a credential header is scoped to.

    Normalized so two spellings of the same origin compare EQUAL: the default port is
    made explicit, and an IP-literal host is canonicalized through `ipaddress` (`::1` and
    `0:0:0:0:0:0:0:1` are the same host; so are `127.0.0.1` and its decimal form)."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        port = None  # malformed port; treat as its own origin rather than crashing here
    host = _canonical_host(parts.hostname)
    return (scheme, host, port if port is not None else _DEFAULT_PORTS.get(scheme))


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-apply every construction-time downstream guard to each REDIRECT HOP.

    `HttpTransport.__init__` validates the configured URL's scheme, refuses the
    instance-metadata address, and refuses cleartext credentials — but `urlopen` follows
    up to 10 redirects, and none of those checks ran again on the hops. Two confirmed
    consequences before this handler existed:

      * A downstream 302 to `http://169.254.169.254/…` was FOLLOWED, and its body was
        enqueued straight into the model's context — the exact SSRF target the
        construction-time guard refuses outright.
      * CPython's `HTTPRedirectHandler.redirect_request` copies every request header
        except content-length/content-type onto the new request, so an
        `Authorization: Bearer …` set via `--header` was re-sent VERBATIM to whatever
        host the redirect named, cleartext included.

    `urllib` also permits `ftp://` on a redirect (`http_error_302` allows
    http/https/ftp), which sidesteps `_ALLOWED_URL_SCHEMES` — re-checking the scheme
    per hop closes that too. Refusals are raised as `HTTPError`, which
    `HttpTransport._post` already converts into a legible JSON-RPC error, so a blocked
    redirect degrades into the same fail-open path as any other downstream failure."""

    def __init__(self, origin_url: str):
        self._origin = _origin(origin_url)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        scheme = parts.scheme.lower()
        if scheme and scheme not in _ALLOWED_URL_SCHEMES:
            raise urllib.error.HTTPError(
                newurl, code, f"terse: refusing redirect to disallowed scheme "
                f"{scheme!r} (only {'/'.join(_ALLOWED_URL_SCHEMES)})", headers, fp)
        if _is_metadata_host(parts.hostname):
            raise urllib.error.HTTPError(
                newurl, code, f"terse: refusing redirect to link-local/metadata address "
                f"{parts.hostname!r} — the cloud instance-metadata SSRF target", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        # Cross-origin hop: the credential was scoped to the configured downstream, not to
        # wherever it points us. Strip rather than refuse — a plain redirect to a CDN or a
        # renamed path is legitimate and should still work, just without the secret.
        if _origin(new.full_url) != self._origin:
            for name in [h for h in new.headers
                         if any(t in h.lower() for t in _SENSITIVE_HEADER_TOKENS)]:
                del new.headers[name]
        return new


def _as_ip(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """`hostname` as an IP address if it is a literal in ANY encoding the resolver
    accepts, else None.

    `ipaddress.ip_address` alone is not enough, and the gap is a live SSRF bypass rather
    than a nicety: it parses only dotted-quad, while glibc's `getaddrinfo` — the thing
    that actually dials — also accepts decimal (`2852039166`), octal
    (`0251.0376.0251.0376`), hex (`0xa9.0xfe.0xa9.0xfe`) and short (`169.254.43518`)
    forms. Every one of those resolves to 169.254.169.254 and every one of them was
    classified "not an IP, therefore allowed" by the guard whose whole job is refusing
    that address. `socket.inet_aton` accepts exactly the same set, so canonicalizing
    through it closes the encodings without resolving DNS (rebinding stays out of scope,
    as documented on `_is_metadata_host`).

    The trailing dot — `169.254.169.254.` — is the fully-qualified form; the resolver
    accepts it and `inet_aton` does not, so strip it first."""
    host = hostname.strip("[]").rstrip(".")
    if not host:
        return None
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None  # a real DNS name, not classifiable without resolving


def _canonical_host(hostname: str | None) -> str:
    """A host's one canonical spelling, for origin comparison. IP literals go through
    `_as_ip` so alternate encodings of one address collapse to a single string."""
    if not hostname:
        return ""
    ip = _as_ip(hostname)
    return str(ip) if ip is not None else hostname.lower().rstrip(".")


def _is_metadata_host(hostname: str | None) -> bool:
    """True for a link-local target — the cloud instance-metadata SSRF endpoint
    (169.254.169.254 across AWS/Azure/GCP/DO, fe80::/10, and GCP's
    metadata.google.internal alias). Never a legitimate MCP server, so refusing it costs
    nothing while closing the highest-value SSRF target. Loopback and ordinary private/LAN
    addresses stay allowed — local and homelab MCP servers are a first-class use case, and
    a DNS name isn't resolved here (rebinding is a deeper problem out of this guard's scope)."""
    if not hostname:
        return False
    if hostname.lower().rstrip(".") == "metadata.google.internal":
        return True
    ip = _as_ip(hostname)   # every encoding the resolver accepts, not just dotted-quad
    return ip is not None and ip.is_link_local


class Transport(Protocol):
    """One downstream MCP peer, abstracted over its wire transport.

    `inbound()` yields server->client JSON-RPC lines (no trailing newline) —
    usable directly as `proxy.pump()`'s `src` (`pump` does `for raw in src:`
    then `raw.rstrip("\\n")`, so a bare `str` iterator or a line-iterable file
    object both work). `outbound()` returns an object with `.write(str)` +
    `.flush()` for client->server lines — usable directly as `pump()`'s `dst`.
    `close()` releases whatever resource backs the peer (a child process, an
    HTTP session) — idempotent, safe to call more than once (`run_proxy` calls
    it from more than one place as a defense-in-depth cleanup).

    `half_close()` and `wait()` exist so callers (`run_proxy`/`multiproxy.py`)
    never need an `isinstance` check against a concrete subtype to tear a peer
    down correctly — every transport-specific teardown detail (does closing
    stdin let a child flush and exit on its own? is there even a process to
    wait for an exit code from?) lives behind these two methods instead of
    leaking into every call site as a runtime type check."""

    def inbound(self) -> Iterator[str]: ...

    def outbound(self) -> Any: ...

    def close(self) -> None: ...

    def half_close(self) -> None: ...

    def wait(self) -> int: ...


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

    def half_close(self) -> None:
        """Close stdin only, signaling EOF so the child can flush any remaining
        reply and exit on its own — `wait()`/`close()` (SIGTERM/SIGKILL escalation
        via `_terminate_child`) still run afterward as the real/last-resort reaper."""
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    def wait(self) -> int:
        return self.proc.wait()


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

    def __init__(self, transport: HttpTransport):
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


def _parse_request_id(line: str) -> tuple[bool, Any]:
    """Parse `line` (an outgoing JSON-RPC request) once: whether it carries an `"id"`
    key at all (a notification has none and per JSON-RPC must never get a reply) and
    that id's value. Shared by every HttpTransport error/reply path so each parses the
    outgoing line only once. If `line` isn't valid JSON (shouldn't happen — it came
    from the client through `note_request`/`pump` unchanged), fails toward "has an id"
    so the caller reports the error rather than silently dropping it."""
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return True, None
    if not isinstance(parsed, dict):
        return True, None
    return "id" in parsed, parsed.get("id")


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

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: int = 60):
        scheme = urlsplit(url).scheme.lower()
        if scheme not in _ALLOWED_URL_SCHEMES:
            # Raised at construction (before any I/O), so build_transport's callers
            # surface it as a clean config error (proxy exit 2 / multiproxy bad-peer),
            # not an uncaught traceback or, worse, a silent file read.
            raise ValueError(
                f"terse: downstream URL scheme {scheme or '(none)'!r} is not allowed — "
                f"only {'/'.join(_ALLOWED_URL_SCHEMES)} (urllib would otherwise honor "
                "file://, ftp://, data:, turning a config-supplied URL into a local-file "
                "read or SSRF vector)")
        self.url = url
        self.headers = dict(headers or {})
        split = urlsplit(url)
        if _is_metadata_host(split.hostname):
            # Same construction-time, before-any-I/O contract as the checks around it.
            raise ValueError(
                f"terse: refusing to connect to link-local/metadata address "
                f"{split.hostname!r} — this is the cloud instance-metadata SSRF target, "
                "never a legitimate MCP endpoint")
        # Same construction-time, before-any-I/O contract as the scheme check above.
        guard_cleartext_credential(url, _has_sensitive_header(self.headers),
                                   what="terse: downstream")
        self.timeout = timeout
        # A PRIVATE opener, not the module-global `urlopen`: it carries the redirect
        # handler that re-applies these same guards to every hop (see
        # `_GuardedRedirectHandler`). Built per-transport because the handler is bound to
        # THIS downstream's origin, which is what decides a cross-origin credential strip.
        self._own_origin = _origin(url)
        self._opener = urllib.request.build_opener(_GuardedRedirectHandler(url))
        self._q: queue.Queue[Any] = queue.Queue()
        # MCP Streamable HTTP session affinity: some servers pin a client to
        # server-side state via this header, set on a prior response. Captured
        # opportunistically and echoed back on every subsequent POST — never
        # required, since plenty of servers don't use it at all.
        self.session: str | None = None

    def inbound(self) -> Iterator[str]:
        return iter(self._q.get, _SENTINEL)

    def outbound(self) -> _HttpSendWriter:
        return _HttpSendWriter(self)

    def close(self) -> None:
        self._q.put(_SENTINEL)

    def half_close(self) -> None:
        """No persistent connection to half-close — client stdin EOF IS the
        transport's whole end-of-life signal, so closing outright is correct."""
        self.close()

    def wait(self) -> int:
        """No child process to wait for an exit code from — always 0. The real
        completion signal for HTTP is the caller joining its inbound pump thread,
        which only finishes once close() has drained the sentinel (see
        run_proxy/run_multi_proxy)."""
        return 0

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
            with self._opener.open(req, timeout=self.timeout) as resp:  # noqa: S310
                # Only the CONFIGURED downstream may set our session id. After a redirect
                # `resp` is the final hop's response, so without this check a hostile
                # server could 302 to a host of its choosing and have that host's
                # `Mcp-Session-Id` stored — which terse then presented on every subsequent
                # POST to the legitimate downstream. That is session fixation: an attacker
                # picks the session the real server sees. (`geturl()` is the post-redirect
                # url; on a non-redirected response it equals `self.url`.)
                if _origin(resp.geturl()) == self._own_origin:
                    sid = resp.headers.get("Mcp-Session-Id")
                    if sid:
                        self.session = sid
                ctype = resp.headers.get("Content-Type", "")
                # Bounded read: one byte past the cap tells us it overflowed without
                # trusting a (possibly absent/lying) Content-Length. An over-limit body
                # is refused with a legible error rather than being buffered to OOM.
                body = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(body) > _MAX_RESPONSE_BYTES:
                    self._maybe_enqueue_error(
                        line, f"terse: downstream response exceeded "
                        f"{_MAX_RESPONSE_BYTES} bytes; refused")
                    return
        except urllib.error.HTTPError as exc:
            # A 4xx/5xx status. Per MCP Streamable-HTTP the server MAY still have sent
            # a legitimate JSON-RPC error object in the body (e.g. "missing
            # Authorization header") — read it and forward it verbatim when it looks
            # like one, instead of discarding the real detail behind a generic
            # wrapper message the client can't act on.
            try:
                raw_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — reading the error body is best-effort
                raw_body = ""
            if raw_body.strip():
                try:
                    parsed = json.loads(raw_body)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict) and "jsonrpc" in parsed:
                    # Forward it verbatim ONLY if the outgoing line actually expected a
                    # reply — a notification (no "id") must never get one, the same
                    # invariant _maybe_enqueue_error enforces for a non-JSON-RPC-shaped
                    # error body just below.
                    has_id, _ = _parse_request_id(line)
                    if has_id:
                        self._q.put(raw_body)
                    return
            detail = raw_body.strip() or str(exc)
            self._maybe_enqueue_error(
                line, f"terse: downstream HTTP {exc.code} {exc.reason}: {detail}")
            return
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            # OSError covers a bare connection-refused/timeout that never got far enough
            # to become a URLError. The other two cover what `urlsplit` accepts but
            # `http.client` rejects at request time, neither of which is an OSError:
            # `ValueError` for an invalid header VALUE, and `HTTPException` for
            # `InvalidURL` — a downstream url of `http://exa mple.com/mcp`, reachable from
            # a repo-committed .mcp.json. (`InvalidURL` derives from `HTTPException`, NOT
            # from `ValueError`; assuming otherwise is what the regression test caught.)
            # Without these the exception escaped `_post` — whose contract is "never
            # raises" — through
            # `_HttpSendWriter.flush()` into `pump()`, killing the client->server thread
            # and making the proxy exit 0 as if it had shut down cleanly, with the
            # client's call unanswered. Either way now: the in-flight request gets a
            # legible error instead of the client hanging forever on a reply that's
            # never coming.
            self._maybe_enqueue_error(line, f"terse: downstream HTTP request failed: {exc}")
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

    def _maybe_enqueue_error(self, line: str, message: str) -> None:
        """Enqueue a synthesized JSON-RPC error for `line`'s in-flight request —
        unless `line` is a notification (has no `"id"` key at all), which per
        JSON-RPC never gets a reply. Enqueuing one anyway would hand the client an
        unsolicited `id: null` message matching no request it sent, which a strict
        client could reject or misinterpret as an unmatched response."""
        has_id, mid = _parse_request_id(line)
        if not has_id:
            return
        self._q.put(self._error_reply(mid, message))

    @staticmethod
    def _error_reply(mid: Any, message: str) -> str:
        """A JSON-RPC 2.0 error object for `mid` (the outgoing line's own request id),
        so the client's matching in-flight call gets a legible failure instead of
        silence. Callers already resolved `mid` via `_parse_request_id`."""
        return json.dumps(
            {"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": message}},
            separators=(",", ":"), ensure_ascii=False,
        )


def build_transport(target: list[str], *, headers: dict[str, str] | None = None) -> Transport:
    """Build the right `Transport` for a proxy `cmd`/downstream target.

    A single element containing `"://"` is a URL -> `HttpTransport`; anything
    else is a stdio launch command -> `StdioTransport`. Mirrors
    `proxy.stdio_transport_error`'s own URL detection so the two can never
    disagree about what counts as a URL downstream."""
    if len(target) == 1 and "://" in target[0]:
        return HttpTransport(target[0], headers=headers)
    return StdioTransport(target)
