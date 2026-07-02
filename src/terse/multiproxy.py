"""Multi-downstream fan-out proxy (#5 Half B): ONE `terse proxy --config` process
fronting N downstream MCP peers (any mix of stdio and HTTP, via `transport.py`),
merging their `tools/list` under name prefixes, routing `tools/call` by prefix, and
sharing the drop-to-retrieve store across all of them.

This is explicitly an ergonomics feature (single policy/primer/process) — MCP clients
already multiplex servers natively — so the v1 scope below is deliberately
proportionate. Documented limitations are fine; silent gaps are not:
  - A server-initiated request FROM a downstream that expects a client reply
    (sampling/createMessage, roots) is forwarded to the client with its id rewritten
    to a router-namespaced one (`_rewrite_server_request`); the client's reply is
    routed back to the originating peer with the original id restored
    (`route_client_line`'s method-is-None branch) — an id whose peer isn't known
    (unrecognized, already answered, or evicted past `_SERVER_REQ_MAX`) is dropped
    rather than guessed at.
  - Any client method other than `initialize`/`tools/list` (broadcast+merge) or
    `tools/call` (routed by prefix) — e.g. `resources/list`, `prompts/list`, `ping` —
    is forwarded to peer 0 ONLY, not broadcast/merged, and this is logged to stderr
    unconditionally (not gated behind --debug — silently dropping N-1 peers' data
    from the reply is exactly the kind of gap this file promises not to hide).
    Building full N-way broadcast/merge logic for every possible MCP method is
    disproportionate for a feature the issue itself calls "ergonomics, modest value".
  - A broadcast (`initialize`/`tools/list`) blocks on every peer up to
    `BROADCAST_TIMEOUT` seconds; a peer that never answers can't wedge it — the reply
    goes out with whatever DID arrive, and the missing peer(s) are logged to stderr.

Reused, unchanged: `Interceptor` (per peer, sharing one drop store via its optional
`store`/`store_lock` kwargs — see proxy.py), `pump()` (one reader thread PER PEER for
the server->client direction — that direction genuinely is 1:1, so pump is the right
tool there), `SWALLOW`, `TERSE_PRIMER`, `RETRIEVE_TOOL_DEF`, `transport.build_transport`.
New here: the client->server fan-out loop (NOT pump — fan-out from one source to N
destinations is a genuinely different shape), tool-name prefixing/routing, and
broadcast id-remapping/merge.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Optional, TextIO

from . import lossy as lossy_mod
from . import policy as policy_mod
from .proxy import (
    RETRIEVE_TOOL_DEF,
    SWALLOW,
    TERSE_PRIMER,
    Interceptor,
    _build_capture_and_audit,
    _ignore_sigterm,
    _install_sigterm_to_exit,
    _restore_sigterm,
    pump,
)
from .transport import Transport, build_transport

# Tool-name prefix separator. Double underscore, not a dot: MCP/OpenAI function names
# are commonly constrained to `^[a-zA-Z0-9_-]+$`, and a real downstream tool name can
# already contain a dot (e.g. `gh.api.items`) — `.` as a separator would be ambiguous
# to split back apart, `__` isn't.
PREFIX_SEP = "__"

# How long a broadcast (initialize/tools/list) waits for every peer before merging
# whatever arrived and moving on. Module-level so a test can shrink it (a dead-peer
# test that waited the real default would need 30+ seconds of real time for no reason).
BROADCAST_TIMEOUT = 30.0

# Client methods that fan out to every peer and get merged into ONE reply. Anything
# else that still carries an id (not tools/call, not one of these) falls through to
# the documented v1 "forward to peer 0 only" path.
_BROADCAST_METHODS = ("initialize", "tools/list")

# Bound on `Router._local_id_map` (broadcast-local id -> broadcast seq). Entries are
# deliberately NOT popped as soon as a broadcast finishes (a late, post-finish reply
# must still resolve to "already done" and be swallowed rather than leak to the client
# — see `_maybe_collect`/`_finish_broadcast`), so growth is bounded by eviction instead,
# the same LRU-cap shape `Interceptor.PENDING_MAX`/`DROPPED_MAX` already use elsewhere
# in this codebase. Sized generously (a broadcast is a rare event, not a hot path).
_LOCAL_ID_MAP_MAX = 4096

# Bound on `Router._server_requests` (router-local id -> (peer_idx, original id)),
# mirroring `_LOCAL_ID_MAP_MAX`'s reasoning: a peer that keeps issuing server-initiated
# requests (sampling/createMessage, roots) the client never answers can't grow this
# unboundedly — an evicted entry just means that particular late reply is dropped.
_SERVER_REQ_MAX = 1024

# Bound on `Router._routed_timed_out`, same eviction reasoning as _LOCAL_ID_MAP_MAX:
# once a routed tools/call has timed out and been answered, its id is remembered only
# long enough to swallow the peer's late real reply instead of double-answering the
# client — an evicted entry just means an extremely late reply falls through to the
# normal path instead (harmless: the client already has its timeout error).
_ROUTED_TIMED_OUT_MAX = 4096


@dataclass
class Peer:
    """One downstream MCP peer, wired into the shared drop store. `inter` is this
    peer's own `Interceptor` — its diff/compress state is per-peer (a `tools/call`
    result is only ever compressed against ITS OWN prior result for the same tool),
    but its drop-to-retrieve store is the one shared across every `Peer` (built once
    in `run_multi_proxy` and injected into each `Interceptor` via `store`/`store_lock`)."""
    name: str
    transport: Transport
    inter: Interceptor


@dataclass
class DownstreamSpec:
    """One parsed `downstreams[]` entry from a `--config` file, before a `Peer` (which
    needs a live `Transport` + `Interceptor`) is built from it."""
    name: str
    target: list[str]              # a stdio command, or a single-element [url]
    headers: dict[str, str]
    policy_path: Optional[str]     # resolved relative to the config file; None = use the default


def load_multi_config(path: str) -> list[DownstreamSpec]:
    """Parse + validate a `--config` JSON file:
    `{"downstreams": [{"name": "gh", "policy": "gh.json", "command": [...]},
                       {"name": "kb", "url": "https://...", "headers": {...}}]}`.
    `name` must be unique — it becomes the tool-prefix/routing key, so a duplicate
    would make `tools/call` routing ambiguous. Each entry needs exactly one of
    `command` (a stdio launch command) or `url` (a single HTTP/SSE endpoint). A
    relative `policy` path is resolved against the CONFIG file's directory (not the
    process cwd) so a policy bundle can ship alongside its config regardless of where
    `terse proxy --config` is invoked from. Raises ValueError with a clear message on
    any malformed entry; raises OSError/json.JSONDecodeError (a ValueError subclass)
    on a missing/unparseable file — the caller reports both as a clean config error."""
    config_path = Path(path)
    doc = json.loads(config_path.read_text(encoding="utf-8"))
    downstreams = doc.get("downstreams")
    if not isinstance(downstreams, list) or not downstreams:
        raise ValueError(f"{path}: 'downstreams' must be a non-empty list")

    specs: list[DownstreamSpec] = []
    seen: set[str] = set()
    for i, d in enumerate(downstreams):
        if not isinstance(d, dict):
            raise ValueError(f"{path}: downstreams[{i}] must be an object")
        name = d.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: downstreams[{i}] is missing a non-empty 'name'")
        if name in seen:
            raise ValueError(f"{path}: duplicate downstream name {name!r} — 'name' must "
                             "be unique (it becomes the tool-prefix/routing key)")
        if PREFIX_SEP in name:
            # `_route_call` splits a routed tool name on the FIRST "__", so a name that
            # itself contains "__" is ambiguous against another peer whose name is its
            # prefix (e.g. "gh" and "gh__api" would both match a "gh__api__foo" call —
            # silently to the wrong one). Reject at config-load time instead.
            raise ValueError(f"{path}: downstream name {name!r} must not contain "
                             f"{PREFIX_SEP!r} — that's the tool-prefix separator")
        seen.add(name)

        url, command = d.get("url"), d.get("command")
        if url and command:
            raise ValueError(f"{path}: downstream {name!r} has both 'url' and 'command' "
                             "— give exactly one")
        if url:
            if not isinstance(url, str):
                raise ValueError(f"{path}: downstream {name!r}: 'url' must be a string")
            target = [url]
        elif command:
            if not isinstance(command, list) or not all(isinstance(c, str) for c in command):
                raise ValueError(f"{path}: downstream {name!r}: 'command' must be a list "
                                 "of strings")
            target = list(command)
        else:
            raise ValueError(f"{path}: downstream {name!r} needs a 'command' or a 'url'")

        headers = d.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError(f"{path}: downstream {name!r}: 'headers' must be an object")

        policy_path = d.get("policy")
        if policy_path is not None:
            if not isinstance(policy_path, str):
                raise ValueError(f"{path}: downstream {name!r}: 'policy' must be a string")
            p = Path(policy_path)
            policy_path = str(p if p.is_absolute() else config_path.parent / p)

        specs.append(DownstreamSpec(name=name, target=target,
                                    headers={str(k): str(v) for k, v in headers.items()},
                                    policy_path=policy_path))
    return specs


@dataclass
class _PendingBroadcast:
    """Bookkeeping for one in-flight broadcast (`initialize` or `tools/list`), keyed by
    a router-local, globally-unique SEQUENCE NUMBER — not the client's request id.
    A client id is only unique among an individual client's in-flight requests; keying
    by seq instead means two broadcasts can never collide even if a client illegally
    reuses an id while the first is still in flight (JSON-RPC forbids that, but a
    misbehaving client is exactly the case worth being safe against — see `_broadcast`).
    `client_id` is kept so the eventual merged reply can be written with the id the
    client actually used. `parts` collects each peer's parsed reply as it arrives (peer
    index -> the full JSON-RPC message, so a peer error is distinguishable from a peer
    that hasn't answered yet); `remaining` shrinks to empty as replies land, or the
    broadcast is force-completed by `Router._timeout_broadcast` once `BROADCAST_TIMEOUT`
    elapses. Peer-local ids aren't stored here — they're deterministic from
    `(seq, peer_idx)` (`f"terse-b{seq}-{i}"`), recomputed wherever needed instead of
    tracked in a second structure that could drift out of sync with `_local_id_map`."""
    kind: str
    client_id: Any
    remaining: set[int]
    parts: dict[int, dict] = field(default_factory=dict)
    timer: Optional[threading.Timer] = None
    done: bool = False


class _PeerSender:
    """Per-peer background writer (#5): `send()` enqueues a line and returns
    immediately; a single worker thread per peer drains its queue and does the actual
    write+flush (which, for an `HttpTransport` peer, is a BLOCKING network round-trip —
    see `transport.HttpTransport._post`). One worker per peer preserves that peer's own
    line ORDER (FIFO queue, single consumer) while letting different peers' sends
    proceed independently.

    Without this, the client->server fan-out ran on ONE thread (`Router.route_client_line`,
    by design — see the module docstring), so writing to a slow HTTP peer blocked that
    same thread from even STARTING to write to any other peer — serializing all peer
    traffic behind whichever peer was currently slowest, and for a broadcast, consuming
    the shared `BROADCAST_TIMEOUT` budget before a healthy peer further down the peer
    list was ever contacted."""

    _STOP: Any = object()

    def __init__(self, transport: Transport, debug: bool = False):
        self._transport = transport
        self._debug = debug
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, line: str) -> None:
        self._q.put(line)

    def _run(self) -> None:
        while True:
            line = self._q.get()
            if line is self._STOP:
                return
            try:
                w = self._transport.outbound()
                w.write(line + "\n")
                w.flush()
            except Exception as exc:  # noqa: BLE001 — one peer's send failure must
                                       # never crash its worker thread or any other peer
                if self._debug:
                    sys.stderr.write(f"[terse-multiproxy] send failed: {exc}\n")

    def close(self) -> None:
        self._q.put(self._STOP)
        self._thread.join(timeout=2.0)


class Router:
    """Owns id-remapping, tool-name prefixing, broadcast/merge, and routing for a
    multi-peer session. One `Router` per `run_multi_proxy` call; long-lived for the
    session's duration.

    Two directions, deliberately asymmetric (see module docstring): `route_client_line`
    is called once per client->server line, from a dedicated loop (fan-out, not pump);
    `from_peer(i)` returns a `pump()`-compatible transform for peer i's server->client
    reader thread (that direction genuinely is 1:1, so pump is the right tool)."""

    def __init__(self, peers: list[Peer], out: TextIO, out_lock: Lock, *,
                 debug: bool = False, broadcast_timeout: float = BROADCAST_TIMEOUT):
        self.peers = peers
        self.by_name = {p.name: i for i, p in enumerate(peers)}
        self.out = out
        self.out_lock = out_lock
        self.debug = debug
        self.broadcast_timeout = broadcast_timeout
        # True if ANY peer's policy enables drop-to-retrieve — gates whether the merged
        # tools/list advertises the single synthetic terse.retrieve tool at all, mirroring
        # single-peer Interceptor.policy.has_drop() (#10).
        self.has_drop = any(p.inter.policy.has_drop() for p in peers)
        # One background sender per peer (index-aligned with `peers`) so a slow HTTP
        # peer's blocking send can't stall routing to any other peer — see _PeerSender.
        self._senders = [_PeerSender(p.transport, debug=debug) for p in peers]

        self._pending_lock = Lock()
        # keyed by broadcast SEQUENCE NUMBER, not client id — see _PendingBroadcast.
        self._pending: dict[int, _PendingBroadcast] = {}
        self._broadcast_seq = 0
        # client id -> the seq of that client id's currently-active broadcast, if any.
        # Lets a reused client id (a protocol violation, but one worth failing safe
        # against) cancel/abandon the PRIOR broadcast's seq without touching its still-
        # live _local_id_map entries — those remain valid pointers to the old (now
        # abandoned) seq, so a late reply for them still resolves through _pending.get()
        # to "not found" and is swallowed, never misattributed to the new broadcast.
        self._active_seq: dict[Any, int] = {}
        # peer-local broadcast id -> the seq it belongs to. Doubles as the "is this id
        # one of ours" check: from_peer() looks here first, so an unknown id (a normal
        # routed tools/call response) is a clean miss, not a string-parse guess. NOT
        # popped when a broadcast finishes — a reply arriving after the merge already
        # went out must still resolve here and be swallowed (see _maybe_collect), not
        # fall through to a peer's Interceptor as an unsolicited message. Bounded by
        # _LOCAL_ID_MAP_MAX eviction instead of eager per-broadcast cleanup.
        self._local_id_map: "OrderedDict[Any, int]" = OrderedDict()
        # router-local id -> (peer_idx, original id), for a server-initiated request
        # forwarded to the client (see _rewrite_server_request) so the client's eventual
        # reply can be routed back to the peer that actually asked, with its original id
        # restored — instead of the prior v1 gap (misdelivered to peer 0). Bounded by
        # _SERVER_REQ_MAX eviction.
        self._server_requests: "OrderedDict[str, tuple[int, Any]]" = OrderedDict()
        self._server_req_seq = 0
        # client id -> Timer, for a routed (single-peer) tools/call awaiting reply —
        # mirrors _broadcast's BROADCAST_TIMEOUT guarantee ("a dead peer can't wedge
        # it") for the routed path, which previously had no bound at all: a hung/dead
        # stdio peer left the client's request unanswered forever. Whichever side wins
        # the race to pop an entry (the real reply in from_peer, or _timeout_routed_call
        # firing) is the one that answers the client; the loser does nothing.
        self._routed_timers: dict[Any, threading.Timer] = {}
        # ids whose routed call already got a synthesized timeout reply — the peer's
        # eventual real (late) reply must be swallowed here, not double-delivered.
        self._routed_timed_out: "OrderedDict[Any, None]" = OrderedDict()

    # ---------- client -> server ----------

    def route_client_line(self, line: str) -> None:
        """Route one raw JSON-RPC line from the real client. Never raises: a malformed
        line (shouldn't happen — it came straight from the client's stdin) is silently
        ignored rather than crashing the whole session over one bad line."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(msg, dict):
            return

        method, mid = msg.get("method"), msg.get("id")

        if method == "tools/call":
            self._route_call(msg)
            return

        if method in _BROADCAST_METHODS:
            self._broadcast(msg, method)
            return

        if mid is None:
            # A notification (no id at all): fan out fire-and-forget, no reply to track.
            if method is not None:
                self._broadcast_notification(line)
            return

        if method is None:
            # A JSON-RPC response (never carries "method") with an id: the client's
            # reply to some peer's earlier server-initiated request, whose id
            # `_rewrite_server_request` rewrote specifically so it could be routed back
            # here — a peer-chosen id isn't namespaced across peers, so two peers could
            # otherwise pick colliding ids for unrelated requests. An id we don't
            # recognize (unknown, already answered, or evicted past _SERVER_REQ_MAX) is
            # dropped rather than forwarded to an arbitrary peer that never asked for it.
            with self._pending_lock:
                entry = self._server_requests.pop(mid, None)
            if entry is not None:
                peer_idx, orig_id = entry
                restored = dict(msg)
                restored["id"] = orig_id
                self._write_peer(peer_idx, json.dumps(restored, separators=(",", ":"),
                                                       ensure_ascii=False))
            elif self.debug:
                sys.stderr.write(f"[terse-multiproxy] reply for unknown id {mid!r} — "
                                 "no peer is waiting on it; dropped\n")
            return

        # v1 scope decision (see module docstring): anything else that still carries an
        # id — resources/list, prompts/list, ping, ... — is NOT worth full broadcast/
        # merge machinery. Forward to peer 0 only, and say so on stderr (unconditionally
        # — silently dropping N-1 peers' data from the reply is exactly the kind of gap
        # this file promises not to hide); a caller who needs more can front that server
        # alone.
        if self.peers:
            sys.stderr.write(
                f"[terse-multiproxy] {method!r} isn't tools/call or a broadcast "
                f"method; forwarding to peer 0 ({self.peers[0].name!r}) only "
                "(v1 scope — see multiproxy.py's module docstring)\n")
            self._write_peer(0, line)

    def _route_call(self, msg: dict) -> None:
        params = msg.get("params") or {}
        name = params.get("name")
        mid = msg.get("id")

        if name == lossy_mod.RETRIEVE_TOOL:
            # Unprefixed and answered by us: the store is shared, so ANY peer's
            # Interceptor sees every handle regardless of which peer dropped it.
            # SWALLOW — this never reaches a downstream, which has no such tool.
            reply = self.peers[0].inter.answer_retrieve(
                json.dumps(msg, separators=(",", ":"), ensure_ascii=False))
            if reply is not None:
                self._write_client(reply)
            return

        if isinstance(name, str) and PREFIX_SEP in name:
            peer_name, _, bare = name.partition(PREFIX_SEP)
            if peer_name in self.by_name:
                idx = self.by_name[peer_name]
                rewritten = dict(msg)
                rewritten["params"] = {**params, "name": bare}
                rewritten_line = json.dumps(rewritten, separators=(",", ":"),
                                            ensure_ascii=False)
                # The CLIENT's original id passes through unchanged: exactly one peer
                # ever answers a routed call, so there's no id collision risk here (that
                # risk only exists for the broadcast methods, which DO remap ids).
                # tool_name is the PEER-QUALIFIED name (e.g. "gh__search") — capture/
                # audit bookkeeping must not collide two different peers' same-named
                # tools into one corpus bucket, even though the wire line sent to the
                # downstream uses the bare name it actually expects.
                self.peers[idx].inter.note_request(rewritten_line, tool_name=name)
                self._write_peer(idx, rewritten_line)
                if mid is not None:
                    # Bound the wait — a hung/dead peer must not wedge this call
                    # forever, matching _broadcast's BROADCAST_TIMEOUT guarantee.
                    timer = threading.Timer(self.broadcast_timeout,
                                            self._timeout_routed_call, args=(mid, idx))
                    timer.daemon = True
                    with self._pending_lock:
                        self._routed_timers[mid] = timer
                    timer.start()
                return

        # Unknown peer prefix (or no prefix at all) — a legible JSON-RPC error, not a
        # crash or a silent hang, so the client sees exactly why the call went nowhere.
        if mid is not None:
            self._write_client(json.dumps(
                {"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32601,
                    "message": f"terse-multiproxy: unknown tool {name!r} (expected "
                              f"'<peer>{PREFIX_SEP}<tool>' for one of: "
                              f"{', '.join(self.by_name)})"}},
                separators=(",", ":"), ensure_ascii=False))

    def _broadcast(self, msg: dict, kind: str) -> None:
        if not self.peers:
            return
        client_id = msg.get("id")
        with self._pending_lock:
            seq = self._broadcast_seq
            self._broadcast_seq += 1
            # A client reusing an id while its prior broadcast is still in flight is
            # already a JSON-RPC protocol violation (ids must be unique among in-flight
            # requests) — fail safe against it anyway rather than trust it never
            # happens: abandon the PRIOR broadcast under ITS OWN seq (cancel its timer,
            # drop it from `_pending`) instead of overwriting shared state it still
            # references. Every broadcast gets a globally unique seq, so there is never
            # a live local-id collision between two broadcasts even under this
            # misbehavior — a late reply for the abandoned one resolves via
            # `_local_id_map` to a seq no longer in `_pending` and is safely swallowed
            # (see `_maybe_collect`), never misattributed to the new broadcast.
            prior_seq = self._active_seq.get(client_id)
            if prior_seq is not None:
                prior = self._pending.pop(prior_seq, None)
                if prior is not None and prior.timer is not None:
                    prior.timer.cancel()
            pb = _PendingBroadcast(kind=kind, client_id=client_id,
                                   remaining=set(range(len(self.peers))))
            self._pending[seq] = pb
            self._active_seq[client_id] = seq
            for i in range(len(self.peers)):
                self._local_id_map[f"terse-b{seq}-{i}"] = seq
            while len(self._local_id_map) > _LOCAL_ID_MAP_MAX:
                self._local_id_map.popitem(last=False)
            timer = threading.Timer(self.broadcast_timeout, self._timeout_broadcast,
                                    args=(seq,))
            timer.daemon = True
            pb.timer = timer
        timer.start()

        for i, peer in enumerate(self.peers):
            rewritten = dict(msg)
            rewritten["id"] = f"terse-b{seq}-{i}"
            line = json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)
            # For "initialize" this also resets this peer's OWN diff/pending/dropped
            # state (note_request's reconnect handling) — correct on a real client
            # reconnect, and since `dropped` is the SHARED store, harmlessly idempotent
            # when called once per peer. For "tools/list" note_request is a no-op (it
            # only acts on initialize/tools/call methods) — calling it unconditionally
            # here is simpler than branching on `kind`.
            peer.inter.note_request(line)
            if kind == "initialize":
                # note_request just set init_id to this broadcast-local id, but this
                # peer's reply is intercepted by _maybe_collect/_merge_initialize below
                # — it never reaches transform_response, so its one-time reset never
                # fires. Clear it immediately: multiproxy builds its own merged
                # initialize reply, so this peer's init_id has no other purpose.
                peer.inter.clear_init_id()
            self._write_peer(i, line)

    def _broadcast_notification(self, line: str) -> None:
        for sender in self._senders:
            sender.send(line)

    def _write_peer(self, idx: int, line: str) -> None:
        # Enqueued to the peer's own background sender (#5): never blocks this thread
        # on a peer's network round-trip, so one slow HTTP peer can't stall routing to
        # any OTHER peer — see _PeerSender's docstring.
        self._senders[idx].send(line)

    def _write_client(self, line: str) -> None:
        with self.out_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def close_senders(self) -> None:
        """Stop every peer's background sender thread (#5). Called once, after the
        client->server fan-out loop and `drain_pending_broadcasts` finish, before
        `run_multi_proxy` tears the peers' transports down — so no sender is still
        mid-write when its transport closes underneath it."""
        for sender in self._senders:
            sender.close()

    # ---------- server -> client ----------

    def from_peer(self, peer_idx: int) -> Callable[[str], Any]:
        """The `pump()` transform for peer `peer_idx`'s server->client reader thread."""
        def _transform(line: str) -> Any:
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                return None  # not JSON — forward unchanged, fail-open like proxy.py
            if isinstance(msg, dict) and msg.get("id") is not None:
                if self._maybe_collect(peer_idx, msg):
                    return SWALLOW  # a broadcast-local-id reply: never forwarded as-is
                mid = msg["id"]
                with self._pending_lock:
                    timer = self._routed_timers.pop(mid, None)
                    already_timed_out = timer is None and mid in self._routed_timed_out
                    if already_timed_out:
                        del self._routed_timed_out[mid]
                if timer is not None:
                    timer.cancel()
                if already_timed_out:
                    # The routed call's timeout already answered the client — this is
                    # the peer's late real reply arriving after the fact; swallow it
                    # rather than double-answering.
                    return SWALLOW
                if (msg.get("method") is not None
                        and "result" not in msg and "error" not in msg):
                    # A server-initiated request FROM this peer (sampling/createMessage,
                    # roots, ...) expecting a client reply: rewrite its id so the reply
                    # can be routed back to THIS peer specifically (see
                    # _rewrite_server_request / route_client_line's method-is-None
                    # branch), instead of the prior v1 gap where it was forwarded
                    # verbatim and any reply defaulted to peer 0.
                    return self._rewrite_server_request(peer_idx, msg)
            # A normal routed response (or the v1 forward-to-peer-0 passthrough): run it
            # through THIS peer's own Interceptor so its per-peer diff/drop/compress
            # state applies correctly. `note_request` was told the REWRITTEN/bare tool
            # name, so `transform_response`'s pending.pop(id) already resolves to the
            # bare name here — exactly right for compression; the client-facing prefix
            # only matters for tools/list, which is handled entirely in the merge path
            # below and never reaches transform_response (SWALLOWed above).
            return self.peers[peer_idx].inter.transform_response(line)
        return _transform

    def _rewrite_server_request(self, peer_idx: int, msg: dict) -> str:
        """Rewrite a server-initiated request's id to a router-local one and remember
        (peer_idx, original id) so `route_client_line` can restore it and deliver the
        client's eventual reply back to THIS peer. A peer-chosen id isn't namespaced —
        two peers could independently pick the same id for unrelated requests — so
        without this rewrite the client's reply would be ambiguous by construction, not
        just misrouted by policy. Bounded by `_SERVER_REQ_MAX` eviction (see module
        docstring): an evicted entry just means that particular late reply is dropped."""
        with self._pending_lock:
            local_id = f"terse-s{self._server_req_seq}"
            self._server_req_seq += 1
            self._server_requests[local_id] = (peer_idx, msg.get("id"))
            while len(self._server_requests) > _SERVER_REQ_MAX:
                self._server_requests.popitem(last=False)
        rewritten = dict(msg)
        rewritten["id"] = local_id
        return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)

    def _maybe_collect(self, peer_idx: int, msg: dict) -> bool:
        """If `msg.id` is a router-issued broadcast-local id, record it into the
        matching pending broadcast and return True (the caller must SWALLOW — the
        peer-local-id message itself is never forwarded). False means this id is NOT
        one of ours — a normal routed response, handle it the usual way."""
        mid = msg["id"]
        with self._pending_lock:
            # `.get`, not `.pop`: a reply that arrives AFTER its broadcast already
            # finished (timed out or fully merged) must still resolve here and be
            # swallowed below (pb is None/done -> True) rather than fall through to
            # transform_response as an unsolicited message. _local_id_map entries are
            # bounded by eviction (_LOCAL_ID_MAP_MAX in _broadcast), not by popping here.
            seq = self._local_id_map.get(mid)
            if seq is None:
                return False
            pb = self._pending.get(seq)
            if pb is None or pb.done:
                return True  # already merged/timed out — a late arrival; swallow, drop
            pb.parts[peer_idx] = msg
            pb.remaining.discard(peer_idx)
            remaining_empty = not pb.remaining
        if remaining_empty:
            self._finish_broadcast(seq)
        return True

    def drain_pending_broadcasts(self) -> None:
        """Block until every currently in-flight broadcast has been merged and replied
        to. Each is still bounded by its own `broadcast_timeout` (via the `threading.
        Timer` already scheduled for it — joining a Timer thread that has already fired
        `.cancel()` returns immediately), so this can never hang. Called once, right
        after the client's stdin hits EOF, before `run_multi_proxy` tears peers down: a
        client that disconnects immediately after e.g. `initialize`, before any peer
        answered, must not lose that reply just because shutdown started before the
        broadcast's own timer had a chance to fire."""
        with self._pending_lock:
            timers = [pb.timer for pb in self._pending.values() if pb.timer is not None]
        for timer in timers:
            timer.join(timeout=self.broadcast_timeout + 1.0)

    def _timeout_broadcast(self, seq: int) -> None:
        with self._pending_lock:
            pb = self._pending.get(seq)
            if pb is None or pb.done:
                return
            missing = sorted(pb.remaining)
            client_id = pb.client_id
        names = [self.peers[i].name for i in missing]
        sys.stderr.write(f"[terse-multiproxy] broadcast (client id {client_id!r}) timed "
                         f"out after {self.broadcast_timeout}s waiting on peer(s) {names}; "
                         "merging with whatever arrived — a dead/slow peer never wedges "
                         "the proxy\n")
        self._finish_broadcast(seq)

    def _timeout_routed_call(self, mid: Any, peer_idx: int) -> None:
        """Fires if a routed (single-peer) `tools/call`'s target peer never replies
        within `broadcast_timeout` — the routed-call counterpart to `_timeout_broadcast`.
        Whichever side (this timer, or the real reply in `from_peer`) wins the race to
        pop `_routed_timers[mid]` is the one that answers the client; if this timer
        wins, the id is remembered in `_routed_timed_out` so the peer's eventual late
        real reply is swallowed instead of double-answering the client."""
        with self._pending_lock:
            timer = self._routed_timers.pop(mid, None)
            if timer is None:
                return  # the real reply already arrived and cancelled this timer
            self._routed_timed_out[mid] = None
            while len(self._routed_timed_out) > _ROUTED_TIMED_OUT_MAX:
                self._routed_timed_out.popitem(last=False)
        peer_name = self.peers[peer_idx].name if 0 <= peer_idx < len(self.peers) else "?"
        sys.stderr.write(f"[terse-multiproxy] peer {peer_name!r} did not answer "
                         f"tools/call id={mid!r} within {self.broadcast_timeout}s — "
                         "replying with a timeout error; a dead/slow peer never wedges "
                         "a routed call\n")
        self._write_client(json.dumps(
            {"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32001,
                "message": f"terse-multiproxy: peer {peer_name!r} timed out"}},
            separators=(",", ":"), ensure_ascii=False))

    def _finish_broadcast(self, seq: int) -> None:
        with self._pending_lock:
            pb = self._pending.pop(seq, None)
            if pb is None or pb.done:
                return
            pb.done = True
            if pb.timer is not None:
                pb.timer.cancel()
            if self._active_seq.get(pb.client_id) == seq:
                self._active_seq.pop(pb.client_id, None)
            # `_local_id_map` entries for this seq are deliberately left in place (see
            # __init__/_maybe_collect) — they're bounded by eviction, not popped here,
            # so a reply that arrives after this point still resolves to "seq not in
            # _pending" and is safely swallowed instead of leaking to the client.

        merged = (self._merge_initialize(pb) if pb.kind == "initialize"
                 else self._merge_tools_list(pb))
        self._write_client(json.dumps(
            {"jsonrpc": "2.0", "id": pb.client_id, "result": merged},
            separators=(",", ":"), ensure_ascii=False))

    # ---------- broadcast merges ----------

    def _merge_tools_list(self, pb: _PendingBroadcast) -> dict:
        """Concat every peer's tools with `{peer}__` prefixes; append the single
        unprefixed `RETRIEVE_TOOL_DEF` exactly once (never per-peer — each peer's own
        `Interceptor._inject_retrieve_tool` is bypassed here since the peer-local-id
        reply never reaches `transform_response`), and only if some peer's policy
        actually enables drop-to-retrieve, matching single-peer behavior."""
        tools: list[dict] = []
        for i, peer in enumerate(self.peers):
            result = pb.parts.get(i, {}).get("result")
            peer_tools = result.get("tools") if isinstance(result, dict) else None
            if not isinstance(peer_tools, list):
                continue  # peer errored, never answered, or replied oddly — skip it
            for t in peer_tools:
                if not (isinstance(t, dict) and isinstance(t.get("name"), str)):
                    continue
                tools.append({**t, "name": f"{peer.name}{PREFIX_SEP}{t['name']}"})
        if self.has_drop:
            tools.append(RETRIEVE_TOOL_DEF)
        return {"tools": tools}

    def _merge_initialize(self, pb: _PendingBroadcast) -> dict:
        """`protocolVersion` = the first-arriving peer's (documented; peers are expected
        to agree in practice). `capabilities` = a shallow dict-union (last-peer-wins on a
        key clash — no ordering guarantee stronger than "arrival order", documented).
        `serverInfo` names US, not any one peer (merging N identities into one isn't
        meaningful). `instructions` = ONE `TERSE_PRIMER`, first, then each peer's own
        non-empty instructions (skipping one that already carries the primer, so a peer
        that is ITSELF a terse proxy doesn't duplicate it)."""
        protocol_version: Optional[str] = None
        capabilities: dict = {}
        instructions_parts: list[str] = []
        # Iterate pb.parts in ITS OWN (insertion) order, not range(len(self.peers))
        # (fixed config order) — pb.parts[peer_idx] = msg is only ever written under
        # _pending_lock as each reply genuinely lands (see _maybe_collect), so its
        # insertion order IS true arrival order, matching this method's own
        # "first-arriving"/"arrival order" contract below.
        for result in (part.get("result") for part in pb.parts.values()):
            if not isinstance(result, dict):
                continue
            if protocol_version is None and isinstance(result.get("protocolVersion"), str):
                protocol_version = result["protocolVersion"]
            caps = result.get("capabilities")
            if isinstance(caps, dict):
                capabilities.update(caps)
            instr = result.get("instructions")
            if isinstance(instr, str) and instr.strip() and TERSE_PRIMER not in instr:
                instructions_parts.append(instr)
        if protocol_version is None:
            protocol_version = "2024-11-05"  # every peer errored/timed out — a safe fallback
        instructions = TERSE_PRIMER
        if instructions_parts:
            instructions += "\n\n" + "\n\n".join(instructions_parts)
        from . import __version__
        return {
            "protocolVersion": protocol_version,
            "capabilities": capabilities,
            "serverInfo": {"name": "terse", "version": __version__},
            "instructions": instructions,
        }


def _build_peers(specs: list[DownstreamSpec], default_policy: policy_mod.Policy, *,
                 debug: bool, capture: Optional[Callable[[str, str], None]],
                 audit: Optional[Callable[[dict], None]],
                 store: "OrderedDict[str, Any]", store_lock: Lock,
                 dropped_bytes: list[int], diff_override: bool = False,
                 diff_keyframe_override: Optional[int] = None) -> list[Peer]:
    """Build every `Peer`: its own `Transport` (stdio or HTTP, via `build_transport`)
    and its own `Interceptor` (per-peer diff/compress state, but the drop store —
    including its byte-eviction counter — is injected shared). Raises on a bad spec —
    OSError if a stdio peer can't be launched, ValueError if a peer's own policy file
    is malformed — and closes whatever peers WERE already built (their live
    children/connections) before re-raising, so a bad Nth peer doesn't orphan an
    earlier peer's already-launched child. Catches Exception broadly (not just
    OSError) since policy loading can fail for reasons unrelated to process launch.

    `diff_override`/`diff_keyframe_override` are applied to EVERY peer's policy, not
    just `default_policy`-derived ones — otherwise a peer with its own `policy_path`
    silently never sees the CLI's `--diff` opt-in, unlike a peer using the default."""
    peers: list[Peer] = []
    try:
        for spec in specs:
            pol = (policy_mod.load_policy(spec.policy_path) if spec.policy_path
                  else default_policy)
            if diff_override:
                pol.diff = True
            if diff_keyframe_override is not None:
                pol.diff_keyframe_interval = diff_keyframe_override
            inter = Interceptor(pol, debug=debug, capture=capture, audit=audit,
                                store=store, store_lock=store_lock,
                                dropped_bytes=dropped_bytes)
            transport = build_transport(spec.target, headers=spec.headers or None)
            peers.append(Peer(name=spec.name, transport=transport, inter=inter))
    except Exception:
        for peer in peers:
            peer.transport.close()
        raise
    return peers


def run_multi_proxy(
    config_path: str,
    default_policy: policy_mod.Policy,
    *,
    debug: bool = False,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    capture_dir: Optional[str] = None,
    debug_log: Optional[str] = None,
    broadcast_timeout: float = BROADCAST_TIMEOUT,
    diff_override: bool = False,
    diff_keyframe_override: Optional[int] = None,
) -> int:
    """Load `config_path`, build one `Peer` per downstream (own `Transport` + own
    `Interceptor`, all sharing one drop store), spawn one `pump()` reader thread per
    peer (server->client) plus run the client->server fan-out loop on this thread, and
    block until the client's stdin hits EOF. `broadcast_timeout` overrides
    `BROADCAST_TIMEOUT` — a test-only knob so a dead-peer test doesn't need to wait out
    the real 30s default. `diff_override`/`diff_keyframe_override` mirror cli.py's
    single-peer `--diff`/`--diff-keyframe-interval` CLI opt-in — applied to EVERY
    peer's policy in `_build_peers`, including one loaded from its own `policy_path`,
    so `--diff` is proxy-wide (not silently skipped for a peer with a custom policy).

    Return code: 2 for a bad/missing config (mirrors `run_proxy`'s `stdio_transport_error`
    path), 127 if a stdio peer can't be launched (mirrors `run_proxy`'s OSError path), 0
    once the client disconnects cleanly. There is no single child exit code to propagate
    (there are N children) — 0 means "the client's stdin hit EOF and every peer was
    closed", same meaning `run_proxy` gives an HTTP downstream today."""
    cin = stdin or sys.stdin
    cout = stdout or sys.stdout

    try:
        specs = load_multi_config(config_path)
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError subclass
        sys.stderr.write(f"[terse-multiproxy] {exc}\n")
        return 2

    capture, audit = _build_capture_and_audit(capture_dir, debug_log, debug, "[terse-multiproxy]")

    store: "OrderedDict[str, Any]" = OrderedDict()
    store_lock = Lock()
    dropped_bytes: list[int] = [0]

    try:
        peers = _build_peers(specs, default_policy, debug=debug, capture=capture,
                             audit=audit, store=store, store_lock=store_lock,
                             dropped_bytes=dropped_bytes, diff_override=diff_override,
                             diff_keyframe_override=diff_keyframe_override)
    except OSError as exc:
        sys.stderr.write(f"[terse-multiproxy] failed to launch a downstream peer: {exc}\n")
        return 127
    except ValueError as exc:
        sys.stderr.write(f"[terse-multiproxy] bad downstream policy: {exc}\n")
        return 2

    out_lock = Lock()
    router = Router(peers, cout, out_lock, debug=debug, broadcast_timeout=broadcast_timeout)

    sigterm_token = _install_sigterm_to_exit()

    try:
        threads = [Thread(target=pump,
                          args=(peer.transport.inbound(), cout, router.from_peer(i)),
                          kwargs={"lock": out_lock}, daemon=True)
                  for i, peer in enumerate(peers)]
        for t in threads:
            t.start()

        # Client->server is a dedicated loop, not a pump thread: fan-out from one source
        # to N destinations is a genuinely different shape than pump's 1:1 line-in/
        # line-out (see module docstring). Running it inline on THIS thread means
        # "block until client stdin EOF" falls out of the for-loop ending, for free.
        for raw in cin:
            line = raw.rstrip("\n")
            if not line:
                continue
            router.route_client_line(line)

        # A client that disconnects the instant after e.g. `initialize` — before any
        # peer answered — must still get its merged reply; wait out any broadcast still
        # in flight (bounded by its own timeout) before tearing peers down.
        router.drain_pending_broadcasts()

        # Stop every peer's background sender BEFORE closing any transport below —
        # otherwise a sender thread could still be mid-write to a transport that's
        # about to be torn down.
        router.close_senders()

        # Client EOF: let each peer wind down like run_proxy's client_to_server finally
        # does for a single stdio downstream — transport.half_close() (Transport
        # method, shared with proxy.py — see transport.py) handles the stdio-vs-HTTP
        # distinction internally, so this loop needs no isinstance check on the
        # concrete transport type. Then join every reader thread so no peer output is
        # still in flight when this function returns.
        for peer in peers:
            peer.transport.half_close()
        for t in threads:
            t.join(timeout=2.0)
        return 0
    finally:
        _ignore_sigterm(sigterm_token)
        # Idempotent defense-in-depth: if the try block raised before reaching the
        # normal close_senders()/transport-teardown sequence above, stop every sender
        # here too before closing transports. A repeat call is a harmless no-op: the
        # sender thread already exited on the first STOP, so this just re-queues one
        # nobody will read and joins an already-finished thread.
        router.close_senders()
        # Idempotent last-resort reaper for every peer (SIGTERM/SIGKILL escalation for a
        # stdio child that didn't exit on its own; a harmless repeat close for HTTP).
        for peer in peers:
            peer.transport.close()
        _restore_sigterm(sigterm_token)
