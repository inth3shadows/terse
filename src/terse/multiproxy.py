"""Multi-downstream fan-out proxy (#5 Half B): ONE `terse proxy --config` process
fronting N downstream MCP peers (any mix of stdio and HTTP, via `transport.py`),
merging their `tools/list` under name prefixes, routing `tools/call` by prefix, and
sharing the drop-to-retrieve store across all of them.

This is explicitly an ergonomics feature (single policy/primer/process) — MCP clients
already multiplex servers natively — so the v1 scope below is deliberately
proportionate. Documented limitations are fine; silent gaps are not:
  - A server-initiated request FROM a downstream that expects a client reply
    (sampling/createMessage, roots) is forwarded to the client, but the client's reply
    is not routed back to the originating peer — there is no reverse-routing table in
    v1. Rare for a compression proxy.
  - Any client method other than `initialize`/`tools/list` (broadcast+merge) or
    `tools/call` (routed by prefix) — e.g. `resources/list`, `prompts/list`, `ping` —
    is forwarded to peer 0 ONLY, not broadcast/merged. Building full N-way
    broadcast/merge logic for every possible MCP method is disproportionate for a
    feature the issue itself calls "ergonomics, modest value".
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
import signal
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Optional, TextIO

from . import lossy as lossy_mod
from . import policy as policy_mod
from .proxy import RETRIEVE_TOOL_DEF, SWALLOW, TERSE_PRIMER, Interceptor, pump
from .transport import HttpTransport, Transport, build_transport

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
    the ORIGINAL client request id. `parts` collects each peer's parsed reply as it
    arrives (peer index -> the full JSON-RPC message, so a peer error is distinguishable
    from a peer that hasn't answered yet); `remaining` shrinks to empty as replies land,
    or the broadcast is force-completed by `Router._timeout_broadcast` once
    `BROADCAST_TIMEOUT` elapses. `local_ids` is every peer's rewritten id for THIS
    broadcast, kept so a late (post-timeout) reply and a leftover `_local_id_map` entry
    can both be cleaned up once the broadcast finishes."""
    kind: str
    remaining: set[int]
    parts: dict[int, dict] = field(default_factory=dict)
    local_ids: dict[int, Any] = field(default_factory=dict)
    timer: Optional[threading.Timer] = None
    done: bool = False


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

        self._pending_lock = Lock()
        self._pending: dict[Any, _PendingBroadcast] = {}
        # peer-local broadcast id -> the ORIGINAL client id it was issued for. Doubles as
        # the "is this id one of ours" check: from_peer() pops from here, so an unknown id
        # (a normal routed tools/call response) is a clean miss, not a string-parse guess.
        self._local_id_map: dict[Any, Any] = {}

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

        # v1 scope decision (see module docstring): anything else that still carries an
        # id — resources/list, prompts/list, ping, ... — is NOT worth full broadcast/
        # merge machinery. Forward to peer 0 only, and say so on stderr rather than
        # silently guessing; a caller who needs more can front that server alone.
        if self.peers:
            if self.debug:
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
                self.peers[idx].inter.note_request(rewritten_line)
                self._write_peer(idx, rewritten_line)
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
        pb = _PendingBroadcast(kind=kind, remaining=set(range(len(self.peers))))
        with self._pending_lock:
            # JSON-RPC forbids reusing an id while the matching request is in flight, so
            # a client doing that is already violating the protocol; fail safe rather
            # than silently corrupting bookkeeping by dropping the stale entry first.
            prior = self._pending.pop(client_id, None)
            if prior is not None and prior.timer is not None:
                prior.timer.cancel()
            for i in range(len(self.peers)):
                local_id = f"terse-b{client_id}-{i}"
                pb.local_ids[i] = local_id
                self._local_id_map[local_id] = client_id
            self._pending[client_id] = pb
            timer = threading.Timer(self.broadcast_timeout, self._timeout_broadcast,
                                    args=(client_id,))
            timer.daemon = True
            pb.timer = timer
        timer.start()

        for i, peer in enumerate(self.peers):
            rewritten = dict(msg)
            rewritten["id"] = pb.local_ids[i]
            line = json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)
            # For "initialize" this also resets this peer's OWN diff/pending/dropped
            # state (note_request's reconnect handling) — correct on a real client
            # reconnect, and since `dropped` is the SHARED store, harmlessly idempotent
            # when called once per peer. For "tools/list" note_request is a no-op (it
            # only acts on initialize/tools/call methods) — calling it unconditionally
            # here is simpler than branching on `kind`.
            peer.inter.note_request(line)
            self._write_peer(i, line)

    def _broadcast_notification(self, line: str) -> None:
        for peer in self.peers:
            try:
                self._write_peer_line(peer, line)
            except Exception as exc:  # noqa: BLE001 — one broken peer must not crash fan-out
                if self.debug:
                    sys.stderr.write(f"[terse-multiproxy] notification to {peer.name!r} "
                                     f"failed: {exc}\n")

    def _write_peer(self, idx: int, line: str) -> None:
        self._write_peer_line(self.peers[idx], line)

    @staticmethod
    def _write_peer_line(peer: Peer, line: str) -> None:
        w = peer.transport.outbound()
        w.write(line + "\n")
        w.flush()

    def _write_client(self, line: str) -> None:
        with self.out_lock:
            self.out.write(line + "\n")
            self.out.flush()

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
            # A normal routed response (or the v1 forward-to-peer-0 passthrough): run it
            # through THIS peer's own Interceptor so its per-peer diff/drop/compress
            # state applies correctly. `note_request` was told the REWRITTEN/bare tool
            # name, so `transform_response`'s pending.pop(id) already resolves to the
            # bare name here — exactly right for compression; the client-facing prefix
            # only matters for tools/list, which is handled entirely in the merge path
            # below and never reaches transform_response (SWALLOWed above).
            return self.peers[peer_idx].inter.transform_response(line)
        return _transform

    def _maybe_collect(self, peer_idx: int, msg: dict) -> bool:
        """If `msg.id` is a router-issued broadcast-local id, record it into the
        matching pending broadcast and return True (the caller must SWALLOW — the
        peer-local-id message itself is never forwarded). False means this id is NOT
        one of ours — a normal routed response, handle it the usual way."""
        mid = msg["id"]
        with self._pending_lock:
            client_id = self._local_id_map.pop(mid, None)
            if client_id is None:
                return False
            pb = self._pending.get(client_id)
            if pb is None or pb.done:
                return True  # already merged/timed out — a late arrival; swallow, drop
            pb.parts[peer_idx] = msg
            pb.remaining.discard(peer_idx)
            remaining_empty = not pb.remaining
        if remaining_empty:
            self._finish_broadcast(client_id)
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

    def _timeout_broadcast(self, client_id: Any) -> None:
        with self._pending_lock:
            pb = self._pending.get(client_id)
            if pb is None or pb.done:
                return
            missing = sorted(pb.remaining)
        names = [self.peers[i].name for i in missing]
        sys.stderr.write(f"[terse-multiproxy] broadcast (client id {client_id!r}) timed "
                         f"out after {self.broadcast_timeout}s waiting on peer(s) {names}; "
                         "merging with whatever arrived — a dead/slow peer never wedges "
                         "the proxy\n")
        self._finish_broadcast(client_id)

    def _finish_broadcast(self, client_id: Any) -> None:
        with self._pending_lock:
            pb = self._pending.pop(client_id, None)
            if pb is None or pb.done:
                return
            pb.done = True
            if pb.timer is not None:
                pb.timer.cancel()
            # Clean up any local ids that never got a reply (a timed-out peer) so
            # _local_id_map can't grow unboundedly over a long session with a
            # persistently dead peer.
            for i in pb.remaining:
                self._local_id_map.pop(pb.local_ids.get(i), None)

        merged = (self._merge_initialize(pb) if pb.kind == "initialize"
                 else self._merge_tools_list(pb))
        self._write_client(json.dumps({"jsonrpc": "2.0", "id": client_id, "result": merged},
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
        for i in range(len(self.peers)):
            result = pb.parts.get(i, {}).get("result")
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
                 store: "OrderedDict[str, Any]", store_lock: Lock) -> list[Peer]:
    """Build every `Peer`: its own `Transport` (stdio or HTTP, via `build_transport`)
    and its own `Interceptor` (per-peer diff/compress state, but the drop store is
    injected shared). Raises OSError if a stdio peer can't be launched — the caller
    closes whatever peers WERE built before re-raising/reporting, so a bad 2nd peer
    doesn't orphan the 1st peer's already-launched child."""
    peers: list[Peer] = []
    for spec in specs:
        pol = (policy_mod.load_policy(spec.policy_path) if spec.policy_path
              else default_policy)
        inter = Interceptor(pol, debug=debug, capture=capture, audit=audit,
                            store=store, store_lock=store_lock)
        transport = build_transport(spec.target, headers=spec.headers or None)
        peers.append(Peer(name=spec.name, transport=transport, inter=inter))
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
) -> int:
    """Load `config_path`, build one `Peer` per downstream (own `Transport` + own
    `Interceptor`, all sharing one drop store), spawn one `pump()` reader thread per
    peer (server->client) plus run the client->server fan-out loop on this thread, and
    block until the client's stdin hits EOF. `broadcast_timeout` overrides
    `BROADCAST_TIMEOUT` — a test-only knob so a dead-peer test doesn't need to wait out
    the real 30s default.

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

    capture: Optional[Callable[[str, str], None]] = None
    if capture_dir is not None:
        from .capture import capture_payload

        def capture(tool: str, raw: str) -> None:
            try:
                capture_payload(tool, raw, capture_dir)
            except Exception as exc:  # noqa: BLE001 — capture is never load-bearing
                if debug:
                    sys.stderr.write(f"[terse-multiproxy] capture_payload failed: {exc}\n")

    audit: Optional[Callable[[dict], None]] = None
    if debug_log is not None:
        from .capture import append_audit

        def audit(record: dict) -> None:
            try:
                append_audit(record, debug_log)
            except Exception as exc:  # noqa: BLE001 — audit is never load-bearing
                if debug:
                    sys.stderr.write(f"[terse-multiproxy] append_audit failed: {exc}\n")

    store: "OrderedDict[str, Any]" = OrderedDict()
    store_lock = Lock()

    try:
        peers = _build_peers(specs, default_policy, debug=debug, capture=capture,
                             audit=audit, store=store, store_lock=store_lock)
    except OSError as exc:
        sys.stderr.write(f"[terse-multiproxy] failed to launch a downstream peer: {exc}\n")
        return 127

    out_lock = Lock()
    router = Router(peers, cout, out_lock, debug=debug, broadcast_timeout=broadcast_timeout)

    # SIGTERM handling mirrors run_proxy's (#21): convert to a clean SystemExit so every
    # peer still gets reaped via `finally` instead of the default action bypassing it.
    prev_sigterm = None
    installed_sigterm = False
    try:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        installed_sigterm = True
    except (ValueError, OSError):
        pass

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

        # Client EOF: let each peer wind down like run_proxy's client_to_server finally
        # does for a single stdio downstream — half-close a stdio peer's stdin so a
        # well-behaved child sees EOF and exits on its own; an HTTP peer has no
        # persistent connection to half-close, so close it outright (pushes the sentinel
        # that ends its inbound() queue iterator). Then join every reader thread so no
        # peer output is still in flight when this function returns.
        for peer in peers:
            if isinstance(peer.transport, HttpTransport):
                peer.transport.close()
            else:
                try:
                    peer.transport.outbound().close()
                except Exception:  # noqa: BLE001 — best-effort half-close
                    pass
        for t in threads:
            t.join(timeout=2.0)
        return 0
    finally:
        if installed_sigterm:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        # Idempotent last-resort reaper for every peer (SIGTERM/SIGKILL escalation for a
        # stdio child that didn't exit on its own; a harmless repeat close for HTTP).
        for peer in peers:
            peer.transport.close()
        if installed_sigterm:
            try:
                signal.signal(signal.SIGTERM,
                              prev_sigterm if prev_sigterm is not None else signal.SIG_DFL)
            except (ValueError, OSError, TypeError):
                pass
