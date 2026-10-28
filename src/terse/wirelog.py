"""The wire log (`proxy --wire-log DIR`): a lossless, append-only record of every JSON-RPC
line crossing terse, at all four hops, so an agent's MCP observations — the original a
server sent AND what the model was actually given — can be rebuilt afterwards and joined to
the client's own transcript.

One file per proxy process: `DIR/<CLAUDE_CODE_SESSION_ID or no-session>/<conn>.jsonl`,
dirs 0700, file 0600. One JSON object per line; see TECHNICAL.md "Wire log" for the schema.

Completeness, not volume, is the point, so the design is about never losing a line
silently and never breaking forwarding:
- `tap()` runs on the forwarding threads and stays small: two clocks, a parse only when the
  exclusion gate needs one (every client line; another line only while something on its
  side is excluded), the decision, a lock, a queue put. Hashing and serializing happen on
  ONE writer thread, which also keeps the file in `seq` order.
- `seq` is assigned under the lock for every record, kept or not. A full queue drops the
  record (its seq becomes a hole) and counts it; the count is written as a `sink_drop` event
  as soon as there is room, so a gap is always both visible and explained.
- An OSError on open or write, or any unexpected exception in the tap or the writer, warns
  once on stderr and disables the tap; the proxy carries on exactly as without
  `--wire-log`.

Exclusion (`capture: false`) is decided in `tap()`, on the forwarding thread that produced
the line, under the same lock that assigns `seq` — so decisions are made in wire order, and
never through the queue (a dropped request record cannot let its reply through in full).
The writer only renders a decision already taken. See `_Scope` for the state machine.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import queue
import re
import stat
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from typing import Any, TextIO

from ._secure_io import mkdir_restricted, open_append_restricted

SCHEMA_VERSION = 1
NO_SESSION = "no-session"
_QUEUE_MAX = 16384
_QUEUE_BYTES_MAX = 64 * 1024 * 1024
_BATCH_MAX = 512
# Per-file size cap. The log is lossless by design, so nothing else bounded it: a chatty or
# hostile downstream could fill the disk through one connection's file. Past the cap one
# `truncated` event is written and the rest of the connection is not logged.
# `TERSE_WIRE_LOG_MAX_BYTES` overrides it; 0 means unlimited.
_FILE_BYTES_MAX = 512 * 1024 * 1024
# Excluded ids/progress tokens still awaiting their reply are never evicted (an evicted one
# would let its late reply through in full); past this many the scope fails closed instead.
_LIVE_MAX = 65536
# Answered ones are kept this long (FIFO), so a late duplicate or a reused id stays excluded.
_DONE_MAX = 4096
# A tool name no real server exports: what `Policy.select` returns for it is the rule an
# UNKNOWN tool on that server falls to — i.e. the server's default. capture:false there
# (secret-broker's `secret-broker.*` default-deny) excludes the whole server.
_PROBE_TOOL = "\x00terse-wire-probe"
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")
# `rule` values for exclusions no policy rule names: a line the gate could not classify (a
# batch element that is not an object, duplicate keys, nesting too deep to parse), the gate
# itself raising, and the in-flight table overflowing.
RULE_UNDECIDABLE = "<undecidable>"
RULE_GATE_ERROR = "<gate-error>"
RULE_OVERFLOW = "<exclusion-overflow>"
# An excluded request went to a server with an id its reply cannot be matched by (null, or a
# line only partly parseable): every later line from that server is excluded (#524).
RULE_UNMATCHED = "<unmatched-excluded-call>"
_CLIENT_HOPS = frozenset(("client_in", "client_out"))
_READ_HOPS = frozenset(("client_in", "server_in"))
# Hops on which a request carries an id the CLIENT chose (and a response answers one the
# SERVER chose); on the other two it is the reverse.
_CLIENT_ID_REQUEST_HOPS = frozenset(("client_in", "server_out"))

_INT_ID = re.compile(r"-?[0-9]{1,19}")
# Where an id-like value sits in a raw line: the values `_Scope.needs_parse` pre-scans
# against answered keys. Key names written with escapes are not seen (see `needs_parse`).
_ID_VALUES = re.compile(r'"(?:id|progressToken|requestId)"\s*:\s*'
                        r'("(?:[^"\\]|\\.)*"|-?[0-9][0-9.eE+-]*)')
_KINDS = ("c", "s", "pc", "ps")
# Past this many id-like values a line is dense structured data, where the scan costs as
# much as the parse it would save: parse it, as before #525.
_SCAN_MAX = 256
# A candidate spelled other than `_spellings` predicts (an escape or non-ASCII in a string,
# a fraction or exponent in a number) is decoded before it is compared.
_ODD_STR = re.compile(r'[\\\x80-\U0010ffff]')
_ODD_NUM = re.compile(r"[.eE+]")


def _spellings(key: str) -> tuple[str, ...]:
    """How a line most likely writes an id whose `_id_key` is `key`."""
    if key[:1] == '"':
        return (key,)
    if _INT_ID.fullmatch(key):
        return (key, f'"{key}"')
    return ()

# `resolve(msg)` for one parsed client message: (rule glob, excluded) or None (no rule).
Resolver = Callable[[dict], "tuple[str | None, bool] | None"]


def _id_key(mid: Any) -> str:
    if isinstance(mid, float) and mid.is_integer():
        mid = int(mid)          # a server echoing 5 as 5.0 is still answering 5
    elif isinstance(mid, str) and _INT_ID.fullmatch(mid):
        mid = int(mid)          # nor is one coercing "5" to 5 (#524); a collision only
                                # ever over-excludes
    try:
        return json.dumps(mid, sort_keys=True)
    except (TypeError, ValueError):
        return repr(mid)


def server_excluded(policy: Any, server: str | None) -> str | None:
    """The rule glob when `server`'s default is capture:false (the whole server is
    excluded), else None."""
    rule = policy.select(_PROBE_TOOL, server)
    return None if rule.capture else rule.tool_glob


def retrieve_handle(params: dict) -> str | None:
    args = params.get("arguments")
    handle = args.get("handle") if isinstance(args, dict) else None
    return handle if isinstance(handle, str) else None


class _DuplicateKey(Exception):
    pass


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        # Python keeps the LAST duplicate; another server's parser may keep the first, so
        # the tool terse resolves need not be the tool that runs.
        raise _DuplicateKey
    return obj


_STRICT = json.JSONDecoder(object_pairs_hook=_strict_pairs)


def _parse(line: str, strict: bool) -> tuple[str, Any]:
    """("ok", value), ("bad", None) for a blank/non-JSON line, or ("undecidable", None) for
    one that may be JSON to someone but cannot be classified here."""
    if not line.strip():
        return "bad", None
    try:
        if strict:
            return "ok", _STRICT.decode(line)   # a leading BOM raises JSONDecodeError
        return "ok", json.loads(line)
    except (_DuplicateKey, RecursionError):
        return "undecidable", None
    except json.JSONDecodeError:
        return "bad", None
    except ValueError:      # valid JSON refused here, e.g. an int past the digit limit
        return "undecidable", None


class _Scope:
    """The exclusion state machine for one connection (one WireTap). Every method runs
    under the tap's lock.

    A key is (side, kind, id): `side` is "c" for the client-facing hops or "p:<peer>" for
    one peer's hops (ids are rewritten between the two in multiproxy, and two peers' ids
    are unrelated); `kind` is "c" for a request id the client side chose, "s" for one the
    server side chose, "pc"/"ps" for the progress token of such a request.

    - A request line that is excluded registers its id (and its progress token) on the
      side it was seen. Write hops inherit the decision of the read they came from (the
      tap's thread-local context, or `inherit=` from a queued sender), so the forwarded
      copy of an excluded request registers the copy's own id: the router's `terse-b*`
      broadcast ids, or the `terse-s*` id a server request is rewritten to.
    - A response is excluded iff its id is registered on that side, and it completes the
      key (moves it to `done`). A progress/cancelled notification is matched by its token
      or requestId.
    - While a peer has an excluded request in flight, every server_in line from it that
      is not a response to a non-excluded id is excluded: server requests (sampling,
      elicitation, roots, ping), notifications, noise. Their ids register, so the client's
      reply to them is excluded on both hops.
    - `excluded_peers` (whole-server capture:false) excludes every server line of that
      peer; `exclude_all` the whole connection.

    - An excluded request sent to a server with an id its reply cannot be matched by
      (null, or a line that cannot be fully parsed) fails closed: every later server_in
      line from that peer is excluded (`unmatched`) until it exits (#524).
    - With `may_exclude` false (no rule anywhere has capture:false), an unparseable line
      is logged like any other instead of excluded as undecidable.

    Live keys are never evicted; at `_LIVE_MAX` the scope fails closed for the rest of the
    connection (`overflow`) instead of forgetting one."""

    def __init__(self, exclude_all: str | None, excluded_peers: dict[str, str],
                 may_exclude: bool = True):
        self.exclude_all = exclude_all
        self.excluded_peers = excluded_peers
        self.may_exclude = may_exclude
        self.resolve: Resolver | None = None
        self.live: dict[tuple[str, str, str], tuple[str, list[tuple[str, str, str]]]] = {}
        self.done: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self.sides: dict[str, int] = {}   # keys (live or done) per side; 0 = nothing to match
        self.live_sides: dict[str, int] = {}   # live keys alone, per side
        self._done_ver = 0                     # bumped on every change to `done`
        self._done_text: tuple[int, dict[str, set[str]]] = (-1, {})
        self.unmatched: dict[str, str] = {}    # peer side -> rule, until the peer exits
        self.calls: dict[str, tuple[int, str]] = {}   # peer side -> (live requests, a rule)
        self.overflow: str | None = None
        self.overflowed = False           # set once, for the caller to emit an event

    @staticmethod
    def side(hop: str, peer: str | None) -> str:
        return "c" if hop in _CLIENT_HOPS else f"p:{peer}"

    def needs_parse(self, hop: str, peer: str | None, inherited: str | None,
                    line: str) -> bool:
        if hop == "client_in" or inherited is not None:
            return True
        if hop not in _CLIENT_HOPS and peer in self.excluded_peers:
            return True                   # its requests must register, for the replies
        side = self.side(hop, peer)
        if self.live_sides.get(side):
            return True
        # Only answered keys left: parse just a line naming one (a late duplicate reply, a
        # reused id), not every line for good (#525). Blind spot: a key name written with
        # escapes ("\u0069d"), which no server sends for a reply it already gave.
        return bool(self.sides.get(side)) and self._names_done(side, line)

    def _names_done(self, side: str, line: str) -> bool:
        ver, text = self._done_text
        if ver != self._done_ver:         # rebuilt once per change, not once per line
            ver, text = self._done_ver, {}
            for done_key in list(self.done):
                text.setdefault(done_key[0], set()).update(_spellings(done_key[2]))
            self._done_text = (ver, text)
        spelled = text.get(side, set())
        for n, m in enumerate(_ID_VALUES.finditer(line)):
            value = m.group(1)
            if value in spelled or n >= _SCAN_MAX:
                return True
            if (_ODD_STR if value[0] == '"' else _ODD_NUM).search(value):
                try:
                    key = _id_key(json.loads(value))
                except ValueError:
                    return True           # cannot tell: parse it
                if any((side, kind, key) in self.done for kind in _KINDS):
                    return True
        return False

    def _lookup(self, key: tuple[str, str, str]) -> str | None:
        hit = self.live.get(key)
        return hit[0] if hit is not None else self.done.get(key)

    def _in_flight(self, side: str) -> str | None:
        n_rule = self.calls.get(side)
        return n_rule[1] if n_rule is not None else None

    def decide(self, hop: str, peer: str | None, state: str, msg: Any,
               inherited: str | None, sent: bool,
               resolved: list[Any] | None = None) -> tuple[str | None, str | None]:
        """(exclusion rule or None, informational rule) for one line, and its bookkeeping.
        `state` is `_parse`'s verdict, or "skip" when the line was not parsed. `resolved`
        is `resolve()` per element of a client_in line, computed before the lock."""
        client = hop in _CLIENT_HOPS
        side = self.side(hop, peer)
        rule = inherited
        if rule is None and not client and peer is not None:
            rule = self.excluded_peers.get(peer)
        if rule is None and hop == "server_in":
            rule = self.unmatched.get(side)
        fail_closed = hop == "server_out" and sent
        if state == "undecidable":
            rule = rule or (RULE_UNDECIDABLE if self.may_exclude else None)
            if rule is not None and fail_closed:
                self.unmatched.setdefault(side, RULE_UNMATCHED)
            return rule, None
        if state == "skip":
            return rule, None
        if state == "bad" or not isinstance(msg, (dict, list)):
            if rule is None and hop == "server_in":
                rule = self._in_flight(side)
            return rule, None
        batch = isinstance(msg, list)
        elems = msg if batch else [msg]
        info = None
        unmatchable = False
        for i, e in enumerate(elems):
            if not isinstance(e, dict):
                if self.may_exclude:
                    rule = rule or RULE_UNDECIDABLE
                unmatchable = True
                continue
            res = resolved[i] if resolved is not None else None
            hit, info_e = self._classify(hop, side, e, res)
            rule = rule or hit
            if not batch:
                info = info_e
            if isinstance(e.get("method"), str) and "id" in e and e["id"] is None:
                unmatchable = True        # a request whose reply carries no usable id
        for e in elems:
            if isinstance(e, dict):
                self._account(hop, side, e, rule, sent)
        if rule is not None and fail_closed and unmatchable:
            self.unmatched.setdefault(side, RULE_UNMATCHED)
        return rule, info

    def _classify(self, hop: str, side: str, e: dict,
                  res: Any) -> tuple[str | None, str | None]:
        mid, method = e.get("id"), e.get("method")
        client_ids = hop in _CLIENT_ID_REQUEST_HOPS
        hit: str | None = None
        if method is None:
            if mid is not None:          # a response: excluded iff its request was
                hit = self._lookup((side, "s" if client_ids else "c", _id_key(mid)))
            elif hop == "server_in":
                hit = self._in_flight(side)
            return hit, None
        if not isinstance(method, str):
            return (self._in_flight(side) if hop == "server_in" else None), None
        params = e.get("params")
        if mid is not None:               # a request reusing a still-excluded id
            hit = self._lookup((side, "c" if client_ids else "s", _id_key(mid)))
        elif isinstance(params, dict):
            if method == "notifications/progress" and "progressToken" in params:
                kind = "ps" if client_ids else "pc"
                hit = self._lookup((side, kind, _id_key(params["progressToken"])))
            elif method == "notifications/cancelled" and "requestId" in params:
                kind = "c" if client_ids else "s"
                hit = self._lookup((side, kind, _id_key(params["requestId"])))
        if hit is None and hop == "server_in":
            hit = self._in_flight(side)
        info = None
        if hop == "client_in" and res is not None:
            info = res[0]
            if res[1] and hit is None:
                hit = res[0] or RULE_UNDECIDABLE
        return hit, info

    def _account(self, hop: str, side: str, e: dict, rule: str | None, sent: bool) -> None:
        mid = e.get("id")
        if mid is None:
            return
        client_ids = hop in _CLIENT_ID_REQUEST_HOPS
        if e.get("method") is None:
            self._complete((side, "s" if client_ids else "c", _id_key(mid)))
            return
        if rule is None or not sent:      # a line discarded unsent is never answered
            return
        kind = "c" if client_ids else "s"
        links: list[tuple[str, str, str]] = []
        params = e.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        token = meta.get("progressToken") if isinstance(meta, dict) else None
        if token is not None:
            links.append((side, "p" + kind, _id_key(token)))
        self._register((side, kind, _id_key(mid)), rule, links)

    def _add(self, key: tuple[str, str, str], rule: str,
             links: list[tuple[str, str, str]]) -> bool:
        if key in self.live:
            return False
        if len(self.live) >= _LIVE_MAX:
            if self.overflow is None:
                self.overflow, self.overflowed = RULE_OVERFLOW, True
            return False
        if self.done.pop(key, None) is None:
            self.sides[key[0]] = self.sides.get(key[0], 0) + 1
        else:
            self._done_ver += 1
        self.live[key] = (rule, links)
        self.live_sides[key[0]] = self.live_sides.get(key[0], 0) + 1
        return True

    def _pop_live(self, key: tuple[str, str, str]) -> Any:
        entry = self.live.pop(key, None)
        if entry is not None:
            n = self.live_sides[key[0]] - 1
            if n:
                self.live_sides[key[0]] = n
            else:
                del self.live_sides[key[0]]
        return entry

    def _register(self, key: tuple[str, str, str], rule: str,
                  links: list[tuple[str, str, str]]) -> None:
        if not self._add(key, rule, links):
            return
        for link in links:
            self._add(link, rule, [])
        side = key[0]
        if key[1] == "c" and side != "c":
            n = self.calls.get(side, (0, rule))[0]
            self.calls[side] = (n + 1, rule)

    def _complete(self, key: tuple[str, str, str]) -> None:
        entry = self._pop_live(key)
        if entry is None:
            return
        rule, links = entry
        self._retire(key, rule)
        for link in links:
            lentry = self._pop_live(link)
            if lentry is not None:
                self._retire(link, lentry[0])
        side = key[0]
        if key[1] == "c" and side in self.calls:
            n, r = self.calls[side]
            if n <= 1:
                del self.calls[side]
            else:
                self.calls[side] = (n - 1, r)

    def _retire(self, key: tuple[str, str, str], rule: str) -> None:
        self.done[key] = rule
        self._done_ver += 1
        while len(self.done) > _DONE_MAX:
            old, _ = self.done.popitem(last=False)
            n = self.sides.get(old[0], 1) - 1
            if n:
                self.sides[old[0]] = n
            else:
                self.sides.pop(old[0], None)

    def peer_gone(self, peer: str | None) -> None:
        """The peer exited: nothing it was asked will be answered any more."""
        side = f"p:{peer}"
        for key in [k for k in self.live if k[0] == side]:
            self._complete(key)
        self.calls.pop(side, None)
        self.unmatched.pop(side, None)


def _segment(value: str) -> str:
    seg = _SAFE_SEGMENT.sub("_", value)[:128]
    return seg if seg.strip(".") else NO_SESSION


def _sha(line: str) -> str:
    try:
        data = line.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:   # a lone surrogate that did not come from surrogateescape
        data = line.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()


def _meta(msg: Any) -> tuple[Any, Any]:
    """(id, method) of a parsed line; both None for anything but an object."""
    if isinstance(msg, dict):
        return msg.get("id"), msg.get("method")
    return None, None


_STOP: Any = object()
_INHERIT: Any = object()


class WireTap:
    def __init__(self, path: Path, fh: TextIO, conn: str, claude_session: str, *,
                 exclude_all: str | None = None,
                 excluded_peers: dict[str, str] | None = None, may_exclude: bool = True,
                 queue_max: int = _QUEUE_MAX, queue_bytes_max: int = _QUEUE_BYTES_MAX,
                 file_bytes_max: int | None = None):
        self.path = path
        self._file_max = _file_bytes_max() if file_bytes_max is None else file_bytes_max
        try:
            self._written = os.fstat(fh.fileno()).st_size   # appending: count what is there
        except (OSError, ValueError, AttributeError):
            self._written = 0
        self._capped = False
        self._fh = fh
        self.conn = conn
        self.claude_session = claude_session
        self.excluded_peers = excluded_peers or {}
        self._scope = _Scope(exclude_all, self.excluded_peers, may_exclude)
        self._ctx = threading.local()
        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_max)
        self._bytes_max = queue_bytes_max
        self._bytes = 0
        self._lock = Lock()
        self._seq = 0
        self._dropped = 0
        self._dead = False
        self._closed = False
        self._warned = False
        self._gate_warned = False
        # Taps not written: after the log turned itself off, and after close() (the
        # latter never get a seq, so `stop` stays the last record and its count exact).
        self.dead_drops = 0
        self.late_drops = 0
        self._thread = Thread(target=self._run, daemon=True, name="terse-wire-log")
        self._thread.start()

    @property
    def resolve(self) -> Resolver | None:
        return self._scope.resolve

    @resolve.setter
    def resolve(self, fn: Resolver | None) -> None:
        """The mode's client-request policy: tools/call (and, under --config, routing)."""
        self._scope.resolve = fn

    def context(self) -> str | None:
        """The exclusion rule of the last line READ on this thread (None: not excluded) —
        captured by a sender that writes on another thread, and passed back as `inherit`."""
        return getattr(self._ctx, "rule", None)

    # ---------- forwarding-thread side ----------

    def tap(self, hop: str, raw: str, *, peer: str | None = None, status: str | None = None,
            origin: str | None = None, inherit: Any = _INHERIT,
            force: str | None = None) -> None:
        """Record one physical line at `hop`. Never raises, never blocks. A write hop
        inherits the exclusion of the read this thread last tapped (or `inherit`); `force`
        excludes the line under that rule regardless."""
        if self._dead:
            with self._lock:
                self.dead_drops += 1
            return
        try:
            line = raw[:-1] if raw.endswith("\n") else raw
            if origin is None:
                origin = getattr(raw, "wire_origin", None)
                if origin is not None and status is None:
                    status = "synthesized"
            read = hop in _READ_HOPS
            if inherit is _INHERIT:
                inherit = None if read else self.context()
            base = force or inherit
            scope = self._scope
            state, msg = "skip", None
            resolved: list[Any] | None = None
            resolve_exc: Exception | None = None
            trivial = scope.overflow or scope.exclude_all
            if trivial is None:
                try:
                    if scope.needs_parse(hop, peer, base, line):
                        # server_out strict too: a duplicate "id" must not register the
                        # key a first-key-wins server will not answer with (#524).
                        state, msg = _parse(line, strict=hop in ("client_in", "server_out"))
                except Exception:  # noqa: BLE001 — fail closed, never into forwarding
                    state = "undecidable"
                # The resolver (routing table, Policy.select) runs before the lock, which
                # every forwarding thread's tap waits on (#525).
                if hop == "client_in" and state == "ok" and scope.resolve is not None:
                    try:
                        resolved = [scope.resolve(e) if isinstance(e, dict) else None
                                    for e in (msg if isinstance(msg, list) else [msg])]
                    except Exception as exc:  # noqa: BLE001 — fail closed, below
                        resolve_exc = exc
            with self._lock:
                if self._closed:
                    self.late_drops += 1
                    return
                rule: str | None
                info: str | None
                if trivial is not None:
                    rule, info = trivial, None
                else:
                    try:
                        if resolve_exc is not None:
                            raise resolve_exc
                        rule, info = scope.decide(hop, peer, state, msg, base,
                                                  sent=status != "dropped",
                                                  resolved=resolved)
                    except Exception as exc:  # noqa: BLE001 — fail closed
                        rule, info = RULE_GATE_ERROR, None
                        self._gate_error(exc)
                if scope.overflowed:
                    scope.overflowed = False
                    self._put_locked(self._event_item(
                        "exclusion_overflow", None, {"live_max": _LIVE_MAX}), 0)
                meta = _meta(msg) if state == "ok" else None
                self._put_locked((hop, peer, line, status, origin, rule, info, meta, None,
                                  None, None), len(line))
            if read:
                self._ctx.rule = rule
        except Exception as exc:  # noqa: BLE001 — the tap is never load-bearing
            self._fail(f"tap failed ({type(exc).__name__}: {exc})")

    def event(self, name: str, *, peer: str | None = None, **detail: Any) -> None:
        if self._dead:
            return
        try:
            with self._lock:
                if self._closed:
                    self.late_drops += 1
                    return
                if name == "peer_exit":
                    self._scope.peer_gone(peer)
                self._put_locked(self._event_item(name, peer, detail), 0)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"event failed ({type(exc).__name__}: {exc})")

    @staticmethod
    def _event_item(name: str, peer: str | None, detail: dict) -> tuple:
        return ("event", peer, None, None, None, None, None, None, time.time_ns(),
                time.monotonic_ns(), (name, detail))

    def _put_locked(self, item: tuple, size: int) -> None:
        if self._dropped and self._try_put(
                self._event_item("sink_drop", None, {"count": self._dropped}), 0):
            self._dropped = 0
        if not self._try_put(item, size):
            self._seq += 1          # the hole IS the evidence; sink_drop explains it
            self._dropped += 1

    def _try_put(self, item: tuple, size: int) -> bool:
        if self._bytes + size > self._bytes_max and self._bytes:
            return False
        # Both clocks are read here, under the lock and as `seq` is taken, so `mono_ns`
        # never decreases in `seq` order (#522). Read earlier, on the producing thread,
        # two threads could stamp in one order and take seqs in the other.
        item = (*item[:8], time.time_ns(), time.monotonic_ns(), item[10])
        try:
            self._q.put_nowait((self._seq, item, size))
        except queue.Full:
            return False
        self._seq += 1
        self._bytes += size
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Write `stop` (and a final `sink_drop`), flush and close. Idempotent. Bounded to
        about a second when the writer is dead."""
        tail: list[tuple] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True     # checked under this lock: no record gets a seq after
            if not self._dead:
                if self._dropped:
                    tail.append((self._seq, self._event_item(
                        "sink_drop", None, {"count": self._dropped}), 0))
                    self._seq += 1
                tail.append((self._seq, self._event_item(
                    "stop", None, {"records": self._seq + 1}), 0))
                self._seq += 1
        if self._dead or not self._thread.is_alive():
            timeout = min(timeout, 1.0)
        # Outside the lock: the writer takes it to settle `_bytes`, so blocking on a full
        # queue while holding it would stall the very thread that drains the queue.
        try:
            for entry in tail:
                self._q.put(entry, timeout=timeout)
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    # ---------- writer thread ----------

    def _run(self) -> None:
        stop = False
        while not stop:
            first = self._q.get()
            batch = [first]
            while len(batch) < _BATCH_MAX:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            entries = []
            freed = 0
            for entry in batch:
                if entry is _STOP:
                    stop = True
                    continue
                entries.append(entry)
                freed += entry[2]
            with self._lock:
                self._bytes -= freed
            if self._dead or self._capped or not entries:
                continue                # keep draining, so close() never waits on us
            try:
                out = self._within_cap([self._render(seq, item) for seq, item, _ in entries],
                                       [seq for seq, _, _ in entries])
                self._fh.write(out)
                self._fh.flush()
            except (OSError, ValueError) as exc:
                self._fail(f"write to {self.path} failed ({exc})")
            except Exception as exc:  # noqa: BLE001 — a writer bug must not go silent
                self._fail(f"writer failed ({type(exc).__name__}: {exc})")
        try:
            self._fh.close()
        except (OSError, ValueError) as exc:
            self._fail(f"write to {self.path} failed ({exc})")

    def _within_cap(self, rendered: list[str], seqs: list[int]) -> str:
        """The prefix of `rendered` that fits under the file cap; past it, a `truncated`
        record in place of the rest, and the log stops for this connection. It takes the
        seq of the first record it replaces: seqs in a batch can have gaps (a dropped tap
        consumes one), so `first + i` could repeat a seq already written."""
        if self._file_max <= 0:
            return "".join(rendered)
        keep: list[str] = []
        for i, rec in enumerate(rendered):
            size = len(rec.encode("utf-8"))
            if self._written + size > self._file_max:
                keep.append(self._render(seqs[i], self._event_item(
                    "truncated", None, {"file_bytes_max": self._file_max})))
                self._capped = True
                sys.stderr.write(f"[terse] wire log {self.path} reached {self._file_max} "
                                 "bytes; the rest of this connection is not logged "
                                 "(TERSE_WIRE_LOG_MAX_BYTES)\n")
                break
            self._written += size
            keep.append(rec)
        return "".join(keep)

    def _fail(self, what: str) -> None:
        self._dead = True
        if self._warned:
            return
        self._warned = True
        sys.stderr.write(f"[terse] wire log {what}; wire log disabled for this connection, "
                         "proxying continues\n")

    def _gate_error(self, exc: BaseException) -> None:
        if self._gate_warned:
            return
        self._gate_warned = True
        sys.stderr.write(f"[terse] wire log exclusion gate raised ({type(exc).__name__}: "
                         f"{exc}); the line is logged as excluded, proxying continues\n")

    def _render(self, seq: int, item: tuple) -> str:
        hop, peer, line, status, origin, rule, info, meta, ts_ns, mono_ns, ev = item
        rec: dict[str, Any] = {"v": SCHEMA_VERSION, "conn": self.conn, "seq": seq,
                               "ts_ns": ts_ns, "mono_ns": mono_ns,
                               "claude_session": self.claude_session, "peer": peer,
                               "hop": hop}
        if ev is not None:
            rec.update(jsonrpc_id=None, method=None, len=None, sha256=None, line=None,
                       capture_status=None, origin=None, rule=None,
                       event=ev[0], detail=ev[1])
            return json.dumps(rec, ensure_ascii=True, default=str) + "\n"
        if meta is None:
            parsed, mid, method = False, None, None
            if line.strip():
                try:
                    mid, method = _meta(json.loads(line))
                    parsed = True
                except (ValueError, RecursionError):
                    pass
        else:
            parsed, (mid, method) = True, meta
        excluded = rule is not None
        if excluded:
            status = "excluded_by_policy"
            if isinstance(mid, bool) or not isinstance(mid, (str, int, float, type(None))):
                mid = None              # not a JSON-RPC id: could be anything
        elif status is None:
            status = "full" if parsed else "parse_error"
        rec.update(jsonrpc_id=mid, method=method if isinstance(method, str) else None,
                   len=len(line), sha256=None if excluded else _sha(line),
                   line=None if excluded else line, capture_status=status, origin=origin,
                   rule=rule if excluded else info)
        return json.dumps(rec, ensure_ascii=True, default=str) + "\n"


def _file_bytes_max() -> int:
    raw = os.environ.get("TERSE_WIRE_LOG_MAX_BYTES")
    if raw is None:
        return _FILE_BYTES_MAX
    try:
        return int(raw)
    except ValueError:
        sys.stderr.write(f"[terse] TERSE_WIRE_LOG_MAX_BYTES={raw!r} is not an integer; "
                         f"using {_FILE_BYTES_MAX}\n")
        return _FILE_BYTES_MAX


def may_exclude(policies: Any) -> bool:
    """Whether any rule in `policies` has capture:false — without one, nothing can be
    excluded, and an unparseable line is logged like any other."""
    return any(not r.capture for p in policies for r in p.rules)


def _own_dir(path: Path, *, managed: bool) -> None:
    """Create `path` 0700 if missing, then refuse it unless this user owns it (#522). The
    file is opened O_NOFOLLOW/0600 either way; this keeps a pre-planted directory on a
    shared host from collecting the log's names. `DIR` is the operator's: a symlink to a
    directory they own is fine and its mode is theirs. The session directory below it is
    terse's alone, so a symlink there is refused and a loose mode is tightened."""
    mkdir_restricted(path)
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        if managed:
            raise OSError(errno.ELOOP, "is a symlink", str(path))
        st = os.stat(path)
    getuid = getattr(os, "getuid", None)        # absent on Windows: no owner to check
    if getuid is not None and st.st_uid != getuid():
        raise PermissionError(errno.EPERM, "owned by another user", str(path))
    if managed and stat.S_IMODE(st.st_mode) & 0o077:
        os.chmod(path, 0o700)


def open_wire_tap(root: str, conn: str, *, exclude_all: str | None = None,
                  excluded_peers: dict[str, str] | None = None,
                  **kw: Any) -> WireTap | None:
    """Build the tap for one proxy process, or None (after one stderr warning) when the
    directory or file cannot be opened — the proxy then runs exactly as without it."""
    session = os.environ.get("CLAUDE_CODE_SESSION_ID") or NO_SESSION
    path = Path(root) / _segment(session) / f"{_segment(conn)}.jsonl"
    try:
        _own_dir(Path(root), managed=False)
        _own_dir(path.parent, managed=True)
        fh = open_append_restricted(path)
    except OSError as exc:
        sys.stderr.write(f"[terse] wire log disabled: cannot open {path} ({exc}); "
                         "proxying continues\n")
        return None
    return WireTap(path, fh, conn, session, exclude_all=exclude_all,
                   excluded_peers=excluded_peers, **kw)
