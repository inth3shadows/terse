"""The wire log (`proxy --wire-log DIR`): a lossless, append-only record of every JSON-RPC
line crossing terse, at all four hops, so an agent's MCP observations — the original a
server sent AND what the model was actually given — can be rebuilt afterwards and joined to
the client's own transcript.

One file per proxy process: `DIR/<CLAUDE_CODE_SESSION_ID or no-session>/<conn>.jsonl`,
dirs 0700, file 0600. One JSON object per line; see TECHNICAL.md "Wire log" for the schema.

Completeness, not volume, is the point, so the design is about never losing a line
silently and never breaking forwarding:
- `tap()` runs on the forwarding threads and does only O(1) work (two clocks, a lock, a
  queue put). Parsing, hashing and serializing happen on ONE writer thread, which also keeps
  the file in `seq` order.
- `seq` is assigned under the lock for every record, kept or not. A full queue drops the
  record (its seq becomes a hole) and counts it; the count is written as a `sink_drop` event
  as soon as there is room, so a gap is always both visible and explained.
- An OSError on open or write warns once on stderr and disables the tap; the proxy carries
  on exactly as without `--wire-log`.

Exclusion (`capture: false`) is decided on the forwarding thread, at request time, by the
caller's gate, and registered synchronously in `_excluded_ids` — never via the queue, so a
dropped request record cannot let its secret-bearing response through in full. Ids stay
excluded until evicted (a later reuse of the id is excluded too: the safe direction).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from typing import Any, NamedTuple, TextIO

from ._secure_io import mkdir_restricted, open_append_restricted

SCHEMA_VERSION = 1
NO_SESSION = "no-session"
_QUEUE_MAX = 16384
_QUEUE_BYTES_MAX = 64 * 1024 * 1024
_EXCLUDED_IDS_MAX = 4096
_BATCH_MAX = 512
# A tool name no real server exports: what `Policy.select` returns for it is the rule an
# UNKNOWN tool on that server falls to — i.e. the server's default. capture:false there
# (secret-broker's `secret-broker.*` default-deny) excludes the whole server.
_PROBE_TOOL = "\x00terse-wire-probe"
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")


class Gate(NamedTuple):
    """What the caller decided about one line at request time."""
    rule: str | None       # the matching policy rule's glob, if one applies
    excluded: bool
    id_key: str | None     # the request's id (as `_id_key`), to exclude its responses


def _id_key(mid: Any) -> str:
    return json.dumps(mid, sort_keys=True)


def server_excluded(policy: Any, server: str | None) -> str | None:
    """The rule glob when `server`'s default is capture:false (the whole server is
    excluded), else None."""
    rule = policy.select(_PROBE_TOOL, server)
    return None if rule.capture else rule.tool_glob


def call_gate(line: str, resolve: Callable[[str, dict], tuple[str | None, bool] | None],
              ) -> Gate | None:
    """Parse a client line only when it can be a tools/call or prompts/get (a substring
    test first keeps every other line free), and ask `resolve(method, params)` for its
    (rule, excluded). None = no decision; the writer falls back to the id map."""
    if "tools/call" not in line and "prompts/get" not in line:
        return None
    try:
        msg = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(msg, dict):
        return None
    method, params = msg.get("method"), msg.get("params")
    if method not in ("tools/call", "prompts/get") or not isinstance(params, dict):
        return None
    res = resolve(method, params)
    if res is None:
        return None
    mid = msg.get("id")
    return Gate(res[0], res[1], None if mid is None else _id_key(mid))


def retrieve_handle(params: dict) -> str | None:
    args = params.get("arguments")
    handle = args.get("handle") if isinstance(args, dict) else None
    return handle if isinstance(handle, str) else None


def _segment(value: str) -> str:
    seg = _SAFE_SEGMENT.sub("_", value)[:128]
    return seg if seg.strip(".") else NO_SESSION


def _sha(line: str) -> str:
    try:
        data = line.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:   # a lone surrogate that did not come from surrogateescape
        data = line.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()


_STOP: Any = object()


class WireTap:
    def __init__(self, path: Path, fh: TextIO, conn: str, claude_session: str, *,
                 exclude_all: str | None = None,
                 excluded_peers: dict[str, str] | None = None,
                 queue_max: int = _QUEUE_MAX, queue_bytes_max: int = _QUEUE_BYTES_MAX):
        self.path = path
        self._fh = fh
        self.conn = conn
        self.claude_session = claude_session
        self._exclude_all = exclude_all
        self.excluded_peers = excluded_peers or {}
        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_max)
        self._bytes_max = queue_bytes_max
        self._bytes = 0
        self._lock = Lock()
        self._seq = 0
        self._dropped = 0
        self._excluded_ids: OrderedDict[str, str | None] = OrderedDict()
        self._dead = False
        self._closed = False
        self._thread = Thread(target=self._run, daemon=True, name="terse-wire-log")
        self._thread.start()

    # ---------- forwarding-thread side ----------

    def tap(self, hop: str, raw: str, *, peer: str | None = None, status: str | None = None,
            origin: str | None = None, gate: Gate | None = None) -> None:
        """Record one physical line at `hop`. Never raises, never blocks."""
        if self._dead or self._closed:
            return
        try:
            line = raw[:-1] if raw.endswith("\n") else raw
            if origin is None:
                origin = getattr(raw, "wire_origin", None)
                if origin is not None and status is None:
                    status = "synthesized"
            item = (hop, peer, line, status, origin, gate, time.time_ns(),
                    time.monotonic_ns(), None)
            with self._lock:
                if gate is not None and gate.excluded and gate.id_key is not None:
                    self._excluded_ids[gate.id_key] = gate.rule
                    self._excluded_ids.move_to_end(gate.id_key)
                    while len(self._excluded_ids) > _EXCLUDED_IDS_MAX:
                        self._excluded_ids.popitem(last=False)
                self._put_locked(item, len(line))
        except Exception:  # noqa: BLE001 — the tap is never load-bearing
            self._dead = True

    def event(self, name: str, *, peer: str | None = None, **detail: Any) -> None:
        if self._dead or self._closed:
            return
        try:
            with self._lock:
                self._put_locked(self._event_item(name, peer, detail), 0)
        except Exception:  # noqa: BLE001
            self._dead = True

    @staticmethod
    def _event_item(name: str, peer: str | None, detail: dict) -> tuple:
        return ("event", peer, None, None, None, None, time.time_ns(), time.monotonic_ns(),
                (name, detail))

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
        try:
            self._q.put_nowait((self._seq, item, size))
        except queue.Full:
            return False
        self._seq += 1
        self._bytes += size
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Write `stop` (and a final `sink_drop`), flush and close. Idempotent."""
        if self._closed:
            return
        tail: list[tuple] = []
        with self._lock:
            self._closed = True     # no producer after this, so seq order is the put order
            if not self._dead:
                if self._dropped:
                    tail.append((self._seq, self._event_item(
                        "sink_drop", None, {"count": self._dropped}), 0))
                    self._seq += 1
                tail.append((self._seq, self._event_item(
                    "stop", None, {"records": self._seq + 1}), 0))
                self._seq += 1
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
            out: list[str] = []
            freed = 0
            for entry in batch:
                if entry is _STOP:
                    stop = True
                    continue
                seq, item, size = entry
                freed += size
                if not self._dead:
                    out.append(self._render(seq, item))
            with self._lock:
                self._bytes -= freed
            if out and not self._dead:
                try:
                    self._fh.write("".join(out))
                    self._fh.flush()
                except (OSError, ValueError) as exc:
                    self._fail(exc)
        try:
            self._fh.close()
        except (OSError, ValueError) as exc:
            if not self._dead:
                self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        self._dead = True
        sys.stderr.write(f"[terse] wire log write to {self.path} failed ({exc}); wire log "
                         "disabled for this connection, proxying continues\n")

    def _render(self, seq: int, item: tuple) -> str:
        hop, peer, line, status, origin, gate, ts_ns, mono_ns, ev = item
        rec: dict[str, Any] = {"v": SCHEMA_VERSION, "conn": self.conn, "seq": seq,
                               "ts_ns": ts_ns, "mono_ns": mono_ns,
                               "claude_session": self.claude_session, "peer": peer,
                               "hop": hop}
        if ev is not None:
            rec.update(jsonrpc_id=None, method=None, len=None, sha256=None, line=None,
                       capture_status=None, origin=None, rule=None,
                       event=ev[0], detail=ev[1])
            return json.dumps(rec, ensure_ascii=True, default=str) + "\n"
        mid: Any = None
        method: Any = None
        parsed = False
        if line.strip():
            try:
                msg = json.loads(line)
                parsed = True
            except (ValueError, RecursionError):
                msg = None
            if isinstance(msg, dict):
                mid, method = msg.get("id"), msg.get("method")
        rule: str | None = None
        excluded = False
        if self._exclude_all is not None:
            excluded, rule = True, self._exclude_all
        elif peer is not None and peer in self.excluded_peers:
            excluded, rule = True, self.excluded_peers[peer]
        elif gate is not None:
            excluded, rule = gate.excluded, gate.rule
        if not excluded and mid is not None and not (hop == "client_in" and gate is not None):
            key = _id_key(mid)
            with self._lock:
                if key in self._excluded_ids:
                    excluded, rule = True, self._excluded_ids[key]
        if excluded:
            status = "excluded_by_policy"
        elif status is None:
            status = "full" if parsed else "parse_error"
        rec.update(jsonrpc_id=mid, method=method if isinstance(method, str) else None,
                   len=len(line), sha256=None if excluded else _sha(line),
                   line=None if excluded else line, capture_status=status, origin=origin,
                   rule=rule)
        return json.dumps(rec, ensure_ascii=True, default=str) + "\n"


def open_wire_tap(root: str, conn: str, *, exclude_all: str | None = None,
                  excluded_peers: dict[str, str] | None = None,
                  **kw: Any) -> WireTap | None:
    """Build the tap for one proxy process, or None (after one stderr warning) when the
    directory or file cannot be opened — the proxy then runs exactly as without it."""
    session = os.environ.get("CLAUDE_CODE_SESSION_ID") or NO_SESSION
    path = Path(root) / _segment(session) / f"{_segment(conn)}.jsonl"
    try:
        mkdir_restricted(root)
        mkdir_restricted(path.parent)
        fh = open_append_restricted(path)
    except OSError as exc:
        sys.stderr.write(f"[terse] wire log disabled: cannot open {path} ({exc}); "
                         "proxying continues\n")
        return None
    return WireTap(path, fh, conn, session, exclude_all=exclude_all,
                   excluded_peers=excluded_peers, **kw)
