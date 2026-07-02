"""MCP proxy: compress downstream tool-call results per policy, transparently.

Sits between an MCP client (e.g. Claude Code) and one downstream MCP server, which it
reaches over a pluggable `Transport` (`transport.py`, #5): a local stdio subprocess, or
an MCP Streamable-HTTP endpoint. Either way it forwards JSON-RPC both ways. The ONLY
thing it changes is the text of a `tools/call` *result*, which it runs through
`policy.apply()` using the tool name recorded from the matching request.

Design guarantees:
  - Transparent: every non-(tools/call-result) message is forwarded byte-for-byte.
  - Fail-open: any parse/compress error forwards the ORIGINAL message. A compression
    layer must never lose or corrupt a tool result.
  - Frame-safe: MCP messages are newline-delimited JSON on the wire (stdio lines, or one
    JSON-RPC message per SSE event over HTTP); terse minified output has no embedded
    newlines, so a compressed result stays one line/event.
  - Transport-independent: `Interceptor` and `pump()` operate on line-in/line-out only —
    neither knows or cares whether the downstream is a subprocess or an HTTP peer.

The pure message logic lives in `Interceptor` (unit-tested without any I/O). The
`run_proxy` shell wires it to a `Transport` with two pump threads.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from collections import OrderedDict
from threading import Lock, Thread
from typing import Any, Callable, Optional, TextIO

from . import lossy as lossy_mod
from . import policy as policy_mod
from . import text_diff
from . import transforms
from .tokenize import count_cl100k
from .transport import HttpTransport, build_transport

# The synthetic tool terse advertises in tools/list when a policy enables drop-to-retrieve
# (#10). The proxy answers its calls itself from the drop store — the downstream server
# never sees it.
RETRIEVE_TOOL_DEF = {
    "name": lossy_mod.RETRIEVE_TOOL,
    "description": ("Fetch the full original value of a field terse dropped from an earlier "
                    "tool result to save context. Pass the handle string shown in the field's "
                    f"{lossy_mod.DROP_KEY!r} marker; returns the exact original value."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "handle": {"type": "string",
                       "description": f"The handle from a {lossy_mod.DROP_KEY!r} marker."},
        },
        "required": ["handle"],
    },
}


def _cost(text: str) -> int:
    """Token cost, falling back to byte length where tiktoken is unavailable."""
    c = count_cl100k(text)
    return c if c is not None else len(text)


# A one-time, system-level explanation of terse's wire forms, injected into the MCP
# `initialize` result's `instructions` field (#13). Measurement showed a *system-level*
# primer recovers comprehension that an inline per-result note cannot (the stdio proxy
# can't set a system prompt); `instructions` is the channel clients add to that context.
# Covers the always-on table/dict forms AND the opt-in diff form, so it helps base
# comprehension too — paid once per session, not per result.
TERSE_PRIMER = (
    "Some tool results are 'terse'-compressed (a lossless, denser JSON encoding); some "
    "are sent as diffs against the previous result of the same tool. Read each as the "
    "equivalent full JSON:\n"
    '- Table {"__terse_table__":1,"n":N,"cols":[...],"rows":[[...]]}: N records, each row '
    'POSITIONAL — its i-th value belongs to the i-th name in "cols". "n" is the exact count.\n'
    '- Dict {"__terse_dict__":1,"legend":{"~0":value,...},"data":...}: every "~K" token '
    'inside "data" stands for legend["~K"] — substitute it back.\n'
    '- Diff {"__terse_diff__":1,"shape":"rows","by":COL,"set":[...],"new":[...],"del":[...],'
    '"n":N}: update the PREVIOUS same-tool result — from its records drop ids in "del", '
    'overwrite/insert each record in "set" matched by its "by" field, append ids in "new"; '
    '"n" is the final record count. A {"shape":"keys","set":{...},"del":[...]} diff instead '
    'removes "del" keys and applies "set" key/values to the previous object. '
    'A text diff {"__terse_textdiff__":1,"ops":[["=",a,b],["+","..."],...]} updates the '
    "PREVIOUS same-tool plain-text result: process ops in order, copying chunks a..b of "
    "that prior text for a `=` op or inserting its literal string for a `+` op, then "
    "concatenating everything.\n"
    '- Dropped field {"__terse_dropped__":"H","bytes":N,"retrieve":"terse.retrieve"}: a '
    "large field value was omitted to save context. It is NOT lost — when you actually need "
    'it, call the terse.retrieve tool with {"handle":"H"} to get the exact original back.\n'
    "Always reason about the fully reconstructed result."
)


class Interceptor:
    """Pure JSON-RPC message logic. Tracks request id -> tool name and compresses
    matching results. No I/O; both methods take and return a single line of text
    (without the trailing newline).

    When `policy.diff` is on, it also keeps the previous per-tool result and emits a
    lossless delta when that is smaller than the full compressed form — the stateful
    cross-call lever. It is fail-open and self-verifying: a diff is sent only when it
    provably reconstructs the result, and the full form is always the fallback. JSON and
    non-JSON (text/log/file) results each get their own diff base and codec (#25) so a
    tool that alternates between the two never mixes bases across shapes."""

    # Cap on in-flight request ids tracked at once. A tools/call that times out with no
    # result body never gets popped from `pending` (#22), so bound the map and evict
    # oldest-first: a long session against a flaky server can't leak unboundedly. An
    # evicted id whose result arrives late just forwards uncompressed — safe, fail-open.
    PENDING_MAX = 1024
    # drop-to-retrieve store bounds (#10): retain at most this many distinct handles AND at
    # most this many bytes of stored originals, evicting least-recently-used first. A dropped
    # field the model never retrieves before eviction just fails its retrieve legibly (Phase
    # 3) — fail-open, never a crash. Both caps guard a long session from unbounded growth.
    DROPPED_MAX = 512
    DROPPED_MAX_BYTES = 8 << 20  # 8 MiB

    def __init__(self, pol: policy_mod.Policy, debug: bool = False,
                 capture: Optional[Callable[[str, str], None]] = None,
                 audit: Optional[Callable[[dict], None]] = None,
                 store: Optional["OrderedDict[str, Any]"] = None,
                 store_lock: Optional[Lock] = None,
                 dropped_bytes: Optional[list[int]] = None):
        self.policy = pol
        self.pending: dict[Any, str] = {}
        self.debug = debug
        self.diff = pol.diff
        # Optional tee of each RAW (pre-compression) tool-result text, keyed by tool name
        # (#32). Keeps the Interceptor I/O-free: the callback owns the disk write. Never
        # affects forwarding — its failures are swallowed at the call site.
        self.capture = capture
        # Optional structured replay log of the raw->decision->emitted triple per result
        # (#23). Like capture, the callback owns I/O and its failures are swallowed: an
        # audit-log write must NEVER change what the client receives.
        self.audit = audit
        self.last: dict[str, Any] = {}  # tool -> previous result object (the diff base)
        # tool -> consecutive diffs emitted since the last full (keyframe) result. Bounds
        # how far a chained diff can drift from a self-contained anchor (#8).
        self.keyframe_interval = pol.diff_keyframe_interval
        self.since_keyframe: dict[str, int] = {}
        # Same two roles as `last`/`since_keyframe` but for non-JSON payloads (#25):
        # the CDC text diff (Tier 0.7 text) needs its own prior-text base, since a
        # non-JSON result never populates `last` (there is no JSON object to diff).
        self.last_text: dict[str, str] = {}
        self.since_text_keyframe: dict[str, int] = {}
        # drop-to-retrieve store (#10): handle -> original field value, filled when a field
        # marked drop-to-retrieve is replaced inline by a handle, and read back by the
        # synthetic terse.retrieve tool. LRU-ordered; bounded by DROPPED_MAX / _MAX_BYTES;
        # cleared on reconnect (like the diff bases) since the model's context — and thus
        # every emitted handle — resets then too.
        #
        # `store`/`store_lock` (#5 Half B): when the caller passes them (multiproxy.py
        # fronting N peers), this Interceptor shares its drop store + lock with every
        # OTHER peer's Interceptor instead of keeping a private one. That is safe because
        # handles are content-addressed and include the bare tool name (lossy._handle) —
        # two peers dropping different values never collide, and equal values dedupe into
        # one slot — so one shared store serves terse.retrieve correctly regardless of
        # which peer answers it. Default (None) is 100% behavior-preserving for every
        # existing single-peer caller: a fresh private OrderedDict + Lock, exactly as
        # before this parameter existed.
        self.dropped: "OrderedDict[str, Any]" = store if store is not None else OrderedDict()
        # `dropped_bytes` (#5 Half B): a 1-element box, not a plain int, specifically so
        # it can be SHARED the same way `store` is. `self.dropped` can be one dict shared
        # across N Interceptors, but a plain `self._dropped_bytes = 0` would still be
        # per-instance — each peer would only ever see bytes IT personally inserted, so
        # the DROPPED_MAX_BYTES eviction check would never fire against the shared dict's
        # TRUE combined size. A shared box keeps the byte tally as cross-peer-accurate as
        # the dict it's tracking. Default (None) is behavior-preserving: a fresh private
        # box, exactly equivalent to a private int.
        self._dropped_bytes_box: list[int] = dropped_bytes if dropped_bytes is not None else [0]
        self.init_id: Any = None        # id of the initialize request, to prime its reply
        # The two proxy pump threads call note_request (client->server) and
        # transform_response (server->client) concurrently, both mutating the shared
        # pending/last/since_keyframe state. One lock serializes each method so the
        # compound eviction + the reconnect reset can't race a response in flight.
        # When `store_lock` is injected, this Interceptor's ENTIRE critical section
        # (not just the drop store) serializes against every other peer sharing that
        # lock too — a peer's own pending/last/since_keyframe state doesn't strictly
        # need cross-peer exclusion, but `self.dropped` does (it's the same physical
        # dict), and `_drop_put` already assumes it's called under `self._lock` (see its
        # docstring) — so reusing one lock for the whole critical section is the minimal,
        # provably-safe change rather than carving out a second, finer-grained lock.
        self._lock = store_lock if store_lock is not None else Lock()

    def note_request(self, line: str) -> None:
        """Record id -> tool name for tools/call requests, and the initialize request id
        (so its reply can carry the format primer). Side-effect only."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        method = msg.get("method")
        with self._lock:
            if method == "initialize":
                # A re-handshake means the client rebuilt its MCP connection — and almost
                # certainly its context window — so the model no longer holds any prior
                # result a diff could reference. Drop every diff base so each tool
                # re-anchors as a full, guarding against a silently-unresolvable delta
                # after a client-side context reset (#20). Also drop pending: a stale
                # pre-reconnect id could otherwise collide with a reused id and mis-route
                # a late response to the wrong tool's codec. Context COMPACTION without a
                # reconnect is unobservable over stdio; that residual risk is why --diff
                # stays opt-in.
                self.last.clear()
                self.since_keyframe.clear()
                self.last_text.clear()
                self.since_text_keyframe.clear()
                self.pending.clear()
                self.dropped.clear()
                self._dropped_bytes_box[0] = 0
                if mid is not None:
                    self.init_id = mid
                return
            if method != "tools/call":
                return
            name = (msg.get("params") or {}).get("name")
            if mid is not None and isinstance(name, str):
                self.pending[mid] = name
                # dict preserves insertion order; drop the oldest tracked id(s) once over
                # cap so abandoned (timed-out) entries can't accumulate (#22). Safe under
                # the lock — no concurrent mutation during the iterate-then-pop.
                while len(self.pending) > self.PENDING_MAX:
                    self.pending.pop(next(iter(self.pending)))

    def transform_response(self, line: str) -> str:
        """Compress the text of a tracked tools/call result; prime the initialize reply;
        else return unchanged."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line
        if not isinstance(msg, dict) or "result" not in msg or msg.get("id") is None:
            return line
        # Held across the whole body so the shared init_id/pending/last/since_keyframe
        # state stays consistent against a concurrent note_request on the other thread.
        with self._lock:
            if msg["id"] == self.init_id:
                self.init_id = None  # one-time
                primed = self._augment_initialize(msg)
                return primed if primed is not None else line
            tool = self.pending.pop(msg["id"], None)
            if tool is None:
                # Not a tracked tools/call response. When a policy enables drop-to-retrieve,
                # a tools/list reply is where we advertise the synthetic terse.retrieve tool
                # so the model knows how to fetch a dropped field back (#10). Anything else
                # (and any non-tools/list message) forwards unchanged.
                if self.policy.has_drop():
                    injected = self._inject_retrieve_tool(msg)
                    if injected is not None:
                        return injected
                return line  # not a tracked tools/call response (tools/list, ...)

            result = msg.get("result")
            content = result.get("content") if isinstance(result, dict) else None
            if not isinstance(content, list):
                return line

            text_blocks = [b for b in content
                           if isinstance(b, dict) and b.get("type") == "text"
                           and isinstance(b.get("text"), str)]

            # Tee the RAW payload before any compression touches it (#32). Strictly a side
            # effect: a capture failure must NEVER affect what the client receives, so it
            # is swallowed here regardless of what the callback does.
            if self.capture is not None:
                for b in text_blocks:
                    try:
                        self.capture(tool, b["text"])
                    except Exception as exc:  # noqa: BLE001 — capture is never load-bearing
                        if self.debug:
                            sys.stderr.write(f"[terse-proxy] {tool}: capture skipped: {exc}\n")

            # Snapshot the raw block texts before any transform mutates them in place, so
            # the audit log can pair each raw payload with what terse actually emitted (#23).
            raw_texts = [b["text"] for b in text_blocks] if self.audit is not None else None

            changed = False
            # Diffing reasons about ONE logical payload, so it only engages for a single
            # text block (the overwhelmingly common tool-result shape); multi-block results
            # take the plain per-block compression path.
            if self.diff and len(text_blocks) == 1:
                changed = self._compress_or_diff(text_blocks[0], tool)
            else:
                for block in text_blocks:
                    new_text = self._compress(block["text"], tool)
                    if new_text != block["text"]:
                        block["text"] = new_text
                        changed = True

            # Audit AFTER the transform, regardless of `changed`: a no-op is itself
            # diagnostic — it confirms terse left a suspect payload untouched.
            if self.audit is not None and raw_texts is not None:
                self._emit_audit(tool, msg["id"], raw_texts, text_blocks, changed)

            if not changed:
                return line
            # Re-serialize compactly. JSON-RPC is semantics, not formatting; no newlines.
            return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def _compress_or_diff(self, block: dict, tool: str) -> bool:
        """Compress one block, preferring a lossless delta vs the prior same-tool result
        when it is smaller. Updates the per-tool diff base. Returns whether the block
        text changed. Fail-open: any error leaves the block untouched and state intact."""
        text = block["text"]
        try:
            applied = policy_mod.apply(text, tool, self.policy, drop_sink=self._drop_put)
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: passthrough on error: {exc}\n")
            return False
        if applied.skipped:
            # Skipped = a passthrough tool (empty tiers) OR a non-JSON result (e.g. an
            # upstream error string, a file read, a log tail) for a normally-compressed
            # one. Either way it carries no JSON the next JSON diff could build on, and
            # it becomes the model's visible "previous same-tool result" — so drop any
            # stale JSON diff base and reset its keyframe counter, forcing the next JSON
            # result to re-anchor as a full (#8).
            self.last.pop(tool, None)
            self.since_keyframe.pop(tool, None)
            if not self.policy.select(tool).tiers:
                return False  # true passthrough policy: hands off entirely, no state kept
            return self._text_diff_or_store(block, tool, text)

        chosen = applied.text
        try:
            curr = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            curr = None
        if curr is not None:
            prev = self.last.get(tool)
            emitted_diff = False
            # A keyframe is due once K diffs have chained off the last full result; force
            # the full compressed form so the chain re-anchors (#8). interval 0 = never.
            keyframe_due = (self.keyframe_interval > 0
                            and self.since_keyframe.get(tool, 0) >= self.keyframe_interval)
            if prev is not None and not keyframe_due:
                wire = self._diff_wire(prev, curr, tool)
                if wire is not None and _cost(wire) < _cost(applied.text):
                    chosen = wire
                    emitted_diff = True
                    if self.debug:
                        sys.stderr.write(
                            f"[terse-proxy] {tool}: diff {_cost(applied.text)}->{_cost(wire)} "
                            f"tok vs full compressed\n")
            if self.debug and keyframe_due:
                sys.stderr.write(f"[terse-proxy] {tool}: keyframe (full) after "
                                 f"{self.since_keyframe.get(tool, 0)} diffs\n")
            # A diff extends the chain; any full result (no prior, diff lost, or keyframe)
            # is a fresh anchor and resets the counter.
            self.since_keyframe[tool] = self.since_keyframe.get(tool, 0) + 1 if emitted_diff else 0
            # Base the NEXT diff on the true current value, whichever form we emit:
            # the model's reconstructable state after this turn is `curr` either way.
            self.last[tool] = curr

        if chosen != text:
            block["text"] = chosen
            return True
        return False

    def _augment_initialize(self, msg: dict) -> Optional[str]:
        """Prepend the terse format primer to the initialize result's `instructions` (#13),
        preserving any the downstream server set. Idempotent. Returns the reserialized
        line, or None to forward unchanged."""
        result = msg.get("result")
        if not isinstance(result, dict):
            return None
        existing = result.get("instructions")
        existing = existing if isinstance(existing, str) else ""
        if TERSE_PRIMER in existing:
            return None
        result["instructions"] = TERSE_PRIMER + (f"\n\n{existing}" if existing else "")
        if self.debug:
            sys.stderr.write("[terse-proxy] injected terse format primer into "
                             "initialize.instructions\n")
        return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def _diff_wire(self, prev: Any, curr: Any, tool: str) -> Optional[str]:
        """The on-the-wire diff envelope, or None if no lossless diff applies. Self-
        describing: it names the prior result (already in the model's context) and
        carries the changes inline, so the model reconstructs without an out-of-band
        retrieve. Shared with the fluency-for-diff eval via `transforms.diff_wire`."""
        try:
            return transforms.diff_wire(prev, curr, tool)
        except Exception:  # noqa: BLE001 — fail-open
            return None

    def _text_diff_or_store(self, block: dict, tool: str, text: str) -> bool:
        """Tier 0.7 text (#25): CDC-diff a non-JSON result against this tool's own prior
        non-JSON result, when diffing is on. Same fail-open/self-verifying/keyframe
        contract as the JSON diff path — a diff is sent only when it provably
        reconstructs the text AND is smaller than the raw payload; the raw text is
        always the fallback, and every Kth result re-anchors as a full (#8)."""
        if not self.diff:
            return False
        prev_text = self.last_text.get(tool)
        keyframe_due = (self.keyframe_interval > 0
                        and self.since_text_keyframe.get(tool, 0) >= self.keyframe_interval)
        changed = False
        if prev_text is not None and not keyframe_due:
            wire = self._text_diff_wire(prev_text, text, tool)
            if wire is not None and _cost(wire) < _cost(text):
                block["text"] = wire
                changed = True
                if self.debug:
                    sys.stderr.write(f"[terse-proxy] {tool}: text diff {_cost(text)}->"
                                     f"{_cost(wire)} tok vs raw\n")
        self.since_text_keyframe[tool] = self.since_text_keyframe.get(tool, 0) + 1 if changed else 0
        self.last_text[tool] = text
        return changed

    def _text_diff_wire(self, prev: str, curr: str, tool: str) -> Optional[str]:
        """Fail-open wrapper mirroring `_diff_wire`, for the CDC text-diff codec."""
        try:
            return text_diff.text_diff_wire(prev, curr, tool)
        except Exception:  # noqa: BLE001 — fail-open
            return None

    def _compress(self, text: str, tool: str) -> str:
        """policy.apply with a hard fail-open: any error returns the original text."""
        try:
            applied = policy_mod.apply(text, tool, self.policy, drop_sink=self._drop_put)
            if self.debug and not applied.skipped and applied.text != text:
                sys.stderr.write(
                    f"[terse-proxy] {tool}: {len(text)}->{len(applied.text)} bytes "
                    f"(tiers={list(applied.tiers)})\n"
                )
            return applied.text
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: passthrough on error: {exc}\n")
            return text

    def _drop_put(self, handle: str, value: Any) -> None:
        """Store a dropped field's original under `handle` for a later terse.retrieve (#10).
        LRU: re-inserting an existing handle refreshes its recency; once over the count or
        byte cap, evict oldest-first. Called from apply() inside transform_response, which
        already holds self._lock — no separate lock needed."""
        size = len(lossy_mod._serialize(value))
        if handle in self.dropped:
            self._dropped_bytes_box[0] -= len(lossy_mod._serialize(self.dropped[handle]))
            self.dropped.move_to_end(handle)
        self.dropped[handle] = value
        self._dropped_bytes_box[0] += size
        while self.dropped and (len(self.dropped) > self.DROPPED_MAX
                                or self._dropped_bytes_box[0] > self.DROPPED_MAX_BYTES):
            _, evicted = self.dropped.popitem(last=False)
            self._dropped_bytes_box[0] -= len(lossy_mod._serialize(evicted))

    def _inject_retrieve_tool(self, msg: dict) -> Optional[str]:
        """If `msg` is a tools/list result, append the synthetic terse.retrieve tool so the
        model can fetch a drop-to-retrieve field back by handle (#10). Idempotent. Returns
        the reserialized line, or None to forward unchanged (not a tools/list, or already
        advertised)."""
        result = msg.get("result")
        if not isinstance(result, dict):
            return None
        tools = result.get("tools")
        if not isinstance(tools, list):
            return None
        if any(isinstance(t, dict) and t.get("name") == lossy_mod.RETRIEVE_TOOL for t in tools):
            return None  # already present — idempotent across re-lists
        tools.append(RETRIEVE_TOOL_DEF)
        if self.debug:
            sys.stderr.write(f"[terse-proxy] injected {lossy_mod.RETRIEVE_TOOL} into tools/list\n")
        return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def answer_retrieve(self, line: str) -> Optional[str]:
        """If `line` is a client tools/call for the synthetic terse.retrieve tool, produce the
        JSON-RPC reply here — from the drop store — instead of forwarding it downstream, which
        has no such tool (#10). Returns the reply line to write back to the client, or None if
        this isn't a retrieve call. A miss (evicted, or a handle from before a reconnect) is a
        legible error result, never a protocol error — the model can just re-run the tool."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(msg, dict) or msg.get("method") != "tools/call":
            return None
        params = msg.get("params") or {}
        if params.get("name") != lossy_mod.RETRIEVE_TOOL:
            return None
        mid = msg.get("id")
        handle = (params.get("arguments") or {}).get("handle")
        value = None
        with self._lock:
            hit = handle in self.dropped
            if hit:
                self.dropped.move_to_end(handle)  # a read refreshes recency
                value = self.dropped[handle]
        if hit:
            result: dict = {"content": [{"type": "text", "text": lossy_mod._serialize(value)}]}
        else:
            result = {"content": [{"type": "text",
                                   "text": (f"terse: dropped-field handle {handle!r} is no "
                                            "longer available (evicted, or the session "
                                            "reconnected). Re-run the original tool to get "
                                            "the value again.")}],
                      "isError": True}
        if self.debug:
            sys.stderr.write(f"[terse-proxy] answered {lossy_mod.RETRIEVE_TOOL} "
                             f"handle={handle!r} hit={hit}\n")
        return json.dumps({"jsonrpc": "2.0", "id": mid, "result": result},
                          separators=(",", ":"), ensure_ascii=False)

    def _emit_audit(self, tool: str, mid: Any, raw_texts: list[str],
                    text_blocks: list[dict], changed: bool) -> None:
        """Hand the audit callback one replay record per result (#23). Strictly a side
        effect: any error is swallowed so an audit-log write can never change what the
        client receives — same fail-open contract as capture."""
        record = {
            "tool": tool,
            "id": mid,
            "diff_mode": self.diff,
            "tiers": list(self.policy.select(tool).tiers),
            "changed": changed,
            "blocks": [{"raw": raw, "emitted": b["text"]}
                       for raw, b in zip(raw_texts, text_blocks)],
        }
        try:
            self.audit(record)  # type: ignore[misc]  — only called when set
        except Exception as exc:  # noqa: BLE001 — audit is never load-bearing
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: audit skipped: {exc}\n")


# Sentinel a transform returns to SWALLOW a line — write nothing to dst — as distinct from
# None, which forwards the line unchanged. Used when the client->server side answers a
# synthetic terse.retrieve call itself and must not forward it downstream (#10).
SWALLOW: Any = object()


def pump(src: TextIO, dst: TextIO, transform: Callable[[str], Any],
         lock: "Optional[Lock]" = None) -> None:
    """Read lines from src, apply transform, write to dst with a single trailing newline.
    transform returns: a string to write, None to forward the line unchanged, or SWALLOW to
    write nothing (the transform handled it out-of-band). Stops at EOF. With `lock`, each
    write+flush is serialized — needed on the shared client-facing stream, which both this
    pump and the retrieve answerer write to (#10)."""
    for raw in src:
        line = raw.rstrip("\n")
        if not line:
            continue
        out = transform(line)
        if out is SWALLOW:
            continue
        if out is None:
            out = line
        if lock is not None:
            with lock:
                dst.write(out + "\n")
                dst.flush()
        else:
            dst.write(out + "\n")
            dst.flush()


def stdio_transport_error(cmd: list[str]) -> Optional[str]:
    """Return a clear error if `cmd` can't be a proxy downstream target at all, else
    None (#19). Currently the only such case is nothing given after `--`. A URL is no
    longer rejected here — `transport.build_transport` dispatches a single `"://"`
    target to `HttpTransport` (#5), so a URL is a valid, launchable-in-spirit target
    same as a stdio command."""
    if not cmd:
        return "no downstream command given after `--`"
    return None


def _terminate_child(proc: "subprocess.Popen[Any]", timeout: float = 2.0) -> None:
    """Reap the downstream server if it is still running, so it shares the proxy's
    lifecycle and is never orphaned (#21). SIGTERM first, then SIGKILL on timeout."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def run_proxy(
    cmd: list[str],
    pol: policy_mod.Policy,
    debug: bool = False,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    capture_dir: Optional[str] = None,
    debug_log: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> int:
    """Launch the downstream MCP peer `cmd` and proxy JSON-RPC through `Interceptor`.
    `cmd` is either a stdio launch command, or a single-element list holding a URL — in
    which case `transport.build_transport` dispatches to `HttpTransport` instead of
    launching a subprocess (#5). `headers` is forwarded to an HTTP downstream only (e.g.
    bearer auth); it is a harmless no-op for a stdio one.

    A stdio child shares this process's lifecycle: it is reaped on normal exit, on a
    crash (via `finally`), and on SIGTERM (the signal a parent MCP client uses to stop
    us), so it is never left orphaned (#21). An HTTP downstream has no child process to
    reap — see the transport-specific control flow below.

    With `capture_dir`, each raw tool-result payload is also teed into that corpus dir
    (#32) for later `terse verify --corpus`/`measure` — opt-in, and strictly a side
    effect that can never change what the client receives.

    With `debug_log`, a structured raw->decision->emitted record per result is appended
    to that JSONL path (#23) for after-the-fact diagnosis/replay of a silent compression
    bug — same opt-in, side-effect-only contract.

    Return code: for a stdio downstream, the child's real exit code (or 127 if it could
    never be launched — #19), exactly as before this function grew a second transport.
    For an HTTP downstream there is no child process to exit; 0 means the client
    disconnected cleanly (its stdin hit EOF, which — via `client_to_server`'s `finally`
    below — closes the transport in turn)."""
    cin = stdin or sys.stdin
    cout = stdout or sys.stdout

    # Fail fast when there is nothing to proxy at all (#19): clearer than a hang or
    # empty result later. A URL is now a valid downstream (build_transport dispatches
    # it to HttpTransport below) — only "nothing after --" is still an error here.
    transport_err = stdio_transport_error(cmd)
    if transport_err is not None:
        sys.stderr.write(f"[terse-proxy] {transport_err}\n")
        return 2

    capture: Optional[Callable[[str, str], None]] = None
    if capture_dir is not None:
        from .capture import capture_payload

        def capture(tool: str, raw: str) -> None:
            # Swallow here too (defense in depth alongside the Interceptor's guard): a
            # read-only or full corpus dir must not break the proxy.
            try:
                capture_payload(tool, raw, capture_dir)
            except Exception as exc:  # noqa: BLE001 — capture is never load-bearing
                if debug:
                    sys.stderr.write(f"[terse-proxy] capture_payload failed: {exc}\n")

    audit: Optional[Callable[[dict], None]] = None
    if debug_log is not None:
        from .capture import append_audit

        def audit(record: dict) -> None:
            # Defense in depth alongside the Interceptor's guard: a read-only or full
            # disk must not break the proxy.
            try:
                append_audit(record, debug_log)
            except Exception as exc:  # noqa: BLE001 — audit is never load-bearing
                if debug:
                    sys.stderr.write(f"[terse-proxy] append_audit failed: {exc}\n")

    inter = Interceptor(pol, debug=debug, capture=capture, audit=audit)

    try:
        transport = build_transport(cmd, headers=headers)
    except OSError as exc:
        # Mistyped path, non-executable, or otherwise unlaunchable STDIO downstream —
        # report it as a config error instead of an uncaught traceback (#19). 127 = the
        # shell convention for "command not found". (An HTTP target never raises here:
        # HttpTransport.__init__ does no I/O — a bad URL/host only ever surfaces later,
        # fail-open, as a synthesized per-request error — see transport.py.)
        sys.stderr.write(f"[terse-proxy] failed to launch downstream server {cmd[0]!r}: "
                         f"{exc}\n")
        return 127

    # The rest of this function is deliberately NOT fully transport-polymorphic: the
    # stdio path keeps its exact pre-#5 control flow (half-close stdin, `proc.wait()`
    # for the real exit code, `_terminate_child` as the last-resort reaper) so every
    # existing behavior/test is byte-for-byte unchanged, while HTTP — which has no
    # child process to wait() on — generalizes to "block until the inbound pump
    # finishes", which for HTTP only happens once `transport.close()` runs.
    is_http = isinstance(transport, HttpTransport)

    # SIGTERM otherwise bypasses `finally` (default action exits immediately), orphaning
    # the child. Convert it to a clean SystemExit so cleanup runs. Only the main thread
    # may install handlers; in a worker (e.g. tests calling run_proxy directly) the
    # try/finally still covers the crash and normal-exit paths.
    prev_sigterm = None
    installed_sigterm = False
    try:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        installed_sigterm = True
    except (ValueError, OSError):
        pass

    # The client-facing stream (cout) now has TWO writers: the server->client pump and the
    # client->server side answering a swallowed terse.retrieve call (#10). Serialize every
    # write+flush to it so a synthesized reply can't interleave mid-line with a result.
    out_lock = Lock()

    try:
        def client_to_server() -> None:
            def fwd(line: str) -> Any:
                # A terse.retrieve call is ours to answer from the drop store — the downstream
                # server has no such tool. Write the reply straight back to the client and
                # SWALLOW the request so it never reaches downstream (and never enters
                # `pending`, since we don't call note_request for it). This never touches
                # `transport` at all — retrieve is a pure client<->proxy exchange, which is
                # exactly why it needed zero HTTP-specific reimplementation for #5.
                if inter.policy.has_drop():
                    reply = inter.answer_retrieve(line)
                    if reply is not None:
                        with out_lock:
                            cout.write(reply + "\n")
                            cout.flush()
                        return SWALLOW
                inter.note_request(line)
                return line  # forward request unchanged; only observe
            try:
                pump(cin, transport.outbound(), fwd)
            finally:
                if is_http:
                    # HTTP has no persistent connection to half-close: client stdin EOF
                    # IS the transport's end-of-life signal. Closing here (not only in
                    # the outer `finally`) is what lets `server_to_client`'s
                    # `pump(transport.inbound(), ...)` below ever terminate —
                    # HttpTransport.inbound() is a queue.Queue iterator with no other
                    # EOF condition (#5).
                    transport.close()
                else:
                    # stdio: half-close only, exactly as before this refactor. Closing
                    # the child's stdin signals EOF so it can flush any remaining reply
                    # and exit on its own; `proc.wait()` below blocks for that real
                    # exit, and the outer `finally`'s transport.close() (SIGTERM/SIGKILL
                    # escalation via `_terminate_child`) stays the last-resort reaper.
                    try:
                        transport.outbound().close()
                    except Exception:  # noqa: BLE001
                        pass

        def server_to_client() -> None:
            pump(transport.inbound(), cout, inter.transform_response, lock=out_lock)

        t_up = Thread(target=client_to_server, daemon=True)
        t_down = Thread(target=server_to_client, daemon=True)
        t_up.start()
        t_down.start()
        if is_http:
            # No child process to wait() on: block until the inbound pump thread itself
            # finishes, which only happens once `transport.close()` runs (above, from
            # client EOF) and drains the sentinel through HttpTransport.inbound()'s
            # queue iterator. No fixed timeout — inbound EOF IS the completion signal.
            t_down.join()
            rc = 0
        else:
            rc = transport.proc.wait()
            t_down.join(timeout=2.0)
        return rc
    finally:
        if installed_sigterm:
            # Ignore further SIGTERM while reaping: a second signal would otherwise
            # re-enter the sys.exit(143) handler and unwind out of transport.close()
            # before the SIGKILL escalation and the restore below ever run.
            try:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        transport.close()
        if installed_sigterm:
            # Restore the prior disposition; SIG_DFL when it wasn't a Python-set handler
            # (getsignal returns None there), so we never leave our lambda installed.
            try:
                signal.signal(signal.SIGTERM,
                              prev_sigterm if prev_sigterm is not None else signal.SIG_DFL)
            except (ValueError, OSError, TypeError):
                pass
