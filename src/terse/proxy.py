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

import hashlib
import json
import signal
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable
from threading import Lock, Thread
from typing import Any, TextIO

from . import lossy as lossy_mod
from . import policy as policy_mod
from . import text_diff, transforms
from .tokenize import count_cl100k
from .transport import HttpTransport, build_transport

# How long to let the inbound pump finish draining the downstream's final reply after the
# child process has exited (stdio). Generous: the child's stdout EOF guarantees the pump
# terminates once buffered data is flushed; this only bounds a pathological stall (e.g. the
# client stopped reading our stdout) instead of the old 2s cap that could truncate a large
# final reply outright.
_STDIO_DRAIN_TIMEOUT = 30.0

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


def _args_key(arguments: Any) -> str:
    """Stable short digest of a tools/call's `arguments`, used to ATTRIBUTE each diff base
    to the call that produced it (Phase 1 instrumentation). Canonical (sorted keys) so
    equal arguments always collide; empty/absent/unserializable -> "". Recorded only — the
    diff base is still keyed by tool name alone at this phase; whether to key ON this is the
    Phase 2 decision the ledger's `diff_reason` breakdown informs."""
    if not arguments:
        return ""
    try:
        canon = json.dumps(arguments, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    except (TypeError, ValueError):
        return ""
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


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
                 capture: Callable[[str, str], None] | None = None,
                 audit: Callable[[dict], None] | None = None,
                 stats: Callable[[str, str, str, bool, str | None, str | None], None] | None = None,
                 server_name: str | None = None,
                 store: OrderedDict[str, Any] | None = None,
                 store_lock: Lock | None = None,
                 dropped_bytes: list[int] | None = None,
                 log_prefix: str = "[terse-proxy]"):
        self.policy = pol
        # The downstream server's name, when the caller knows it (`proxy --server-name`,
        # or a multiproxy peer's config name). Passed to every `policy.select`/`apply` so
        # a server-scoped rule (`runecho.*`) matches a server that doesn't self-prefix its
        # own tool names (#83). None = no qualified candidate, i.e. exactly the pre-#83
        # matching behavior.
        self.server_name = server_name
        # id -> (policy_tool, capture_tool): policy_tool drives compression/policy-tier
        # lookup and MUST be the bare name the policy's rules match against; capture_tool
        # is what capture()/audit() see and defaults to policy_tool, but multiproxy
        # overrides it to a peer-qualified name (see note_request's tool_name) so two
        # peers' same-named tools don't collide into one capture-corpus bucket.
        self.pending: dict[Any, tuple[str, str, str]] = {}
        self.debug = debug
        self.diff = pol.diff
        # Join every text block of a multi-block result into one record array before
        # compressing (#116) — folds records across blocks AND makes the result
        # diff-eligible. Independent of `diff`: with diffing off it still folds, just
        # never diffs.
        self.join_blocks = pol.join_blocks
        # Optional tee of each RAW (pre-compression) tool-result text, keyed by tool name
        # (#32). Keeps the Interceptor I/O-free: the callback owns the disk write. Never
        # affects forwarding — its failures are swallowed at the call site.
        self.capture = capture
        # Optional structured replay log of the raw->decision->emitted triple per result
        # (#23). Like capture, the callback owns I/O and its failures are swallowed: an
        # audit-log write must NEVER change what the client receives.
        self.audit = audit
        # Optional payload-FREE savings ledger callback: (tool, raw, emitted,
        # passthrough) per result block (see stats.py). Unlike capture/audit it is safe
        # to leave always-on — it records sizes and decisions, never content — but it
        # keeps their exact contract: callback owns I/O, failures are swallowed.
        self.stats = stats
        # Side-effect sinks (capture/audit/stats) swallow their failures to stay fail-open,
        # but a sink that fails on EVERY call — a full disk, a bad path — would then stop
        # writing forever with the failure only visible under --debug. Warn ONCE per sink,
        # unconditionally, the first time it fails, so a silently-dead ledger is noticed.
        # This is the ONLY place a sink failure is reported, so the callbacks must let
        # their exceptions out (#131) — see `_build_capture_and_audit`.
        self._sink_warned: set[str] = set()
        # Prefix on this Interceptor's stderr lines, so a multiproxy peer's sink failure
        # is attributed to `[terse-multiproxy]` rather than the single-proxy default.
        self.log_prefix = log_prefix
        self.last: dict[str, Any] = {}  # tool -> previous result object (the diff base)
        # tool -> args-key of the call that produced the base above (Phase 1). Recorded
        # only, to classify WHY a diff did/didn't fire (same-args miss vs different-args
        # base); the base itself is still keyed by tool name alone. Cleared everywhere
        # `last` is (skip path, reconnect) so the two never disagree.
        self.last_args: dict[str, str] = {}
        # tool -> whether the base above came from a JOINED multi-block result (#116). A
        # result that joins on one call and doesn't on the next flips array<->object, so a
        # diff across the flip would be unresolvable; when this flag differs from the
        # current result the base is dropped and the result re-anchors as a full. Kept in
        # lockstep with `last` (cleared wherever `last` is).
        self.last_joined: dict[str, bool] = {}
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
        self.dropped: OrderedDict[str, Any] = store if store is not None else OrderedDict()
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
        # `clientInfo.name` from the handshake, when the client declared one (#128). Drives
        # `"structured": "auto"`; None until an initialize is seen, and None means "leave".
        self.client_name: str | None = None
        # The two proxy pump threads call note_request (client->server) and
        # transform_response (server->client) concurrently, both mutating pending/last/
        # since_keyframe/init_id state. `_local_lock` serializes each method against the
        # other so the compound eviction + the reconnect reset can't race a response in
        # flight — it is ALWAYS private to this Interceptor, never shared with another
        # peer's, so it never blocks another peer's compression/capture/audit work.
        self._local_lock = Lock()
        # `_store_lock` guards ONLY `self.dropped`/`_dropped_bytes_box` (see `_drop_put`/
        # `answer_retrieve`), which multiproxy.py DOES share across every peer's
        # Interceptor via `store_lock` — that dict is the same physical object across
        # peers, so mutating it needs cross-peer exclusion. Splitting this out from
        # `_local_lock` means a slow peer's compression/disk-I/O (held under its own
        # PRIVATE `_local_lock`) no longer serializes every other peer's response
        # processing behind it — only the brief drop-store dict mutation does, and that
        # happens on the order of microseconds, not a full compress/capture/audit pass.
        #
        # INVARIANT (read this before adding a new lock-acquiring method to this class):
        # whenever a method needs BOTH locks, it must acquire `_local_lock` OUTER and
        # `_store_lock` INNER, never the reverse — a method that acquires `_store_lock`
        # first and then something needing `_local_lock` (directly, or by calling back
        # into another method of this class) creates a lock-order cycle with any method
        # that already does local-then-store, which can deadlock under concurrent
        # multi-peer load. This is enforced only by this comment, not by the type system
        # or a runtime check — `answer_retrieve` already acquires `_store_lock` alone
        # with no nesting, so a future method extending that pattern must not also reach
        # for `_local_lock` while still holding `_store_lock`.
        self._store_lock = store_lock if store_lock is not None else Lock()

    def note_request(self, line: str, *, tool_name: str | None = None) -> None:
        """Record id -> tool name for tools/call requests, and the initialize request id
        (so its reply can carry the format primer). Side-effect only.

        `tool_name`, if given, overrides the name parsed from `line`'s own
        `params.name` — used by multiproxy to track a peer-qualified name (e.g.
        `"gh__search"`) for capture/audit bookkeeping, even though `line` itself
        (sent to the downstream) carries the bare name the peer actually expects."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        method = msg.get("method")
        with self._local_lock:
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
                self.last_args.clear()
                self.last_joined.clear()
                self.since_keyframe.clear()
                self.last_text.clear()
                self.since_text_keyframe.clear()
                self.pending.clear()
                # self.dropped is the (possibly cross-peer-shared) drop store — its own
                # lock guards this reset, consistent with _drop_put/answer_retrieve.
                # Lock order is always _local_lock then _store_lock, never reversed
                # anywhere in this class, so nesting them here is deadlock-safe.
                with self._store_lock:
                    self.dropped.clear()
                    self._dropped_bytes_box[0] = 0
                # The client's DECLARED identity, straight off the handshake. This is
                # what lets `"structured": "auto"` compress the typed `structuredContent`
                # field only for clients measured not to validate it (#128) — an observed
                # name, not a heuristic. Absent/malformed leaves it None, which the
                # resolver treats as "unknown" and therefore "leave".
                info = (msg.get("params") or {}).get("clientInfo")
                if isinstance(info, dict) and isinstance(info.get("name"), str):
                    self.client_name = info["name"]
                    if self.debug:
                        sys.stderr.write(
                            f"{self.log_prefix} client: {info['name']} "
                            f"{info.get('version', '?')} -> structured=auto resolves to "
                            f"{policy_mod.structured_mode_for_client('auto', info['name'])}\n")
                if mid is not None:
                    self.init_id = mid
                return
            if method != "tools/call":
                return
            params = msg.get("params") or {}
            name = params.get("name")
            if mid is not None and isinstance(name, str):
                self.pending[mid] = (name, tool_name if tool_name is not None else name,
                                     _args_key(params.get("arguments")))
                # dict preserves insertion order; drop the oldest tracked id(s) once over
                # cap so abandoned (timed-out) entries can't accumulate (#22). Safe under
                # the lock — no concurrent mutation during the iterate-then-pop.
                while len(self.pending) > self.PENDING_MAX:
                    self.pending.pop(next(iter(self.pending)))

    def clear_init_id(self) -> None:
        """Reset the one-time initialize-reply marker `note_request` just set, without
        waiting for `transform_response` to see it. Used by multiproxy for a broadcast-
        rewritten `initialize`: that peer's real reply is intercepted and merged by the
        broadcast collector, never reaching `transform_response`, so its normal one-time
        reset (`transform_response`'s `msg["id"] == self.init_id` branch) never fires and
        `init_id` would otherwise stay stale, risking a later unrelated reply being
        misidentified as the initialize reply if its id ever collides."""
        with self._local_lock:
            self.init_id = None

    def transform_response(self, line: str) -> str:
        """Compress the text of a tracked tools/call result; prime the initialize reply;
        else return unchanged."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line
        if not isinstance(msg, dict) or msg.get("id") is None:
            return line
        # A server-initiated REQUEST (it carries "method" alongside an id) is NOT a reply to
        # anything this proxy sent. JSON-RPC gives each direction its OWN id space, and both
        # sides conventionally number from 1, so a server's `roots/list` /
        # `sampling/createMessage` / `elicitation/create` id routinely collides with an
        # in-flight tools/call id. Falling through would pop that call's `pending` entry (the
        # pop below is deliberately unconditional so an error-shaped reply still frees it),
        # and the REAL result would then arrive untracked — silently forwarded uncompressed
        # and missing from the ledger. Forward it untouched instead; a server request is not
        # ours to answer or rewrite.
        # Predicate deliberately identical to multiproxy's `from_peer` guard: a message
        # carrying BOTH `method` and a `result`/`error` is not a server-initiated request
        # under any reading of JSON-RPC, and must still take the response path.
        if msg.get("method") is not None and "result" not in msg and "error" not in msg:
            return line
        # Held across the whole body so the init_id/pending/last/since_keyframe state
        # stays consistent against a concurrent note_request on the other thread.
        # ALWAYS this Interceptor's own private lock — never blocks another peer's
        # transform_response, even under multiproxy's shared drop store (see
        # _drop_put/_store_lock for the piece that DOES need cross-peer exclusion).
        with self._local_lock:
            if msg["id"] == self.init_id:
                self.init_id = None  # one-time
                primed = self._augment_initialize(msg)
                return primed if primed is not None else line
            # Pop BEFORE the "result" check (not after, as a top-level early-return
            # guard would do): an error-shaped reply for a tracked id — including
            # HttpTransport's own synthesized fail-open error — must still free its
            # `pending` entry, or it lingers until PENDING_MAX eviction instead of
            # being cleaned up immediately.
            tracked = self.pending.pop(msg["id"], None)
            if tracked is None or "result" not in msg:
                # Either not a tracked tools/call response at all (tools/list, ...),
                # or an error reply for one we WERE tracking (already popped above).
                # When a policy enables drop-to-retrieve, a tools/list reply is where
                # we advertise the synthetic terse.retrieve tool so the model knows
                # how to fetch a dropped field back (#10) — only for the untracked
                # case, never for a tracked call's error reply.
                if tracked is None and self.policy.has_drop():
                    injected = self._inject_retrieve_tool(msg)
                    if injected is not None:
                        return injected
                return line
            tool, capture_tool, args_key = tracked

            result = msg.get("result")
            content = result.get("content") if isinstance(result, dict) else None
            if not isinstance(content, list):
                return line

            text_blocks = [b for b in content
                           if isinstance(b, dict) and b.get("type") == "text"
                           and isinstance(b.get("text"), str)]

            # An `isError` result is a failure the model has to READ to act on — a stack
            # trace or a "server said no" message. Compression is fine (it's lossless and
            # the text stays legible), but a LOSSY transform must not put an extra
            # retrieve round-trip between the model and an error at exactly the moment it
            # is trying to recover. Forced fully lossless, same suppression the never-lossy
            # server floor applies, so an error payload is never evicted to a handle.
            error_result = bool(result.get("isError")) if isinstance(result, dict) else False

            # `"capture": false` on the matching rule — never PERSIST this tool's payloads
            # (#85). Gates BOTH sinks that write raw content to disk: the corpus tee below
            # and the audit/replay log further down (its records embed the raw payload too,
            # so gating only the tee would be half a guard). The in-memory compression path
            # is untouched — this is about what survives on disk, and the client's result
            # is identical either way.
            persist = self.policy.select(tool, self.server_name).capture

            # Snapshot the raw block texts before any transform mutates them in place: the
            # capture tee, the audit log's raw side (#23), and the stats ledger all read the
            # ORIGINAL payload. The stats ledger is payload-FREE (sizes + decision only), so
            # it is never gated by `capture: false` — a credential-returning tool still gets
            # counted, just never quoted.
            wants_raw = ((self.capture is not None and persist)
                         or (self.audit is not None and persist)
                         or self.stats is not None)
            raw_texts = [b["text"] for b in text_blocks] if wants_raw else None

            changed = False
            diff_reason: str | None = None
            joined_block: dict | None = None   # set when the multi-block join fires (#116)
            joined_curr: list | None = None    # its parsed pre-lossy array, for capture

            # `"structured": "replace"` (#128) — is this result's text block a dead mirror
            # of `structuredContent`? Decided HERE, against the RAW block, because every
            # branch below rewrites that text in place and the comparison is only
            # meaningful before they do.
            mirror = self._mirror_to_drop(result, text_blocks, tool,
                                          error_result=error_result)

            if mirror is not None:
                # Do not compress a block that is about to be deleted: it is wasted work,
                # and it would leave a diff base the client never received — the next
                # result would then diff against text nobody has seen.
                diff_reason = "mirror_dropped"
                # Set here, not at the drop below, so the audit record — emitted further
                # down, and deliberately before the block is removed — reports `changed`
                # truthfully. A trace saying "changed: false" next to an emitted "" would
                # be the replay log lying about the one decision it exists to record.
                changed = True
                if self.diff:
                    self.last.pop(tool, None)
                    self.last_args.pop(tool, None)
                    self.last_joined.pop(tool, None)
                    self.since_keyframe.pop(tool, None)
                # Emitted side is the empty string, which is the literal wire truth: the
                # ledger must show this block costing zero, not show it "unchanged".
                emitted_pairs = ([(r, "") for r in raw_texts]
                                 if raw_texts is not None else [])
            # #116: a result with >=2 text blocks is tried as ONE joined record array first
            # — the per-block path can reach neither cross-record folding nor the diff tier
            # (71% of real traffic was stuck there). A refusal falls back to per-block and
            # records WHY (`multiblock_<reason>`); the join itself is gated by
            # `join_blocks`, independent of `diff`.
            elif len(text_blocks) >= 2:
                new_text, diff_reason, joined_curr = self._compress_or_diff_joined(
                    text_blocks, tool, args_key, force_lossless=error_result)
                if new_text is not None:
                    joined_block = {"type": "text", "text": new_text}

            if mirror is not None:
                pass                       # handled above; the drop itself happens below
            elif joined_block is not None:
                # Collapse the N text blocks to the single joined block, in place; non-text
                # blocks keep their positions. This is the one path that changes the number
                # of content blocks the client sees — defensible because the MCP spec puts
                # no meaning on block count (2025-06-18 server/tools).
                self._collapse_text_blocks(content, text_blocks, joined_block)
                changed = True
                emitted_pairs = ([("\n".join(raw_texts), joined_block["text"])]
                                 if raw_texts is not None else [])
            elif self.diff and len(text_blocks) == 1:
                changed, diff_reason = self._compress_or_diff(
                    text_blocks[0], tool, args_key, force_lossless=error_result)
                emitted_pairs = ([(r, b["text"]) for r, b in zip(raw_texts, text_blocks, strict=True)]
                                 if raw_texts is not None else [])
            else:
                for block in text_blocks:
                    new_text = self._compress(block["text"], tool,
                                              force_lossless=error_result)
                    if new_text != block["text"]:
                        block["text"] = new_text
                        changed = True
                if len(text_blocks) == 1:
                    diff_reason = "diff_off"   # diffing disabled for a single-block result
                # A per-block result the model receives as N blocks has no single JSON value
                # a later diff could reference (and its actual prior same-tool result was
                # these N blocks, not the stale base) — drop any base so the next result
                # re-anchors, the same discipline the skipped path applies (#116).
                if self.diff:
                    self.last.pop(tool, None)
                    self.last_args.pop(tool, None)
                    self.last_joined.pop(tool, None)
                    self.since_keyframe.pop(tool, None)
                emitted_pairs = ([(r, b["text"]) for r, b in zip(raw_texts, text_blocks, strict=True)]
                                 if raw_texts is not None else [])

            # Tee the RAW payload (#32), AFTER the path is known so a joined result is
            # captured ONCE as the array terse actually compresses — not N per-block
            # envelopes that would make the corpus misrepresent multi-block tools (the
            # corpus feeds measure / fluency / policy-generate). Strictly a side effect:
            # a capture failure never changes what the client receives.
            if self.capture is not None and persist and raw_texts is not None:
                if joined_block is not None:
                    payloads = [json.dumps(joined_curr, separators=(",", ":"),
                                           ensure_ascii=False)]
                else:
                    payloads = raw_texts
                for payload in payloads:
                    try:
                        self.capture(capture_tool, payload)
                    except Exception as exc:  # noqa: BLE001 — capture is never load-bearing
                        self._warn_sink("capture", capture_tool, exc)

            # Audit AFTER the transform, regardless of `changed`: a no-op is itself
            # diagnostic — it confirms terse left a suspect payload untouched. On the joined
            # path both sinks see ONE (raw, emitted) pair — raw = the N originals joined by
            # newline (the true wire cost the model saw), emitted = the single joined block.
            if self.audit is not None and persist:
                self._emit_audit(tool, msg["id"], emitted_pairs, changed,
                                 display_tool=capture_tool)
            # `structuredContent` rides alongside the text blocks and is what some clients
            # actually give the model (#128). Compress it when the rule opts in; either
            # way its EMITTED size is what the ledger must count, so the reported saving
            # tracks the whole result rather than the text block alone.
            structured, rewrote_structured = self._compress_structured(
                result, tool, force_lossless=error_result)
            changed = changed or rewrote_structured

            # The mirror drop happens LAST, after the typed field is final and after both
            # sinks have seen the raw block: capture feeds the corpus and audit is the
            # record of what the server sent, and neither may be told the block never
            # existed. Removal is by identity — an `==`-based remove could take a
            # different block that happens to compare equal.
            if mirror is not None:
                result["content"] = [b for b in content if b is not mirror]
                if self.debug:
                    sys.stderr.write(
                        f"[terse-proxy] {tool}: dropped {len(mirror['text'])}-char text "
                        f"mirror of structuredContent (structured=replace)\n")

            if self.stats is not None:
                self._emit_stats(tool, emitted_pairs, display_tool=capture_tool,
                                 diff_reason=diff_reason, structured=structured)

            if not changed:
                return line
            # Re-serialize compactly. JSON-RPC is semantics, not formatting; no newlines.
            return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def _compress_or_diff(self, block: dict, tool: str, args_key: str = "",
                          force_lossless: bool = False) -> tuple[bool, str]:
        """Compress one block, preferring a lossless delta vs the prior same-tool result
        when it is smaller. Updates the per-tool diff base. Returns `(changed, reason)`,
        where `reason` is the Phase 1 instrumentation datum — a short label for WHY the
        diff did/didn't fire (no_prior | keyframe | emitted | not_smaller_same_args |
        not_smaller_diff_args | text_emitted | text_dropped | non_json | passthrough |
        error), for the
        ledger. Fail-open: any error leaves the block untouched and state intact."""
        text = block["text"]
        try:
            applied = policy_mod.apply(text, tool, self.policy, drop_sink=self._drop_put,
                                       server=self.server_name,
                                       force_lossless=force_lossless)
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: passthrough on error: {exc}\n")
            return False, "error"
        if applied.skipped:
            # Skipped = a passthrough tool (empty tiers) OR a non-JSON result (e.g. an
            # upstream error string, a file read, a log tail) for a normally-compressed
            # one. Either way it carries no JSON the next JSON diff could build on, and
            # it becomes the model's visible "previous same-tool result" — so drop any
            # stale JSON diff base and reset its keyframe counter, forcing the next JSON
            # result to re-anchor as a full (#8).
            self.last.pop(tool, None)
            self.last_args.pop(tool, None)
            self.last_joined.pop(tool, None)
            self.since_keyframe.pop(tool, None)
            if applied.text != text:
                # A text-payload drop-to-retrieve fired (`$text.code_blocks`): the payload
                # is non-JSON, so no tier ran and `skipped` stays True, but the emitted
                # text is NOT the raw text. Emit it and skip the CDC text diff entirely —
                # chaining a diff onto a dropped payload would make the base depend on
                # which spans happened to clear the size floor, so clear that state too
                # and let the next raw text re-anchor as a full.
                block["text"] = applied.text
                self.last_text.pop(tool, None)
                self.since_text_keyframe.pop(tool, None)
                return True, "text_dropped"
            if not self.policy.select(tool, self.server_name).tiers:
                return False, "passthrough"  # true passthrough: hands off, no state kept
            # A CDC text diff that actually shipped is a real diff hit — bucket it as
            # `text_emitted`, not `non_json`, or the ledger's emitted-vs-non_json split
            # misreports file-read/log-tail traffic.
            changed = self._text_diff_or_store(block, tool, text)
            return changed, ("text_emitted" if changed else "non_json")

        chosen = applied.text
        reason = "non_json"  # curr unparseable/too-deep: no JSON diff decision was possible
        try:
            curr = json.loads(text)
            # Depth guard (#79): a payload past the codec-wide cap must not become the
            # diff base — the diff encoders/decoders recurse and deep-compare without a
            # depth argument. Treat it like non-JSON: no diff in, no base stored.
            if transforms.exceeds_depth(curr):
                curr = None
        except (json.JSONDecodeError, ValueError, RecursionError):
            curr = None
        if curr is not None:
            chosen, reason = self._diff_decision(applied.text, curr, tool, args_key,
                                                 joined=False)

        if chosen != text:
            block["text"] = chosen
            return True, reason
        return False, reason

    def _diff_decision(self, full_text: str, curr: Any, tool: str, args_key: str,
                       *, joined: bool) -> tuple[str, str]:
        """Decide diff-vs-full for one reconstructable payload `curr` whose full compressed
        form is `full_text`; update the per-tool diff base; return `(emitted_text, reason)`.
        Shared by the single-block path and the multi-block join (#116).

        `joined` records whether this result collapsed N blocks into one. When it differs
        from the tool's previous result the shapes are incompatible (array vs object), so a
        diff across the flip would be unresolvable — the base is treated as absent and this
        result re-anchors as a full (`reason == "reanchor"`)."""
        prev = self.last.get(tool)
        prev_args = self.last_args.get(tool)
        prev_joined = self.last_joined.get(tool)
        # A shape flip (join<->single) makes the stored base structurally incompatible.
        shape_flip = prev is not None and prev_joined is not None and prev_joined != joined
        chosen = full_text
        emitted_diff = False
        # A keyframe is due once K diffs have chained off the last full result; force the
        # full compressed form so the chain re-anchors (#8). interval 0 = never.
        keyframe_due = (self.keyframe_interval > 0
                        and self.since_keyframe.get(tool, 0) >= self.keyframe_interval)
        if prev is None or shape_flip:
            reason = "reanchor" if shape_flip else "no_prior"
        elif keyframe_due:
            reason = "keyframe"              # forced full to re-anchor the chain
        else:
            wire = self._diff_wire(prev, curr, tool)
            if wire is not None and _cost(wire) < _cost(full_text):
                chosen = wire
                emitted_diff = True
                reason = "emitted"
                if self.debug:
                    sys.stderr.write(
                        f"[terse-proxy] {tool}: diff {_cost(full_text)}->{_cost(wire)} "
                        f"tok vs full compressed\n")
            else:
                # A base existed but the delta didn't win. Split by whether that base came
                # from a DIFFERENT-args call (arg-keying could offer a better, same-args
                # base) or the SAME args (a genuine encoding miss arg-keying wouldn't fix).
                reason = ("not_smaller_diff_args" if prev_args != args_key
                          else "not_smaller_same_args")
        if self.debug and keyframe_due and not shape_flip:
            sys.stderr.write(f"[terse-proxy] {tool}: keyframe (full) after "
                             f"{self.since_keyframe.get(tool, 0)} diffs\n")
        # A diff extends the chain; any full result (no prior, diff lost, keyframe, flip)
        # is a fresh anchor and resets the counter.
        self.since_keyframe[tool] = self.since_keyframe.get(tool, 0) + 1 if emitted_diff else 0
        # Base the NEXT diff on the true current value, whichever form we emit: the model's
        # reconstructable state after this turn is `curr` either way.
        self.last[tool] = curr
        self.last_args[tool] = args_key
        self.last_joined[tool] = joined
        return chosen, reason

    def _compress_or_diff_joined(self, text_blocks: list[dict], tool: str, args_key: str,
                                 force_lossless: bool = False
                                 ) -> tuple[str | None, str, list | None]:
        """#116: compress a multi-block result as ONE joined record array, preferring a
        cross-call diff when it wins. Returns `(emitted_text, reason, raw_array)`:

          - `emitted_text is None` — the join was declined; the caller falls back to the
            per-block path. `reason` is `multiblock_<why>` and `raw_array` is None.
          - otherwise `emitted_text` is the single joined block's text, `reason` is the
            diff decision (`emitted` | `no_prior` | `keyframe` | `reanchor` |
            `not_smaller_*`) or `joined` when diffing is off, and `raw_array` is the parsed
            pre-lossy blocks (the value captured to the corpus).

        Fail-open: any error declines the join, leaving the per-block path and diff state
        untouched."""
        raws = [b["text"] for b in text_blocks]
        try:
            applied, curr, refuse = policy_mod.apply_joined(
                raws, tool, self.policy, drop_sink=self._drop_put,
                server=self.server_name, force_lossless=force_lossless)
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: join passthrough on error: {exc}\n")
            return None, "multiblock_error", None
        if applied is None:
            return None, f"multiblock_{refuse}", None
        if self.diff:
            chosen, reason = self._diff_decision(applied.text, curr, tool, args_key,
                                                 joined=True)
        else:
            # Diffing off: emit the full joined-and-compressed form, keep no base.
            chosen, reason = applied.text, "joined"
        return chosen, reason, curr

    @staticmethod
    def _collapse_text_blocks(content: list, old_text_blocks: list[dict],
                              joined_block: dict) -> None:
        """Replace the N text blocks in `content` (in place) with the single `joined_block`,
        positioned where the FIRST text block was; every non-text block (image / audio /
        resource_link / embedded resource) keeps its place and order (#116). Matched by
        object identity — the blocks in `old_text_blocks` are the very dicts in `content`."""
        old_ids = {id(b) for b in old_text_blocks}
        out: list = []
        placed = False
        for b in content:
            if id(b) in old_ids:
                if not placed:
                    out.append(joined_block)
                    placed = True
                # else: a subsequent text block, now subsumed into joined_block — drop it
            else:
                out.append(b)
        content[:] = out

    def _augment_initialize(self, msg: dict) -> str | None:
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

    def _diff_wire(self, prev: Any, curr: Any, tool: str) -> str | None:
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

    def _text_diff_wire(self, prev: str, curr: str, tool: str) -> str | None:
        """Fail-open wrapper mirroring `_diff_wire`, for the CDC text-diff codec."""
        try:
            return text_diff.text_diff_wire(prev, curr, tool)
        except Exception:  # noqa: BLE001 — fail-open
            return None

    def _compress(self, text: str, tool: str, force_lossless: bool = False) -> str:
        """policy.apply with a hard fail-open: any error returns the original text."""
        try:
            applied = policy_mod.apply(text, tool, self.policy, drop_sink=self._drop_put,
                                       server=self.server_name,
                                       force_lossless=force_lossless)
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

    def _structured_mode(self, tool: str) -> str:
        """This tool's `structured` setting, resolved against the connected client. One
        place, so the mirror-drop guard and the codec can never disagree about the mode."""
        return policy_mod.structured_mode_for_client(
            self.policy.select(tool, self.server_name).structured, self.client_name)

    def _mirror_to_drop(self, result: Any, text_blocks: list[dict], tool: str, *,
                        error_result: bool) -> dict | None:
        """The text block to delete under `"structured": "replace"` (#128 option 2), or
        None to leave the result's blocks alone.

        MCP 2025-06-18 has a structured tool return the serialized JSON in a text block
        *for backwards compatibility*. Measured against `claude` 2.1.218, that client reads
        `structuredContent` and discards the block — so once the typed field is compressed
        (#134/#135) the block is the entire remaining wire cost and nobody's input. Dropping
        it is measured-safe for that client: a result with `content: []` and a populated
        typed field reaches the model complete and without error
        (`scripts/probe/structured_content/`, the `nomirror` probe).

        Measured-safe, and for that client measured-worthless: context cost went
        2,596 -> 1,008 chars under "compress" and 1,008 -> 1,008 under "replace", because
        the block it removes was already being thrown away. The mode exists for a client
        that forwards both fields (unmeasured — see `policy.Rule.structured`), which is
        also the only client that would otherwise see a diffed block contradicting a
        full-envelope typed field.

        Every condition below must hold; any failure returns None and the result takes the
        ordinary compress path, exactly as `"compress"` would have produced it:

        * mode resolves to "replace" — never on "auto"/"leave"/"compress"
        * the rule actually has tiers. `tiers: []` is the "hands off this tool" switch, and
          removing a block is the most hands-on thing terse does. It also keeps the ledger
          honest: a passthrough-labelled row whose out_chars fell would be the #133 error
          again.
        * not an error result — error text is usually the only thing there, and a model
          recovering from a failure has to be able to read it
        * exactly one text block, and it is a FAITHFUL mirror: its parsed JSON equals
          `structuredContent`. A block that merely accompanies the typed field carries
          information the typed field does not, and dropping it would lose data. This is
          the guard that makes the whole thing safe rather than merely measured, since it
          is checked per result rather than assumed from the spec's SHOULD.

        Deliberately NOT a guard: whether the tool declared an `outputSchema`. That was the
        expected gate — a client should only prefer the typed field when a schema says it
        exists — and it was measured false: the `noschema` probe's mirror-less-equivalent
        tool declares no `outputSchema` and the client forwarded `structuredContent`
        anyway. Keeping a guard whose premise had just been disproved would be superstition,
        and it would have cost per-tool `tools/list` state to enforce.

        Also NOT a guard: whether the codec managed to shrink the typed field. If it did
        not, the field is still there, still complete, still the field the client reads —
        and the mirror is still dead weight."""
        if self._structured_mode(tool) != "replace" or error_result:
            return None
        if not self.policy.select(tool, self.server_name).tiers:
            return None
        if len(text_blocks) != 1:
            return None
        if not isinstance(result, dict) or "structuredContent" not in result:
            return None
        try:
            mirrored = json.loads(text_blocks[0]["text"])
        except (json.JSONDecodeError, ValueError):
            return None                       # not JSON: not a mirror of anything
        return text_blocks[0] if mirrored == result["structuredContent"] else None

    def _compress_structured(self, result: Any, tool: str, *,
                             force_lossless: bool = False) -> tuple[str | None, bool]:
        """Run a result's `structuredContent` through the codec in place, when the matching
        rule opted in with `"structured": "compress"` (#128). Returns the serialized field
        as it will go out (compressed or not) for the ledger, or None when absent.

        Why this exists: measured against `claude` 2.1.218, the client forwards the TYPED
        field to the model and discards the text block terse compresses, so on a tool that
        emits both, compressing only the block delivers ~0%. Why it is opt-in, and why the
        default must stay "leave": see `policy.Rule.structured`.

        No diff. Diffing the typed field needs its own per-tool base and keyframe
        accounting; mixing that in here would double the surface with none of the evidence
        the text-block diff tier earned before it was turned on.

        It is otherwise the SAME path the text block takes — `policy.apply` — so a rule
        that declares `drop-to-retrieve` fields sees them applied here too, and the typed
        field can come out carrying a `__terse_dropped__` marker. That is deliberate (the
        mirrored payload has the mirrored shape, so the same field paths match) and it
        inherits the same guards: the never-lossy SERVER floor is enforced inside `apply`
        on the verified server identity, and `force_lossless` suppresses it on an error
        result. Handles are content-derived, so the same value dropped from both the block
        and the field resolves to one store entry, not two.

        Fail-open like everything else on this path: a field that does not survive a
        round-trip through `json.dumps` is left exactly as it was."""
        if not isinstance(result, dict) or "structuredContent" not in result:
            return None, False
        try:
            original = json.dumps(result["structuredContent"], separators=(",", ":"),
                                  ensure_ascii=False)
        except (TypeError, ValueError):
            return None, False                # unserializable: not ours to touch
        if self._structured_mode(tool) not in policy_mod.STRUCTURED_REWRITING:
            return original, False            # untouched, but still counted by the ledger
        emitted = self._compress(original, tool, force_lossless=force_lossless)
        if emitted == original:
            return original, False
        try:
            result["structuredContent"] = json.loads(emitted)
        except json.JSONDecodeError:
            # The codec's output is always JSON, so this cannot normally fire — but the
            # typed field is the one a client may hand straight to a schema validator, so
            # an unparseable replacement must never be written. Keep the original.
            return original, False
        return emitted, True

    def _drop_put(self, handle: str, value: Any) -> None:
        """Store a dropped field's original under `handle` for a later terse.retrieve (#10).
        LRU: re-inserting an existing handle refreshes its recency; once over the count or
        byte cap, evict oldest-first. Called from apply() inside transform_response, while
        that method's own `_local_lock` is held — acquires `_store_lock` itself here
        rather than assuming a caller-held lock already covers it, since `_store_lock` is
        the one that's actually shared across peers under multiproxy (`_local_lock` isn't)."""
        with self._store_lock:
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

    def _inject_retrieve_tool(self, msg: dict) -> str | None:
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

    def answer_retrieve(self, line: str) -> str | None:
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
        if not isinstance(handle, str):
            handle = ""  # a malformed/absent handle can only ever be a miss below
        value = None
        with self._store_lock:
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

    def _warn_sink(self, kind: str, tool: str, exc: Exception) -> None:
        """Announce a side-effect sink failure. The FIRST failure of each kind is written
        unconditionally — a sink failing on every call (full disk, bad path) would else go
        silent forever without --debug — and further ones only under --debug, so a
        persistently-failing sink can't flood stderr on the hot path."""
        first = kind not in self._sink_warned
        if first or self.debug:
            self._sink_warned.add(kind)
            tail = " (further occurrences silenced unless --debug)" if first else ""
            sys.stderr.write(f"{self.log_prefix} {tool}: {kind} skipped: {exc}{tail}\n")

    def _emit_audit(self, tool: str, mid: Any, pairs: list[tuple[str, str]],
                    changed: bool, *, display_tool: str | None = None) -> None:
        """Hand the audit callback one replay record per result (#23). Strictly a side
        effect: any error is swallowed so an audit-log write can never change what the
        client receives — same fail-open contract as capture.

        `pairs` is one `(raw, emitted)` per emitted block — N pairs on the per-block path,
        exactly ONE on the joined path (#116), where `raw` is the N originals joined by
        newline and `emitted` is the single joined block. `tool` drives
        `self.policy.select(tool, self.server_name)` and MUST be the bare/policy-matching
        name. `display_tool`, if given, overrides only the record's `"tool"` field (e.g.
        multiproxy's peer-qualified name) without affecting which policy rule's tiers get
        reported."""
        shown_tool = display_tool if display_tool is not None else tool
        record = {
            "tool": shown_tool,
            "id": mid,
            "diff_mode": self.diff,
            "tiers": list(self.policy.select(tool, self.server_name).tiers),
            "changed": changed,
            "blocks": [{"raw": raw, "emitted": emitted} for raw, emitted in pairs],
        }
        audit = self.audit
        if audit is None:
            return  # caller already gates on this; kept for local type-narrowing too
        try:
            audit(record)
        except Exception as exc:  # noqa: BLE001 — audit is never load-bearing
            self._warn_sink("audit", shown_tool, exc)

    def _emit_stats(self, tool: str, pairs: list[tuple[str, str]], *,
                    display_tool: str | None = None, diff_reason: str | None = None,
                    structured: str | None = None) -> None:
        """Hand the stats callback one (tool, raw, emitted, passthrough, diff_reason) per
        emitted block, for the payload-free savings ledger (stats.py). Same fail-open
        contract as capture/audit: the callback owns I/O and a failure can never change
        what the client receives. `pairs`/`tool`/`display_tool` as in `_emit_audit`. The
        diff decision is per-result, so `diff_reason` is attributed to every pair — which
        is exactly one pair on the joined path and the common single-block shape.

        `structured` is the serialized `structuredContent` this result carried, if any. It
        is per-RESULT, not per-block, so on a multi-block result it is attributed to the
        first pair only — counting it once per block would inflate the very number this is
        meant to make honest (#128)."""
        stats = self.stats
        if stats is None:
            return
        shown_tool = display_tool if display_tool is not None else tool
        passthrough = not self.policy.select(tool, self.server_name).tiers
        for index, (raw, emitted) in enumerate(pairs):
            try:
                stats(shown_tool, raw, emitted, passthrough, diff_reason,
                      structured if index == 0 else None)
            except Exception as exc:  # noqa: BLE001 — stats is never load-bearing
                self._warn_sink("stats", shown_tool, exc)


# Sentinel a transform returns to SWALLOW a line — write nothing to dst — as distinct from
# None, which forwards the line unchanged. Used when the client->server side answers a
# synthetic terse.retrieve call itself and must not forward it downstream (#10).
SWALLOW: Any = object()


def pump(src: Iterable[str], dst: Any, transform: Callable[[str], Any],
         lock: Lock | None = None) -> None:
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


def stdio_transport_error(cmd: list[str]) -> str | None:
    """Return a clear error if `cmd` can't be a proxy downstream target at all, else
    None (#19). Currently the only such case is nothing given after `--`. A URL is no
    longer rejected here — `transport.build_transport` dispatches a single `"://"`
    target to `HttpTransport` (#5), so a URL is a valid, launchable-in-spirit target
    same as a stdio command."""
    if not cmd:
        return "no downstream command given after `--`"
    return None


def _terminate_child(proc: subprocess.Popen[Any], timeout: float = 2.0) -> None:
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


# Sentinel distinguishing "SIGTERM handler installation was attempted and failed" (a
# non-main thread — signal.signal only works there; a caller-held finally must still
# run cleanup regardless) from "installed, and the prior disposition was None" (no
# Python-set handler; restore to SIG_DFL, not None). `_install_sigterm_to_exit`'s
# return value is opaque to callers — pass it straight to `_ignore_sigterm`/
# `_restore_sigterm`, which both already no-op correctly for this sentinel.
_SIGTERM_NOT_INSTALLED: Any = object()


def _install_sigterm_to_exit() -> Any:
    """SIGTERM otherwise bypasses a caller's `finally` (default action exits
    immediately), orphaning a child process/peer. Convert it to a clean
    `sys.exit(143)` so cleanup runs. Shared by `run_proxy` and
    `multiproxy.run_multi_proxy` (#21) — install/ignore/restore is identical in both,
    differing only in what teardown work happens between `_ignore_sigterm` and
    `_restore_sigterm`. Returns an opaque token for those two functions."""
    try:
        prev = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        return prev
    except (ValueError, OSError):
        # Only the main thread may install signal handlers; in a worker (e.g. a test
        # calling run_proxy directly) this silently no-ops — the caller's own
        # try/finally still covers crash and normal-exit paths regardless.
        return _SIGTERM_NOT_INSTALLED


def _ignore_sigterm(token: Any) -> None:
    """Ignore further SIGTERM while reaping: a second signal would otherwise
    re-enter the `sys.exit(143)` handler and unwind out of teardown before the
    SIGTERM/SIGKILL escalation and `_restore_sigterm` below ever run."""
    if token is _SIGTERM_NOT_INSTALLED:
        return
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, OSError):
        pass


def _restore_sigterm(token: Any) -> None:
    """Restore the prior disposition; SIG_DFL when it wasn't a Python-set handler
    (`token is None`), so a caller never leaves the `sys.exit(143)` lambda installed."""
    if token is _SIGTERM_NOT_INSTALLED:
        return
    try:
        signal.signal(signal.SIGTERM, token if token is not None else signal.SIG_DFL)
    except (ValueError, OSError, TypeError):
        pass


def _build_capture_and_audit(
    capture_dir: str | None, debug_log: str | None
) -> tuple[Callable[[str, str], None] | None, Callable[[dict], None] | None]:
    """Build the (capture, audit) callback pair from --capture-dir/--debug-log, shared
    by `run_proxy` and `multiproxy.run_multi_proxy` (identical logic, differing only in
    which process's downstream target they're wired to).

    These callbacks own I/O and NOTHING else: a failure propagates to the caller. Both
    sinks are still strictly side effects — a read-only or full disk must never break
    the proxy — but that fail-open guarantee is enforced by the one caller that has the
    bookkeeping for it, `Interceptor` (see `_warn_sink`), which swallows the failure AND
    announces the first one of each kind. Catching here as well made that unconditional
    first warning dead code, so a dead sink stayed invisible without --debug (#131)."""
    capture: Callable[[str, str], None] | None = None
    if capture_dir is not None:
        from .capture import capture_payload

        def capture(tool: str, raw: str) -> None:
            capture_payload(tool, raw, capture_dir)

    audit: Callable[[dict], None] | None = None
    if debug_log is not None:
        from .capture import append_audit

        def audit(record: dict) -> None:
            append_audit(record, debug_log)

    return capture, audit


def run_proxy(
    cmd: list[str],
    pol: policy_mod.Policy,
    debug: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    capture_dir: str | None = None,
    debug_log: str | None = None,
    headers: dict[str, str] | None = None,
    stats_log: str | None = None,
    server_name: str | None = None,
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

    With `stats_log`, a payload-FREE savings record per result (sizes + decision, never
    content — see stats.py) is appended to that JSONL ledger for `terse stats`. Unlike
    the two above this is ON by default (cli.py resolves the default path; None here
    means disabled) — safe because no payload content is stored — but it keeps the
    identical side-effect-only, fail-open contract.

    `server_name` is this downstream's name in the MCP config. It makes a server-scoped
    policy rule (`runecho.*`) match a server whose tools aren't self-prefixed, and labels
    the stats ledger with the real server identity instead of the command basename (#83).

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

    capture, audit = _build_capture_and_audit(capture_dir, debug_log)

    stats = None
    if stats_log is not None:
        from .stats import build_stats_writer, server_label

        # `server_name` (the MCP config's own name for this server) is the truthful
        # identity when the caller knows it; `server_label(cmd)` is the fallback guess
        # from the command basename, which misreads a launcher-wrapped server (kb behind
        # secret-broker's `sb-run` labels itself "sb-run") — #83.
        label = server_name or server_label(cmd)
        stats = build_stats_writer(stats_log, label)

    inter = Interceptor(pol, debug=debug, capture=capture, audit=audit, stats=stats,
                        server_name=server_name)

    try:
        transport = build_transport(cmd, headers=headers)
    except OSError as exc:
        # Mistyped path, non-executable, or otherwise unlaunchable STDIO downstream —
        # report it as a config error instead of an uncaught traceback (#19). 127 = the
        # shell convention for "command not found".
        sys.stderr.write(f"[terse-proxy] failed to launch downstream server {cmd[0]!r}: "
                         f"{exc}\n")
        return 127
    except ValueError as exc:
        # An HTTP target does no I/O in __init__, but it DOES now reject a disallowed URL
        # scheme (file://, ftp://, …) up front — a config error, so exit 2 like any other
        # bad downstream spec rather than crashing on the traceback (see transport.py).
        sys.stderr.write(f"[terse-proxy] {exc}\n")
        return 2

    # `half_close()`/`wait()` (Transport methods) hide every transport-specific
    # teardown/exit-code detail behind polymorphism — no isinstance check needed for
    # those. `is_http` is still needed for ONE genuinely irreducible difference: an
    # HTTP downstream has no process exit code at all, so "how long do we block
    # joining the inbound pump thread, and what's the resulting rc" differs in KIND,
    # not just in which method to call (see the join/rc branch below).
    is_http = isinstance(transport, HttpTransport)

    sigterm_token = _install_sigterm_to_exit()

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
                # transport.half_close() is what lets server_to_client's
                # pump(transport.inbound(), ...) below ever terminate: for HTTP
                # (a queue.Queue iterator with no other EOF condition) it closes
                # outright; for stdio it closes the child's stdin so the child can
                # flush any remaining reply and exit on its own (transport.wait()
                # below blocks for that real exit; the outer finally's
                # transport.close() — SIGTERM/SIGKILL escalation — stays the
                # last-resort reaper either way).
                transport.half_close()

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
            rc = transport.wait()
            # The child has exited, so its stdout reaches EOF and the inbound pump WILL
            # terminate once it drains the last buffered reply — give it a generous window
            # to do so. The old 2s cap could kill the daemon thread mid-drain on a large
            # final reply, silently truncating the client's last message(s). If the drain
            # still hasn't finished (e.g. the client stopped reading our stdout), announce
            # it rather than truncating in silence.
            t_down.join(timeout=_STDIO_DRAIN_TIMEOUT)
            if t_down.is_alive():
                sys.stderr.write(
                    "[terse] downstream exited but its final reply did not finish "
                    f"draining within {_STDIO_DRAIN_TIMEOUT:.0f}s; last message(s) may "
                    "be truncated\n")
        return rc
    finally:
        _ignore_sigterm(sigterm_token)
        transport.close()
        _restore_sigterm(sigterm_token)
