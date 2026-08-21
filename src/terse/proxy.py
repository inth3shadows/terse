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
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import partial
from threading import Lock, Thread
from typing import Any, TextIO

from . import lossy as lossy_mod
from . import policy as policy_mod
from . import text_diff, transforms

# Only the two cadence labels, at module scope: they are string constants, `stats` has no
# module-level dependency on anything heavy (json/os/re/time plus `_secure_io` and the
# `tokenize` this module already imports), and there is no cycle -- `stats` never imports
# `proxy`. The WRITER is still imported lazily in `run_proxy` as before. Naming the site's
# cadence from the one definition beats re-spelling the literal at each call site.
from .stats import PRIMER_CADENCE_ONCE
from .tokenize import count_cl100k
from .transport import HttpTransport, build_transport

# How long to let the inbound pump finish draining the downstream's final reply after the
# child process has exited (stdio). Generous: the child's stdout EOF guarantees the pump
# terminates once buffered data is flushed; this only bounds a pathological stall (e.g. the
# client stopped reading our stdout) instead of the old 2s cap that could truncate a large
# final reply outright.
_STDIO_DRAIN_TIMEOUT = 30.0

# The corpus tee: (tool, raw, server, result_id). `server` and `result_id` are what let a
# tune-time reader reconstruct which downstream sent a payload and which result's blocks
# arrived together, instead of inferring both from capture timing (#148, #152). Keyword-only
# past `raw` so the pre-#148 two-positional-arg call shape still reads correctly.
CaptureFn = Callable[..., None]

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
# The primer is assembled per-server from these sections, not shipped whole (#168). Each
# section documents ONE wire form, and the form is gated by policy the proxy already knows
# at initialize time — so a server explains only what it can actually put on the wire.
#
# Measured cost of the whole primer, cl100k: header 41, table 155, dict 44, diff 190,
# embedded 53, dropped 64, tail 8 = 555 with every gate on. A default policy emits head +
# table + dict + tail = 248, since `diff` is off (#170) and embedded/dropped are per-rule.
# Paid per wrapped server, ONCE per session since #211 — it was re-read every turn as
# cache_read before that, which is why terse measured a 14.0% win at one wrapped server
# and a loss at three. A router still pays the per-turn cadence; see `stats.py`.
#
# `table` went 55 -> 155, and this is the section to attack first if #168's per-server tax
# reopens. Two things drove it:
#
#   * Union-schema tabularize's absent-cell vocabulary (+74). NOT optional decoration:
#     measured on 24 absent-vs-null questions over a non-uniform table, three models scored
#     54.2% / 54.2% / 79.2% against the old 55-token paragraph — a binary question answered
#     near chance, because a `null` filling a hole is indistinguishable from a real null
#     without the rule — and 95.8% / 100% / 100% with the shipped paragraph (single
#     trial, n=24; the wording measured before `subcols` was folded in scored 100% on
#     all three). A cheaper encoding exists (one
#     ABSENT_MARKER in EVERY absent cell, no `absent_cols`/`sentinel_cols` arrays: 87 tokens
#     total, also 100% on the same probe), traded away because it costs more on the wire at
#     scale (31.8% vs 33.5% saved on a 200-record table) — the primer is paid once per
#     session (per turn for a router), the wire per payload. The trade only got better
#     for standalone entries when #211 made the primer lazy.
#   * `subcols` (+26), which the codec has emitted since nested key folding shipped and this
#     paragraph never named. Found by making the coupling a test rather than a promise: the
#     guard derives the required vocabulary from a real emission, so a header key added to
#     the codec fails there instead of reaching a model with no rule for reading it.
#
# Reconfirmed 2026-08-04 (issue #168): the 87-token encoding was NOT adopted. #168's own
# weighted-cost model (cache_read 0.1x, cache_write 1.25x) makes the 68-token primer delta
# worth only ~6.8 weighted tokens/turn, while the 1.7pp of wire savings it trades away
# lands on cache_write-priced payload tokens — ~12.5x more expensive per token. This
# repo's own live ledger (`terse stats`) has single table-shaped tools (e.g.
# kb.read.list_principles: 823 calls, 435,876 raw tokens) large enough that a sliver of
# non-uniform traffic erases the entire primer-side gain accumulated across every logged
# call in the ledger's history. Don't re-litigate this without a traffic mix that inverts
# that ratio.
#
# PRIMER_HEAD is the idempotency sentinel: it appears in every non-empty assembly, so
# `_augment_initialize` can detect its own prior injection without knowing which sections
# were selected.
PRIMER_HEAD = (
    "Some tool results are 'terse'-compressed (a lossless, denser JSON encoding); some "
    "are sent as diffs against the previous result of the same tool. Read each as the "
    "equivalent full JSON:\n"
)
PRIMER_TABLE = (
    '- Table {"__terse_table__":1,"n":N,"cols":[...],"rows":[[...]]}: N records, each row '
    'POSITIONAL — its i-th value belongs to the i-th name in "cols". "n" is the exact count. '
    'Records need not share a key set: "absent_cols":[i,...] lists columns where some records '
    'have NO such key, written null there — or "__terse_absent__" for columns also in '
    '"sentinel_cols", where a null is a real null. An absent cell means the key is missing '
    'from that record, not that its value is null. "subcols":{name:SPEC} means that column\'s '
    'cells are themselves positional rows, read against SPEC the same way.\n'
)
PRIMER_DICT = (
    '- Dict {"__terse_dict__":1,"legend":{"~0":value,...},"data":...}: every "~K" token '
    'inside "data" stands for legend["~K"] — substitute it back.\n'
)
PRIMER_DIFF = (
    '- Diff {"__terse_diff__":1,"shape":"rows","by":COL,"set":[...],"new":[...],"del":[...],'
    '"n":N}: update the PREVIOUS same-tool result — from its records drop ids in "del", '
    'overwrite/insert each record in "set" matched by its "by" field, append ids in "new"; '
    '"n" is the final record count. A {"shape":"keys","set":{...},"del":[...]} diff instead '
    'removes "del" keys and applies "set" key/values to the previous object. '
    'A text diff {"__terse_textdiff__":1,"ops":[["=",a,b],["+","..."],...]} updates the '
    "PREVIOUS same-tool plain-text result: process ops in order, copying chunks a..b of "
    "that prior text for a `=` op or inserting its literal string for a `+` op, then "
    "concatenating everything.\n"
)
PRIMER_EMBEDDED = (
    '- Embedded JSON {"__terse_json__":1,"f":F,"v":...}: "v" is a JSON document the tool '
    'returned as a STRING — read it as that document. "f" is a re-serialization tag; '
    "ignore it.\n"
)
PRIMER_DROPPED = (
    '- Dropped field {"__terse_dropped__":"H","bytes":N,"retrieve":"terse.retrieve"}: a '
    "large field value was omitted to save context. It is NOT lost — when you actually need "
    'it, call the terse.retrieve tool with {"handle":"H"} to get the exact original back.\n'
)
PRIMER_TAIL = "Always reason about the fully reconstructed result."

# The full assembly, for tests and for callers that want every section regardless of
# policy. `build_primer(default_policy())` reproduces this exactly.
TERSE_PRIMER = (PRIMER_HEAD + PRIMER_TABLE + PRIMER_DICT + PRIMER_EMBEDDED + PRIMER_DIFF
                + PRIMER_DROPPED + PRIMER_TAIL)


def build_primer(pol: policy_mod.Policy, server: str | None = None) -> str:
    """The primer for a server governed by `pol` — only the wire forms it can emit (#168).

    Returns "" when the policy can emit no compressed form at all (every reachable rule is
    `tiers: ()` with diffing therefore dead and no field drop) — before #168 such a server
    paid a full primer to explain forms it is structurally forbidden from producing.

    That is a property of the POLICY, not of any particular server: `secret-broker` pays
    248 under both shipped policies, though by different routes — an explicit
    `secret-broker.secret.list_credentials` carve-out sitting ahead of `secret-broker.*`
    in the live policy, and `defaults` in `policy.example.json`, which has no
    secret-broker rule at all. `server_never_lossy` (#199) only suppresses the
    dropped-field paragraph. The default-deny shape this returns "" for is deliberate.

    `minify` alone is deliberately NOT a reason to emit a primer: minified JSON is just
    JSON, carries no terse marker, and needs no explanation.
    """
    return _assemble_primer(
        table=pol.emits_table(server), dictionary=pol.emits_dict(server),
        embedded=pol.emits_embedded(server),
        diff=pol.emits_diff(server), dropped=pol.has_drop(server),
    )


def _assemble_primer(*, table: bool, dictionary: bool, diff: bool, dropped: bool,
                     embedded: bool = False) -> str:
    """Join the selected sections, or "" when none are selected."""
    body = "".join(s for gate, s in (
        (table, PRIMER_TABLE),
        (dictionary, PRIMER_DICT),
        (embedded, PRIMER_EMBEDDED),
        (diff, PRIMER_DIFF),
        (dropped, PRIMER_DROPPED),
    ) if gate)
    return PRIMER_HEAD + body + PRIMER_TAIL if body else ""


def union_primer(pairs: list[tuple[policy_mod.Policy, str | None]]) -> str:
    """One primer covering every form ANY of `policies` can emit (#168).

    `pairs` is [(policy, server_name)] — each peer is gated against its OWN name, since a
    rule like `kb.*` totally covers the kb peer and is irrelevant to the runecho one.

    For a router fronting several peers with independent policies: a form no peer can emit
    is a form the client will never see, but a form any single peer can emit must still be
    documented once. Erring toward inclusion — a surplus paragraph costs tokens, a missing
    one costs comprehension."""
    return _assemble_primer(
        table=any(p.emits_table(s) for p, s in pairs),
        dictionary=any(p.emits_dict(s) for p, s in pairs),
        embedded=any(p.emits_embedded(s) for p, s in pairs),
        diff=any(p.emits_diff(s) for p, s in pairs),
        dropped=any(p.has_drop(s) for p, s in pairs),
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
                 capture: CaptureFn | None = None,
                 audit: Callable[[dict], None] | None = None,
                 stats: Callable[[str, str, str, bool, str | None, str | None,
                                  str | None], None] | None = None,
                 server_name: str | None = None,
                 store: OrderedDict[str, Any] | None = None,
                 store_lock: Lock | None = None,
                 dropped_bytes: list[int] | None = None,
                 origins: dict[str, tuple[str, str, str]] | None = None,
                 stats_retrieve: Callable[[str, str, str, bool, str], None] | None = None,
                 stats_primer: Callable[[str, str, bool], None] | None = None,
                 ledger_label: str | None = None,
                 log_prefix: str = "[terse-proxy]",
                 lazy_primer: bool = True):
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
        # Bumped on every `initialize`, and folded into the result id handed to `capture`
        # so a reconnecting client's restarted JSON-RPC ids stay distinguishable (#148).
        self._result_gen = 0
        self.debug = debug
        self.diff = pol.diff
        # Join every text block of a multi-block result into one record array before
        # compressing (#116) — folds records across blocks AND makes the result
        # diff-eligible. Independent of `diff`: with diffing off it still folds, just
        # never diffs.
        self.join_blocks = pol.join_blocks
        # Optional tee of each RAW (pre-compression) tool-result text, keyed by tool name
        # (#32). Keeps the Interceptor I/O-free: the callback owns the disk write. Never
        # affects forwarding — its failures are swallowed at the call site. Also handed
        # this peer's `server` and the `result_id` the block belonged to, so the corpus can
        # answer "which server" and "which call" instead of leaving both to be guessed at
        # tune time (#148, #152).
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
        # `handle -> (tool, rule path)` for everything in `self.dropped` (#251). SHARED
        # whenever `store` is, and for the same reason: under multiproxy any peer's
        # Interceptor may be the one that answers a `terse.retrieve` for a handle a
        # DIFFERENT peer dropped, so a private origins map would lose the attribution on
        # exactly the fleet shape that has a lossy-by-default rule. Guarded by
        # `_store_lock` alongside the dict it mirrors, and evicted in lockstep with it.
        self._drop_origin: dict[str, tuple[str, str, str]] = (origins if origins is not None
                                                              else {})
        # The ledger `server` label THIS Interceptor's drops should be billed to. Stored
        # into `_drop_origin` at drop time rather than read at retrieve time, because under
        # multiproxy the router answers EVERY terse.retrieve through `peers[0]` (see
        # `_route_call`) — so the answering Interceptor is almost never the one that
        # dropped the value, and its own label would mislabel the row. Falls back to
        # `server_name`; `run_proxy` passes the resolved ledger identity, which can differ.
        self._ledger_label = ledger_label or server_name or ""
        # Optional payload-FREE ledger callback for a retrieve round-trip:
        # (server, tool, path, hit, payload). Same fail-open contract as `stats`.
        self.stats_retrieve = stats_retrieve
        # Optional callback for a primer that was actually EMITTED (#311): (cadence, text).
        # Same payload-free, fail-open contract as `stats`. Fires at most once per session
        # on the lazy path and exactly once on the eager one, so it is nowhere near a hot
        # path -- which is why it is a separate writer rather than a branch inside `stats`.
        self.stats_primer = stats_primer
        self.init_id: Any = None        # id of the initialize request, to prime its reply
        # `clientInfo.name` from the handshake, when the client declared one (#128). Drives
        # `"structured": "auto"`; None until an initialize is seen, and None means "leave".
        self.client_name: str | None = None
        # Lazy primer (#168 phase 2): when True, `initialize` is left unprimed and the
        # primer instead attaches to the first `tools/call` result that actually carries a
        # terse wire form — paid once per SESSION, not once per TURN. False preserves the
        # old always-eager `_augment_initialize` behavior; multiproxy passes False for every
        # peer, since the router already primes eagerly once via `union_primer` and a peer
        # going lazy too would just double the explanation on top of that (see
        # `_build_peers`). Computed once here, not lazily: `pol` is finalized by every
        # caller before construction, so there's nothing to gain by deferring it, and
        # deferring would mean recomputing `build_primer` on every reconnect reset instead
        # of once.
        self._lazy_primer = lazy_primer
        self._primer_text = build_primer(pol, server_name) if lazy_primer else ""
        # True = "nothing left to do" — collapses `lazy_primer=False` and "this policy
        # emits no compressible form at all" (`build_primer` returns "" for a default-deny
        # policy) into the same no-op state, so neither needs its own branch later.
        self._primer_sent = not (lazy_primer and self._primer_text)
        # Whether this process has already recorded a SUPPRESSED primer (#286). Separate
        # from `_primer_sent` because they answer different questions and a session can do
        # both -- suppressed on an early `structuredContent` result, attached on a later
        # text-only one. One row of each is the truth; the reader treats an attach as
        # authoritative. Bounded to one per process so a server that returns
        # `structuredContent` on every call writes one row, not one per result.
        self._primer_suppressed_logged = False
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
        # `msg.get("params") or {}` only neutralises FALSY junk: a client that sends
        # `"params": "oops"` (or a list) passes that guard and then raises AttributeError
        # on the first `.get` below. That exception escapes into the client->server pump
        # THREAD and kills forwarding for the rest of the session — a malformed request
        # taking the whole proxy down, when this method is side-effect-only bookkeeping
        # and should simply decline to record anything it cannot parse.
        raw_params = msg.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
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
                # Same reasoning as the diff-base clears above, applied to the lazy primer:
                # a reconnect means the model's context — and with it, any primer it
                # previously read — is gone. Without this reset, a reconnected session's
                # tools/call results would compress freely with no primer ever explaining
                # the wire forms to the NEW context.
                self._primer_sent = not (self._lazy_primer and self._primer_text)
                self._primer_suppressed_logged = False
                # A reconnecting client restarts its JSON-RPC ids at 1 while this process
                # keeps one session id, so `sess:1` from before and after the reconnect
                # would name two unrelated results the same and the corpus would fuse them
                # — the #148 defect, arriving by the one door the process-scoped id leaves
                # open. Same reasoning as the `pending` reset directly above, applied to
                # the identity the corpus stores rather than the one this class routes by.
                self._result_gen += 1
                # self.dropped is the (possibly cross-peer-shared) drop store — its own
                # lock guards this reset, consistent with _drop_put/answer_retrieve.
                # Lock order is always _local_lock then _store_lock, never reversed
                # anywhere in this class, so nesting them here is deadlock-safe.
                with self._store_lock:
                    self.dropped.clear()
                    self._dropped_bytes_box[0] = 0
                    # Cleared with the store, for the same reason the store is: every
                    # handle the model still holds became unresolvable at the reconnect,
                    # so its attribution describes a drop that can no longer be retrieved.
                    self._drop_origin.clear()
                # The client's DECLARED identity, straight off the handshake. This is
                # what lets `"structured": "auto"` compress the typed `structuredContent`
                # field only for clients measured not to validate it (#128) — an observed
                # name, not a heuristic. Absent/malformed leaves it None, which the
                # resolver treats as "unknown" and therefore "leave".
                info = params.get("clientInfo")
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
            name = params.get("name")
            # `mid` becomes a dict KEY below, so a non-hashable id (`"id": {"a": 1}`) would
            # raise TypeError out of this same pump thread — the identical failure the
            # params guard above closes, by a different door.
            if isinstance(mid, (str, int)) and isinstance(name, str):
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
        # Sink calls (capture/audit/stats) are QUEUED here and invoked after the lock is
        # released. A `try/except` around a sink catches one that RAISES; it cannot catch
        # one that BLOCKS — a full disk mid-retry, a stalled network mount, a slow fsync
        # would hold `_local_lock` indefinitely, freeze `note_request` (which takes the
        # same lock), and with it every later tools/call on this connection. Deferring the
        # I/O is what makes the fail-open contract — "a sink failure or slowness never
        # affects forwarding" — true for slowness and not merely for failure.
        deferred: list[tuple[str, str, Callable[[], None]]] = []
        # Held across the whole body so the init_id/pending/last/since_keyframe state
        # stays consistent against a concurrent note_request on the other thread.
        # ALWAYS this Interceptor's own private lock — never blocks another peer's
        # transform_response, even under multiproxy's shared drop store (see
        # _drop_put/_store_lock for the piece that DOES need cross-peer exclusion).
        with self._local_lock:
            if msg["id"] == self.init_id:
                self.init_id = None  # one-time
                if self._lazy_primer:
                    # #168 phase 2: no eager priming — the primer attaches to the first
                    # qualifying tools/call result instead (see the end of this method).
                    return line
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
                if tracked is None and self.policy.has_drop(self.server_name):
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
            partial_done = False               # set when a PARTIAL multi-block join fires (#140)
            partial_pairs: list[tuple[str, str]] = []
            partial_payloads: list[str] = []

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
                    # ALL SIX state maps, not just the JSON four. The client's actual
                    # previous result for this tool is an empty content array, so a later
                    # CDC text diff whose `=` ops reference the dropped block's text would
                    # be unrecoverable — the model never received the text being referenced.
                    # Same discipline as the per-block path below and the reconnect reset.
                    self.last.pop(tool, None)
                    self.last_args.pop(tool, None)
                    self.last_joined.pop(tool, None)
                    self.since_keyframe.pop(tool, None)
                    self.last_text.pop(tool, None)
                    self.since_text_keyframe.pop(tool, None)
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
                elif diff_reason in ("multiblock_non_json", "multiblock_heterogeneous"):
                    # #140: the FULL join refused because at least one block is not a
                    # record (a bare error string, a JSON array). Rather than drop the
                    # whole result to the per-block path — losing cross-record folding on
                    # the records that ARE there — fold each contiguous run of >=2 object
                    # blocks and leave the rest per-block. A server emitting an error block
                    # beside good records is ordinary MCP, not an edge case (the single
                    # most common diff reason in the live ledger).
                    partial_result = self._partial_join(content, tool,
                                                        force_lossless=error_result)
                    if partial_result is not None:
                        partial_pairs, partial_payloads, partial_changed = partial_result
                        partial_done = True
                        changed = partial_changed
                        diff_reason = "multiblock_partial"

            if mirror is not None:
                pass                       # handled above; the drop itself happens below
            elif partial_done:
                # #140: `_partial_join` already rebuilt `content` in place and dropped any
                # diff base; just carry its (raw, emitted) pairs to the sinks below.
                emitted_pairs = partial_pairs
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
                    # Diffing is off BY POLICY here — this is the `else` of
                    # `elif self.diff and len(text_blocks) == 1`. The block count only
                    # decides whether the label applies, it is not the cause. The old
                    # wording ("disabled for a single-block result") read as a structural
                    # exclusion and helped convince a reader diffing was unimplemented
                    # rather than deliberately off (#181, #170).
                    diff_reason = "diff_off"
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
                elif partial_done:
                    # Each folded run captured as its own array, each leftover block as
                    # itself — the corpus mirrors exactly what terse compressed (#140).
                    payloads = partial_payloads
                else:
                    payloads = raw_texts
                # The JSON-RPC id identifies the RESULT these blocks came from, so the
                # corpus no longer has to infer "same call" from write timing (#148).
                # Scoped twice, because the bare id is unique only within one connection:
                # by handshake generation here, and by process in the callback — many
                # connections, and many processes, write into one corpus dir. Resolved
                # under the lock (`_result_gen` is reset by note_request on re-handshake)
                # so the deferred write carries the generation this result was read in,
                # not whatever a reconnect may have moved it to meanwhile.
                result_id = f"{self._result_gen}.{msg['id']}"
                for payload in payloads:
                    deferred.append((
                        "capture", capture_tool,
                        partial(self.capture, capture_tool, payload,
                                server=self.server_name, result_id=result_id),
                    ))

            # Audit AFTER the transform, regardless of `changed`: a no-op is itself
            # diagnostic — it confirms terse left a suspect payload untouched. On the joined
            # path both sinks see ONE (raw, emitted) pair — raw = the N originals joined by
            # newline (the true wire cost the model saw), emitted = the single joined block.
            if self.audit is not None and persist:
                deferred.append((
                    "audit", capture_tool,
                    partial(self._emit_audit, tool, msg["id"], emitted_pairs, changed,
                            display_tool=capture_tool),
                ))
            # `structuredContent` rides alongside the text blocks and is what some clients
            # actually give the model (#128). Compress it when the rule opts in; either
            # way its EMITTED size is what the ledger must count, so the reported saving
            # tracks the whole result rather than the text block alone.
            structured_raw, structured_out, rewrote_structured = self._compress_structured(
                result, tool, force_lossless=error_result)
            changed = changed or rewrote_structured

            # The mirror drop happens LAST, after the typed field is final and after both
            # sinks have seen the raw block: capture feeds the corpus and audit is the
            # record of what the server sent, and neither may be told the block never
            # existed. Removal is by identity — an `==`-based remove could take a
            # different block that happens to compare equal.
            if mirror is not None:
                # Slice-assign the list `result["content"]` already points at rather than
                # rebinding the key: same effect, and it works off `content`, which the
                # isinstance check above narrowed to a list (`result` is still `Any | None`
                # to a type checker at this point).
                content[:] = [b for b in content if b is not mirror]
                if self.debug:
                    sys.stderr.write(
                        f"[terse-proxy] {tool}: dropped {len(mirror['text'])}-char text "
                        f"mirror of structuredContent (structured=replace)\n")

            # #168 phase 2 (lazy primer): attach the format primer to the FIRST result that
            # actually carries a terse wire form, instead of eagerly at `initialize`. Once
            # per session (`_primer_sent` latches True below); a no-op after that or when
            # `lazy_primer` is off.
            #
            # The trigger is "does the FINAL content contain a terse marker", not the
            # coarser `changed` flag — `changed` is also True for a pure mirror-drop
            # (nothing terse-encoded survives to explain) and for a `structuredContent`-only
            # rewrite (see the guard below).
            #
            # `structuredContent` guard: measured (`scripts/probe/structured_content/`) that
            # Claude Code discards the text block ENTIRELY whenever a result also carries
            # `structuredContent` — a primer text block inserted here would be thrown away
            # right alongside it, regardless of whether the marker itself landed in text or
            # in `structuredContent` (an untouched `structuredContent` still wins the
            # client's preference over text). Skip attaching on THIS result rather than
            # attach somewhere it can't be seen; a later, text-only compressible result
            # still gets it. A session whose every wrapped-tool call carries
            # `structuredContent` never finds that later call — a known, narrow, accepted
            # gap (see #168 plan notes), not silently pretended away.
            structured_present = isinstance(result, dict) and "structuredContent" in result
            # Did terse put a wire form on THIS result? Computed once and shared by both
            # branches below, because both ask the same question and only differ on what to
            # do with the answer.
            #
            # An earlier revision asked it with `'"__terse_' in json.dumps(result)` and a
            # comment claiming the short-circuit made that run "at most once per process".
            # The claim was backwards (found in review): `_primer_suppressed_logged` only
            # latches when the marker IS found, so on a server that never produces one --
            # all-passthrough rules, a shape the codec never wins on -- every preceding term
            # stayed true and the whole result was re-serialized on EVERY response, forever,
            # inside `_local_lock`. Measured at ~265ms per response on 2MB payloads, on the
            # very path `proxy.py` otherwise takes care never to serialize.
            #
            # `rewrote_structured` is the precise answer for the structured side (terse
            # rewrote that field, so the marker is in it by construction) and a text-block
            # scan is the answer for the other. No serialization, and it drops the
            # quoted-marker false positive the dump inherited from the attach gate as well.
            # Gated on `primer_pending`, not computed unconditionally. BOTH consumers below
            # already require it, so without this guard the scan ran on every response for
            # the entire life of the process -- long after the primer was sent, and even with
            # `lazy_primer` off -- at ~0.8ms per 2MB no-match payload, under `_local_lock`.
            # Hoisting the scan out of the old branch was right; hoisting it past its own
            # precondition was not.
            #
            # Deliberately NOT covered by a dedicated test: "this expression did not
            # evaluate" is not observable from outside without contriving a fixture that
            # tests the contrivance. Its one behavioural consequence -- no suppression is
            # recorded once the primer has been sent -- IS pinned, by
            # `test_no_suppression_is_recorded_after_the_primer_has_already_attached`.
            primer_pending = not self._primer_sent and self._lazy_primer
            marker_in_text = primer_pending and any(
                isinstance(b, dict) and b.get("type") == "text"
                and isinstance(b.get("text"), str) and '"__terse_' in b["text"]
                for b in content)
            # NO `changed` term, deliberately: the attach gate four lines below is
            # `marker_in_text` alone, and these two must agree on what "the client will see a
            # terse envelope" means or the same content records a primer when it is text-only
            # and records NOTHING when `structuredContent` rides along. A round-2 revision had
            # `changed and ...` here and disagreed on 96 of a 504-case matrix -- every one of
            # them a doubly-wrapped peer or a downstream whose own text is already a terse
            # envelope, a shape `_cadence`'s docstring names as live. It failed safe (an
            # estimate, never a fabricated zero) but re-opened #286's bill for exactly that
            # shape. `primer_pending` already does the cost-avoidance `changed` was added for.
            wire_form_emitted = (primer_pending
                                 and (marker_in_text or rewrote_structured))
            suppressed_owed = (structured_present and wire_form_emitted
                               and not self._primer_suppressed_logged)
            if suppressed_owed:
                # The SUPPRESSION, written down rather than left to be inferred (#286,
                # #317-redesign). This branch is the one #286 is about: the result carries a
                # terse wire form, so a primer is owed, and the guard above refuses to
                # attach it because the client would discard the text block unread. The
                # cost is therefore genuinely zero -- and until this row existed, nothing
                # in the ledger said so, leaving `primer_liability` to bill the full primer
                # on the strength of "the server was called".
                #
                # Inference from a MISSING row cannot substitute: the primer decision
                # happens once, at the session's first compressible result, while result
                # rows accrue for hours afterwards, so any `--since` window or ledger
                # rotation starting mid-session drops it and keeps the rest. A positive
                # record survives both, because a truncated window then simply falls back
                # to the estimate instead of publishing a fabricated zero.
                #
                # Gated on a wire form being present for the same reason the attach is: a
                # result carrying no terse marker owes no primer, so suppressing one is not
                # a fact worth recording. Checked across the WHOLE result, not just text --
                # `structuredContent` itself may be what got compressed (#141).
                self._primer_suppressed_logged = True
                deferred.append((
                    "primer ledger", "(primer)",
                    partial(self._emit_primer, PRIMER_CADENCE_ONCE,
                            self._primer_text, False),
                ))
            if primer_pending and not structured_present and marker_in_text:
                content.insert(0, {"type": "text", "text": self._primer_text})
                self._primer_sent = True
                changed = True
                # DEFERRED, not called here (review of #311). The decision to bill is
                # made inside this branch -- the branch that actually attached it, so
                # everything outside it (a result carrying `structuredContent`, one
                # with no terse marker, a session that never produces a compressible
                # result at all) writes nothing, which is the whole point of #286. But
                # the WRITE has to happen after `_local_lock` is released, like every
                # other sink: `append_stats` stats/rotates/reads/appends a file, and a
                # blocking one -- stalled mount, full disk mid-rotation -- would hold
                # this lock, freeze `note_request`, and wedge every later tools/call on
                # the connection. try/except catches a sink that RAISES and does
                # nothing for one that BLOCKS; see the note above `deferred`.
                deferred.append((
                    "primer ledger", "(primer)",
                    partial(self._emit_primer, PRIMER_CADENCE_ONCE,
                            self._primer_text),
                ))
            if self.stats is not None:
                deferred.append((
                    "stats", capture_tool,
                    partial(self._emit_stats, tool, emitted_pairs,
                            display_tool=capture_tool, diff_reason=diff_reason,
                            structured=structured_raw, structured_out=structured_out),
                ))

            if not changed:
                out = line
            else:
                # Re-serialize compactly. JSON-RPC is semantics, not formatting; no
                # newlines.
                out = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

        # Lock RELEASED. The reply is already decided, so a sink that blocks here delays
        # only this response's own return — it can no longer wedge `note_request` or any
        # later call. `_emit_audit`/`_emit_stats` swallow their own errors (their fail-open
        # contract); the capture callback does not, so it is guarded here.
        for label, sink_tool, run_sink in deferred:
            try:
                run_sink()
            except Exception as exc:  # noqa: BLE001 — sinks are never load-bearing
                self._warn_sink(label, sink_tool, exc)
        return out

    def _compress_or_diff(self, block: dict, tool: str, args_key: str = "",
                          force_lossless: bool = False) -> tuple[bool, str]:
        """Compress one block, preferring a lossless delta vs the prior same-tool result
        when it is smaller. Updates the per-tool diff base. Returns `(changed, reason)`,
        where `reason` is the Phase 1 instrumentation datum — a short label for WHY the
        diff did/didn't fire (no_prior | keyframe | emitted | not_smaller_same_args |
        not_smaller_diff_args | text_emitted | text_dropped | non_json | passthrough |
        error), for the ledger.

        Two labels the ledger carries do NOT originate here, so the enumeration above is
        not the whole value set: `diff_off` and `mirror_dropped` are both set by
        `transform_response`. `mirror_dropped` means the text block was deleted as a
        redundant `structuredContent` mirror (#128) and no diff decision was reached at
        all — which is why it displaces the `diff_off` a single-block result would
        otherwise carry.

        Fail-open: any error leaves the block untouched and state intact."""
        text = block["text"]
        try:
            applied = policy_mod.apply(text, tool, self.policy, drop_sink=self._drop_put,
                                       server=self.server_name,
                                       force_lossless=force_lossless)
            self._note_drop_origins(applied)
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
            self._note_drop_origins(applied)
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

    def _partial_join(self, content: list, tool: str,
                      force_lossless: bool = False
                      ) -> tuple[list[tuple[str, str]], list[str], bool] | None:
        """#140: fold each maximal run of >=2 ADJACENT JSON-object text blocks into one
        joined record array, leaving every other block (non-text, a JSON array, a
        non-JSON error string, or a lone object) in place and per-block compressed.

        Rebuilds `content` in place and returns `(emitted_pairs, capture_payloads,
        changed)`, or None when nothing folds (no run of >=2 object blocks clears the
        codec) — in which case `content` is left untouched and the caller keeps the plain
        per-block path.

        Establishes NO cross-call diff base. A partial layout's block boundaries can shift
        call to call (an error block appears, records move), so anchoring a diff on the
        folded subset needs its own keyframe accounting — deferred (#140). All diff state
        for the tool is dropped instead, the same discipline the mirror-drop and per-block
        paths use, so no stale base is diffed against and no partial base misleads the next
        result.

        Adjacency is measured over `content`, not over the text blocks alone: a non-text
        block (image / resource_link) breaks a run, because folding across it would reorder
        content the client may rely on."""
        def _is_record(b: Any) -> bool:
            if not (isinstance(b, dict) and b.get("type") == "text"
                    and isinstance(b.get("text"), str)):
                return False
            try:
                return isinstance(json.loads(b["text"]), dict)
            except (json.JSONDecodeError, TypeError, RecursionError):
                return False

        # Partition `content` into maximal runs of adjacent object-text blocks vs
        # everything else, preserving order. A run has len>=1; a non-record is its own
        # singleton segment.
        segments: list[list] = []
        run: list = []
        for b in content:
            if _is_record(b):
                run.append(b)
            else:
                if run:
                    segments.append(run)
                    run = []
                segments.append([b])
        if run:
            segments.append(run)

        # Decide each segment WITHOUT mutating content, so a run that only refuses
        # (a reserved marker or over-depth on its sub-array) leaves nothing half-applied.
        # A len>=2 segment is a record run by construction (singletons are the non-records).
        plan: list[tuple[str, Any]] = []
        folded_any = False
        for seg in segments:
            if len(seg) >= 2:
                raws = [b["text"] for b in seg]
                try:
                    applied, curr, _refuse = policy_mod.apply_joined(
                        raws, tool, self.policy, drop_sink=self._drop_put,
                        server=self.server_name, force_lossless=force_lossless)
                    self._note_drop_origins(applied)
                except Exception as exc:  # noqa: BLE001 — fail-open per run
                    if self.debug:
                        sys.stderr.write(f"[terse-proxy] {tool}: partial-join run "
                                         f"passthrough on error: {exc}\n")
                    applied, curr = None, None
                if applied is not None:
                    plan.append(("fold", (applied.text, curr, raws)))
                    folded_any = True
                    continue
            plan.append(("pass", seg))

        if not folded_any:
            return None

        # Apply the plan. This mutates leftover blocks in place (`b["text"] = new`) BEFORE
        # the atomic `content[:] = out` below, so its all-or-nothing property rests on
        # `_compress` being hard fail-open (it returns the original text on any error and
        # never raises) — a raise mid-loop would leave `content` half-compressed.
        out: list = []
        emitted_pairs: list[tuple[str, str]] = []
        capture_payloads: list[str] = []
        changed = False
        for kind, item in plan:
            if kind == "fold":
                text, curr, raws = item
                out.append({"type": "text", "text": text})
                emitted_pairs.append(("\n".join(raws), text))
                capture_payloads.append(json.dumps(curr, separators=(",", ":"),
                                                   ensure_ascii=False))
                changed = True
            else:
                for b in item:
                    if (isinstance(b, dict) and b.get("type") == "text"
                            and isinstance(b.get("text"), str)):
                        raw = b["text"]
                        new = self._compress(raw, tool, force_lossless=force_lossless)
                        if new != raw:
                            b["text"] = new
                            changed = True
                        emitted_pairs.append((raw, b["text"]))
                        capture_payloads.append(raw)
                    out.append(b)
        content[:] = out

        if self.diff:
            self.last.pop(tool, None)
            self.last_args.pop(tool, None)
            self.last_joined.pop(tool, None)
            self.since_keyframe.pop(tool, None)
            self.last_text.pop(tool, None)
            self.since_text_keyframe.pop(tool, None)
        return emitted_pairs, capture_payloads, changed

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
        line, or None to forward unchanged.

        The primer is assembled from this server's policy (#168), so it documents only the
        wire forms this server can emit — and is empty, injecting nothing, for a policy
        that can emit none. Idempotency keys on PRIMER_HEAD rather than the whole string,
        since the assembly varies per policy."""
        result = msg.get("result")
        if not isinstance(result, dict):
            return None
        primer = build_primer(self.policy, self.server_name)
        if not primer:
            if self.debug:
                sys.stderr.write("[terse-proxy] policy emits no terse wire form; "
                                 "no primer injected\n")
            return None
        existing = result.get("instructions")
        existing = existing if isinstance(existing, str) else ""
        if PRIMER_HEAD in existing:
            return None
        result["instructions"] = primer + (f"\n\n{existing}" if existing else "")
        # NOT recorded, deliberately (#311). This site emits unconditionally whenever the
        # policy yields a primer at all -- the same predicate `primer_liability` already
        # evaluates from the INSTALLED policy -- so inference here is exact and a ledger
        # row would cost bytes without adding knowledge. Only the LAZY attach is
        # unobservable from outside the process, and that is the one #286 got wrong.
        if self.debug:
            sys.stderr.write(f"[terse-proxy] injected terse format primer "
                             f"({len(primer)} chars) into initialize.instructions\n")
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
            self._note_drop_origins(applied)
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
            # Compare CANONICAL SERIALIZATIONS, not the parsed values. Python `==` treats
            # True == 1 and 1 == 1.0, so a block reading `{"ok":true,"n":1.0}` would count
            # as a faithful mirror of `{"ok":1,"n":1}` and be deleted — handing the model
            # `1` where the block said `true`. The guard's whole claim is that a block
            # carrying anything the typed field does not is never dropped; value-level
            # equality is a hole in it.
            mirrored = json.dumps(json.loads(text_blocks[0]["text"]),
                                  sort_keys=True, separators=(",", ":"))
            typed = json.dumps(result["structuredContent"],
                               sort_keys=True, separators=(",", ":"))
            match = mirrored == typed
        except (ValueError, TypeError, RecursionError):
            # `RecursionError` covers BOTH statements, and both can raise it: nesting deep
            # enough blows the C parser's stack on the way in, and a deep `==` recurses on
            # the way out. This method runs outside `_compress`'s fail-open wrapper, so an
            # escaping exception would take down a tool call rather than pass it through —
            # the one failure mode the proxy exists to never have. (json.JSONDecodeError is
            # a ValueError; the depth cap that motivates this is #79.)
            return None
        return text_blocks[0] if match else None

    def _compress_structured(self, result: Any, tool: str, *,
                             force_lossless: bool = False
                             ) -> tuple[str | None, str | None, bool]:
        """Run a result's `structuredContent` through the codec in place, when the matching
        rule opted in with `"structured": "compress"` (#128). Returns
        `(raw_serialization, out_serialization, rewrote)` — the field's size on the RAW and
        EMITTED sides of the ledger, which DIFFER only when it was actually compressed
        (#141), and `(None, None, False)` when absent. Equal on both sides when the field is
        left untouched.

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
            return None, None, False
        try:
            original = json.dumps(result["structuredContent"], separators=(",", ":"),
                                  ensure_ascii=False)
        except (TypeError, ValueError):
            return None, None, False          # unserializable: not ours to touch
        if self._structured_mode(tool) not in policy_mod.STRUCTURED_REWRITING:
            return original, original, False  # untouched, but still counted by the ledger
        emitted = self._compress(original, tool, force_lossless=force_lossless)
        if emitted == original:
            return original, original, False
        try:
            result["structuredContent"] = json.loads(emitted)
        except json.JSONDecodeError:
            # The codec's output is always JSON, so this cannot normally fire — but the
            # typed field is the one a client may hand straight to a schema validator, so
            # an unparseable replacement must never be written. Keep the original.
            return original, original, False
        return original, emitted, True

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
                evicted_handle, evicted = self.dropped.popitem(last=False)
                self._dropped_bytes_box[0] -= len(lossy_mod._serialize(evicted))
                # Evict the attribution with the value, never after it. A handle left
                # behind here would still be content-addressed over (tool, path, bytes),
                # so it could only ever be re-created with the SAME origin — but the map
                # would grow without bound alongside a store that is explicitly capped.
                self._drop_origin.pop(evicted_handle, None)

    def _note_drop_origins(self, applied: Any) -> None:
        """Record `handle -> (server, tool, rule path)` for the drops an `apply`/`apply_joined`
        call actually COMMITTED (#251), so a later `terse.retrieve` is billed to the rule —
        and to the peer — that caused it.

        The server label is captured HERE, at drop time, not read at retrieve time: under
        multiproxy `_route_call` answers every retrieve through `peers[0]`, so the answering
        Interceptor is almost never the dropping one and its label would name the wrong peer.

        Reads provenance off the returned `Applied` rather than through `drop_sink`: the
        staged sink is `dict.__setitem__`, which cannot take extra arguments, and widening
        the sink protocol would have broken every 2-arg test fake. `getattr` with a default
        keeps this tolerant of an `Applied`-shaped object from an older caller or a fake."""
        origins = getattr(applied, "drop_origins", None)
        if not origins:
            return
        with self._store_lock:
            for handle, (otool, opath) in origins.items():
                # Only attribute what actually reached the store. A drop whose value was
                # evicted between commit and here has nothing to retrieve, and recording it
                # would inflate the rule's drop count against retrieves that cannot happen.
                if handle in self.dropped:
                    self._drop_origin[handle] = (self._ledger_label, otool, opath)

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
        # Same trap as note_request: `or {}` only neutralises FALSY junk, so a truthy
        # non-object `params` (or `arguments`) raised AttributeError straight out of `fwd`
        # into the client->server pump THREAD and stopped forwarding for the session. This
        # path runs BEFORE note_request whenever the policy has a drop rule — which the
        # default policy does not, but a deployed one does — so it is the branch that
        # actually fires in production.
        raw_params = msg.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        if params.get("name") != lossy_mod.RETRIEVE_TOOL:
            return None
        mid = msg.get("id")
        raw_args = params.get("arguments")
        handle = raw_args.get("handle") if isinstance(raw_args, dict) else None
        if not isinstance(handle, str):
            handle = ""  # a malformed/absent handle can only ever be a miss below
        value = None
        with self._store_lock:
            hit = handle in self.dropped
            if hit:
                self.dropped.move_to_end(handle)  # a read refreshes recency
                value = self.dropped[handle]
            # Read the attribution under the SAME lock acquisition that read the value:
            # taken separately, an interleaved eviction between the two could drop the
            # origin and bill this retrieve to `unknown` even though it hit.
            origin = self._drop_origin.get(handle)
        if hit:
            # Serialized ONCE and reused for the ledger's size measurement below: the drop
            # store is capped at 8 MiB, so a large field would otherwise pay a full second
            # JSON encode purely to take a `len()`.
            served = lossy_mod._serialize(value)
            result: dict = {"content": [{"type": "text", "text": served}]}
        else:
            served = ""
            result = {"content": [{"type": "text",
                                   "text": (f"terse: dropped-field handle {handle!r} is no "
                                            "longer available (evicted, or the session "
                                            "reconnected). Re-run the original tool to get "
                                            "the value again.")}],
                      "isError": True}
        if self.debug:
            sys.stderr.write(f"[terse-proxy] answered {lossy_mod.RETRIEVE_TOOL} "
                             f"handle={handle!r} hit={hit}\n")
        # The drop rule's COST, billed to the rule that caused it (#251). Fail-open like
        # every other side-effect sink: a ledger problem must never turn a served retrieve
        # into a failed one, because the value is already resolved at this point.
        if self.stats_retrieve is not None:
            # A miss is ALWAYS unattributed by construction: `_drop_origin` is popped in
            # lockstep with eviction and cleared with the store, so every path that makes
            # `hit` false has already discarded the origin. Billed to this proxy's own
            # label rather than dropped — the call was spent either way, and hiding it
            # would under-count the cost side.
            oserver, otool, opath = origin if origin is not None else (
                self._ledger_label, lossy_mod.RETRIEVE_TOOL, "")
            try:
                self.stats_retrieve(oserver, otool, opath, hit, served)
            except Exception as exc:  # noqa: BLE001 — stats is never load-bearing
                self._warn_sink("stats", otool, exc)
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

    def _emit_primer(self, cadence: str, text: str, attached: bool = True) -> None:
        """Record a primer that actually went out, with the cadence of the site that sent
        it (#311, #286).

        Called from the two branches that make the decision, and only from inside them: the
        one that ATTACHED a primer (`attached=True`, its real size) and the one that DECLINED
        to, because the result carried `structuredContent` and the client would have discarded
        it unread (`attached=False`, zero). A session that simply never produces a
        compressible result writes nothing at all, which is correct -- it made no decision.

        Recording the refusal rather than staying silent about it is the whole of #286.
        Silence cannot be read: a window with no primer row is indistinguishable from one
        whose row aged out of a `--since` or a rotation, so `primer_liability` had to fall
        back to billing every called server a full primer.

        Same fail-open contract as `_emit_stats`: the callback owns its I/O and a failure
        here can never change what the client receives, so a dead ledger degrades the report
        rather than the proxy."""
        emit = self.stats_primer
        if emit is None:
            return
        # NOT wrapped in try/except here: this runs from the `deferred` drain loop, which
        # already catches and routes to `_warn_sink` under the "primer ledger" label. A
        # second catch here would swallow the error before the loop could see it, and the
        # warning would be lost entirely. The label is deliberately NOT "stats": `_warn_sink`
        # writes the first failure of each kind unconditionally and silences the rest, so
        # sharing a kind would let this consume the result ledger's one guaranteed warning,
        # and a dead ledger going silent is what #131 exists to prevent. It also avoids the
        # substring "stats skipped", which is what a reader (and a test) greps for to count
        # RESULT-ledger failures.
        emit(cadence, text, attached)

    def _emit_stats(self, tool: str, pairs: list[tuple[str, str]], *,
                    display_tool: str | None = None, diff_reason: str | None = None,
                    structured: str | None = None,
                    structured_out: str | None = None) -> None:
        """Hand the stats callback one
        (tool, raw, emitted, passthrough, diff_reason, structured, structured_out) per
        emitted block, for the payload-free savings ledger (stats.py). Same fail-open
        contract as capture/audit: the callback owns I/O and a failure can never change
        what the client receives. `pairs`/`tool`/`display_tool` as in `_emit_audit`. The
        diff decision is per-result, so `diff_reason` is attributed to every pair — which
        is exactly one pair on the joined path and the common single-block shape.

        `structured`/`structured_out` are the serialized `structuredContent` this result
        carried, if any, on the raw and emitted sides (they differ only when the typed field
        was itself compressed, #141). Per-RESULT, not per-block, so both are attributed to
        the first pair only — counting them once per block would inflate the very number
        this is meant to make honest (#128)."""
        stats = self.stats
        if stats is None:
            return
        shown_tool = display_tool if display_tool is not None else tool
        passthrough = not self.policy.select(tool, self.server_name).tiers
        for index, (raw, emitted) in enumerate(pairs):
            try:
                stats(shown_tool, raw, emitted, passthrough, diff_reason,
                      structured if index == 0 else None,
                      structured_out if index == 0 else None)
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


def _new_session_id() -> str:
    """A fresh id for one proxy process's run, used to scope captured result ids.

    The single point of nondeterminism on the capture path, minted at the EDGE
    (`run_proxy` / `run_multi_proxy`) so `Interceptor` and `capture_payload` stay pure —
    the same reason `captured_at` is stamped in one place (principle #31)."""
    return uuid.uuid4().hex[:8]


def _build_capture_and_audit(
    capture_dir: str | None, debug_log: str | None, session: str | None = None
) -> tuple[CaptureFn | None, Callable[[dict], None] | None]:
    """Build the (capture, audit) callback pair from --capture-dir/--debug-log, shared
    by `run_proxy` and `multiproxy.run_multi_proxy` (identical logic, differing only in
    which process's downstream target they're wired to).

    These callbacks own I/O and NOTHING else: a failure propagates to the caller. Both
    sinks are still strictly side effects — a read-only or full disk must never break
    the proxy — but that fail-open guarantee is enforced by the one caller that has the
    bookkeeping for it, `Interceptor` (see `_warn_sink`), which swallows the failure AND
    announces the first one of each kind. Catching here as well made that unconditional
    first warning dead code, so a dead sink stayed invisible without --debug (#131)."""
    capture: CaptureFn | None = None
    if capture_dir is not None:
        from .capture import capture_payload

        def capture(tool: str, raw: str, server: str | None = None,
                    result_id: Any = None) -> None:
            # A JSON-RPC id repeats across sessions (every client starts counting again),
            # and one corpus dir accumulates many sessions — so the id alone would collide
            # and fuse unrelated calls into one "result". `session` is minted per proxy
            # process by the caller; without it the id is dropped rather than stored
            # ambiguously.
            key = None if result_id is None or session is None else f"{session}:{result_id}"
            capture_payload(tool, raw, capture_dir, server=server, result_id=key)

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
    lazy_primer: bool = True,
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
    If the policy marks any rule `require_server_name`, omitting it is refused outright
    rather than silently falling through to a less restrictive rule.

    `lazy_primer` (#168 phase 2), default True, is the CLI's real default and not exposed
    as a flag — passed through from here only so a test that isn't about primer behavior
    can pin the old always-eager `Interceptor` shape (`lazy_primer=False`) instead of
    threading a leading primer block through every first-compressed-result assertion.

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

    # Fail fast, before a single tool is proxied, rather than fail open on the first
    # call to a tool a `require_server_name` rule was meant to guard. Without
    # `server_name`, `Policy._match_candidates` never synthesizes the server-qualified
    # candidate such a rule needs to ever match — the rule goes silently unreachable and
    # the tool falls through to the unmatched-tool default instead (full tiers,
    # `capture=True`). Refusing to start is the loud failure that gap should have had.
    #
    # `not server_name`, not `is None`: `_match_candidates` gates the same candidate on
    # `if server` (falsy), so `--server-name ""` is already just as unreachable there as
    # omitting the flag entirely — an `is None` check here would let an empty string
    # slip past this refusal into the exact silent-fallback gap it exists to close.
    if not server_name:
        needs_name = sorted({r.tool_glob for r in pol.rules if r.require_server_name})
        if needs_name:
            sys.stderr.write(
                "[terse-proxy] refusing to start: policy rule(s) "
                f"{needs_name} set \"require_server_name\": true, but no --server-name "
                "was given. Without it these rules can never match and the tools they "
                "guard fall through to the policy's permissive default. Pass "
                "--server-name <name>.\n")
            return 2

    capture, audit = _build_capture_and_audit(capture_dir, debug_log, _new_session_id())

    stats = None
    stats_retrieve = None
    stats_primer = None
    ledger_label = None
    if stats_log is not None:
        from .stats import (
            build_primer_writer,
            build_retrieve_writer,
            build_stats_writer,
            resolve_ledger_identity,
        )

        label = ledger_label = resolve_ledger_identity(server_name, cmd)
        stats = build_stats_writer(stats_log, label)
        # Same ledger identity as the result writer, so a drop rule's saving and its
        # retrieve cost group under one `server` key (#251).
        stats_retrieve = build_retrieve_writer(stats_log, label)
        # Same ledger identity again: the primer's cost has to group under the same
        # `server` key as the savings it is weighed against, or break-even compares two
        # different servers (#285 was that bug for the wrap label).
        stats_primer = build_primer_writer(stats_log, label)

    inter = Interceptor(pol, debug=debug, capture=capture, audit=audit, stats=stats,
                        stats_retrieve=stats_retrieve, stats_primer=stats_primer,
                        ledger_label=ledger_label,
                        server_name=server_name, lazy_primer=lazy_primer)

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
                # ANSWERING is deliberately ungated, while ADVERTISING (the tools/list gate
                # above) is gated per-server (#168). Answer >= advertise, matching what
                # multiproxy already does — and it is the asymmetry that matters: a retrieve
                # call arriving at a server this build thinks cannot drop is exactly the
                # symptom of `_glob_covers_server`'s unsound cases (#199), and forwarding it
                # downstream turns "one wasted paragraph" into an unredeemable handle and a
                # -32601 from a server that never had the tool. `answer_retrieve` costs
                # nothing when nothing was dropped: it returns a legible miss and swallows.
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
