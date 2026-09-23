"""Live savings ledger: payload-free per-result stats from the proxy + aggregation.

The measurement story had a gap: `terse measure` proves savings over a captured
corpus, and `--debug-log` records full raw->emitted replays, but neither answers
"how much did terse save in my real sessions?" — the debug log embeds raw tool
payloads (the same secrets exposure as capture), so nobody leaves it on.

This ledger stores ONLY sizes and decisions — never payload content — so it is safe
to leave always-on (the proxy default; `--no-stats` opts out). One JSON line per
tool-result block: ts, version (the terse that wrote it), server, tool, decision,
diff_reason, raw/out chars, raw/out cl100k tokens, structured_* sizes
(null when tiktoken is unavailable — `terse stats` then reports chars, showing the
gap explicitly rather than substituting, same contract as report.py). Writes are
fail-open side effects with the same contract as capture/audit: a full disk can
never affect forwarding.

One shared default file serves every proxy process: each append is a single
O_APPEND write far under PIPE_BUF, which POSIX keeps atomic, so concurrent proxies
interleave whole lines. Rotation renames the live file to `.1` at the size cap
(keeping one generation, so the ledger is bounded at ~2x the cap); a cross-process
rotation race is benign — rename is atomic and the loser just appends to the fresh
file. Timestamps are real wall-clock here (unlike the corpus, principle #31): a
"how much this week" query is inherently a time series.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

from ._secure_io import append_restricted, mkdir_restricted
from .tokenize import count_cl100k

MAX_LEDGER_BYTES = 10 * 1024 * 1024  # rotate the live file past this size

# Decision labels — derived by sniffing the emitted text, not by threading state out
# of the proxy's compression paths, so adding stats changed no compression logic.
PASSTHROUGH = "passthrough"  # policy has no tiers for this tool: terse hands off
UNCHANGED = "unchanged"      # compression ran but nothing smaller was emitted
DIFF = "diff"                # a cross-call delta shipped (JSON row/key or text diff)
COMPRESSED = "compressed"    # the full encoded form shipped (incl. keyframes)

_DIFF_MARKERS = ('"__terse_diff__"', '"__terse_textdiff__"')


def default_stats_log() -> Path:
    """$XDG_STATE_HOME/terse/stats.jsonl (fallback ~/.local/state) — the XDG home for
    machine-local, non-config state, which is exactly what a savings ledger is."""
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(state) / "terse" / "stats.jsonl"


def server_label(cmd: list[str]) -> str:
    """A short downstream identity for the ledger: the command basename for a stdio
    target, the host for an HTTP one. Purely a grouping key for `terse stats` — two
    wrapped servers' same-named tools shouldn't collapse into one row."""
    if not cmd:
        return "unknown"
    target = cmd[0]
    if "://" in target:
        return urlparse(target).netloc or target
    return Path(target).name or target


def resolve_ledger_identity(server_name: str | None, cmd: list[str]) -> str:
    """The identity a ledger record for this server is written under: `server_name` (the
    MCP config's own name for it) when the caller knows it, else a guess from the command
    basename (`server_label`) — which misreads a launcher-wrapped server (kb behind
    secret-broker's `sb-run` labels itself "sb-run") — #83.

    The ONE fallback rule terse uses for this identity. `proxy.py`'s live write path and
    `install_mcp.py`'s `mcp-status` drift detector both call this rather than each
    re-deriving `server_name or server_label(...)` — a review round caught the two copies
    diverging in principle (not yet in practice) before this existed."""
    return server_name or server_label(cmd)


def _ledger_version() -> str:
    """The terse that wrote this record, resolved once per process.

    Records carried no version, so a ledger spanning a week of releases could not be
    asked "did that change help?" — the only available axis was time, and time is
    confounded by payload mix (a handful of huge calls moves a day's average far more
    than any codec change). Cached at module scope because `build_record` is on the
    proxy's hot path and this is a constant for the process's lifetime.

    Forward-only by construction: records written before this field cannot be
    backfilled, and `aggregate` reports them as `unversioned` rather than guessing.
    """
    global _VERSION_CACHE
    if _VERSION_CACHE is None:
        from . import __version__
        _VERSION_CACHE = __version__
    return _VERSION_CACHE


_VERSION_CACHE: str | None = None


def canonical_tool(server: str, tool: str) -> str:
    """Strip a router's `<server>__` qualification when it merely repeats `server`.

    multiproxy qualifies a peer's tool names on a real cross-peer collision (#168), so
    the SAME tool appears in the ledger under two spellings depending on which topology
    handled it — `kb.read.get` behind a plain proxy, `kb__kb.read.get` behind a router.
    They aggregated as two rows, splitting one tool's stats and understating whichever
    row an operator happened to read.

    Only an EXACTLY redundant prefix is removed — the qualifier must equal this record's
    own server label. A genuine cross-peer qualification (`gh__search` recorded under
    server `kb`) is left alone, because there the prefix is the only surviving evidence
    of which peer answered.

    The strip is unconditional, NOT collision-gated against the downstream's advertised
    tool names: a server labelled `kb` that genuinely exports both `x` and a tool named
    literally `kb__x` would sum two distinct tools into one row. multiproxy cannot
    produce that case (it qualifies every peer tool unconditionally — `multiproxy.py`
    `f"{peer}{PREFIX_SEP}{bare}"`), so it needs a plain `terse proxy` whose command
    basename happens to match a real tool's own `__` prefix. Accepted rather than
    threading a live tool list into a pure report helper to defend against a name no
    observed server has.

    The `len(tool) > len(prefix)` guard is load-bearing: without it a tool named exactly
    `kb__` canonicalizes to the empty string and silently joins whatever other row is
    keyed on `""`.
    """
    prefix = f"{server}__"
    return tool[len(prefix):] if tool.startswith(prefix) and len(tool) > len(prefix) else tool


def classify_decision(raw: str, emitted: str, passthrough: bool) -> str:
    """What the proxy did with one result, derived from the texts alone. A keyframe is
    reported as `compressed` (it IS the full form) — the diff hit-rate is the metric
    this exists for, not keyframe accounting."""
    if passthrough:
        return PASSTHROUGH
    if emitted == raw:
        return UNCHANGED
    if any(m in emitted[:40] for m in _DIFF_MARKERS):
        return DIFF
    return COMPRESSED


def build_record(server: str, tool: str, raw: str, emitted: str,
                 passthrough: bool, diff_reason: str | None = None,
                 structured: str | None = None,
                 structured_out: str | None = None) -> dict[str, Any]:
    """One ledger line. Sizes and labels only — never payload content (the property
    that makes always-on safe). Token counts are None without tiktoken.

    `diff_reason` (Phase 1 instrumentation) records WHY the cross-call diff did or
    did not fire for this result — the datum that decides whether arg-keying the diff base
    is worth building. See proxy `_compress_or_diff` for the value set. None on older
    records (the field post-dates them) and on writers that don't supply it.

    `structured` / `structured_out` are the serialized `structuredContent` riding alongside
    this result (#128), on the RAW and EMITTED sides respectively. They are usually equal —
    terse leaves the typed field alone by default, so it costs full price on both sides, and
    `raw_chars`/`out_chars` are the whole result's cost, not the text block's alone. Without
    that a `structuredContent`-emitting tool reported a text-block reduction the model never
    received, because the client reads the typed field and discards the block terse
    compressed (`scripts/probe/structured_content/`).

    Since #134 the field can ITSELF be compressed (`"structured": "compress"/"replace"`), and
    then the two sides DIFFER — the raw side must carry the original size, the out side the
    compressed one. Charging the compressed size to both (the pre-#141 bug) understated the
    real wire saving by ~15 points whenever the typed field was compressed. `structured_out`
    defaults to `structured` so a caller that passes one value — an untouched field, and
    every record written before the split — still lands the same size on both sides, which
    is exactly right for that case. `structured_chars` (raw side) and `structured_out_chars`
    (emitted side) keep the split recoverable.

    `decision` stays keyed on the TEXT BLOCK alone — it names what terse did, and terse
    genuinely did compress it. The size fields say what that was worth."""
    extra_raw = structured or ""
    extra_out = structured_out if structured_out is not None else extra_raw
    # Tokenize each side once and reuse — the same value feeds both the folded *_tokens
    # total and the recoverable structured_*_tokens split.
    extra_raw_tok = count_cl100k(extra_raw)
    extra_out_tok = count_cl100k(extra_out)
    return {
        "ts": int(time.time()),
        # The writer's version, so a later "did release X help?" is answerable from the
        # ledger instead of from a time slice that payload mix confounds.
        "version": _ledger_version(),
        "server": server,
        "tool": tool,
        "decision": classify_decision(raw, emitted, passthrough),
        "diff_reason": diff_reason,
        "raw_chars": len(raw) + len(extra_raw),
        "out_chars": len(emitted) + len(extra_out),
        "raw_tokens": _sum_tokens(count_cl100k(raw), extra_raw_tok),
        "out_tokens": _sum_tokens(count_cl100k(emitted), extra_out_tok),
        "structured_chars": len(extra_raw),
        "structured_out_chars": len(extra_out),
        # Both token sides recorded, mirroring the char split above — the emitted side is
        # what a downstream needs to recover the structured field's own token saving.
        "structured_tokens": extra_raw_tok if extra_raw else 0,
        "structured_out_tokens": extra_out_tok if extra_out else 0,
    }


RETRIEVE_EVENT = "retrieve"

PRIMER_EVENT = "primer"

# The two primer cadences, public because the WRITE sites (proxy, multiproxy) have to name
# one and the read side buckets on it. `_PER_TURN`/`_ONCE` below are aliases, not copies:
# the report has always spelled these strings and a second literal would be a silent
# divergence waiting to happen the first time one side is reworded.
PRIMER_CADENCE_PER_TURN = "per-turn"
PRIMER_CADENCE_ONCE = "once/session"


def build_primer_record(server: str, *, cadence: str, primer: str,
                        attached: bool = True) -> dict[str, Any]:
    """One ledger line for a primer that was ACTUALLY emitted (#311, #286).

    `primer_liability` sizes the primer from the INSTALLED policy and then uses the ledger
    only to decide who was called. Its own docstring concedes what that cannot see: a
    session whose every compressible result also carried `structuredContent` never reaches
    the lazy attach (`proxy.py`'s guard), so the server was called, paid nothing, and was
    billed anyway. #286 is that bill observed in the wild -- `searxng-mcp` charged 312
    tok/session for a primer it is structurally incapable of sending.

    No read-side cleverness can recover this, because the fact is only known at the attach
    site. So the attach site records it. That is the whole design, and it is deliberately
    less than #312 attempted: no session id, no epoch id, no cross-process correlation.
    A record here says "this text went out, once, now" and nothing more.

    `cadence` is `_ONCE` or `_PER_TURN` -- stored rather than re-derived, because which one
    applies is a property of the SITE that emitted it (a lazy attach is once per session, an
    `initialize` injection is re-read every turn) and the reader cannot recover the site.
    The two are different units and must never be summed; storing the label is what keeps a
    consumer from having to guess.

    `attached=False` records the OPPOSITE fact: a primer this proxy decided NOT to send,
    because the result carried `structuredContent` and a text block inserted beside it would
    be discarded by the client unread. That row is the whole of #286. Its `tokens` is 0
    because nothing went out -- the cost really is zero, and the row exists to say so.

    Recording the suppression rather than inferring it from a MISSING row is deliberate and
    was arrived at the hard way (#317, redesigned after review). Inference cannot work: the
    primer decision happens once, at a session's first compressible result, while result
    rows accrue for hours after it, so any `--since` window or ledger rotation that starts
    mid-session drops the primer row and keeps the rest. Absence therefore means "this
    window cannot say", permanently and unfixably. Presence means something. So the proxy
    writes down both answers and the reader never has to guess.

    Payload-free like every other record: the primer is measured and discarded. It is
    policy-derived text the operator's own configuration produced, never tool output.

    Deliberately carries NO `raw_chars`/`out_chars`. `aggregate` skips any row lacking both
    ("not a ledger record"), so a primer can never be counted as a compressed block or fold
    its bytes into the published savings percentage. That skip is load-bearing, not
    incidental -- see `test_a_primer_record_never_enters_the_savings_total`.
    """
    return {
        "ts": int(time.time()),
        "version": _ledger_version(),
        "server": server,
        "event": PRIMER_EVENT,
        # Which cadence this site pays on. NOT a total: `_PER_TURN` rows are billed again
        # every turn by the client re-reading `instructions`, `_ONCE` rows are not.
        "cadence": cadence,
        # Whether the primer actually went out. False is not a failure -- it is a
        # measurement of zero, and the ONLY thing that can distinguish "this server never
        # pays a primer" from "this window does not contain the row".
        "attached": bool(attached),
        # 0 on a suppressed row: nothing was sent, so nothing was spent. Not `None` --
        # `None` means "emitted, size unknown", which is a different claim entirely.
        "bytes": len(primer) if attached else 0,
        # None (not 0) without tiktoken, matching `build_record`: unknown is not zero.
        "tokens": count_cl100k(primer) if attached else 0,
    }


def build_retrieve_record(server: str, tool: str, path: str, *,
                          hit: bool, payload: str = "") -> dict[str, Any]:
    """One ledger line for a `terse.retrieve` round-trip — a drop rule's COST (#251).

    Until this existed the ledger measured only the saving side of `drop-to-retrieve`: the
    tokens a dropped field never spent. It could not see the model spending a whole extra
    tool call to fetch that field back, so a rule that drops a field the model always needs
    was indistinguishable from one that drops a field it never needs. That comparison is
    what `terse tune` needs to retune a drop rule from evidence rather than judgement.

    Payload-free like every other record: `payload` is measured and discarded exactly as
    `build_record` measures `raw`, the rule's `path` is policy text the operator wrote, and
    the value itself never reaches the file. `hit=False` is a handle that had been evicted
    or predates a reconnect — it cost the model a call and returned nothing, which is the
    worst outcome there is and has to stay visible.

    A miss is **unattributable by construction**, and the report says so rather than
    implying otherwise: `_drop_origin` is popped in lockstep with eviction and cleared with
    the store, so every path that produces `hit=False` has already discarded the origin. A
    real miss therefore always lands on the `(unattributed)` row, never beside a rule. This
    function still accepts a `path` with `hit=False` because it is a pure record builder and
    pinning that combination is how the field's independence is tested — but the proxy
    cannot produce it.

    Deliberately carries NO `raw_chars`/`out_chars`. `aggregate` skips any row lacking
    both (`"not a ledger record"`), so a retrieve can never be counted as a compressed
    block or fold its bytes into the published savings percentage. That skip is load-
    bearing, not incidental — see `test_a_retrieve_record_never_enters_the_savings_total`.
    """
    return {
        "ts": int(time.time()),
        "version": _ledger_version(),
        "server": server,
        "tool": tool,
        "event": RETRIEVE_EVENT,
        # The policy rule path that dropped the value ('$text.code_blocks', '$.a.b'), so a
        # tool carrying two drop rules bills each separately.
        "path": path,
        "hit": bool(hit),
        # What the round-trip cost in the model's context: the bytes/tokens it had to
        # re-read to get the dropped value back. On a miss both are 0 — nothing came back.
        "bytes": len(payload),
        # None (not 0) without tiktoken, matching `build_record`: unknown is not zero.
        "tokens": count_cl100k(payload),
    }


def _sum_tokens(a: int | None, b: int | None) -> int | None:
    """None means "not tokenized" and must stay None — `aggregate` reports those rows
    separately as `untokenized` rather than blending them into a total, so silently
    coercing a None to 0 here would move an unknown into the known column."""
    if a is None or b is None:
        return None
    return a + b


def append_stats(record: dict[str, Any], log_path: str | Path,
                 max_bytes: int = MAX_LEDGER_BYTES) -> None:
    """Append one record, rotating the live file to `.1` once it passes `max_bytes`.
    Restricted perms for consistency with every other terse-managed file, even though
    records are payload-free."""
    p = Path(log_path)
    mkdir_restricted(p.parent)
    try:
        if p.stat().st_size >= max_bytes:
            p.replace(p.with_name(p.name + ".1"))
    except OSError:
        pass  # no live file yet, or a concurrent proxy already rotated it
    # Self-heal after a torn tail (a writer that died mid-line): without this, the next
    # record concatenates onto the torn fragment and a GOOD record becomes unparseable
    # too. A racing double-heal just writes a blank line, which load_stats skips.
    prefix = ""
    try:
        with p.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                prefix = "\n"
    except OSError:
        pass  # no live file yet (or empty) — nothing to heal
    append_restricted(p, prefix + json.dumps(record, ensure_ascii=False) + "\n")


def load_stats(log_path: str | Path, since_ts: int | None = None) -> list[dict[str, Any]]:
    """Every record from the rotated generation + the live file, in append order,
    optionally filtered to ts >= since_ts. Unparseable lines are skipped (a torn line
    from a crashed writer must not sink the whole report)."""
    p = Path(log_path)
    records: list[dict[str, Any]] = []
    for path in (p.with_name(p.name + ".1"), p):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if since_ts is not None and not (isinstance(rec.get("ts"), int)
                                             and rec["ts"] >= since_ts):
                continue
            records.append(rec)
    return records


_WINDOW = re.compile(r"^(\d+)([smhdw])$")
_WINDOW_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_window(window: str) -> int:
    """`30m`/`24h`/`7d` -> seconds. Raises ValueError on anything else."""
    m = _WINDOW.match(window.strip())
    if m is None:
        raise ValueError(f"bad --since window {window!r} — use e.g. 30m, 24h, 7d")
    return int(m.group(1)) * _WINDOW_SECONDS[m.group(2)]


def _context_tokens(rec: dict[str, Any], raw_t: int, out_t: int) -> tuple[int, int]:
    """One record's raw/out token pair on the CONTEXT basis — what the MODEL received —
    where `raw_t`/`out_t` are the wire pair `aggregate` already validated as ints.

    `build_record` folds the text block and the typed field into one figure. That is a
    deliberate WIRE basis and #141 fixed the opposite bug, so it is not changing. But terse
    also knows the mirror text block never reaches the model: `policy.py` records the
    measurement — raw 2,596 chars in context -> "compress" 1,008 -> "replace" 1,008, no
    change, because the client had ALREADY discarded the block. `"replace"` saves stdio
    bytes and nothing at all off the model's context.

    So the split is:

      typed field present   the model reads the typed field ALONE, and `structured_tokens`
                            / `structured_out_tokens` are recorded (#134/#141) precisely so
                            that is recoverable. No new ledger field is needed.
      no typed field        the text block IS the context, so context == wire.

    WHAT THIS ASSUMES, stated because it is load-bearing for about a third of the ledger.
    A record says a typed field was present; it never says which client read it. Partitioned
    over the live ledger (4,479 tokenized rows, 2026-09-23):

        2,266  no typed field                          the block IS the context
          825  typed field COMPRESSED (out < raw)      the field was rewritten
        1,388  typed field present but untouched       says nothing either way

    (measured 2026-09-23 over 4,479 tokenized rows; the three parts sum to the total, which
    is the only part of this worth asserting — the counts drift as the ledger grows.)

    The middle group is NOT proof of a safe client, and an earlier draft of this comment
    said it was. `structured_mode_for_client` gates only `"auto"` on
    `STRUCTURED_SAFE_CLIENTS`; an explicit `"compress"` or `"replace"` in a rule is returned
    unchanged for any client, including none at all. So a rewritten field is a property of
    the POLICY, not of the client — it happens to imply a safe client on this fleet only
    because all nine rules resolve through `auto`.

    The third group is ambiguous in both directions and no reading of the record settles it:
    `auto` resolves to "leave" for a non-safe client, an explicit `"leave"` rule is
    indistinguishable, and `compress` on an already-compact field also leaves the size
    unchanged. Reading every one of them the other way (both fields forwarded) moves the
    ledger-wide saving from 1,423,614 to 1,470,859 and `turns_covered` from 2,836 to 2,930 —
    a 3.3% band, three orders of magnitude clear of the 1.0 break-even threshold.

    THE BAND IS A FACT ABOUT THIS FLEET, not about the basis. On a fleet whose rules name
    `"compress"` explicitly, the middle group stops being evidence too, and reading every
    typed row the other way returns the whole distance to wire — a 92.5% band, not 3.3%.
    The conclusion survives there (no verdict flips at 1,700x coverage) but the argument
    does not, so do not quote the 3.3% as a property of the method.

    So this takes the typed-only reading and states the band. A published RANGE was
    rejected: it doubles the surface of every figure for an uncertainty that changes no
    verdict here. A ledger `client` field was rejected because it cannot repair the records
    already written.

    A MULTI-BLOCK result is a known hole, and the partition above quietly includes it.
    `_emit_stats` attaches `structured`/`structured_out` to the first pair only ("per-RESULT,
    not per-block" — counting them once per block would inflate the very number they exist
    to make honest), so blocks 1..N-1 of such a result carry no typed-field split and land
    in the first group above. They are typed-field rows wearing a no-typed-field record,
    and their text is counted as model context when the client discarded all of it. Live
    magnitude: 3 follower rows, 39 raw tokens. Structural, not material, and it wants a
    per-result grouping the ledger cannot express today (#437).

    Untokenized is never zero. A record whose typed field exists but whose
    `structured_tokens` is not an int (no tiktoken at write time) cannot state its context
    payload, so it falls back to the wire pair rather than claiming a zero-token field."""
    st, so = rec.get("structured_tokens"), rec.get("structured_out_tokens")
    if not isinstance(st, int) or st <= 0:
        return raw_t, out_t
    # `structured_out_tokens` defaults to the raw side for a field terse left alone, exactly
    # as `build_record` does — a record predating the split carries only the one value.
    return st, so if isinstance(so, int) else st


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the ledger up: overall totals, per server/tool rows, decision counts.

    Token totals sum only records where BOTH token counts are present (a mixed sum
    would silently blend tokenizers-with-chars); `untokenized` counts the rest so the
    report can show the gap instead of hiding it. Char totals always cover everything.
    """
    # `blocks`, not `results`: one ledger record is emitted per tool-result text BLOCK
    # (the proxy tees `_emit_stats` per emitted pair), so a multi-block result contributes
    # N and a joined/partial one contributes 1 per folded unit. Naming it `blocks` keeps
    # the count honest — it moves with join behaviour by design, not by call volume (#141).
    total = {"blocks": 0, "raw_chars": 0, "out_chars": 0,
             "raw_tokens": 0, "out_tokens": 0, "untokenized": 0, "unversioned": 0,
             # The same traffic on a CONTEXT basis (#420) — see `_context_tokens`. Beside
             # the wire figures, never instead of them: what left the pipe and what the
             # model was charged for are both real, and #141 was the bug of publishing one
             # while meaning the other.
             "ctx_raw_tokens": 0, "ctx_out_tokens": 0}
    decisions: dict[str, int] = {}
    # Which terse wrote each record (forward-only — see `_ledger_version`), carrying the
    # SAME token sums as the per-tool rows. A `{version: count}` counter would have been
    # cheaper and useless: block counts across versions say how much traffic each build
    # handled, not what it saved, so the "did that release help?" question this field
    # exists for would still only be answerable through the payload-mix-confounded time
    # axis it was added to replace.
    versions: dict[str, dict[str, int]] = {}
    # Phase 1: why the cross-call diff did/didn't fire (only present on newer records).
    diff_reasons: dict[str, int] = {}
    tools: dict[tuple[str, str], dict[str, int]] = {}
    # Drop-rule COST rows (#251): a `terse.retrieve` round-trip, keyed by the rule that
    # caused the drop. Deliberately accumulated in its own map, never folded into `tools`
    # — a retrieve is not a compressed block, and adding it to a tool's `blocks` would
    # make the same call show up twice in the saving denominator.
    retrieves: dict[tuple[str, str, str], dict[str, int]] = {}
    # Primer-emission rows (#311): a primer that was ACTUALLY sent, keyed by the ledger
    # label that sent it. Its own map for the same reason `retrieves` has one -- a primer
    # is not a compressed block, and it must never reach the saving denominator. Keyed by
    # (server, cadence) because the two cadences are different units and summing them is
    # the exact error `primer_liability` splits its report to avoid.
    # Keyed by (server, cadence, attached): an attach and a suppression are opposite facts
    # about the same server and must never merge into one row. That third key is what makes
    # "this server provably pays nothing" expressible at all (#286).
    primers: dict[tuple[str, str, bool], dict[str, int]] = {}
    for rec in records:
        if rec.get("event") == PRIMER_EVENT:
            psrv = str(rec.get("server", "unknown"))
            # A row with no `attached` key predates the field and can only be an attach --
            # the suppression row did not exist then. Defaulting to True keeps an old ledger
            # reading exactly as it did.
            # `is not False`, NOT `bool(...)`. `.get` returns the default only when the key
            # is ABSENT: an explicit `"attached": null` returns None, and `bool(None)` is
            # False -- so a row that is self-evidently an attach (496 tokens, 992 bytes) was
            # bucketed as a suppression and the entry published a measured zero. Same
            # hand-edited/foreign-writer threat class the `rec_tok > 0` guard defends
            # against. Only an explicit `false` means the primer was declined.
            pkey = (psrv, str(rec.get("cadence", "unknown")),
                    rec.get("attached", True) is not False)
            prow = primers.setdefault(pkey, {"emissions": 0, "tokens": 0,
                                             "bytes": 0, "untokenized": 0})
            # Counts recorded DECISIONS, not emissions, on an `attached: false` row -- the
            # decision there was to send nothing. The name predates the suppression row and
            # is kept because it is a published `--json` field; `attached` is what tells a
            # consumer which sense applies, and `primer_liability` never divides by an
            # emission count from a suppressed row.
            prow["emissions"] += 1
            pb = rec.get("bytes")
            if isinstance(pb, int):
                prow["bytes"] += pb
            pt = rec.get("tokens")
            # None-is-not-zero, same as every other record: a ledger written without
            # tiktoken knows the primer went out but not what it cost.
            if isinstance(pt, int):
                prow["tokens"] += pt
            else:
                prow["untokenized"] += 1
            continue
        if rec.get("event") == RETRIEVE_EVENT:
            rsrv = str(rec.get("server", "unknown"))
            rkey = (rsrv, canonical_tool(rsrv, str(rec.get("tool", "unknown"))),
                    str(rec.get("path", "")))
            rrow = retrieves.setdefault(rkey, {"calls": 0, "hits": 0, "misses": 0,
                                               "bytes": 0, "tokens": 0, "untokenized": 0})
            rrow["calls"] += 1
            rrow["hits" if rec.get("hit") else "misses"] += 1
            rb = rec.get("bytes")
            if isinstance(rb, int):
                rrow["bytes"] += rb
            rt = rec.get("tokens")
            # Same None-is-not-zero discipline as the result rows: a record written
            # without tiktoken carries an unknown, and blending it in as 0 would
            # understate the very cost this row exists to expose.
            if isinstance(rt, int):
                rrow["tokens"] += rt
            else:
                rrow["untokenized"] += 1
            continue
        raw_c, out_c = rec.get("raw_chars"), rec.get("out_chars")
        if not (isinstance(raw_c, int) and isinstance(out_c, int)):
            continue  # not a ledger record
        total["blocks"] += 1
        total["raw_chars"] += raw_c
        total["out_chars"] += out_c
        decision = str(rec.get("decision", "unknown"))
        decisions[decision] = decisions.get(decision, 0) + 1
        reason = rec.get("diff_reason")
        if isinstance(reason, str):
            diff_reasons[reason] = diff_reasons.get(reason, 0) + 1
        srv = str(rec.get("server", "unknown"))
        # Canonicalized at READ time, not at write time: this also repairs the 2,000+
        # records already on disk, which a writer-side fix could never reach.
        key = (srv, canonical_tool(srv, str(rec.get("tool", "unknown"))))
        ver = rec.get("version")
        vrow = None
        if isinstance(ver, str):
            vrow = versions.setdefault(ver, {"blocks": 0, "tokenized": 0,
                                             "raw_tokens": 0, "out_tokens": 0})
            vrow["blocks"] += 1
        else:
            # Counted, never bucketed under a "None" key: a record written before the
            # field existed is an unknown writer, not a writer named None.
            total["unversioned"] += 1
        row = tools.setdefault(key, {"blocks": 0, "tokenized": 0, "encoded": 0,
                                     "raw_tokens": 0, "out_tokens": 0,
                                     "ctx_raw_tokens": 0, "ctx_out_tokens": 0,
                                     "raw_chars": 0, "out_chars": 0, "diffs": 0})
        row["blocks"] += 1
        # Blocks on which a terse WIRE FORM shipped, as opposed to blocks emitted at all.
        # `unchanged` ran the codec and shipped the original; `passthrough` never ran it.
        # Only these two decisions can put a `__terse_` marker on the wire — which is what
        # the lazy primer attaches to (`proxy.py`'s attach guard), so `primer_liability`
        # needs the distinction to tell a server that was merely CALLED from one whose
        # primer could actually have fired. An UPPER bound in one direction and sound in
        # the other: a minify-only `compressed` block carries no marker either, so a
        # non-zero count does not prove the primer attached, but a zero count proves it
        # could not have.
        # Stated as an EXCLUSION — "not one of the two decisions that rule a marker out" —
        # rather than as "one of the two that may have shipped one". They are the same test
        # for any record terse wrote (`classify_decision` returns exactly one of the four)
        # and differ only on a record whose `decision` this function could not read, which
        # `aggregate` tolerates as `"unknown"` a few lines up. Counting an unknown as encoded
        # over-bills that server's primer; NOT counting it under-bills, and under-billing is
        # the direction `_cadence` argues is unsafe (it moves a paying server into `free`,
        # where the operator reads "costs nothing at all"). No record terse has ever written
        # lacks the field — 0 of 2,115 in the live ledger — so this only ever decides a
        # hand-written or third-party line, where the conservative answer is the right one.
        #
        # NOT a proof that the excluded two shipped no marker: the lazy-primer attach fires
        # on `'"__terse_'` anywhere in the final content, which can come from the DOWNSTREAM
        # payload, so a `passthrough` result quoting a terse wire form attaches the primer
        # while classifying as `passthrough`. `_cadence`'s docstring carries that caveat;
        # this counter is evidence, not a demonstration.
        if decision not in (PASSTHROUGH, UNCHANGED):
            row["encoded"] += 1
        row["raw_chars"] += raw_c
        row["out_chars"] += out_c
        if decision == DIFF:
            row["diffs"] += 1
        raw_t, out_t = rec.get("raw_tokens"), rec.get("out_tokens")
        if isinstance(raw_t, int) and isinstance(out_t, int):
            total["raw_tokens"] += raw_t
            total["out_tokens"] += out_t
            row["raw_tokens"] += raw_t
            row["out_tokens"] += out_t
            # Accumulated in the SAME branch, so both bases cover exactly the same record
            # set. Summing the context basis over a wider set than the wire one would make
            # the two incomparable, which is the whole point of publishing them together.
            ctx_raw, ctx_out = _context_tokens(rec, raw_t, out_t)
            total["ctx_raw_tokens"] += ctx_raw
            total["ctx_out_tokens"] += ctx_out
            row["ctx_raw_tokens"] += ctx_raw
            row["ctx_out_tokens"] += ctx_out
            # Per-row, not just `total["untokenized"]`: the #175 break-even divides this
            # row's savings by its call count, and `blocks` counts untokenized records the
            # token sums skipped. Only this counter makes the two halves the same row set.
            row["tokenized"] += 1
            if vrow is not None:
                vrow["raw_tokens"] += raw_t
                vrow["out_tokens"] += out_t
                # Same tokenized/blocks split as the per-tool rows: a per-version rate
                # divided by `blocks` would be diluted by exactly the untokenized share.
                vrow["tokenized"] += 1
        else:
            total["untokenized"] += 1
    return {"total": total, "decisions": decisions, "diff_reasons": diff_reasons,
            "versions": versions,
            # Primers actually emitted this window, by label and cadence (#311). Empty on
            # every ledger written before this shipped -- readers MUST treat empty as "this
            # ledger cannot say", not as "no primer was sent".
            "primers": [{"server": s, "cadence": c, "attached": a, **row}
                        for (s, c, a), row in sorted(primers.items())],
            # Costliest rule first: tokens the model spent fetching dropped values back.
            "retrieves": [{"server": s, "tool": t, "path": p, **row}
                          for (s, t, p), row in sorted(
                              retrieves.items(),
                              key=lambda kv: kv[1]["tokens"], reverse=True)],
            "tools": [{"server": s, "tool": t, **row}
                      for (s, t), row in sorted(
                          tools.items(),
                          key=lambda kv: kv[1]["raw_tokens"] - kv[1]["out_tokens"],
                          reverse=True)]}


def _pct_saved(raw: int, out: int) -> str:
    return f"{(raw - out) / raw * 100:5.1f}%" if raw else "    –"


def _hit_rate(diffs: int, blocks: int) -> str:
    """diffs / blocks as a percent — the cross-call diff hit rate this ledger exists
    to measure (a raw count alone is meaningless without its denominator). Blank on a
    zero denominator, which can't happen per-row but keeps the helper total."""
    return f"{diffs / blocks * 100:4.0f}%" if blocks else "    –"


def build_stats_report(agg: dict[str, Any], *, log_path: str | Path,
                       window: str | None = None,
                       liability: dict[str, Any] | None = None) -> str:
    """Human-readable rollup. Tokens are the headline when available; chars are the
    honest fallback, labeled as such (never silently presented as tokens)."""
    total, decisions, tools = agg["total"], agg["decisions"], agg["tools"]
    scope = f"last {window}" if window else "all time"
    lines = [f"terse stats — {scope}  (ledger: {log_path})", ""]
    if total["blocks"] == 0:
        if window:
            # The ledger isn't necessarily empty — the window filtered everything out.
            # Point at the window, not at "nothing ever recorded" (the wrong cause).
            lines.append(f"no results in the last {window} — widen --since or drop it "
                         "(older results may still be in the ledger).")
        else:
            lines.append("no results recorded — has a terse-wrapped server handled a "
                         "tool call since stats shipped?")
        # Still print the liability: an install with wrapped servers and NO recorded results
        # is the purest net-negative case there is, and suppressing the line here would hide
        # it behind "nothing to report" (#168).
        if liability:
            lines += build_primer_section(liability)
        return "\n".join(lines) + "\n"
    tok_raw, tok_out = total["raw_tokens"], total["out_tokens"]
    lines.append(f"blocks: {total['blocks']}   "
                 f"decisions: " + ", ".join(f"{k}={v}" for k, v in sorted(decisions.items())))
    diff_reasons = agg.get("diff_reasons") or {}
    if diff_reasons:
        # Phase 1: the diff hit-rate breakdown. `no_prior` = tool never re-called;
        # `not_smaller_diff_args` = base was a different-args call (arg-keying opportunity);
        # `not_smaller_same_args` = same-args base but the delta didn't win (encoding, not
        # keying); `emitted` = a JSON diff shipped; `text_emitted` = a CDC text diff
        # shipped; `keyframe` = forced full to re-anchor.
        lines.append("diff reasons: "
                     + ", ".join(f"{k}={v}" for k, v in sorted(diff_reasons.items())))
        # `diff_off` on every block is the single most misread line in this report: it looks
        # like a missing feature, and a reader who then finds no `diff` key in their policy
        # concludes diffing was never wired up (#181). It is implemented and deliberately
        # off (#170 — the 190-token primer paragraph outweighed what the tier saved at a
        # 0.38% hit rate; the ~900-2,700x once quoted here was computed against the pre-#211
        # per-turn charge, see `policy.py`). Say so where the question is actually asked,
        # not only in the dataclass.
        if diff_reasons.get("diff_off") and len(diff_reasons) == 1:
            lines.append("  (diff_off = cross-call diffing is OFF by default since #170 — "
                         "measured cost > saving; enable per server with `--diff`)")
    if tok_raw or tok_out:
        lines.append(f"tokens (cl100k): {tok_raw:,} -> {tok_out:,}   "
                     f"saved {tok_raw - tok_out:,} ({_pct_saved(tok_raw, tok_out).strip()})")
        if total["untokenized"]:
            lines.append(f"  ({total['untokenized']} result(s) uncounted — tiktoken "
                         f"unavailable when they were recorded; chars below cover them)")
    else:
        lines.append("tokens: unavailable (tiktoken not installed when recording) — "
                     "char totals below")
    lines.append(f"chars: {total['raw_chars']:,} -> {total['out_chars']:,} "
                 f"({_pct_saved(total['raw_chars'], total['out_chars']).strip()} saved)")
    lines.append("")
    # Mirror the header's unit choice per-row: tokens when the ledger has any, else
    # chars — otherwise a tiktoken-less ledger renders the whole (most useful) per-tool
    # table as a wall of zeros while the header above honestly shows char savings.
    use_tokens = bool(tok_raw or tok_out)
    raw_col, out_col = ("tok raw", "tok out") if use_tokens else ("chr raw", "chr out")
    lines.append(f"{'server':<18} {'tool':<34} {'blocks':>7} {'diffs':>5} "
                 f"{'diff%':>5} {raw_col:>10} {out_col:>10} {'saved':>6}")
    for row in tools:
        raw_n = row["raw_tokens"] if use_tokens else row["raw_chars"]
        out_n = row["out_tokens"] if use_tokens else row["out_chars"]
        lines.append(f"{row['server'][:18]:<18} {row['tool'][:34]:<34} "
                     f"{row['blocks']:>7} {row['diffs']:>5} "
                     f"{_hit_rate(row['diffs'], row['blocks']):>5} "
                     f"{raw_n:>10,} {out_n:>10,} "
                     f"{_pct_saved(raw_n, out_n):>6}")
    lines += build_retrieve_section(agg, use_tokens)
    lines += build_version_section(agg)
    if liability:
        lines += build_primer_section(liability)
    return "\n".join(lines) + "\n"


def build_retrieve_section(agg: dict[str, Any], use_tokens: bool = True) -> list[str]:
    """What each `drop-to-retrieve` rule COST — the other half of the lossy ledger (#251).

    Rendered, not merely carried in `--json`, for the same reason the version section is:
    a drop rule's saving has always been visible in the per-tool table above, so showing
    only that half systematically flattered every lossy rule. A rule whose field the model
    fetches back on most calls is not saving what the table above credits it with.

    Suppressed entirely when no retrieve was ever recorded — which is every ledger written
    before this shipped, and every install with no drop rule (the default). An empty
    section would read as "your drop rules cost nothing", a claim this data cannot make
    about a ledger that could not record it."""
    rows = agg.get("retrieves") or []
    if not rows:
        return []
    # Mirror the per-tool table's unit choice (`use_tokens`): on a tiktoken-less ledger the
    # token column is all zeros, and rendering only that turned the entire cost table into
    # a wall of nothing while the byte counts sitting right there were known. Same fallback,
    # same reason — a cost the operator cannot read is a cost they will not act on.
    cost_col = "tok" if use_tokens else "chr"
    out = ["", "drop-to-retrieve cost — round-trips the model spent fetching dropped values back:",
           f"{'server':<18} {'tool':<28} {'rule path':<22} {'calls':>6} "
           f"{'miss':>5} {cost_col:>9}"]
    for r in rows:
        # An empty path is a MISS (the origin is always gone by then — see
        # `build_retrieve_record`), or a handle stored without provenance: billed, never
        # hidden, but named as unknown rather than guessed at.
        path = r["path"] or "(unattributed)"
        cost = r["tokens"] if use_tokens else r["bytes"]
        out.append(f"{r['server'][:18]:<18} {r['tool'][:28]:<28} {path[:22]:<22} "
                   f"{r['calls']:>6} {r['misses']:>5} {cost:>9,}")
    misses = sum(r["misses"] for r in rows)
    if misses:
        # The worst outcome there is: the model spent a whole call and got nothing back,
        # so this is pure cost with no recovered value on the other side. Always on the
        # `(unattributed)` row — a miss cannot name the rule it came from.
        out.append(f"  ({misses} miss(es) — handle evicted or the session reconnected; "
                   "the call was spent and returned nothing. A miss cannot be attributed "
                   "to a rule: the origin is discarded with the value.)")
    untok = sum(r["untokenized"] for r in rows)
    if untok and use_tokens:
        out.append(f"  ({untok} retrieve(s) uncounted — tiktoken unavailable when recorded)")
    return out


def build_version_section(agg: dict[str, Any]) -> list[str]:
    """Per-writer-version savings — "did that release help?", asked directly.

    Rendered, not just carried in `--json`: the field's whole purpose is to replace a
    time-slice comparison an operator would otherwise do by eye, and one that only
    appears under `--json` does not replace anything.

    Suppressed entirely when no record carries a version, which is every ledger written
    before the field shipped — an all-`unversioned` table states the obvious at the cost
    of a screen. The `unversioned` count still prints whenever versioned rows exist
    beside it, because THAT is the case where a reader would otherwise mistake a partial
    table for the whole window.
    """
    versions = agg.get("versions") or {}
    if not versions:
        return []
    lines = ["", f"  {'version':<28} {'blocks':>7} {'tok raw':>10} {'tok out':>10} "
                 f"{'saved':>6}"]
    for ver, row in sorted(versions.items()):
        raw_n, out_n = row["raw_tokens"], row["out_tokens"]
        lines.append(f"  {ver[:28]:<28} {row['blocks']:>7} {raw_n:>10,} {out_n:>10,} "
                     f"{_pct_saved(raw_n, out_n):>6}")
    unversioned = (agg.get("total") or {}).get("unversioned") or 0
    if unversioned:
        lines.append(f"  ({unversioned:,} record(s) predate the version field and are NOT "
                     f"in this table — compare versions only within it.)")
    return lines


# --- primer liability (#168) ----------------------------------------------------------
#
# The ledger charges terse for the payloads it compresses and never for the context it
# adds, so `terse stats` can report a win in a session that was a net loss. A primer that
# rides `initialize.instructions` is re-read as `cache_read` EVERY turn, so its cost scales
# with (servers x turns) while the savings scale with (compressible tool calls). Measured
# from outside terse: a 14.0% win at one wrapped server, a 2.1% LOSS at three.
#
# #211 removed that scaling for standalone entries — the primer now attaches once, to the
# first compressible result — so this section reports the two cadences separately and never
# adds them. See `primer_liability`'s docstring for why summing them was the defect.
#
# What this deliberately does NOT do is charge a per-turn cost into the ledger. `turns` is
# not observable: a `terse proxy` is a stdio process that sees one `initialize` per process
# lifetime and then `tools/call` requests, and several calls can share a turn. Nothing in
# MCP reports the client's turn count. Inventing one would be the same defect family as
# #144/#186/#188 — a number describing something the code did not measure.
#
# So the output is a BREAK-EVEN statement instead: what each cadence costs, what the window
# banked, and how far those savings go against it. Same decision for the operator, no
# fabricated denominator.

# States that unconditionally run their own `terse proxy`/`multiproxy` and pay a primer.
# A `folded-and-live` peer is conditional: its live half may run its own proxy, or may be a
# raw re-add that pays nothing beyond the router's union primer. The render loop admits only
# the former, using the launch fields `scan_scopes` fills when it proves that proxy exists.
_PAYS_PRIMER = ("wrapped", "wrapped-unstashed", "router", "router-ambiguous")

# States whose entry WRITES ITS OWN LEDGER ROWS. A different question from the one above,
# and it was a real defect that they shared a filter (#309): a `folded-and-live` peer pays
# no primer of its own — correctly absent from `_PAYS_PRIMER` — but its live entry does run
# its own proxy and does write records under its own guessed label, so it is every bit as
# able to collide with another entry over a launcher basename as a `wrapped` one is.
#
# The `launches_via_terse` half of that condition is NOT repeated here, because it cannot
# be: this tuple sees only a state name. `scan_scopes` carries it instead — a
# `folded-and-live` row that is live as a raw re-add leaves `wraps`/`ledger_identity`
# None, so `_guessed_label` returns "" and the count below skips it anyway. Coupled on
# purpose, and pinned from both ends (see `test_install_mcp.py`'s scan-contract test as
# well as the collision tests here).
_WRITES_LEDGER_ROWS = _PAYS_PRIMER + ("folded-and-live",)

# Of those, the states that still prime EAGERLY at `initialize.instructions`, which the
# client re-reads every turn as `cache_read` — a recurring per-turn charge. Everything else
# in `_PAYS_PRIMER` is a standalone `run_proxy` entry, lazy since #211: one attach, to the
# first compressible result, and nothing at all if that result never comes.
_PRIMES_EAGERLY = ("router", "router-ambiguous")

# Command basenames that name a LAUNCHER, not a server. `server_label` of such a command is
# a label every unrelated `python -m ...` / `npx ...` wrap in the fleet shares, so reading
# ledger rows under it hands one server another server's savings — the manufactured-KEEP
# half of #285. Only ever consulted for a GUESSED label; an explicit `--server-name` is the
# server's own name and is trusted whatever it says.
_LAUNCHER_BASENAMES = frozenset({
    "node", "nodejs", "npx", "npm", "pnpm", "pnpx", "yarn", "bun", "bunx", "deno",
    "uv", "uvx", "pipx", "pip", "poetry", "pdm", "conda", "mise", "nix", "nix-shell",
    "ruby", "perl", "java", "dotnet", "sh", "bash", "zsh", "pwsh", "powershell",
    "env", "sudo", "docker", "podman",
})

# `python`, `python3`, `python3.12`, `py` — the versioned family the set above can't list.
# No anchors: every call site uses `fullmatch`, and a `$` here would quietly survive a
# future switch to `search` while turning names like `pypy-server` into false positives.
_LAUNCHER_RE = re.compile(r"(?:python|pypy|py)\d*(?:\.\d+)*", re.IGNORECASE)


def _is_launcher_basename(label: str) -> bool:
    """True for a command basename that identifies no server on its own. Case- and
    `.exe`-insensitive; deliberately narrow — `pymcp`, `pypi-server`, `node-thing`,
    `envoy` and `docker-mcp` all name a SERVER and are not launchers."""
    name = label.lower().removesuffix(".exe")
    return name in _LAUNCHER_BASENAMES or bool(_LAUNCHER_RE.fullmatch(name))


# How the CLIENT resolves a server defined in more than one scope: it connects once, to the
# highest-precedence definition. Lower number wins. An unknown scope sorts last — it cannot
# outrank a scope we do understand on the strength of not being recognised.
_SCOPE_PRECEDENCE = {"local": 0, "project": 1, "user": 2}


def _precedence_winner(scan_rows: list[dict[str, Any]]) -> dict[str, int]:
    """`{server name: index in scan_rows}` for the one row per name that the client would
    actually launch, among rows `eligible` accepts.

    De-duplicating by name is right in kind — the same server in two scopes is one server to
    the client, and counting it twice would manufacture a collision with itself — but both
    consumers did it FIRST-WINS over rows emitted user -> project -> local, so the row that
    took the slot was the **user** one: the single definition guaranteed not to be running
    (#398). A shadowed `kb` in user scope could therefore delete the measurement of an
    unrelated entry that legitimately owned every row under its label.

    Returns INDICES rather than rows, so a caller can keep iterating in emission order and
    skip the losers. The alternative — iterating in precedence order — would also reorder
    the report, which is a second change wearing the first one's clothes.

    EVERY row is admitted, whatever its state, and the two consumers filter afterwards.
    Twice now the alternative has been tried and has failed the same way:

    - Admitting only rows that GUESS a label let a shadowed user-scope entry beat the
      project-scope one the client launches — #398 one level in.
    - Giving each consumer its own admission set (`_WRITES_LEDGER_ROWS` for ambiguity,
      `_PAYS_PRIMER` for liability) made them disagree about which row speaks for a name:
      for a project-scope `folded-and-live` shadowing a user-scope `wrapped`, ambiguity
      picked the folded row (guessing nothing, so no collision) while liability rendered
      the shadowed one AND read its ledger blocks — two entries reporting the same blocks,
      #285's double count restored (review of PR #415).

    The client's choice is a fact about the FLEET, not about the question being asked of
    it, so it is computed once here. A consumer that cannot speak for the winning row must
    say nothing for that name — not fall through to a definition that is not running.

    This covers, but does not wholly subsume, the guard added in review of #309: within ONE
    scope precedence cannot separate two rows, so a same-scope row that guesses nothing can
    still take the slot from one that does. `scan_scopes` emits one row per (scope, server)
    and cannot produce that, but a hand-built row list can — see
    `test_two_rows_in_ONE_scope_keep_the_first`."""
    def rank(row: dict[str, Any]) -> int:
        # ONE spelling of the default, used for both sides of the comparison below. Two
        # copies is not a style point: mutating one of them left the other disagreeing, so
        # an unknown scope lost anyway and the mutant read as equivalent (review of #398).
        return _SCOPE_PRECEDENCE.get(str(row.get("scope")), len(_SCOPE_PRECEDENCE))

    best: dict[str, int] = {}
    for i, row in enumerate(scan_rows):
        name = row.get("server")
        if not name:
            continue
        name = str(name)
        prev = best.get(name)
        # Strict `<`: ties keep the FIRST row at that rank, which is the pre-existing
        # behaviour for two entries in one scope (impossible through the config format, but
        # a hand-merged file can carry it) and keeps this function order-stable.
        if prev is None or rank(row) < rank(scan_rows[prev]):
            best[name] = i
    return best


def _guessed_label(row: dict[str, Any]) -> str:
    """The ledger label a WRAPPED entry's records are GUESSED to carry — the downstream
    command's basename. Empty when the entry baked an explicit `--server-name` (nothing is
    guessed then, the flag says outright what the proxy writes) or there is no downstream
    to guess from."""
    if row.get("ledger_identity_explicit"):
        return ""
    ident = row.get("ledger_identity")
    if ident:
        return str(ident)
    wraps = row.get("wraps") or ""
    return server_label(wraps.split()) if wraps else ""


def _ambiguous_labels(scan_rows: list[dict[str, Any]],
                      ledger_path: str | None = None) -> set[str]:
    """Guessed launcher labels that MORE THAN ONE installed entry resolves to — the only
    state in which reading ledger rows under one is provably attributing another server's
    traffic (#285).

    Both halves are load-bearing. Launcher-ness alone is not ambiguity: a lone `node
    /srv/x.js` wrap owns every `node` row in the ledger, and dropping its label would
    delete a correct measurement from the one population that cannot fix it by re-running
    `install-mcp` — the first draft of this fix did exactly that. Sharing alone is not
    ambiguity either: two entries cannot share a NON-launcher basename without launching
    the same binary, which is one logical server and one honest label.

    De-duplicated by server NAME first: the same entry present in both project and user
    scope is one server to the client, and counting it twice would manufacture a collision
    with itself. A row that guesses NOTHING never takes that slot, though — it would
    SHADOW a same-named row in a lower-priority scope that does guess, silently deleting a
    real collision (found in review of #309, and reproduced: a raw `folded-and-live`
    re-add in user scope hid a genuine `python` collision between two project-scope
    entries, which the pre-#309 code caught).

    Which row wins the slot when several DO guess is `_precedence_winner`'s job (#398):
    rows arrive user → project → local and first-wins kept the USER row, while the client
    connects to the highest-precedence definition (local > project > user), so the kept row
    was the one guaranteed not to be running. Both this function and `primer_liability`
    share that helper now; they differ only in which rows are admitted to the contest."""
    winners = _precedence_winner(scan_rows)
    by_name: dict[str, str] = {}
    for i, row in enumerate(scan_rows):
        name = row.get("server")
        # `winners` already holds ONE index per name, so this needs no `seen` set of its
        # own — and a name whose winning row is in a state that writes no ledger rows
        # contributes nothing, rather than falling back to a definition that is not running.
        if not name or winners.get(str(name)) != i:
            continue
        if row.get("state") not in _WRITES_LEDGER_ROWS:
            continue
        # An entry that writes no rows into this ledger cannot be one of the two servers
        # fighting over a label (#397). Counting it flagged the label ambiguous and deleted
        # the measurement of the entry that legitimately owned every row under it.
        #
        # Skipping it HERE ALONE was tried in PR #395 and reverted with the measurement in
        # hand: `_wrapped_labels` still handed the same entry the label, so both entries
        # then reported the SAME blocks — #285's double count, relocated rather than fixed.
        # The two paths only make sense changed together, and they are: see
        # `_wrapped_labels`.
        if not _writes_ledger_rows(row, ledger_path):
            continue
        lbl = _guessed_label(row)
        if not lbl:
            continue
        by_name[str(name)] = lbl
    counts: dict[str, int] = {}
    for lbl in by_name.values():
        if lbl and _is_launcher_basename(lbl):
            counts[lbl] = counts.get(lbl, 0) + 1
    return {lbl for lbl, n in counts.items() if n > 1}


def _live_labels(scan_rows: list[dict[str, Any]]) -> set[str]:
    """Every ledger label some INSTALLED entry currently answers to — a router's peer names,
    and each wrapped entry's own identity or guess. The set `_superseded_labels` has to
    subtract, so it cannot hand one server another server's live rows."""
    live: set[str] = set()
    for row in scan_rows:
        state = row.get("state")
        wraps = row.get("wraps") or ""
        if state in ("router", "router-ambiguous"):
            live.update(p for p in (q.strip() for q in wraps.split(",")) if p)
            continue
        # Folded peers are the router's labels too — they write their own ledger records
        # even though they never pay a primer, so `_PAYS_PRIMER` is the wrong filter here.
        # `ident` ALONE is what makes a baked entry's command basename correctly absent
        # here: `resolve_ledger_identity` already collapsed the two cases, returning the
        # baked name for a baked entry (whose basename is exactly what is no longer live —
        # the thing `_superseded_labels` exists to name) and the basename itself for an
        # unbaked one. The `elif` is only the fallback for a row predating that field.
        ident = row.get("ledger_identity")
        if ident:
            live.add(str(ident))
        elif wraps:
            live.add(server_label(wraps.split()))
        name = row.get("server")
        if name and state and state.startswith("folded"):
            live.add(str(name))
    return live


class _Contest(NamedTuple):
    """Who writes one contested ledger label. `writers` is every installed entry; `standalone`
    is those that are not routers.

    Both sets, published once, because three consumers need different slices and re-deriving
    any of them is how this subsystem keeps breaking. Round 2 re-derived a claimant's label
    through `_wrapped_labels` and lost the `ambiguous` case; round 3 re-derived it inline and
    lost the `_writes_ledger_rows` gate, so a `--no-stats` entry was named as a writer and
    the remedy for the real collision was suppressed. The render loop asks "does THIS entry
    write a contested label" (`writers`), the primer path asks "is this entry the only
    standalone writing it" (`standalone`), and the report's remedy line asks the same
    question per label. One computation answers all three."""

    writers: frozenset[str]
    standalone: frozenset[str]


def _contested_labels(scan_rows: list[dict[str, Any]],
                      ledger_path: str | None = None) -> dict[str, _Contest]:
    """Ledger labels that MORE THAN ONE running terse process writes, each mapped to the
    NON-ROUTER entries writing it — provably unattributable, so no entry may count them
    (#396 review).

    The map, not just the set, because every consumer needs a different slice of it — see
    `_Contest`. One in particular: a measurement on a contested row survives. A primer record is written at exactly one site, `run_proxy`'s
    `build_primer_writer` — a router's peers are built `lazy_primer=False`, so no peer attach
    can fire, and the router's own union primer is eager and recorded nowhere. So a
    `once/session` primer row under a contested label can only have been written by a
    STANDALONE proxy answering to it. Where exactly one non-router entry writes the label,
    that record is fully attributable however many routers also claim it, and blacking it out
    turned a measured 777-token attach into `session_once_tokens: 0` under the line "Only
    servers that were actually called are billed here" (re-review finding 1).

    A multiproxy router tags each peer's records with that PEER's name. Any other entry
    running its own `terse proxy` under that same name writes records identical in
    provenance: two processes, one label, no field that distinguishes them. Summing the
    label's blocks and savings into both liability rows let each clear break-even off the
    other's traffic and report KEEP — #285's double count, one state over.

    **Counted by WRITER, not matched between two fixed roles.** The first draft intersected
    "a router's peers" with "a folded-and-live duplicate's label", which missed two shapes
    review then reproduced:

      * a duplicate at ANOTHER SCOPE. `folded` is computed per scope from that scope's peers
        file, so a peer folded behind a user-scope router and separately wrapped at project
        scope is `wrapped`, never `folded-and-live`. Two plain `install-mcp` runs reproduced
        the full double count (review of PR #422, finding 2). ONE ORIENTATION ONLY: when the
        duplicate sits at a LOWER-precedence scope than the router, the router's own scope
        contributes a `folded` row for that name which wins `_precedence_winner` and deletes
        the writer, so the contest never sees it. That is #424's mechanism and is not fixed
        here — do not read this bullet as closing the cross-scope case in general.
      * a SECOND ROUTER folding the same peer name. A router is itself an entry running a
        proxy under that name, but the intersection could only ever put a router on the
        left-hand side (finding 1 of the re-review). `--router-name` defeats
        `_precedence_winner`'s name dedup, so two distinct routers really can coexist.

    Both fall out of asking one question per label — *how many installed entries write it?*
    — instead of two. That also answers the `router-ambiguous` pair (two routers fronting
    one peers file), which was reported as a pre-existing hole in the same review.

    The label a claimant WRITES, not the label it may COUNT. `_wrapped_labels` was tried
    here and is wrong: it returns [] for a guess in `ambiguous`, so a router peer named
    `python` beside another entry guessing `python` stopped being contested and the router
    reported a cleared KEEP over rows three processes wrote (re-review finding 2, a
    regression against the commit it was remediating). An ambiguous entry still WRITES under
    that label — it merely may not bank it — and writing is the whole question here.

    One side must be a ROUTER. Two plain wrapped entries resolving to one non-launcher
    label are one logical server installed twice, which #285 ruled an honest label and
    `test_ambiguity_needs_a_LAUNCHER_not_merely_a_shared_label` pins; widening to any shared
    label would delete that measurement. A router's peer NAME is different in kind — the
    router chose it, so a second entry answering to it duplicates that peer.

    Gated on `_writes_ledger_rows`: an entry baked `--no-stats`, or pointed at another
    ledger, writes none of the rows under that label and cannot be one of the writers
    fighting over it — the manufactured-collision direction #397 removed from
    `_ambiguous_labels`.

    A plain `folded` peer is not a writer. It is launched BY the router, so the router's
    process writes its rows; admitting it would have every fleet contest its own peers.
    `_WRITES_LEDGER_ROWS` states that, and `scan_scopes` independently leaves such a row's
    `ledger_identity` and `wraps` None, so it claims nothing either way (mutation F2b
    survives on widening the tuple and is disclosed as equivalent, not a kill — the same
    note `measured_zero`'s `all()` carries). The behaviour is pinned by
    `test_a_plain_folded_peer_never_contests_its_own_router`."""
    from .install_mcp import NO_PEERS  # local: same cycle break as `primer_liability`

    winners = _precedence_winner(scan_rows)
    # label -> the server NAMES writing it. Names, not row indices: `_precedence_winner`
    # already admits one row per name, so a count of names is a count of processes.
    writers: dict[str, set[str]] = {}
    # Routers are tracked separately from the writer count: they contest a label but never
    # write a primer record under it, which is what makes the single-standalone case
    # attributable.
    non_router_writers: dict[str, set[str]] = {}
    # A contest needs a ROUTER on one side, and that is a deliberate limit rather than an
    # oversight. Two plain wrapped entries that resolve to one non-launcher label are ONE
    # logical server installed twice — #285 decided that is an honest label, and
    # `test_ambiguity_needs_a_LAUNCHER_not_merely_a_shared_label` pins it. A router's peer
    # name is different in kind: the router chose it, so a second entry answering to it is a
    # duplicate of that peer, which is the population #396 is about.
    router_labels: set[str] = set()
    for i, row in enumerate(scan_rows):
        name = row.get("server")
        if not name or winners.get(str(name)) != i:
            continue
        state, wraps = row.get("state"), row.get("wraps") or ""
        if not _writes_ledger_rows(row, ledger_path):
            continue
        if state in ("router", "router-ambiguous"):
            # `(no peers)` is the scan's empty-peers-file SENTINEL, never a peer name.
            written = ([] if wraps == NO_PEERS
                       else [p for p in (q.strip() for q in wraps.split(",")) if p])
        elif state in _WRITES_LEDGER_ROWS:
            # `resolve_ledger_identity`'s answer, which is what the proxy puts in the record:
            # the baked `--server-name`, else the downstream command's basename. A raw
            # `claude mcp add` re-add has neither and writes nothing to contest.
            ident = row.get("ledger_identity") or (server_label(wraps.split()) if wraps
                                                   else "")
            written = [str(ident)] if ident else []
        else:
            continue
        if state in ("router", "router-ambiguous"):
            router_labels.update(written)
        for lbl in written:
            writers.setdefault(lbl, set()).add(str(name))
            if state not in ("router", "router-ambiguous"):
                non_router_writers.setdefault(lbl, set()).add(str(name))
    return {lbl: _Contest(frozenset(writers[lbl]),
                          frozenset(non_router_writers.get(lbl, ())))
            for lbl in router_labels if len(writers.get(lbl, ())) > 1}


def _superseded_labels(row: dict[str, Any], labels: list[str],
                       by_label: dict[str, int], live: set[str]) -> list[str]:
    """Ledger labels that hold rows this entry almost certainly WROTE, but which its current
    identity no longer claims — the history stranded when `--server-name` was baked into an
    entry that had been guessing (#285 review).

    The ledger is append-only and records the identity in force at write time, so baking the
    flag renames the server from that moment on and silently splits its history in two: the
    live fleet has `secret-broker` 214 rows next to `sb-run` 200, and `runecho` 245 next to
    `runecho-mcp` 14. Merging them is NOT the fix — that would be the guessing this issue
    removed, and two labels can equally be two servers — so the rows are reported, never
    counted. Every published rate stays measured against one identity.

    Two subtractions keep "almost certainly WROTE" honest, and review found the second one
    missing. First, never a launcher basename: `searxng-mcp` guesses `python`, and the
    `python` rows in the ledger are an unrelated demo server, not its own history — a
    launcher basename is exactly where "these rows are probably yours" cannot be said.
    Second, never a label some OTHER installed entry still answers to (`live`): a
    `--server-name kb` entry over `sb-run` sits next to an unbaked `legacy-kb` reading those
    same `sb-run` rows as its live rate, and an entry over `kb` sits next to a router
    fronting `kb` as a peer. Claiming either as stranded history would print one server's
    live traffic as another's past, in the same report."""
    guess = server_label((row.get("wraps") or "").split()) if row.get("wraps") else ""
    # `guess in labels` is what makes an UNBAKED entry report nothing without a separate
    # `ledger_identity_explicit` check: with no flag its rows ARE under the guess, so
    # `_wrapped_labels` already returned it and nothing was stranded. Only a baked name can
    # move the live identity off the guess and leave the old rows behind.
    if not guess or guess in labels or _is_launcher_basename(guess) or guess in live:
        return []
    return [guess] if by_label.get(guess) else []


def _wrapped_labels(row: dict[str, Any], wraps: str, ambiguous: set[str],
                    ledger_path: str | None = None) -> list[str]:
    """The ledger label(s) a WRAPPED (standalone) entry's records are written under.

    `--server-name` (#83, baked by `install-mcp` since #152) overrides what the proxy
    writes to `server`, so the scan's `ledger_identity` — `resolve_ledger_identity`, the
    ONE resolution rule, shared with `proxy.py`'s write path — is the answer whenever the
    flag was explicit. Re-deriving the label from the downstream COMMAND here instead (what
    this did before #285) misreads every entry that passes the flag, and both directions
    are reachable: `searxng-mcp` (`.venv/bin/python -m searxng_mcp` -> `python`) billed its
    break-even against unrelated `python` rows, manufacturing a cleared verdict out of
    another server's savings, while `secret-broker` (`... python3 ...` -> `python3`)
    matched nothing and read as `never called` — the fleet's second-best compressor
    reported as pure cost, exactly the claim #175 exists to prevent.

    A scan row without an explicit flag still falls back to the command basename — that IS
    what the proxy guessed too, so the two paths still agree. The one exception is a guess
    in `ambiguous` (see `_ambiguous_labels`), where two installed entries provably resolve
    to the same launcher basename: `python` is then a real label with real rows belonging
    to more than one server, so no missing-label guard fires and no honest reading exists.
    Saying "cannot say" there is the same unknown-is-not-zero discipline `_break_even`'s
    vocabulary already keeps — and it gets its OWN reason string, because `no ledger label`
    is documented as "matched no ledger rows", which is not what happened here."""
    # The other half of #397's fix, and the half PR #395's reverted attempt lacked. An
    # entry that writes no rows into this ledger owns no label in it — not even an explicit
    # `--server-name`, which says what it WOULD write, not that it wrote anything. Without
    # this, excluding it from the ambiguity contest just lets it claim another server's
    # rows outright.
    if not _writes_ledger_rows(row, ledger_path):
        return []
    ident = row.get("ledger_identity")
    if ident and row.get("ledger_identity_explicit"):
        return [ident]
    guess = ident or (server_label(wraps.split()) if wraps else "")
    return [] if not guess or guess in ambiguous else [guess]


# Per-server cadence labels. The whole point of splitting them is that `tok/turn` and
# `tok/session` are different units and summing them was the defect (#211 follow-up).
_PER_TURN = PRIMER_CADENCE_PER_TURN
_ONCE = PRIMER_CADENCE_ONCE
_ONCE_FREE = "once/session (unpaid)"
_ONCE_UNKNOWN = "once/session (?)"

# The same four, short enough for a table cell that has to fit inside 80 columns alongside
# five other. The prose labels stay the `--json` contract; these are render-only.
_CADENCE_ABBR = {_PER_TURN: "/turn", _ONCE: "1x", _ONCE_FREE: "1x-", _ONCE_UNKNOWN: "1x?"}

# --- wrap/don't-wrap verdict (#238) ---------------------------------------------------
#
# A rollup, not a new measurement. Every input below is a field `_break_even` and `_cadence`
# already publish, and the 1.0 threshold is the one `build_primer_section` already draws
# (`turns_covered < 1` / `session_covered < 1` -> NET NEGATIVE). Nothing here is a number the
# code did not previously compute -- that is the #238 contract.
KEEP, TUNE, UNWRAP, INSUFFICIENT = "KEEP", "TUNE", "UNWRAP", "INSUFFICIENT"

# `_break_even` reasons that describe MISSING DATA rather than a measured outcome. `never
# called` is NOT here: it is a real measurement whose meaning depends on cadence (see
# `_recommend`), which is the whole idle-router/free-standalone split (#211).
#
# `writes no ledger rows` is listed for MEANING, not for behaviour: measured 2026-09-16, a
# silent entry reaches the same INSUFFICIENT through branch 6 of `_recommend` (no labels ->
# no token data), so removing it from this tuple changes no output and no test can tell the
# difference. It belongs here anyway — it is missing data, not a measured outcome — and any
# future branch that reads this tuple inherits the right answer instead of the accidental
# one.
_NO_DATA_REASONS = ("no ledger label", "ambiguous ledger label", "no token data",
                    "primer unknown", "writes no ledger rows",
                    "router/live duplicate label")

# A guessed label two installed entries share (#285). Distinct from `no ledger label`, whose
# documented meaning is "matched no ledger rows": here rows exist, they just belong to more
# than one server. Collapsing the two would tell an operator to go looking for traffic that
# is sitting right there under a name they can fix with `--server-name`.
_R_AMBIGUOUS = "ambiguous ledger label"

# An entry configured not to write into the ledger `terse stats` reads (#397). Distinct
# again from both neighbours: `no ledger label` means the rows were looked for and not
# found, `ambiguous` means they exist and belong to more than one server, and this one
# means the entry never wrote any to look for. The operator's fix is different in each
# case — bake `--server-name`, or turn stats back on, or nothing at all.
_R_NO_LEDGER_ROWS = "writes no ledger rows"

# A label claimed by BOTH a multiproxy router and the separately-live duplicate of one of
# its peers (#396 review). The fourth distinct cause of "no attributable label", and the
# only one whose fix is not `--server-name` on a launcher guess: the router tags each peer's
# records with that PEER's name, and a `folded-and-live` entry running its own proxy under
# the same name writes rows that are byte-identical in provenance. An explicit
# `--server-name kb` does not help — it is what makes the collision EXACT — so
# `_ambiguous_labels`, which only contests guessed launcher basenames, never sees it. Left
# unfixed, both rows consumed the same blocks and the same savings and each independently
# reported KEEP: #285's double count, one state over.
_R_ROUTER_DUP = "router/live duplicate label"


def _writes_ledger_rows(row: dict[str, Any], ledger_path: str | None = None) -> bool:
    """Does this entry write records into the ledger THIS REPORT IS READING?

    False in exactly two configured cases: `--no-stats` (the proxy writes nothing at all)
    and `--stats-log` pointing at a DIFFERENT file from the one being read (it writes real
    records this report never sees). Both make the entry unable to own a ledger label, and
    unable to be one of two servers colliding over one.

    `ledger_path` is the file the report was built from (`terse stats --log FILE`, else
    `default_stats_log()`). Comparing against it is not a refinement — without it the
    predicate deletes exactly the measurement it exists to protect (review of PR #417):
    `--stats-log /srv/kb.jsonl` read back with `terse stats --log /srv/kb.jsonl` is the
    entry that owns every row in the file, and a hand-edited entry that simply spells the
    default path out is the same case. `parse_proxy_opts`' docstring calls hand-edited
    entries "the one population this has to read correctly".

    Resolved before comparing, so `~/x.jsonl` and `/home/u/x.jsonl` are one file — a tilde
    stays LITERAL inside a JSON MCP config while `default_stats_log()` is already expanded,
    which is the spelling that matters.

    A RELATIVE baked path is resolved against the cwd of whoever runs `terse stats`, NOT the
    config's directory, so it can compare equal to a different file (or unequal to the same
    one). That is a real hole, not an excluded case: `install-mcp` has no `--stats-log` flag
    and never writes one, so EVERY entry carrying it is hand-edited — the population the
    caller's docstring calls the one that has to be read correctly. Left as a limitation
    because there is no defensible base to resolve against (the proxy's cwd at launch is the
    client's, not the config's), and the cost is a mislabelled row in a report. An earlier
    draft of this paragraph claimed `install-mcp` writes the flag; it does not.

    BOTH sides are resolved, which the first cut got half-right: an entry with NO
    `--stats-log` writes to `default_stats_log()`, and that is not the file being read
    whenever `terse stats --log FILE` names another one. Unconditionally answering True
    there let an entry claim blocks out of a file it never writes to, and made it contest a
    label against the entry that owns every row in it (review of PR #417).

    A row carrying NEITHER `stats` nor `stats_log` short-circuits to True and skips the path
    comparison entirely: a scan predating #397 cannot say where the entry writes, and
    neither half of the rule may be applied to it. Reading absence as silence would retract
    the measurement of every entry recorded before #397. Unknown is not zero — the same discipline `codec_call_rate` keeps for a missing
    counter. `stats` PRESENT AND NOT TRUE is False, and the distinction is #416's: it sets
    the field to None deliberately, for an entry whose args carry no `proxy` subcommand, so
    that this function would not read "runs no proxy at all" as "writes normally". The first
    cut used `is False` and read that None as True — the exact assertion #416's comment says
    it exists to prevent."""
    if "stats" in row and row["stats"] is not True:
        return False
    if "stats" not in row and "stats_log" not in row:
        # A row from a scan predating #397 carries NEITHER field. It cannot say where — or
        # whether — this entry writes, so it keeps the measurement it had. Applying the
        # default-ledger rule below would retract every such row the moment the report reads
        # a non-default ledger, on evidence the row does not carry. Either field present
        # means the scan looked.
        return True
    read = ledger_path or str(default_stats_log())
    baked = row.get("stats_log")
    return _same_file(str(baked) if baked else str(default_stats_log()), read)


def _same_file(a: str, b: str) -> bool:
    """Do two ledger paths name one file? Expanded and normalised, never `os.path.samefile`
    — that needs both to EXIST, and a ledger the proxy has not written yet is the common
    case for a freshly installed entry."""
    try:
        return (os.path.realpath(os.path.expanduser(a))
                == os.path.realpath(os.path.expanduser(b)))
    except Exception:  # noqa: BLE001 - a malformed path is "not the same file", not a crash
        return a == b

# Verdict reason vocabulary beyond the `_break_even` strings it passes through. Closed set --
# `--json` consumers switch on these.
_R_CLEARED = "cleared"                 # coverage >= 1 against its own primer
_R_SHORT = "short of break-even"       # positive rate, coverage < 1
_R_EXPANDING = "expanding"             # negative rate with no primer to offset it

# Render order for the recommend table: what needs ACTION first. Deliberately different from
# the break-even table, which sorts by rate — that one answers "which server is the best
# codec fit", this one answers "what should I change today", and a KEEP row is the one thing
# on the screen that needs nothing.
_VERDICT_ORDER = {UNWRAP: 0, TUNE: 1, INSUFFICIENT: 2, KEEP: 3}


def _cadences_of(servers: list[dict[str, Any]]) -> set[str]:
    """Which cadences an install actually has — the gate for every per-cadence line.

    ONE function rather than a set comprehension at each call site, because the two sites
    drifted: the prose gated on `s.get("cadence") or _PER_TURN` and the table's legend on a
    bare `s.get("cadence")`. On a liability blob round-tripped through `--json` by a
    pre-cadence terse — the exact backward-compat path both were written for — that set was
    `{None}`, so the table suppressed the `/turn` legend and printed the standalone one
    instead, directly under prose that had just declared the whole figure recurring.

    `or _PER_TURN` is the honest default for a blob with no cadence at all: an older terse
    measured the always-eager model, so per-turn is what its numbers mean."""
    return {s.get("cadence") or _PER_TURN for s in servers}


def _cadence(state: str | None, blocks: int | None, encoded: int | None,
             recorded: bool = False, unpaid: bool = False) -> str:
    """How often this entry actually pays its primer, post-#211.

    `blocks` is the ledger's answer to "was it called", and its three-way None/0/N is load
    bearing here exactly as it is in `_break_even`: None means no label was recoverable, so
    we never found the rows to ask — NOT that the server was idle. Collapsing it to "never
    called" would move an unknown into the `free` bucket and quietly under-report.

    `encoded` is the sharper question, and "was it called" was the wrong one to bill on
    (review finding). The lazy primer attaches to a result carrying a terse wire form, so a
    standalone entry that was called a thousand times and never produced one — an
    all-passthrough policy, non-JSON payloads, a shape the codec never wins on — paid
    NOTHING, and billing it a full primer is the same mis-bucketing this split exists to
    fix, just in the other direction. `blocks` counts every emitted block regardless of
    decision, so it cannot see that; `encoded` counts every block EXCEPT the `passthrough`
    and `unchanged` ones (so an unreadable `decision` counts — see `aggregate`, where the
    predicate lives and why it is stated as an exclusion).

    The inference is one-directional but NOT a proof, and an earlier revision of this
    docstring overclaimed it (found in review). `encoded == 0` is strong evidence the primer
    never attached, not a demonstration: the attach guard at `proxy.py` fires on
    `'"__terse_'` appearing anywhere in the FINAL content, and that text can come from the
    downstream payload rather than from the codec — a `passthrough` result that happens to
    quote a terse wire form (a code-search tool returning terse's own source, a doubly
    wrapped peer) attaches the primer while classifying as `passthrough`, so `encoded` stays
    0 and this returns `_ONCE_FREE` for a server that did pay. That path is still open and
    is the same under-billing class as the one this argument closes.

    `encoded > 0` likewise does not prove the primer attached — a minify-only `compressed`
    block carries no marker, and the `structuredContent` gap at the same guard can suppress
    the attach on results that do. So a non-zero count bills, which stays the over-billing
    direction the module argues is the safe one."""
    if state in _PRIMES_EAGERLY:
        return _PER_TURN
    # `recorded` ENDS the inference above, and is checked before every other branch (#311
    # review). Everything this docstring argues is about evidence: `encoded` is "strong
    # evidence, not a demonstration", `blocks is None` is "we never found the rows to ask".
    # A recorded emission is neither -- the attach site wrote down that it happened. The
    # very path this docstring names as still open (a `passthrough` result whose downstream
    # payload quotes a terse marker attaches the primer while `encoded` stays 0) is exactly
    # where the two disagree, and the report used to call such a server MEASURED and list it
    # as "costing nothing at all" in the same breath.
    if unpaid:
        # Proof, from a recorded suppression, that the primer never went out. Checked before
        # `recorded` and before every inference: this is the one branch backed by a row the
        # proxy wrote at the moment it made the decision.
        return _ONCE_FREE
    if recorded:
        return _ONCE
    if blocks is None:
        return _ONCE_UNKNOWN
    # `encoded is None` is the hand-rolled/pre-counter agg: fall back to `blocks`, which is
    # the OLD behaviour — coarser, and over-billing rather than under-billing.
    return _ONCE if (blocks if encoded is None else encoded) else _ONCE_FREE


def _break_even(primer_tokens: int | None, blocks: int | None,
                tokenized: int | None, saved: int,
                *, no_label_reason: str | None = None) -> dict[str, Any]:
    """Per-server `saved/block` and `blocks to break even` (#175).

    The rule the positioning issue states — *wrap a server when its typical payload saves
    more than `primer x (turns per call)` tokens* — is only actionable if the operator can
    read both halves per server. #175 computed this table by hand from the ledger; this puts
    it in `terse stats`.

    The arithmetic is `primer / saved_per_block` either way, but the UNIT the answer is in
    depends on how often that primer is charged, so the caller pairs it with `_cadence`:
    blocks per TURN for an eagerly-primed router, blocks once per SESSION for a lazily-primed
    standalone entry (#211). Same number, two very different bars — which is why the rendered
    column header no longer hard-codes `/turn`.

    Stated per BLOCK, not per call, because a block is what the ledger counts: one record
    per emitted tool-result text block, which is >= 1 per call and moves with join behaviour
    by design (#141, and the `blocks` naming decision in `aggregate`). Calling it `/call`
    would silently 3x the reported break-even for a server that emits three blocks per call,
    in the pessimistic direction — found in review of #197, where this table shipped one
    round with a `calls` header over a block counter.

    A rate is a number OR a `verdict` naming why there isn't one — never a 0 standing in for
    a missing measurement, because each of these accuses the install of something different
    and `None` alone cannot tell them apart (either in the table or in `--json`):

      no ledger label   the config entry's downstream command matched no ledger rows, so we
                        cannot even say whether it was called. NOT "never called".
      ambiguous ledger label
                        the entry bakes no `--server-name` and its downstream command's
                        basename is a launcher (`python`, `npx`, ...) that ANOTHER installed
                        entry also resolves to, so its rows cannot be told from that
                        server's. Rows exist; attribution does not (#285).
      writes no ledger rows
                        the entry is configured not to write into the ledger this report
                        reads — `--no-stats`, or `--stats-log` naming a different file. It
                        owns no label here and cannot collide over one; there is nothing to
                        look for, as opposed to looking and not finding (#397).
      router/live duplicate label
                        MORE THAN ONE installed entry writes under this label — a router's
                        peer name that a second entry also answers to. Rows exist and this
                        entry wrote some of them, but nothing in a record says which, so no
                        claimant may count it (#396). Distinct from `ambiguous ledger label`,
                        whose fix is baking `--server-name`: here an explicit `--server-name`
                        is typically what MADE the collision exact.
      never called      installed, pays a primer, banked nothing this window.
      no token data     called, but every matching row was recorded without tiktoken. The
                        savings in TOKENS are unknown, not zero — and dividing a cl100k
                        primer by a char-derived rate would silently mix units.
      primer unknown    the rate is real but the server's policy could not be read, so the
                        threshold it must clear is unknown. Reporting this as `never` would
                        condemn a server on evidence about a missing FILE (found in review
                        of #197).
      no primer         nothing to earn back, at any rate, positive or negative. TWO
                        distinguishable causes share this string, deliberately: a
                        default-deny policy that emits no compressed form and therefore no
                        primer, and (#286) a server whose primer is DECLINED on every result
                        because `structuredContent` would make the client discard it. A
                        `--json` consumer separates them without a new verdict value —
                        `primer_source == "recorded"` with `cadence == "once/session
                        (unpaid)"` is the second; the first has neither. The value is not
                        split because the closed set is a published contract and both cases
                        give the operator the same answer: this wrap costs no primer.
      never             a known non-positive rate: no block volume earns this primer back.
                        The one verdict here that should stop an operator, which is why it
                        is a word and not a very large number.

    `tokenized` — not `blocks` — is the denominator. `aggregate` counts every record in
    `blocks` but only tokenized ones in the token sums, so a ledger that spans an offline
    session (`count_cl100k` returns None) and later online ones carries a full block count
    against a partial savings sum. Dividing by `blocks` there under-reports the rate by
    exactly the untokenized fraction, and always in the direction that argues for unwrapping
    a server that is in fact paying for itself (found in review of #197).
    """
    if blocks is None:
        # `no_label_reason` refines WHY there is no label when the caller knows:
        # `ambiguous ledger label` (#285), `writes no ledger rows` (#397), or
        # `router/live duplicate label` (#396). Defaulted, so
        # every existing caller keeps the original string and this stays additive to the
        # `--json` vocabulary.
        return {"saved_per_block": None, "blocks_to_break_even": None,
                "break_even_verdict": no_label_reason or "no ledger label"}
    if blocks == 0:
        return {"saved_per_block": None, "blocks_to_break_even": None,
                "break_even_verdict": "never called"}
    if not tokenized:
        return {"saved_per_block": None, "blocks_to_break_even": None,
                "break_even_verdict": "no token data"}
    per_call = saved / tokenized
    if primer_tokens is None:
        return {"saved_per_block": per_call, "blocks_to_break_even": None,
                "break_even_verdict": "primer unknown"}
    if primer_tokens == 0:
        return {"saved_per_block": per_call, "blocks_to_break_even": 0.0,
                "break_even_verdict": "no primer"}
    if per_call <= 0:
        return {"saved_per_block": per_call, "blocks_to_break_even": None,
                "break_even_verdict": "never"}
    return {"saved_per_block": per_call, "blocks_to_break_even": primer_tokens / per_call,
            "break_even_verdict": None}


def _recommend(srv: dict[str, Any]) -> dict[str, Any]:
    """One wrap/don't-wrap word per INSTALLED ENTRY, rolled up from fields already published
    on `srv` -- no new arithmetic (#238).

    Takes the whole row rather than scalars (unlike `_break_even`, just above) deliberately:
    the verdict must be a pure function of what the break-even table PRINTS, so an operator
    can re-derive it from the two columns in front of them. It is also why this reads
    `break_even_verdict` rather than re-testing the conditions that produced it -- a second
    copy of that precedence chain is a second thing to drift.

    THE ENTRY, NOT THE PEER. `primer_liability` has already pooled a router's peers into one
    row against ONE shared union primer. Computing this per peer would charge each peer the
    full shared primer and tell the operator to unwrap peers that are collectively paying for
    themselves -- the inversion `primer_liability`'s own docstring warns about, one level up.
    `_PAYS_PRIMER` keeps a plain `folded` peer from ever reaching here. `folded-and-live` is
    the one exception since #396 — its live half is a SECOND process paying its own lazy
    primer, so the row that arrives is an entry, not a peer, and the rule above still holds.

    `break_even_coverage` is `tokenized_blocks / blocks_to_break_even`, which reduces
    algebraically to `entry_saved / primer_tokens` -- the per-entry twin of `turns_covered`
    and `session_covered`. TOKENIZED blocks, never `blocks`: `saved = rate * tokenized` is an
    exact identity, and dividing by `blocks` would credit savings to blocks that were never
    measured, manufacturing coverage out of an offline session.

    The 1.0 threshold is not chosen here; it is inherited. `build_primer_section` already
    prints NET NEGATIVE below 1.0 in both units and a neutral-positive sentence above it, so
    thresholding anywhere else would make the summary word disagree with the paragraph it
    summarizes. What 1.0 MEANS differs by cadence and both readings are optimistic bounds --
    one turn for a router, one session for a lazy entry -- which is stated in the rendered
    legend rather than corrected by a margin, because a margin would be a fabricated turn
    count (the #144/#186/#188 family). The only thing that could turn either bound into a
    measurement is a ledger-side session/turn marker, which is a ledger SHAPE change and a
    separate decision -- deliberately not in #238.
    """
    v = srv.get("break_even_verdict")
    rate = srv.get("saved_per_block")
    tok = srv.get("tokenized_blocks")
    need = srv.get("blocks_to_break_even")
    cadence = srv.get("cadence")
    primer = srv.get("primer_tokens")

    def out(verdict, reason, coverage=None):
        return {"verdict": verdict, "verdict_reason": reason,
                "break_even_coverage": coverage}

    # 1. Missing data is never an accusation. A verdict on any of these would be a verdict
    #    about a missing FILE or a missing tokenizer -- the #197 review finding, one level up.
    if v in _NO_DATA_REASONS:
        return out(INSUFFICIENT, v)

    # 2. `never called` is a MEASUREMENT, and what it measures depends on who pays for idling.
    #    An eagerly-primed router ships its union primer every turn whether or not anyone
    #    calls it, so zero calls is a complete measurement of a total loss -- exactly what
    #    `idle` already reports as "pure cost". A lazily-primed standalone entry paid NOTHING
    #    (#211, `free`), and "unwrap the servers costing you least" is the inversion the #211
    #    follow-up exists to prevent. The truthy-primer guard mirrors `idle`'s own: a
    #    default-deny router costs nothing to leave idle either.
    if v == "never called":
        if cadence == _PER_TURN and primer:
            return out(UNWRAP, v)
        return out(INSUFFICIENT, v)

    # 3. A non-positive rate against a real primer: no block volume earns it back, at any
    #    volume. This is the ONLY structural impossibility here, and it is what separates
    #    UNWRAP from TUNE.
    #
    #    Coverage is still well-defined here, and computed directly rather than via
    #    `blocks_to_break_even` (which `_break_even` deliberately leaves `None` for this
    #    branch -- "blocks needed to reach break-even" has no answer when no volume ever
    #    gets there). But `break_even_coverage` is a DIFFERENT quantity -- "how much of the
    #    primer this window's savings recovered" -- and that reduces to `entry_saved /
    #    primer_tokens` same as every other branch, negative and all. Found in review:
    #    an earlier cut returned `None` for a genuinely negative rate, which is
    #    indistinguishable in `--json` from the `no primer` branch's TRULY undefined `None`
    #    (division by zero) -- the exact "a None can't tell two accusations apart" failure
    #    this module's own docstring warns against, just re-introduced one function up.
    if v == "never":
        coverage = (rate * tok) / primer if rate is not None and tok is not None and primer else None
        return out(UNWRAP, v, coverage)

    # 4/5. `no primer`: nothing to earn back "at any rate, positive or negative"
    #      (`_break_even`). Coverage is undefined -- `blocks_to_break_even` is 0.0 and the
    #      division has no meaning -- so it is None, not a fabricated infinity.
    #
    #      One deliberate divergence from `_break_even`'s ordering, and it is not an
    #      arithmetic change: `_break_even` short-circuits on `primer == 0` BEFORE testing the
    #      rate, so a server with no primer that is actively EXPANDING payloads reports
    #      `no primer`. The primer is free; the expansion is not. Reachable when a policy was
    #      edited mid-window. Verdicting that KEEP would be the one flatly wrong word this
    #      table can print, so the verdict layer -- which reads both fields -- says UNWRAP and
    #      the reason string says why the two cells disagree.
    if v == "no primer":
        if rate is not None and rate < 0:
            return out(UNWRAP, _R_EXPANDING)
        return out(KEEP, v)

    # 6. A real finite break-even. `v is None` here by `_break_even`'s construction, but the
    #    guards are explicit rather than assumed: this function is also handed rows from a
    #    `--json` blob written by an older terse (see `_fmt_verdict`), and a missing field
    #    must degrade, never raise.
    if not need or tok is None:
        return out(INSUFFICIENT, v or "no token data")
    coverage = tok / need
    return (out(KEEP, _R_CLEARED, coverage) if tok >= need
            else out(TUNE, _R_SHORT, coverage))


def _contributors(labels: list[str], blocks_by: dict[str, int], saved_by: dict[str, int],
                  tokenized_by: dict[str, int]) -> list[dict[str, Any]]:
    """Per-ledger-label evidence UNDER an installed-entry verdict -- a ranking of who
    contributes, never a verdict of its own (#238).

    Deliberately verdict-INCAPABLE: no `primer_tokens`, no `blocks_to_break_even`, no
    `break_even_verdict`, no `verdict`. The break-even question is `saved / primer`, and a
    peer has no primer -- a router pays ONE union primer for the whole fleet. Anyone wanting
    a per-peer KEEP/UNWRAP would have to ADD a field with no honest per-peer value, which is
    a visible act rather than an accident. The absence is asserted by name in the contract
    test, not just left implicit.
    """
    rows: list[dict[str, Any]] = []
    for lbl in labels:
        tok = tokenized_by.get(lbl, 0)
        saved = saved_by.get(lbl, 0)
        rows.append({"label": lbl, "blocks": blocks_by.get(lbl, 0),
                     "tokenized_blocks": tok, "saved_tokens": saved,
                     "saved_per_block": (saved / tok) if tok else None})
    return sorted(rows, key=lambda r: r["saved_tokens"], reverse=True)


def primer_liability(scan_rows: list[dict[str, Any]], agg: dict[str, Any],
                     ledger_path: str | None = None) -> dict[str, Any]:
    """Primer cost of the INSTALLED wrapped servers, against the window's savings.

    Installed, not ledger-derived, and that distinction is the whole point: an eagerly-primed
    router nobody called still ships its union primer every turn and contributes zero ledger
    rows. Sizing this from the ledger would hide exactly the worst case — the install that is
    pure cost. (The ledger is still what decides whether a LAZILY-primed entry was billed at
    all; see below. Installed sizes it, the ledger says who owes it.)

    TWO CADENCES, never summed (#211 follow-up). Before the lazy primer every entry here
    paid at `initialize` and `per_turn_tokens` was one honest total. It no longer is:

      router / router-ambiguous   still prime EAGERLY, one `union_primer` in the router's
                                  own merged `initialize.instructions`, re-read every turn
                                  as `cache_read`. RECURRING — `per_turn_tokens`.
      wrapped / wrapped-unstashed lazy since #211: the primer attaches to the FIRST result
                                  carrying a terse wire form. Paid ONCE per session if that
                                  result comes, and NOT AT ALL if it never does.
      folded-and-live             the same lazy charge, but only when the live half really
                                  launches a second `terse proxy` (#396). A raw re-add runs
                                  nothing and pays nothing beyond the router's union primer.

    Plain `folded` / `folded-unstashed` peers are rendered by nobody: the router pays one
    union primer covering the fleet, so charging a peer as well would double-bill it.

    Adding those two into one `tok/turn` headline overstated a standalone install by the
    session's whole turn count, and — worse — the `idle` line then accused a never-called
    standalone server of being "pure cost" when #211 is precisely what made it free. That
    inverted advice (unwrap the servers that cost nothing) is why this is split rather than
    footnoted; the previous fix only added a "treat as worst-case" caveat, and a bound that
    is structurally zero is not a bound.

    "Called" is read from the ledger, so a standalone server is billed its one-time primer
    when it produced ANY block this window. That is an upper bound in one narrow direction:
    a session whose every compressible result also carried `structuredContent` never gets
    the attach (the accepted gap at `proxy.py`'s lazy-attach guard), so it was called and
    still paid nothing. The ledger cannot distinguish that, and inventing a discount for it
    would be the #144/#186/#188 defect family again — a number describing something the
    code did not measure. Over-billing by an unobservable exception is the safe direction.

    Each server is sized from ITS OWN policy via `build_primer`, not from a shared constant:
    the primer is assembled per-server from policy-gated sections, so a `minify`-only or
    default-deny server legitimately pays 0. A server whose policy cannot be read is counted
    as `unresolved` and left OUT of the total, which is therefore a LOWER bound and is
    labelled as one — better than substituting the built-in default and overstating.

    Entries are de-duplicated by server name across scopes: the same name in project and user
    scope is one server to the client, not two primers.
    """
    # Imported here, not at module scope: `stats` is on the proxy's hot path and these pull
    # in the policy/primer machinery only when a report is actually being rendered.
    from .install_mcp import NO_PEERS
    from .policy import default_policy, load_policy
    from .proxy import build_primer, union_primer

    # Primers ACTUALLY emitted this window, by ledger label (#311). Only the once/session
    # cadence: the eager sites emit unconditionally and are already exact by inference, so
    # they are not recorded and must not be looked for here.
    #
    # TOKENS AND EMISSIONS, not tokens alone. `primer_tokens` is a PER-SESSION charge --
    # `_break_even` divides by it to get "blocks once per session", the report renders it
    # "N tok/session" -- while `aggregate` sums a whole window, and a standalone proxy is
    # one process per session and writes one row per session. Assigning the window sum to a
    # per-session field over-bills by the session count and flips break-even to NET
    # NEGATIVE on any multi-session window, which is the normal case (`terse stats` with no
    # `--since` reads all history). That is the same mis-denomination #312 was closed for,
    # one layer down, and it made the "measured" figure WORSE than the estimate it replaced
    # (caught in review before merge; see
    # `test_a_multi_session_window_reports_ONE_primer_not_the_windows_sum`).
    #
    # The divisor is TOKENIZED emissions, not `emissions`: `tokens` only accumulates rows
    # where `count_cl100k` returned an int, so dividing by a count that also includes
    # untokenized rows would understate the per-session charge in proportion to how much of
    # the window a tiktoken-less terse wrote.
    recorded_tokens: dict[str, int] = {}
    recorded_emissions: dict[str, int] = {}
    # Labels with a recorded SUPPRESSION and no attach: proof the primer costs nothing
    # (#286). Tracked separately because it is the opposite fact, and because an attach
    # anywhere in the window overrides it -- a session can suppress on an early
    # `structuredContent` result and attach on a later text-only one, and having paid once
    # is what matters.
    suppressed_label: set[str] = set()
    # Labels with ANY attach row, recorded before the tokenization skip below. Kept separate
    # from `recorded_emissions` because those two answer different questions: "did it pay?"
    # and "can we size what it paid?". An attach written without tiktoken carries
    # `tokens: None`, so it can only answer the first -- and folding the two together meant a
    # session that suppressed early and attached late inverted to "provably free" whenever
    # tiktoken was unavailable, publishing a fabricated zero AS A MEASUREMENT. That is the
    # precise failure this whole design exists to prevent, re-entered through the divisor
    # (found in review of #320; see `test_an_untokenized_attach_still_beats_a_suppression`).
    attached_label: set[str] = set()
    for prow in (agg.get("primers") or []):
        if prow.get("cadence") != _ONCE:
            continue
        plbl = str(prow.get("server", ""))
        if not plbl:
            continue
        # The `True` default is defensive only and CANNOT be mutation-pinned: `aggregate`
        # always injects `attached` into every published row, so this default never fires on
        # its output -- only on a hand-built or foreign blob. The default that does matter is
        # the one in `aggregate` itself, which is pinned. Flagged rather than left to look
        # guarded (a mutation of it passes the whole suite).
        if not prow.get("attached", True):
            # Defence in depth: a suppression is a claim that NOTHING went out, so a row
            # claiming it while carrying a size is self-contradictory and cannot be trusted
            # as proof of non-payment. Ignoring it falls back to the estimate, which is the
            # safe direction; believing it publishes a fabricated zero.
            if not (prow.get("tokens") or 0) and not (prow.get("bytes") or 0):
                suppressed_label.add(plbl)
            continue
        # BEFORE the tokenization skip: an attach is proof of payment whether or not we can
        # size it.
        attached_label.add(plbl)
        tokenized_emissions = (prow.get("emissions") or 0) - (prow.get("untokenized") or 0)
        if tokenized_emissions <= 0:
            # Emissions happened but none carries a token count. We know it was PAID and not
            # what it COST, so there is nothing to measure with -- leave the label out and
            # let the policy-derived estimate stand, labelled `estimated`. Inventing a size
            # here, or dividing by an emission count the tokens do not cover, would publish
            # a fabricated measurement.
            continue
        recorded_tokens[plbl] = recorded_tokens.get(plbl, 0) + (prow.get("tokens") or 0)
        recorded_emissions[plbl] = recorded_emissions.get(plbl, 0) + tokenized_emissions
    by_label: dict[str, int] = {}
    # Savings too, not just call counts (#175): the per-server break-even below divides one
    # by the other, and rolling them up in the same pass keeps them from being taken over
    # different row sets.
    saved_by_label: dict[str, int] = {}
    tokenized_by_label: dict[str, int] = {}
    encoded_by_label: dict[str, int | None] = {}
    for trow in agg.get("tools", []):
        by_label[trow["server"]] = by_label.get(trow["server"], 0) + trow["blocks"]
        lbl = trow["server"]
        # CONTEXT basis, not wire (#420). Every figure this feeds — `saved_per_block`,
        # `blocks_to_break_even`, `break_even_coverage` — is compared against a primer that
        # is charged in CONTEXT, every turn, so a wire numerator is the wrong unit and
        # over-credits the entry by the mirror block it also paid nothing for. Ledger-wide
        # that inflation is 1.93x.
        #
        # `or` the wire pair, never 0: an agg from an older terse (or a hand-rolled one in
        # a test) carries no `ctx_*` key at all, and defaulting that to zero would publish
        # "this server saved nothing" for every such caller — the unknown-is-not-zero
        # failure, in the direction that tells an operator to unwrap a server that is
        # paying for itself.
        #
        # But wire is NOT a silently-equivalent fallback, and an earlier draft of this
        # comment claimed it was ("what the two bases agree on for a record with no typed
        # field, which is what every pre-#134 ledger is made of"). That conflates the LEDGER
        # version with the AGGREGATOR version. The realistic legacy shape is a pre-#420
        # `aggregate` over a POST-#134 ledger — no `ctx_*` keys, plenty of typed fields —
        # and there the two bases differ by the full inflation: measured 3.33x on one
        # fixture. So the fallback restores the exact over-credit #420 removes, and the row
        # has to SAY so rather than publish a number that looks measured. `saved_basis`
        # below is that marker, the same discipline `primer_source` already keeps for the
        # primer half of this row (review of PR #435).
        raw_t, out_t = trow.get("raw_tokens") or 0, trow.get("out_tokens") or 0
        ctx_raw = trow.get("ctx_raw_tokens")
        ctx_out = trow.get("ctx_out_tokens")
        ctx_raw = raw_t if ctx_raw is None else ctx_raw
        ctx_out = out_t if ctx_out is None else ctx_out
        saved_by_label[lbl] = saved_by_label.get(lbl, 0) + (ctx_raw - ctx_out)
        # `.get("tokenized")`, defaulted: a caller may hand us a hand-rolled agg (the tests
        # do) or one produced before this counter existed. Falling back to `blocks` there
        # keeps the old — merely coarser — behaviour instead of reporting `no token data`
        # for a row that plainly has token sums.
        tok = trow.get("tokenized")
        if tok is None:
            tok = trow["blocks"] if raw_t or out_t else 0
        tokenized_by_label[lbl] = tokenized_by_label.get(lbl, 0) + tok
        # Left as None for the whole label if ANY contributing row predates the counter —
        # not defaulted to 0, which would claim "this server never shipped a wire form" on
        # the strength of a row that simply could not say. `_cadence` reads None as "fall
        # back to `blocks`", the old coarser behaviour.
        enc = trow.get("encoded")
        if enc is None or encoded_by_label.get(lbl, 0) is None:
            encoded_by_label[lbl] = None
        else:
            encoded_by_label[lbl] = (encoded_by_label.get(lbl) or 0) + enc
    def _label_has_rows(lbl: str) -> bool:
        """Did this label record any TOOL rows in the window? One definition, because the
        `labels` filter and the blackout must agree about it — they disagreed once already
        and an idle duplicate deleted a complete measurement (round 4).

        TOOL rows only, and the name says so on purpose: primer records live in a different
        population and this must not be read as "this label is silent" (see `primer_labels`,
        where reading it that way published one attach as two).

        `by_label` alone decides it against any real aggregate — `aggregate` does
        `row["blocks"] += 1` per record, so a label with token sums always has a block count.
        The other two terms are defensive against a hand-rolled `agg` and cannot fire
        otherwise; all three mutations that drop them survive, and they are kept for the same
        reason `measured_zero`'s `all()` is."""
        return bool(by_label.get(lbl) or tokenized_by_label.get(lbl)
                    or saved_by_label.get(lbl))

    servers: list[dict[str, Any]] = []
    # Which row speaks for a name defined in several scopes — the one the client launches,
    # not the first emitted (#398). Rendering order below is unchanged: the loop still walks
    # `scan_rows` as emitted and skips the rows that lost.
    winners = _precedence_winner(scan_rows)
    # Computed over ALL rows before any is rendered: whether one entry's guessed label is
    # ambiguous is a fact about the fleet, not about that row (#285).
    ambiguous = _ambiguous_labels(scan_rows, ledger_path)
    # Same fleet-level discipline, a different collision: a label shared by a router and the
    # separately-live duplicate of one of its peers (#396 review).
    contests = _contested_labels(scan_rows, ledger_path)
    contested = set(contests)
    live = _live_labels(scan_rows)
    for i, row in enumerate(scan_rows):
        name, state = row.get("server"), row.get("state")
        # The winner check subsumes the old `seen` dedup: one index per name, fleet-wide.
        if not name or winners.get(str(name)) != i:
            continue
        # A folded peer normally pays nothing itself: the router covers it in one union
        # primer. `folded-and-live` is the exception only when its live duplicate launches
        # another terse proxy. `scan_scopes` proves that by filling `wraps`; a raw re-add
        # leaves it None. State alone cannot distinguish the two (#396).
        live_duplicate_proxy = state == "folded-and-live" and row.get("wraps") is not None
        if state not in _PAYS_PRIMER and not live_duplicate_proxy:
            continue
        is_router = state in ("router", "router-ambiguous")
        # `wraps` means two different things by state, and reading it the wrong way is how
        # a busy router reports as never-called: for a wrapped entry it is the downstream
        # COMMAND, for a router it is the comma-joined peer NAMES — and multiproxy tags each
        # peer's ledger records with its own name, so those names ARE the labels. A wrapped
        # entry's label is NOT `server_label(wraps)`: `--server-name` overrides it, which is
        # `_wrapped_labels`' whole subject (#285).
        wraps = row.get("wraps") or ""
        # `(no peers)` is the scan's SENTINEL for an empty peers file, not a peer name.
        # Reading it as one gave a peerless router a ledger label of "(no peers)", which
        # matched nothing and produced the right verdict (UNWRAP — it primes every turn and
        # fronts nothing) off a fabricated label that `--json` published in `ledger_labels`
        # and `contributors`. Peerless is the one state where zero is KNOWN rather than
        # unknown: there is no peer that could have banked anything, so `blocks` is a real 0
        # and the verdict is earned instead of lucky (#285 review).
        peerless = is_router and wraps == NO_PEERS
        peers = ([] if peerless
                 else [p for p in (q.strip() for q in wraps.split(",")) if p]
                 ) if is_router else []
        # A SILENT router owns no labels either (review of PR #417). The peer names are
        # the labels a router's rows carry, but a router baked `--no-stats` (or writing to
        # another ledger) wrote none of them — and leaving this branch ungated let it claim
        # every peer's savings, including rows a `folded-and-live` peer wrote through its
        # own proxy. That is the same "excluded from the contest, still owning the label"
        # shape this fix exists to remove, one state over.
        writes = _writes_ledger_rows(row, ledger_path)
        claimed = (peers if writes else []) if is_router \
            else _wrapped_labels(row, wraps, ambiguous, ledger_path)
        # Dropped from BOTH claimants, not arbitrated between them: the ledger records no
        # fact that could pick a winner, and handing the label to either one is the
        # fabricated attribution #285 removed. A router keeps its uncontested peers and
        # stays measurable on those; only the shared name goes dark (#396 review).
        # STRAIGHT OUT of the contest, never re-derived. Twice now a second derivation of
        # "which contested labels does this entry write" has drifted from the first: round 2
        # routed it through `_wrapped_labels` and lost the `ambiguous` case, round 3 inlined
        # it and lost the `_writes_ledger_rows` gate — so an entry that writes nothing was
        # named as a writer, and because a remedy is chosen from that set, the real
        # collision's remedy was suppressed. `_contested_labels` already applied every gate.
        #
        # This also settles the reason for a contested entry whose own guess is ambiguous.
        # Contested outranks ambiguous, and must: the ambiguity line prescribes baking
        # `--server-name`, and on an entry named after a router's peer that bakes the
        # collision EXACT.
        contested_here = sorted(lbl for lbl, c in contests.items() if str(name) in c.writers)
        # A contested label with NO rows this window is contested in NAME only: attributing
        # zero to each of two claimants is arithmetically impossible to get wrong, so it stays
        # in `labels` and the entry keeps its measurement. Dropping it unconditionally was the
        # round-3 stand-down applied one step too late — the sums were already gone by the
        # time `blackout` declined to null them, so an entry whose ONLY label was the idle one
        # still went dark, which is the exact outcome that fix claimed to remove (round 4).
        labels = [lbl for lbl in claimed
                  if lbl not in contested or not _label_has_rows(lbl)]
        pol_path = row.get("policy")
        tokens: int | None = None
        try:
            pol = load_policy(pol_path) if pol_path else default_policy()
            # A router ships ONE primer covering every form any peer can emit, and each
            # peer's policy is gated against its OWN name — gating the union on the router's
            # name would test rules like `kb.*` against "terse" and silently under-report.
            tokens = count_cl100k(union_primer([(pol, p) for p in peers]) if is_router
                                  else build_primer(pol, name))
        except Exception:  # noqa: BLE001 — an unreadable policy is reported, never raised
            tokens = None
        # A primer this entry is KNOWN to have sent, because the attach site wrote it down
        # (#311). Recorded beats inferred whenever it exists: `tokens` above is what the
        # installed policy WOULD assemble, which is the right size but says nothing about
        # whether the lazy attach ever fired. #286 is that gap billed as a fact.
        #
        # Absence is read as evidence of NOTHING, permanently and by design. A window with
        # no primer row cannot be told apart from one whose row simply aged out: the primer
        # decision happens once, at a session's first compressible result, while result rows
        # accrue for hours after it, so any `--since` or ledger rotation starting mid-session
        # drops it and keeps the rest. Such an entry keeps the policy estimate and is
        # labelled `estimated`.
        #
        # A ledger-version floor was tried for this (#319) and does NOT close it -- a floor
        # proves the WRITER could record, never that the WINDOW is session-complete. Do not
        # re-attempt it; the answer is the `attached: false` row, which is positive evidence.
        # The MEAN recorded primer, which is the per-session unit this field is read in --
        # never the window sum. Summed across labels first so an entry answering to several
        # labels is one mean over all its emissions, not a mean of means.
        # A contested label is unattributable for BLOCKS, which several processes write, and
        # attributable for a PRIMER record when this entry is the only standalone proxy
        # answering to it — see `_contested_labels`. Keeping the two questions apart is what
        # stops the blackout reporting a measured 777-token attach as zero.
        #
        # EVERY contested label reaches this through the sole-standalone term, and the
        # subtraction on the left is what guarantees it. `labels` deliberately keeps a
        # contested label with no TOOL rows — attributing zero blocks to two claimants
        # cannot be wrong — but that reasoning does not transfer: primer records are a
        # separate population (`agg["primers"]`) that `_label_has_rows` never reads, so an
        # idle-by-blocks label can still carry a real attach. Letting it in on the left
        # published one 777-token record as `recorded` on BOTH claimants (1,554 tok/session
        # fleet-wide), and the suppression variant — reachable from a `structuredContent`-only
        # downstream, #286's flagship shape, which writes a decline row and no tool row —
        # declared both entries provably `free` off a record only one of them wrote.
        primer_labels = [lbl for lbl in labels if lbl not in contested] + \
            [lbl for lbl in contested_here
             if not is_router and contests[lbl].standalone == {str(name)}]
        # A contested label this entry could not claim for the primer leaves the primer
        # question unanswerable for it -- distinct from answering "unpaid".
        primer_unknown = any(lbl not in primer_labels for lbl in contested_here)
        rec_tok = sum(recorded_tokens.get(lbl, 0) for lbl in primer_labels)
        rec_em = sum(recorded_emissions.get(lbl, 0) for lbl in primer_labels)
        # `rec_tok > 0` mirrors the accumulator's `tokenized_emissions <= 0` skip. The
        # proxy cannot produce emissions totalling zero tokens -- an empty primer sets
        # `_primer_sent` at construction so the attach never fires -- but a hand-edited or
        # truncated ledger can, and without this it publishes `primer_tokens: 0` under the
        # `recorded` label: a claim that we MEASURED a free primer. Unknown must degrade to
        # the labelled estimate, never to a fabricated zero.
        measured = rec_em > 0 and rec_tok > 0 and not is_router
        # A measured ZERO: the proxy recorded a SUPPRESSION for every label of this entry
        # and no attach anywhere. `not any(... attached_label)` is what makes an attach win
        # and is the load-bearing term -- `not measured` alone was NOT enough, because an
        # untokenized attach fails `measured` while still being proof of payment.
        #
        # The `all()` is defensive only -- `_wrapped_labels` returns at most
        # one label and multi-label entries are routers, excluded below -- so `all` and `any`
        # are structurally identical today and no test can distinguish them. Kept because it
        # states the intent for whoever makes an entry multi-label, not because it fires.
        #
        # Routers are excluded because they prime EAGERLY at `initialize` and no eager site
        # records anything -- a router has no primer rows by construction, and reading that
        # as proof of non-payment would zero the recurring cost of the one shape that
        # genuinely pays every turn.
        #
        # An entry with NO primer rows reaches neither branch and keeps the policy estimate.
        # That fallback is what makes a truncated `--since` window or a rotated ledger safe:
        # absence means "this window cannot say", never "it costs nothing".
        measured_zero = (not measured and not is_router and bool(primer_labels)
                         and not any(lbl in attached_label for lbl in primer_labels)
                         and all(lbl in suppressed_label for lbl in primer_labels))
        if measured_zero:
            tokens = 0
        if measured:
            tokens = round(rec_tok / rec_em)
        # None, not 0, when no label could be recovered: "unknown" and "never called" are
        # different claims, and only the second one accuses an install of being pure cost.
        blocks = (sum(by_label.get(lbl, 0) for lbl in labels) if labels
                  else (0 if peerless else None))
        # Same `if labels else None` guard as `blocks`: with no recoverable label, 0 would
        # claim "nothing was tokenized" to a `--json` consumer when the truth is that we
        # never found the rows to ask (review of #197).
        tokenized = (sum(tokenized_by_label.get(lbl, 0) for lbl in labels)
                     if labels else (0 if peerless else None))
        # None if no label, or if any contributing label could not report — same
        # unknown-is-not-zero discipline as `blocks`.
        per_label_enc = [encoded_by_label.get(lbl) for lbl in labels]
        encoded = (0 if peerless
                   else None if (not labels or any(e is None for e in per_label_enc))
                   else sum(e or 0 for e in per_label_enc))
        # A row that had a label TAKEN AWAY publishes no measurement at all. `blocks == 0`
        # is the hard claim "never called", which `_break_even` turns into that verdict,
        # `_recommend` escalates to UNWRAP for a per-turn row, and `idle` prints as "pure
        # cost" — three conclusions drawn from a sum the contest emptied. But a NON-zero
        # surviving sum is no safer, and the first fix said it was.
        #
        # That claim was "dropping labels only understates savings, so the verdict is
        # conservative". Both halves are false, each demonstrated on a real fleet in review
        # of this PR:
        #
        #   * COVERAGE is `saved/primer`. Dropping a label whose savings are NEGATIVE raises
        #     `saved`, so the verdict moves the optimistic way: a fleet reading UNWRAP/`never`
        #     — documented as the one verdict that should stop an operator, structurally
        #     unearnable — read TUNE, "arithmetically reachable", once its loss-making peer
        #     was contested away.
        #   * The RATE is `saved/tokenized`, which has no monotone direction at all. Dropping
        #     a high-volume/low-rate label inflates it without bound; a measured row rendered
        #     `600` against a true pooled `266.7`, with nothing in the row saying a label was
        #     missing.
        #
        # A sign test on the dropped label would fix only the first. So the rule is the
        # simple one the module's own discipline already implies: a label taken away leaves
        # a row that cannot be measured, whatever the surviving sum happens to be.
        # `contributors` still lists the labels that DID survive, so nothing attributable is
        # hidden — only the entry-level verdict declines to be stated.
        # ...but only where the contest can actually have moved a number. A contested label
        # with NO rows in this window contributes zero to every sum, so what survives IS the
        # pooled truth and there is nothing to misattribute — blacking it out would turn a
        # complete measurement into INSUFFICIENT because an idle duplicate is installed
        # (re-review finding 5, which also falsified the "a label taken away leaves a row
        # that cannot be measured" wording above for this sub-case). The duplication is
        # still reported: `contested_labels` is populated either way, so the entry is still
        # named in the stanza and the operator still sees it.
        blackout = any(_label_has_rows(lbl) for lbl in contested_here)
        if blackout:
            blocks = tokenized = encoded = None
        row_out: dict[str, Any] = {
            "server": name, "scope": row.get("scope"), "state": state,
            "primer_tokens": tokens, "ledger_labels": labels, "blocks": blocks,
            # Where `primer_tokens` came from, published so a --json consumer never has to
            # guess which of its numbers is a measurement (#311). "recorded" = the proxy
            # wrote down an emission this window; "estimated" = sized from the installed
            # policy and inferred to have been paid, the pre-#311 behaviour.
            "primer_source": "recorded" if (measured or measured_zero) else "estimated",
            "tokenized_blocks": tokenized,
            # `unpaid=measured_zero` forces the UNPAID bucket. `_cadence` alone returns
            # `_ONCE` ("pays once per session") whenever `encoded > 0`, and #286's shape
            # compresses plenty -- it just never attaches. Without this the flagship case
            # renders `1x` beside a zero and never reaches the `free` list.
            #
            # `blocks`/`encoded` are passed as None -- "cannot say" -- when a contested
            # label was kept for BLOCKS but refused for the PRIMER. Those two fields are
            # honest about blocks and `_cadence` reads them as evidence about the primer, so
            # an idle contested label made them argue `never called` -> `_ONCE_FREE`, and
            # the entry was declared provably FREE while an unattributable attach record sat
            # under its label. `1x?` is the answer: some process paid, and nothing in the
            # ledger says which.
            "cadence": _cadence(state, None if primer_unknown else blocks,
                                None if primer_unknown else encoded, recorded=measured,
                                unpaid=measured_zero),
            **_break_even(tokens, blocks, tokenized,
                          sum(saved_by_label.get(lbl, 0) for lbl in labels),
                          # Same shape as `no ledger label` — nothing measurable — but a
                          # different CAUSE, and the only one with an actionable fix.
                          no_label_reason=(
                              # Order matters: a silent entry has no labels for a reason
                              # that is not ambiguity, and its guess may well ALSO be in
                              # `ambiguous` because another entry shares the basename.
                              # Reporting that would send the operator to bake
                              # `--server-name` on an entry whose rows do not exist.
                              _R_NO_LEDGER_ROWS
                              if not labels and not writes
                              # Before the ambiguity branch, and applying to a ROUTER too
                              # (the one no-label cause that does): this entry HAS a label
                              # and a second process answers to it. Reporting `ambiguous`
                              # would send the operator to bake `--server-name`, which for
                              # this collision is what created it (#396 review).
                              else _R_ROUTER_DUP
                              if blackout
                              else _R_AMBIGUOUS
                              if not labels and not is_router
                              and _guessed_label(row) in ambiguous
                              else None)),
            "contributors": _contributors(labels, by_label, saved_by_label,
                                          tokenized_by_label),
            # Reported, never summed into anything above — see `_superseded_labels`.
            "superseded_labels": _superseded_labels(row, labels, by_label, live),
            # Also reported, never counted, and for the same reason: these rows exist and
            # this entry wrote SOME of them, but nothing in the ledger says which. A router
            # that keeps other labels carries no `break_even_verdict` to say so, so without
            # this field its blocks would simply be smaller than last release with no
            # published reason (#396 review).
            "contested_labels": contested_here,
        }
        # Last, and from the finished row: the verdict is a rollup of the published fields,
        # so it is computed from them rather than alongside them.
        row_out.update(_recommend(row_out))
        servers.append(row_out)

    # Each server lands in exactly one bucket, and only two of them are money.
    per_turn = sum(s["primer_tokens"] or 0 for s in servers if s["cadence"] == _PER_TURN)
    once = sum(s["primer_tokens"] or 0 for s in servers if s["cadence"] == _ONCE)
    unresolved = sum(1 for s in servers if s["primer_tokens"] is None)
    total = agg.get("total") or {}
    wire_saved = (total.get("raw_tokens") or 0) - (total.get("out_tokens") or 0)
    # Which basis `saved_tokens` is actually in, published so a consumer never has to guess
    # — an aggregate from a pre-#420 terse cannot supply the context figure and falls back
    # to wire, which is the inflated number this change exists to stop publishing silently.
    saved_basis = "context" if total.get("ctx_raw_tokens") is not None else "wire"
    # Same fallback as the per-label sums above, and for the same reason.
    ctx_raw = total.get("ctx_raw_tokens")
    ctx_out = total.get("ctx_out_tokens")
    saved = ((total.get("raw_tokens") or 0) if ctx_raw is None else ctx_raw) \
        - ((total.get("out_tokens") or 0) if ctx_out is None else ctx_out)
    return {
        "servers": servers,
        # REDEFINED by the #211 follow-up: recurring (eager-priming) entries only. It used
        # to sum every wrapped server, which is why a standalone install's headline used to
        # be large and wrong. A `--json` consumer comparing across versions will see this
        # drop; that drop IS the correction, not a regression.
        "per_turn_tokens": per_turn,
        "session_once_tokens": once,
        "unresolved": unresolved,
        # Pays every turn and banked nothing this window — pure cost. Now routers ONLY: a
        # never-called standalone entry is the `free` list below, not this one.
        "idle": [s["server"] for s in servers
                 if s["cadence"] == _PER_TURN and s["blocks"] == 0 and s["primer_tokens"]],
        # Installed and lazy, and it cost nothing at all this window. TWO ways to earn
        # that, and the list covers both: never triggered (the thing #211 bought), or
        # triggered plenty and provably never primed (#286 -- every result carried
        # `structuredContent`, so the attach was suppressed and the proxy recorded it).
        #
        # Guarded on a non-zero primer for the same reason `idle` is: a default-deny server
        # is free because it emits nothing, which is a different fact with its own `no
        # primer` verdict in the table. `or primer_source == "recorded"` re-admits the
        # second case, whose primer_tokens is a MEASURED 0 rather than an absent one -- and
        # that is a real distinction, not a loophole: only a recorded suppression sets it.
        "free": [s["server"] for s in servers
                 if s["cadence"] == _ONCE_FREE
                 and (s["primer_tokens"] or s.get("primer_source") == "recorded")],
        # Lazy, but no ledger label was recoverable, so we cannot say whether the attach
        # ever fired. Neither total counts it — same discipline as `unresolved`.
        "uncertain": [s["server"] for s in servers if s["cadence"] == _ONCE_UNKNOWN],
        # REDEFINED by #420: this is now the CONTEXT saving, what the model was charged
        # for, because it is the numerator of `turns_covered` and the primer in that
        # denominator is charged in context. A `--json` consumer comparing across versions
        # will see it roughly halve on a fleet whose results carry `structuredContent`;
        # that drop IS the correction, exactly as `per_turn_tokens`' drop was in the #211
        # follow-up. The wire figure is published beside it, never lost.
        "saved_tokens": saved,
        # What left the stdio pipe — the honest "what terse actually removed" number, and
        # the one the codec is judged on. Reported, never divided by a primer (#420).
        "wire_saved_tokens": wire_saved,
        # "context" or "wire". `wire` means this aggregate predates #420 and could not
        # supply the context figure, so every saving here — fleet and per server — is on
        # the inflated basis. Never absent, so a consumer reads it unconditionally.
        "saved_basis": saved_basis,
        # NOT `(saved - once) / per_turn`. `once` is charged per SESSION and `saved` is the
        # whole window, which spans an unknown number of sessions — a `terse proxy` is one
        # process per session and `Interceptor._primer_sent` re-arms at every `initialize`.
        # Netting a per-session charge out of a multi-session pot subtracts one primer where
        # K were paid, and `sessions` is no more observable from this ledger than `turns` is
        # (the same reason there is no per-turn charge in it at all). So the recurring figure
        # is left dividing like against like. None, not 0 or infinity, with nothing recurring
        # to pay: rendering "inf turns" would read as a measurement.
        "turns_covered": (saved / per_turn) if per_turn else None,
        # An UPPER bound, and labelled as one wherever it is rendered: this treats the whole
        # window as a single session, which is the most favourable reading available. A
        # window covering K sessions paid K one-time primers, so the true coverage is this
        # over K. Kept as a bound rather than dropped because it is still decisive in one
        # direction — if even the best case does not clear 1.0, the install is genuinely
        # net negative and no session count can rescue it.
        "session_covered": (saved / once) if once else None,
        # Per-LABEL, for the remedy line. Round 4: the stanza scanned every server's
        # `contested_labels` fleet-wide, so a fleet with two independent contests printed one
        # sentence claiming that renaming one entry addressed both labels — and the other
        # label's real remedy was never printed at all. A remedy is a fact about a label.
        "contests": {lbl: sorted(c.standalone) for lbl, c in contests.items()},
    }


def _labels_of(liab: dict[str, Any], server: str) -> list[str]:
    """The contested labels published for one server — read off the rendered rows rather
    than recomputed, so the remedy cannot disagree with the table above it."""
    return next((s.get("contested_labels") or [] for s in liab.get("servers") or []
                 if s.get("server") == server), [])


def build_primer_section(liab: dict[str, Any]) -> list[str]:
    """The break-even lines for `build_stats_report`. Empty when nothing pays a primer."""
    per_turn, servers = liab["per_turn_tokens"], liab["servers"]
    if not servers:
        return []
    # `.get` on every key added by the #211 follow-up: `build_stats_report` also renders a
    # liability blob handed back through `--json` by an older terse, which carries none of
    # them. Degrade to the recurring half rather than raising — a report is never
    # load-bearing (#197).
    once = liab.get("session_once_tokens") or 0
    # Which stanzas apply is read from the SERVERS, not from the totals: an install whose
    # lazy entries were all uncalled has `once == 0` and still needs the one-time stanza to
    # explain why, whereas an install with no lazy entry at all should not be told about a
    # cadence it does not have. A blob from an older terse has no `cadence` and gets the
    # recurring stanza only — which is exactly what it measured.
    cadences = _cadences_of(servers)
    lines = ["", f"primer liability across {len(servers)} wrapped server(s) — NOT in the "
                 f"totals above:"]
    if _PER_TURN in cadences:
        lines.append(f"  recurring  {per_turn:,} tok/turn — a multiproxy router primes "
                     f"eagerly at `initialize`, and the")
        lines.append("             client re-reads those instructions every turn as "
                     "cache_read.")
    if cadences - {_PER_TURN}:
        lines.append(f"  one-time   {once:,} tok/session — a standalone `terse proxy` "
                     f"attaches its primer to the")
        lines.append("             first compressible result and not again (#211). Only "
                     "servers that were")
        lines.append("             actually called are billed here.")
    if len(cadences) > 1 and _PER_TURN in cadences:
        lines.append("  the two figures are different units and are deliberately not "
                     "summed.")
    # TWO kinds of measurement and they are opposite facts, so one sentence cannot describe
    # both: "recorded the emission" is false for a server whose primer was DECLINED, which is
    # the whole point of #286. Split by the token count -- a measured zero is the suppression.
    paid = [s_["server"] for s_ in servers
            if s_.get("primer_source") == "recorded" and s_.get("primer_tokens")]
    never = [s_["server"] for s_ in servers
             if s_.get("primer_source") == "recorded" and not s_.get("primer_tokens")]
    if paid or never:
        lines.append(f"  {len(paid) + len(never)} of {len(servers)} server(s) are MEASURED, "
                     f"not inferred:")
        if paid:
            lines.append(f"             {len(paid)} recorded an emission.")
        if never:
            # NOT "every result carried `structuredContent`": the row proves only that the
            # result which WOULD have carried the primer did, and that none attached later.
            # A server that emits one such result and then only passthrough text produces
            # the same row. The verdict stays true; that reason would not.
            lines.append(f"             {len(never)} recorded that the primer was DECLINED "
                         f"— `structuredContent` on the")
            lines.append("             result that would have carried it, and none attached "
                         "later, so they pay")
            lines.append("             nothing at all (#286).")
        if len(paid) + len(never) < len(servers):
            # "except any listed as free below": an estimated entry that was never triggered
            # appears in BOTH this sentence and the free list below, which reads as a
            # contradiction. The estimate is about the primer's SIZE, not about whether it
            # was paid; the free list is the authority on the latter.
            lines.append("             The rest are sized from policy and assumed paid — "
                         "except any listed as")
            lines.append("             free below, where the ledger settles it.")
    if liab["unresolved"]:
        lines.append(f"  {liab['unresolved']} server(s) have an unreadable policy and are "
                     f"NOT counted — treat both figures as lower bounds.")
    # Silent entries are named whatever their CADENCE. `uncertain` is `_ONCE_UNKNOWN`, which
    # a router never is (it primes eagerly, so `_cadence` returns `_PER_TURN` before it ever
    # looks at `blocks`), so a silent ROUTER printed the verdict in a table cell that nothing
    # in the report explained — the very gap this split was widened to close, one state over
    # (review of PR #417).
    silent_any = sorted(s["server"] for s in liab["servers"]
                        if s.get("break_even_verdict") == _R_NO_LEDGER_ROWS)
    # On `contested_labels`, not on the verdict, and not via `_named` (which intersects with
    # `uncertain`, a set no router can be in). A fully-dark claimant carries the verdict,
    # but a router that keeps other peers does NOT — it reports a real KEEP over a block
    # count that quietly shrank, with nothing in the report saying a label was taken from
    # it. Both need naming, and only the field is true of both.
    dup_any = sorted(s["server"] for s in liab["servers"] if s.get("contested_labels"))
    if liab.get("uncertain") or silent_any or dup_any:
        # Split by CAUSE, because only one of the two has a fix the operator can act on.
        # `mcp-status` already tells this entry to bake `--server-name`; saying "no ledger
        # label" here and nothing else made the two commands read as unrelated complaints.
        unknown = set(liab.get("uncertain") or [])

        def _named(reason: str) -> list[str]:
            return sorted(s["server"] for s in liab["servers"]
                          if s["server"] in unknown
                          and s.get("break_even_verdict") == reason)

        amb = _named(_R_AMBIGUOUS)
        # THREE causes now, not two (review of PR #417). A silent entry reached the `rest`
        # bucket and was told "no ledger label", whose documented meaning is "rows were
        # looked for and not found" — sending the operator to hunt for rows that do not
        # exist, four lines under a table that already said `writes no ledger rows`.
        silent = silent_any
        # Named by its OWN reason rather than "everything left over": a fourth reason added
        # later would otherwise inherit the "no ledger label" caption, which is the same
        # mislabelling this split exists to remove.
        #
        # Measured equivalent TODAY — `uncertain` is exactly the `blocks is None` population
        # and those three reasons are its only causes, so no test can tell the two spellings
        # apart (mutation P5 survives, and is listed as equivalent rather than as a kill).
        # It is the cheap half of a defect that has now been fixed twice.
        # No subtraction: a server carries ONE verdict string, so these three are disjoint
        # by construction (the filter that used to sit here was dead — review of PR #417).
        rest = _named("no ledger label")
        if rest:
            lines.append(f"  no ledger label, so it is unknown whether the lazy primer ever "
                         f"attached: {', '.join(rest)}")
        if silent:
            lines.append("  writes no ledger rows — baked `--no-stats`, or `--stats-log` "
                         "pointing elsewhere, so this report")
            lines.append(f"  cannot see what they banked and they claim no ledger label: "
                         f"{', '.join(silent)}")
        if amb:
            lines.append(f"  ambiguous ledger label — these entries share a launcher "
                         f"basename, so their rows cannot be told apart: {', '.join(amb)}")
            lines.append("  bake `--server-name <name>` into each (re-run `install-mcp`) to "
                         "make them measurable.")
        # FOURTH cause, and the one whose fix is the opposite of the line above: here the
        # `--server-name` is exactly what both writers agree on (#396 review). Telling this
        # operator to bake one would be advice to do again what already broke it.
        # NOT `_named`, which intersects with `uncertain` (cadence `1x?`). A router is
        # always `_PER_TURN` — `_cadence` answers that for `_PRIMES_EAGERLY` before it ever
        # looks at `blocks` — so a contested ROUTER could never appear in that set, and
        # printed this verdict in a table cell nothing in the report explained. Exactly the
        # gap `silent_any` was added for in #417, reintroduced one reason later (review of
        # PR #422, finding 4). This cause is the first that applies to a router, so it is
        # the first that has to read the verdict directly.
        dup = dup_any
        if dup:
            lines.append(f"  router/live duplicate label — more than one installed entry "
                         f"writes under one name: {', '.join(dup)}")
            # ONE LINE PER LABEL, from the contest's own per-label writer sets. A remedy is a
            # fact about a label, not about the fleet: scanning every server's
            # `contested_labels` made a fleet with two independent contests print a single
            # sentence claiming that renaming one entry addressed both, while the other
            # label's remedy went unprinted (round 4).
            #
            # Which remedy applies is also per label. `--server-name` is REFUSED with
            # `--config` (`cli.py` exits 2), so it is not an action a router can take — and
            # where every writer of a label is a router there is no standalone to rename.
            # `router-ambiguous` is the sub-case with no "one of" to pick either: by
            # `_detect_routers`' definition those entries front ONE peers file, so renaming
            # the peer there moves it for both. Advice that does not fire for a case the line
            # itself names is what the `mcp-status` pointer was removed for, and what the
            # blanket `--server-name` line was removed for after that.
            states = {s["server"]: s.get("state") for s in liab["servers"]}
            for lbl, standalone in sorted((liab.get("contests") or {}).items()):
                if standalone:
                    who = (f"one of {', '.join(standalone)}" if len(standalone) > 1
                           else standalone[0])
                    fix = f"give {who} a DISTINCT `--server-name`"
                elif all(states.get(w) == "router-ambiguous"
                         for w in dup if lbl in _labels_of(liab, w)):
                    fix = "point one of them at its own peers file"
                else:
                    fix = "rename the peer in one of the peers files"
                lines.append(f"  `{lbl}`: remove one of them, or {fix}.")
    # Reported here rather than folded into the rate above: the split is a FACT about the
    # ledger, and merging the two identities would be the guessing #285 removed.
    for srv in liab["servers"]:
        sup = srv.get("superseded_labels") or []
        if sup:
            lines.append(f"  {srv['server']}: ledger rows under `{', '.join(sup)}` predate "
                         f"its `--server-name` and are NOT counted in its rate above.")
    saved = liab["saved_tokens"]
    turns = liab["turns_covered"]
    # Which basis every figure above is in, said once. The primer is charged in CONTEXT and
    # every break-even here divides by it, so publishing a WIRE numerator beside it
    # over-credited the fleet by 1.93x ledger-wide (#420). The wire figure is still the
    # honest "what terse removed from the pipe" number and is published in `--json`; it is
    # simply not the unit a primer break-even is denominated in.
    if turns is not None:
        # Below 1.0 the ratio is not the interesting number — the shortfall is. `~0.4 turns`
        # and `~0x over` both round to a `0` that reads as a measured zero, the same
        # print-as-zero hole `_fmt_rate` closed in review of #197.
        if turns >= 1:
            lines.append(f"  the {saved:,} tok of CONTEXT saved in this window pays for "
                         f"~{turns:,.0f} turn(s) of the recurring primer.")
        else:
            lines.append(f"  NET NEGATIVE over this window: the {saved:,} tok of CONTEXT "
                         f"saved does not cover even a single turn of the {per_turn:,} tok "
                         f"recurring primer.")
    if liab.get("session_covered") is not None:
        # Rendered ALONGSIDE the recurring line, not instead of it, and never netted into
        # it: the two divide different things. And stated as a ceiling, because the window
        # spans an unknown number of sessions and each one paid this charge again.
        #
        # Both lines credit the SAME savings in full, which is a second reason each is only
        # a ceiling and not a verdict (review finding). Not netting them is right — the
        # units differ, see `turns_covered` — but silence about it let a mixed install read
        # two individually-true lines as jointly true: at saved=600, per_turn=555, once=555
        # they say "pays for ~1 turn" and "covers the one-time charge at most ~1x" while the
        # real cost is 555*turns + 555 and the install is deeply negative. Said once, below,
        # rather than hedged into both sentences.
        covered = liab["session_covered"]
        if covered >= 1:
            lines.append(f"  the same {saved:,} tok of CONTEXT covers the {once:,} tok "
                         f"one-time charge "
                         f"at most ~{covered:,.0f}x — fewer if this")
            lines.append("  window spans more than one session, which it usually does; "
                         "each of them paid that charge again.")
        else:
            lines.append(f"  NET NEGATIVE over this window: the {saved:,} tok of CONTEXT "
                         f"saved does not cover the {once:,} tok one-time charge even "
                         f"once, and each "
                         f"session in the window paid it again.")
    if turns is not None and liab.get("session_covered") is not None:
        lines.append("  each line above credits the SAME savings against its own charge "
                     "alone — a mixed install pays both,")
        lines.append("  so clearing one of them is not clearing the pair.")
    if liab["idle"]:
        lines.append(f"  paying every turn but never called here: "
                     f"{', '.join(sorted(liab['idle']))} — pure cost until they handle a "
                     f"compressible result.")
    if liab.get("free"):
        # TWO ways to be free and the line has to allow both, or it states a falsehood about
        # the server this list was widened for: #286's shape is triggered heavily and still
        # pays nothing, because every result carried `structuredContent` and the primer was
        # declined. Saying "not triggered" printed a flat contradiction of the `blocks`
        # column three lines below it.
        lines.append("  cost nothing at all this window — never triggered, or triggered "
                     "and the primer was")
        lines.append(f"  declined and never attached (#211, #286): "
                     f"{', '.join(sorted(liab['free']))}")
    wire = liab.get("wire_saved_tokens")
    # Gated on a CONTEXT figure actually being rendered below, not merely on the two bases
    # differing. Both ratios are None for an entry that was never called, and then this
    # printed the WIRE number as the section's only figure — directly under a header
    # announcing the context basis, which is the #141 confusion re-created (review of PR
    # #435). Said BELOW the turns lines, because the figures it labels are the ones after
    # it, including `_build_break_even_table`'s `saved/block` column.
    shows_context = turns is not None or liab.get("session_covered") is not None
    if wire is not None and wire != saved and shows_context:
        lines.append(f"  the figures above and the saved/block column below are CONTEXT "
                     f"tokens. {wire:,} tok left the stdio")
        lines.append("  pipe; a mirror text block a `structuredContent`-reading client "
                     "discards is wire saving, not context (#420).")
    lines += _build_break_even_table(servers)
    return lines


def _fmt_rate(per_block: float) -> str:
    """`saved/block`, rendered so that ONLY an exactly-zero rate prints as zero.

    `:,.0f` alone collapsed all of (-1, 1) to `0`/`-0`, which sits in the same column as a
    finite break-even and reads as a contradiction; the first fix moved that hole to
    (-0.05, 0.05) rather than closing it (both found in review of #197). A tiny-but-nonzero
    rate is squashed to a bound instead, which is a true statement at any magnitude."""
    if per_block == 0:
        return "0"
    if abs(per_block) < 0.01:
        return "<0.01" if per_block > 0 else ">-0.01"
    return f"{per_block:,.2f}" if abs(per_block) < 10 else f"{per_block:,.0f}"


# Reasons too long for the break-even table's last cell, and the text used there instead.
# Render-only, exactly like `_CADENCE_ABBR`: the full string stays the `--json` contract and
# the `build_primer_section` stanza, both of which have room for it.
#
# `_BREAK_EVEN_MAX_WIDTH` is 84 because the fixed cells are 62 and the widest reason was
# `ambiguous ledger label` (22). `router/live duplicate label` is 27, which renders the row
# at 89 and folds the table — the overflow that comment predicts, and which
# `test_a_verdict_string_row_stays_inside_the_documented_bound` could not catch because its
# fixture never produced this reason (review of PR #422, finding 3).
_REASON_ABBR = {_R_ROUTER_DUP: "router/live dup label"}


def _fmt_break_even(srv: dict[str, Any]) -> tuple[str, str]:
    """The two right-hand columns of the break-even table, as text.

    A verdict from `_break_even` wins over the number: it is the reason there ISN'T one."""
    per_block, verdict = srv.get("saved_per_block"), srv.get("break_even_verdict")
    calls = srv.get("blocks_to_break_even")
    rate = "–" if per_block is None else _fmt_rate(per_block)
    # `verdict` is read with `.get`, so a liability blob round-tripped through `--json` by an
    # older terse carries no verdict at all and would take the numeric branch with a None.
    # Degrade to a dash rather than raising: this is a report, never load-bearing (#197).
    if verdict or calls is None:
        return rate, _REASON_ABBR.get(verdict or "", verdict or "–")
    return rate, f"{calls:,.2f}"


def _fmt_verdict(srv: dict[str, Any]) -> tuple[str, str, str]:
    """`(verdict, coverage, reason)` as text, for the recommend table.

    Same degrade-don't-raise property as `_fmt_break_even` directly above, and for the same
    reason: `build_primer_section` and `--json` are public, so a liability blob round-tripped
    through a terse that predates #238 carries none of these keys. A report is never
    load-bearing (#197) -- it dashes out.

    Coverage reuses `_fmt_rate`, which already closed the print-as-zero hole: a 0.004 coverage
    rendering as `0.00x` beside the word TUNE would read as a measured zero."""
    verdict = srv.get("verdict") or "–"
    cov = srv.get("break_even_coverage")
    coverage = "–" if cov is None else _fmt_rate(cov) + "x"
    return verdict, coverage, srv.get("verdict_reason") or "–"


def _fmt_denominator(srv: dict[str, Any]) -> str:
    """The call count the rate was taken over.

    Shown as `tokenized/blocks` when they differ, so a partially-untokenized ledger says so
    in the column whose number it changes rather than only in the header's `N uncounted`
    aside (found in review of #197)."""
    blocks, tok = srv["blocks"], srv.get("tokenized_blocks")
    if blocks is None:
        return "–"
    if tok is not None and tok != blocks:
        # No thousands separators in the PAIR form. This cell is a ratio to be compared, not
        # a magnitude to be read at a glance, and the separators cost four characters
        # exactly where the cell is widest — the live ledger already renders `1,790/1,799`
        # at 11, which was the whole column. Bare digits keep a million-block ledger inside
        # the width below; the single-value form keeps its separators, where it is the
        # magnitude that matters and the string is half as long.
        return f"{tok}/{blocks}"
    return f"{blocks:,}"


# The widest a rendered break-even row may get, reason strings included. 79 is the numeric
# guarantee; the slack is the longest reason the last cell can carry (`ambiguous ledger
# label`). A new reason longer than that folds the table and must be abbreviated instead.
_BREAK_EVEN_MAX_WIDTH = 84


def _build_break_even_table(servers: list[dict[str, Any]]) -> list[str]:
    """Per-server `saved/block` and the block count that pays for that server's primer.

    Gated on whether any server was CALLED in this window, not on whether any produced a
    rate. An install where nothing was called renders a table of dashes that says nothing
    the `idle` line above did not already say — but a server that WAS called and still has
    no rate is the tiktoken-missing case, and "no token data" against a real block count is
    exactly the row that sends the operator to fix it.

    A SECOND way to reach the empty case, added by #307: `blocks` is also `None` for a row
    whose ledger label is `ambiguous ledger label` (a hand-edited launcher wrap sharing a
    basename with a sibling). An all-ambiguous fleet has no row with a `blocks` count either,
    so this gate suppresses the table there too — silently, from this function's point of
    view. That silence is not a second bug: `build_primer_section`, which calls this, already
    prints the ambiguity line and the `--server-name` fix before falling through to here (see
    `test_the_rendered_section_separates_the_two_causes_of_an_unknown_label` and
    `test_the_break_even_legend_does_not_re_collapse_the_two_causes`), so the operator is
    told why in the paragraph above the missing table, not left with an unexplained gap.
    Reviewed and kept as the cheapest of three options in #310."""
    if not any(s.get("blocks") for s in servers):
        return []
    # Widths are load-bearing twice over: four tests match right-aligned cells, and the row
    # has to stay inside 80 columns or the last cells fold onto the next line and the table
    # stops being one. Adding `cadence` at its full label width took the row to 104, so the
    # cadence values are abbreviated (`_CADENCE_ABBR`) and `server` gives back space it was
    # not using (the longest real name in the fleet this targets is `secret-broker`, 13).
    #
    # `blocks` was first narrowed to 11 on the reasoning that it only ever holds `N` or
    # `tokenized/N` — but `_fmt_denominator` rendered the pair with thousands separators,
    # and the LIVE ledger already produced `1,790/1,799`, exactly 11. One more order of
    # magnitude would have overflowed and broken the 80-column guarantee this comment makes.
    # 15 with bare digits in the pair form holds a million-block ledger; the sum below is
    # 2 + 14+1 + 6+1 + 15+1 + 11+1 + 9+1 + 17 = 79, and `test_the_break_even_row_stays_
    # inside_eighty_columns` fails if any of these change without the others.
    #
    # The 80 is a guarantee about the NUMERIC cells only, and always has been: the last cell
    # also carries a reason string, and `ambiguous ledger label` (22) has pushed that row to
    # 84 since #285 — `writes no ledger rows` (21) is one of the same family, not a new one.
    # Named here because the test measured only the numeric row and so could never have
    # caught either (review of PR #417). `_BREAK_EVEN_MAX_WIDTH` below is the real bound.
    lines = ["", f"  {'server':<14} {'primer':>6} {'blocks':>15} {'saved/block':>11} "
                 f"{'cadence':>9} {'to break even':>17}"]
    # Rateless rows sort last as a group rather than tying with a 0.0 rate: `or -1` treated
    # a break-even server (0.0, falsy) as worse than one actively LOSING tokens (-0.5,
    # truthy), inverting the two rows an operator most needs ordered (review of #197).
    for srv in sorted(servers, reverse=True,
                      key=lambda s: (s.get("saved_per_block") is not None,
                                     s.get("saved_per_block") or 0.0)):
        per_block, calls = _fmt_break_even(srv)
        primer = "?" if srv["primer_tokens"] is None else f"{srv['primer_tokens']:,}"
        name = srv["server"] if len(srv["server"]) <= 14 else srv["server"][:13] + "…"
        # `.get`, defaulted to a dash: an older `--json` blob has no cadence at all, and a
        # blank there is honest where guessing `per-turn` would re-assert the old defect.
        cadence = _CADENCE_ABBR.get(srv.get("cadence") or "", "–")
        lines.append(f"  {name:<14} {primer:>6} {_fmt_denominator(srv):>15} "
                     f"{per_block:>11} {cadence:>9} {calls:>17}")
    lines.append("  wrap a server when it clears its own row: below that rate the primer "
                 "costs more than the codec banks (#175).")
    # Each cadence explains itself only to an install that HAS it — the same reason the
    # prose above does not tell a router-only install about a one-time charge.
    shown = _cadences_of(servers)
    if _PER_TURN in shown:
        lines.append("  /turn = an eagerly-primed router: the break-even is blocks per "
                     "TURN, and it recurs.")
    if shown - {_PER_TURN}:
        lines.append("  1x = a lazily-primed standalone entry (#211): the break-even is "
                     "blocks ONCE PER SESSION, a far lower bar.")
        if _ONCE_FREE in shown or _ONCE_UNKNOWN in shown:
            lines.append("  1x? = called-ness unknown (no ledger label, an ambiguous one, "
                         "a label shared with another writer, or no rows")
            lines.append("  written at all); 1x- = unpaid, either")
            lines.append("  because it was never triggered or because every primer was "
                         "declined (#286).")
    lines.append("  a BLOCK is one emitted tool-result text block — >=1 per call, so this "
                 "is a conservative bar (#141).")
    return lines


def build_recommend_section(liab: dict[str, Any]) -> list[str]:
    """One verdict word per installed entry — the `--recommend` body (#238).

    A SEPARATE block rather than a column on the break-even table, and that is forced rather
    than preferred: the break-even row is already 79 of a pinned 80 columns (its own comment,
    and `test_the_break_even_row_stays_inside_eighty_columns`), while `INSUFFICIENT` is 12
    characters. Shrinking a neighbour to make room re-opens the overflow hole review closed
    once already, when `blocks` was narrowed and the live ledger immediately produced an
    11-character cell.

    Empty when nothing pays a primer — the same contract as `build_primer_section`, and for
    the same reason: absent and zero are different claims.

    Every legend line is gated on being APPLICABLE, mirroring `_build_break_even_table`'s
    `_cadences_of` discipline. An install with no TUNE row is not lectured about autotune,
    and an install with no lazy entry is not told about a one-session reading it never gets.
    """
    servers = liab.get("servers") or []
    if not servers:
        return []
    # 2 + 14+1 + 12+1 + 9+1 + 9+1 = 50 of fixed cells, plus the reason word — the longest in
    # the closed vocabulary is `router/live duplicate label` (27, #396 review), so the widest
    # row is 77. The slack is deliberate: this table's whole justification is that the
    # break-even row had none — and it is what lets this one print the reason in full where
    # `_build_break_even_table` has to abbreviate it (`_REASON_ABBR`).
    lines = [f"  {'server':<14} {'verdict':<12} {'cadence':>9} {'coverage':>9} why"]
    # Sorted by WHAT NEEDS ACTION, not by rate — deliberately different from the break-even
    # table directly above it. That one ranks servers by how well the codec fits them; this
    # one answers "what should I change today", and a KEEP row is the one line on the screen
    # that needs nothing, so it sorts last. Ties break on ascending coverage (the thinnest
    # margin first) and then on name, so the order is total and the output is deterministic.
    for srv in sorted(servers, key=lambda s: (_VERDICT_ORDER.get(s.get("verdict"), 99),
                                              s.get("break_even_coverage")
                                              if s.get("break_even_coverage") is not None
                                              else -1.0,
                                              s.get("server") or "")):
        verdict, coverage, why = _fmt_verdict(srv)
        name = srv["server"] if len(srv["server"]) <= 14 else srv["server"][:13] + "…"
        # `.get`, defaulted to a dash, exactly as the break-even table does: an older `--json`
        # blob has no cadence and a blank is honest where guessing would re-assert the defect.
        cadence = _CADENCE_ABBR.get(srv.get("cadence") or "", "–")
        lines.append(f"  {name:<14} {verdict:<12} {cadence:>9} {coverage:>9} {why}")
    verdicts = {s.get("verdict") for s in servers}
    if KEEP in verdicts or UNWRAP in verdicts:
        lines.append("  KEEP = cleared its own primer in this window. UNWRAP = no block "
                     "volume ever will.")
    if TUNE in verdicts:
        # The wording is load-bearing and pinned by a test. TUNE is a REACHABILITY statement:
        # terse modelled no policy change and structurally cannot — the ledger is payload-free
        # by design, so re-encoding a real payload under a hypothetical gate is not something
        # any amount of code in this module can do. Saying "a policy change would help" would
        # be the #144/#186/#188 family again, so this points at the tool that actually answers
        # the what-if instead of claiming the answer.
        lines.append("  TUNE = arithmetically reachable, not a modelled improvement.")
        lines.append("  terse has tested no policy change here — `terse policy autotune` is "
                     "the command that does.")
    if INSUFFICIENT in verdicts:
        lines.append("  INSUFFICIENT = the ledger cannot answer yet (no label, an "
                     "ambiguous one, a label shared with another writer,")
        lines.append("  no rows written at all, no token data, unreadable policy, or not "
                     "called).")
    shown = _cadences_of(servers)
    if _PER_TURN in shown:
        lines.append("  coverage on a /turn row is against ONE turn's charge — a router "
                     "pays it every turn, so compare it")
        lines.append("  against your own turn count.")
    if shown - {_PER_TURN}:
        lines.append("  coverage on a 1x row treats the whole window as one session — the "
                     "most favourable reading;")
        lines.append("  every further session paid the charge again.")
    # Contributors print ONLY where the pooling is otherwise invisible. A single-label entry's
    # contributor list is a copy of its own row and says nothing; a router's is the answer to
    # "which peer is carrying this?", which the entry-level verdict deliberately does not ask.
    pooled = [s for s in servers if len(s.get("contributors") or []) >= 2]
    if pooled:
        lines.append("  a router's peers are pooled under its one shared primer — the "
                     "verdict is per installed entry,")
        lines.append("  never per peer (#238).")
        for srv in pooled:
            ranked = "; ".join(f"{c['label']} {c['saved_tokens']:,}"
                               for c in srv["contributors"])
            lines.append(f"    {srv['server']} pools, by tokens saved: {ranked}")
    return lines


def build_recommend_report(agg: dict[str, Any], *, log_path: str | Path,
                           window: str | None = None,
                           liability: dict[str, Any] | None = None) -> str:
    """`terse stats --recommend` — the verdict screen, arithmetic one flag away.

    The signature mirrors `build_stats_report` exactly, including the `agg` this renderer
    never reads, so `_cmd_stats` swaps one call for the other rather than growing a second
    argument-marshalling path that can drift from the first. The verdict is computed from the
    LIABILITY blob alone; `agg` is already folded into it upstream (`primer_liability` takes
    it) and re-reading it here would be a second, independently-drifting view of the same
    numbers.

    `--recommend` REPLACES the ledger tables rather than appending to them. Replacing leaves
    the default report's bytes completely untouched, which is what keeps this change out of
    the several negative text assertions `test_primer_liability.py` already makes.
    """
    scope = f"last {window}" if window else "all time"
    lines = [f"terse recommend — {scope}  (ledger: {log_path})", ""]
    if liability is None:
        # Never an empty document, and never the WRONG refusal. The install could not be
        # scanned at all (`_cmd_stats` has already named the cause on stderr), which is not
        # the same claim as "nothing here is wrapped" — the same absent-vs-zero discipline
        # the rest of this module keeps, one layer up in the renderer.
        lines.append("no verdict: the install could not be sized, so there is no primer to "
                     "weigh this window's")
        lines.append("savings against (the reason is on stderr). `terse stats` still reports "
                     "the ledger totals.")
        return "\n".join(lines) + "\n"
    section = build_recommend_section(liability)
    if not section:
        lines.append("no verdict: nothing in this install pays a primer, so there is nothing "
                     "to weigh a window's")
        lines.append("savings against. `terse stats` still reports the ledger totals.")
        return "\n".join(lines) + "\n"
    lines += section
    # The basis, on the VERDICT screen. `--recommend` replaces the ledger tables rather than
    # appending to them, so the line `build_primer_section` adds never reaches this output —
    # and every number here moved when the basis did (`coverage` 2,840 -> 1,739 on the live
    # fleet, and the pooled "by tokens saved" ranking halved). USAGE documents this as the
    # screen an operator reads for the verdict, so it was the one screen that said nothing
    # about why its numbers changed (review of PR #435).
    if (liability or {}).get("saved_basis") == "context":
        lines.append("")
        lines.append("coverage and the pooled savings above are CONTEXT tokens — what the "
                     "model received, not what left")
        lines.append("the stdio pipe. A mirror text block a `structuredContent`-reading "
                     "client discards is wire saving (#420).")
    lines.append("")
    lines.append("run without --recommend for the arithmetic behind these words.")
    return "\n".join(lines) + "\n"


def build_stats_writer(stats_log: str | Path, server: str):
    """The proxy-side callback: (tool, raw, emitted, passthrough) -> appended record.
    Owns all I/O and NOTHING else, kept here so both run_proxy and run_multi_proxy wire
    it identically. A write failure propagates: stats is still never load-bearing, but
    the swallow-and-announce lives in the one caller with the bookkeeping for it,
    `proxy.Interceptor._warn_sink` — catching here too made its unconditional
    first-failure warning dead code, so a dead ledger stayed silent (#131)."""
    def stats(tool: str, raw: str, emitted: str, passthrough: bool,
              diff_reason: str | None = None, structured: str | None = None,
              structured_out: str | None = None) -> None:
        append_stats(build_record(server, tool, raw, emitted, passthrough, diff_reason,
                                  structured, structured_out),
                     stats_log)

    return stats


def build_primer_writer(stats_log: str | Path, server: str):
    """The proxy-side callback for a primer that actually went out (#311, #286).

    A THIRD writer for the same reason `build_retrieve_writer` is a second one: a primer is
    not a result, shares none of the result record's size fields, and fires from a different
    code path -- at most once per session, on the lazy attach. Widening the result writer
    would put a branch taken at most once on the hot path for every compressed block.

    Wired at exactly ONE site, `run_proxy`, and deliberately NOT alongside the other two in
    `multiproxy._build_peers` (#311; an earlier draft of this docstring claimed otherwise
    and was corrected in review). Two independent reasons, either sufficient:

      * Peers run `lazy_primer=False`, so a peer's `_primer_sent` starts True and its lazy
        attach can never fire. A per-peer writer would be dead code.
      * The router emits ONE `union_primer` for N peers. Wiring this into the peer loop with
        `spec.name` would bill N primers for the one the client receives -- the over-count
        that sank #312's design.

    The router's own union primer is therefore not recorded at all. It does not need to be:
    it is emitted unconditionally at `initialize`, which is the same predicate
    `primer_liability` already evaluates from the installed policy, so inference there is
    exact. Recording it would also require a ledger identity the router does not have --
    the reader derives a router's identity from its PEER names, so no synthetic label would
    join."""
    def primer(cadence: str, text: str, attached: bool = True) -> None:
        append_stats(build_primer_record(server, cadence=cadence, primer=text,
                                         attached=attached), stats_log)

    return primer


def build_retrieve_writer(stats_log: str | Path, server: str):
    """The proxy-side callback for a `terse.retrieve` round-trip (#251).

    Deliberately a SECOND writer rather than another parameter on `build_stats_writer`'s
    closure: a retrieve is not a result, shares none of the result record's size fields,
    and rides a different code path (`answer_retrieve`, which never forwards downstream).
    Widening the result writer would have put a rarely-taken branch on the hot path for
    every compressed block. Wired at the same two sites, so the two stay in lockstep.

    `server` here is only the DEFAULT. The caller passes the label captured at drop time,
    because under multiproxy the router answers every retrieve through `peers[0]`
    (`_route_call`) — so this closure's own label names the answering peer, which is
    almost never the peer whose rule dropped the value."""
    def retrieve(origin_server: str, tool: str, path: str, hit: bool, payload: str) -> None:
        append_stats(build_retrieve_record(origin_server or server, tool, path,
                                           hit=hit, payload=payload),
                     stats_log)

    return retrieve
