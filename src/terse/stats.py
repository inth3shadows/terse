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
from typing import Any
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
             "raw_tokens": 0, "out_tokens": 0, "untokenized": 0, "unversioned": 0}
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
    for rec in records:
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
        row = tools.setdefault(key, {"blocks": 0, "tokenized": 0,
                                     "raw_tokens": 0, "out_tokens": 0,
                                     "raw_chars": 0, "out_chars": 0, "diffs": 0})
        row["blocks"] += 1
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
        # off (#170 — the primer paragraph cost ~900-2,700x what the tier saved at a 0.38%
        # hit rate). Say so where the question is actually asked, not only in the dataclass.
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
    lines += build_version_section(agg)
    if liability:
        lines += build_primer_section(liability)
    return "\n".join(lines) + "\n"


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

# States whose entry runs its own `terse proxy`/`multiproxy` and therefore pays a primer.
# "folded*" peers are stashed BEHIND a router — the router pays one union primer for the
# fleet and the peers pay nothing, so counting them would double-charge.
_PAYS_PRIMER = ("wrapped", "wrapped-unstashed", "router", "router-ambiguous")

# Of those, the states that still prime EAGERLY at `initialize.instructions`, which the
# client re-reads every turn as `cache_read` — a recurring per-turn charge. Everything else
# in `_PAYS_PRIMER` is a standalone `run_proxy` entry, lazy since #211: one attach, to the
# first compressible result, and nothing at all if that result never comes.
_PRIMES_EAGERLY = ("router", "router-ambiguous")

# Per-server cadence labels. The whole point of splitting them is that `tok/turn` and
# `tok/session` are different units and summing them was the defect (#211 follow-up).
_PER_TURN = "per-turn"
_ONCE = "once/session"
_ONCE_FREE = "once/session (unpaid)"
_ONCE_UNKNOWN = "once/session (?)"

# The same four, short enough for a table cell that has to fit inside 80 columns alongside
# five other. The prose labels stay the `--json` contract; these are render-only.
_CADENCE_ABBR = {_PER_TURN: "/turn", _ONCE: "1x", _ONCE_FREE: "1x-", _ONCE_UNKNOWN: "1x?"}


def _cadence(state: str | None, blocks: int | None) -> str:
    """How often this entry actually pays its primer, post-#211.

    `blocks` is the ledger's answer to "was it called", and its three-way None/0/N is load
    bearing here exactly as it is in `_break_even`: None means no label was recoverable, so
    we never found the rows to ask — NOT that the server was idle. Collapsing it to "never
    called" would move an unknown into the `free` bucket and quietly under-report."""
    if state in _PRIMES_EAGERLY:
        return _PER_TURN
    if blocks is None:
        return _ONCE_UNKNOWN
    return _ONCE if blocks else _ONCE_FREE


def _break_even(primer_tokens: int | None, blocks: int | None,
                tokenized: int | None, saved: int) -> dict[str, Any]:
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
      never called      installed, pays a primer, banked nothing this window.
      no token data     called, but every matching row was recorded without tiktoken. The
                        savings in TOKENS are unknown, not zero — and dividing a cl100k
                        primer by a char-derived rate would silently mix units.
      primer unknown    the rate is real but the server's policy could not be read, so the
                        threshold it must clear is unknown. Reporting this as `never` would
                        condemn a server on evidence about a missing FILE (found in review
                        of #197).
      no primer         a default-deny policy emits no compressed form and therefore no
                        primer — nothing to earn back, at any rate, positive or negative.
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
        return {"saved_per_block": None, "blocks_to_break_even": None,
                "break_even_verdict": "no ledger label"}
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


def primer_liability(scan_rows: list[dict[str, Any]], agg: dict[str, Any]) -> dict[str, Any]:
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
    from .policy import default_policy, load_policy
    from .proxy import build_primer, union_primer

    by_label: dict[str, int] = {}
    # Savings too, not just call counts (#175): the per-server break-even below divides one
    # by the other, and rolling them up in the same pass keeps them from being taken over
    # different row sets.
    saved_by_label: dict[str, int] = {}
    tokenized_by_label: dict[str, int] = {}
    for trow in agg.get("tools", []):
        by_label[trow["server"]] = by_label.get(trow["server"], 0) + trow["blocks"]
        lbl = trow["server"]
        raw_t, out_t = trow.get("raw_tokens") or 0, trow.get("out_tokens") or 0
        saved_by_label[lbl] = saved_by_label.get(lbl, 0) + (raw_t - out_t)
        # `.get("tokenized")`, defaulted: a caller may hand us a hand-rolled agg (the tests
        # do) or one produced before this counter existed. Falling back to `blocks` there
        # keeps the old — merely coarser — behaviour instead of reporting `no token data`
        # for a row that plainly has token sums.
        tok = trow.get("tokenized")
        if tok is None:
            tok = trow["blocks"] if raw_t or out_t else 0
        tokenized_by_label[lbl] = tokenized_by_label.get(lbl, 0) + tok
    servers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in scan_rows:
        name, state = row.get("server"), row.get("state")
        if state not in _PAYS_PRIMER or not name or name in seen:
            continue
        seen.add(name)
        is_router = state in ("router", "router-ambiguous")
        # `wraps` means two different things by state, and reading it the wrong way is how
        # a busy router reports as never-called: for a wrapped entry it is the downstream
        # COMMAND (the ledger keys on `server_label` of that, not on the MCP entry name),
        # for a router it is the comma-joined peer NAMES — and multiproxy tags each peer's
        # ledger records with its own name, so those names ARE the labels.
        wraps = row.get("wraps") or ""
        peers = [p for p in (q.strip() for q in wraps.split(",")) if p] if is_router else []
        labels = peers if is_router else ([server_label(wraps.split())] if wraps else [])
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
        # None, not 0, when no label could be recovered: "unknown" and "never called" are
        # different claims, and only the second one accuses an install of being pure cost.
        blocks = sum(by_label.get(lbl, 0) for lbl in labels) if labels else None
        # Same `if labels else None` guard as `blocks`: with no recoverable label, 0 would
        # claim "nothing was tokenized" to a `--json` consumer when the truth is that we
        # never found the rows to ask (review of #197).
        tokenized = (sum(tokenized_by_label.get(lbl, 0) for lbl in labels)
                     if labels else None)
        servers.append({"server": name, "scope": row.get("scope"), "state": state,
                        "primer_tokens": tokens, "ledger_labels": labels, "blocks": blocks,
                        "tokenized_blocks": tokenized, "cadence": _cadence(state, blocks),
                        **_break_even(tokens, blocks, tokenized,
                                      sum(saved_by_label.get(lbl, 0) for lbl in labels))})

    # Each server lands in exactly one bucket, and only two of them are money.
    per_turn = sum(s["primer_tokens"] or 0 for s in servers if s["cadence"] == _PER_TURN)
    once = sum(s["primer_tokens"] or 0 for s in servers if s["cadence"] == _ONCE)
    unresolved = sum(1 for s in servers if s["primer_tokens"] is None)
    total = agg.get("total") or {}
    saved = (total.get("raw_tokens") or 0) - (total.get("out_tokens") or 0)
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
        # Installed, lazy, not triggered this window — costs nothing at all. Reported
        # because it is the thing #211 bought, and because the old code billed exactly these
        # servers as pure cost. Guarded on a non-zero primer for the same reason `idle` is:
        # a default-deny server is free because it emits nothing, which is a different fact
        # and already has its own `no primer` verdict in the table.
        "free": [s["server"] for s in servers
                 if s["cadence"] == _ONCE_FREE and s["primer_tokens"]],
        # Lazy, but no ledger label was recoverable, so we cannot say whether the attach
        # ever fired. Neither total counts it — same discipline as `unresolved`.
        "uncertain": [s["server"] for s in servers if s["cadence"] == _ONCE_UNKNOWN],
        "saved_tokens": saved,
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
    }


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
    cadences = {s.get("cadence") or _PER_TURN for s in servers}
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
    if liab["unresolved"]:
        lines.append(f"  {liab['unresolved']} server(s) have an unreadable policy and are "
                     f"NOT counted — treat both figures as lower bounds.")
    if liab.get("uncertain"):
        lines.append(f"  no ledger label, so it is unknown whether the lazy primer ever "
                     f"attached: {', '.join(sorted(liab['uncertain']))}")
    saved = liab["saved_tokens"]
    turns = liab["turns_covered"]
    if turns is not None:
        # Below 1.0 the ratio is not the interesting number — the shortfall is. `~0.4 turns`
        # and `~0x over` both round to a `0` that reads as a measured zero, the same
        # print-as-zero hole `_fmt_rate` closed in review of #197.
        if turns >= 1:
            lines.append(f"  the {saved:,} tok saved in this window pays for ~{turns:,.0f} "
                         f"turn(s) of the recurring primer.")
        else:
            lines.append(f"  NET NEGATIVE over this window: the {saved:,} tok saved does "
                         f"not cover even a single turn of the {per_turn:,} tok recurring "
                         f"primer.")
    if liab.get("session_covered") is not None:
        # Rendered ALONGSIDE the recurring line, not instead of it, and never netted into
        # it: the two divide different things. And stated as a ceiling, because the window
        # spans an unknown number of sessions and each one paid this charge again.
        covered = liab["session_covered"]
        if covered >= 1:
            lines.append(f"  the same {saved:,} tok covers the {once:,} tok one-time charge "
                         f"at most ~{covered:,.0f}x — fewer if this")
            lines.append("  window spans more than one session, which it usually does; "
                         "each of them paid that charge again.")
        else:
            lines.append(f"  NET NEGATIVE over this window: the {saved:,} tok saved does "
                         f"not cover the {once:,} tok one-time charge even once, and each "
                         f"session in the window paid it again.")
    if liab["idle"]:
        lines.append(f"  paying every turn but never called here: "
                     f"{', '.join(sorted(liab['idle']))} — pure cost until they handle a "
                     f"compressible result.")
    if liab.get("free"):
        lines.append(f"  installed but not triggered this window, so costing nothing at "
                     f"all (#211): {', '.join(sorted(liab['free']))}")
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
        return rate, verdict or "–"
    return rate, f"{calls:,.2f}"


def _fmt_denominator(srv: dict[str, Any]) -> str:
    """The call count the rate was taken over.

    Shown as `tokenized/blocks` when they differ, so a partially-untokenized ledger says so
    in the column whose number it changes rather than only in the header's `N uncounted`
    aside (found in review of #197)."""
    blocks, tok = srv["blocks"], srv.get("tokenized_blocks")
    if blocks is None:
        return "–"
    if tok is not None and tok != blocks:
        return f"{tok:,}/{blocks:,}"
    return f"{blocks:,}"


def _build_break_even_table(servers: list[dict[str, Any]]) -> list[str]:
    """Per-server `saved/block` and the block count that pays for that server's primer.

    Gated on whether any server was CALLED in this window, not on whether any produced a
    rate. An install where nothing was called renders a table of dashes that says nothing
    the `idle` line above did not already say — but a server that WAS called and still has
    no rate is the tiktoken-missing case, and "no token data" against a real block count is
    exactly the row that sends the operator to fix it."""
    if not any(s.get("blocks") for s in servers):
        return []
    # Widths are load-bearing twice over: four tests match right-aligned cells, and the row
    # has to stay inside 80 columns or the last cells fold onto the next line and the table
    # stops being one. Adding `cadence` at its full label width took the row to 104, so the
    # cadence values are abbreviated (`_CADENCE_ABBR`) and `blocks` — which only ever holds
    # `N` or `tokenized/N` — gives back the space it was never using.
    lines = ["", f"  {'server':<18} {'primer':>6} {'blocks':>11} {'saved/block':>11} "
                 f"{'cadence':>9} {'to break even':>17}"]
    # Rateless rows sort last as a group rather than tying with a 0.0 rate: `or -1` treated
    # a break-even server (0.0, falsy) as worse than one actively LOSING tokens (-0.5,
    # truthy), inverting the two rows an operator most needs ordered (review of #197).
    for srv in sorted(servers, reverse=True,
                      key=lambda s: (s.get("saved_per_block") is not None,
                                     s.get("saved_per_block") or 0.0)):
        per_block, calls = _fmt_break_even(srv)
        primer = "?" if srv["primer_tokens"] is None else f"{srv['primer_tokens']:,}"
        name = srv["server"] if len(srv["server"]) <= 18 else srv["server"][:17] + "…"
        # `.get`, defaulted to a dash: an older `--json` blob has no cadence at all, and a
        # blank there is honest where guessing `per-turn` would re-assert the old defect.
        cadence = _CADENCE_ABBR.get(srv.get("cadence") or "", "–")
        lines.append(f"  {name:<18} {primer:>6} {_fmt_denominator(srv):>11} "
                     f"{per_block:>11} {cadence:>9} {calls:>17}")
    lines.append("  wrap a server when it clears its own row: below that rate the primer "
                 "costs more than the codec banks (#175).")
    # Each cadence explains itself only to an install that HAS it — the same reason the
    # prose above does not tell a router-only install about a one-time charge.
    shown = {s.get("cadence") for s in servers}
    if _PER_TURN in shown:
        lines.append("  /turn = an eagerly-primed router: the break-even is blocks per "
                     "TURN, and it recurs.")
    if shown - {_PER_TURN}:
        lines.append("  1x = a lazily-primed standalone entry (#211): the break-even is "
                     "blocks ONCE PER SESSION, a far lower bar.")
        if _ONCE_FREE in shown or _ONCE_UNKNOWN in shown:
            lines.append("  1x? = called-ness unknown (no ledger label); 1x- = installed "
                         "but not triggered, so unpaid.")
    lines.append("  a BLOCK is one emitted tool-result text block — >=1 per call, so this "
                 "is a conservative bar (#141).")
    return lines


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
