"""Live savings ledger: payload-free per-result stats from the proxy + aggregation.

The measurement story had a gap: `terse measure` proves savings over a captured
corpus, and `--debug-log` records full raw->emitted replays, but neither answers
"how much did terse save in my real sessions?" — the debug log embeds raw tool
payloads (the same secrets exposure as capture), so nobody leaves it on.

This ledger stores ONLY sizes and decisions — never payload content — so it is safe
to leave always-on (the proxy default; `--no-stats` opts out). One JSON line per
tool-result block: ts, server, tool, decision, raw/out chars, raw/out cl100k tokens
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

from ._secure_io import append_restricted
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
                 passthrough: bool, diff_reason: str | None = None) -> dict[str, Any]:
    """One ledger line. Sizes and labels only — never payload content (the property
    that makes always-on safe). Token counts are None without tiktoken.

    `diff_reason` (Phase 1 instrumentation) records WHY the cross-call diff did or
    did not fire for this result — the datum that decides whether arg-keying the diff base
    is worth building. See proxy `_compress_or_diff` for the value set. None on older
    records (the field post-dates them) and on writers that don't supply it."""
    return {
        "ts": int(time.time()),
        "server": server,
        "tool": tool,
        "decision": classify_decision(raw, emitted, passthrough),
        "diff_reason": diff_reason,
        "raw_chars": len(raw),
        "out_chars": len(emitted),
        "raw_tokens": count_cl100k(raw),
        "out_tokens": count_cl100k(emitted),
    }


def append_stats(record: dict[str, Any], log_path: str | Path,
                 max_bytes: int = MAX_LEDGER_BYTES) -> None:
    """Append one record, rotating the live file to `.1` once it passes `max_bytes`.
    Restricted perms for consistency with every other terse-managed file, even though
    records are payload-free."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
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
    total = {"results": 0, "raw_chars": 0, "out_chars": 0,
             "raw_tokens": 0, "out_tokens": 0, "untokenized": 0}
    decisions: dict[str, int] = {}
    # Phase 1: why the cross-call diff did/didn't fire (only present on newer records).
    diff_reasons: dict[str, int] = {}
    tools: dict[tuple[str, str], dict[str, int]] = {}
    for rec in records:
        raw_c, out_c = rec.get("raw_chars"), rec.get("out_chars")
        if not (isinstance(raw_c, int) and isinstance(out_c, int)):
            continue  # not a ledger record
        total["results"] += 1
        total["raw_chars"] += raw_c
        total["out_chars"] += out_c
        decision = str(rec.get("decision", "unknown"))
        decisions[decision] = decisions.get(decision, 0) + 1
        reason = rec.get("diff_reason")
        if isinstance(reason, str):
            diff_reasons[reason] = diff_reasons.get(reason, 0) + 1
        key = (str(rec.get("server", "unknown")), str(rec.get("tool", "unknown")))
        row = tools.setdefault(key, {"results": 0, "raw_tokens": 0, "out_tokens": 0,
                                     "raw_chars": 0, "out_chars": 0, "diffs": 0})
        row["results"] += 1
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
        else:
            total["untokenized"] += 1
    return {"total": total, "decisions": decisions, "diff_reasons": diff_reasons,
            "tools": [{"server": s, "tool": t, **row}
                      for (s, t), row in sorted(
                          tools.items(),
                          key=lambda kv: kv[1]["raw_tokens"] - kv[1]["out_tokens"],
                          reverse=True)]}


def _pct_saved(raw: int, out: int) -> str:
    return f"{(raw - out) / raw * 100:5.1f}%" if raw else "    –"


def _hit_rate(diffs: int, results: int) -> str:
    """diffs / results as a percent — the cross-call diff hit rate this ledger exists
    to measure (a raw count alone is meaningless without its denominator). Blank on a
    zero denominator, which can't happen per-row but keeps the helper total."""
    return f"{diffs / results * 100:4.0f}%" if results else "    –"


def build_stats_report(agg: dict[str, Any], *, log_path: str | Path,
                       window: str | None = None) -> str:
    """Human-readable rollup. Tokens are the headline when available; chars are the
    honest fallback, labeled as such (never silently presented as tokens)."""
    total, decisions, tools = agg["total"], agg["decisions"], agg["tools"]
    scope = f"last {window}" if window else "all time"
    lines = [f"terse stats — {scope}  (ledger: {log_path})", ""]
    if total["results"] == 0:
        if window:
            # The ledger isn't necessarily empty — the window filtered everything out.
            # Point at the window, not at "nothing ever recorded" (the wrong cause).
            lines.append(f"no results in the last {window} — widen --since or drop it "
                         "(older results may still be in the ledger).")
        else:
            lines.append("no results recorded — has a terse-wrapped server handled a "
                         "tool call since stats shipped?")
        return "\n".join(lines) + "\n"
    tok_raw, tok_out = total["raw_tokens"], total["out_tokens"]
    lines.append(f"results: {total['results']}   "
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
    lines.append(f"{'server':<18} {'tool':<34} {'results':>7} {'diffs':>5} "
                 f"{'diff%':>5} {raw_col:>10} {out_col:>10} {'saved':>6}")
    for row in tools:
        raw_n = row["raw_tokens"] if use_tokens else row["raw_chars"]
        out_n = row["out_tokens"] if use_tokens else row["out_chars"]
        lines.append(f"{row['server'][:18]:<18} {row['tool'][:34]:<34} "
                     f"{row['results']:>7} {row['diffs']:>5} "
                     f"{_hit_rate(row['diffs'], row['results']):>5} "
                     f"{raw_n:>10,} {out_n:>10,} "
                     f"{_pct_saved(raw_n, out_n):>6}")
    return "\n".join(lines) + "\n"


def build_stats_writer(stats_log: str | Path, server: str):
    """The proxy-side callback: (tool, raw, emitted, passthrough) -> appended record.
    Owns all I/O and NOTHING else, kept here so both run_proxy and run_multi_proxy wire
    it identically. A write failure propagates: stats is still never load-bearing, but
    the swallow-and-announce lives in the one caller with the bookkeeping for it,
    `proxy.Interceptor._warn_sink` — catching here too made its unconditional
    first-failure warning dead code, so a dead ledger stayed silent (#131)."""
    def stats(tool: str, raw: str, emitted: str, passthrough: bool,
              diff_reason: str | None = None) -> None:
        append_stats(build_record(server, tool, raw, emitted, passthrough, diff_reason),
                     stats_log)

    return stats
