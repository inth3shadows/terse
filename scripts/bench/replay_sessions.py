#!/usr/bin/env python3
"""Replay real Claude Code sessions to price what terse did to billed cost.

The cost-per-task suite (`cost_per_task/`) runs short, single-question tasks with three or
four tools allowed. It is a regression check. It cannot say what terse is worth in real
sessions, where a result stays in context and is re-read on every later API call. This
script answers that from data already on disk, with no model calls:

  ledger       `terse stats`'s payload-free rows: per result, the raw and emitted sizes.
  transcripts  Claude Code's own records: each tool result as the model saw it, and the
               billed `usage` of every API call.

Method
------
1. Group ledger rows into results (a multi-block result writes one row per block) and join
   each to a transcript tool result: same tool, nearest timestamp at or before it, one
   ledger result per transcript result. A join is kept only if the size the ledger recorded
   agrees with the size the transcript shows; a group that fails is split into its rows
   and tried again (parallel calls of one tool land in the same second), and whatever
   still disagrees is left out and counted.
2. delta = (what the raw result would have cost in context) - (what the model was shown),
   in cl100k tokens. The shown side is tokenized from the transcript, so the primer and any
   wrapper are charged to terse automatically.
3. Price delta on every API call that carried it: the write rate on the call that first
   included it, the read rate on later calls, the write rate again on a call that rebuilt
   the cache, and nothing after a compaction. Rates come from each call's own usage block
   and model.
4. Charge a `terse_retrieve` round trip its result tokens (same carry) plus the cached
   prefix its extra API call re-read.
5. Report the net twice: against what the results terse handled would have cost with
   terse off (what an MCP proxy can reach), and against the whole bill of every session
   scanned (mostly shell output, file reads and model output, which it cannot).

What it cannot show: whether the model would have made the same calls with terse off. The
replay assumes it would. `recall_3` (an identical-argument call of the same tool within
three calls) is reported for changed and unchanged results as the visible proxy.

Not charged to terse (each would lower the result): a primer sent at `initialize` by a
router older than #212, the `terse_retrieve` tool definition, and the few output tokens of
a retrieve's own tool call.

Units: the ledger counts cl100k and Claude bills its own tokenizer. Deltas are scaled by a
per-model ratio measured from the transcripts (`tokenizer_ratio`); the cl100k figure is
kept beside it.

Rows are written to `<out>/sessions.jsonl` as each transcript finishes; `<out>/summary.json`
and a text table follow at the end.
"""

from __future__ import annotations

import argparse
import bisect
import calendar
import importlib.util
import json
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab_session = _load("ab_session", HERE / "ab_session.py")
cost_model = _load("cost_model", HERE / "cost_per_task" / "cost_model.py")

from terse.tokenize import count_cl100k  # noqa: E402 -- after the sys.path setup

# Claude Code saves an MCP result to a file past either limit (code.claude.com/docs/en/mcp;
# the 50,000-char one measured in #496). The model then sees only a short preview.
OFFLOAD_TOKENS = 25_000
OFFLOAD_CHARS = 50_000
OFFLOAD_MARGIN = 0.7
# How the saved-to-file notice opens: the current wording, and the one older CLI versions
# used (both read off the live transcripts, 2026-07 to 2026-10).
OFFLOAD_MARKS = ("<persisted-output>",)
OFFLOAD_NOTICE_RE = re.compile(r"Error: result \([\d,]+ characters\) exceeds maximum allowed tokens")

# Ledger `ts` is whole seconds at emit; the transcript stamps the result when the client
# records it. Measured on the live data: 5th..99th percentile of (transcript - ledger) is
# 0.0..1.0 s, min -3.0 (ledger rounds down, clocks are the same host), max 7.0.
JOIN_BEFORE = 10.0
JOIN_AFTER = 3.0
CLUSTER_GAP = 1  # seconds between rows of one multi-block result
PARALLEL_SLACK = 2.0  # seconds around a group's rows in which a result counts as parallel

# How far the size the model was shown may sit from the size the ledger recorded and still
# be the same result. The client wraps and joins blocks (a constant 13-16 chars on the
# live data); a result that carried the primer is longer by the primer and its wrapper.
# The primer is assembled per policy: 379 chars for a dict-only one, about 2,240 for a
# router's union (`proxy._assemble_primer`).
SIZE_SLACK_CHARS = 40
SIZE_SLACK_PER_ROW = 16
SIZE_SLACK_SHARE = 0.02
PRIMER_BAND = (350, 3200)
CALIBRATION_MIN_SAMPLES = 20

# A call whose prompt is under this share of the previous call's prompt lost its history
# without a `compact_boundary` record (an older CLI, or a cleared context).
DROP_RATIO = 0.5
# A call that read under this share of the previous prompt from cache rebuilt the prefix.
REBUILD_RATIO = 0.5

RETRIEVE_SUFFIX = "terse_retrieve"
RECALL_WINDOW = 3
CALIBRATION_MIN_TOKENS = 3000

LENGTH_BUCKETS = ((5, "1-5 calls"), (20, "6-20 calls"), (100, "21-100 calls"),
                  (10**9, "over 100 calls"))


def norm_tool(name: str) -> str:
    """One spelling for a tool across the ledger (`kb__kb.read.search`, `kb.read.search`)
    and a transcript (`mcp__terse__kb_read_search`): the last `__` segment, with every
    non-alphanumeric folded to `_`."""
    return re.sub(r"[^a-z0-9]", "_", (name or "").lower().split("__")[-1])


def parse_ts(stamp: str) -> float:
    """A transcript's ISO-8601 UTC timestamp as epoch seconds."""
    base = calendar.timegm(time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S"))
    frac = stamp[19:].rstrip("Z")
    return base + (float("0" + frac) if frac.startswith(".") else 0.0)


# ---------------------------------------------------------------- ledger


@dataclass
class LedgerResult:
    """One tool result as the ledger saw it: every block's row, summed."""
    tool: str
    ts: int
    rows: list = field(default_factory=list)
    # A row split out of a group: it may only pair with a result of its own size. The
    # primer band is wide enough that a whole multi-block result would otherwise "fit"
    # its first block.
    exact_only: bool = False

    def _sum(self, key: str) -> int:
        return sum(r.get(key) or 0 for r in self.rows)

    @property
    def text_out_chars(self) -> int:
        return self._sum("out_chars") - self._sum("structured_out_chars")

    @property
    def typed_out_chars(self) -> int:
        return self._sum("structured_out_chars")

    @property
    def has_tokens(self) -> bool:
        return all(isinstance(r.get("raw_tokens"), int) for r in self.rows)

    def raw_side(self, typed: bool) -> tuple[int, int]:
        """(tokens, chars) of the raw result on the basis the model read: the typed field
        alone, or the text blocks alone."""
        if typed:
            return self._sum("structured_tokens"), self._sum("structured_chars")
        return (self._sum("raw_tokens") - self._sum("structured_tokens"),
                self._sum("raw_chars") - self._sum("structured_chars"))

    @property
    def changed(self) -> bool:
        return any(r.get("decision") in ("compressed", "diff") for r in self.rows)

    def fit(self, shown_chars: int) -> int | None:
        """How `shown_chars` matches what this ledger result would look like to the model,
        as its text blocks or as its typed field: 0 for the same size, 1 for that size
        plus a primer, None for neither."""
        best = None
        for expected in (self.text_out_chars, self.typed_out_chars):
            if expected <= 0:
                continue
            slack = (SIZE_SLACK_CHARS + SIZE_SLACK_PER_ROW * len(self.rows)
                     + SIZE_SLACK_SHARE * expected)
            extra = shown_chars - expected
            if abs(extra) <= slack:
                return 0
            if not self.exact_only and PRIMER_BAND[0] <= extra <= PRIMER_BAND[1] + slack:
                best = 1
        return best

    def near_offload_limit(self) -> bool:
        """Could the client have saved this result to a file? Its emitted size is within
        OFFLOAD_MARGIN of a limit: the limits are in the client's own tokenizer and the
        primer adds to what terse recorded."""
        out_tokens = self._sum("out_tokens")
        return (max(self.text_out_chars, self.typed_out_chars) > OFFLOAD_MARGIN * OFFLOAD_CHARS
                or out_tokens > OFFLOAD_MARGIN * OFFLOAD_TOKENS)

    def size_agrees(self, shown_chars: int) -> bool:
        return self.fit(shown_chars) is not None


def load_ledger(path: Path) -> list[LedgerResult]:
    """Result rows grouped into results: consecutive rows of one tool from one writer no
    more than CLUSTER_GAP seconds apart. `join` splits a group that turns out to be
    parallel calls."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event", "result") == "result" and "decision" in rec:
                rows.append(rec)
    rows.sort(key=lambda r: r["ts"])
    out: list[LedgerResult] = []
    last: dict[tuple, LedgerResult] = {}
    for rec in rows:
        key = (norm_tool(rec.get("tool", "")), rec.get("router") or rec.get("server"))
        cur = last.get(key)
        if cur is not None and rec["ts"] - cur.rows[-1]["ts"] <= CLUSTER_GAP:
            cur.rows.append(rec)
            continue
        cur = LedgerResult(key[0], rec["ts"], [rec])
        last[key] = cur
        out.append(cur)
    return out


# ---------------------------------------------------------------- transcripts


@dataclass
class Call:
    """One billed API response."""
    model: str | None
    inp: int
    read: int
    w5: int
    w1: int
    out: int
    # False for a response another transcript already recorded: a resumed or forked
    # session copies history, and the copy was not billed again.
    billed: bool = True

    @property
    def prompt(self) -> int:
        return self.inp + self.read + self.w5 + self.w1

    def read_weight(self) -> float:
        return cost_model._cache_read_weight(self.model, ab_session)

    def tail_weight(self) -> float:
        """Rate paid for content this call saw for the first time: its own mix of
        uncached input and cache writes. A fully cached call wrote nothing new, so new
        content there can only have been read."""
        new = self.inp + self.w5 + self.w1
        if new <= 0:
            return self.read_weight()
        return (self.inp * ab_session.W_INPUT + self.w5 * cost_model.W_CACHE_WRITE_5M
                + self.w1 * cost_model.W_CACHE_WRITE_1H) / new

    def weighted(self) -> float:
        return (self.inp * ab_session.W_INPUT + self.read * self.read_weight()
                + self.w5 * cost_model.W_CACHE_WRITE_5M
                + self.w1 * cost_model.W_CACHE_WRITE_1H + self.out * ab_session.W_OUTPUT)

    def usd_per_weighted(self) -> float | None:
        price = cost_model._price_for_model(self.model)
        return price["input"] / 1e6 if price else None


@dataclass
class Result:
    """One MCP tool result as the model was shown it."""
    tool_use_id: str
    name: str
    ts: float
    chars: int
    tokens: int | None
    offloaded: bool
    args: str


def _usage_call(usage: dict, model: str | None) -> Call:
    total_write = usage.get("cache_creation_input_tokens", 0) or 0
    split = usage.get("cache_creation")
    if isinstance(split, dict) and ("ephemeral_5m_input_tokens" in split
                                    or "ephemeral_1h_input_tokens" in split):
        w5 = split.get("ephemeral_5m_input_tokens", 0) or 0
        w1 = split.get("ephemeral_1h_input_tokens", 0) or 0
    else:
        w5, w1 = total_write, 0
    return Call(model, usage.get("input_tokens", 0) or 0,
                usage.get("cache_read_input_tokens", 0) or 0, w5, w1,
                usage.get("output_tokens", 0) or 0)


def is_offload_notice(text: str) -> bool:
    """Is this what the model was shown in place of a result the client saved to a file?"""
    head = text.lstrip()[:200]
    return head.startswith(OFFLOAD_MARKS[0]) or bool(OFFLOAD_NOTICE_RE.match(head))


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def scan_transcript(path: Path, billed_elsewhere: set[tuple] | None = None) -> list[tuple]:
    """The transcript as an ordered event list: ("call", Call), ("result", Result),
    ("compact",). A response split over several records is one call (the dedup key is
    cost_model's); a synthetic all-zero usage record is not a call.

    `billed_elsewhere` is the set of responses earlier transcripts already recorded, and
    is added to. A response found there keeps its place in the order, so later calls
    still compare against the right prompt size, but is marked unbilled."""
    events: list[tuple] = []
    seen: set[tuple] = set()
    if billed_elsewhere is None:
        billed_elsewhere = set()
    uses: dict[str, tuple[str, str]] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("type")
            if kind == "system" and rec.get("subtype") == "compact_boundary":
                events.append(("compact",))
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if kind == "assistant":
                if isinstance(content, list):
                    for block in content:
                        if (isinstance(block, dict) and block.get("type") == "tool_use"
                                and str(block.get("name", "")).startswith("mcp__")):
                            uses[block.get("id")] = (
                                block["name"],
                                json.dumps(block.get("input"), sort_keys=True))
                key = (rec.get("requestId"), msg.get("id"))
                usage = msg.get("usage")
                if key in seen or not cost_model.is_real_turn_usage(usage):
                    continue
                seen.add(key)
                call = _usage_call(usage, msg.get("model"))
                if None not in key:
                    call.billed = key not in billed_elsewhere
                    billed_elsewhere.add(key)
                events.append(("call", call))
            elif kind == "user" and isinstance(content, list):
                for block in content:
                    if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                        continue
                    use = uses.get(block.get("tool_use_id"))
                    if use is None or not rec.get("timestamp"):
                        continue
                    text = _result_text(block.get("content"))
                    events.append(("result", Result(
                        block["tool_use_id"], use[0], parse_ts(rec["timestamp"]),
                        len(text), count_cl100k(text),
                        is_offload_notice(text), use[1])))
    return events


# ---------------------------------------------------------------- join


def _fit(res: Result, led: LedgerResult) -> int | None:
    """`LedgerResult.fit` for `res`. A file preview's size says nothing, so an offloaded
    result fits any row of its tool big enough to have been offloaded, ranked with the
    primer-band fits. A small row is never its record: taking one would cost the row's
    owner its match."""
    if res.offloaded:
        return 1 if led.near_offload_limit() else None
    return led.fit(res.chars)


def _in_window(res: Result, led: LedgerResult) -> bool:
    return led.ts - JOIN_AFTER <= res.ts <= led.ts + JOIN_BEFORE


def _alongside(res: Result, led: LedgerResult) -> bool:
    """Did `res` arrive while `led`'s rows were being written? Parallel calls of one tool
    do; a second call a few seconds later does not. Tighter than `_in_window` on purpose:
    with the full window a real multi-block result was split whenever its tool was called
    again within ten seconds, and its rows were then lost (measured: 205 groups)."""
    return led.rows[0]["ts"] - PARALLEL_SLACK <= res.ts <= led.rows[-1]["ts"] + PARALLEL_SLACK


def _claim(ledger: list[LedgerResult], results: list[Result]) -> dict[str, int]:
    """tool_use_id -> index into `ledger`, each index claimed once, and only by a result
    whose size fits (`_fit`): a result of another server with the same tool name must not
    take a row from its owner. The ledger stamp is floored to the second at emit, so the true
    match is at or before the transcript's time: those rank first, then nearest."""
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, led in enumerate(ledger):
        index[led.tool].append((led.ts, i))
    for rows in index.values():
        rows.sort()
    taken: set[int] = set()
    out: dict[str, int] = {}
    for res in sorted(results, key=lambda r: r.ts):
        rows = index.get(norm_tool(res.name))
        if not rows:
            continue
        lo = bisect.bisect_left(rows, (res.ts - JOIN_BEFORE, -1))
        hi = bisect.bisect_right(rows, (res.ts + JOIN_AFTER, 10**12))
        # A same-size fit outranks a primer-band fit whatever the timing: the band is
        # wide, and two parallel calls of different sizes would otherwise swap rows.
        free = [(fit, ts > res.ts, abs(res.ts - ts), i) for ts, i in rows[lo:hi]
                if i not in taken and (fit := _fit(res, ledger[i])) is not None]
        if not free:
            continue
        i = min(free)[3]
        taken.add(i)
        out[res.tool_use_id] = i
    return out


def _rows_of(led: LedgerResult) -> list[LedgerResult]:
    return [LedgerResult(led.tool, row["ts"], [row], exact_only=True) for row in led.rows]


def join(ledger: list[LedgerResult],
         results: list[Result]) -> tuple[dict[str, LedgerResult], dict]:
    """(tool_use_id -> ledger result, counts).

    Rows of one tool written in the same second are either blocks of one result or
    parallel calls. A multi-row group is split into its rows up front when two or more
    transcript results of that tool arrived alongside it (`_alongside`), and after the
    first pass when no result fits it whole. Whatever no result fits is left out, so a wrong pairing cannot
    be priced; `mismatch_offsets` buckets how far the nearest candidate was off, so a
    systematic drop is visible.

    A tool_use_id that recurs (a resumed or forked session copies history) is joined
    once; every copy then replays against the same ledger result.

    `counts["ledger_results"]` is the denominator of the overall join rate. It is counted
    after the splits, so merging rows can never raise the rate."""
    first: dict[str, Result] = {}
    for res in results:
        first.setdefault(res.tool_use_id, res)
    by_tool: dict[str, list[Result]] = defaultdict(list)
    for res in first.values():
        by_tool[norm_tool(res.name)].append(res)

    def near(led: LedgerResult, pool) -> list[Result]:
        return [r for r in pool if _in_window(r, led)]

    splits = 0
    pool: list[LedgerResult] = []
    for led in ledger:
        if len(led.rows) > 1 and sum(
                _alongside(r, led) for r in by_tool.get(led.tool, ())) >= 2:
            pool.extend(_rows_of(led))
            splits += 1
        else:
            pool.append(led)

    good: dict[str, LedgerResult] = {}
    claims = _claim(pool, list(first.values()))
    claimed = set(claims.values())
    good.update((tid, pool[i]) for tid, i in claims.items())
    rest: list[LedgerResult] = []
    for i, led in enumerate(pool):
        if i in claimed:
            continue
        if len(led.rows) > 1:
            rest.extend(_rows_of(led))
            splits += 1
        else:
            rest.append(led)
    claims = _claim(rest, [r for r in first.values() if r.tool_use_id not in good])
    claimed = set(claims.values())
    good.update((tid, rest[i]) for tid, i in claims.items())

    mismatch = 0
    offsets: Counter = Counter()
    for i, led in enumerate(rest):
        if i in claimed:
            continue
        cands = near(led, (r for r in by_tool.get(led.tool, ()) if r.tool_use_id not in good))
        if not cands:
            continue
        mismatch += 1
        off = min((r.chars - led.text_out_chars for r in cands), key=abs)
        offsets[_offset_bucket(off)] += 1
    total = len(good) + len(rest) - len(claimed)

    def saved(led: LedgerResult) -> int:
        return led._sum("raw_tokens") - led._sum("out_tokens")

    saved_all = sum(saved(led) for led in ledger)
    saved_joined = sum(saved(led) for led in good.values())
    unjoined = [led for i, led in enumerate(rest) if i not in claimed]
    changed_joined = sum(1 for led in good.values() if led.changed)
    changed_all = changed_joined + sum(1 for led in unjoined if led.changed)
    counts = {
        "ledger_results": total,
        "joined": len(good),
        "size_mismatch": mismatch,
        "no_transcript": total - len(good) - mismatch,
        "groups_split": splits,
        # Results terse changed are the only ones with a delta to price, so their join
        # rate is what the gate tests. Rows terse passed through also come from callers
        # that leave no transcript (a corpus capture script, another client).
        "changed_results": changed_all,
        "changed_joined": changed_joined,
        # The same coverage weighted by what the ledger says each result saved on the
        # wire: a dropped 90-row result and a dropped 1-row one are not the same loss.
        "saved_tokens_joined_pct": round(100 * saved_joined / saved_all, 2) if saved_all else None,
        "mismatch_offsets": dict(sorted(offsets.items())),
    }
    return good, counts


def _offset_bucket(off: int) -> str:
    """Shown minus recorded chars for a dropped pairing, as a coarse label."""
    for top, label in ((-2000, "under -2000"), (-200, "-2000..-200"), (-41, "-200..-41"),
                       (40, "-40..40"), (349, "41..349"), (3200, "350..3200")):
        if off <= top:
            return label
    return "over 3200"


# ---------------------------------------------------------------- replay


def classify(res: Result, led: LedgerResult) -> tuple[str, int, int]:
    """(status, delta_tokens, kept_inline_tokens) for one joined result.

    delta is raw minus shown, cl100k. Statuses that carry no delta:
      offloaded      the model saw a file preview; the raw result was over a limit too.
      offload_only_out  only terse's output crossed a limit (the primer tipped it): no
                     saving is claimed for moving a result out of context by accident.
      kept_inline    terse kept in context a result whose raw form would have gone to a
                     file. The ledger's rule (#420) is to claim nothing either way;
                     `kept_inline_tokens` is the shown size, for the pessimistic bound in
                     which the model would never have opened the file.
      no_tokens      a row written without tiktoken, or an untokenizable result.
    """
    if not led.has_tokens or res.tokens is None:
        return "no_tokens", 0, 0
    # <=: terse often emits the text blocks and the typed field at the same size, and the
    # shown size then cannot tell them apart. The client shows the typed field when a
    # result has one (untouched `secret.list_credentials` results in the transcripts are
    # `structured_chars` long to the character), and the raw text blocks are the larger
    # side, so a tie read as text overstated the saving.
    typed = (led.typed_out_chars > 0
             and abs(res.chars - led.typed_out_chars) <= abs(res.chars - led.text_out_chars))
    raw_tokens, raw_chars = led.raw_side(typed)
    raw_offloads = raw_tokens > OFFLOAD_TOKENS or raw_chars > OFFLOAD_CHARS
    if res.offloaded:
        return ("offloaded" if raw_offloads else "offload_only_out"), 0, 0
    if raw_offloads:
        return "kept_inline", 0, res.tokens
    return "priced", raw_tokens - res.tokens, 0


def carry(calls: list[Call], start: int, stop: int) -> tuple[float, float | None, int]:
    """Weighted tokens (input-equivalent), USD, and the number of calls that carried one
    token placed in context just before `calls[start]`, up to but excluding `stop`.

    The first carrying call pays its tail rate. A later call pays the read rate, unless it
    rebuilt the cache (read under REBUILD_RATIO of the previous prompt), where it pays its
    tail rate again. Carrying ends early at a call whose prompt fell under DROP_RATIO of
    the previous one: the history, and this token with it, was dropped."""
    weighted = 0.0
    usd: float | None = 0.0
    n = 0
    for j in range(start, stop):
        call = calls[j]
        prev = calls[j - 1] if j > 0 else None
        if j > start and prev is not None and prev.prompt > 0:
            if call.prompt < DROP_RATIO * prev.prompt:
                break
            rebuilt = call.read < REBUILD_RATIO * prev.prompt
        else:
            rebuilt = True
        if not call.billed:
            continue
        rate = call.tail_weight() if rebuilt else call.read_weight()
        weighted += rate
        per = call.usd_per_weighted()
        usd = None if (usd is None or per is None) else usd + rate * per
        n += 1
    return weighted, usd, n


def replay(events: list[tuple], joined: dict[str, LedgerResult],
           ratio: dict[str | None, float] | None = None) -> dict:
    """Price every joined result and every retrieve in one transcript. `ratio` maps a
    model to billed tokens per cl100k token (key None is the fallback); without it deltas
    stay in cl100k."""
    ratio = ratio or {}
    calls: list[Call] = []
    marks: list[tuple] = []        # (Result, index of the next call)
    compacts: list[int] = []       # call index at which history restarts
    for ev in events:
        if ev[0] == "call":
            calls.append(ev[1])
        elif ev[0] == "result":
            marks.append((ev[1], len(calls)))
        else:
            compacts.append(len(calls))

    def stop_for(start: int) -> int:
        # >=: a compaction right after the result means no call ever carried it.
        later = [c for c in compacts if c >= start]
        return later[0] if later else len(calls)

    items = []
    net_weighted = 0.0
    net_usd: float | None = 0.0
    for pos, (res, start) in enumerate(marks):
        is_retrieve = norm_tool(res.name) == RETRIEVE_SUFFIX
        led = joined.get(res.tool_use_id)
        if led is None and not is_retrieve:
            continue
        if is_retrieve:
            status, delta, inline = ("retrieve", -(res.tokens or 0), 0)
        else:
            status, delta, inline = classify(res, led)
        per_w, per_usd, n = carry(calls, start, stop_for(start))
        model = calls[start].model if start < len(calls) else None
        scale = ratio.get(model, ratio.get(None, 1.0))
        weighted = delta * scale * per_w
        usd = None if per_usd is None else delta * scale * per_usd
        # What this result would have cost in context with terse off: the denominator
        # for "share of what an MCP result costs". Nothing for a result the raw path
        # would have saved to a file, and nothing for a retrieve, which it never makes.
        raw_weighted = (delta + res.tokens) * scale * per_w if status == "priced" else 0.0
        if is_retrieve and start < len(calls) and calls[start].billed:
            # The round trip is an API call the raw path never makes: it re-read the
            # whole cached prefix once.
            extra = calls[start]
            cost = extra.read * extra.read_weight()
            per = extra.usd_per_weighted()
            weighted -= cost
            usd = None if (usd is None or per is None) else usd - cost * per
        # Did the model ask the same tool the same thing again right away?
        recall = any(norm_tool(r.name) == norm_tool(res.name) and r.args == res.args
                     for r, s in marks[pos + 1:] if s - start <= RECALL_WINDOW)
        items.append({
            "tool": norm_tool(res.name), "status": status, "delta_tokens": delta,
            "kept_inline_tokens": inline, "carried_calls": n,
            "weighted": round(weighted, 2), "usd": None if usd is None else round(usd, 6),
            "model": model, "changed": bool(led and led.changed), "recall_3": recall,
            "shown_tokens": res.tokens,
            # The call that first carried this result was billed in another transcript:
            # this is copied history. Its weight here is what THIS session paid to keep
            # carrying it; the result itself is counted where it was first billed.
            "copy": start < len(calls) and not calls[start].billed,
            "kept_inline_weighted": round(inline * scale * per_w, 2),
            "raw_weighted": round(raw_weighted, 2),
        })
        net_weighted += weighted
        net_usd = None if (net_usd is None or usd is None) else net_usd + usd

    billed = [c for c in calls if c.billed]
    total_weighted = sum(c.weighted() for c in billed)
    total_usd: float | None = 0.0
    for c in billed:
        per = c.usd_per_weighted()
        total_usd = None if (total_usd is None or per is None) else total_usd + c.weighted() * per
    models = Counter(c.model for c in calls if c.model)
    return {
        "calls": len(calls),
        "billed_calls": len(billed),
        "model": models.most_common(1)[0][0] if models else None,
        "session_weighted": round(total_weighted, 1),
        "session_usd": None if total_usd is None else round(total_usd, 4),
        "net_weighted": round(net_weighted, 1),
        "net_usd": None if net_usd is None else round(net_usd, 4),
        "items": items,
    }


def calibration_samples(events: list[tuple]) -> list[tuple[str, float]]:
    """(model, billed tokens / cl100k tokens) for each large MCP result that was the only
    new content between two API calls. The ledger counts cl100k; Claude bills its own
    tokenizer. The previous call's output joins the next prompt, so it is subtracted."""
    out = []
    for i in range(1, len(events) - 1):
        ev = events[i]
        if ev[0] != "result" or (ev[1].tokens or 0) < CALIBRATION_MIN_TOKENS or ev[1].offloaded:
            continue
        before, after = events[i - 1], events[i + 1]
        if before[0] != "call" or after[0] != "call":
            continue
        grown = after[1].prompt - before[1].prompt - before[1].out
        if grown > 0 and after[1].model and after[1].billed and before[1].billed:
            out.append((after[1].model, grown / ev[1].tokens))
    return out


# ---------------------------------------------------------------- report


def tokenizer_ratios(samples: dict[str, list[float]]) -> dict[str | None, float]:
    """Median billed-per-cl100k ratio per model with enough samples; key None is the
    pooled median, used for every other model (1.0 with no samples at all)."""
    pooled = [v for vals in samples.values() for v in vals]
    out: dict[str | None, float] = {None: st.median(pooled) if pooled else 1.0}
    for model, vals in samples.items():
        if len(vals) >= CALIBRATION_MIN_SAMPLES:
            out[model] = st.median(vals)
    return out


def length_bucket(calls: int) -> str:
    for top, label in LENGTH_BUCKETS:
        if calls <= top:
            return label
    return LENGTH_BUCKETS[-1][1]


def summarize(sessions: list[dict], join_counts: dict,
              calib: dict[str, list[float]]) -> dict:
    def fold(rows, key):
        acc: dict[str, dict] = {}
        for row in rows:
            slot = acc.setdefault(key(row), {
                "results": 0, "delta_tokens": 0, "weighted": 0.0, "usd": 0.0,
                "usd_missing": 0, "raw_weighted": 0.0})
            if not row["copy"]:
                slot["results"] += 1
                slot["delta_tokens"] += row["delta_tokens"]
            slot["weighted"] += row["weighted"]
            slot["raw_weighted"] += row["raw_weighted"]
            if row["usd"] is None:
                slot["usd_missing"] += 1
            else:
                slot["usd"] += row["usd"]
        for slot in acc.values():
            slot["weighted"] = round(slot["weighted"], 1)
            slot["raw_weighted"] = round(slot["raw_weighted"], 1)
            slot["usd"] = round(slot["usd"], 4)
        return acc

    items = [dict(it, bucket=length_bucket(s["calls"])) for s in sessions for it in s["items"]]
    touched = [s for s in sessions if s["items"]]
    by_model_session: dict[str, dict] = {}
    for s in touched:
        slot = by_model_session.setdefault(s["model"] or "unknown", {
            "sessions": 0, "session_weighted": 0.0, "net_weighted": 0.0})
        slot["sessions"] += 1
        slot["session_weighted"] += s["session_weighted"]
        slot["net_weighted"] += s["net_weighted"]
    for slot in by_model_session.values():
        slot["net_share_pct"] = (round(100 * slot["net_weighted"] / slot["session_weighted"], 3)
                                 if slot["session_weighted"] else None)
        slot["session_weighted"] = round(slot["session_weighted"], 1)
        slot["net_weighted"] = round(slot["net_weighted"], 1)

    def recall_rate(rows):
        rows = list(rows)
        return {"n": len(rows),
                "recall_pct": round(100 * sum(r["recall_3"] for r in rows) / len(rows), 2)
                if rows else None}

    plain = [it for it in items if it["status"] != "retrieve" and not it["copy"]]
    all_weighted = sum(s["session_weighted"] for s in sessions)
    net = sum(s["net_weighted"] for s in sessions)
    mcp_raw = sum(it["raw_weighted"] for it in items)
    unpriced = sum(it["weighted"] for it in items if it["usd"] is None)
    # terse left these alone, so raw and shown should be the same size. A sum far from
    # zero means the two sides are not being measured alike.
    untouched = [it for it in plain if it["status"] == "priced" and not it["changed"]]
    total = join_counts["ledger_results"]
    changed_total = join_counts.get("changed_results")
    return {
        "join": dict(join_counts,
                     rate_pct=round(100 * join_counts["joined"] / total, 2) if total else None,
                     changed_rate_pct=round(100 * join_counts["changed_joined"] / changed_total, 2)
                     if changed_total else None),
        "transcripts": len(sessions), "transcripts_with_terse": len(touched),
        "net_weighted": round(net, 1),
        "net_usd_priced_models": round(sum(it["usd"] or 0 for it in items), 2),
        "items_without_usd_price": sum(1 for it in items if it["usd"] is None),
        "weighted_without_usd_price": round(unpriced, 1),
        "untouched_check": {
            "results": len(untouched),
            "delta_tokens": sum(it["delta_tokens"] for it in untouched),
            "shown_tokens": sum(it["shown_tokens"] or 0 for it in untouched)},
        "copied_results": sum(1 for it in items if it["copy"]),
        "ratios_applied": {str(k): round(v, 3) for k, v in tokenizer_ratios(calib).items()},
        "all_sessions_weighted": round(all_weighted, 1),
        "net_share_of_all_sessions_pct": round(100 * net / all_weighted, 4) if all_weighted else None,
        # The same net against what terse could reach: the cost the joined, priced
        # results (changed or not) would have had with terse off. Retrieves are in the
        # net. A result that did not join is in neither side; an untouched one would
        # only have added to the denominator, so read this beside the join rate.
        "mcp_raw_weighted": round(mcp_raw, 1),
        "net_share_of_mcp_results_pct": round(100 * net / mcp_raw, 2) if mcp_raw else None,
        "touched_sessions_weighted": round(sum(s["session_weighted"] for s in touched), 1),
        "kept_inline_pessimistic_weighted": round(sum(it["kept_inline_weighted"] for it in items), 1),
        "by_status": fold(items, lambda r: r["status"]),
        "by_model": fold(items, lambda r: r["model"] or "unknown"),
        "by_tool": fold(items, lambda r: r["tool"]),
        "by_length": fold(items, lambda r: r["bucket"]),
        "by_model_sessions": by_model_session,
        "recall": {"changed": recall_rate(r for r in plain if r["changed"]),
                   "unchanged": recall_rate(r for r in plain if not r["changed"])},
        "tokenizer_ratio": {m: {"n": len(v), "median": round(st.median(v), 3)}
                            for m, v in calib.items() if v},
    }


def render(summary: dict) -> str:
    lines = []
    j = summary["join"]
    lines.append(f"join: {j['joined']} of {j['ledger_results']} ledger results ({j['rate_pct']}%); "
                 f"left out: {j['size_mismatch']} size mismatch, {j['no_transcript']} with no "
                 f"transcript result; {j['groups_split']} groups split into rows")
    lines.append(f"of results terse changed: {j.get('changed_joined')} of "
                 f"{j.get('changed_results')} joined ({j['changed_rate_pct']}%), holding "
                 f"{j.get('saved_tokens_joined_pct')}% of the wire saving the ledger recorded")
    lines.append(f"size mismatches by (shown - recorded) chars: {j['mismatch_offsets']}")
    lines.append(f"results seen again in a resumed or forked transcript: "
                 f"{summary['copied_results']} (weight counted, result counted once)")
    lines.append(f"transcripts: {summary['transcripts']} scanned, "
                 f"{summary['transcripts_with_terse']} carried a terse result")
    lines.append(f"net effect: {summary['net_weighted']:,.0f} input-equivalent tokens saved "
                 f"(negative = terse cost more)")
    lines.append(f"usd: ${summary['net_usd_priced_models']:,.2f} on models with a known price; "
                 f"{summary['weighted_without_usd_price']:,.0f} of the net is on models without "
                 f"one ({summary['items_without_usd_price']} results) and is NOT in the usd figure")
    u = summary["untouched_check"]
    lines.append(f"check: {u['results']} results terse left alone sum to a delta of "
                 f"{u['delta_tokens']:,} tokens against {u['shown_tokens']:,} shown (expect ~0)")
    if summary["mcp_raw_weighted"]:
        lines.append(f"share of what the joined, priced MCP results would have cost: "
                     f"{summary['net_share_of_mcp_results_pct']}% of "
                     f"{summary['mcp_raw_weighted']:,.0f} (results that did not join, or "
                     f"that either path saved to a file, are in neither side)")
    lines.append(f"share of all scanned sessions' billed cost: "
                 f"{summary['net_share_of_all_sessions_pct']}%")
    lines.append(f"pessimistic bound for results kept inline: "
                 f"-{summary['kept_inline_pessimistic_weighted']:,.0f}")

    def table(title, rows, order=None):
        lines.append("")
        lines.append(f"{title:<34}{'results':>9}{'delta tok':>13}{'weighted':>15}{'usd':>11}"
                     f"{'raw path':>15}{'saved':>8}")
        keys = order or sorted(rows, key=lambda k: -rows[k]["weighted"])
        for key in keys:
            if key not in rows:
                continue
            r = rows[key]
            usd = f"{r['usd']:,.2f}" + ("*" if r["usd_missing"] else "")
            share = (f"{100 * r['weighted'] / r['raw_weighted']:.1f}%"
                     if r["raw_weighted"] else "-")
            lines.append(f"{str(key)[:33]:<34}{r['results']:>9}{r['delta_tokens']:>13,}"
                         f"{r['weighted']:>15,.0f}{usd:>11}{r['raw_weighted']:>15,.0f}"
                         f"{share:>8}")

    table("by status", summary["by_status"])
    table("by model", summary["by_model"])
    table("by session length", summary["by_length"], [b for _, b in LENGTH_BUCKETS])
    table("by tool", summary["by_tool"])
    lines.append("")
    lines.append("sessions that carried a terse result, by main model:")
    for model, r in sorted(summary["by_model_sessions"].items(),
                           key=lambda kv: -kv[1]["session_weighted"]):
        lines.append(f"  {model:<28} {r['sessions']:>5} sessions  net {r['net_weighted']:>14,.0f}"
                     f"  = {r['net_share_pct']}% of their cost")
    rc = summary["recall"]
    lines.append("")
    lines.append(f"same call repeated within {RECALL_WINDOW} calls: changed results "
                 f"{rc['changed']['recall_pct']}% of {rc['changed']['n']}, unchanged "
                 f"{rc['unchanged']['recall_pct']}% of {rc['unchanged']['n']}")
    lines.append("billed tokens per cl100k token (large single results): "
                 + ", ".join(f"{m} {v['median']} (n={v['n']})"
                             for m, v in summary["tokenizer_ratio"].items()))
    lines.append(f"ratio applied to deltas: {summary['ratios_applied']} (None = every other model)")
    lines.append("* some results on a model without a price; usd covers the priced ones only")
    return "\n".join(lines)


def find_transcripts(root: Path, since: float) -> list[Path]:
    paths = list(root.glob("*/*.jsonl")) + list(root.glob("*/*/subagents/*.jsonl"))
    return sorted(p for p in paths if p.stat().st_mtime >= since)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", type=Path,
                    default=Path.home() / ".local/state/terse/stats.jsonl")
    ap.add_argument("--transcripts", type=Path, default=Path.home() / ".claude/projects",
                    help="Claude Code projects directory")
    ap.add_argument("--out", type=Path, required=True, help="results directory")
    ap.add_argument("--min-join", type=float, default=90.0,
                    help="stop before pricing if under this percent of the results terse "
                         "changed, or of the saved tokens they hold, join a transcript")
    args = ap.parse_args(argv)

    ledger = load_ledger(args.ledger)
    if not ledger:
        print("no result rows in the ledger", file=sys.stderr)
        return 2
    paths = find_transcripts(args.transcripts, ledger[0].ts - 86400)
    print(f"{len(ledger)} ledger results, {len(paths)} transcripts", file=sys.stderr)

    # Oldest first, so a response is billed to the transcript that made it and a later
    # copy of that history is the one marked unbilled.
    paths.sort(key=lambda p: p.stat().st_mtime)
    scanned: list[tuple[Path, list[tuple]]] = []
    billed: set[tuple] = set()
    for n, path in enumerate(paths, 1):
        try:
            scanned.append((path, scan_transcript(path, billed)))
        except OSError as exc:
            print(f"skipped {path}: {exc}", file=sys.stderr)
        if n % 500 == 0:
            print(f"  scanned {n}/{len(paths)}", file=sys.stderr)

    results = [ev[1] for _, events in scanned for ev in events
               if ev[0] == "result" and ev[1].ts >= ledger[0].ts - JOIN_BEFORE]
    joined, counts = join(ledger, results)
    overall = 100 * counts["joined"] / counts["ledger_results"]
    changed = (100 * counts["changed_joined"] / counts["changed_results"]
               if counts["changed_results"] else 0.0)
    tokens = counts["saved_tokens_joined_pct"] or 0.0
    print(f"join: {counts}", file=sys.stderr)
    print(f"join rates: all results {overall:.1f}%, results terse changed {changed:.1f}%, "
          f"saved tokens {tokens:.1f}%", file=sys.stderr)
    if min(changed, tokens) < args.min_join:
        print(f"join rate under {args.min_join}%: stopping before pricing", file=sys.stderr)
        return 3

    calib: dict[str, list[float]] = defaultdict(list)
    for _, events in scanned:
        for model, sample in calibration_samples(events):
            calib[model].append(sample)
    ratio = tokenizer_ratios(calib)

    args.out.mkdir(parents=True, exist_ok=True)
    sessions = []
    with (args.out / "sessions.jsonl").open("w") as fh:
        for path, events in scanned:
            row = replay(events, joined, ratio)
            row["transcript"] = str(path)
            row["subagent"] = path.parent.name == "subagents"
            sessions.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["items"]:
                print(f"{path.stem[:36]:<36} {str(row['model'])[:22]:<22} calls {row['calls']:>4}"
                      f"  results {len(row['items']):>3}  net {row['net_weighted']:>12,.0f}",
                      file=sys.stderr)

    summary = summarize(sessions, counts, calib)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render(summary)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
