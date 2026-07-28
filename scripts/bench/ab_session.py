#!/usr/bin/env python3
"""A/B two Claude Code sessions on real billed tokens — measure terse WITHOUT trusting terse.

terse's own ledger (`terse stats`) cannot settle whether terse is a net win. Three
costs are structurally outside its accounting:

  1. It counts in cl100k (tiktoken), not Claude's tokenizer (see src/terse/tokenize.py).
  2. It never charges itself for TERSE_PRIMER — 402 cl100k tokens prepended to each
     wrapped server's `instructions` (src/terse/proxy.py) — plus one `terse_retrieve`
     tool definition per server. That is fixed system-prompt weight on EVERY request.
  3. It cannot see rebound: `terse.retrieve` round-trips, or the model re-reading a
     file because a dropped field mattered.

This script reads none of terse's numbers. It reads the `usage` block the Anthropic API
returned, as recorded in the Claude Code transcript, and diffs two sessions.

Protocol
--------
  terse uninstall-mcp && <fresh session> && <run the fixed task>   -> transcript A
  terse install-mcp   && <fresh session> && <run the same task>    -> transcript B
  python scripts/bench/ab_session.py --a <A.jsonl> --b <B.jsonl>

Both runs must use the same model, the same prompt, and the same repo state. The
comparison is only as good as that control; the script reports version/model/branch
skew it can detect, but it cannot detect a differently-worded prompt.

Weighted tokens
---------------
Raw token sums overstate terse's cost. The primer is stable and lands in the cache;
tool results mostly do not. Anthropic prices cache writes at 1.25x and cache reads at
0.1x of base input. `weighted` applies those multipliers so the comparison reflects
spend rather than raw count. Both are reported — raw is the honest context-pressure
number, weighted is the honest money number.

First results (2026-07-28)
--------------------------
Identical workload in every arm: four `runecho structure` calls on the same four files,
model pinned to sonnet, same prompt. Every arm returned the same answer (377) with zero
`terse.retrieve` round-trips, so the deltas are cost, not behavior.

    wrapped servers            RAW input      WEIGHTED    n
    1 (runecho only)             -14.0%         -9.2%     1
    3 (codegraph, kb, runecho)    +2.1%         +4.9%     3   <- signal, spread +/-332
    6, one proxy each            +23.1%        +17.4%     1
    6, behind one multiproxy      +0.0%         +4.5%     4   <- within spread

terse WINS at one wrapped server and LOSES at three. The variable is not server count per
se: each standalone `terse proxy` injects its own copy of TERSE_PRIMER (402 cl100k tokens)
into that server's MCP `instructions`, and the client re-reads all of them every turn as
cache_read. Cost therefore scales with (servers x turns) while savings scale with
(compressible tool calls). At four calls, three primers are already too many to amortize.

`multiproxy` collapses N primers to one and erased the six-server penalty (+23.1% ->
+0.0% RAW), which is the cleanest evidence that the primer, not the codec, is the
regression. See terse#168 for the amortization fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Anthropic input-token price multipliers relative to base input.
W_INPUT = 1.0
W_CACHE_WRITE = 1.25
W_CACHE_READ = 0.1
W_OUTPUT = 5.0  # output is ~5x input on every current Claude model


class SessionStats:
    def __init__(self, path: Path):
        self.path = path
        self.input = 0
        self.cache_write = 0
        self.cache_read = 0
        self.output = 0
        self.turns = 0
        self.mcp_calls = Counter()
        self.tool_calls = Counter()
        self.retrieve_calls = 0
        self.models: Counter[str] = Counter()
        self.versions: Counter[str] = Counter()
        self.branches: Counter[str] = Counter()
        self.sidechain_turns = 0
        self._seen: set[tuple] = set()
        self._seen_tools: set[str] = set()
        self._parse()

    def _parse(self) -> None:
        with self.path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[warn] {self.path.name}:{lineno} unparseable, skipped",
                          file=sys.stderr)
                    continue
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message") or {}

                # Tool calls are counted BEFORE the usage dedup below, and deduped on
                # their own block id. Claude Code splits one API response across several
                # transcript records (a text record, then a tool_use record) that share
                # requestId AND message.id while each repeating the full `usage` block.
                # So usage must be deduped per response, but tool_use blocks live in the
                # records that dedup would discard — counting them there undercounts.
                for block in msg.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    bid = block.get("id")
                    if bid is not None:
                        if bid in self._seen_tools:
                            continue
                        self._seen_tools.add(bid)
                    name = block.get("name", "?")
                    self.tool_calls[name] += 1
                    if name.startswith("mcp__"):
                        self.mcp_calls[name] += 1
                    if "terse_retrieve" in name or "terse.retrieve" in name:
                        self.retrieve_calls += 1

                # Bill each API response exactly once.
                key = (rec.get("requestId"), msg.get("id"))
                if key in self._seen:
                    continue
                self._seen.add(key)

                if v := rec.get("version"):
                    self.versions[v] += 1
                if b := rec.get("gitBranch"):
                    self.branches[b] += 1
                if m := msg.get("model"):
                    self.models[m] += 1
                if rec.get("isSidechain"):
                    self.sidechain_turns += 1

                # Top-level usage already aggregates `iterations`; do NOT also sum those.
                usage = msg.get("usage") or {}
                self.input += usage.get("input_tokens", 0)
                self.cache_write += usage.get("cache_creation_input_tokens", 0)
                self.cache_read += usage.get("cache_read_input_tokens", 0)
                self.output += usage.get("output_tokens", 0)
                self.turns += 1

    @property
    def raw_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    @property
    def weighted(self) -> float:
        return (self.input * W_INPUT
                + self.cache_write * W_CACHE_WRITE
                + self.cache_read * W_CACHE_READ
                + self.output * W_OUTPUT)

    @property
    def total_mcp(self) -> int:
        return sum(self.mcp_calls.values())


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _delta_line(label: str, a: float, b: float, unit: str = "") -> str:
    d = b - a
    pct = (d / a * 100) if a else float("nan")
    sign = "+" if d > 0 else ""
    verdict = "worse" if d > 0 else ("better" if d < 0 else "same")
    pct_s = "   n/a" if a == 0 else f"{sign}{pct:6.1f}%"
    return (f"  {label:<22} {_fmt(a):>12} {_fmt(b):>12} "
            f"{sign}{_fmt(d):>12}{unit} {pct_s}  {verdict}")


def _skew_warnings(a: SessionStats, b: SessionStats) -> list[str]:
    warns = []
    for name, va, vb in (("model", set(a.models), set(b.models)),
                         ("cli version", set(a.versions), set(b.versions)),
                         ("git branch", set(a.branches), set(b.branches))):
        if va and vb and va != vb:
            warns.append(f"{name} differs: A={sorted(va)} B={sorted(vb)}")
    if a.total_mcp != b.total_mcp:
        warns.append(
            f"MCP call count differs: A={a.total_mcp} B={b.total_mcp} — the two runs did "
            f"not do the same work, so the token delta is not attributable to terse")
    # Turn count is deliberately NOT a skew warning. With MCP call counts equal, a turn
    # delta is a measured effect of the treatment (fewer/more round-trips to reach the
    # same answer), not an uncontrolled variable. It only invalidates the comparison when
    # the call counts already disagree, which is caught above.
    return warns


def report(a: SessionStats, b: SessionStats, *, label_a: str, label_b: str) -> int:
    print(f"\nA (control): {a.path}")
    print(f"B (terse)  : {b.path}\n")
    print(f"  {'metric':<22} {label_a:>12} {label_b:>12} {'delta':>13} {'pct':>7}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 13} {'-' * 7}")
    print(_delta_line("input (uncached)", a.input, b.input))
    print(_delta_line("cache write", a.cache_write, b.cache_write))
    print(_delta_line("cache read", a.cache_read, b.cache_read))
    print(_delta_line("output", a.output, b.output))
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 13} {'-' * 7}")
    print(_delta_line("RAW input total", a.raw_input, b.raw_input))
    print(_delta_line("WEIGHTED (spend)", a.weighted, b.weighted))
    print()
    print(f"  assistant turns        {a.turns:>12} {b.turns:>12}")
    print(f"  MCP tool calls         {a.total_mcp:>12} {b.total_mcp:>12}")
    print(f"  terse.retrieve calls   {a.retrieve_calls:>12} {b.retrieve_calls:>12}")
    warns = _skew_warnings(a, b)
    # Per-call amortization is only meaningful when both runs did the same work. On a
    # skewed pair it produces a large, confident, meaningless number — so don't print it.
    if b.total_mcp and not warns:
        per_call = (b.weighted - a.weighted) / b.total_mcp
        print(f"\n  weighted delta per MCP call: {per_call:+,.0f}")
        if per_call < 0:
            # Fixed primer overhead is paid once; savings accrue per call.
            print("  (negative = terse pays for itself at this call volume)")

    if warns:
        print("\n  UNCONTROLLED SKEW — the delta below is not clean:")
        for w in warns:
            print(f"    ! {w}")

    d = b.weighted - a.weighted
    if d == 0:
        print("\n  verdict (weighted): NO DIFFERENCE\n")
    else:
        verdict = "TERSE WINS" if d < 0 else "TERSE LOSES"
        print(f"\n  verdict (weighted): {verdict} by {abs(d):,.0f} weighted tokens\n")
    return 0 if not warns else 2


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_dir():
        jsonls = sorted(path.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
        if not jsonls:
            sys.exit(f"no *.jsonl transcripts in {path}")
        return jsonls[-1]
    if not path.exists():
        sys.exit(f"no such transcript: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, metavar="PATH",
                    help="control transcript (terse NOT installed): a .jsonl file, or a "
                         "session dir whose newest .jsonl is used")
    ap.add_argument("--b", required=True, metavar="PATH",
                    help="treatment transcript (terse installed)")
    ap.add_argument("--json", action="store_true",
                    help="emit the raw numbers as JSON instead of the table")
    args = ap.parse_args(argv)

    a = SessionStats(_resolve(args.a))
    b = SessionStats(_resolve(args.b))

    if args.json:
        def dump(s: SessionStats) -> dict:
            return {"path": str(s.path), "input": s.input, "cache_write": s.cache_write,
                    "cache_read": s.cache_read, "output": s.output,
                    "raw_input": s.raw_input, "weighted": round(s.weighted, 1),
                    "turns": s.turns, "mcp_calls": s.total_mcp,
                    "retrieve_calls": s.retrieve_calls,
                    "models": sorted(s.models), "versions": sorted(s.versions)}
        print(json.dumps({"a": dump(a), "b": dump(b),
                          "delta_raw_input": b.raw_input - a.raw_input,
                          "delta_weighted": round(b.weighted - a.weighted, 1),
                          "skew": _skew_warnings(a, b)}, indent=2))
        return 0

    return report(a, b, label_a="A no-terse", label_b="B terse")


if __name__ == "__main__":
    raise SystemExit(main())
