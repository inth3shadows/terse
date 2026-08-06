#!/usr/bin/env python3
"""A/B two Claude Code sessions on real billed tokens — measure terse WITHOUT trusting terse.

terse's own ledger (`terse stats`) cannot settle whether terse is a net win. Three
costs are structurally outside its accounting:

  1. It counts in cl100k (tiktoken), not Claude's tokenizer (see src/terse/tokenize.py).
  2. It never charges itself for the primer — up to 555 cl100k tokens with every gate on,
     248 for a default policy (src/terse/proxy.py) — plus one `terse_retrieve` tool
     definition per server. Since #211 a standalone wrapped server attaches that lazily,
     once per session; a ROUTER still rides `initialize.instructions` and carries it as
     fixed system-prompt weight on every request. (This harness predates #211 and its
     measured tables below are all of the pre-#211 eager architecture.)
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
Identical workload in every arm; model pinned to sonnet; same prompt. Every arm returned
the same answer with zero `terse.retrieve` round-trips, so the deltas are cost, not
behavior.

    wrapped servers                calls   RAW input   n   note
    1 (runecho only)                   4     -14.0%    1
    2 (runecho, kb)                    4      +1.9%    3   inside spread
    3 (codegraph, kb, runecho)         4      +2.1%    3   pre diff-flip
    3, after the #170 diff flip        4      +1.4%    3
    3, after the #170 diff flip       16      +0.8%    6   modal-turn filtered
    6, one proxy each                  4     +23.1%    1
    6, behind one multiproxy           4      +0.0%    4   inside spread

terse WINS at one wrapped server and LOSES from two upward, and MORE CALLS DID NOT SAVE
IT: 4 calls and 16 calls both land net-negative at three servers. Each standalone
`terse proxy` injects its own TERSE_PRIMER into that server's MCP `instructions`, and the
client re-read all of them every turn as cache_read, as it did then, so cost scaled with
(servers x turns). #170 cut the primer 402 -> 212 tokens and moved 3 servers from +2.1%
to +1.4% — real, but not enough to change the sign.

`multiproxy` collapses N primers to one and erased the six-server penalty (+23.1% ->
+0.0% RAW), which is the cleanest evidence that the primer, not the codec, is the
regression. See terse#168 for the amortization fix.

A caution on the 16-call row: turn counts there are BIMODAL (a ~3-turn batched path and a
~18-turn sequential one), not unimodal-with-outliers. The modal filter selects the larger
cluster, so it is choosing a behavioral mode rather than discarding noise. Both clusters
agree on the sign (+0.4% at 3 turns, +0.8% at 18), which is why the conclusion stands.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
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


# --------------------------------------------------------------------------- arms

# A run whose assistant-turn count deviates from the modal value across both arms is
# DROPPED, not reported. `cache_read` scales with turns and dominates RAW input, so a
# single extra round-trip moves the total by more than the entire treatment effect.
# Measured: a 16-call pair where one run per arm took 18 turns instead of 3 reported
# RAW -3.0%, while the modal-turn runs showed +0.4% — the aggregate was reporting
# round-trip luck, not compression.
#
# This is a deliberate reversal. An earlier revision argued a turn delta was a measured
# EFFECT of the treatment (fewer round-trips to the same answer) as long as MCP call
# counts matched, and so refused to flag it. That holds for a 6-vs-7 difference; it does
# not survive 18-vs-3, where the variance swamps the signal it was supposed to preserve.


class Arm:
    """One side of the comparison: N runs of the same task under the same condition."""

    def __init__(self, paths: list[Path], label: str):
        self.label = label
        # The same transcript passed twice is not two replicates. It would report sd=0
        # from non-independent data and manufacture confidence out of n=1 — the exact
        # failure this outlier control exists to prevent. Drop repeats, keep order.
        seen: set[Path] = set()
        uniq: list[Path] = []
        for p in paths:
            rp = p.resolve()
            if rp in seen:
                print(f"  ! {label}: ignoring duplicate run {p.name[:8]}", file=sys.stderr)
                continue
            seen.add(rp)
            uniq.append(p)
        self.runs = [SessionStats(p) for p in uniq]
        self.kept: list[SessionStats] = list(self.runs)
        self.dropped: list[SessionStats] = []

    def restrict_to_turns(self, turns: int) -> None:
        self.kept = [r for r in self.runs if r.turns == turns]
        self.dropped = [r for r in self.runs if r.turns != turns]

    def stat(self, field: str) -> tuple[float, float]:
        """(mean, sample stdev) of `field` over the kept runs. sd is 0.0 for n<2 —
        reported as such rather than hidden, since n=1 has no spread to speak of."""
        vals = [getattr(r, field) for r in self.kept]
        if not vals:
            return (0.0, 0.0)
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

    def union(self, field: str) -> set:
        out: set = set()
        for r in self.kept or self.runs:
            out |= set(getattr(r, field))
        return out


def modal_turns(a: Arm, b: Arm) -> int:
    """The most common turn count across BOTH arms, so the filter is applied
    symmetrically and cannot be tuned to favor one side. Ties resolve to the lower
    count (statistics.mode is order-dependent on ties; sorting makes it deterministic)."""
    counts = Counter(r.turns for r in a.runs + b.runs)
    top = max(counts.values())
    return min(t for t, c in counts.items() if c == top)


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _row(label: str, a: Arm, b: Arm, field: str) -> tuple[str, bool]:
    """One metric line, plus whether the delta clears the noise floor."""
    ma, sa = a.stat(field)
    mb, sb = b.stat(field)
    d = mb - ma
    # A zero control mean has no percentage — printing +0.0% next to a large absolute
    # delta reads as "no change", the opposite of the truth. Say n/a.
    pct = f"{d / ma * 100:>+7.1f}%" if ma else f"{'n/a':>8}"
    pooled = (sa + sb) / 2
    # A delta smaller than twice the pooled spread is not distinguishable from run-to-run
    # variance at these sample sizes. Crude on purpose: with n<=6 a real significance test
    # would imply a precision this harness does not have.
    #
    # BOTH arms need n>=2 or there is no spread to compare against. A one-run arm reports
    # sd=0, which halves `pooled` and makes almost any delta look like SIGNAL — the exact
    # false confidence this outlier control exists to remove. Say "n<2" instead.
    #
    # pooled == 0 with n>=2 in both arms means every replicate landed on the same number:
    # zero observed spread is the STRONGEST evidence a nonzero delta is real, not a reason
    # to withhold judgement. Token counts here are deterministic given the same turns, so
    # this is the common case for a clean pair, and `abs(d) > 0` is the right test.
    enough = len(a.kept) > 1 and len(b.kept) > 1
    clears = enough and abs(d) > 2 * pooled
    flag = "  n<2" if not enough else ("  SIGNAL" if clears else "  noise")
    return (f"  {label:<18} {ma:>11,.0f} +/-{sa:>9,.0f} {mb:>11,.0f} +/-{sb:>9,.0f} "
            f"{d:>+11,.0f} {pct}{flag}", clears)


def _skew_warnings(a: Arm, b: Arm) -> list[str]:
    warns = []
    for name, field in (("model", "models"), ("cli version", "versions"),
                        ("git branch", "branches")):
        va, vb = a.union(field), b.union(field)
        if va and vb and va != vb:
            warns.append(f"{name} differs: A={sorted(va)} B={sorted(vb)}")
    mcp_a = {r.total_mcp for r in a.kept}
    mcp_b = {r.total_mcp for r in b.kept}
    if mcp_a and mcp_b and mcp_a != mcp_b:
        warns.append(
            f"MCP call count differs: A={sorted(mcp_a)} B={sorted(mcp_b)} — the runs did "
            f"not do the same work, so the token delta is not attributable to terse")
    return warns


def report(a: Arm, b: Arm, *, drop_outliers: bool = True) -> int:
    turns = modal_turns(a, b)
    if drop_outliers:
        a.restrict_to_turns(turns)
        b.restrict_to_turns(turns)

    print(f"\nA ({a.label}): {len(a.kept)}/{len(a.runs)} runs")
    print(f"B ({b.label}): {len(b.kept)}/{len(b.runs)} runs")
    if drop_outliers:
        print(f"modal assistant turns: {turns}"
              + (f"  (dropped {len(a.dropped) + len(b.dropped)} outlier run(s))"
                 if a.dropped or b.dropped else ""))
        for arm in (a, b):
            for r in arm.dropped:
                print(f"  ! dropped {arm.label} {r.path.name[:8]}: {r.turns} turns "
                      f"(RAW {r.raw_input:,})")
    if not a.kept or not b.kept:
        print("\n  NO COMPARABLE RUNS — every run in one arm deviated from the modal "
              "turn count. Re-run, or pass --keep-outliers to see the raw spread.\n")
        return 2

    print(f"\n  {'metric':<18} {'A control':>23} {'B terse':>23} {'delta':>11} {'pct':>8}")
    print(f"  {'-'*18} {'-'*23} {'-'*23} {'-'*11} {'-'*8}")
    for label, fld in (("cache write", "cache_write"), ("cache read", "cache_read"),
                       ("output", "output")):
        line, _ = _row(label, a, b, fld)
        print(line)
    print(f"  {'-'*18} {'-'*23} {'-'*23} {'-'*11} {'-'*8}")
    raw_line, raw_clears = _row("RAW input", a, b, "raw_input")
    w_line, w_clears = _row("WEIGHTED", a, b, "weighted")
    print(raw_line)
    print(w_line)
    print()
    print(f"  MCP calls          {a.kept[0].total_mcp:>11} {b.kept[0].total_mcp:>23}")
    print(f"  terse.retrieve     {a.kept[0].retrieve_calls:>11} "
          f"{b.kept[0].retrieve_calls:>23}")

    warns = _skew_warnings(a, b)
    if warns:
        print("\n  UNCONTROLLED SKEW — the delta above is not clean:")
        for w in warns:
            print(f"    ! {w}")

    mw, _ = b.stat("weighted")
    ma, _ = a.stat("weighted")
    d = mw - ma
    if len(a.kept) < 2 or len(b.kept) < 2:
        print(f"\n  verdict (weighted): INSUFFICIENT DATA — delta {d:+,.0f}, but an arm "
              f"has fewer than 2 surviving runs so there is no spread to judge it "
              f"against\n")
        return 2
    if not w_clears:
        print(f"\n  verdict (weighted): INCONCLUSIVE — delta {d:+,.0f} is inside the "
              f"run-to-run spread\n")
    else:
        print(f"\n  verdict (weighted): {'TERSE WINS' if d < 0 else 'TERSE LOSES'} "
              f"by {abs(d):,.0f} weighted tokens\n")
    return 2 if warns else 0


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
    ap.add_argument("--a", required=True, nargs="+", metavar="PATH",
                    help="control transcript(s) (terse NOT installed). Repeatable — pass "
                         "every replicate; a .jsonl file, or a session dir whose newest "
                         ".jsonl is used")
    ap.add_argument("--b", required=True, nargs="+", metavar="PATH",
                    help="treatment transcript(s) (terse installed)")
    ap.add_argument("--keep-outliers", action="store_true",
                    help="do NOT drop runs whose turn count deviates from the modal "
                         "value. Shows the raw spread; the aggregate is then dominated "
                         "by round-trip luck rather than by compression")
    ap.add_argument("--json", action="store_true",
                    help="emit the per-run numbers as JSON instead of the table")
    args = ap.parse_args(argv)

    a = Arm([_resolve(p) for p in args.a], "no-terse")
    b = Arm([_resolve(p) for p in args.b], "terse")

    if args.json:
        turns = modal_turns(a, b)
        if not args.keep_outliers:
            a.restrict_to_turns(turns)
            b.restrict_to_turns(turns)

        def dump(arm: Arm) -> dict:
            return {"label": arm.label, "modal_turns": turns,
                    "kept": [{"path": str(r.path), "turns": r.turns,
                              "raw_input": r.raw_input, "weighted": round(r.weighted, 1),
                              "mcp_calls": r.total_mcp} for r in arm.kept],
                    "dropped": [{"path": str(r.path), "turns": r.turns} for r in arm.dropped],
                    "mean_raw_input": arm.stat("raw_input")[0],
                    "sd_raw_input": arm.stat("raw_input")[1],
                    "mean_weighted": arm.stat("weighted")[0],
                    "sd_weighted": arm.stat("weighted")[1]}
        print(json.dumps({"a": dump(a), "b": dump(b), "skew": _skew_warnings(a, b)},
                         indent=2))
        return 0

    return report(a, b, drop_outliers=not args.keep_outliers)


if __name__ == "__main__":
    raise SystemExit(main())
