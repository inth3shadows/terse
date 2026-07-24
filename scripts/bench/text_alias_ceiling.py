#!/usr/bin/env python3
"""Ceiling measurement for #137's second lever: lossless repeated-span aliasing over text.

#137 proposes a "text dictionary" — the long-text analogue of terse's `dictionary` tier,
which folds repeated JSON VALUES behind an inline legend. The premise is that
`codegraph_explore`'s markdown scaffolding repeats enough to pay for a legend. This script
exists to test that premise before anything is built, and to keep the answer re-derivable:
a future session proposing a text dictionary should meet a measurement, not a verdict.

Nothing here is modelled. Every reported percentage is produced by actually building the
encoded payload — legend included, in the format the primer already teaches — and running
cl100k over it. The round-trip is re-checked for every payload at measurement time and a
failure RAISES, so a lossy encoding can never be reported as a saving.

Three encoders, weakest to strongest:

    line    repeated identical LINES
    phrase  repeated word n-grams (1..32 words) — the direct analogue of `dictionary`
    both    repeated multi-line RUNS first, then phrases over the residual

Selection is single-pass greedy: enumerate every repeated n-gram, rank by a character
heuristic, then walk that ranking accepting any candidate whose EXACT tiktoken saving is
positive and whose occurrences are still unclaimed.

Run:
    uv run scripts/bench/text_alias_ceiling.py --corpus ~/.config/terse/session-corpus
    uv run scripts/bench/text_alias_ceiling.py --files ~/src --sample 300
    uv run scripts/bench/text_alias_ceiling.py --corpus <dir> --policy ~/.config/terse/policy.json

`--corpus` takes a `terse capture` envelope directory and reports per shape, including the
marginal column that actually decides the question: what an aliaser finds in terse's own
output, after the shipped lossless tiers have already run. That baseline goes through
`policy.apply` — the same entry point the proxy uses — so passthrough rules, the marker
guard, the depth guard and the codec's minify fallback are the shipped behavior rather than
a re-implementation. Without `--policy` it runs the DEFAULT policy (all three lossless tiers
on every tool), which is the strongest lossless baseline; with `--policy` it runs a real
deployment's rules.

`--files` takes a source tree and reports the distribution over real files — the stand-in
for the `read_text_file` class, which no captured corpus here contains.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from terse import capture
from terse import policy as policy_mod
from terse.tokenize import count_cl100k as TOK

UNIT_RE = re.compile(r"\s+|\S+")
DICT_MARKER = "__terse_textdict__"

# Alias overhead charged per legend entry: the quoting, separator and colon a JSON legend
# costs on top of the alias and the value themselves. Deliberately small — the point is a
# CEILING, so every judgement call here is made in the idea's favour.
LEGEND_ENTRY_OVERHEAD = 4

# Candidate alias sigils, cheapest-first at measurement time. `~` matches the JSON
# dictionary tier and the primer, so it is tried first — but it CANNOT be assumed: terse's
# own `dictionary` mints `~0, ~1, ...` from the same namespace (transforms.ALIAS_SIGIL), so
# on any payload that tier has touched, `~`-aliases would collide with real content. The
# encoder picks the first sigil absent from the payload, exactly as `transforms._alias_gen`
# avoids literals already present. Getting this wrong is not a rounding error: scoring a
# collision as 0.0% silently zeroed 84 of 626 live payloads, all of them the one tool #137
# is about, and biased the headline toward "no headroom".
SIGILS = ("~", "¤", "§", "†", "‡", "¶", "‰", "±", "¬")

SOURCE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".md", ".rs", ".java", ".sh",
              ".sql", ".yaml", ".yml", ".c", ".h", ".cpp", ".rb"}
SOURCE_SKIP = {"node_modules", ".git", ".venv", "dist", "build", "__pycache__", "vendor",
               ".codegraph", "site-packages", ".cache", ".bare", "target", ".mypy_cache"}
MIN_FILE_BYTES, MAX_FILE_BYTES = 2_000, 400_000


class NotLossless(AssertionError):
    """An encoder produced output that does not restore to its input.

    A real exception rather than a bare `assert`, because `assert` vanishes under
    `python -O` — and this check is the single thing standing between a measurement and a
    fabricated one.
    """


# Counters for anything that silently bounds a reported number. Printed when non-zero: a
# cap that binds without saying so is indistinguishable from a genuine ceiling.
LIMITS: Counter = Counter()


# ---------------------------------------------------------------- encoders


def pick_sigil(text: str) -> str:
    """The cheapest sigil the payload does not already contain.

    Every alias starts with the sigil, so absence of the sigil is absence of collisions —
    one check for the whole legend, and no payload has to be scored 0 for containing one.
    """
    free = [s for s in SIGILS if s not in text]
    if not free:
        LIMITS["no_free_sigil"] += 1
        i = 0
        while f"«{i}»" in text:
            i += 1
        return f"«{i}»"
    return min(free, key=lambda s: _tok(s + "0"))


def _tok(s: str) -> int:
    n = TOK(s)
    if n is None:  # tiktoken absent — every number here is a token count, so refuse
        raise SystemExit("tiktoken is required for this measurement; pip install tiktoken")
    return n


def legend_cost(legend: dict[str, str]) -> int:
    """Tokens the legend block itself costs. Counted against every reported saving."""
    if not legend:
        return 0
    return _tok(json.dumps({DICT_MARKER: 1, "legend": legend}, ensure_ascii=False) + "\n")


def _alias_ngrams(units: list[str], anchors: list[int], lengths: tuple[int, ...],
                  legend: dict[str, str], sigil: str, budget: int, max_aliases: int,
                  min_chars: int = 6,
                  blocked: set[int] | None = None) -> tuple[list[str], dict[str, str]]:
    """Greedy n-gram aliasing over a unit stream. Returns (new units, legend).

    `anchors` are the unit indices a span may start or end on, which is what keeps a
    phrase span from ending on whitespace and a block span from covering half a line.
    Lossless because ''.join(units) reproduces the input exactly and every replacement
    covers a whole unit range.

    `budget` caps how many candidates get exact (tiktoken) scoring. The ranking is by
    CHARACTERS and the accept test is by TOKENS, so the two disagree and a payable
    candidate can in principle sit past the cut — hence `LIMITS`, which counts every time
    a cap actually binds instead of asserting it never matters.
    """
    cands: list[tuple[int, str, list[tuple[int, int]]]] = []
    for nw in lengths:
        if nw > len(anchors):
            break
        occ: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for k in range(len(anchors) - nw + 1):
            s, e = anchors[k], anchors[k + nw - 1]
            occ["".join(units[s:e + 1])].append((s, e))
        for txt, spans in occ.items():
            if len(spans) < 2 or len(txt) < min_chars:
                continue
            cands.append(((len(spans) - 1) * len(txt), txt, spans))
    cands.sort(key=lambda c: -c[0])
    if len(cands) > budget:
        LIMITS["budget_bound"] += 1

    # An already-emitted alias unit is BLOCKED, not merely a non-anchor: a span that
    # merely contains one would nest aliases inside a legend value, and nested aliases
    # make decoding order-dependent. That is a real defect this guard exists to prevent —
    # it silently corrupted 10 of 296 files before it was added.
    claimed = bytearray(len(units))
    for i in blocked or ():
        claimed[i] = 1

    repl: list[tuple[int, int, str]] = []
    for i, (_, txt, spans) in enumerate(cands):
        if i >= budget:
            break
        if len(legend) >= max_aliases:
            LIMITS["max_aliases_bound"] += 1
            break
        free: list[tuple[int, int]] = []
        last = -1
        for s, e in spans:
            if s > last and not any(claimed[s:e + 1]):
                free.append((s, e))
                last = e
        if len(free) < 2:
            continue
        n, t = len(free), _tok(txt)
        a = f"{sigil}{len(legend)}"
        at = _tok(a)
        if (n * t) - (n * at) - (t + at + LEGEND_ENTRY_OVERHEAD) <= 0:
            continue
        legend[a] = txt
        for s, e in free:
            claimed[s:e + 1] = b"\x01" * (e - s + 1)
            repl.append((s, e, a))

    if not repl:
        return units, legend
    repl.sort()
    out: list[str] = []
    prev = 0
    for s, e, a in repl:
        out.extend(units[prev:s])
        out.append(a)
        prev = e + 1
    out.extend(units[prev:])
    return out, legend


def encode_lines(text: str, sigil: str | None = None) -> tuple[str, dict[str, str]]:
    """Alias repeated identical lines — what a naive text dictionary would do."""
    sigil = sigil or pick_sigil(text)
    lines = text.split("\n")
    counts = Counter(lines)
    legend: dict[str, str] = {}
    sub: dict[str, str] = {}
    for line, n in sorted(((line, n) for line, n in counts.items() if n >= 2 and line.strip()),
                          key=lambda x: -(len(x[0]) * x[1])):
        t = _tok(line)
        a = f"{sigil}{len(legend)}"
        at = _tok(a)
        if (n * t) - (n * at) - (t + at + LEGEND_ENTRY_OVERHEAD) <= 0:
            continue
        legend[a] = line
        sub[line] = a
    if not legend:
        return text, {}
    return "\n".join(sub.get(line, line) for line in lines), legend


def encode_phrases(text: str, sigil: str | None = None, max_aliases: int = 300,
                   budget: int = 4000) -> tuple[str, dict[str, str]]:
    """Alias repeated word n-grams — the direct analogue of the JSON `dictionary` tier."""
    sigil = sigil or pick_sigil(text)
    units = UNIT_RE.findall(text)
    if "".join(units) != text:  # pragma: no cover — defensive
        raise NotLossless("unit split is not lossless")
    anchors = [i for i, u in enumerate(units) if u.strip()]
    out, legend = _alias_ngrams(units, anchors, (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
                                {}, sigil, budget, max_aliases)
    return ("".join(out), legend) if legend else (text, {})


def encode_blocks_then_phrases(text: str, sigil: str | None = None, max_aliases: int = 300,
                               budget: int = 4000) -> tuple[str, dict[str, str]]:
    """Repeated multi-LINE runs first, then word n-grams over what is left.

    Blocks first because one long repeated run is worth far more per legend entry than the
    short phrases inside it, and a purely phrase-ranked pass fragments those runs.
    """
    sigil = sigil or pick_sigil(text)
    lines = text.split("\n")
    units: list[str] = []
    for i, line in enumerate(lines):
        units.append(line)
        if i < len(lines) - 1:
            units.append("\n")
    if "".join(units) != text:  # pragma: no cover — defensive
        raise NotLossless("line split is not lossless")

    legend: dict[str, str] = {}
    units, legend = _alias_ngrams(units, list(range(0, len(units), 2)),
                                  (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
                                  legend, sigil, budget, max_aliases)
    body = "".join(units)

    u2 = UNIT_RE.findall(body)
    blocked = {i for i, u in enumerate(u2) if u in legend}
    anchors = [i for i, u in enumerate(u2) if u.strip() and i not in blocked]
    u2, legend = _alias_ngrams(u2, anchors, (1, 2, 3, 4, 6, 8, 12, 16, 24),
                               legend, sigil, budget, max_aliases, blocked=blocked)
    body = "".join(u2)
    return (body, legend) if legend else (text, {})


ENCODERS = (("line", encode_lines), ("phrase", encode_phrases),
            ("both", encode_blocks_then_phrases))


def decode(body: str, legend: dict[str, str]) -> str:
    """Longest alias first, so `~1` can never eat the `~1` inside `~10`."""
    for a in sorted(legend, key=len, reverse=True):
        body = body.replace(a, legend[a])
    return body


def score(text: str, encoder: str) -> float:
    """Percent of cl100k tokens saved by `encoder`, legend included. 0.0 when it declines.

    Raises `NotLossless` if the encoding does not round-trip — a lossy encoder must not be
    able to report a saving, and that has to hold under `python -O` too.
    """
    fn = dict(ENCODERS)[encoder]
    raw_t = _tok(text)
    if not raw_t:
        return 0.0
    body, legend = fn(text)
    if not legend:
        return 0.0
    if decode(body, legend) != text:
        raise NotLossless(f"{encoder} encoding does not restore to its input")
    return 100.0 * (raw_t - (legend_cost(legend) + _tok(body))) / raw_t


def best(text: str) -> tuple[float, str]:
    """Best of the three encoders, and which one won."""
    top, how = 0.0, "-"
    for name, _ in ENCODERS:
        pct = score(text, name)
        if pct > top:
            top, how = pct, name
    return top, how


# ---------------------------------------------------------------- baseline


def make_baseline(pol: policy_mod.Policy):
    """`raw -> what terse emits for it` under `pol`, via the proxy's own entry point.

    `policy.apply` rather than `transforms.compress`: the passthrough rule (`tiers: []`),
    the reserved-marker guard, the depth guard and the codec's minify-on-gate-failure
    fallback are all shipped behavior, and re-implementing any of them here would make the
    marginal column measure a terse that does not exist. A dict drop-sink is supplied so a
    `drop-to-retrieve` rule actually fires instead of warning and staying lossless.

    Caveat worth stating: capture envelopes record the tool name UNPREFIXED (#152), so a
    server-scoped rule (`secret-broker.*`) will not match here even though it matches at
    runtime. `--policy` runs are therefore a close model of a deployment, not a replay of
    one; `rules_hit` in the output says how many payloads selected a non-default rule.
    """
    def baseline(raw: str, tool: str) -> str:
        sink: dict[str, object] = {}
        try:
            applied = policy_mod.apply(raw, tool, pol, drop_sink=sink.__setitem__)
        except Exception:  # noqa: BLE001 — the proxy is fail-open; so is the measurement
            LIMITS["baseline_error"] += 1
            return raw
        if applied.tiers:
            LIMITS["rules_hit"] += 1
        return applied.text
    return baseline


def gzip_floor(text: str) -> float:
    """BYTE compression ratio — a different domain from every other column here.

    Reported for scale only, and never mixed into a token average: gzip's wins are short
    high-frequency repeats that BPE already encodes in one or two tokens, so a byte
    percentage next to a token percentage invites exactly the wrong inference.
    """
    data = text.encode()
    return 100.0 * (1 - len(gzip.compress(data)) / len(data)) if data else 0.0


# ---------------------------------------------------------------- corpora


MIN_SCORED_TOKENS = 200


def run_corpus(corpus: Path, pol: policy_mod.Policy) -> dict:
    """Per-shape aliasing ceiling over a capture corpus.

    Two different populations, deliberately:
      * the ALIASING columns skip payloads under `MIN_SCORED_TOKENS`, which are too small
        for a legend to ever pay;
      * the INFLATION counter sees every payload, because inflation happens precisely on
        the small ones a token floor would hide. Counting it over the filtered set would
        make "zero inflated payloads" a property of the filter, not of terse.
    """
    baseline = make_baseline(pol)
    agg: dict[str, Counter] = defaultdict(Counter)
    winners: Counter = Counter()
    infl = Counter()
    for env in capture.load_corpus(corpus):
        raw, tool = env.get("raw"), env.get("tool", "?")
        if not isinstance(raw, str) or not raw:
            continue
        prod = baseline(raw, tool)
        raw_t, prod_t = _tok(raw), _tok(prod)
        infl["n"] += 1
        if prod_t > raw_t:
            infl["inflated_n"] += 1
            infl["inflated_tok"] += prod_t - raw_t
        if raw_t < MIN_SCORED_TOKENS:
            infl["below_floor"] += 1
            continue
        shape = env.get("shape") or capture.classify_shape(raw)
        a_raw, how = best(raw)
        a_post, _ = best(prod)
        winners[how] += 1
        c = agg[shape]
        c["n"] += 1
        c["raw"] += raw_t
        c["prod"] += prod_t
        c["alias_raw"] += raw_t * a_raw / 100
        c["alias_post"] += prod_t * a_post / 100
        c["gzip_bytes"] += len(raw.encode()) * gzip_floor(raw) / 100
        c["bytes"] += len(raw.encode())
    return {"by_shape": {k: dict(v) for k, v in agg.items()},
            "winners": dict(winners), "inflation": dict(infl),
            "limits": dict(LIMITS)}


def iter_source(root: Path, cap: int) -> list[Path]:
    """Candidate source files under `root`, capped at `cap`.

    The cap stops the walk rather than sampling it, so a cap that actually bites biases the
    pool toward directories `os.walk` reaches first. The default (`--sample` x 40) is set
    well above the trees this is run on; `pool=` in the output is what tells you whether it
    bit. The size window is reported too — it truncates BOTH tails of the distribution, and
    the upper bound in particular drops the large generated files that would dominate a
    token-weighted average for a real `read_text_file` workload.
    """
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SOURCE_SKIP and not d.startswith(".")]
        for f in filenames:
            if Path(f).suffix in SOURCE_EXT:
                p = Path(dirpath) / f
                try:
                    if MIN_FILE_BYTES <= p.stat().st_size <= MAX_FILE_BYTES:
                        out.append(p)
                except OSError:
                    pass
        if len(out) > cap:
            LIMITS["walk_cap_bound"] += 1
            return out
    return out


def run_files(root: Path, sample: int, seed: int) -> dict:
    pool = iter_source(root, sample * 40)
    random.Random(seed).shuffle(pool)
    rows = []
    skipped = 0
    for p in pool[:sample]:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped += 1
            continue
        pct, how = best(text)
        rows.append({"file": str(p), "tok": _tok(text), "saved": pct, "encoder": how,
                     "gzip_bytes": gzip_floor(text)})
    rows.sort(key=lambda r: -r["saved"])
    return {"pool": len(pool), "skipped": skipped, "rows": rows, "limits": dict(LIMITS)}


# ---------------------------------------------------------------- reporting


def print_corpus(res: dict, policy_label: str) -> None:
    by_shape, tot = res["by_shape"], Counter()
    print(f"baseline: {policy_label}\n")
    print(f"{'shape':18} {'n':>5} {'raw tok':>10} {'terse':>8} {'alias:raw':>10} "
          f"{'alias:post':>11} {'combined':>9} {'gzip(bytes)*':>13}")
    for shape, c in sorted(by_shape.items(), key=lambda x: -x[1]["raw"]):
        for k, v in c.items():
            tot[k] += v
        print(_row(shape, c))
    if tot["raw"]:
        print(_row("TOTAL", tot))
    print("\n* gzip is a BYTE ratio, byte-weighted — a different domain from every token "
          "column beside it,\n  and an unreachable floor besides (the model must still "
          "read the output). Scale only; not comparable.")
    print("winning encoder on raw:", res["winners"])

    inf = res.get("inflation", {})
    print(f"\ninflation check over ALL {inf.get('n', 0)} payloads "
          f"({inf.get('below_floor', 0)} of them below the {MIN_SCORED_TOKENS}-token "
          f"scoring floor):")
    print(f"  terse emits LARGER than raw on {inf.get('inflated_n', 0)} "
          f"(+{inf.get('inflated_tok', 0):,} tok) — no emit-only-if-smaller guard on the "
          "lossless stage")
    _print_limits(res.get("limits", {}))


def _print_limits(limits: dict) -> None:
    if limits:
        print("\nlimits that bound this run:", limits)


def _row(label: str, c: Counter | dict) -> str:
    raw, prod = c["raw"], c["prod"]
    prod_pct = 100 * (raw - prod) / raw
    combined = 100 * (raw - (prod - c["alias_post"])) / raw
    gz = 100 * c["gzip_bytes"] / c["bytes"] if c.get("bytes") else 0.0
    return (f"{label:18} {int(c['n']):>5} {int(raw):>10,} {prod_pct:>7.1f}% "
            f"{100 * c['alias_raw'] / raw:>9.1f}% {100 * c['alias_post'] / prod:>10.1f}% "
            f"{combined:>8.1f}% {gz:>12.1f}%")


BANDS = ((30, 1e9), (20, 30), (10, 20), (5, 10), (2, 5), (-1e9, 2))


def print_files(res: dict) -> None:
    rows = res["rows"]
    if not rows:
        print("no files matched")
        return
    tot = sum(r["tok"] for r in rows)
    wsum = sum(r["tok"] * r["saved"] / 100 for r in rows)
    print(f"pool={res['pool']} scored={len(rows)} skipped={res['skipped']} "
          f"total={tot:,} tok  token-weighted saving={100 * wsum / tot:.1f}%")
    print(f"size window: {MIN_FILE_BYTES:,}-{MAX_FILE_BYTES:,} bytes — BOTH tails of the "
          "file-size\ndistribution are excluded, so this is a saving over the middle of "
          "it, not over all of it.")
    print(f"\n{'saving band':>14} {'files':>6} {'share of files':>15} {'share of tokens':>16}")
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r["saved"] < hi]
        label = f">={lo}%" if hi > 1e8 else (f"<{hi}%" if lo < -1e8 else f"{lo}-{hi}%")
        print(f"{label:>14} {len(sel):>6} {100 * len(sel) / len(rows):>14.1f}% "
              f"{100 * sum(r['tok'] for r in sel) / tot:>15.1f}%")
    print(f"\ntop 10:\n{'file':64} {'tok':>7} {'saved':>7} {'enc':>7}")
    for r in rows[:10]:
        print(f"{r['file'][-64:]:64} {r['tok']:>7,} {r['saved']:>6.1f}% {r['encoder']:>7}")
    _print_limits(res.get("limits", {}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, help="terse capture envelope directory")
    ap.add_argument("--files", type=Path, help="source tree (the read_text_file stand-in)")
    ap.add_argument("--policy", type=Path,
                    help="a real deployment's policy.json; default is the DEFAULT policy "
                         "(all three lossless tiers on every tool)")
    ap.add_argument("--sample", type=int, default=300, help="files to score (default 300)")
    ap.add_argument("--seed", type=int, default=1337, help="sampling seed (default 1337)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if not args.corpus and not args.files:
        ap.error("give --corpus, --files, or both")

    if args.policy:
        pol, label = policy_mod.load_policy(args.policy), f"policy.apply({args.policy})"
    else:
        pol, label = policy_mod.default_policy(), "policy.apply(default policy — all tiers)"

    out: dict = {}
    if args.corpus:
        out["corpus"] = run_corpus(args.corpus, pol)
        out["corpus"]["baseline"] = label
    if args.files:
        out["files"] = run_files(args.files, args.sample, args.seed)

    if args.json:
        json.dump(out, sys.stdout, indent=2, default=float)
        print()
        return 0
    if "corpus" in out:
        print(f"### capture corpus: {args.corpus}\n")
        print_corpus(out["corpus"], label)
    if "files" in out:
        print(f"\n### source tree: {args.files}\n")
        print_files(out["files"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
