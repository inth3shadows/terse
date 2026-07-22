# terse — Benchmarks

**Last updated: 2026-07-17.** All figures below were produced on that date and are
**reproducible end-to-end** with the commands shown — nothing here is hand-typed or
estimated. If you re-run and get different numbers, the code changed; open an issue.

## What is being measured

- **Token reduction** = how many fewer tokens the compressed form costs vs the raw JSON,
  counted in **`cl100k_base`** (the tiktoken vocabulary terse uses). Higher is better.
  A payload that is *already* compact (no pretty-print whitespace) makes every number here
  a *pure structural* gain — the hardest honest case.
- **Lossless** = `decompress(compress(x)) == x` exactly. Every terse row below is verified
  lossless per payload; a payload is dropped from a total if either tool fails its round-trip.
- "raw", "terse", "TOON" columns are % fewer tokens than the raw JSON.

## Reproduce everything

```bash
uv sync
cd scripts/bench && npm install          # pins the official @toon-format/toon encoder
cd -
uv run scripts/bench/benchmark.py        # §1  terse vs TOON on real GitHub API payloads
uv run scripts/bench/width_sweep.py      # §2  the column-width sweep
uv run scripts/bench/diff_demo.py        # §3  cross-call diff (terse's own axis)
```

---

## §1 — terse vs TOON on real, public GitHub API payloads

The corpus is real GitHub API output (`scripts/bench/corpus/`) — the nested, record-shaped
tool traffic terse targets. `cl100k` tokens, all lossless.

| payload | records | raw tok | **terse** | TOON |
|---|--:|--:|--:|--:|
| gh_pulls | 30 | 151,165 | **76.1%** | −8.4% |
| gh_workflow_runs | 20 | 76,032 | **80.3%** | −7.5% |
| gh_issues | 30 | 48,032 | **32.7%** | −8.0% |
| gh_commits | 30 | 69,652 | **26.5%** | −4.5% |
| gh_dir_listing | 24 | 6,736 | **31.4%** | −7.7% |
| gh_rate_limit | 1 obj | 357 | **13.4%** | −36.7% |
| gh_repo_single | 1 obj | 1,652 | 0.0% | −4.4% |
| gh_commits_flat | 30 | 10,886 | **2.4%** | 1.7% |
| gh_labels | 9 | 632 | 15.2% | **19.0%** |
| **weighted total** | | **365,144** | **58.3%** | **−7.1%** |

**Plain reading:** on real nested records terse cuts tokens **58%**; TOON *regresses* to −7%
(worse than raw) because it adds a key-path per nesting level, while terse folds the repeated
subtrees and long repeated strings (e.g. `gh_pulls` = 60 copies of the same repo object
collapsed to one legend entry → 76%). TOON wins only on `gh_labels` — a flat, short-valued
uniform table, its designed sweet spot.

---

## §2 — Column-width sweep: is there a "narrow vs wide" crossover? (No.)

A natural hypothesis is that TOON overtakes terse once records get *wide* (many columns),
because TOON writes the header once per table. We tested it directly: **40 rows held fixed,
column count swept 2→12**, seeded, each row verified lossless for both tools.

| columns | terse% | TOON% | winner |
|--:|--:|--:|--:|
| 2 | 40.4 | 44.0 | TOON +3.6 |
| 3 | 52.1 | 48.7 | terse +3.4 |
| 4 | 46.8 | 48.8 | TOON +2.0 |
| 5 | 52.6 | 50.7 | terse +1.9 |
| 6 | 48.9 | 50.5 | TOON +1.6 |
| 7 | 52.7 | 51.5 | terse +1.2 |
| 8 | 50.0 | 51.2 | TOON +1.2 |
| 9 | 52.9 | 52.1 | terse +0.8 |
| 10 | 50.6 | 51.7 | TOON +1.1 |
| 11 | 52.9 | 52.3 | terse +0.6 |
| 12 | 51.1 | 52.1 | TOON +1.0 |

**Plain reading:** there is **no clean column-count crossover.** The winner oscillates by
parity, the margins are ~1–4 points, and they **converge toward a tie** as width grows — the
opposite of "TOON pulls decisively ahead when wide." (An earlier draft of this repo's README
claimed a ≤3/≥4-column boundary from a single synthetic construction; a seeded sweep does not
reproduce it, and the claim was corrected.)

**The real dividing axis is value repetition, not width.** On these stripped-flat synthetic
tables — no nesting, no long repeated strings — terse's dictionary/subtree tiers have little
to fold, so the two tools tie. terse's decisive §1 win comes precisely from the redundancy
that real records have and synthetic flat tables don't.

---

## §3 — Cross-call diff (an axis no stateless encoder has)

When the same tool is called again (poll a list, re-read a file), terse can emit a lossless
*delta* against the prior result instead of the whole payload. TOON, minify, and terse's own
single-shot codec all pay the full column every call. Modeling one repeat call per payload
(`diff_demo.py`), the **second** call costs:

| repeated call | full re-send | diff | smaller by |
|---|--:|--:|--:|
| gh_commits_flat | 10,681 | 812 | **92.4%** |
| gh_issues | 32,608 | 4,448 | **86.4%** |
| gh_pulls | 37,776 | 15,292 | **59.5%** |
| **weighted total** | 152,837 | 40,138 | **73.7%** |

**Honest caveat (read this):** these are *modeled* repeat-call savings. How *often* the
pattern occurs in a real agent loop is workload-dependent and is being measured directly (the
proxy now records a per-result `diff_reason` — run `terse stats` to see the breakdown for your
own traffic). Do **not** read §3 as a claim about aggregate real-world savings; read §1 for that.

---

## §4 — Competitor landscape (hands-on, tested 2026-07-17)

Installed and tested, not cited from marketing. Only TOON (§1) is directly comparable on a
lossless token axis; the rest measure *different guarantees*, so no head-to-head % is claimed.

| Tool | What it is (verified) | Comparable? |
|---|---|---|
| **headroom** (`headroom-ai`, v0.32.0) | JSON compressor is a **deterministic Rust transform, not ML**: lossless on uniform arrays, but **drops rows** on larger/irregular sets, recoverable only via a `retrieve` round-trip against a **time-boxed cache** (default 30-min TTL). Measured on our corpus: 33.1% (lossless reformat) to 42.5%/64.1% (lossy, dropping 13/30 and 13/20 rows). A separate optional text compressor *is* ML. | No — its larger numbers come from *dropping data* with time-limited recovery; terse's are unconditionally lossless. |
| **LLMLingua-2** (Microsoft) | Lossy prompt token-classifier. Fed JSON it strips syntax (`{`,`}`,`:`,`"`) as low-information and emits **invalid, unparseable JSON**; truncates past 512 tokens. ~50% on both prose and JSON. | No — different axis (prompts, not tool output), lossy, corrupts structure. |
| **Atlassian mcp-compressor** | Primarily lossless schema/description compression at connect time — **complementary and stackable** with terse (`terse proxy -- mcp-compressor -- <server>`). An opt-in `--toonify` flag also reformats results into TOON (off by default; no diffing/policy/state). | Adjacent, not competing. |
| **Anthropic / OpenAI context editing** | Native, server-side, **lossy** history-pruning; no local artifact to run keylessly. | Different mechanism (drops old results server-side). |

---

## §5 — In production: the live ledger (not just the curated corpus)

§1–3 are a fixed corpus. This section is **real proxied traffic** — the author's own kb /
secret-broker / runecho / codegraph sessions — read from terse's always-on, **payload-free**
savings ledger (sizes + decisions only, never content). Unlike §1–3, a stranger can't
reproduce *these* numbers (they're one person's traffic); the point is the opposite — here is
what an honest production figure looks like, and the one command that gives you *yours*.

**Headline (measured 2026-07-22, `terse stats`, ledger spans 7 days):**

```
1,526 results   470,609 -> 427,378 tok   9.2% blended
```

That 9.2% is honest and *incomplete*, for two reasons — and both are the point of this section.

**1. This ledger straddles the #116 transition.** Most of its records predate cross-block
joining (the `multiblock` diff-reason bucket is still 345 of them), so it mostly reflects the
old per-block path. `terse stats` on post-#116 traffic reads higher for repeat-heavy loops
(below). The number is *composition*, not a constant — which is why we publish a range.

**2. Savings track payload shape.** Which tools you call sets the mix. Measured on the real
captured records (production policy, deduplicated to one call's worth per tool):

| shape | example tool | codec: per-block → joined (#116) | an *unchanged* repeat |
|---|---|--:|--:|
| wide, low-cardinality | `kb.read.changelog` | 21% → **38%** | ~99% |
| | `kb.read.recent_rejections` | 17% → **33%** | ~99% |
| | `kb.read.for_repo` | 15% → **24%** | ~98% |
| prose-heavy records | `kb.read.list_principles` | 3% → 3% *(hard ceiling)* | **~99.9%** |
| | `kb.read.get` | 2% → 2% | ~99.9% |
| already-projected small | `kb.read.query_stats` | 41% → 41% | — |
| tiny status objects | *(policy `tiers:[]`)* | 0% → 0% *(correct — already minimal)* | — |

The prose ceiling is structural, exactly as predicted: long unique text in
`principle`/`rationale`/`evidence` has nothing to fold, and no tier combination changes that.

**What #116 actually changed here.** The codec fold (per-block → joined) helps the wide
low-cardinality tools and does ~nothing for prose. The real lever is the **diff tier**, which
the per-block path could never reach:

- **76% of ledger tokens** are the multiblock JSON shape #116 targets.
- **71% of ledger tokens are now diff-eligible** — a join fires *and* a repeat produces a
  lossless delta — where **before #116 that share was 100% excluded** from diffing.
- On an *unchanged* repeat, those results collapse **~99%** (the right column above). kb data
  changes slowly and these tools are re-read many times per session (`list_principles`: 865
  calls in this 7-day ledger), so in a real agent loop a large fraction of calls after the
  first are near-empty diffs.

**So the production figure is a range, not a point:**

- **Floor** — every call, no repeats: the joined codec alone, ~9–12%, dominated by prose
  ceilings.
- **Ceiling** — repeat-heavy loop with data stable between calls: re-weighting the ledger's
  own token mix, the diff-eligible 71% collapsing ~99% each puts the aggregate near **~71%**.
- **Reality sits between**, set by *your* repeat rate and how fast *your* data changes — which
  no benchmark can tell you. So measure it:

```bash
terse stats                 # rollup: results, decisions, tokens saved, per-tool rows,
                            #          and the diff-reason breakdown (how often diffs fire)
terse stats --since 7d      # windowed
```

Wrap your servers (`terse install-mcp …`), use them for a week, read your own ledger. That
converts "trust our benchmark" into "run it on your traffic" — the honest version of the
claim, and a better pitch besides.

---

## Methodology & honesty notes

- Tokenizer is `cl100k_base`; absolute % shift under a different vocabulary but the ranking is
  stable (terse's cross-tokenizer-invariance claim, tested separately in the suite).
- §1 corpus is real, public GitHub API output. §2 is **synthetic and seeded** — illustrative
  of a mechanism, not production-representative; the exact numbers depend on the construction
  (short keys, value cardinality), which is why §2's takeaway is "no crossover," not a constant.
- Every terse figure is verified lossless per payload. §4's headroom row is the only place a
  "reduction %" is reported for a tool that achieves it by *discarding* data — flagged inline.
- Adoption honesty: terse is new (pre-PyPI as of this date); TOON and headroom are far more
  established. terse's wedge is narrow and specific — *unconditionally lossless, no ML, no
  egress* — not breadth of adoption.
