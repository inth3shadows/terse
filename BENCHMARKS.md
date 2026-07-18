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

## §5 — Generate your own real-session evidence

The benchmarks above are a fixed corpus. terse also keeps an always-on, **payload-free**
savings ledger from your real sessions (sizes + decisions only, never content), so you can
measure what terse saved *you*, not a synthetic corpus:

```bash
terse stats                 # rollup: results, decisions, tokens saved, per-tool rows
terse stats --since 7d      # windowed
```

This is terse's "evidence over stars" path: your own numbers, on your own traffic, verifiable
without trusting this file.

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
