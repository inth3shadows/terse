# terse

The **lossless-first** MCP compression proxy: it makes tool output smaller without
ever changing what your agent reads — byte-faithful by default, lossy only where you
explicitly opt in.

terse reduces tokens two ways, and they are not equally easy to copy. That split is
the whole positioning.

**1. The lossless codec — the reach.** terse removes only *structural* overhead:
pretty-print whitespace, keys repeated once per record, repeated values, repeated
nested schema. The transformed bytes **are** the model's input — a denser but still
legible representation, not an offload. There is no decode step, no ML model in the
loop, and every transform has an exact inverse (a round-trip gate asserts
`decompress(compress(x)) == x` over the whole corpus). This is the guarantee most
tools in this space decline to make: headroom's default JSON path is a **lossy** ML
model behind a `retrieve` round-trip; Anthropic/OpenAI context-editing **drops** old
tool results server-side. terse never silently mutates what the model sees — and
"lossless" is the category, not the token count. The tabularization primitive itself
is public (formats like [TOON](https://toonformat.dev/) publish it standalone, MIT-
licensed, ~40% on flat arrays), so the codec is terse's *demo*, not its moat: a
motivated competitor could clone it.

**2. The stateful cross-call diff — the moat.** When the same tool is called again —
poll a list, re-read a file — terse emits a lossless *delta* against the prior result
instead of the whole payload (**73% smaller on repeated calls** in the benchmark
below). This is the one axis a stateless encoder **architecturally cannot reach**:
TOON, headroom's per-call model, and server-side history-pruning all pay the full
column every call because none of them remember the last result. terse can only do it
because it lives in the session as a transparent proxy. Cloning the codec is a
weekend; cloning the diff means becoming a session-spanning proxy — a different
product. Default-on since its validation program completed (see Status).

Around those two sits the **bundle** that turns a byte filter into a control plane you
don't want to rip out: MCP-native proxy packaging (transparent to any downstream
server, no client-side reformatting), a **live savings ledger** (`terse stats`), a
fluency-gated lossy escape hatch, and self-installing ops tooling (`install-mcp`,
`mcp-status`) — each diff/lossy tier validated by a behavioral eval before it was ever
turned on by default.

It is **selective by design**. Measurement on real tool output showed the win is
strongly per-tool (0–30%): large on record/symbol-shaped verbose output, near-zero
on already-minified or already-projected tools. So terse applies per-tool policy
rather than compressing everything blindly.

## How It Works

terse transforms a tool's JSON output through a tiered, fully-lossless pipeline,
then (optionally) serves it through a per-tool policy that decides which tiers run.

- **Tier 0 — minify**: strip insignificant whitespace.
- **Tier 0 — tabularize**: a list of uniform records becomes one header + value
  rows (keys written once, not once per record), recursively hoisting nested
  uniform-dict columns into a shared header.
- **Tier 0.5 — dictionary code**: repeated string values *and repeated whole subtrees*
  are folded into an inline legend (`~0`, `~1`, …) proven disjoint from every literal in
  the payload. Committed only when it actually saves tokens, so it never regresses.
- **Tier 0.7 — cross-call diff (stateful, ON by default)**: when the same tool is called
  repeatedly, the proxy emits a lossless delta against the prior result instead of
  the full payload (the 91%-overlap headroom). Self-describing, verified to reconstruct
  exactly, and emitted only when smaller — falls back to the full form otherwise.
  Default-on since its validation program completed (fluency, nested-record coverage,
  and the drift soak — see Status); opt out with `proxy --no-diff` / `install-mcp
  --no-diff` or a policy-file `"diff": false`.
  Record-shaped JSON gets a row/key diff; non-JSON results (file reads, source excerpts,
  log tails) get a separate content-defined-chunking (CDC) diff — a rolling hash cuts
  chunk boundaries by content, not position, so an edit anywhere only perturbs the
  chunk(s) it overlaps and the rest is sent as references to the prior result. Each
  shape keeps its own diff base per tool.
- **Tier 1 — lossy (opt-in, per field)**: `truncate` caps and annotates a field marked
  `{"lossy":"truncate","max":N}`, gated by an acceptable-loss check (only marked,
  non-`critical` fields may differ, each only as a valid truncation). `drop-to-retrieve`
  replaces a marked field with a handle, stores the original per session, and serves it
  back via a synthetic `terse.retrieve` tool the proxy injects — gated so a drop is
  accepted only if the handle resolves to the exact original. `summarize` (needs a model)
  is still parsed but deferred — warned and left lossless. Off everywhere by default.

Every transform has an exact inverse, and a round-trip gate asserts
`decompress(compress(x)) == x` over the whole corpus. The transformed bytes *are*
the model's input — a denser but still-readable representation, not an offload.

The proxy also keeps a **live savings ledger** (on by default; `--no-stats` to opt
out): one payload-free JSONL record per result — sizes, tokens, and the decision
taken, never content — so `terse stats` can answer "how much did terse actually save
me this week?" from real sessions, not just the synthetic corpus.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- `tiktoken` (installed via `uv sync`) for token counting

## Quick Start

```bash
uv sync

# Is a payload losslessly compressible, and by how much?
echo '[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]' | uv run terse gate -

# Compress a tool output through a per-tool policy
some-tool-emitting-json | uv run terse compress --tool gh.api.repos --policy policy.example.json -

# Run an MCP server behind terse: it compresses that server's tool results live
uv run terse proxy --policy policy.example.json -- uvx some-mcp-server --flags

# Run the test suite (it IS the lossless gate)
uv run pytest
```

## Project Structure

```
src/terse/
  transforms.py  lossless tiers (minify, tabularize, dict coding) + round-trip gate
  policy.py      selective per-tool policy: load, match, apply
  proxy.py       MCP stdio middleware: compress a downstream server's tool results
  stats.py       live savings ledger (payload-free) + the `terse stats` aggregation
  capture.py     corpus capture (shape-tagged envelopes) + shape classifier
  measure.py     per-payload + cross-tokenizer token measurement
  probes.py      value-redundancy + cross-call-overlap ceiling probes
  fluency/       does a model read the compressed form as accurately as raw JSON?
                 (questions / scoring / answerers / harnesses / pack behind one facade)
  tokenize.py    cl100k / o200k token counting
  report.py      markdown reports (savings, per-tool, probes, tokenizer, fluency)
  html_report.py charted HTML companion (inline SVG, no JS/CDN) for measure/verify
  cli.py         entrypoint: gate / capture / measure / probe / validate / compress / proxy / stats / fluency
scripts/
  gen_stress_corpus.py  synthetic stress corpus for the fluency eval
  bench/                terse-vs-TOON token benchmark on a real GitHub-API corpus
                        (fetch_corpus.sh, benchmark.py, diff_demo.py, toon_encode.mjs)
tests/           round-trip, measurement, probe, policy, and fluency tests
policy.example.json   selective policy encoding the measured per-tool insight
corpus/          captured tool outputs (gitignored; may contain real data)
```

## Verify it yourself

terse sits in your agent's critical path, so it earns trust by inspection. See
[VERIFY.md](VERIFY.md) for the full walkthrough — or generate a self-contained
report (lossless gate + per-tool token savings) in one command:

```bash
terse verify --out reports/verify-report.md          # bundled sample, zero setup
terse verify --corpus corpus --out report.md         # your own captured traffic
terse verify --html --out reports/verify-report.md   # + a charted HTML report alongside it
terse verify --corpus corpus --json                  # machine-readable gate + savings (CI-checkable)
```

## Benchmarks: terse vs alternatives

A head-to-head token-reduction benchmark on a corpus of **real, public GitHub API
payloads** (`scripts/bench/`) — the nested, record-shaped output that dominates real MCP
tool traffic. Everything below is **lossless and verified per payload** (a row is dropped
from the total if either encoder fails its round-trip), counted in `cl100k_base` (the same
tokenizer terse uses). Reproduce end-to-end:

```bash
cd scripts/bench && npm install          # pins the official @toon-format/toon encoder
./fetch_corpus.sh                        # OR use the committed corpus snapshot as-is
uv run scripts/bench/benchmark.py        # terse vs TOON vs baselines
uv run scripts/bench/diff_demo.py        # terse's cross-call diff (its own axis)
```

The only directly-comparable public tool is **[TOON](https://toonformat.dev/)** — a
lossless encoding that shares terse's tabularization primitive. The corpus is compact JSON
(no pretty-print whitespace), so `minify` saves ~0% and every number below is *pure
structural* gain, the hardest honest case:

| payload (real GitHub API) | records | raw tok | terse | TOON |
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
| **weighted total** | | 365,144 | **58.3%** | **−7.1%** |

*(% = fewer cl100k tokens than raw; higher is better; **bold** = winner.)*

**Honest reading of this:**

- **On real nested API records, terse wins decisively and TOON regresses** (−7.1% overall
  — *worse* than raw). TOON is built for **flat, uniform arrays**; GitHub records are
  deeply nested and non-uniform (a PR embeds repeated `user`/`head`/`base`/`repo`
  subtrees), which terse's dictionary tier folds and TOON's tabular layout cannot — it
  adds key-path overhead instead. terse's headline 76% on `gh_pulls` is exactly this:
  60 repeated copies of the same repo object collapsed to one legend entry.
- **TOON is not beaten everywhere — and we show where it wins.** On `gh_labels` (a flat,
  short-valued uniform table — TOON's designed sweet spot) TOON leads, **+19.0% vs terse's
  +15.2%**. TOON's own published ~40% figures are real *for that input shape*; this corpus
  deliberately tests the different thing (nested tool output), so treat this as
  "different niche," not "terse strictly dominant."
- **Neither tool helps much when free text dominates** (`gh_commits_flat`: long commit
  messages, ~2% either way) or on tiny single objects — matching terse's own "selective,
  0–30%, per-tool" claim rather than contradicting it.
**Cross-call diff — the axis no stateless encoding has.** When the same tool is called
again (poll a list, re-read a file), terse emits a lossless *delta* against the prior
result instead of the whole payload. TOON, minify, and terse's own single-shot codec all
pay the full column every call. Modeling one repeated call per payload (two records
changed, one appended — the poll-again pattern), the **second** call costs
(`uv run scripts/bench/diff_demo.py`):

| repeated call | records | full re-send (terse) | diff | diff smaller |
|---|--:|--:|--:|--:|
| gh_commits_flat | 30 | 10,681 | 812 | **92.4%** |
| gh_commits | 30 | 51,623 | 6,273 | **87.8%** |
| gh_issues | 30 | 32,608 | 4,448 | **86.4%** |
| gh_dir_listing | 24 | 4,779 | 977 | **79.6%** |
| gh_pulls | 30 | 37,776 | 15,292 | **59.5%** |
| gh_workflow_runs | 20 | 15,370 | 12,336 | 19.7% |
| **weighted total** | | 152,837 | 40,138 | **73.7%** |

The diff cost scales with *what changed*, not with payload size — so its win compounds
exactly where token cost otherwise does: a long agent loop re-fetching mostly-unchanged
results. (`gh_workflow_runs` is lower here only because its records are large and this
model changed a big nested field; a status/timestamp churn would diff far smaller.) This
is on top of the single-shot reduction above, and stacks with it.

**Tools not benchmarked head-to-head, and why (no invented numbers):**

| Tool | Why not a like-for-like row |
|---|---|
| **headroom** (headroomlabs-ai, ~60k★) | The closest *product* competitor and far more adopted, but its default JSON path is **lossy** (an ML model) with a `retrieve` round-trip — a different guarantee than terse's lossless-first, not comparable on a lossless token axis. (The `headroom` package on PyPI is an unrelated CLI assistant.) |
| **LLMLingua-2** (Microsoft, 6.4k★) | Lossy prompt compression via a trained classifier; operates on **input prompts**, not structured tool output. Different axis. |
| **Anthropic context editing** / OpenAI equivalents | **Native, server-side, lossy** history-pruning (drop oldest tool results past a threshold), no local artifact to run keylessly. This — not any third-party tool — is the real strategic overlap with terse for first-party API users. |
| **Atlassian mcp-compressor** (97★) | Compresses tool **schemas/descriptions** at connect time, not call **results** — adjacent, not competing. |

Adoption honesty: terse is new (pre-PyPI, few/no stars); TOON (24.9k★) and headroom
(~60k★) are far more established. terse's defensible wedge is narrow and specific —
*lossless-first, no retrieve round-trip, no ML dependency, MCP-transparent, plus cross-call
diffing* — not breadth of adoption.

## Related Documentation

- [Verify it yourself](VERIFY.md) — prove losslessness, savings, and no-egress locally
- [Technical Reference](TECHNICAL.md) — architecture, pipeline, policy schema, limitations
- [Usage Guide](USAGE.md) — running the CLI day-to-day and reading its output
- [Changelog](CHANGELOG.md) — notable changes per release

## Status

A working, measured, selective **lossless** library, CLI, and MCP
stdio proxy. The proxy's open question — *does a model read the compressed form as
well as raw JSON?* — now has a measured answer: on a stress corpus, Claude Haiku 4.5
and Gemini 2.5 Flash match raw-JSON accuracy on the compressed form (100% paired) at a
37% token saving (`terse fluency`; see TECHNICAL.md). Whole-subtree aliasing (folding
repeated objects, not just strings) is built. Cross-call diffing is a lossless tier
that is now **on by default** — its full validation program passed: pair fluency
(`fluency --diff`, 4-model panel 100%), the nested-record surface (`structure`: diff
100% vs full-terse 94%), and long-chain drift soaked from both sides — mechanically
(`tests/test_diff_soak.py` — exact reconstruction hundreds of chained hops deep) and
behaviorally (`fluency --diff-soak` — no depth-correlated accuracy loss up to the
keyframe bound). Opt out per proxy (`--no-diff`) or per policy (`"diff": false`).
The Tier 1 lossy modes `truncate` and
`drop-to-retrieve` are built (opt-in, off by default); `summarize` remains designed but
not yet built — see TECHNICAL.md "Known Limitations".
