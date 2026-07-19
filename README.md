# terse

The **lossless-first** MCP compression proxy: it makes tool output smaller without
ever changing what your agent reads — byte-faithful by default, lossy only where you
explicitly opt in.

terse reduces tokens two ways: one that carries the day-to-day value, and one that is
harder for a competitor to copy. Keeping those straight is the whole positioning.

**1. The lossless codec — the value.** This is the lever that does the work. terse
removes only *structural* overhead: pretty-print whitespace, keys repeated once per
record, repeated values, repeated nested schema. The transformed bytes **are** the
model's input — a denser but still legible representation, not an offload. There is no
decode step, no ML model in the loop, and every transform has an exact inverse (a
round-trip gate asserts `decompress(compress(x)) == x` over the whole corpus). This is
the guarantee most tools in this space decline to make: headroom's JSON path is lossless
on uniform arrays but **falls back to dropping rows** on larger/irregular record sets,
recoverable only via a `retrieve` round-trip against a cache that expires (verified,
v0.32.0; default 30-min TTL); Anthropic/OpenAI context-editing **drops** old tool results
server-side. terse never silently mutates what the model sees — and "lossless" is the
category, not the token count. In terse's own production ledger this codec is where
essentially all the savings come from (see Status). Its one honest caveat: the
tabularization primitive is public (formats like [TOON](https://toonformat.dev/) publish
it standalone, MIT-licensed, ~40% on flat arrays), so a motivated competitor could clone
the codec in a weekend.

**2. The stateful cross-call diff — the defensible axis.** When the same tool is called
again — poll a list, re-read a file — terse emits a lossless *delta* against the prior
result instead of the whole payload (**~73% smaller on the repeated call** in the model
below). This is the one axis a stateless encoder **architecturally cannot reach**: TOON,
headroom's stateless per-call compressor, and server-side history-pruning all pay the
full column every call because none of them remember the last result — terse can only do
it because it lives in the session as a transparent proxy. That makes it the harder half
to copy. But it is a **bonus tier, not the headline**: it only pays off when a workload
actually repeats a call with a similar-enough payload, which is rarer than it sounds — in
terse's own 7-day production traffic the diff tier fired on ~0.4% of results (2 of 464).
When your loop *does* re-fetch mostly-unchanged results it compounds hard; when it
doesn't, it costs nothing (lossless, and emitted only when smaller). Default-on since its
validation program completed (see Status).

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

## Install

Needs Python 3.11+.

```bash
uv tool install terse-mcp   # global `terse` CLI  (or: pipx install terse-mcp)
```

Or `pip install terse-mcp` into a virtualenv for library/embedded use.

## Quick Start (under a minute)

terse sits between your MCP client and a server and shrinks the server's tool results in
flight. **No config needed** — the proxy is lossless-everywhere by default:

```bash
# 1. Wrap ANY stdio MCP server. Your agent talks to it exactly as before;
#    terse compresses the results it returns, losslessly.
terse proxy -- uvx some-mcp-server --flags

# 2. See what it saved (the payload-free ledger is on by default):
terse stats
```

Want to eyeball the codec first, no server involved?

```bash
echo '[{"id":1,"state":"open","repo":"acme/widgets"},{"id":2,"state":"open","repo":"acme/widgets"},{"id":3,"state":"open","repo":"acme/widgets"},{"id":4,"state":"open","repo":"acme/widgets"},{"id":5,"state":"open","repo":"acme/widgets"},{"id":6,"state":"open","repo":"acme/widgets"}]' | terse gate -
# → round-trip lossless: PASS ; ~36% fewer cl100k tokens
```

(Savings grow with record count and repetition; on a single tiny object terse correctly
declines and passes it through unchanged — it never inflates what it can't shrink.)

## Does terse help my server?

The win is per-tool and terse only keeps what pays, so it never hurts — but it helps a lot
more on some shapes than others. Point it at a server and run `terse stats` to see for real;
as a rule of thumb:

| terse helps most | terse barely moves |
|---|---|
| record/array JSON (lists of objects) | already-minified or already-projected output |
| repeated values or nested repeated subtrees | free-text-dominated results (logs, prose, diffs) |
| verbose REST-ish payloads (GitHub, Jira, DB rows) | tiny single objects |
| tools you call repeatedly (cross-call diff) | binary / non-JSON blobs (passed through untouched) |

## Wire it into your MCP client (permanent)

`install-mcp` rewrites your MCP config to launch a server *through* terse — reversible, and
transparent to the client. It needs a per-tool policy; the smallest useful one is:

```bash
echo '{"version":1,"defaults":{"tiers":["minify","tabularize","dictionary"]}}' > terse-policy.json
```

```bash
# Claude Code, user scope (~/.claude.json) — wrap a server you've already registered by name:
terse install-mcp --policy terse-policy.json <server-name>

# Project scope (a committed .mcp.json instead):
terse install-mcp --policy terse-policy.json --scope project --file .mcp.json <server-name>

terse mcp-status                       # confirm what's wrapped
terse uninstall-mcp <server-name>      # cleanly restore the original entry
```

Other MCP clients (Cursor, etc.) read the same config shape — wherever a server is launched
as `cmd --flags`, launch it as `terse proxy -- cmd --flags` to get the same effect.
See [USAGE.md](USAGE.md) for tuning a policy (`terse tune`) and reading `terse stats`.

**From source** (contributors): `uv sync` then `uv run terse ...`; `uv run pytest` is the
lossless gate.

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
- **TOON is not beaten everywhere — and the boundary is value repetition, not column
  width.** On `gh_labels` (9 records × 7 columns — TOON's designed sweet spot) TOON leads,
  **+19.0% vs terse's +15.2%**. terse's decisive corpus win comes from **nested repeated
  subtrees and long repeated string values** — its dictionary and subtree-aliasing tiers fold
  them, TOON's flat tabular layout cannot — which is exactly what real GitHub records carry.
  On *stripped-flat synthetic tables* with none of that redundancy, the two converge: a seeded
  column-width sweep (`uv run scripts/bench/width_sweep.py`) shows them within a few points at
  every width, trading the lead by parity, with **no clean column-count crossover** (an earlier
  claim of a ≤3/≥4 boundary did not reproduce — see BENCHMARKS.md). So the honest frame is:
  terse wins where records repeat or nest (real tool output); TOON stays competitive on flat,
  low-redundancy uniform tables — "different niche," not "terse strictly dominant."
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
is on top of the single-shot reduction above, and stacks with it. These figures are the
diff's win *when it fires* on a repeated call; how often that actually happens is a
separate question — in terse's own production ledger it is ~0.4% of results, so treat
this as a defensible bonus tier, not the headline lever (that's the codec — see the
positioning note at the top and Status).

**Tools not benchmarked head-to-head, and why (no invented numbers):**

| Tool | Why not a like-for-like row |
|---|---|
| **headroom** (`headroom-ai`, headroomlabs-ai) | The closest *product* competitor and far more adopted (star figures cited vary, ~29–49k — unverified). But its JSON compressor is a **deterministic Rust transform, not an ML model** (verified, v0.32.0): lossless on uniform arrays, yet **falling back to dropping rows** on larger/irregular record sets, recoverable only via a `retrieve` round-trip against a **time-boxed, backend-dependent cache** (default 30-min TTL; SQLite in the proxy path, in-memory if constructed directly — gone after expiry, eviction, or process exit). A separate, optional text/log compressor *is* ML (a keyless model download). Not comparable on a lossless token axis: terse's guarantee is unconditional — no cache, no TTL, no ML, no egress. (The `headroom` package on PyPI is a different, unrelated CLI.) |
| **LLMLingua-2** (Microsoft, 6.4k★) | Lossy prompt compression via a trained token-classifier; operates on **input prompts**, not structured tool output. Verified on a JSON payload it strips the syntax (`{`, `}`, `:`, `"`) as low-information and emits **invalid, unparseable JSON** (and silently truncates past its 512-token window). Different axis entirely. |
| **Anthropic context editing** / OpenAI equivalents | **Native, server-side, lossy** history-pruning (drop oldest tool results past a threshold), no local artifact to run keylessly. This — not any third-party tool — is the real strategic overlap with terse for first-party API users. |
| **Atlassian mcp-compressor** (97★) | Primarily compresses tool **schemas/descriptions** at connect time (lossless deferred-disclosure) — complementary and stackable with terse (`terse proxy -- mcp-compressor -- <server>`). Caveat: an opt-in `--toonify` flag *does* reformat call **results** into TOON, so "schemas only" isn't strictly true — but that path is off by default and is a single static pass with no diffing, per-tool policy, or cross-call state, out-competed by terse's codec on that axis. |

Adoption honesty: terse is new (just published to PyPI, few/no stars); TOON (24.9k★) and headroom
(widely adopted, star figures cited vary ~29–49k) are far more established. terse's
defensible wedge is narrow and specific — *unconditionally lossless (no expiring
retrieve-cache), no ML dependency, MCP-transparent, plus cross-call diffing* — not breadth
of adoption.

## Related Documentation

- [Benchmarks](BENCHMARKS.md) — dated, reproducible terse-vs-TOON + competitor numbers
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
