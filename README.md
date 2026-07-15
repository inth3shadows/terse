# terse

Lossless-first compression layer for AI-agent tool outputs — byte-faithful on
fields you mark critical, configurable lossy reduction only where you opt in.

terse is lossless-first. Unlike blanket lossy offload (e.g. headroom's "drop to a
retrieve-cache"), it keeps everything **resident and legible** by default and removes
only *structural* overhead: pretty-print whitespace, keys repeated once per record,
repeated values, repeated nested schema. Tokens go down; nothing the model needs
leaves the window; there is no decode step. Lossy reduction is strictly opt-in, per
field — including `drop-to-retrieve`, a deliberate escape hatch that evicts a marked
field to a handle the model can fetch back with a `terse.retrieve` tool. It is off by
default and never the rule.

It is **selective by design**. Measurement on real tool output showed the win is
strongly per-tool (0–30%): large on record/symbol-shaped verbose output, near-zero
on already-minified or already-projected tools. So terse applies per-tool policy
rather than compressing everything blindly.

Lossless tabularization of uniform JSON arrays isn't unique to terse — formats like
[TOON](https://toonformat.dev/) publish the same primitive as a standalone, MIT-
licensed encoding (~40% token reduction, independently benchmarked). terse's
differentiation isn't the tabularization trick alone; it's the bundle: MCP-native
proxy packaging (transparent to any downstream server, no client-side reformatting
required), per-tool policy, cross-call diffing, a fluency-gated lossy escape hatch,
and self-installing ops tooling (`install-mcp`, `mcp-status`) — combined, and each
diff/lossy tier validated by a behavioral eval before it was ever turned on by
default.

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
```

## Related Documentation

- [Verify it yourself](VERIFY.md) — prove losslessness, savings, and no-egress locally
- [Technical Reference](TECHNICAL.md) — architecture, pipeline, policy schema, limitations
- [Usage Guide](USAGE.md) — running the CLI day-to-day and reading its output

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
