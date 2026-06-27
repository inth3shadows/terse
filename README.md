# terse

Lossless-first compression layer for AI-agent tool outputs — byte-faithful on
fields you mark critical, configurable lossy reduction only where you opt in.

terse is the inverse of blanket lossy offload (e.g. headroom's "drop to a
retrieve-cache"). Instead of evicting data and making the model call `retrieve`,
terse keeps everything **resident and legible** in context and removes only
*structural* overhead: pretty-print whitespace, keys repeated once per record,
repeated values, repeated nested schema. Tokens go down; nothing the model needs
leaves the window; there is no decode step.

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
- **Tier 0.5 — dictionary code**: repeated string values are folded into an inline
  legend (`~0`, `~1`, …) proven disjoint from every literal in the payload.
- **Tier 1 — lossy (opt-in, NOT YET BUILT)**: per marked field — truncate /
  summarize / drop-to-retrieve. The policy schema accepts these today but the
  engine warns and leaves them lossless until the tier exists.

Every transform has an exact inverse, and a round-trip gate asserts
`decompress(compress(x)) == x` over the whole corpus. The transformed bytes *are*
the model's input — a denser but still-readable representation, not an offload.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- `tiktoken` (installed via `uv sync`) for token counting
- Optional: the `anthropic` extra + an API key, only if you want a real
  Anthropic `count_tokens` point-check (not required; see USAGE)

## Quick Start

```bash
uv sync

# Is a payload losslessly compressible, and by how much?
echo '[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]' | uv run terse gate -

# Compress a tool output through a per-tool policy
some-tool-emitting-json | uv run terse compress --tool gh.api.repos --policy policy.example.json -

# Run the test suite (it IS the lossless gate)
uv run pytest
```

## Project Structure

```
src/terse/
  transforms.py  lossless tiers (minify, tabularize, dict coding) + round-trip gate
  policy.py      selective per-tool policy: load, match, apply
  capture.py     corpus capture (shape-tagged envelopes) + shape classifier
  measure.py     per-payload + cross-tokenizer token measurement
  probes.py      value-redundancy + cross-call-overlap ceiling probes
  tokenize.py    cl100k / o200k token counting (+ optional Anthropic)
  report.py      markdown reports (savings, per-tool, probes, tokenizer)
  cli.py         entrypoint: gate / capture / measure / probe / validate / compress
tests/           round-trip, measurement, probe, and policy tests
policy.example.json   selective policy encoding the measured per-tool insight
corpus/          captured tool outputs (gitignored; may contain real data)
```

## Related Documentation

- [Technical Reference](TECHNICAL.md) — architecture, pipeline, policy schema, limitations
- [Usage Guide](USAGE.md) — running the CLI day-to-day and reading its output

## Status

Phase-0 spike: a working, measured, selective **lossless** library + CLI. The MCP
proxy wrapper and the Tier 1 lossy modes are designed but not yet built — see
TECHNICAL.md "Known Limitations".
