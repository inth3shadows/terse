# terse

Lossless-first compression layer for AI-agent tool outputs — byte-faithful on
fields you mark critical, configurable lossy reduction only where you opt in.

The inverse of blanket lossy offload: data stays **resident and legible** in
context (no decode/retrieve step); tokens are reduced by lossless structural
encoding where redundancy exists, and lossy reduction is strictly opt-in per
field. See the full design at
`~/.claude/plans/terse-lossless-tool-output-compression.md`.

> Status: **Phase-0 spike**. The lossless spine (minify + tabularize + round-trip
> gate) and token counting are implemented; corpus capture, the ceiling probes,
> and the per-tier/per-bucket report are stubbed to the plan.

## Try the lossless gate

```bash
uv sync
echo '[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]' | uv run terse gate -
```

Prints whether the Tier-0 pipeline round-trips losslessly, the shape bucket, and
the cl100k token delta.

## Tests

```bash
uv run pytest
```

The test suite *is* the lossless gate: every transform must satisfy
`decompress(compress(x)) == x`.

<!-- Full README / TECHNICAL / USAGE generated via /docs before first push. -->
