# Technical Reference: terse

## Architecture

terse is a pure-Python library with a thin CLI. The core is a deterministic,
fully-lossless transform pipeline; everything else (measurement, probes, policy)
is built around it.

```
raw tool output (JSON text)
        │
        ▼
 json.loads ──► compress_structure  (Tier 0: tabularize, recursive nested-key fold)
        │              │
        │              ▼
        │        dict_encode        (Tier 0.5: fold repeated string values -> legend)
        │              │
        │              ▼
        │          minify           (Tier 0: strip whitespace, serialize)
        ▼              │
   (policy decides ◄───┘
    which tiers run per tool)
        │
        ▼
   compressed JSON text  ──► decompress ──► byte-identical original
```

**Design decisions:**

- **Lossless gate is non-negotiable.** `transforms.roundtrip_ok(obj)` runs
  `decompress(compress(obj)) == obj`. Token availability changes *which* values get
  aliased (a performance choice) but never correctness. The test suite is a
  parametrized battery of this gate.
- **Representation transform, not offload.** The compressed form is valid input the
  model reads directly (a table, an inline legend). There is no `retrieve` step.
  This is the core difference from lossy-offload approaches.
- **Row-major tables.** Tabularization keeps rows as rows (positional cells mapped to
  a `cols` header), including nested `subcols`, so the table stays as legible as
  CSV/markdown — the model already does position→header mapping for the outer table.
- **Aliases are collision-proof.** Dictionary aliases come from a `~`-sigil namespace
  that is checked disjoint from every literal string (keys and values) in the
  payload, so decode is an exact legend lookup with no ambiguity.
- **Selective, fail-closed policy.** Value is per-tool, so a policy gates which tiers
  run. An unmatched tool gets the lossless default and never a lossy op.
- **Determinism.** No clock/random in the transform path; same input → same output.

## File Descriptions

- **`transforms.py`** — the lossless core.
  - `minify` / `compress_structure` (+ `_fold_records`) / `dict_encode` and their
    exact inverses (`decompress_structure` + `_unfold_row`, `dict_decode`).
  - `compress_with(obj, tabularize, dictionary)` applies a selectable subset of
    tiers; `compress`/`decompress`/`roundtrip_ok` are the full pipeline + gate.
  - Markers: `TABLE_MARKER`, `DICT_MARKER`, `ALIAS_SIGIL` (`~`).
  - Depends on `tokenize.count_cl100k` for the tokenizer-aware aliasing threshold.
- **`policy.py`** — `Rule`/`Policy` dataclasses, `load_policy` (JSON parse + validate),
  `default_policy`, `Policy.select` (first tool-glob match wins), and `apply()` which
  returns an `Applied` record (text, tiers run, skipped, warnings). The only module
  that knows about lossy field modes (which it parses and warns about, never executes).
- **`capture.py`** — `classify_shape` (pretty/compact JSON, array-of-records,
  long-text), `capture_payload` (writes a sha-idempotent envelope to `corpus/`),
  `load_corpus`, `coverage`, `extract_records`.
- **`measure.py`** — `measure_payload` (per-tier cl100k decomposition: `minify +
  tabularize + dictionary == tier_total`, re-runs the gate), `measure_corpus`, and
  `cross_tokenizer_savings` (cl100k vs o200k invariance).
- **`probes.py`** — `value_redundancy` and `cross_call_overlap`: upper-bound
  estimators for whether higher-ceiling levers (dictionary, cross-call diffing) are
  worth building. They measure, they do not compress.
- **`tokenize.py`** — `count(text, encoding)` over named tiktoken vocabs (cl100k,
  o200k), `encode_cl100k` (token ids for probes), `count_anthropic` (optional, needs
  a key).
- **`report.py`** — markdown renderers: `build_report` (savings by shape + per-tool +
  tier attribution + coverage + gate banner), `build_probe_report`,
  `build_tokenizer_report`.
- **`cli.py`** — argparse dispatch for the six subcommands.

## API Integrations

- **tiktoken (local)** — token counting under `cl100k_base` and `o200k_base`. No
  network at runtime after the one-time vocab download. Used for measurement and for
  the dictionary coder's cost-aware aliasing threshold.
- **Anthropic `count_tokens` (optional)** — `count_anthropic` calls the Messages
  `count_tokens` endpoint if the `anthropic` extra is installed and a key is present.
  There is **no public local tokenizer for Claude 3+**, so this is the only way to
  get true Claude token counts; it is off by default. Sending a payload to this
  endpoint transmits it to Anthropic — run it on public data only.
- No other external services. terse does not call any tool APIs itself; it compresses
  output that is piped or passed to it.

## Configuration

### Policy file (JSON)

```jsonc
{
  "version": 1,                       // only 1 is supported; anything else is rejected
  "defaults": { "tiers": ["minify", "tabularize", "dictionary"] },
  "policies": [
    {
      "match": { "tool": "gh.*" },    // fnmatch glob on the tool name
      "tiers": ["minify", "tabularize", "dictionary"],   // [] = passthrough (skip)
      "fields": {                     // optional, per-field
        "result[].id":   { "critical": true },           // honored trivially in v1 (lossless)
        "result[].body": { "lossy": "drop-to-retrieve" } // parsed, WARNED, not executed in v1
      }
    }
  ]
}
```

- **Matching:** rules are evaluated in order; the first whose `match.tool` glob
  matches wins. No match → `defaults.tiers`.
- **Tiers:** any subset of `minify` / `tabularize` / `dictionary`. `minify` is implied
  by serialization (a warning is emitted if omitted). `[]` = passthrough.
- **`critical`:** a denylist against lossy ops. In v1 (lossless only) it is honored
  automatically since nothing is dropped.
- **`lossy`:** `truncate` / `summarize` / `drop-to-retrieve`. Accepted by the schema
  for forward-compatibility; v1 emits a warning and leaves the field lossless.
- Validation: unknown tiers and unsupported versions raise `ValueError` at load time.

`policy.example.json` ships a policy that encodes the measured insight (gh/runecho
full tiers, kb drops dictionary, `*.rate_limit` skipped).

### Environment

- `ANTHROPIC_API_KEY` — only read by `count_anthropic` / `terse measure --anthropic`.
  Absent by default; everything else runs without it.

## Deployment

terse is a library + CLI, not a service.

```bash
uv sync                       # create the venv, install pinned deps (uv.lock)
uv run terse <subcommand>     # run the CLI
```

To use it as a library, import `terse.policy.apply` / `terse.transforms.compress`.
A future MCP proxy would call `policy.apply(raw, tool, policy)` on each tool result
(not yet built). There is no remote infrastructure and nothing to roll back; reverting
is `git checkout` of a prior commit.

## Maintenance Commands

```bash
uv run pytest                 # full suite (the lossless gate + measurement + policy)
uv run pytest -q tests/test_roundtrip.py   # just the lossless gate
uv run terse measure          # token savings report over corpus/ -> reports/
uv run terse probe            # ceiling probes (value redundancy, cross-call overlap)
uv run terse validate         # cross-tokenizer invariance (cl100k vs o200k)
```

Reports are written under `reports/` (gitignored). The corpus under `corpus/` is
gitignored because captured tool output may contain real data.

## Known Limitations

- **Tier 1 (lossy) is not built.** `truncate` / `summarize` / `drop-to-retrieve` are
  in the policy schema but warned-and-skipped. Today terse is 100% lossless regardless
  of policy.
- **No MCP proxy yet.** The selective library + CLI exists; the thin proxy that would
  intercept live tool results and call `policy.apply()` is designed but unbuilt.
- **Whole-subtree aliasing and cross-call diffing are unbuilt.** Probes show headroom
  (repeated whole objects; 91% overlap between successive same-tool calls) but the
  coders for them do not exist yet.
- **Marker collision.** A payload that genuinely contains a top-level
  `__terse_table__` / `__terse_dict__` key, or whose strings exhaust the entire
  `~`-alias namespace, is a theoretical edge not specially handled. Real tool output
  does not contain these.
- **Dictionary coding trades some direct legibility for tokens.** A `~0` reference is
  resolved by reading the inline legend in the same payload (no retrieve step), but it
  is less immediately readable than the literal value. It is gated on real token
  savings, so it only fires where it pays.
- **No true Claude token counts without a key.** All shipped numbers use cl100k;
  robustness is established via cross-tokenizer invariance (cl100k vs o200k, ~0.5 pt
  mean divergence), not a direct Claude measurement.
- **Shape classifier is shallow.** `classify_shape` inspects one level; deeply-nested
  record structures may bucket as `compact-json` even though terse folds them. The
  per-tool report exists precisely because shape buckets can mislead.
