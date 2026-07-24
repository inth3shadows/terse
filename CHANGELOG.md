# Changelog

All notable changes to terse are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from git tags (`vX.Y.Z`, via hatch-vcs) — an entry moves from
`[Unreleased]` to a versioned section when its tag is pushed.

## [Unreleased]

### Fixed
- **A capture/audit/stats sink that HUNG — rather than raised — froze every later tool
  call on the connection.** The sinks were invoked inside `transform_response`'s
  `_local_lock`, and the fail-open `try/except` around them only ever caught a sink that
  raised. A sink that blocks (full disk mid-retry, stalled network mount, slow fsync) held
  the lock indefinitely; `note_request` takes that same lock, so the next `tools/call`
  wedged behind it and never recovered. Sink calls are now queued under the lock and
  invoked after it is released — the reply is already decided by then, so a slow sink
  delays only its own response instead of the whole connection. This makes the documented
  contract ("a sink failure or slowness never affects forwarding") true for *slowness*,
  which it previously was not. Pinned by
  `test_blocking_sink_does_not_stall_a_concurrent_note_request`, which times out against
  the old code.

### Added
- **The corpus is bounded per tool (`MAX_SAMPLES_PER_TOOL`, default 200).** `capture.py`
  was the only disk sink with no retention — `stats.py` rotates at 10 MB and `history.py`
  at 5 MB, but envelopes accumulated forever. Since envelopes hold *raw* tool payloads
  (credentials, PII, private source), unbounded retention widened the blast radius of any
  later disk compromise as much as it risked disk exhaustion. The cap is per tool, not
  global: every consumer (measure, probes, `policy generate`/`autotune`) reasons per tool,
  so a global byte cap would let one chatty tool evict the only samples a quiet one ever
  produced and silently narrow what a generated policy can see. Eviction is oldest-first
  by mtime. `capture_payload(..., max_per_tool=None)` restores unlimited retention for a
  deliberate one-shot corpus build.
- **Coverage instrumentation** (`pytest-cov`, `[tool.coverage.*]`). The suite had no
  coverage number at all; the first measured run is **89% branch coverage** over
  `src/terse`. Reported, not gated — `--cov` is opt-in per run so the default `pytest`
  stays fast, and no threshold is set until the baseline has been looked at.

### Fixed
- **The capture envelope recorded neither which result nor which server a payload came
  from, so autotune had to guess both (#148, #152).** Two defects, one absent pair of
  fields, now written by the proxy (`server`, `result_id`) and read at tune time:
  - *Results were reconstructed from capture timing.* Consecutive envelopes within 50 ms
    were taken to be one result. A burst of independent parallel calls has no gap between
    it, so 200 separate single-block results chained into ONE 200-block group and scored
    **63.4% saved with `dictionary` enabled** where the truth — each scored alone, which is
    what the proxy does — is **25.0% and no dictionary**. Grouping is now exact wherever
    result ids are present. Corpora captured before them keep the heuristic, which also
    gained a total-span cap so an unbroken run can no longer chain without bound, and
    `policy generate`/`autotune` now say how many payloads were grouped that way rather
    than presenting a guessed number as a measured one.
  - *A generated rule could be unreachable.* `select` tries the `{server}.{tool}`
    candidate against **every** rule before the bare name, so with `runecho.*` deployed a
    corpus-derived `structure` rule is dead on arrival — position cannot save it, only the
    qualified name can. Generated rules are now authored under the same name the runtime
    looks them up by. On the live 1663-payload corpus this was ~35 shadow rules in the
    autotune diff, all of which a human had to hand-filter.
  - *The merge's shadow check resolves on the `(bare tool, server)` pair, candidate-major.*
    Naming the rule is only half of it: the check has to find the rule the LOADER would.
    Both a deployed `runecho.*` and a deployed bare `structure` govern a tool captured from
    runecho, and either one's operator-owned keys must be inherited — otherwise autotune
    hands a tool from a `capture: false` rule to a fresh one with capture ON, silently
    reversing the #85 decision and reporting it as a benign "(new rule)". Candidate-major
    also matters when both are deployed: rule-major picks `structure`, the loader picks
    `runecho.*`, and inheriting the wrong rule's keys is worse than inheriting none.
  - *A corpus spanning the upgrade no longer splits one tool in two.* A payload with no
    server is folded into the single server observed for that same bare tool; two servers
    for one bare name is genuinely ambiguous and stays unattributed rather than guessed.
    Without this, the half captured before the upgrade is measured on half the sample —
    and is dead at runtime besides.
  - *`tune --drop-eval` looks its rule up the way the proxy does.* It resolved by bare tool
    name, which on a server-tagged corpus falls through to the defaults, finds no `fields`,
    and scores **nothing** while still reporting that it verified the suggested drops — the
    #149 failure mode with one lookup removed.
  Both fields are optional and omitted when unknown, so an existing corpus stays loadable
  and needs no migration; they are preserved together on an idempotent rewrite, since a
  first sighting's timestamp beside a later sighting's result id would place one block in
  two calls at once. Result ids are scoped by proxy process *and* by handshake generation,
  because a reconnecting client restarts its JSON-RPC ids at 1. `terse capture` gained
  `--server` for the hand-captured case.
- **`policy generate` scored payloads per-BLOCK, so every multi-block tool was
  under-measured (#147).** The proxy compresses a multi-block result as one joined record
  array (#116); the generator scored each captured block alone. For a server that returns
  one record per content block — common — those are wildly different numbers: measured on
  real kb traffic, `changelog` is 23.3% per-block and **48.4% joined**. Payloads are now
  grouped back into results (by capture-time proximity) and each result is scored the way
  the proxy would, falling back to per-block exactly where `apply_joined` would refuse.
- **One non-JSON payload no longer disqualifies a whole tool (#147).** A single
  `Error executing tool …` text block among a tool's records forced `passthrough` for all
  of it. The premise was wrong — `policy.apply` passes a non-JSON payload through untouched
  at runtime, so the tier costs nothing on those results. On a real corpus this alone was
  zeroing the highest-volume tool in the fleet: `kb.read.search` measured **16.7% saved**
  and was written as passthrough because 4 of its 436 payloads were error text. A
  mostly-text tool is still suppressed, now for the right reason — non-JSON contributes 0
  saved while its raw tokens stay in the denominator, so it falls below the threshold on
  its own (`codegraph_explore`, 61/61 non-JSON, scores 0.0%).

### Added
- **`terse policy autotune` — re-tune an EXISTING policy instead of overwriting it (#136).**
  `policy generate` authors from nothing and is *total*: run it on a deployed policy and it
  silently drops every decision the corpus cannot see. It already warned about that for
  `capture: false`; the same was true of `never_lossy_servers`, any `structured` override,
  hand-written active `fields`, any rule for a tool the corpus never saw, and rule ORDER
  (first match wins). `autotune` merges instead, split by what a corpus can possibly know:
  **the corpus decides `tiers`** (including removing one — the motivating case is a stale
  tier decision that predates a codec change), **the operator owns everything else**. It
  prints a per-rule diff, names what it deliberately did not regenerate, and writes
  **nothing** without `--apply`. New rules are inserted before any existing glob that would
  shadow them, since a `kb.read.search` rule appended after `kb.*` is dead on arrival.
  A new rule **inherits the operator-owned keys of whatever rule it displaces** — inserting
  it ahead of a broader rule must not quietly hand that tool `capture: true` or
  `structured: "auto"` — and a rule whose `tiers: []` is suppressing a lossy `$text.*`
  selector keeps `tiers: []`, because turning them on would ACTIVATE that selector and this
  merge is documented as lossless. Warns before applying a tier *downgrade*: the corpus is
  a sample (idempotent by sha, and empty for a `capture: false` tool), so a removal should
  be cross-checked against `terse stats`, which counts every call.

### Added
- **`"structured": "compress"` — compress `structuredContent` too (#128).** New per-rule
  policy knob. MCP 2025-06-18 lets a tool return a typed `structuredContent` field beside
  a text block that mirrors it; terse compressed only the block. Measured against `claude`
  2.1.218 with a read-only proxy, the client forwards the **typed field** to the model and
  discards the block entirely — so on such a tool terse was delivering ~0% however good
  the ledger looked. With the knob on, the same fixture measures **61.2%** of the model's
  real context (2,596 → 1,008 chars), captured end to end rather than inferred.
  Affected servers are not exotic: filesystem (14/14 tools), memory (9/9) and kb (27/27)
  all declare an `outputSchema`.
  Codec only, no diff. See the `structured: "auto"` entry under **Changed** for how the
  default now decides this per connected client.
- **`"structured": "replace"` — drop the redundant text mirror (#128), and the measurement
  saying you probably shouldn't.** Compresses the typed field *and* deletes the text block
  that duplicates it. Measured on the reference fixture: context cost goes 2,596 → 1,008
  chars under `"compress"` and **1,008 → 1,008** under `"replace"` — no change, because
  Claude Code had already discarded the block. What it removes is stdio bytes, not context.
  Shipped as an explicit opt-in that `"auto"` never selects, because it is correct for a
  client that forwards *both* fields (which `"compress"` can leave holding a cross-call
  diff in the block contradicting a full envelope in the typed field); no such client has
  been measured. Five independently-tested guards must all hold before a block is dropped
  — explicit `"replace"`, non-empty `tiers`, not an `isError`, exactly one text block, and
  that block's parsed JSON **equal** to `structuredContent`. Whether the tool declared an
  `outputSchema` is deliberately *not* a guard: the new `noschema` probe shows the client
  reads the typed field from a tool that declares none.

### Fixed
- **The savings ledger no longer reports a saving terse did not deliver (#128).** terse
  compresses a result's text block but leaves `structuredContent` untouched, and the
  ledger counted only the block — so a tool emitting both was credited with the block's
  full reduction while the untouched duplicate rode along at full size. `build_record`
  now counts that duplicate on *both* sides, making `raw_chars`/`out_chars` the whole
  result's cost, and records the split as `structured_chars`/`structured_tokens`. On the
  reference fixture the same call now reports **33.9%** where it previously reported
  58.7%. `decision` is unchanged — it names what terse did to the block, and terse did
  compress it. Records predating the field had no duplicate, so a missing value reads
  as 0. Measured against a live client, the honest figure may be lower still: see
  `scripts/probe/structured_content/`, which found that `claude` 2.1.218 reads
  `structuredContent` and discards the compressed block entirely.
- **A broken capture/stats/audit sink now says so, instead of failing silently.** The
  callbacks handed to the `Interceptor` caught their own exceptions behind a `--debug`
  gate, so the `try/except → _warn_sink` around them never saw one and `_warn_sink`'s
  unconditional first-failure warning was dead code. A `--capture-dir` pointing at a
  regular file, or a `--stats-log` pointing at a directory, produced a completely
  normal-looking run — every tool call answered, exit 0 — with zero payloads captured,
  no ledger written, and nothing on stderr; a later `terse measure --corpus` then
  reported a percentage over whatever subset happened to land. The callbacks now own
  I/O only and let failures propagate to the single caller that has the per-sink
  bookkeeping. The fail-open contract is unchanged: a sink failure still never changes
  what the client receives — it is now merely *audible*: once per sink kind, and under
  `multiproxy` (where the `Interceptor` and its bookkeeping are per-peer) once per
  peer, so a dead shared sink is attributed to each downstream that hit it.
- **A server-initiated request no longer silently disables compression for an in-flight
  call.** A server→client request (`roots/list`, `sampling/createMessage`,
  `elicitation/create`) carries a `method` alongside an id, and JSON-RPC gives each
  direction its own id space — both sides conventionally numbering from 1 — so such an id
  routinely collides with an in-flight `tools/call` id. `transform_response` popped
  `pending[id]` unconditionally (deliberately, so an error-shaped reply still frees its
  entry), which also consumed the entry for a server request; the real tool result then
  arrived untracked and was forwarded **uncompressed and absent from the savings ledger**,
  with nothing logged to say so. The `initialize` path had the same exposure — a colliding
  server request consumed `init_id`, so the real reply never received the terse primer.
  Method-bearing messages are now forwarded untouched, using the same predicate
  `multiproxy` already applied one layer up.

### Added
- **Cross-block join (`join_blocks`, ON by default) — #116.** When every text content
  block of a tool result is a JSON object, the proxy now joins them into one record array
  before compressing, so `tabularize`/`dictionary` fold across records *and* the whole
  result becomes eligible for the cross-call diff tier. Several MCP servers return one
  record per block, a shape that was 71% of terse's own live traffic and could reach
  neither cross-record folding nor diffing (the diff path only ran for single-block
  results). Measured on a realistic 80-record `kb.read.list_principles` payload: per-block
  +9.6% → joined codec +24.9%, and a near-identical repeat call collapses ~6900 tokens to
  ~100 via a diff. Lossy field rules resolve **per block, before the join**, so a path
  authored against one record's shape is unaffected. Opt out with `proxy --no-join-blocks`
  / `install-mcp --no-join-blocks` or a policy-file `"join_blocks": false`.

### Changed
- **`structured` now defaults to `"auto"`, which decides per connected client (#128).**
  This **changes default wire behavior for Claude Code users**: `structuredContent` will
  carry a terse envelope where it previously carried the server's own object. That is the
  intended effect — with the previous `"leave"` default terse was a measured no-op on
  filesystem, memory and kb — but it is a real behavior change, stated here rather than
  buried under Added.
  The `"leave"` default shipped alongside the knob rested on "terse cannot detect which
  client it sits behind." That was wrong: the MCP `initialize` request carries
  `params.clientInfo`, a name the client *declares*, and the proxy proxies that request.
  `"auto"` compresses the typed field only for clients measured not to validate it
  (`policy.STRUCTURED_SAFE_CLIENTS`, currently `claude-code`, evidenced by the `badtype`
  and `enveloped` probes) and **fails closed** for an unlisted client, a client that omits
  `clientInfo`, and a library caller that never handshakes. Explicit `"compress"`/`"leave"`
  still win. Measured with a stock policy — no `structured` key anywhere — the fixture's
  context cost drops 2,596 → 1,008 chars (61.2%) against `claude-code`, and is untouched
  against anything else.
- **A joined result changes the content-block count the client sees (N → 1).** This is the
  first time terse changes anything but block *text*. The MCP spec (2025-06-18) puts no
  meaning on block count — blocks carry no index a payload can reference — and non-text
  blocks (image/audio/resource) keep their positions. The savings ledger's blanket
  `multiblock` reason is replaced by reasons that name why a join did or didn't fire
  (`multiblock_non_json` / `_heterogeneous` / `_marker` / `_depth` / `_passthrough` /
  `_off`, plus a `reanchor` reason when a join↔single shape flip forces a full).

## [0.4.1] - 2026-07-21

### Fixed
- **`install-mcp` no longer writes a launcher path that can never resolve.** A wrapped
  entry is spawned from JSON via `execve` with no shell, so a quoted
  `TERSE_MCP_CMD='~/.local/bin/terse'` wrote a literal tilde and the entry silently
  failed to start. The override's `argv[0]` is now `expanduser`ed, and a path that does
  not exist is rejected at install time before the config is touched — the same
  treatment `--policy` already got. A bare name (`terse`) still passes through, since it
  resolves against the launcher's `PATH`.
- **`mcp-status` flags a wrapped entry whose launcher stopped resolving.** This is the
  failure mode an upgrade causes when a versioned `uv tool`/`pipx` venv moves out from
  under every wrapped entry at once, and it was invisible everywhere: the client cannot
  spawn the proxy, so the server just appears with no tools. New `launcher` /
  `launcher_missing` fields in `mcp-status --json`.

### Added
- `$TERSE_MCP_CMD` is documented in `USAGE.md` (it previously existed only in a
  docstring) and now has test coverage, along with the two `install-mcp` footguns that
  only surface after an upgrade or an uninstall.

## [0.4.0] - 2026-07-21

### Fixed
- **`$`-prefixed JSON keys are drop-eligible again.** Reserving the whole `$` sigil for
  text selectors silently disabled `drop-to-retrieve` on ordinary JSON keys like
  `$schema`/`$ref`/`$id` (every JSON Schema payload has them). Only the `$text.` prefix
  is reserved now. Regression introduced with the text selector, caught in review before
  any release carried it.
- **A known text selector carrying an unsupported mode now warns** instead of doing
  nothing silently — `{"$text.code_blocks": {"lossy": "truncate"}}` was the one config
  that failed with no signal at all.
- **Fence scanning follows CommonMark 4.5**: a backtick fence's info string may not
  contain backticks. Permitting them let an inline-code prose line (```` ```py``` ````)
  open a phantom fence, so a prose region was evicted as if it were source. The recovery
  gate could not catch this — it proves a span is restorable, never that it was code.
- **`isError` tool results are never evicted to a handle.** An error is what the model
  must read to recover; a lossy transform must not put a retrieve round-trip in front of
  it. Added a per-result `force_lossless` override, the response-level twin of the
  never-lossy server floor.

### Added
- **`drop-to-retrieve` for non-JSON payloads, addressed by span.** A policy field can now
  name `"$text.code_blocks"`, which evicts each fenced code block over `min` chars from a
  long-text tool result to a `terse.retrieve` handle while leaving the surrounding prose
  resident. This reaches a payload class the lossless codec structurally cannot help with:
  measured over 60 real captured `codegraph_explore` results, 89.2% of their tokens were
  fenced source and terse saved **0.0%**; with the selector enabled the same corpus drops
  **87.0%** of its tokens, with byte-exact restore verified on all 57 transformed payloads
  and zero gate failures. Opt-in and off by default — no existing policy changes behavior.
  The gate is stronger than its JSON sibling's: rather than proving only marked paths
  changed, it reconstructs the entire payload from the emitted text plus the session store
  and requires byte-for-byte equality. Suppressed on never-lossy servers, by `critical`,
  and by `"tiers": []`, each with an explicit warning rather than silence. The behavioral
  fluency harness (`dropeval`) gained the matching text recall/precision questions.

### Changed
- Docs: install instructions now lead with `pip install terse-mcp` from PyPI rather than a
  git clone (#113), and the positioning is corrected — the always-on lossless codec is the
  core value, with cross-call diffing a bonus tier layered on top (#114).

## [0.3.1] - 2026-07-18

### Added
- Automated PyPI publishing via GitHub Actions **Trusted Publishing** (OIDC) with
  PEP 740 provenance attestations — tagged releases upload to PyPI with no stored
  token. This is the first PyPI release of `terse-mcp`. (#111)

### Fixed
- `release` workflow: corrected the built-wheel name check for the `terse-mcp`
  rename (`terse_mcp-*.whl`), which had blocked the tagged release run. (#110)

## [0.3.0] - 2026-07-18

### Added
- **Installable package** — MIT license and PyPI-ready metadata. The distribution is
  named `terse-mcp` (the bare `terse` is taken on PyPI); the import package stays
  `terse`, so `python -m terse` is unchanged. (#103)
- `verify --json`: emit the lossless-gate verdict and cl100k savings totals as JSON
  on stdout instead of the markdown report — CI-checkable (`… | jq -e
  .lossless_gate.ok`), parity with `stats --json` / `mcp-status --json`. (#107)
- diff-moat instrumentation: `stats` records why the cross-call diff tier did or did
  not fire per call, to measure the diff feature's real-world reach (Phase 0+1). (#101)
- Property-based fuzzing of the lossless round-trip guarantee. (#105)

### Documentation
- Onboarding quickstart + per-client install recipes. (#108)
- Codeshot architecture diagram embedded in TECHNICAL.md. (#104)

## [0.2.0] - 2026-07-17

### Added
- `mcp-status`: each wrapped server now shows what it actually fronts
  (`wraps=<downstream cmd/url>`), whether the cross-call diff tier is
  on/off/default, and whether the stats ledger is on; a policy file that has gone
  missing since install is flagged `(MISSING)`; new `--json` output for
  scripts/CI. (#97)
- `tune`: each drop-candidate bucket ends with a savings rollup — estimated gross
  tokens the whole bucket would evict and its share of the corpus. (#98)
- `fluency --html`: writes the forest-plot comprehension-gap report next to
  `--out` for the paired diff-family evals (`--diff`, `--diff-soak`,
  `--text-diff-eval`); same inline-SVG/no-JS/no-CDN form as `measure --html`. (#99)

### Fixed
- `stats`: the per-tool table falls back to character columns when tiktoken token
  counts are absent (was rendering as all-zeros); an empty `--since` window now
  reports the window rather than "nothing ever recorded"; added a per-tool
  cross-call diff hit-rate (`diff%`) column. (#96)
