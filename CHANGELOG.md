# Changelog

All notable changes to terse are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from git tags (`vX.Y.Z`, via hatch-vcs) — an entry moves from
`[Unreleased]` to a versioned section when its tag is pushed.

## [Unreleased]

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
