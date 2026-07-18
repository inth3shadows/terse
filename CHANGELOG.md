# Changelog

All notable changes to terse are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from git tags (`vX.Y.Z`, via hatch-vcs) — an entry moves from
`[Unreleased]` to a versioned section when its tag is pushed.

## [Unreleased]

### Added
- `verify --json`: emit the lossless-gate verdict and cl100k savings totals as JSON
  on stdout instead of the markdown report — CI-checkable (`… | jq -e
  .lossless_gate.ok`), parity with `stats --json` / `mcp-status --json`.

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
