# Verify terse yourself

terse sits in the critical path of your agent's tool calls, so it should earn trust by
inspection — not by claims. This walks through verifying both questions that matter:

- **Is it safe?** — does it ever corrupt, drop, or leak a tool result?
- **Is it worth it?** — how many tokens does it actually save on *your* traffic?

Everything below runs locally. terse has no telemetry and the proxy makes no network
calls (see step 4).

## 0. Get it

```bash
git clone https://github.com/inth3shadows/terse.git && cd terse
uv sync            # or: python -m venv .venv && . .venv/bin/activate && pip install -e .
```

## 1. Run the correctness suite (the load-bearing safety check)

The losslessness guarantee *is* the test suite — round-trip, diff, and proxy tests.

```bash
uv run pytest -q          # expect: all green; also runs in CI on Python 3.11–3.13
```

## 2. Round-trip YOUR own data

`terse gate` compresses a payload and checks it reconstructs byte-for-byte. Exit code is
the gate: `0` = lossless, `1` = not. Feed it real output from a tool you care about:

```bash
uv run terse gate path/to/your-tool-output.json
# round-trip lossless: PASS
# cl100k tokens:       <before> -> <after>  (N% saved)
```

## 3. One-command verification report

`terse verify` emits a single self-contained markdown report: the **lossless gate**
(which voids the savings if any payload fails) plus **per-shape / per-tool token
savings**, with a header stating exactly what it does and does not prove.

```bash
# zero-setup: runs on a bundled deterministic sample
uv run terse verify --out reports/verify-report.md

# on YOUR traffic: capture real tool outputs first, then point verify at them
uv run terse capture --tool runecho.structure path/to/output.json --corpus corpus
uv run terse verify --corpus corpus --out reports/verify-report.md
```

Or capture your *live* traffic automatically: run the proxy with `--capture-dir`, use
your agent normally, then verify what it actually saw — no per-payload `capture` step.

```bash
uv run terse proxy --capture-dir corpus --policy policy.json -- some-mcp-server
uv run terse verify --corpus corpus --out reports/verify-report.md
```

`--capture-dir` tees only the *raw* (pre-compression) payloads and is a pure side
effect — a capture failure never changes what your agent receives.

The bundled sample proves the *mechanism* (lossless + the transforms firing); your own
captured corpus proves the *savings on your workload* — terse's win is shape-dependent
(large on record/symbol-shaped output, ~0% on already-compact or single-object tools),
so the per-tool table is the number that matters for you.

## 4. Confirm no egress, and read the fail-open path

```bash
# the only network code is fluency.py (an explicit, opt-in model eval the proxy
# never calls); the proxy is stdio-only and persists nothing
grep -rnE "requests|urllib|socket" src/terse        # → only src/terse/fluency.py
```

Then read `src/terse/proxy.py` (~300 lines): every `tools/call` result is run through
`policy.apply`, and **any** parse/compress error forwards the original message unchanged
(fail-open). Non-`tools/call` messages are forwarded byte-for-byte.

## What terse does and doesn't keep

- **No database, no runtime state on disk.** The only persisted artifacts are ones *you*
  create: `corpus/` (from `terse capture`) and `reports/` (from `measure`/`verify`), plus
  the installer's `~/.terse-mcp-stash.json` config backup.
- Cross-call diffing (opt-in, `--diff`) keeps the previous result of a tool **in memory
  only**, to emit a delta; it is lost on restart and written nowhere.
