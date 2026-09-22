# terse

[![tests](https://github.com/inth3shadows/terse/actions/workflows/tests.yml/badge.svg)](https://github.com/inth3shadows/terse/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/terse-mcp.svg)](https://pypi.org/project/terse-mcp/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![terse MCP server](https://glama.ai/mcp/servers/inth3shadows/terse/badges/score.svg)](https://glama.ai/mcp/servers/inth3shadows/terse)

The **lossless-first** MCP compression proxy: it makes tool output smaller without
ever changing what your agent reads — **lossless by value** by default (what decodes back
out is the same JSON, value for value), lossy only where you explicitly opt in.

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
v0.34.0; default 30-min TTL); Anthropic/OpenAI context-editing **drops** old tool results
server-side. terse never silently mutates what the model sees — and "lossless" is the
category, not the token count. In terse's own production ledger this codec is where
essentially all the savings come from (see Status). Its one honest caveat: the
tabularization primitive is public (formats like [TOON](https://toonformat.dev/) publish
it standalone, MIT-licensed, ~40% on flat arrays), so a motivated competitor could clone
the codec in a weekend.

**2. The stateful cross-call diff — the axis nobody else can reach. It does not pay as
shipped, but the reason is not what it looks like.** When the same tool is called again —
poll a list, re-read a file — terse emits
a lossless *delta* against the prior result instead of the whole payload (**~73% smaller on
the repeated call** in the model below). This is the one axis a stateless encoder
**architecturally cannot reach**: TOON, headroom's stateless per-call compressor, and
server-side history-pruning all pay the full column every call because none of them
remember the last result — terse can only do it because it lives in the session as a
transparent proxy.

That is an architectural claim, and it survives. The economic one does not, at this
workload.

**Re-derived 2026-09-22 over the full all-time ledger (4,461 blocks), superseding the
14-day #250 window that reported 1.9% / 3.7%:**

```
decisions:    compressed=3356  passthrough=659  unchanged=416  diff=30
diff reasons: diff_off=1387  joined=587  multiblock=444  no_prior=318
              not_smaller_diff_args=251  emitted=26  not_smaller_same_args=4
```

The tier fires on **0.67% of blocks** (30 of 4,461; 26 emitted diffs), while the live
router's primer costs **502 tokens every turn** — re-read as `cache_read` on each one.
`diff_off=1387` is the largest single reason and is *expected*: #170 made diffing opt-in
precisely because the primer paragraph costs more per turn than this hit rate returns.

**But segment the attempts by whether the diff BASE had matching arguments, and the
picture inverts (#419):**

```
base args MATCH      emitted 26   not_smaller_same_args   4    ->  87% win rate
base args DIFFER     emitted  0   not_smaller_diff_args 251    ->   0% win rate
never re-called      no_prior 318
```

**When the diff gets a same-args base it wins 87% of the time. It almost never gets one.**
`proxy.py:345` keeps **one base slot per tool** (`self.last: dict[str, Any]  # tool ->
previous result object`), so any interleaved call to that tool with different arguments
evicts the base that would have paid. 251 attempts diffed against the wrong base.

So the tier is **starved rather than weak**, and an earlier edition of this section — which
called the remaining exclusions "the workload" and said "no codec change reaches that" — is
**withdrawn**. `no_prior` (318) is genuinely workload; the 251 are a data-structure choice.

**But most of those 251 are not recoverable, and the probe says so.** Classifying them by
whether that tool's arguments can recur at all within a session:

```
115 (45.5%)  WRITE     kb.propose.merge_or_create / extend / delete — payload differs every call
 60 (23.7%)  QUERY     kb.read.search — a different query string nearly every time
 69 (27.3%)  RE-READ   runecho structure (36), kb.read.get (32) — args CAN recur
  9 ( 3.6%)  other
```

Arg-keying moves the writes and queries from `not_smaller_diff_args` to `no_prior` — the
same non-win, relabelled. **Only ~69 are addressable**, worth at most ~60 extra diffs at the
measured 87%: roughly 26 → 86, a **~3.3x on a tier that fires on 0.67% of blocks**, which
still has to clear the 502 tok/turn primer. Real and cheap, but small. #419 carries it at
that reduced size. The drop tier is where the measured tokens are (#252, #271, #273):
`codegraph_explore`'s single rule accounts for 604,892 saved tokens here, 22.2% of all
savings.

It remains **opt-in and not the headline** (#170). When a loop really does re-fetch
mostly-unchanged results it compounds hard; short sessions with wide tool variety never
exercise it. How often *your* loop repeats a call is yours to measure (`terse stats`).

Around those two sits the **bundle** that turns a byte filter into a control plane you
don't want to rip out: MCP-native proxy packaging (transparent to any downstream
server, no client-side reformatting), a **live savings ledger** (`terse stats`), a
fluency-gated lossy escape hatch, and self-installing ops tooling (`install-mcp`,
`mcp-status`) — each diff/lossy tier validated by a behavioral eval before it was ever
shipped.

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
- **Tier 0.7 — cross-call diff (stateful, OPT-IN — `"diff": true`)**: when the same tool is called
  repeatedly, the proxy emits a lossless delta against the prior result instead of
  the full payload (the 91%-overlap headroom). Self-describing, verified to reconstruct
  exactly, and emitted only when smaller — falls back to the full form otherwise.
  Its validation program completed (fluency, nested-record coverage, and the drift
  soak — see Status), but the tier stays OFF by default since #170; opt in with
  `proxy --diff` / `install-mcp --diff` or a policy-file `"diff": true`.
  Record-shaped JSON gets a row/key diff; non-JSON results (file reads, source excerpts,
  log tails) get a separate content-defined-chunking (CDC) diff — a rolling hash cuts
  chunk boundaries by content, not position, so an edit anywhere only perturbs the
  chunk(s) it overlaps and the rest is sent as references to the prior result. Each
  shape keeps its own diff base per tool.
- **Cross-block join (ON by default)**: some MCP servers return one record per content
  block, so each block is a lone object the codec above can barely fold and the diff tier
  skips entirely (it reasons about one logical payload). When every text block of a result
  is a JSON object, the proxy joins them into one record array before compressing — so
  `tabularize`/`dictionary` fold across records *and* the whole result becomes
  diff-eligible. This changes the number of content blocks the client sees (N → 1), which
  the MCP spec permits (block count carries no meaning). Opt out with
  `proxy --no-join-blocks` / `install-mcp --no-join-blocks` or a policy-file
  `"join_blocks": false`; lossy field rules still resolve per block, before the join.
- **Tier 1 — lossy (opt-in, per field)**: `truncate` caps and annotates a field marked
  `{"lossy":"truncate","max":N}`, gated by an acceptable-loss check (only marked,
  non-`critical` fields may differ, each only as a valid truncation). `drop-to-retrieve`
  replaces a marked field with a handle, stores the original per session, and serves it
  back via a synthetic `terse.retrieve` tool the proxy injects — gated so a drop is
  accepted only if the handle resolves to the exact original. `summarize` (needs a model)
  is parsed but **will not be built** — it is closed permanently (#261), because a
  model in the proxy breaks terse's no-ML guarantee. It is warned and left lossless,
  and the schema keeps accepting it only so an existing policy does not fail to load.
  Off everywhere by default.

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

### Docker

```bash
docker build -t terse .
docker run -i --rm terse                      # proxies the bundled demo server
docker run -i --rm terse uvx some-mcp-server  # proxies a real one
```

`-i` is required — MCP stdio *is* stdin/stdout. Everything after the image name is the
downstream server command; terse has no tools of its own, so with no downstream there is
nothing to compress. The default is `examples/demo_mcp_server.py`, a stdlib-only server
whose `demo_orders` tool returns a 40-record order book — you get back terse's compressed
form of it (9,019 → 3,187 chars), which is the fastest way to see what the proxy does.

The image ships `uv`/`uvx` so `uvx`-launched servers work out of the box. Node is **not**
installed — for an `npx`-launched server, build `FROM` this image and add it.

The build reads the version from `git describe`, so build from a normal git clone. Any
context without a resolvable `.git` — a source tarball, or a git worktree whose `.git` is
a pointer file — needs `--build-arg TERSE_VERSION=0.3.1` instead.

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
  cli.py         entrypoint: gate / policy / compress / capture / measure / probe / validate /
                 proxy / stats / fluency / tune / install-mcp / uninstall-mcp / mcp-status / verify
scripts/
  gen_stress_corpus.py  synthetic stress corpus for the fluency eval; `--fleet-shapes`
                        adds `synthetic.*` shapes a capture corpus cannot hold (#403)
  bench/                terse-vs-TOON token benchmark on a real GitHub-API corpus
                        (fetch_corpus.sh, benchmark.py, diff_demo.py, toon_encode.mjs)
  bench/mcp_servers/    what terse does, zero-config, to popular third-party MCP servers
                        (mcp_probe.py harness + pinned repo/web fixtures; BENCHMARKS §6)
examples/
  demo_mcp_server.py    stdlib-only stdio MCP server; the container's default downstream,
                        so `docker run` demonstrates the proxy without a real server
tests/           round-trip, measurement, probe, policy, and fluency tests
Dockerfile       terse + the demo downstream, for registries and one-command trials
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

Head-to-head token reduction on **real, public GitHub API payloads** (`scripts/bench/`) —
the nested, record-shaped output that dominates real MCP tool traffic. Lossless and
verified per payload, counted in `cl100k_base`. The corpus is already-compact JSON, so
every number is *pure structural* gain, the hardest honest case.

**terse is not alone in this niche.** A prior-art sweep (#298) found at least ten projects
in the same space; three were installed and measured on 2026-09-22 against the same corpus
(`BENCHMARKS.md` §4). The closest architectural match is **compressmcp**, an MCP-layer
lossless JSON compressor — not TOON. An earlier edition of this section called TOON "the
only directly-comparable public tool"; that is **withdrawn**.

Measured head-to-head, cl100k, all lossless-verified per payload:

| payload (real GitHub API) | records | raw tok | terse | TOON 4.1.1 | compressmcp 0.4.0 |
|---|--:|--:|--:|--:|--:|
| gh_pulls | 30 | 151,165 | **76.1%** | −8.0% | 6.3% |
| gh_labels | 9 | 632 | 15.2% | **19.0%** | −3.0% |
| **weighted total** | | 365,144 | **59.1%** | −6.5% | 4.5% |

*(% = fewer cl100k tokens than raw; higher is better; **bold** = winner.)*

**terse wins decisively on real nested records.** TOON still regresses in aggregate to
−6.5% (worse than raw): it is built for flat, uniform arrays, while GitHub records are
deeply nested and repeat subtrees (a PR embeds the same `user`/`repo` object 60 times),
which terse's dictionary tier folds and TOON's tabular layout cannot. compressmcp is
genuinely lossless but abbreviates keys only, which is a small win on nested records and a
**net loss on small objects**, where its `Keys:` legend costs more than it saves.

TOON leads on flat, short-valued, uniform tables like `gh_labels` — the boundary is
**value repetition, not column width** (a seeded width sweep found no clean crossover).
Note that TOON 4.x narrowed this: it now wins 3 of the 9 payloads, not 1, all on the small
and shallow end.

**On cross-call diff:** terse has it, and an earlier edition claimed competitors had "no
answer for" it. That is too strong — `@sliday/tamp` ships `diff` and `read-diff` stages in
its default pipeline. What it does not do is offer them losslessly. Separately, be aware
that terse's diff tier is **off by default** (#170) and fires on **0.67%** of this
author's live traffic (30 of 4,461 blocks) — though it wins **87%** of the times its base
has matching arguments, and is starved rather than weak (#419). See the note on it below
before weighting it.

Competitor-by-competitor notes (headroom, LLMLingua-2, mcp-compressor, code-execution
approaches, and more — all hands-on tested, no invented numbers), the full per-payload
tables, the column-width sweep, the live production ledger, and a repo-size sweep across
popular third-party MCP servers all live in **[BENCHMARKS.md](BENCHMARKS.md)**, along
with the exact reproduce commands. The running competitor queue, including everything
screened out and why, is
[docs/competitors-to-benchmark.md](docs/competitors-to-benchmark.md).

Adoption honesty: terse has been on PyPI since v0.3.1 (2026-07-18) and is at v0.33.8;
TOON, headroom and the other measured entrants are far more established. terse's wedge is
narrow and specific — unconditionally lossless, no expiring retrieve-cache, no ML
dependency, MCP-transparent — not breadth of adoption.

*(Star counts were removed from these docs on 2026-09-22: they drifted constantly and
never informed whether terse works. What a project measurably **does** is the comparison
worth carrying.)*

**And a token saved is not a dollar saved.** Measured over 1,093 of this author's own
sessions (2026-09-22, 30-day window), 96.9% of raw input tokens are **cache reads**,
billed at a tenth of input price — so a raw-token basis overstates input cost by
**7.47x**. terse shrinks the prefix that gets re-read every turn and delays the context
window filling up; it does not cut the invoice by the codec percentage. Full derivation in
[BENCHMARKS.md](BENCHMARKS.md) → Methodology & honesty notes.

## Related Documentation

- [When to use terse (and when not to)](docs/POSITIONING.md) — the economic model
  (primer cost vs. per-call savings), the break-even rule, and where the codec does
  and doesn't pay off
- [Benchmarks](BENCHMARKS.md) — dated, reproducible numbers: terse-vs-TOON (§1–2), the
  cross-call diff axis (§3), competitors (§4), the live production ledger (§5), and
  popular third-party MCP servers + a repo-size sweep (§6)
- [Verify it yourself](VERIFY.md) — prove losslessness, savings, and no-egress locally
- [Technical Reference](TECHNICAL.md) — architecture, pipeline, policy schema, limitations
- [Usage Guide](USAGE.md) — running the CLI day-to-day and reading its output
- [Changelog](CHANGELOG.md) — notable changes per release

## Status

A working, measured, selective **lossless** library, CLI, and MCP
stdio proxy. The proxy's open question — *does a model read the compressed form as
well as raw JSON, and does it need the format primer to do so?* — has a real,
model-dependent answer as of #249 (2026-08-19), though not yet a final one. The
earlier claim here (Claude Haiku 4.5 and Gemini 2.5 Flash match raw-JSON accuracy,
100% paired, 37% token saving) was a joint panel figure whose "Haiku 4.5" entry
predates the fix that let this harness reach real Anthropic models at all — the
eval gateway's `claude-*` ids were DeepSeek aliases (see #249) — so the 100%
comprehension figure is unconfirmed rather than cleanly reattributable to Gemini
alone. The 37% token saving is deterministic tokenizer arithmetic, unaffected by
that bug, and still holds. A same-day frontier-panel run, via a real `claude -p`
OAuth backend on the stress corpus (`--trials 1` throughout), found: Opus 5
unaffected either way (raw = terse = 100%, 0 regressions); Haiku 4.5 shows a real
8-point gap without the primer (92% vs. 100% raw), fully recovered by it; Sonnet 5
shows a smaller gap (96% -> 100%). Opus and Sonnet are both single-trial reads and
unconfirmed — a later same-day deepening pass re-ran four other (substitute,
non-Anthropic) models at `--trials 3-5` and found three of four flipped
conclusion under more trials. Only Haiku, among the named models, was re-run at
`--trials 3`, and it held (92% -> 99%, essentially unchanged) — the one figure in
this table that is independently replicated. That same deepening pass also found
the primer can measurably *hurt* a smaller substitute model at higher trial
counts. Net: the primer's effect is real but model-dependent, not "unnecessary"
or "always required." See #249 for the full panel, the replication story, and the
still-open real-payload-corpus precondition before any default changes.
Whole-subtree aliasing (folding
repeated objects, not just strings) is built. Cross-call diffing is a lossless tier
that is **off by default** (#170) — not for lack of confidence, but on cost: its primer
paragraph adds 190 cl100k tokens to that server's primer — the largest single section —
attached once per session to each wrapped server that emits a terse form
(#211), against a measured 0.38% hit rate. The tier banked 5,052 tokens over 13.3 days,
which roughly 27 primer attaches erase — about two a day across that window, which an
active fleet passes immediately. That is the standalone cadence; behind a router the
same paragraph rides `initialize` and is re-read every turn, which only widens the gap.
**Re-derived 2026-09-22 over the full all-time ledger (4,461 blocks), superseding the
14-day #250 window (17 of 908 blocks, 1.9% / 3.7%, 1,344 tokens):** the structural
exclusion is gone and the verdict is unchanged — the tier fires on **30 of 4,461 blocks
(0.67%)**, 26 of them emitted, against a live router primer of **502 tokens per turn**.

**Correction to an earlier edition of this line, which called the remaining exclusions
"workload, not structure".** That is true of `no_prior` (318) and false of
`not_smaller_diff_args` (**251**), which is a data-structure choice: the diff base is one
slot per tool (`proxy.py:345`), so an interleaved different-args call evicts the base that
would have paid. Segmented by whether the base's args matched, the tier wins **87%** of
the time it gets a same-args base (26 of 30) and **0%** when it does not (0 of 251). It is
**starved, not weak**. #419 proposes `(tool, args_key)` keying, with a probe first — the
251 do not all convert, and 87% rests on 30 events.
Its full
validation program did pass: pair fluency
(`fluency --diff`, 4-model panel 100% — per the 0.26.0 changelog entry (#249), this
panel's `claude-sonnet-5`/`claude-fable-5`/`claude-haiku-4-*` gateway ids are
confirmed DeepSeek aliases, so it was two DeepSeek models measured twice under
Anthropic names, not four; the 100% pass itself is unaffected by which name each
run carried), the nested-record surface (`structure`: diff
100% vs full-terse 94%), and long-chain drift soaked from both sides — mechanically
(`tests/test_diff_soak.py` — exact reconstruction hundreds of chained hops deep) and
behaviorally (`fluency --diff-soak` — no depth-correlated accuracy loss up to the
keyframe bound). Opt IN per proxy (`--diff`) or per policy (`"diff": true`) — worth it for a
workload that really does re-call the same tool with the same arguments.
Cross-block joining (N content blocks
folded into one record array before compressing) is built and on by default — it removed
the structural exclusion that kept 71% of real traffic out of the diff tier entirely, and
the re-measurement above is what that removal was worth: a higher hit rate on payloads too
small to pay for the primer.
The Tier 1 lossy modes `truncate` and
`drop-to-retrieve` are built (opt-in, off by default). **`summarize` is closed
permanently (#261), not pending** — terse is deterministic-only by decision, and a
model-in-the-proxy tier is out of scope for good. See TECHNICAL.md "Known Limitations".

Evidence now spans three kinds: a fixed public corpus (BENCHMARKS §1–4), terse's own
live production ledger (§5), and **popular third-party MCP servers** measured zero-config
with pinned fixtures (§6) — filesystem, git, memory, serena, sequential-thinking and
everything, re-run cold on 2026-09-22 against terse, TOON 4.1.1 **and** compressmcp on the
identical captured bytes.

The §6 headline, stated as a pair because one payload dominates the basis: over all 12
JSON payloads **terse 77.4% / TOON 48.2% / compressmcp 51.8%**, but the 133k `directory_tree`
is 96.4% of that basis — excluding it the same three read **52.2% / 50.8% / 45.1%**. So
**terse's margin scales with payload size**: below a few thousand tokens the three encoders
land within ~7 points of each other and TOON wins the smallest row outright; on the large
nested tree terse pulls 30 points clear, because that is where repeated subtrees exist to
fold. Every text-shaped tool is 0% one-shot for all three — the codec is the JSON-specific
lever.

<!-- docvet:anchors
install-mcp -> src/terse/cli.py
mcp-status -> src/terse/cli.py
policy -> src/terse/cli.py
tune -> src/terse/cli.py
uninstall-mcp -> src/terse/cli.py
verify -> src/terse/cli.py
-->

