# terse — Benchmarks

**Last updated: 2026-09-22.** Every figure is dated by section. What was re-derived in the
2026-09-22 pass, and what was not:

| section | status |
|---|---|
| §1 terse vs TOON | **re-run 2026-09-22** on TOON **4.1.1** (pin was 2.3.1). The 2.3.1 run was executed first and reproduced every published cell byte-identically, so the deltas are TOON's upgrade, not a terse regression. |
| §2 width sweep | produced 2026-07-17, re-runs byte-identical; **not re-run** this pass |
| §4 competitors | **new entrants measured 2026-09-22** (compressmcp, @sliday/tamp, token-smithers). headroom **not re-run** — its integration point needs provider credentials; its 2026-08-05 table is retained at that date. |
| §5 live ledger | **re-derived 2026-09-22**: 4,461 blocks, 23.3% blended / 19.3% lossless-only. Supersedes the 2026-08-11 snapshot (2,357 blocks, 15.1%). |
| §6 popular MCP servers | **re-run 2026-09-22** on v0.33.8, fresh cold round, plus a **third encoder column** (compressmcp). Two cells reproduce the 2026-08-04 round byte-identically. |
| §7 topology | **new 2026-09-22.** Its A/B table is explicitly pre-#211 and is history, not guidance. |
| honesty notes | **new cost-basis derivation 2026-09-22** over 1,093 sessions. |

Nothing here is hand-typed or estimated. If you re-run and get different numbers, the code
changed; open an issue.

Two different kinds of evidence live here, and the difference matters:

- **§1–4, §6 are reproducible by anyone** — fixed corpora, pinned fixtures, credential-free
  servers, commands shown below.
- **§5 is one person's live traffic** and is *not* stranger-reproducible by design; it is
  there to show what an honest production number looks like, and to hand you the one
  command that gives you your own.

## What is being measured

- **Token reduction** = how many fewer tokens the compressed form costs vs the raw JSON,
  counted in **`cl100k_base`** (the tiktoken vocabulary terse uses). Higher is better.
  A payload that is *already* compact (no pretty-print whitespace) makes every number here
  a *pure structural* gain — the hardest honest case.
- **Lossless** = `decompress(compress(x)) == x` exactly. Every terse row below is verified
  lossless per payload; a payload is dropped from a total if either tool fails its round-trip.
- "raw", "terse", "TOON" columns are % fewer tokens than the raw JSON.

## Reproduce everything

```bash
uv sync
cd scripts/bench && npm install          # pins the official @toon-format/toon encoder
cd -
uv run scripts/bench/benchmark.py        # §1  terse vs TOON on real GitHub API payloads
uv run scripts/bench/width_sweep.py      # §2  the column-width sweep
uv run scripts/bench/diff_demo.py        # §3  cross-call diff (terse's own axis)

terse stats                              # §5  YOUR live ledger (your traffic, not ours)
cat scripts/bench/mcp_servers/README.md  # §6  popular third-party MCP servers + repo-size
                                         #     sweep (pinned fixtures, credential-free)
```

---

## §1 — terse vs TOON on real, public GitHub API payloads

The corpus is real GitHub API output (`scripts/bench/corpus/`) — the nested, record-shaped
tool traffic terse targets. `cl100k` tokens, all lossless.

Re-measured **2026-09-22** against **TOON 4.1.1** (the harness had been pinned to 2.3.1;
there is no 3.x — TOON went 2.3.1 → 4.0.0). Both columns from the same
`uv run scripts/bench/benchmark.py` run, no hand-patched cells.

**The pinned 2.3.1 run was executed first, on the same day, and reproduced every published
cell byte-identically** — so the changes below are TOON's upgrade, not a terse regression.
terse's column is unchanged at **59.1%** under both encoders. See
[`scripts/bench/version_sweep.md`](scripts/bench/version_sweep.md) for why 58.3% stood
unchanged from `v0.5.1` to `v0.17.0` before #202's union-schema tabularize moved it.

| payload | records | raw tok | **terse** | TOON 4.1.1 | TOON 2.3.1 |
|---|--:|--:|--:|--:|--:|
| gh_pulls | 30 | 151,165 | **76.1%** | −8.0% | −8.4% |
| gh_workflow_runs | 20 | 76,032 | **80.4%** | −7.5% | −7.5% |
| gh_issues | 30 | 48,032 | **38.8%** | −8.0% | −8.0% |
| gh_commits | 30 | 69,652 | **26.5%** | −4.5% | −4.5% |
| gh_dir_listing | 24 | 6,736 | **31.4%** | 10.4% | −7.7% |
| gh_rate_limit | 1 obj | 357 | 13.4% | **18.8%** | −36.7% |
| gh_repo_single | 1 obj | 1,652 | **0.0%** | −3.9% | −4.4% |
| gh_commits_flat | 30 | 10,886 | **2.4%** | 1.7% | 1.7% |
| gh_labels | 9 | 632 | 15.2% | **19.0%** | 19.0% |
| **weighted total** | | **365,144** | **59.1%** | **−6.5%** | **−7.1%** |

**Plain reading:** on real nested records terse cuts tokens **59%**; TOON still *regresses*
in aggregate to −6.5% (worse than raw) because it adds a key-path per nesting level, while
terse folds the repeated subtrees and long repeated strings (e.g. `gh_pulls` = 60 copies of
the same repo object collapsed to one legend entry → 76%).

**What changed in TOON 4.x, and it is worth stating plainly: TOON now wins three of the nine
payloads, not one.** The whole of its improvement is on the small and shallow end —
`gh_rate_limit` −36.7% → **+18.8%** and `gh_dir_listing` −7.7% → **+10.4%**, with
`gh_pulls` and `gh_repo_single` moving less than half a point. The five large nested payloads
that carry the weighted total are **byte-identical** across the two versions. So the aggregate
barely moved while the per-payload story changed materially, and the previous edition of this
section — which said TOON "wins only on `gh_labels`" — is **withdrawn**.

---

## §2 — Column-width sweep: is there a "narrow vs wide" crossover? (No.)

A natural hypothesis is that TOON overtakes terse once records get *wide* (many columns),
because TOON writes the header once per table. We tested it directly: **40 rows held fixed,
column count swept 2→12**, seeded, each row verified lossless for both tools.

| columns | terse% | TOON% | winner |
|--:|--:|--:|--:|
| 2 | 40.4 | 44.0 | TOON +3.6 |
| 3 | 52.1 | 48.7 | terse +3.4 |
| 4 | 46.8 | 48.8 | TOON +2.0 |
| 5 | 52.6 | 50.7 | terse +1.9 |
| 6 | 48.9 | 50.5 | TOON +1.6 |
| 7 | 52.7 | 51.5 | terse +1.2 |
| 8 | 50.0 | 51.2 | TOON +1.2 |
| 9 | 52.9 | 52.1 | terse +0.8 |
| 10 | 50.6 | 51.7 | TOON +1.1 |
| 11 | 52.9 | 52.3 | terse +0.6 |
| 12 | 51.1 | 52.1 | TOON +1.0 |

**Plain reading:** there is **no clean column-count crossover.** The winner oscillates by
parity, the margins are ~1–4 points, and they **converge toward a tie** as width grows — the
opposite of "TOON pulls decisively ahead when wide." (An earlier draft of this repo's README
claimed a ≤3/≥4-column boundary from a single synthetic construction; a seeded sweep does not
reproduce it, and the claim was corrected.)

**The real dividing axis is value repetition, not width.** On these stripped-flat synthetic
tables — no nesting, no long repeated strings — terse's dictionary/subtree tiers have little
to fold, so the two tools tie. terse's decisive §1 win comes precisely from the redundancy
that real records have and synthetic flat tables don't.

---

## §3 — Cross-call diff (an axis no stateless encoder has)

When the same tool is called again (poll a list, re-read a file), terse can emit a lossless
*delta* against the prior result instead of the whole payload. TOON, minify, and terse's own
single-shot codec all pay the full column every call. Modeling one repeat call per payload
(`diff_demo.py`), the **second** call costs:

| repeated call | full re-send | diff | smaller by |
|---|--:|--:|--:|
| gh_commits_flat | 10,681 | 812 | **92.4%** |
| gh_issues | 29,611 | 4,448 | **85.0%** |
| gh_pulls | 37,746 | 15,292 | **59.5%** |
| **weighted total** | 149,741 | 40,138 | **73.2%** |

**Honest caveat (read this):** these are *modeled* repeat-call savings. How *often* the
pattern occurs in a real agent loop is workload-dependent and is being measured directly (the
proxy now records a per-result `diff_reason` — run `terse stats` to see the breakdown for your
own traffic). Do **not** read §3 as a claim about aggregate real-world savings; read §1 for that.

---

## §4 — Competitor landscape (hands-on; new entrants tested 2026-09-22)

Installed and tested, not cited from marketing. **Every row in this section was executed
here.** A project that could not be run is named with the reason and gets no number —
this section deliberately does not reprint anyone's self-reported figures.

**terse is not alone in this niche, and the previous edition of this section said it was.**
A prior-art sweep (#298, 2026-08-18) found at least ten projects occupying the same space.
Three of them were installed and measured on 2026-09-22 against the §1 corpus, same cl100k
basis, same lossless bar — results below. The claim that TOON was "the only directly
comparable public tool" is **withdrawn**: `compressmcp` is an MCP-layer lossless JSON
compressor, which is a closer architectural match to terse than TOON is.

TOON's row was re-measured 2026-09-22 on **4.1.1** (see §1); headroom was last tested
2026-08-05 on real proxied traffic and is **not re-run this round** — see the version note
below. LLMLingua-2, mcp-compressor and the native context-editing row are unchanged from
the 2026-07-17 hands-on test.

### New entrants, measured 2026-09-22

Same method as §1: each tool's own encode path, its own defaults, cl100k tokens, and a
per-payload round-trip check. A row that fails its round-trip is reported as failing, not
dropped silently.

| Tool | weighted | lossless? | what actually happened |
|---|--:|---|---|
| **terse** (reference) | **59.1%** | yes, all 9 | — |
| **compressmcp** 0.4.0 | **4.5%** | **yes, all 9** | Its "TerseJSON" tier abbreviates keys and prepends a `Keys:` legend. Real and genuinely lossless — but key abbreviation alone is a small win on nested records, and it **regresses on small objects** where the legend costs more than it saves: `gh_rate_limit` −23.2%, `gh_repo_single` −29.4%, `gh_labels` −3.0%. |
| **@sliday/tamp** 0.8.21 | **not measurable here** | **no** | Declined **7 of 9** payloads at its own defaults. Cause executed, not guessed: `compress.js:499` returns `null` when `minified.length >= text.length`, and 7 of the 9 corpus files are already byte-minified. Of the two it did process, `gh_commits_flat` (1.7%) **emitted output that does not parse as JSON**. Its default stage list is `cmd-strip, minify, toon, strip-lines, whitespace, llmlingua, dedup, diff, read-diff, prune` — lossy by construction. |
| **token-smithers** | **no codec measured** | n/a | Its `--pipe` CLI returned every payload **byte-identical (0.0%)**. That is not its compressor: `src/token_sieve/cli/main.py: create_pipeline()` registers `PassthroughStrategy()` for `ContentType.TEXT`, so pipe mode is a no-op by construction. Its real codec runs only in MCP-proxy mode, which this round did not stand up. |

**On `tamp` specifically:** it confirms #298's observation that terse's cross-call diff tier
is *not* a unique axis — `diff` and `read-diff` are in tamp's default pipeline. What tamp
does not do is offer it losslessly.

### Registry name collisions — verify identity before benchmarking anything here

Three of the four projects worth testing are **not** the package their obvious registry name
resolves to. Getting this wrong publishes a number attributed to the wrong project:

| You might type | What actually installs | The real project |
|---|---|---|
| `npm i tamp` | `tamp@0.0.3` — oztu, "Encoder for the Tamper protocol" | **`@sliday/tamp`** (scoped), v0.8.21 |
| `npm i llmtrim` | `llmtrim@1.4.0` — VincenzoManto, a token-trimming library | `fkiene/llmtrim` — GitHub only (Rust) |
| `pip install headroom` | a different tool entirely | **`headroom-ai`** |
| `pip install token-smithers` | 404 — on no registry | `shacharbard/token-smithers` — GitHub only |

`npm i compressmcp` is the one that resolves correctly (TheDecipherist's).

**Version drift note:** `headroom-ai` has moved 0.34.0 (tested 2026-08-05) → **0.38.0**
(published, checked 2026-09-22). Its measurable integration point is a live LLM-API proxy
against a real provider, which needs provider credentials, so it was **not re-measured this
round**. The table below is retained at its 2026-08-05 date and should be read as such.

**Headroom re-tested 2026-08-05, on its real integration point.** `headroom-ai` moved from
v0.33.0 (2026-07-30) to v0.34.0 (this run) — same architecture, same numbers below,
reproduced exactly. It pivoted from "JSON compressor" to a full "Context Optimization
Layer" — it's now an **LLM API proxy** (`headroom proxy --backend anthropic`) that sits
between a coding agent and the model provider, compressing `tool_result` blocks inside real
Anthropic Messages API traffic. Calling its old standalone compressor function directly (as
attempted here previously) returns 0% — that function isn't the real path anymore. Correct
method: stood up a mock Anthropic endpoint, routed realistic `tool_use` → `tool_result`
conversations (our corpus JSON as the tool output) through a real `headroom proxy`
(`--mode token`, `--stateless`), and measured what it actually forwarded upstream, cl100k,
same method as §1:

| file | raw tok | **terse** (lossless) | headroom, CCR default (lossy, recoverable) | headroom `--lossless` |
|---|--:|--:|--:|--:|
| gh_pulls | 151,165 | **76.1%** | 42.5% | 0.0% |
| gh_issues | 48,032 | **38.8%** | 33.1% | 0.0% |
| gh_commits | 69,652 | 26.5% | **46.6%** | 0.0% |
| gh_rate_limit | 357 | **13.4%** | 0.0% | 0.0% |

The only headroom mechanism that moved anything on this corpus is **CCR offloading**: a
span is deleted and replaced with a `<<ccr:HASH ...>>` stub the model must call a
`headroom_retrieve` tool to recover — a lossy, stateful contract, not a smaller lossless
encoding. What gets offloaded differs per file: `gh_pulls`/`gh_commits` each get one
`N_rows_offloaded` marker (row-drop), but `gh_issues` gets **30 markers, all large-string
offloads** (`string,3.5KB` etc.) — zero rows dropped, same mechanism family, different unit.
`gh_rate_limit` is untouched content-wise (0%) but still pays for it: CCR unconditionally
injects a `headroom_retrieve` tool definition (114 cl100k tokens) into every request whether
or not anything was offloaded, so on this file the net effect of CCR is **negative** — the
0.0% in the table undercounts the real cost.

Its explicit `--lossless` mode (no CCR, format-native compaction only) gave **0% on all four
files** — content forwarded byte-identical to the corpus input, verified directly, not
inferred from the reported percentage. Re-tested with the optional `[ml,code]` extras
installed (`torch`, `transformers`, `tree-sitter`) to check whether an unhealthy `kompress`
backend (`/readyz` still reports it unhealthy) explains the 0%: it doesn't — same 0% on all
four files with the ML backend nominally available. The lossless path is 0% on this corpus
independent of the ML backend, not a missing-dependency artifact.

**Read the table honestly, not as a sweep:** on `gh_commits`, headroom's *lossy* number is
larger than terse's *lossless* one — that is a real result, not spun away. On the other
three files terse's unconditionally-lossless number wins outright even against headroom's
lossy mode. The honest framing is the trade, not a single winner: headroom can go further
by deleting data recoverably (or not, with `--no-ccr`); terse never deletes anything.

The terse column was re-measured 2026-08-04 (#207); headroom's was independently re-run
2026-08-05 (#215) and reproduced its 2026-07-30 numbers exactly — both sides of this table
are now current, not just directionally comparable. `gh_issues` moved 32.7% → 38.8% under
union-schema tabularize (#202) and so crossed above headroom's unchanged 33.1%, which is why
this paragraph names one file where it used to name two.

| Tool | What it is (verified) | Comparable? |
|---|---|---|
| **headroom** (`headroom-ai`, v0.34.0) | LLM-API proxy compressing `tool_result` blocks in live Anthropic/OpenAI traffic. Its only active mechanism on this corpus is **CCR offloading** — rows or large strings, lossy, recoverable via a `retrieve`-tool round-trip against a cache — or `--no-ccr` (lossy, unrecoverable); its `--lossless` mode measured **0%** here. Measured 2026-08-05 on real proxied traffic: 0%–46.6%, see table above (reproduces the 2026-07-30 v0.33.0 run exactly). | Partially — same lossy/lossless split as before, but re-measured on its actual integration point (an LLM message proxy) rather than a removed standalone function. |
| **LLMLingua-2** (Microsoft) | Lossy prompt token-classifier. Fed JSON it strips syntax (`{`,`}`,`:`,`"`) as low-information and emits **invalid, unparseable JSON**; truncates past 512 tokens. ~50% on both prose and JSON. | No — different axis (prompts, not tool output), lossy, corrupts structure. |
| **Atlassian mcp-compressor** | Primarily lossless schema/description compression at connect time — **complementary and stackable** with terse (`terse proxy -- mcp-compressor -- <server>`). An opt-in `--toonify` flag also reformats results into TOON (off by default; no diffing/policy/state). | Adjacent, not competing. |
| **Anthropic / OpenAI context editing** | Native, server-side, **lossy** history-pruning; no local artifact to run keylessly. | Different mechanism (drops old results server-side). |

---

## §5 — In production: the live ledger (not just the curated corpus)

§1–4 and §6 are things anyone can re-run. This section is **real proxied traffic** — the author's own kb /
secret-broker / runecho / codegraph sessions — read from terse's always-on, **payload-free**
savings ledger (sizes + decisions only, never content). Unlike those, a stranger can't
reproduce *these* numbers (they're one person's traffic); the point is the opposite — here is
what an honest production figure looks like, and the one command that gives you *yours*.

**Headline (measured 2026-09-22, `terse stats`, all-time, ledger spans 2026-07-15 to
2026-09-22):**

```
4,461 blocks   11,676,854 -> 8,956,106 tok   23.3% blended
                                             19.3% lossless-only
```

*(Previous edition: 2,357 blocks, 4,701,499 → 3,991,185, 15.1%, snapshot 2026-08-11.)*

**Two numbers, because one would overstate the lossless claim.** The fleet now includes
exactly one lossy row — `codegraph_explore`, whose `$text.code_blocks` drop-to-retrieve rule
(#139) takes 738,172 → 133,280 tokens, i.e. **81.9%, and 604,892 of the 2,720,748 total
saved — 22.2% of all savings from one tool.** Excluding it, the strictly-lossless fleet
figure is 10,938,682 → 8,822,826 = **19.3%**. terse's wedge is "unconditionally lossless",
so the lossless-only number is the one that belongs next to that claim; 23.3% is what the
context actually shrank by, with one opt-in lossy rule live.

The top of the per-tool table, same run:

```
secret-broker  secret.list_credentials   103 blocks   1,879,016 ->   647,232   65.6%
codegraph      codegraph_explore         203 blocks     738,172 ->   133,280   81.9%  (LOSSY)
kb             kb.read.list_principles   896 blocks   2,675,293 -> 2,274,721   15.0%
kb             kb.read.list_nodes         63 blocks   2,044,614 -> 1,801,313   11.9%
kb             kb.read.search            573 blocks     557,489 ->   484,668   13.1%
```

**A second fleet makes the point much harder to dismiss.** The same operator runs terse on
a second machine with a different workload. Read the same day, same command:

| fleet | terse | blocks | blended | **lossless-only** | share of savings from the one lossy row |
|---|---|--:|--:|--:|--:|
| A (this section) | 0.33.8 | 4,461 | 23.3% | **19.3%** | 22.2% |
| B (second machine) | 0.14.2 | 389 | **51.8%** | **8.8%** | **92.7%** |

Fleet B's headline is **2.2x higher** than fleet A's while its lossless result is **2.2x
lower**. It leans heavily on `codegraph_explore`, whose drop-to-retrieve rule takes
839,904 → 133,467 there: **92.7% of everything that fleet saved is one lossy rule.**

**So the blended headline mostly measures how much of your traffic hits a drop rule, not
how good the codec is** — across these two fleets it is actually *anti-correlated* with
lossless codec performance. Anyone choosing terse on a 51.8% figure would be choosing it
for a drop rule they have to opt into, on one tool.

**And fleet B's *lossless* figure is low for a reason that is config, not workload.** Its
`mcp-status` reads:

```
codegraph       folded  behind=terse
kb              folded  behind=terse
runecho         folded  behind=terse
secret-broker   unwrapped
```

`secret-broker` is installed on that machine and **is not wrapped**, so terse never sees
it and it cannot appear in that ledger at all. On fleet A the same server is the single
largest lossless contributor — `secret.list_credentials`, **65.6% over 1,879,016 raw
tokens**. Fleet B's lossless surface is therefore almost entirely kb prose tools, which
cap in the low teens on any version.

By §7's own decision rule that is the **worst** server to leave unwrapped: its results
carry `structuredContent`, so its primer is free (fleet A: 62 emissions, every one
`attached: false`, 0 tokens). It is unconditionally-lossless savings available at zero
primer cost, not taken.

**The lesson worth carrying off this pair: a fleet's lossless headline is mostly a
function of WHICH SERVERS YOU WRAPPED, and it is easy to leave the best one out.** Run
`terse mcp-status` before reading anything into a low number.

*Caveat, stated rather than buried: fleet B runs terse **0.14.2**, nineteen minor versions
behind fleet A, so its 8.8% is NOT a clean read on the current codec (it predates #202's
union-schema tabularize, among others). Comparing the tools both fleets DO wrap, the older
build runs 2–4 points lower on five of seven — real, but far too small to explain 8.8% vs
19.3%. The unwrapped server and the drop-rule composition are what explain it.*

That 23.3% is honest and *incomplete*, for three reasons now — the two below plus a
third this section didn't have when it was first written.

**1. This ledger spans multiple codec changes, not one fixed codec.** An all-time ledger from
2026-07-15 to 2026-08-11 straddles #116 (cross-block join), #202 (union-schema tabularize),
and structured-content compression going live for `claude-code` sessions — each shipped
mid-window, so early and late records in the same ledger were produced by measurably
different pipelines. `terse stats --since 7d` / `--since 14d` read higher than the all-time
figure for exactly this reason. The number is *composition*, not a constant — which is why we
publish a range.

**2. Savings track payload shape.** Which tools you call sets the mix. Measured on the real
captured records (production policy, deduplicated to one call's worth per tool):

| shape | example tool | codec: per-block → joined (#116) | an *unchanged* repeat |
|---|---|--:|--:|
| wide, low-cardinality | `kb.read.changelog` | 21% → **38%** | ~99% |
| | `kb.read.recent_rejections` | 17% → **33%** | ~99% |
| | `kb.read.for_repo` | 15% → **24%** | ~98% |
| prose-heavy records | `kb.read.list_principles` | 3% → 3% | **~99.9%** |
| | `kb.read.get` | 2% → 2% | ~99.9% |
| already-projected small | `kb.read.query_stats` | 41% → 41% | — |
| tiny status objects | *(policy `tiers:[]`)* | 0% → 0% *(correct — already minimal)* | — |

That per-block-vs-joined comparison is real and reproducible — re-run 2026-08-11 against 141
freshly captured `kb.read.list_principles` payloads (`terse measure --corpus`) and it lands at
3.5%, matching this table almost exactly. **But the earlier claim that nothing could move this
number was wrong, and the live ledger is the counter-evidence.** `kb.read.list_principles` reads **15.1%** blended
in production today (875 blocks, 2,067,745 → 1,754,482 tokens, `terse stats`) — five times this
table's figure. The gap is not a third tier reaching the prose; it is call-shape composition,
same as reason 1 above, concentrated: 59 of those 875 blocks are large calls routed through
`multiproxy`, each spanning many joined content blocks, and those 59 alone carry 88% of the
tool's raw tokens. A handful of big, well-shaped calls can outweigh hundreds of small
prose-ceiling ones in a token-weighted average — which means **"ceiling" was the wrong word for
what this table measures.** It correctly bounds one shape (a single small record, isolated) and
says nothing about a tool's real production number, which depends on how large and how joined
its actual calls are. See `docs/POSITIONING.md` for the current production breakdown.

**What #116 actually changed here.** The codec fold (per-block → joined) helps the wide
low-cardinality tools and does ~nothing for prose. The real lever is the **diff tier**, which
the per-block path could never reach:

- **76% of ledger tokens** are the multiblock JSON shape #116 targets.
- **71% of ledger tokens are now diff-eligible** — a join fires *and* a repeat produces a
  lossless delta — where **before #116 that share was 100% excluded** from diffing.
- On an *unchanged* repeat, those results collapse **~99%** (the right column above). kb data
  changes slowly and these tools are re-read many times per session (`list_principles`: 865
  calls in this 7-day ledger), so in a real agent loop a large fraction of calls after the
  first are near-empty diffs.

> **Correction (2026-09-22): the "~71% ceiling" below was never reached, and the live ledger
> says why.** That projection assumed the diff tier would fire on the diff-eligible share.
> It does not, because #170 made diffing **opt-in** and nothing turned it on. The decision
> mix over all 4,461 blocks:
>
> ```
> compressed=3356   passthrough=659   unchanged=416   diff=30
> diff reasons: diff_off=1387  joined=587  multiblock=444  no_prior=318  emitted=26
> ```
>
> **26 emitted diffs in 4,461 blocks.** `diff_off=1387` is the single largest reason and is
> *expected*, not a misconfiguration — the primer paragraph costs more every turn than a
> ~0.6% hit rate returns. The honest reading is that terse's production win is the
> **always-on codec**, not cross-call diffing; the range below describes a tier that is dark
> in this deployment. It is kept for the mechanism, not as a forecast.

**So the production figure is a range, not a point:**

- **Floor** — every call, no repeats: the joined codec alone, ~9–12%, dominated by prose
  ceilings.
- **Ceiling (unrealised — see the correction above)** — repeat-heavy loop with data stable
  between calls: re-weighting the ledger's own token mix, the diff-eligible 71% collapsing
  ~99% each puts the aggregate near **~71%**.
- **Reality sits between**, set by *your* repeat rate and how fast *your* data changes — which
  no benchmark can tell you. So measure it:

```bash
terse stats                 # rollup: results, decisions, tokens saved, per-tool rows,
                            #          and the diff-reason breakdown (how often diffs fire)
terse stats --since 7d      # windowed
```

Wrap your servers (`terse install-mcp …`), use them for a week, read your own ledger. That
converts "trust our benchmark" into "run it on your traffic" — the honest version of the
claim, and a better pitch besides.

---

## §6 — Popular third-party MCP servers (re-measured 2026-09-22)

> **Re-measured against the merge of #202** (union-schema tabularize), superseding the
> 2026-07-30 numbers below. This was a genuinely fresh, cold run — new `express`/`fastapi`
> clones (the pinned tags were already checked out but as **shallow** clones, `git log`
> from a single commit; unshallowed to get a real `git_log` payload), a `memory` graph
> started empty and verified so before populating it, and every call's exact tool
> arguments now persisted beside its corpus (`mcp_probe.py`'s new `_calls.json` sidecar,
> #138 step 0) so the next re-measure never has to guess. Net effect on the codec column:
> small and mixed — `filesystem/directory_tree` moved 58.0% → 57.3% (a wash, within
> measurement noise, not the direction #202 usually pushes); every other row that
> reproduces the same shape is unchanged. **The bigger finding this round is unrelated to
> #202**: see the "diff tier now needs `--diff`" note below.

§5 is one person's traffic. This section is the other half: what terse does **automatically,
zero-config** to the output of widely-used, **credential-free** MCP servers that anyone can
run. Reproduce with `scripts/bench/mcp_servers/` (pinned repo fixtures, a static local web
fixture, one command per server).

### Re-measured 2026-09-22 — three-way, on record-shaped public payloads

A fresh cold round against **v0.33.8**: new full (non-shallow) clones of the pinned repo
fixtures, every server relaunched, every payload re-captured, and a **third encoder column
added** — compressmcp, which §4 identifies as terse's closest architectural match.
`toon_column.py` now measures terse, TOON **4.1.1** and compressmcp **0.4.0** on the
*identical captured bytes*, so no encoder gets a different input.

| server | tool | shape | raw tok | **terse** | TOON | compressmcp |
|---|---|---|--:|--:|--:|--:|
| filesystem (django, whole pkg) | `directory_tree` | array-of-records | 133,370 | **78.3%** | 48.1% | 52.0% |
| filesystem (django, `db/`) | `directory_tree` | array-of-records | 2,696 | **57.3%** | 56.3% | 50.4% |
| filesystem (fastapi) | `directory_tree` | array-of-records | 1,328 | **50.5%** | 49.5% | 49.3% |
| filesystem (express) | `directory_tree` | array-of-records | 116 | 54.3% | **69.0%** | 41.4% |
| memory | `search_nodes` | array-of-records | 168 | **44.0%** | 36.9% | 31.5% |
| memory | `read_graph` | array-of-records | 215 | **43.7%** | 37.2% | 34.4% |
| memory | `create_entities` | array-of-records | 203 | **42.4%** | 36.0% | 36.5% |
| sequential-thinking | `sequentialthinking` | pretty-json | 41 | **34.1%** | 31.7% | −22.0% |
| serena | `get_symbols_overview` | compact-json | 130 | **18.5%** | 16.9% | 6.9% |
| everything | `get-structured-content` | compact-json | 14 | 0.0% | 0.0% | −107.1% |
| filesystem | `read_text_file` | long-text | 3,524 | 0.0% | n/a | n/a |
| git | `git_log` | long-text | 4,541 | 0.0% | n/a | n/a |

**Two weighted totals, because one of them is really one payload:**

```
all 12 JSON payloads          raw 138,323   terse 77.4%   TOON 48.2%   compressmcp 51.8%
excluding the 133k tree       raw   4,953   terse 52.2%   TOON 50.8%   compressmcp 45.1%
```

`fs-django-big` alone is **96.4%** of the weighted basis. Quoting 77.4% without that is
quoting one directory tree. The honest reading of the pair: **terse's margin scales with
payload size.** Below a few thousand tokens all three encoders land within ~7 points of
each other and TOON actually wins the smallest row (express, 116 tok, 69.0%); on the 133k
tree terse pulls 30 points clear, because that is where repeated subtrees and repeated
long strings exist to fold.

**compressmcp goes negative on small payloads** — −22.0%, −107.1%, −300.0% on the three
smallest — for the same reason §4 found: its `Keys:` legend is a fixed cost that a tiny
object cannot amortise. Its 51.8% on the full set is almost entirely the one large tree.

**Reproductions of the 2026-08-04 round:** `directory_tree` on django's `db/` returns
**57.3%**, byte-identical to the previously published cell, and `sequentialthinking`
returns **34.1%**, likewise. Those two are unchanged across seven weeks and a terse major.

**Two upstream changes found while re-running, both of which would have silently corrupted
the round:**

- `everything`'s `get-structured-content` now **requires** a `location` enum argument. The
  2026-08-04 probe called it with `{}`; that now returns `isError: true`. Per this repo's
  own rule a `TOOL ERROR` payload is discarded and re-run, which is what happened here.
- The repo fixtures must be verified non-shallow before `git_log` is trusted — the
  2026-08-04 round was bitten by 1-commit shallow clones. Checked explicitly this round
  (express 6,103 / fastapi 7,521 commits).

### Still not covered here

`playwright`, `fetch`, `duckduckgo` and `hn` were **not re-run** this round — all four are
text-shaped, where every encoder scores `n/a`/0%, and playwright additionally needs a
Chromium build. Dropping them costs no comparison. `mcp-server-sqlite` would be the ideal
record-array addition (SQL result sets are uniform record arrays) but remains unrunnable:
it is built on the `mcp` SDK's removed `@server.list_tools()` API with no known-good pin,
the same wall that keeps `mcp-server-time` and `mcp-server-calculator` out of this table.

### The 2026-08-04 round, retained for the rows the 2026-09-22 round did not re-run

Superseded where the two overlap — the table above is current for `filesystem`, `memory`,
`sequential-thinking`, `serena`, `git` and `everything`. This one is kept because it is the
only measurement of the text-shaped servers (`playwright`, `fetch`, `duckduckgo`, `hn`) and
of the repeat/diff column, and because its methodology notes still apply.

Servers: the official reference set (`modelcontextprotocol/servers`) plus four widely-used
credential-free third-party servers — **serena**, **playwright-mcp**,
**@modelcontextprotocol/server-sequential-thinking**'s companion tools, and two more
credential-free community servers added in this round, **duckduckgo-mcp-server** and
**@devabdultech/hn-mcp-server** (Hacker News), to broaden shape coverage beyond the original
six.

| server | tool | output shape | codec % (1-shot) | TOON % | an *unchanged* repeat† | reaches the model? |
|---|---|---|--:|--:|---|---|
| filesystem | `directory_tree` | JSON, pretty-printed | **57.3%** | 56.3% | diff | ⚠️ no — see below |
| filesystem | `read_text_file` | source text | 0% | n/a | text-diff | ⚠️ no — see below |
| git | `git_log` | long text | 0% | n/a | text-diff | yes |
| memory | `read_graph`, `search_nodes`, `create_entities` | JSON | **35.1–41.1%** | 26.6–42.1% | diff (on `read_graph`, `search_nodes`) | ⚠️ no — see below |
| serena | `get_symbols_overview`, `find_symbol` | JSON, already compact | **21.4–21.7%** | 4.3–7.1% | — (too small to beat a re-send at this size; diff does land at ~300 tok, see below) | yes |
| playwright | `browser_snapshot` | accessibility tree (text) | 0% | n/a | text-diff | yes |
| fetch | `fetch` | markdown | 0% | n/a | text-diff | yes |
| sequential-thinking | `sequentialthinking` | JSON, pretty-printed | **34.1%** | 31.7% | — (identical args, diff not smaller) | yes |
| everything | `get-structured-content` | JSON, tiny (14 raw tok) | 0% (below the small-payload floor) | 0% | — | ⚠️ no — declares `outputSchema` |
| duckduckgo-mcp-server | `search` | formatted text | 0% | n/a | text-diff | yes |
| hn-mcp-server | `getStories` | formatted text | 0% | n/a | text-diff | yes |

**† Since #170 (2026-07-28) cross-call diffing is no longer the zero-config default** —
every cell in this column was measured with `--diff` passed explicitly (a tiny `TERSE_BIN`
wrapper this round, since `mcp_probe.py` has no CLI hook to add a proxy flag; see the
methodology note below the headline). Run these servers with a plain `terse proxy`
(no `--diff`) today and every one of these rows reads `diff_off` in the ledger instead —
a real behavior change from when this table was first published, not a measurement
artifact. The **codec %** and **TOON %** columns are unaffected either way: they come from
the raw payload terse tees to the capture dir *before* any diff decision, so #170 doesn't
touch them.

**`n/a` is not `0%`.** TOON is a JSON serialization; on the six text-shaped tools it cannot
encode the payload at all, which is a different fact from "encoded it and tied". terse also
scores 0% one-shot there — the honest reading of those rows is that *neither* tool claims
anything on prose, and both fall back to the diff tier. 14 of the 28 payloads captured this
round (across the 11 headline tools, the repo-size sweep, and one supplementary larger
serena call — see below) are non-JSON.

Weighted over only the 14 payloads TOON *can* encode (6,243 raw cl100k tokens):
**terse 48.0%, TOON 47.5%.** terse still wins the largest JSON payload
(`directory_tree` 2,696 tok, 57.3% vs 56.3%) and the next-largest (`memory.read_graph`
610 tok, 40.3% vs 37.4%). The bigger change from the 2026-07-30 round is **serena**: both
its rows flipped from TOON's best case to terse's — `get_symbols_overview` 21.4% vs 7.1%,
`find_symbol` 21.7% vs 4.3%, where the previous round had TOON winning the smaller of the
two. The most likely explanation is #202's union-schema tabularize, whose fold targets
exactly serena's shape (an array of symbol records) — plausible, not proven here (the
original round's exact serena arguments were never recorded, which is the gap step 0
fixes going forward). TOON still wins the smallest, most uniform rows —
`filesystem-sweep-express`'s 116-token `directory_tree` (69.0% vs 54.3%, unchanged from
before) and two of memory's four sub-tools (`create_relations` 57.1% vs 49.0%,
`search_nodes` 42.1% vs 41.1%, both near-ties). Published as the trade, not smoothed into a
single verdict: TOON's header-once row format wins where a payload is small and perfectly
uniform, which is the same shape-conditional result §1 and §2 found.

Reproduce the TOON column from the same capture dirs the codec column came from:

```bash
uv run python scripts/bench/mcp_servers/toon_column.py "$CORPUS" [more CORPUS dirs...]
```

> **Discrepancy, resolved for serena and re-scoped for memory (2026-08-04, #138 step 0-3).**
> The 2026-07-30 round left this exact gap open because the original probe's call
> arguments were never recorded — `mcp_probe.py` now writes them beside every corpus
> (`_calls.json`), so this round's numbers are reproducible in a way the old ones weren't.
>
> **serena had no warm-state excuse to begin with** (it carries no persistent graph), and
> two independent re-measures now agree it doesn't reach the published high end: the
> 2026-07-30 reprocessing of the saved capture landed at 22.2–28.5%, and this round's fresh
> cold probe — same two tools, a freshly-indexed project, arguments now on record — lands
> at **21.4–21.7%**, tighter still. Neither run gets anywhere near the published 37%. The
> most likely explanation is that the original high end came from a different (larger or
> deeper) symbol query that was simply never written down — not a measurement error, just
> an unrecorded one, which is exactly the gap step 0 closes for every future round. The
> table above now publishes the reproducible range. Separately, this round found serena's
> diff tier *does* land once a payload clears roughly 300 tokens (a supplementary
> `get_symbols_overview` on a larger file measured 42.3% saved on an identical repeat,
> against 0% at the ~30-token size in the headline row) — consistent with the repo-size
> sweep's floor finding below, just observed on a new server.
>
> **memory does carry real state**, so "cold" only means "verified-empty before this
> round's own graph was written" — it says nothing about what the original round's graph
> held. This round's graph was built from scratch and confirmed empty first
> (`rm`'d `MEMORY_FILE_PATH` under a fresh path, checked it didn't exist, then populated
> it), and measured **35.1–41.1%** across `read_graph`/`search_nodes`/`create_entities` —
> inside the originally published 27–52%, but below the 2026-07-30 reprocessed-capture's
> 43.9–54.1%. Unlike serena, this gap is not resolved to one number, because it can't be:
> the percentage is a function of *what's in the graph*, which is round-specific by
> design, so a different but equally "cold" graph legitimately lands somewhere else in a
> wide band. The published range below is this round's actual measurement, stated as
> such rather than reconciled with a number built from different content.

**Two rows don't fit the "one shape, one number" mold** (both reproduce byte-for-byte this
round): `sequential-thinking`'s single-thought payload is small enough (82 raw tokens) that
the 34% comes almost entirely from minify, not structural folding — read it as a bound, not
a ceiling: real chains with many thoughts will look more like a JSON-array-of-records row.
`everything`'s `get-structured-content` payload (14 raw tokens, a toy weather object) is
the smallest thing measured in this table and it demonstrates the **floor**, not a
weakness: terse correctly does nothing to a payload with no redundancy left to fold rather
than emitting a larger "compressed" form. `duckduckgo` and `hn-mcp-server` both return
**prose, not JSON** — 0% one-shot like `git_log`/`fetch`, and both still won an
*unchanged*-repeat text-diff; note duckduckgo's second call hits a live search endpoint
(not a pinned fixture like the rest of this table), so unlike every other row here its
repeat isn't guaranteed byte-identical run to run.

### Honest scope note: on two of these servers the codec % never reaches the model

Added 2026-07-23, after measuring it. **These percentages describe the text content block,
which on some servers the client discards.**

MCP 2025-06-18 lets a tool return `structuredContent` alongside a text block that
serializes the same data for backwards compatibility. terse compresses the text block and
leaves `structuredContent` alone (#128). Measured with a read-only proxy on the real
client (`claude` 2.1.218, `scripts/probe/structured_content/`), the client forwards
**`structuredContent`** to the model and discards the text block entirely — so wherever a
server emits it, the codec % above is a reduction of a payload the model never sees.

Which servers do, measured by `outputSchema` declarations and confirmed on the wire:

| server | tools declaring `outputSchema` |
|---|--:|
| filesystem | **14 / 14** |
| memory | **9 / 9** |
| serena | 0 / 21 |
| playwright | 0 / 24 |
| git | 0 / 12 |
| fetch | 0 / 1 |
| sequential-thinking | **1 / 1** |
| everything | 1 / 13 |
| duckduckgo-mcp-server | **2 / 2** |
| hn-mcp-server | 0 / 9 |

It splits along SDK generation, not by accident: the newer TypeScript servers declare
schemas on every tool. Expect this to grow — of the four servers added this round, both
`sequential-thinking` and `duckduckgo-mcp-server` (newer, actively-maintained) declare it
on everything they return; `everything`'s lone declaring tool (`get-structured-content`) is
too small (14 raw tokens) for the distinction to matter in practice; `hn-mcp-server`
declares it on nothing, consistent with returning formatted text rather than JSON.

`filesystem`/`directory_tree`, re-measured end to end:

| quantity | tokens |
|---|--:|
| text block, raw | 1,658 |
| text block, terse | 816 |
| — the 50.8% this table reports | |
| `structuredContent` (untouched) | 2,047 |
| **what the model actually receives** | **2,047** |
| **saving in the model's context** | **0%** |
| honest whole-result wire saving | 22.7% |

Note the structured form is *larger* than the text block it mirrors — the JSON wrapper
re-encodes newlines as `\n` escapes — so on this tool the client's choice costs more than
the text block would have, before terse enters the picture at all.

**What survives this correction:** serena's 21.4–21.7% (this round's re-measure, above) is
the only non-zero one-shot codec number in the table that reaches the model, and the diff
tier still lands on `git`, `playwright` and `fetch` — and on `serena` too, just not at the
headline row's small size (see the size-floor note above the table: it needs roughly
300 raw tokens before a diff beats a re-send). The claim below — codec narrow, diff
broad — holds; it is narrower than first published, and the two rows carrying the biggest
codec numbers are the two that don't count.

Tracked in #128. The ledger was corrected first (it had the same flaw one level down, and
now counts the untouched duplicate on both sides); re-running these numbers per-server is
the remaining work.

**Recovering it: `"structured": "compress"`, and the mirror drop that adds nothing.**
Putting the codec on the field the client actually reads takes the reference fixture from
2,596 to **1,008 chars of the model's real context (61.2%)**, measured end to end by the
same probe rather than inferred from the ledger. Going one step further and deleting the
now-redundant text block (`"structured": "replace"`) measures **1,008 → 1,008**: no
change, because the client had already discarded that block. Worth recording as a negative
result — the duplicate is a *wire* cost on this client, not a context one, so the
2,596-char mirror shows up in the ledger's `raw_chars` and never in a token bill. A client
that forwarded both fields would be the one that benefits, and none has been measured.

### The headline: the codec is narrow, the diff tier is broad — and opt-in since #170

Two things fall out, and they matter more than any single percentage. A third, new this
round: **since #170 (2026-07-28) the diff tier is opt-in, not automatic.** Everything the
repeat column shows below describes `terse proxy --diff`, not the zero-config default a
user gets by just pointing `terse proxy` at a server — that default now reads `diff_off`
in the ledger for every row in this table. The codec % and TOON % columns are unaffected
(they're measured off the raw payload, captured before any diff decision); only the
*repeat* story requires the flag.

1. **The one-shot codec pays only on JSON — and how much depends on whether the server
   pretty-prints.** filesystem pretty-prints its tree, so minify alone is most of 50–58%.
   memory returns compact-ish JSON records (35–41% this round). serena emits
   *already-compact* JSON, so only the structural fold is left (21–22%, this round tighter
   than the 22–37% first published — see the discrepancy note above) — the hardest honest
   case, and exactly the "pure structural gain" framing at the top of this file.
2. **Every text-shaped tool is 0% on the codec — and every one still wins on an *unchanged*
   repeat, once `--diff` is on.** `read_text_file`, `git_log`, `browser_snapshot`, and
   `fetch` are all uncompressible one-shot, yet all four emit a content-defined-chunking
   text diff the second time they are called under `--diff`.

**Read the repeat column as a ceiling, not a typical delta.** Both calls send identical
arguments against an unchanged fixture, so `prev == curr`: the diff encodes an empty
changeset and the wire is near-fixed overhead once the payload clears the small-payload
floor described below. That is the *upper bound* of the diff tier — the same discipline §5 applies when it reports ~99%
on an unchanged repeat and then frames production as a floor/ceiling range — note §5's
column is a *number* and this one is only qualitative (`diff` / `text-diff` / `—`). A real agent
loop re-fetches results that have **changed**, and how much the delta grows with the change
is workload-specific and **not measured here**.

So on third-party servers the **cross-call diff is the broad, shape-independent win once
opted in, and the codec is the JSON-specific, zero-config one** — narrowed further by the
`structuredContent` note above, which removes both of the codec's biggest rows from what
the model actually receives, and leaves the diff tier landing on 8 of the 11 rows measured
this round — every row except `sequential-thinking`, `everything`, and serena's headline
row, all three blocked by the same small-payload floor, not by shape (serena's own diff
does land once its payload clears ~300 tokens, see above). That is a sharper claim than a
blended average, and it
predicts where terse helps: agent loops that call the same tool repeatedly, run with
`--diff` turned on. Browser automation is the shape it should suit best — navigate →
snapshot → act → snapshot produces consecutive, largely-overlapping accessibility trees.
Stated as a *prediction*, not a result: what was measured is an identical repeat; a
post-click tree is a different and untested experiment.

**Which command produces which column:** codec % comes from `terse measure --corpus`; the
repeat column comes from `terse stats --log` (its `diff_reason` breakdown) on a proxy run
with **`--diff`** (no longer the default post-#170 — `mcp_probe.py` has no CLI hook to add
the flag itself, so this round used a one-line `TERSE_BIN` wrapper that inserts it; see
`scripts/bench/mcp_servers/README.md`). Capture is content-addressed, so two identical
repeats collapse into one corpus file — the corpus alone can never evidence the repeat
column, only the ledger can, and only when that ledger came from a `--diff` run.

### Repo size barely moves the codec

`directory_tree` across three pinned fixtures — express v5.2.1 (218 tracked files), fastapi
0.139.2 (3,131), django 5.2.16 (6,922):

| fixture | raw tok | codec % | repeat† |
|---|--:|--:|---|
| express | 116 | 54.3% | diff not smaller (payload too small) |
| fastapi | 1,328 | 50.5% | **diff emitted** |
| django | 2,696 | 57.3% | **diff emitted** |

Reproduces the 2026-07-30 round almost exactly (express and fastapi byte-identical; django
58.0% → 57.3%, inside measurement noise and not the direction #202 usually pushes). The
codec sits in a **50–58% band across a 23× payload-size range** — it tracks JSON
structure, not repo size. What size *does* change is the **diff**: below roughly a thousand
tokens the delta loses to simply re-sending the compressed form; above it the diff wins and
keeps winning. † Same `--diff`-required caveat as the main table.

### Zero-config auto-policy holds up

`terse policy generate` was run against each captured corpus and authored a correct,
conservative, lossless policy every time with no hand-tuning: `directory_tree` →
`minify,tabularize` (dictionary auto-dropped as below the 5% threshold), `read_text_file` →
`tiers: []` passthrough (detected as non-JSON), memory's three record tools → all folded.
That is the "does it just work on a server it has never seen" question, answered yes.

### Aside: three archived reference servers are currently broken, unrelated to terse

Not a terse finding, but worth recording since it shaped which servers ended up in this
table. `mcp-server-time`, `mcp-server-sqlite`, and `mcp-server-calculator` (all Python,
built on the low-level `mcp` SDK) fail to start against the `mcp` package version `uvx`
resolves today — `AttributeError: 'Server' object has no attribute 'list_tools'` /
`'list_resources'`, and a separate `ImportError: cannot import name 'McpError'` (renamed to
`MCPError`). None of these servers pin an upper bound on their `mcp` dependency, so `uvx`
always resolves the latest release, which has since dropped the decorator API they were
written against. `mcp-server-git` and `mcp-server-fetch` hit the identical failure and were
only recovered by forcing an older SDK: `uvx --from mcp-server-git --with 'mcp<1.10' ...`.
Time and sqlite were swapped for `sequential-thinking`, `everything`,
`duckduckgo-mcp-server`, and `hn-mcp-server` — all four launch clean with no pin.

### Transports: HTTP downstream and multi-peer fan-out

Everything above is a **stdio** downstream. terse also proxies an MCP **Streamable-HTTP**
endpoint and can front *N* servers from one process. Both were re-exercised this round
against the reference `everything` server run in `streamableHttp` mode.
**Scope: a single run on 2026-07-30, not part of the pinned size sweep** — these establish
that the transports work end-to-end, not a measured savings result:

- **HTTP downstream** — `terse proxy -- http://127.0.0.1:3001/mcp`. `initialize`,
  `tools/list` (13 tools), `tools/call` and the capture tee all behave as on stdio; the URL
  form is selected automatically (a single target containing `://`).
- **Multi-peer fan-out with mixed transports** — one process fronting three peers,
  **two stdio + one HTTP**, via `proxy --config`:

  | check | result |
  |---|---|
  | merged `tools/list` | 36 tools, **unqualified** (`fs`=14, `mem`=9, `ev`=13, no name collisions) |
  | `initialize` primer | injected **exactly once** across all peers |
  | call routing | each bare tool name reached its own peer, including the HTTP one |
  | per-peer compression | `directory_tree` 54.3% (express v5.2.1 `lib/`), `read_graph` 54.1% |
  | ledger attribution | per-peer internally (`fs.directory_tree`, `mem.read_graph`), regardless of the client-facing name |

  **Correction from the 2026-07-22 measurement:** that run predates #168 (`feat(multiproxy)!:
  qualify tool names only on a real cross-peer collision`) and reported tools as
  peer-prefixed (`fs__directory_tree`). Since fs/mem/ev's 36 tools have zero name overlap,
  none are qualified now — a client calls `directory_tree`, not `fs__directory_tree`. The
  qualifier only appears when two peers genuinely share a tool name.

This round also turned up a real defect, now fixed: a server-initiated request
(`roots/list`, `sampling/createMessage`) uses its **own** id space, so its id can collide
with an in-flight `tools/call`. terse consumed the call's tracking entry on that collision
and then forwarded the real result **uncompressed and unrecorded**, silently. See the
`### Fixed` entry in CHANGELOG.

### Honest scope note: #116's cross-block join does *not* apply here

terse's cross-block join folds a result that arrives as *N* content blocks into one record
array. **Every server measured above returns a single content block per result**, so the
join never fires — their wins come from the codec and the diff tier. The join targets
servers that emit one record per block (the kind measured in §5). Worth stating plainly:
a feature that is decisive on one traffic mix can be inert on another.

---

## §7 — Topology: one proxy per server, or all of them behind one router?

`terse install-mcp` writes one wrapped entry per server by default; `--multiproxy` folds the
named servers into a single router process instead. **This choice is worth more than most
policy tuning, and the advice on it inverted in #211 — so the older measurement below must
not be read as current guidance.**

### The pre-#211 measurement (`scripts/bench/ab_session.py`, 2026-07-28)

Real billed tokens from Claude Code transcripts, not terse's own ledger — the `usage` block
the API returned, diffed across two matched sessions:

| wrapped servers | calls | RAW input | n |
|---|--:|--:|--:|
| 1 (runecho only) | 4 | **−14.0%** | 1 |
| 3 (codegraph, kb, runecho) | 4 | +1.4% | 3 |
| 6, one proxy each | 4 | **+23.1%** | 1 |
| 6, behind one multiproxy | 4 | **+0.0%** | 4 |

At the time each standalone proxy injected its primer into that server's MCP `instructions`,
which the client re-read **every turn**, so cost scaled with (servers × turns). Folding
collapsed N primers into one and erased the six-server penalty. Hence the old advice: fold.

### What #211 changed, and why the advice is no longer "always fold"

The two topologies now pay on **different cadences**, and `src/terse/stats.py:1492` states
the contract:

> ```
> router / router-ambiguous    still prime EAGERLY, one union_primer in the router's own
>                              merged initialize.instructions, re-read every turn as
>                              cache_read. RECURRING.
> wrapped / wrapped-unstashed  lazy since #211: the primer attaches to the FIRST result
>                              carrying a terse wire form. Paid ONCE per session if that
>                              result comes, and NOT AT ALL if it never does.
> ```

So the arithmetic is no longer "N primers vs one". It is:

```
standalone wrapped:   sum(per-server primers)  x  1        (per session, and only if called)
multiproxy router:    one union primer         x  turns    (every turn, called or not)
```

**Live evidence, this author's fleet, `terse stats --json`, 2026-09-22:**

| entry | state | primer | cadence | source |
|---|---|--:|---|---|
| `terse` (router: kb, codegraph, runecho) | `router` | 502 tok | **per-turn** | estimated |
| `codegraph` standalone | — | 502 tok / 4 emissions | **once/session** | **recorded** |
| `runecho` standalone | — | 248 tok / 1 emission | **once/session** | **recorded** |
| `secret-broker` | `wrapped` | 0 tok | once/session (**unpaid**) | recorded |

`secret-broker`'s row is the clearest demonstration of the lazy primer: 62 emissions, all
`attached: false`, **0 tokens** — its results carry `structuredContent`, so the client would
have discarded the primer unread and the proxy declined to send it. A standalone wrap of that
server is free.

### The break-even, stated plainly

A router costs `502 × turns`. The same three servers wrapped standalone cost roughly
`502 + 248 + kb` **once**. On those figures the router is the cheaper topology only for about
the first two turns of a session, and is more expensive after that. The live fleet's
`turns_covered` is **5,419** — the savings cover that many turns of router primer, which is
comfortable here, but it is a budget being spent every turn rather than once.

**Current guidance, matching `install-mcp --multiproxy --help`:** consolidate for *operational*
reasons — one policy, one process, one permission surface — **not** to escape a token tax.
Since #211 the token argument points the other way for small fleets. The six-server
penalty the old table shows was a property of the eager primer, and that property is gone.

### Which servers to wrap alone, and which to pool — a decision rule from the cadence

The cadence contract above turns into a rule, and the live fleet supplies the decisive
case for each branch.

**Wrap alone (the default) when any of these is true:**

1. **The server's results carry `structuredContent`.** Its primer is then *free*, not
   cheap. Measured: `secret-broker` has **62 primer emissions, every one `attached:
   false`, 0 tokens** — the client would have discarded the primer unread, so the proxy
   declines to send it. That server compresses at 65.6% over 774 blocks and pays nothing
   at all for the privilege.
2. **The server is called rarely, or not every session.** Since #211 an unc­alled wrapped
   server pays **zero** — the primer attaches to the first compressible result, and no
   result means no charge. A router charges whether or not any peer is ever called.
3. **The server is record-shaped and high-volume.** It clears a once-per-session primer
   almost immediately: `secret-broker` banks 11,959 tokens per block against a primer
   ceiling of ~502.

**Pool behind a router for operational reasons, not token reasons:** one policy file, one
process, one permission surface to approve. That is a real benefit and it is why
`--multiproxy` exists. Accept that it costs **502 tokens every turn** (this fleet's live
figure) in exchange.

**The anti-pattern, stated plainly: never pool a server that would otherwise pay zero.**
Folding `secret-broker` behind the router would move it from 0 tokens to a share of a
recurring per-turn charge, for a server whose results the primer never reaches anyway.
Before #211 pooling was the way to escape N primers; now it can *create* a charge that
standalone wrapping avoids entirely.

`terse stats --recommend` prints the verdict per installed entry rather than per peer,
because a router pays one union primer for its whole fleet — see the `contributors[]`
note in USAGE.md for why there is no honest per-peer primer to divide by.

### Server suitability by output shape — measured anchors, and what they imply

Everything below is measured in §1, §4 or §6 except where marked **inferred**.

| output shape | measured anchor | terse | verdict |
|---|---|--:|---|
| large nested records, repeated subtrees | `directory_tree` (django, 133k tok) | **78.3%** | ideal — wrap it |
| nested API records | `gh_pulls`, `gh_workflow_runs` | **76–80%** | ideal |
| small record arrays | `memory/read_graph`, `search_nodes` | **42–44%** | worth it |
| pretty-printed JSON | `sequentialthinking` | **34.1%** | worth it |
| already-compact JSON | `serena/get_symbols_overview` | **18.5%** | marginal |
| already field-projected | `kb.read.get` (live ledger) | **3.3%** | not worth a primer |
| tiny objects (< ~50 tok) | `everything/get-structured-content` | **0.0%** | below the floor |
| prose / source text / logs | `git_log`, `read_text_file` | **0.0%** | codec cannot help |

**The rule this gives you:** terse pays where a payload repeats structure — the same keys
across many records, the same subtree across many rows. It cannot help where there is no
repetition to fold (prose) or none left (an already-projected result). Payload *size*
matters too, and independently: §6's small rows land within ~7 points of TOON and
compressmcp, while the 133k tree puts terse 30 points clear.

### Not yet measured: frontend / UI-focused servers

An open hypothesis, recorded because it is plausible and **untested**: design- and
browser-tool servers (Figma, Chrome DevTools, Storybook) return deeply nested node trees
and uniform lists — network-request tables, DOM/accessibility node arrays, style and fill
objects repeated across hundreds of nodes. That is structurally the same shape as
`directory_tree`, terse's single best measured result at 78.3%, so these servers **should
compress well**. That is an **inference from shape, not a measurement**, and no number
should be quoted for it until it is run.

Status of the obvious candidates, checked 2026-09-22:

- **`figma-developer-mcp` 0.13.2** — needs a Figma API token. Out of scope for this
  benchmark set, which is credential-free by design.
- **`chrome-devtools-mcp` 1.9.0** — credential-free (drives a local Chrome), and the
  strongest candidate to test the hypothesis. Needs a Chromium build, the same dependency
  that keeps `playwright` out of the 2026-09-22 round.
- **`storybook-mcp` 0.5.1** — needs a running Storybook instance to point at.

Note the one counter-signal already in the data: `playwright`'s `browser_snapshot` returns
an accessibility tree as **text**, and scored **0%** one-shot in the 2026-08-04 round. "UI
server" is not itself a predictor — whether the server hands back JSON or rendered text is.

### What is still unmeasured here

The table above is **pre-#211 and has not been re-run post-#211.** A clean A/B at 1/3/6
servers on the current lazy-primer build is the missing experiment; `scripts/bench/ab_session.py`
is the harness and its protocol is in its docstring. Until that runs, the cadence contract and
the live primer rows above are the evidence, and the percentages in the old table are history.
Related open work: #270 (a router's `initialize` blocks on its slowest peer) and router-level
lazy priming, which would remove the per-turn charge entirely.

---

## Methodology & honesty notes

- Tokenizer is `cl100k_base`; absolute % shift under a different vocabulary but the ranking is
  stable (terse's cross-tokenizer-invariance claim, tested separately in the suite).
- §1 corpus is real, public GitHub API output. §2 is **synthetic and seeded** — illustrative
  of a mechanism, not production-representative; the exact numbers depend on the construction
  (short keys, value cardinality), which is why §2's takeaway is "no crossover," not a constant.
- Every terse figure in §1–§4 is verified lossless per payload. §4's headroom and tamp rows,
  and §5's `codegraph_explore` row, are the only places a "reduction %" is reported for a
  mechanism that achieves it by *discarding* data — each flagged inline.
- Adoption honesty: terse has been on PyPI since v0.3.1 (2026-07-18) and is at v0.33.8;
  TOON, headroom and the other §4 entrants are far more established. terse's wedge is narrow
  and specific — *unconditionally lossless, no ML, no egress* — not breadth of adoption.
- **Star counts have been deliberately removed from these docs.** They were a popularity
  proxy that drifted constantly and never informed a decision about whether terse works.
  What a project *does*, measured here, is the comparison worth carrying.

### A token reduction is not a proportional cost reduction

This is the most important caveat in this document, and earlier editions did not state it.

Every percentage above is a **token** figure. Billed cost does not move with it one-for-one,
because context is not billed at one price. Measured 2026-09-22 over this author's own
Claude Code transcripts — **1,093 sessions, 118,288 messages carrying a `usage` block, last
30 days** — using Anthropic's published multipliers relative to base input price (fresh
input 1.00x, 5-minute cache write 1.25x, cache read 0.10x):

| component | raw input tokens | share of raw | billed-equivalent | share of billed |
|---|--:|--:|--:|--:|
| fresh input | 125,545,227 | 0.6% | 125,545,227 | 4.7% |
| cache write | 495,678,913 | 2.5% | 619,598,641 | 23.0% |
| **cache read** | **19,493,766,658** | **96.9%** | 1,949,376,666 | 72.3% |
| **total** | **20,114,990,798** | | **2,694,520,534** | |

**96.9% of raw input tokens are cache reads, billed at a tenth of input price. A raw-token
basis therefore overstates input cost by 7.47x.** Tool results are a large share of what is
*in* the context — 47.2% of message content by character, in the same window — so terse is
compressing a real and substantial part of it. But a 59% cut in tool-result tokens is not a
59% cut in the bill.

Where the saving is real: a smaller result shrinks the prefix that is re-read on every
subsequent turn, and it delays the point at which the window fills. Both matter. Neither is
the headline percentage.

This converges with published work reaching the same conclusion from the opposite direction
— notably *Token Reduction Is Not Cost Reduction* (arXiv:2607.12161), which measured a 38.4%
cut in delivered tool-output tokens alongside a **6.8% increase** in billed cost. **Citation
caveat, carried deliberately:** that paper is unreviewed and authored by a cloud-cost
optimisation vendor. It is cited here as convergent, not as authority; the table above is
this project's own measurement and does not depend on it.
