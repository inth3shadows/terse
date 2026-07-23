# terse — Benchmarks

**Last updated: 2026-07-22.** Every figure is dated by section — §1–4 were produced
2026-07-17, §5–6 on 2026-07-22 — and nothing here is hand-typed or estimated. If you
re-run and get different numbers, the code changed; open an issue.

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

| payload | records | raw tok | **terse** | TOON |
|---|--:|--:|--:|--:|
| gh_pulls | 30 | 151,165 | **76.1%** | −8.4% |
| gh_workflow_runs | 20 | 76,032 | **80.3%** | −7.5% |
| gh_issues | 30 | 48,032 | **32.7%** | −8.0% |
| gh_commits | 30 | 69,652 | **26.5%** | −4.5% |
| gh_dir_listing | 24 | 6,736 | **31.4%** | −7.7% |
| gh_rate_limit | 1 obj | 357 | **13.4%** | −36.7% |
| gh_repo_single | 1 obj | 1,652 | 0.0% | −4.4% |
| gh_commits_flat | 30 | 10,886 | **2.4%** | 1.7% |
| gh_labels | 9 | 632 | 15.2% | **19.0%** |
| **weighted total** | | **365,144** | **58.3%** | **−7.1%** |

**Plain reading:** on real nested records terse cuts tokens **58%**; TOON *regresses* to −7%
(worse than raw) because it adds a key-path per nesting level, while terse folds the repeated
subtrees and long repeated strings (e.g. `gh_pulls` = 60 copies of the same repo object
collapsed to one legend entry → 76%). TOON wins only on `gh_labels` — a flat, short-valued
uniform table, its designed sweet spot.

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
| gh_issues | 32,608 | 4,448 | **86.4%** |
| gh_pulls | 37,776 | 15,292 | **59.5%** |
| **weighted total** | 152,837 | 40,138 | **73.7%** |

**Honest caveat (read this):** these are *modeled* repeat-call savings. How *often* the
pattern occurs in a real agent loop is workload-dependent and is being measured directly (the
proxy now records a per-result `diff_reason` — run `terse stats` to see the breakdown for your
own traffic). Do **not** read §3 as a claim about aggregate real-world savings; read §1 for that.

---

## §4 — Competitor landscape (hands-on, tested 2026-07-17)

Installed and tested, not cited from marketing. Only TOON (§1) is directly comparable on a
lossless token axis; the rest measure *different guarantees*, so no head-to-head % is claimed.

| Tool | What it is (verified) | Comparable? |
|---|---|---|
| **headroom** (`headroom-ai`, v0.32.0) | JSON compressor is a **deterministic Rust transform, not ML**: lossless on uniform arrays, but **drops rows** on larger/irregular sets, recoverable only via a `retrieve` round-trip against a **time-boxed cache** (default 30-min TTL). Measured on our corpus: 33.1% (lossless reformat) to 42.5%/64.1% (lossy, dropping 13/30 and 13/20 rows). A separate optional text compressor *is* ML. | No — its larger numbers come from *dropping data* with time-limited recovery; terse's are unconditionally lossless. |
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

**Headline (measured 2026-07-22, `terse stats`, ledger spans 7 days):**

```
1,526 results   470,609 -> 427,378 tok   9.2% blended
```

That 9.2% is honest and *incomplete*, for two reasons — and both are the point of this section.

**1. This ledger straddles the #116 transition.** Most of its records predate cross-block
joining (the `multiblock` diff-reason bucket is still 345 of them), so it mostly reflects the
old per-block path. `terse stats` on post-#116 traffic reads higher for repeat-heavy loops
(below). The number is *composition*, not a constant — which is why we publish a range.

**2. Savings track payload shape.** Which tools you call sets the mix. Measured on the real
captured records (production policy, deduplicated to one call's worth per tool):

| shape | example tool | codec: per-block → joined (#116) | an *unchanged* repeat |
|---|---|--:|--:|
| wide, low-cardinality | `kb.read.changelog` | 21% → **38%** | ~99% |
| | `kb.read.recent_rejections` | 17% → **33%** | ~99% |
| | `kb.read.for_repo` | 15% → **24%** | ~98% |
| prose-heavy records | `kb.read.list_principles` | 3% → 3% *(hard ceiling)* | **~99.9%** |
| | `kb.read.get` | 2% → 2% | ~99.9% |
| already-projected small | `kb.read.query_stats` | 41% → 41% | — |
| tiny status objects | *(policy `tiers:[]`)* | 0% → 0% *(correct — already minimal)* | — |

The prose ceiling is structural, exactly as predicted: long unique text in
`principle`/`rationale`/`evidence` has nothing to fold, and no tier combination changes that.

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

**So the production figure is a range, not a point:**

- **Floor** — every call, no repeats: the joined codec alone, ~9–12%, dominated by prose
  ceilings.
- **Ceiling** — repeat-heavy loop with data stable between calls: re-weighting the ledger's
  own token mix, the diff-eligible 71% collapsing ~99% each puts the aggregate near **~71%**.
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

## §6 — Popular third-party MCP servers (measured 2026-07-22)

§5 is one person's traffic. This section is the other half: what terse does **automatically,
zero-config** to the output of widely-used, **credential-free** MCP servers that anyone can
run. Reproduce with `scripts/bench/mcp_servers/` (pinned repo fixtures, a static local web
fixture, one command per server).

Servers: the official reference set (`modelcontextprotocol/servers`, 88.8k★) plus the two
most-starred credential-free third-party servers — **serena** (26.7k★) and
**playwright-mcp** (35.4k★).

| server | tool | output shape | codec % (1-shot) | an *unchanged* repeat | reaches the model? |
|---|---|---|--:|---|---|
| filesystem | `directory_tree` | JSON, pretty-printed | **50–58%** | diff | ⚠️ no — see below |
| filesystem | `read_text_file` | source text | 0% | text-diff | ⚠️ no — see below |
| git | `git_log` | long text | 0% | text-diff | yes |
| memory | `read_graph`, `search_nodes`, `create_entities` | JSON | **40–42%** | — | ⚠️ no — see below |
| serena | `get_symbols_overview`, `find_symbol` | JSON, already compact | **18–22%** | diff | yes |
| playwright | `browser_snapshot` | accessibility tree (text) | 0% | text-diff | yes |
| fetch | `fetch` | markdown | 0% | text-diff | yes |

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

It splits along SDK generation, not by accident: the newer TypeScript servers declare
schemas on every tool. Expect this to grow.

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

**What survives this correction:** serena's 18–22% is the only non-zero one-shot codec
number in the table that reaches the model, and the diff tier still lands on `git`,
`serena`, `playwright` and `fetch`. The claim below — codec narrow, diff broad — holds;
it is narrower than first published, and the two rows carrying the biggest codec numbers
are the two that don't count.

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

### The headline: the codec is narrow, the diff tier is universal

Two things fall out, and they matter more than any single percentage:

1. **The one-shot codec pays only on JSON — and how much depends on whether the server
   pretty-prints.** filesystem pretty-prints its tree, so minify alone is most of 50–58%.
   memory returns compact-ish JSON records (40–42%). serena emits *already-compact* JSON, so
   only the structural fold is left (18–22%) — the hardest honest case, and exactly the
   "pure structural gain" framing at the top of this file.
2. **Every text-shaped tool is 0% on the codec — and every one still wins on an *unchanged*
   repeat.** `read_text_file`, `git_log`, `browser_snapshot`, and `fetch` are all
   uncompressible one-shot, yet all four emit a content-defined-chunking text diff the
   second time they are called.

**Read the repeat column as a ceiling, not a typical delta.** Both calls send identical
arguments against an unchanged fixture, so `prev == curr`: the diff encodes an empty
changeset and the wire is near-fixed overhead once the payload clears the small-payload
floor described below. That is the *upper bound* of the diff tier — the same discipline §5 applies when it reports ~99%
on an unchanged repeat and then frames production as a floor/ceiling range — note §5's
column is a *number* and this one is only qualitative (`diff` / `text-diff` / `—`). A real agent
loop re-fetches results that have **changed**, and how much the delta grows with the change
is workload-specific and **not measured here**.

So on third-party servers the **cross-call diff is the broad, shape-independent win, and the
codec is the JSON-specific one** — narrowed further by the `structuredContent` note above,
which removes both of the codec's biggest rows from what the model actually receives, and
leaves the diff tier landing on four of the six servers. That is a sharper claim than a blended average, and it
predicts where terse helps: agent loops that call the same tool repeatedly. Browser
automation is the shape it should suit best — navigate → snapshot → act → snapshot produces
consecutive, largely-overlapping accessibility trees. Stated as a *prediction*, not a
result: what was measured is an identical repeat; a post-click tree is a different and
untested experiment.

**Which command produces which column:** codec % comes from `terse measure --corpus`; the
repeat column comes from `terse stats --log` (its `diff_reason` breakdown). Capture is
content-addressed, so two identical repeats collapse into one corpus file — the corpus
alone can never evidence the repeat column, only the ledger can.

### Repo size barely moves the codec

`directory_tree` across three pinned fixtures — express v5.2.1 (218 files), fastapi 0.139.2
(3,131), django 5.2.16 (6,926):

| fixture | raw tok | codec % | repeat |
|---|--:|--:|---|
| express | 116 | 54.3% | diff not smaller (payload too small) |
| fastapi | 1,328 | 50.3% | **diff emitted** |
| django | 2,696 | 58.0% | **diff emitted** |

The codec sits in a **50–58% band across a 23× payload-size range** — it tracks JSON
structure, not repo size. What size *does* change is the **diff**: below roughly a thousand
tokens the delta loses to simply re-sending the compressed form; above it the diff wins and
keeps winning.

### Zero-config auto-policy holds up

`terse policy generate` was run against each captured corpus and authored a correct,
conservative, lossless policy every time with no hand-tuning: `directory_tree` →
`minify,tabularize` (dictionary auto-dropped as below the 5% threshold), `read_text_file` →
`tiers: []` passthrough (detected as non-JSON), memory's three record tools → all folded.
That is the "does it just work on a server it has never seen" question, answered yes.

### Transports: HTTP downstream and multi-peer fan-out

Everything above is a **stdio** downstream. terse also proxies an MCP **Streamable-HTTP**
endpoint and can front *N* servers from one process, and neither had third-party evidence.
Both were exercised against the reference `everything` server run in `streamableHttp` mode.
**Scope: a single run on 2026-07-22, not part of the pinned size sweep** — these establish
that the transports work end-to-end, not a measured savings result:

- **HTTP downstream** — `terse proxy -- http://127.0.0.1:3001/mcp`. `initialize`,
  `tools/list` (13 tools), `tools/call` and the capture tee all behave as on stdio; the URL
  form is selected automatically (a single target containing `://`).
- **Multi-peer fan-out with mixed transports** — one process fronting three peers,
  **two stdio + one HTTP**, via `proxy --config`:

  | check | result |
  |---|---|
  | merged `tools/list` | 36 tools, peer-prefixed (`fs`=14, `mem`=9, `ev`=13) |
  | `initialize` primer | injected **exactly once** across all peers |
  | call routing | each `peer__tool` reached its own peer, including the HTTP one |
  | per-peer compression | `fs__directory_tree` 54.3% (express v5.2.1 `lib/`), `mem__read_graph` 42.1% |
  | ledger attribution | per-peer, under the peer-qualified tool name |

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

## Methodology & honesty notes

- Tokenizer is `cl100k_base`; absolute % shift under a different vocabulary but the ranking is
  stable (terse's cross-tokenizer-invariance claim, tested separately in the suite).
- §1 corpus is real, public GitHub API output. §2 is **synthetic and seeded** — illustrative
  of a mechanism, not production-representative; the exact numbers depend on the construction
  (short keys, value cardinality), which is why §2's takeaway is "no crossover," not a constant.
- Every terse figure is verified lossless per payload. §4's headroom row is the only place a
  "reduction %" is reported for a tool that achieves it by *discarding* data — flagged inline.
- Adoption honesty: terse is new (pre-PyPI as of this date); TOON and headroom are far more
  established. terse's wedge is narrow and specific — *unconditionally lossless, no ML, no
  egress* — not breadth of adoption.
