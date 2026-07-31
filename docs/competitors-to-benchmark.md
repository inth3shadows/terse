# Competitors to benchmark — working backlog

**Status: backlog, not published claims.** Nothing here has been installed and
measured. It is the queue feeding BENCHMARKS.md §4, whose bar is *"installed and
tested, not cited from marketing."* A candidate graduates from this file to §4 (and
to the README's "not benchmarked head-to-head" table) only after a hands-on run.

Numbers quoted below are **the vendor's own claims**, attributed as such. Do not
copy them into README.md or BENCHMARKS.md as if they were reproduced here.

Last swept: 2026-07-31 (Glama directory + web).

---

## Screening axis

**Popularity is not a screen.** Entries stay in this file with zero stars, zero
downloads, and no README — terse itself has no adoption, and a list that filters on
stars would exclude terse. What decides inclusion is the *axis* a tool works on and
whether the comparison is honest. Star counts are recorded as context, and are
labelled unverified unless someone checked.

terse competes on one narrow axis: **re-encoding tool-call results, losslessly, in a
transparent stdio proxy, with no network egress and no expiring retrieve cache.**

Every candidate gets sorted by which of those it does *not* do:

| Axis | Meaning |
|---|---|
| **result-side** | Touches the tool's returned payload (terse's axis) |
| **schema-side** | Only shrinks tool *definitions* at connect time — complementary, stackable |
| **avoidance** | Stops the payload reaching context at all (code execution, history pruning) |
| **agent-callable** | A tool the model must explicitly invoke — not transparent |

---

## Tier 1 — benchmark these

### Code Mode / code execution with MCP

- Cloudflare Code Mode (Agents SDK, open source) and Anthropic's
  "code execution with MCP" pattern.
- **Axis:** avoidance. The model writes code against a typed SDK; the code filters
  results server-side, so raw payloads never enter the context window.
- **Their claims:** Anthropic reports 150k → 2k tokens (98.7%) on a Drive→Salesforce
  workflow; Cloudflare reports ~81% vs direct tool calling, and an entire API
  exposed in ~1,000 tokens via `search()` + `execute()`.
- **Why it matters to positioning:** this is the strongest "why not just do this
  instead?" a reader will raise, second only to native context editing. It is
  *lossy by selection* (the model decides which fields survive) and needs a code
  sandbox plus a scriptable API — it is not a drop-in for an arbitrary stdio MCP
  server. terse and code mode are complementary in principle: code mode avoids the
  payload, terse compresses whatever still reaches the model.
- **To measure:** not a token bake-off (different mechanism). The honest comparison
  is *fidelity*: what fraction of a real payload's fields does a model-written
  filter drop that a later turn then needs? That is terse's fidelity axis (issue
  #138) applied to a rival.
- Links: <https://blog.cloudflare.com/code-mode-mcp/>

### token-optimizer-mcp (ooples)

- ~456★ as listed on GitHub, 2026-07-31 (unverified).
- **Axis:** agent-callable + result-side. Lossless **Brotli** compression with
  cache-and-retrieve-by-key, plus its own `smart_read` / `smart_grep` / `smart_glob`
  replacements for the built-in file tools. `optimize_text` is the generic path and
  can be pointed at any server's output. Claims 60–90% context reduction,
  "2–4x typical, up to 82x for repetitive content."
- **Why it matters to positioning:** it is the closest *mechanism* rival and it
  sharpens terse's wedge rather than blunting it. Two structural differences to
  verify and then state plainly:
  1. **Not a proxy.** The agent has to call its tools; nothing happens
     transparently to an existing MCP server.
  2. **Brotli output is not model-readable.** So it is eviction-to-cache — the
     content leaves context and comes back only via a `get_cached` round trip.
     terse's compressed form is *still read directly by the model* (see the
     fluency results). That is the same distinction already drawn against
     headroom's 30-min-TTL retrieve cache, and it should be drawn here too.
- **To measure:** whether `optimize_text` on our GitHub corpus leaves anything
  readable in context, and what the cache's lifetime/backend is (persistent? TTL?
  process-scoped?). If it is process-scoped, the losslessness claim carries the
  same asterisk as headroom's.
- Links: <https://github.com/ooples/token-optimizer-mcp>

---

## Tier 2 — one line each, no benchmark warranted

### Schema-side proxies (a category now, not one entrant)

`atlassian-labs/mcp-compressor` (~97★) already has a README row. Add siblings to
that row rather than new rows:

- **mcpproxy-go** (smart-mcp-proxy, ~305★) — federates many servers behind one
  `retrieve_tools` call; claims ~99% token reduction and +43% accuracy, all of it
  on tool *definitions*. Results untouched.
- **Tool Filter MCP** (`respawn-llc/tool-filter-mcp`, 36★) — exposes a filtered
  subset of tools.

All complementary and stackable with terse; none competes on the result axis.

## Tier 3 — logged, off-axis (kept, not dropped)

Surfaced by the Glama sweep. These are **not** excluded for being small or unknown —
see the screening note above. They are here because of *what they do*, and each is a
one-line entry someone can promote if that changes.

| Candidate | Axis | Note |
|---|---|---|
| entroly-context-engine (433★) | result-side (claimed) | Claims 78% average via compression + reinforcement learning. Would be Tier 1 *if the claim held on structured tool output* — the mechanism is on-axis, the evidence is a marketing line. 433★ is real adoption, more than most of this table; worth 20 minutes to install and point at the GitHub corpus, then promote or park with a reason. |
| Agent Context Optimizer MCP (1★) | result-side (claimed) | Claims "up to 85% less context waste". Same treatment as entroly — unverified, on-axis, cheap to check. |
| mcp-compress (ShipItAndPray) | agent-callable | Returns gzip/brotli/deflate bytes with lossless verification. Same non-readable-output problem as token-optimizer-mcp, without the caching design. Useful as a *second* data point that "compress the bytes" is a recurring wrong turn. |
| JSON Skeleton MCP | agent-callable, lossy | Truncates string values and dedupes arrays. Comparable to terse's Tier 1 lossy `truncate` mode, not to the codec. |
| shared-context-cache-mcp | avoidance | Cross-session cache dedup, not an encoding. |
| context-diamond | avoidance | Compresses *handoffs* into structured capsules — a summarization product. |
| mcp-json, mcp-json-tools, json-mcp, mcp-code-context | agent-callable | JSON/AST utilities the model invokes. Not proxies, not compressors. Listed so a later sweep doesn't re-surface them as new. |

**Structural finding from the sweep:** the MCP directories cannot find terse's real
rivals. headroom, TOON, and LLMLingua-2 are not MCP servers, so a Glama query
returns only the long tail of agent-callable JSON helpers. Directory search is a
poor discovery channel here; release notes and the token-optimization blog genre
are better.

---

## Glama "alternatives" list — verified identifiers

**Verified 2026-07-31.** Every row below was checked twice: the repo against the
GitHub API (`gh api repos/<owner>/<name>`), and the Glama listing by fetching
`https://glama.ai/mcp/servers/<owner>/<name>` and reading the page `<title>`.

*Method note, because the obvious check is wrong:* Glama is an SPA that returns
**HTTP 200 for a nonexistent server**, so status codes prove nothing. Its 404 shell
is a constant **21,211 bytes** (calibrated against a deliberately fake slug) — that
size, or a `<title>` naming a *different* project, is the real signal.

### Listed on Glama — selectable now

| Pick | GitHub `owner/name` | ★ | Glama page title |
|---|---|--:|---|
| mcp-compressor | `atlassian-labs/mcp-compressor` | 104 | mcp-compressor by atlassian-labs |
| token-optimizer-mcp | `ooples/token-optimizer-mcp` | 456 | token-optimizer-mcp by ooples |
| mcpproxy-go | `smart-mcp-proxy/mcpproxy-go` | 305 | mcpproxy-go by smart-mcp-proxy |
| Tool Filter MCP | `respawn-llc/tool-filter-mcp` | 36 | Tool Filter MCP by respawn-llc |
| entroly-context-engine | `juyterman1000/entroly` | 433 | entroly-context-engine by juyterman1000 |
| Agent Context Optimizer | `AiAgentKarl/agent-context-optimizer-mcp` | 1 | Agent Context Optimizer MCP by AiAgentKarl |
| shared-context-cache | `AiAgentKarl/shared-context-cache-mcp-server` | 1 | shared-context-cache-mcp-server by AiAgentKarl |
| context-diamond | `RainCherb/context-diamond` | 20 | context-diamond by RainCherb |
| mcp-compress | `ShipItAndPray/mcp-compress` | 2 | mcp-compress by ShipItAndPray |
| JSON Skeleton MCP | `jskorlol/json-skeleton-mcp` | 10 | JSON Skeleton MCP Server by jskorlol |
| mcp-code-context | `achatainga/mcp-code-context` | 3 | mcp-code-context by achatainga |
| mcp-json | `pipeworx-io/mcp-json` | 0 | mcp-json by pipeworx-io |
| JSON Filter MCP | `kehvinbehvin/json-mcp-filter` | 25 | JSON Filter MCP by kehvinbehvin |
| mcp-json-tools | `rog0x/mcp-json-tools` | 0 | mcp-json-tools by rog0x |
| MCP Code Execution Mode | `elusznik/mcp-server-code-execution-mode` | 338 | MCP Server Code Execution Mode by elusznik |
| TOON MCP Server | `elminson/toon-mcp` | 3 | TOON MCP Server by elminson |

Two zero-star entries sit in that table on purpose. See the screening note: terse has
none either.

### NOT on Glama — the two biggest competitors, and why

| Project | GitHub `owner/name` | ★ | Why absent |
|---|---|--:|---|
| headroom | `headroomlabs-ai/headroom` | **63,510** | Confirmed 404 (21,211-byte shell). It is a library + proxy that *also* ships an MCP server, but nobody has listed the repo in the registry. |
| TOON | `toon-format/toon` | **25,035** | Confirmed 404. TOON is an encoding, not a server — structurally unlistable. |

This is the finding that matters: **the two projects terse most needs to be compared
against cannot be picked in Glama at all.** The registry indexes MCP servers, and
terse's real rivals aren't MCP servers. Any Glama alternatives list is therefore a
partial picture by construction, and the README table — not Glama — stays the
authoritative comparison.

Options if a Glama-native stand-in is wanted, both imperfect:

- Submit `headroomlabs-ai/headroom` to the registry ourselves. It does ship a server,
  so it plausibly qualifies. Listing a competitor is a real (small) favour to them.
- Pick a third-party TOON *wrapper* instead of TOON: `elminson/toon-mcp` (3★, listed),
  or `mhabedini/json-to-toon-mcp-server`, `HasnainAli47/toon-mcp-server`,
  `v3nom/toon-fetch` (unverified). This compares terse to someone's wrapper rather
  than to TOON, so only do it knowingly.

### Traps found while verifying

- **`microsoft/LLMLingua` is a false positive.** That Glama path returns a live page
  titled *"DebugMCP by microsoft"* — a different repo by the same owner. LLMLingua is
  not on Glama. (It should not be listed regardless: wrong axis, input prompts.)
- **`cloudflare/agents` resolves, but titled *"Cloudflare MCP Server by cloudflare"***
  — not obviously the Agents SDK that carries Code Mode. Check by eye before picking;
  `elusznik/mcp-server-code-execution-mode` (338★) is the cleaner stand-in for the
  *pattern*.
- **`respawn-app/tool-filter-mcp` does not exist** — the owner is `respawn-llc`. The
  wrong form 404s on both GitHub and Glama.
- **`kehvinbehvin/json-mcp` redirects** to `json-mcp-filter`; use the canonical name.

### Star counts, corrected

Verification moved four numbers that were in this repo's docs. All now reflect the
GitHub API on 2026-07-31: headroom **63.5k** (README previously said "~29–49k,
unverified"), TOON **25.0k** (was 24.9k), mcp-compressor **104** (was 97),
LLMLingua **6.5k** (was 6.4k). `entroly` has **433★**, not the "no adoption" this
file asserted in its first draft.
