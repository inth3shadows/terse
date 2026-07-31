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
- **Tool Filter MCP** (respawn-app) — exposes a filtered subset of tools.

All complementary and stackable with terse; none competes on the result axis.

## Tier 3 — logged, off-axis (kept, not dropped)

Surfaced by the Glama sweep. These are **not** excluded for being small or unknown —
see the screening note above. They are here because of *what they do*, and each is a
one-line entry someone can promote if that changes.

| Candidate | Axis | Note |
|---|---|---|
| entroly-context-engine | result-side (claimed) | Claims 78% average via compression + reinforcement learning. Would be Tier 1 *if the claim held on structured tool output* — the mechanism is on-axis, the evidence is a marketing line. Worth 20 minutes to install and point at the GitHub corpus; promote or park with a reason. |
| Agent Context Optimizer MCP | result-side (claimed) | Claims "up to 85% less context waste". Same treatment as entroly — unverified, on-axis, cheap to check. |
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

## Glama "alternatives" list — exact identifiers

Glama's picker matches against **its own registry of MCP servers**, so an entry is
only selectable if someone has listed that repo there. Two consequences worth
knowing before hunting: TOON and headroom are *not primarily MCP servers* (TOON is an
encoding; headroom is a library + proxy that also ships a server), so a Glama search
for them surfaces **third-party wrappers, not the upstream project** — pick those
only knowingly. LLMLingua-2 is not on Glama at all.

Slugs below are `owner/name` as they appear on GitHub. Glama paths sometimes carry an
`@` prefix (`@owner/name`) and sometimes not — both forms resolve.

### Directly on-axis — pick these first

| Pick | GitHub `owner/name` | Glama path | State |
|---|---|---|---|
| Atlassian mcp-compressor | `atlassian-labs/mcp-compressor` | `/mcp/servers/@atlassian-labs/mcp-compressor` | Tested (§4). Safe to list. |
| token-optimizer-mcp | `ooples/token-optimizer-mcp` | search "token-optimizer-mcp" | Untested. Listed in README with an explicit not-installed caveat. |
| mcpproxy-go | `smart-mcp-proxy/mcpproxy-go` | `/mcp/servers/smart-mcp-proxy/mcpproxy-go` | Schema-side, complementary. |
| Tool Filter MCP | `respawn-app/tool-filter-mcp` | `/mcp/servers/@respawn-app/tool-filter-mcp` | Schema-side, complementary. |

### Upstream projects — not Glama-native

| Project | GitHub `owner/name` | Note |
|---|---|---|
| headroom | `headroomlabs-ai/headroom` | Closest product competitor, tested (§4). Its README now says "Library, proxy, **MCP server**" — so a first-party server entry may exist; check before settling for a wrapper. **Also re-check the star figure:** one aggregator now cites 63.3k against our README's "~29–49k, unverified". |
| TOON | `toon-format/toon` | The encoding itself, benchmarked in §1. Not an MCP server. |
| LLMLingua-2 | `microsoft/LLMLingua` | Different axis (input prompts). Do not list as an alternative. |

Third-party TOON MCP wrappers, if a Glama-native TOON entry is wanted — these are
*other people's* wrappers, so listing one compares terse to a wrapper rather than to
TOON: `elminson/toon-mcp`, `mhabedini/json-to-toon-mcp-server`,
`HasnainAli47/toon-mcp-server`, `v3nom/toon-fetch`.

### Tier 3 slugs — list them too

Zero stars is not a reason to omit (see the screening note). Exact identifiers:

`juyterman1000/entroly` (entroly-context-engine) ·
`AiAgentKarl/agent-context-optimizer-mcp` ·
`AiAgentKarl/shared-context-cache-mcp-server` ·
`RainCherb/context-diamond` ·
`ShipItAndPray/mcp-compress` ·
`jskorlol/json-skeleton-mcp` ·
`achatainga/mcp-code-context` ·
`pipeworx-io/mcp-json` ·
`kehvinbehvin/json-mcp` ·
`rog0x/mcp-json-tools`

### Code Mode

`cloudflare/agents` carries the Code Mode SDK; it is an SDK, not a listed server.
The Glama-native stand-in is `elusznik/mcp-server-code-execution-mode`. Pick that
only if the listing should represent the *pattern* rather than Cloudflare's
implementation.

**Every slug on this page came from a directory listing or search result, not from a
repo I opened.** Confirm each resolves before selecting it in Glama.
