# When to use terse — and when not to

terse has never stated this plainly, and a full measurement session (2026-07-28)
showed why it needs to: the same codec measures **58.5%** on public GitHub API
payloads and **6.8%** on this operator's own personal MCP fleet. Both numbers are
correct — they are the same tool measured on two different shapes of input. Quoted
without the shape attached, either one misrepresents what terse does. This doc is
that attachment.

## The economic model, stated once

terse's cost and its payoff are charged on different clocks:

- **Savings are paid ONCE, per tool call.** Compress a payload, bank the tokens.
- **The primer is paid EVERY TURN, PER WRAPPED SERVER.** 212 cl100k tokens (402
  before #170) are injected into that server's MCP `initialize.instructions` and
  re-read by the client every request — whether or not that server is actually
  called on that turn.

Break-even at one wrapped server therefore needs **212 tokens of savings per
turn**. That single line explains every result below, with no A/B harness required
to evaluate a new candidate server: estimate its typical payload size and call
frequency, multiply, compare to 212.

For a router/multiproxy setup wrapping several servers behind one shared primer, the
break-even arithmetic is different in *kind*, not degree. The router's
`union_primer` is a boolean OR over five fixed sections, not a concatenation of N
per-peer primers — so it is O(1) in peer count, not O(N):

| section | tokens |
|---|--:|
| head | 41 |
| table | 155 |
| dict | 44 |
| embedded | 53 |
| diff | 190 |
| dropped | 64 |
| tail | 8 |
| **full (all sections gated on)** | **555** |

555 cl100k tokens is the hard ceiling, paid once at `initialize`, regardless of
whether 1 or 20 peers sit behind the router — the opposite shape of the standalone
case #211 fixed, where N wrapped servers meant N separate primers riding N
`initialize` replies, scaling linearly with server count. A pre-#211 A/B run at 6
idle peers behind a router (same code path, unchanged since) measured +4.5%
weighted, inside the noise floor, versus +17.4% weighted for the same 6 servers
standalone. Full measurement: #212, closed as no-op — no regression found, no code
change needed.

## Where terse pays off — public API servers

Measured from `scripts/bench/corpus/` (real GitHub REST payloads), current codec:

| payload | raw tok | saved | saved% | calls/turn to break even |
|---|--:|--:|--:|--:|
| gh_pulls | 152,047 | 115,861 | 76.2% | 0.002 |
| gh_workflow_runs | 76,260 | 61,318 | 80.4% | 0.003 |
| gh_commits | 69,910 | 18,702 | 26.8% | 0.011 |
| gh_issues | 48,811 | 16,483 | 33.8% | 0.013 |
| gh_dir_listing | 6,736 | 2,114 | 31.4% | 0.100 |
| gh_commits_flat | 11,015 | 387 | 3.5% | 0.548 |
| gh_labels | 632 | 96 | 15.2% | 2.208 |
| gh_rate_limit | 357 | 48 | 13.4% | 4.417 |
| gh_repo_single | 1,655 | 3 | 0.2% | 70.7 |
| **weighted** | **367,423** | **215,012** | **58.5%** | |

Mean 23,890 tokens saved per call. One `gh_pulls` call pays the primer for roughly
546 turns.

## Where it does not — pre-projected personal servers

From the live proxy ledger (1,999 blocks, 13.3 days), same codec:

| tool | calls | saved/call | calls/turn to break even |
|---|--:|--:|--:|
| kb.read.list_nodes | 6 | 2,176 | 0.1 |
| secret.list_credentials | 10 | 2,080 | 0.1 |
| runecho structure (large) | 6 | 898 | 0.2 |
| codegraph_explore | 7 | 440 | 0.5 |
| kb.read.get | 45 | 99 | 2.1 |
| kb.read.list_principles | 962 | 39 | 5.4 |
| kb.read.search | 188 | 28 | 7.5 |
| runecho structure (small) | 190 | 6 | 35.7 |
| secret.* proxy ops | 54 | 0 | never |

Mean 53 tokens saved per call — three orders of magnitude below the public corpus.
The highest-*volume* tools here are also the *worst* compressors. The live policy
already says why for kb: "already field-projected + high-cardinality content."
These servers pre-optimize their own output before terse ever sees it, so there is
nothing structural left to remove.

## The rule

terse pays off in proportion to how raw and verbose a server's output is. It is a
big-payload tool.

> A server that already projects its fields is not a terse candidate, no matter
> how often it is called.

Concretely: **wrap a server when its typical payload saves more than
`212 * (turns per call)` tokens.** Do not wrap it otherwise — the primer is charged
whether the server is called that turn or not.

## codegraph — a third category, not an average

`codegraph_explore` doesn't fit either table above cleanly, and averaging it into
one would hide why it wins. Its payload is markdown-plus-source, not JSON — the
JSON-tier codec measures 0.0% on it because it cannot parse it at all. Its real
saving comes from a dedicated lossy rule (`$text.code_blocks -> terse.retrieve`,
#139) — the fleet's *only* lossy-by-default rule — which evicts the fenced-source
field for a ~90% saving on that field. The 440 tokens/call it shows in the personal
fleet table above is only the JSON-tier's leftover; the drop rule is where the real
win lives, and it isn't represented in either table's methodology.

CodeGraph is structurally closer to the GitHub-API case than to the kb case: it's a
code-intelligence server whose output is inherently large and un-projected. Its win
comes from eviction rather than encoding, so it deserves its own axis rather than a
place in either table — a non-JSON payload class terse handles, but doesn't yet
measure the same way it measures JSON.

## Future work

Router-level primer economics are resolved (see above, #212, closed as no-op): the
shared `union_primer` is bounded at ~555 tokens independent of peer count, not a
scaling liability. Diff-tier defaults and the mcp-status classifier that surfaces
this trade-off at wrap time are tracked in #168 and #174.

## Related

- [README.md](../README.md) — "Does terse help my server?" quick heuristic table
- [BENCHMARKS.md](../BENCHMARKS.md) — full dated numbers behind the tables above
  (§5 live ledger, §6 third-party servers)
- [TECHNICAL.md](../TECHNICAL.md) — policy schema, pipeline, known limitations
