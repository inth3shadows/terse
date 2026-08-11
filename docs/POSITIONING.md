# When to use terse — and when not to

terse has never stated this plainly, and a full measurement session showed why it
needs to: the same codec measures **59.1%** on public GitHub API payloads
(`BENCHMARKS.md` §1, re-measured 2026-08-04) and **15.1%** on this operator's own
personal MCP fleet, all-time (`terse stats`, 2,357 blocks, snapshot 2026-08-11 —
re-run the command for a current figure, this one will drift). Both numbers are
correct — they are the same tool measured on two different shapes of input. Quoted
without the shape attached, either one misrepresents what terse does. This doc is
that attachment.

The personal-fleet figure moves with call composition, not just time: a handful of
large, multi-block calls routed through the multiproxy fleet dominate the token
total and pull the blended number up from what a typical single small call sees.
Re-run `terse stats` on your own ledger rather than trusting this snapshot once
your fleet's mix has changed (new servers wrapped, a router added, etc.).

## The economic model, stated once

terse's cost and its payoff are charged on different clocks:

- **Savings are paid ONCE, per tool call.** Compress a payload, bank the tokens.
- **The primer is paid ONCE, PER SESSION, lazily.** As of #211 (`lazy_primer=True`,
  the CLI's actual default — see `run_proxy` in `src/terse/proxy.py`), a standalone
  wrapped server no longer injects its primer into `initialize.instructions` at
  all. It attaches once, to the first `tools/call` result that actually carries a
  terse wire form. A session that never calls a wrapped tool pays zero primer
  bytes, instead of paying servers × turns — the architecture a pre-#211 version of
  this doc described.

Break-even at one wrapped server is therefore a ONE-TIME question, not a per-turn
one: does this server's typical call save more tokens than the primer it will
attach to costs? A policy with no rules at all pays **248 cl100k tokens** — head
41 + table 155 + dict 44 + tail 8 — because `diff` is off by default (#170) and
`embedded` / `dropped` are gated per server. Add the 64-token dropped-field
paragraph and it is **312**.

Which of the two a given server pays is decided by `Policy.has_drop`, and the
answer is not "does this server have a drop rule of its own". That gate is
deliberately conservative (#168/#199): under a policy that contains **any**
drop-to-retrieve rule — `policy.example.json` and the live fleet policy both do —
a server pays 312 unless an earlier rule covers it or it is structurally
never-lossy — 312 is the fallthrough, and 248 is what requires an earlier covering
rule or never-lossy status. That is the opposite of what "carries a drop rule" would
suggest: `kb` has no drop rule of its own and pays 312, while `gh` pays 248 because
`gh.*` terminates the walk first. `terse stats` prints the real figure for each wrapped
server; do not infer it from the rules by eye.

The 555 in the router table below is the all-gates-on ceiling: reachable for a
router whose peers collectively enable every gate, not for a standalone entry
under a default policy.

A single call that saves more than that server's own primer repays the entire
session's primer cost by itself; every call after that — and every call in a
session where the server is never invoked at all — costs nothing further toward
the primer. There is no recurring ratio to maintain, because #211 removed the
recurring charge.

**The question is per server, not per install.** Each wrapped server attaches its
own primer, once per session, so a six-server fleet pays six of them. What #211
removed was the *turns* factor, not the *servers* factor — standalone cost went
from `servers x turns` to `servers x 1`. Only the router is O(1) in peer count — but
it pays that O(1) primer EVERY TURN rather than once per session, so at any real turn
count a router is the more expensive shape, not the cheaper one. Consolidate for the
operational reasons (one policy, one process, one permission surface), not to save
tokens; `USAGE.md` and `install-mcp --multiproxy` say the same.

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

555 cl100k tokens is the hard ceiling, sent once at `initialize` — and then
re-read every turn as `cache_read`, so unlike a standalone entry's lazy primer
it is a RECURRING charge, paid from the first turn whether or not any peer is
ever called. The diff paragraph alone is 190 of 555 cl100k tokens, so a router at
that ceiling spends 34% of a recurring charge explaining one wire form — which is
the shape of the cost #170 weighed. A router that enables diffing and nothing else
does not sit at the ceiling: it pays 438, of which the same paragraph is 43%. That is why
`terse stats` reports it under a separate cadence from the standalone one, and never
sums the two. The ceiling holds regardless of
whether 1 or 20 peers sit behind the router — the opposite shape of the standalone
case #211 fixed, where N wrapped servers meant N separate primers riding N
`initialize` replies, scaling linearly with server count. A pre-#211 A/B run at 6
idle peers behind a router (same code path, unchanged since) measured +4.5%
weighted, inside the noise floor, versus +17.4% weighted for the same 6 servers
standalone. Full measurement: #212, closed as no-op — no regression found, no code
change needed.

## Where terse pays off — public API servers

Measured from `scripts/bench/corpus/` (real GitHub REST payloads) via
`uv run scripts/bench/benchmark.py`, current codec — identical to `BENCHMARKS.md`
§1:

| payload | raw tok | saved | saved% | calls to clear the one-time 248-tok primer |
|---|--:|--:|--:|--:|
| gh_pulls | 151,165 | 114,979 | 76.1% | 0.002 |
| gh_workflow_runs | 76,032 | 61,090 | 80.3% | 0.004 |
| gh_issues | 48,032 | 18,629 | 38.8% | 0.013 |
| gh_commits | 69,652 | 18,444 | 26.5% | 0.013 |
| gh_dir_listing | 6,736 | 2,114 | 31.4% | 0.117 |
| gh_commits_flat | 10,886 | 258 | 2.4% | 0.961 |
| gh_labels | 632 | 96 | 15.2% | 2.583 |
| gh_rate_limit | 357 | 48 | 13.4% | 5.167 |
| gh_repo_single | 1,652 | 0 | 0.0% | never (lossless, nothing to compress) |
| **weighted** | **365,144** | **215,658** | **59.1%** | |

This table prices the primer at 248 — what a rules-free policy pays, and what `gh`
pays under `policy.example.json` because `gh.*` sits ahead of that policy's drop
rule and ends the walk. A server whose walk instead reaches the dropped-field
paragraph pays 312, which raises every figure in the last column by 26%.

Mean 23,962 tokens saved per call. A single `gh_pulls` call alone saves 114,979
tokens — 464x the entire one-time 248-token primer — so it clears the whole
session's primer cost by itself; every call before or after that, on this server
or in any session where it's never invoked, adds nothing further to the primer
side of the ledger.

## Where it does not — pre-projected personal servers

From the live proxy ledger (2,357 blocks, spans 2026-07-15 to 2026-08-11, snapshot
2026-08-11 — `terse stats`), same codec:

| tool | calls | saved/call | its server's primer | calls to clear it |
|---|--:|--:|--:|--:|
| kb.read.list_nodes | 27 | 3,351 | 312 | 0.093 |
| codegraph_explore | 72 | 2,797 | 312 | 0.112 |
| secret.list_credentials | 10 | 2,080 | 248 | 0.119 |
| kb.read.list_principles | 875 | 358 | 312 | 0.872 |
| kb.read.get | 100 | 67 | 312 | 4.63 |
| kb.read.search | 251 | 47 | 312 | 6.71 |
| runecho structure | 229 | 29 | 248 | 8.45 |
| secret.* proxy ops | 59 | 0 | 248 | never |

The primer column is per server, and rule ORDER decides it — not whether a server
has a drop rule of its own. `codegraph` carries the example policy's only
drop-to-retrieve rule. `kb` sits *after* it in the walk and so inherits the
64-token dropped-field paragraph at 312, while dropping nothing itself; `runecho`
sits *before* it and pays 248; `secret-broker` is structurally never-lossy (#199),
which suppresses the dropped-field paragraph specifically — the remaining 248 comes
from whatever grants its tiers, which is a carve-out rule in the live policy and
`defaults` in the example policy, where it matches no rule at all. Both the live policy and
`policy.example.json` produce exactly these values.

Mean 301 tokens saved per call across the whole ledger (710,314 saved / 2,357
blocks) — roughly two orders of magnitude below the public corpus (23,962), not
three; that gap narrowed as the fleet's call mix shifted toward bigger calls.

The relationship between call volume and compression quality used to be uniformly
inverse here — the more a tool was called, the worse it compressed. That has
broken for `kb.read.list_principles` specifically: at 875 calls it is now the
single most-called tool in the ledger, and it no longer compresses worst —
currently 358 tokens/call, ahead of both `kb.read.get` (67/call, 100 calls) and
`kb.read.search` (47/call, 251 calls), which remain the pattern's clearest
holdouts. The live policy's stated reason for `kb` overall — "already
field-projected + high-cardinality content" — is still true for `kb.read.get` and
`kb.read.search`; it has not stopped being a real constraint for those two. What
changed for `list_principles` is call shape: a growing share of its raw tokens now
arrive as large, multi-block calls routed through multiproxy (`BENCHMARKS.md` §5
has the breakdown), and the codec's tabularize tier folds that shape far better
than the small single-record calls this section originally measured. **This is a
composition effect, not evidence the codec learned to compress prose better** — a
one-shot `terse compress` on an isolated captured payload for this tool still
lands close to the original 3% (see `BENCHMARKS.md` §5).

## The rule

terse pays off in proportion to how raw and verbose a server's output is. It is a
big-payload tool.

> A server that already projects its fields is not a terse candidate, no matter
> how often it is called.

Concretely: **wrap a server when its typical session-lifetime savings clear its
own one-time primer — 248 tokens, or 312 where the dropped-field paragraph is
reachable for it.** That can be a single big call (`gh_pulls` clears it 464x over
on its own) or accumulate across a session — on the current ledger,
`kb.read.list_principles` clears its own primer in under one call on average
(0.87), a genuine change from an earlier snapshot where it took about 6; see the
composition caveat above before reading that as the tool having gotten "better."
Once cleared, every further call — in
that session or any other session where the server never gets invoked at all —
costs nothing more toward the primer. Do not wrap a server whose realistic session
savings can't clear ~250; that bar is paid once per session, not every turn, and
each wrapped server carries its own. Read the exact figure off `terse stats`
rather than inferring it from the policy rules — as the 248/312 split above shows,
rule order decides it and the answer is easy to get wrong by eye.

This rule now has a machine rollup: `terse stats --recommend` prints the comparison
above as one word per **installed entry** (`KEEP` / `TUNE` / `UNWRAP` /
`INSUFFICIENT`) beside the coverage ratio it was derived from. Installed entry, not
peer — a router pays one union primer for its whole fleet. No numbers here change;
it is the same comparison, made once by the tool instead of by eye.

## codegraph — a third category, not an average

`codegraph_explore` doesn't fit either table above cleanly, and averaging it into
one would hide why it wins. Its payload is markdown-plus-source, not JSON — the
JSON-tier codec measures 0.0% on it because it cannot parse it at all. Its real
saving comes from a dedicated lossy rule (`$text.code_blocks -> terse.retrieve`,
#139) — the fleet's *only* lossy-by-default rule — which evicts the fenced-source
field for a ~90% saving on that field. The 1,589 tokens/call it shows in the
personal fleet table above is only the JSON-tier's leftover; the drop rule is
where the real win lives, and it isn't represented in either table's methodology.

CodeGraph is structurally closer to the GitHub-API case than to the kb case: it's a
code-intelligence server whose output is inherently large and un-projected. Its win
comes from eviction rather than encoding, so it deserves its own axis rather than a
place in either table — a non-JSON payload class terse handles, but doesn't yet
measure the same way it measures JSON.

## Future work

Router-level primer economics are resolved (see above, #212, closed as no-op): the
shared `union_primer` is bounded at ~555 tokens independent of peer count, not a
scaling liability. Standalone-server primer economics are resolved too (#211,
lazy primer). There is no open tracker for diff-tier defaults or an mcp-status
classifier surfacing this trade-off at wrap time — #168 (per-server primer
gating) and #172 (mcp-status stash-membership classification) are both closed;
nothing currently open covers either follow-up.

## Related

- [README.md](../README.md) — "Does terse help my server?" quick heuristic table
- [BENCHMARKS.md](../BENCHMARKS.md) — full dated numbers behind the tables above
  (§5 live ledger, §6 third-party servers)
- [TECHNICAL.md](../TECHNICAL.md) — policy schema, pipeline, known limitations
