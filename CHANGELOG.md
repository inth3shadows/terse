# Changelog

All notable changes to terse are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from git tags (`vX.Y.Z`, via hatch-vcs) — an entry moves from
`[Unreleased]` to a versioned section when its tag is pushed.

## [Unreleased]

### Added
- **`policy generate` / `autotune` can now recommend the `embedded` tier.** It shipped
  opt-in but invisible to the generator, so nothing would ever turn it on and an operator
  had to hand-edit a policy — which #144 is the standing proof goes stale, since nothing
  re-derives a hand-authored tier decision after a codec change lands under it. `measure`
  now reports `embedded` as its own marginal step (exactly like `dictionary`), and the
  generator adds the tier per tool when that margin clears the threshold: a double-encoding
  tool scores `41.0% saved (embedded +41.0%)` and gains it, an ordinary record tool scores
  `+0.0%` and is not offered it — nor charged its primer paragraph. `tier_total` counts the
  embedded step deliberately: a tool whose body is one JSON string saves ~0% under the other
  tiers, so scoring it without `embedded` would mark it passthrough and permanently hide the
  very tool the tier exists for.
- **New `embedded` tier: compress JSON the server delivered as a STRING.** `minify`,
  `tabularize` and `dictionary` all walk parsed structure, so a body returned double-encoded
  (`{"response_text": json.dumps(body)}`) is a leaf none of them can reach. Measured on
  identical data: **41.9% saved as a real record array, 0.0% inside a string** — and #143
  measured ~21.6% of one fleet's tokens sitting at 0.0% from exactly that one return
  convention, across seven tools. Adding `"embedded"` to a rule's `tiers` folds such a string
  into `{"__terse_json__":1,"f":F,"v":...}`, after which the other tiers apply inside `v`
  normally; the reference double-encoded payload goes **0.0% → 41.4%**.

  **It fires only when it can rebuild the original string byte-for-byte.** `f` names the
  serialization that reproduces it, chosen from a fixed registry (`json.dumps` defaults,
  minified, `indent=2`, `indent=4`, each with/without `ensure_ascii`); when none match, terse
  leaves the string alone. That bar is deliberately stricter than "parses to the same data",
  because `json.dumps(json.loads(s))` is not `s`: duplicate keys collapse to the last one and
  `1.50`/`1.5e0` renormalize to `1.5`. Both decline here rather than decode to bytes the
  server never sent — the guarantee terse sells is byte-faithfulness, not equivalent JSON.
  Also declines an embedded doc carrying a `__terse_*` key, one past the depth cap, and any
  occurrence where the envelope would not pay for itself (a per-occurrence size guard, since
  the whole-payload guard cannot see one small document growing inside a payload that shrank).

  **Opt-in, and `VALID_TIERS` is now distinct from `DEFAULT_TIERS`** so widening the valid set
  cannot silently switch a tier on for policy files already on disk. Each documented form
  costs a primer paragraph the client re-reads every turn (#168), and #170 is the precedent
  for what that costs when a tier rarely fires — so this one is charged only to policies that
  enable it (53 tok), and `policy generate`/`autotune` can enable it per tool on measured
  evidence. Verified: 10 mutations of the codec's guards each caught by the new suite, 0
  losslessness failures and 0 token regressions across the committed corpus plus every
  payload re-wrapped double-encoded in all five serializations.
- **`install-mcp --multiproxy` folds a fleet into ONE proxy (#179).** This is the step
  that banks #168's measured win: six standalone proxies cost +23.1% raw input against an
  unwrapped control, the same six behind one router cost +0.0%, because each standalone
  proxy injects its own primer that the client re-reads every turn — cost scales with
  (servers x turns) while savings scale with (compressible calls). The named entries are
  replaced by a single `terse proxy --config` entry (`--router-name`, default `terse`)
  plus a peers file next to the config. **The stash stays 1:1**, so `uninstall-mcp --all`
  restores every original byte-for-byte with no special case, and uninstalling ONE peer
  detaches it from the peers file while the router keeps serving the rest (the router
  entry is removed once its last peer leaves). An already-wrapped entry is reduced to the
  downstream it wraps before being folded in, so a proxy is never nested inside the
  router even when its stash lives under another scope. `--print` reports the permission
  rewrite: consolidating N servers changes the `mcp__<server>__` segment for every
  wrapped tool, and the tool segment changes too for any name two or more peers export —
  the latter needs live tool names, so it is flagged rather than guessed. It also
  **warns that the switch WIDENS permissions**: N per-server grants collapse onto one
  `mcp__terse` segment, so a whole-server grant now reaches every peer. Re-running is
  **additive** (folding one more server in keeps the fleet instead of evicting it), the
  peers file is namespaced per scope (a local-scope fleet can't overwrite the user-scope
  one), and every runtime flag a single-server wrap takes — `--capture-dir`, `--diff` /
  `--no-diff`, `--no-stats`, `--no-join-blocks`, `--never-lossy` — applies to the router
  too instead of being silently dropped.
- **A multiproxy peer's `env`/`cwd` are honored at LAUNCH (#179).** The peers file
  recorded them and the router ignored them: `DownstreamSpec` had no such fields and
  `StdioTransport` called `Popen` without `env=`/`cwd=`, so a folded server lost a pinned
  `PATH` (codegraph's node@22) and any `env`-borne credential started unauthenticated.
  `env` is MERGED over the router's own environment, never a replacement — a bare mapping
  would launch the child with no `PATH` or `HOME` at all. Setting either on a `url` peer
  is now a config error rather than a silent no-op.
- **A peer's `env` values are coerced to strings on the way into the peers file (#179).**
  An MCP client's own spawn coerces, so `{"PORT": 3000}` is a working config entry and a
  plain wrap preserves it — but the router parses the peers file, and one non-string value
  there took down the WHOLE fleet at launch, on an install that had reported success.
  `load_multi_config` coerces scalars too; containers and null stay hard errors, and an
  empty `cwd` is rejected rather than surfacing as `[Errno 2] ... : ''`.
- **Config-destroying edges around the router entry closed (#179).** The router entry is
  written over rather than stashed, so `--multiproxy` now REFUSES a router name already
  held by an unrelated live server (`terse` is the default name — this needed no unusual
  flag to destroy a third party's entry with nothing to restore from). A `--router-name`
  change now MOVES the router instead of leaving a second one on the same peers file —
  two such entries launch every peer twice, and make the config uncleanable by terse,
  since ambiguous detection returns None. Folding a wrapped-but-unstashed entry stashes
  the unnested DOWNSTREAM, not the wrapper, so `uninstall` no longer reports
  `restored: True` while writing a proxy line back. Re-running plain `install-mcp` on an
  already-folded server — or on the router itself — is refused instead of running that
  downstream twice or nesting a proxy inside a proxy. A router name belonging to a FOLDED
  peer is refused too (a folded peer has no live entry, so a liveness check could not see
  it, and a later `uninstall` wrote that peer's original over the router, stranding the
  rest of the fleet while reporting success), as is folding the router into its own peers
  file (a router that spawns a router, unbounded, at the next client restart). A rename
  carries the router's hand-edited keys — an `env.PATH` pin is the base environment every
  peer inherits. A peer leaves the fleet only when its live entry is BACK, never because
  its stash entry drifted: the peers file is then the last record of how to launch it.
  Runtime-flag inheritance is all-or-nothing, so `--no-stats`/`--capture-dir` (which have
  no inverse flag) can still be cleared, and the result reports the flags actually baked
  in rather than only those named on the command line.
- **`--multiproxy` writes recovery data before the destructive write (#179).** The three
  files (stash, peers, client config) are each written atomically but not atomically
  together, and folding DELETES a peer's live entry rather than rewriting it the way a
  plain wrap does. Config-first left a window — one SIGKILL, OOM, or full disk wide — where
  the live entry was already gone while the stash still described the previous state: the
  original existed nowhere terse looks, so status reported nothing missing and
  `uninstall --all` never mentioned the server. Only the timestamped config backup held it,
  which no recovery path reads. The config is now written last. Also: a duplicated argument
  (`install-mcp kb kb --multiproxy`) no longer folds the same peer twice, and folding a
  terse router that fronts a DIFFERENT peers file is refused rather than nesting a proxy
  inside a proxy (a router has `--config` and no `--`, so `_unnest` passed it through
  verbatim).
- **Every bad multiproxy state is now recoverable and reported (#179).** `uninstall-mcp
  --all` restores a folded peer whose stash entry drifted away, rebuilding it from the
  peers file (reported as a PARTIAL restore — the peers file records launch fields only);
  it removes a stranded router whenever no peers remain, including when the peers file was
  simply deleted, which no longer reads as "no multiproxy involved". A corrupt peers file
  is reported with its path instead of tracebacking out of `mcp-status` (whose contract
  says it never raises) and blocking every other command with a message naming no file.
  Two entries fronting one peers file are NAMED rather than guessed through — the peers
  file is left in place, status says `router-ambiguous`, and the old `--router-name` advice
  (which added a third router and poisoned detection permanently) is gone. The peers
  filename now always carries a hash tail: `local:/home/e/a/b` and `local:/home/e/a-b`
  slugified identically, so two repos shared one fleet and one repo's router launched the
  other's servers. Folding a server whose `env` is malformed fails with a message naming
  it instead of an `AttributeError`, and a peers record that cannot launch anything is no
  longer "restored" as `{"url": null}`.
- **`mcp-status` understands a folded fleet (#179).** A healthy multiproxy install used
  to read as drift in both directions — every peer as `orphaned-stash`, the router as
  `wrapped-unstashed` ("original command unrecoverable"). New states: `router` (with
  `wraps=` listing its fleet) and `folded` (naming the router it sits behind).
  `uninstall-mcp` also no longer mistakes an unrelated server whose own CLI takes a
  `--config` flag for the router.

### Changed
- **BREAKING (multiproxy): tool and prompt names are now qualified only on a genuine
  cross-peer collision (#168).** `terse proxy --config peers.json` used to rename *every*
  tool to `{peer}__{tool}` — `kb.read.search` surfaced to a client as
  `mcp__terse__kb__kb.read.search`. An allowlist written against the unwrapped servers
  stopped matching, which is the sole reason multiproxy was never shippable, even though
  it is the only thing that erases the multi-server primer cost (measured: six standalone
  proxies cost +23.1% raw input against an unwrapped control; the same six behind one
  multiproxy cost +0.0%). A name exported by exactly one peer is now advertised
  **verbatim**, so a
  fleet with distinct tool names is a drop-in. Only a name two or more peers both export
  is qualified, on both sides. `terse.retrieve` is reserved, so a peer exporting it is
  qualified rather than shadowing the router's own. Routing consults the advertised-name
  table first — a tool whose own name contains `__` is no longer misread as a peer prefix
  — with the `{peer}__` split kept ONLY until the first listing installs, after which an
  unadvertised name is a clean -32601 rather than a speculative dispatch. Ledger
  and capture bookkeeping still record the **peer-qualified** name, so per-server corpus
  attribution is unchanged. The routing table is exactly what the most recent `tools/list`
  advertised — a peer missing from a listing is missing from the client's tool list too, so
  calling it is a clean -32601 (carrying routes forward across a missed listing was tried
  and withdrawn; see #178). A listing that completes late by timeout is answered but not
  installed, so it cannot clobber a newer one. Migration: if you had allowlisted
  `{peer}__{tool}` names, switch them to the bare names now advertised by `tools/list`.
- **Releases are now zero-touch.** `release.yml` runs on every push to `main`, derives the
  next version from the Conventional-Commit types since the last tag (`feat` → minor,
  `fix`/`perf` → patch, breaking → minor while 0.x; docs/chore/test/ci release nothing),
  creates the tag, and builds → GitHub Release → PyPI (Trusted Publishing) in one run after
  re-running the suite against the tagged tree. No manual tag, version bump, or changelog
  graduation. Manual overrides stay: push a `vX.Y.Z` tag, or the Actions Run-workflow button
  with a forced bump. Reuses the one `release.yml` publisher identity, so PyPI needs no
  reconfiguration. See TECHNICAL.md → Releasing.
- **`terse report` coverage and `measure` rows now name a tool the way the policy does
  (#158).** `capture.coverage` keyed `by_tool` on the bare `env["tool"]`, so a server-tagged
  corpus reported `structure` while `policy generate` on the same corpus authored
  `runecho.structure` — an operator cross-checking a rule against its coverage count had to
  know the two named one tool. Both `coverage` and `measure`'s per-row labels now use
  `qualified_tool(env)` (`qualify(bare, server)`), the runtime lookup name. A legacy
  envelope with no server qualifies to its bare name, unchanged.
- **`probes.server_of_tool` reads the envelope's `server` instead of guessing it (#158).**
  Since #156 the envelope records `server` straight from the wrap, so it is now returned
  verbatim; the hand-maintained `_RUNECHO_TOOLS` name heuristic is the fallback for legacy
  envelopes only. The point is that the hardcoded list — which silently went stale every
  time runecho gained a tool — is off the primary path.

### Fixed
- **`embedded` re-defaulted `tabularize` inside the fold, leaking an undocumented marker
  (adversarial review of #183).** Folding a string opens a new structural walk, and
  `_embed_json_string` called `compress_structure(parsed, embedded=True)` without forwarding
  the caller's `tabularize`. A policy with `tiers: ["minify", "embedded"]` therefore emitted
  `__terse_table__` inside `v` while `emits_table()` was correctly False — so the primer never
  documented the form and the model received an envelope with nothing explaining it. Lossless
  either way, but it is precisely the failure `reachable_tiers` exists to prevent. Unreachable
  via `policy generate` (which always pairs the two tiers) and untested because every test in
  the suite paired them too; both gaps now closed.
- **`measure` gated a different pipeline than the one it scored.** `embedded`/`tier_total` are
  computed from `compress_with(..., embedded=True)`, but the round-trip gate ran the default
  combination, so a failure that appeared only with the tier enabled would have kept its
  savings banked and fed them to `policy generate`. The gate now validates the embedded
  pipeline too. (The runtime was never at risk — `_lossless_stage` independently self-checks
  the actually-applied combination.)
- **`_default_diff_label` now survives a pathologically deep policy file** (`RecursionError`
  joins the caught set, matching every other `json.loads`-on-file site in the codebase).
  Its truthiness check is deliberately UNCHANGED and now pinned by test: `load_policy` builds
  the policy with `bool(doc.get("diff", False))`, so `"diff": "false"` genuinely diffs at
  runtime, and reporting `policy (on)` is correct. A review flagged the truthiness as a bug;
  tightening it to `is True` would have made the label print "off" while the proxy diffs —
  reintroducing the label-vs-reality divergence #181 was filed to kill.
- **`mcp-status` reported `diff=default` when the default is OFF, and the docs said the
  opposite (#181).** #170 flipped cross-call diffing off, but three signals still pointed the
  other way, and together they produced a repeatable misdiagnosis: a real session saw
  `diffs=0` with `diff_off` on every block, found no `diff` key in its policy, and concluded
  diffing "isn't wired into policy-based tools at all" and "may have been scoped out". It is
  implemented and deliberately off. Fixed all three: `mcp-status` now RESOLVES the value
  rather than naming it (`default (off)`, or `policy (on|off)` when the entry's own policy
  file says), reading the built-in from the `Policy.diff` dataclass field so the label cannot
  drift from the value it describes; `terse stats` explains `diff_off` where the question is
  actually asked, when it is the only diff reason present; and the misleading comment at the
  `diff_off` assignment (which read as a structural exclusion for single-block results, when
  the real cause is policy) is corrected. **README and USAGE both still claimed diffing was
  "on by default"** — the strongest of the misleading signals, now corrected with the measured
  reason it is off (primer 190/402 tokens re-read every turn vs a 0.38% hit rate).
- **`policy.example.json` disabled `dictionary` on `kb.*` from a measurement that predates
  the #116 multi-block join (#144).** The original call (+2.6% total, not worth the tier)
  was made before that join gave `dictionary` a multi-block record array to fold into.
  Re-measured on 1,657 real captured payloads with the join in the path: fleet-wide 7.5% →
  8.0%, and per-tool up to `lodestone_search` 10.9% → 44.0%. `kb.*` now ships with
  `["minify", "tabularize", "dictionary"]`. The underlying `policy autotune` generator was
  never stale — it re-derives the marginal-savings threshold from fresh measurements every
  run — only this hand-authored example had gone stale after a codec change landed under it.
- **The savings ledger charged `structuredContent` at its COMPRESSED size on the raw side,
  understating the real wire saving by ~15 points (#141, part 1).** Since #134 the typed
  field can itself be compressed (`"structured": "compress"/"replace"`), but `_emit_stats`
  passed only the emitted serialization, and `build_record` added that one value to *both*
  `raw_chars` and `out_chars` — so the raw side was charged the compressed size and the
  saving looked smaller than it was. On the reference fixture the ledger reported 47.6%
  where the wire truth is 62.9% (`compress`) and 73.8% vs 81.4% (`replace`). The two sides
  are now tracked separately end to end: `_compress_structured` returns `(raw, out)`, and
  `build_record` charges each side its own size (`structured_chars` raw, new
  `structured_out_chars` emitted). A caller passing one value — an untouched field, and
  every record written before the split — still lands the same size on both sides, so older
  records need no migration. This matters beyond tidiness: the ledger feeds
  `policy generate` and the #136 autotune loop, so a skewed per-tool saving produced a
  skewed policy. *(#141 part 2 — `terse stats` counts BLOCKS but labels them "results" — is
  tracked separately; it's a wider naming change.)*
- **The lossless codec could emit MORE tokens than the server sent, silently (#154).** On a
  record set too small to amortize the `__terse_table__` header — a 2-row `list_*`, a
  filtered query, a shrunk result — `tabularize` produced a form larger than the raw
  payload, and nothing compared the two, so terse shipped the inflated version. The
  reported saving is an average, so a long tail of inflated small payloads hid behind a
  positive headline. `compress_with` now holds the same emit-only-if-smaller contract the
  diff tier already does and the dictionary tier held per-alias: the tiered form is emitted
  only when it tokenizes strictly smaller than plain minify, else the plain lossless form
  ships. Compared on the tokenizer, since a shorter byte string can tokenize longer. Zero
  occurrences on the live corpus (0 of 624) — a latent hole closed before it bit, and the
  `text_alias_ceiling` tripwire now reads zero by construction rather than by luck.
- **A generated rule name carrying a glob metacharacter governed more than its own tool
  (#157).** `policy generate`/`autotune` author `"match": {"tool": name}`, and
  `Policy.select` reads `match.tool` as an fnmatch glob (hand-authored rules use `gh.*`,
  `*.rate_limit`). A tool or server name containing `*`, `?`, or `[` therefore authored a
  rule that silently over-matched: `qualify("*", "runecho")` → `runecho.*`, one tool's
  tiers — and its `capture: false` — landing on every tool of that server. The generated
  name is now `glob.escape`d at the single serialization point, so it matches its own
  literal name and nothing else. Names without metacharacters (the common case) are
  unchanged, so the stored policy stays readable, and `select`/`_shadowing`/`merge_policy`
  all keep reading the one stored string, so the escaped form is both a correct pattern
  and a stable merge key. Neither name is attacker-controlled in any supported deployment,
  so this was a robustness gap, not a vulnerability.
- **A capture/audit/stats sink that HUNG — rather than raised — froze every later tool
  call on the connection.** The sinks were invoked inside `transform_response`'s
  `_local_lock`, and the fail-open `try/except` around them only ever caught a sink that
  raised. A sink that blocks (full disk mid-retry, stalled network mount, slow fsync) held
  the lock indefinitely; `note_request` takes that same lock, so the next `tools/call`
  wedged behind it and never recovered. Sink calls are now queued under the lock and
  invoked after it is released — the reply is already decided by then, so a slow sink
  delays only its own response instead of the whole connection. This makes the documented
  contract ("a sink failure or slowness never affects forwarding") true for *slowness*,
  which it previously was not. Pinned by
  `test_blocking_sink_does_not_stall_a_concurrent_note_request`, which times out against
  the old code.

### Added
- **The corpus is bounded per tool (`MAX_SAMPLES_PER_TOOL`, default 200).** `capture.py`
  was the only disk sink with no retention — `stats.py` rotates at 10 MB and `history.py`
  at 5 MB, but envelopes accumulated forever. Since envelopes hold *raw* tool payloads
  (credentials, PII, private source), unbounded retention widened the blast radius of any
  later disk compromise as much as it risked disk exhaustion. The cap is per tool, not
  global: every consumer (measure, probes, `policy generate`/`autotune`) reasons per tool,
  so a global byte cap would let one chatty tool evict the only samples a quiet one ever
  produced and silently narrow what a generated policy can see. Eviction is oldest-first
  by mtime. `capture_payload(..., max_per_tool=None)` restores unlimited retention for a
  deliberate one-shot corpus build.
- **Coverage instrumentation** (`pytest-cov`, `[tool.coverage.*]`). The suite had no
  coverage number at all; the first measured run is **89% branch coverage** over
  `src/terse`. Reported, not gated — `--cov` is opt-in per run so the default `pytest`
  stays fast, and no threshold is set until the baseline has been looked at.

### Fixed
- **The capture envelope recorded neither which result nor which server a payload came
  from, so autotune had to guess both (#148, #152).** Two defects, one absent pair of
  fields, now written by the proxy (`server`, `result_id`) and read at tune time:
  - *Results were reconstructed from capture timing.* Consecutive envelopes within 50 ms
    were taken to be one result. A burst of independent parallel calls has no gap between
    it, so 200 separate single-block results chained into ONE 200-block group and scored
    **63.4% saved with `dictionary` enabled** where the truth — each scored alone, which is
    what the proxy does — is **25.0% and no dictionary**. Grouping is now exact wherever
    result ids are present. Corpora captured before them keep the heuristic, which also
    gained a total-span cap so an unbroken run can no longer chain without bound, and
    `policy generate`/`autotune` now say how many payloads were grouped that way rather
    than presenting a guessed number as a measured one.
  - *A generated rule could be unreachable.* `select` tries the `{server}.{tool}`
    candidate against **every** rule before the bare name, so with `runecho.*` deployed a
    corpus-derived `structure` rule is dead on arrival — position cannot save it, only the
    qualified name can. Generated rules are now authored under the same name the runtime
    looks them up by. On the live 1663-payload corpus this was ~35 shadow rules in the
    autotune diff, all of which a human had to hand-filter.
  - *The merge's shadow check resolves on the `(bare tool, server)` pair, candidate-major.*
    Naming the rule is only half of it: the check has to find the rule the LOADER would.
    Both a deployed `runecho.*` and a deployed bare `structure` govern a tool captured from
    runecho, and either one's operator-owned keys must be inherited — otherwise autotune
    hands a tool from a `capture: false` rule to a fresh one with capture ON, silently
    reversing the #85 decision and reporting it as a benign "(new rule)". Candidate-major
    also matters when both are deployed: rule-major picks `structure`, the loader picks
    `runecho.*`, and inheriting the wrong rule's keys is worse than inheriting none.
  - *A corpus spanning the upgrade no longer splits one tool in two.* A payload with no
    server is folded into the single server observed for that same bare tool; two servers
    for one bare name is genuinely ambiguous and stays unattributed rather than guessed.
    Without this, the half captured before the upgrade is measured on half the sample —
    and is dead at runtime besides.
  - *`tune --drop-eval` looks its rule up the way the proxy does.* It resolved by bare tool
    name, which on a server-tagged corpus falls through to the defaults, finds no `fields`,
    and scores **nothing** while still reporting that it verified the suggested drops — the
    #149 failure mode with one lookup removed.
  Both fields are optional and omitted when unknown, so an existing corpus stays loadable
  and needs no migration; they are preserved together on an idempotent rewrite, since a
  first sighting's timestamp beside a later sighting's result id would place one block in
  two calls at once. Result ids are scoped by proxy process *and* by handshake generation,
  because a reconnecting client restarts its JSON-RPC ids at 1. `terse capture` gained
  `--server` for the hand-captured case.
- **`policy generate` scored payloads per-BLOCK, so every multi-block tool was
  under-measured (#147).** The proxy compresses a multi-block result as one joined record
  array (#116); the generator scored each captured block alone. For a server that returns
  one record per content block — common — those are wildly different numbers: measured on
  real kb traffic, `changelog` is 23.3% per-block and **48.4% joined**. Payloads are now
  grouped back into results (by capture-time proximity) and each result is scored the way
  the proxy would, falling back to per-block exactly where `apply_joined` would refuse.
- **One non-JSON payload no longer disqualifies a whole tool (#147).** A single
  `Error executing tool …` text block among a tool's records forced `passthrough` for all
  of it. The premise was wrong — `policy.apply` passes a non-JSON payload through untouched
  at runtime, so the tier costs nothing on those results. On a real corpus this alone was
  zeroing the highest-volume tool in the fleet: `kb.read.search` measured **16.7% saved**
  and was written as passthrough because 4 of its 436 payloads were error text. A
  mostly-text tool is still suppressed, now for the right reason — non-JSON contributes 0
  saved while its raw tokens stay in the denominator, so it falls below the threshold on
  its own (`codegraph_explore`, 61/61 non-JSON, scores 0.0%).

### Added
- **`terse policy autotune` — re-tune an EXISTING policy instead of overwriting it (#136).**
  `policy generate` authors from nothing and is *total*: run it on a deployed policy and it
  silently drops every decision the corpus cannot see. It already warned about that for
  `capture: false`; the same was true of `never_lossy_servers`, any `structured` override,
  hand-written active `fields`, any rule for a tool the corpus never saw, and rule ORDER
  (first match wins). `autotune` merges instead, split by what a corpus can possibly know:
  **the corpus decides `tiers`** (including removing one — the motivating case is a stale
  tier decision that predates a codec change), **the operator owns everything else**. It
  prints a per-rule diff, names what it deliberately did not regenerate, and writes
  **nothing** without `--apply`. New rules are inserted before any existing glob that would
  shadow them, since a `kb.read.search` rule appended after `kb.*` is dead on arrival.
  A new rule **inherits the operator-owned keys of whatever rule it displaces** — inserting
  it ahead of a broader rule must not quietly hand that tool `capture: true` or
  `structured: "auto"` — and a rule whose `tiers: []` is suppressing a lossy `$text.*`
  selector keeps `tiers: []`, because turning them on would ACTIVATE that selector and this
  merge is documented as lossless. Warns before applying a tier *downgrade*: the corpus is
  a sample (idempotent by sha, and empty for a `capture: false` tool), so a removal should
  be cross-checked against `terse stats`, which counts every call.

### Added
- **`"structured": "compress"` — compress `structuredContent` too (#128).** New per-rule
  policy knob. MCP 2025-06-18 lets a tool return a typed `structuredContent` field beside
  a text block that mirrors it; terse compressed only the block. Measured against `claude`
  2.1.218 with a read-only proxy, the client forwards the **typed field** to the model and
  discards the block entirely — so on such a tool terse was delivering ~0% however good
  the ledger looked. With the knob on, the same fixture measures **61.2%** of the model's
  real context (2,596 → 1,008 chars), captured end to end rather than inferred.
  Affected servers are not exotic: filesystem (14/14 tools), memory (9/9) and kb (27/27)
  all declare an `outputSchema`.
  Codec only, no diff. See the `structured: "auto"` entry under **Changed** for how the
  default now decides this per connected client.
- **`"structured": "replace"` — drop the redundant text mirror (#128), and the measurement
  saying you probably shouldn't.** Compresses the typed field *and* deletes the text block
  that duplicates it. Measured on the reference fixture: context cost goes 2,596 → 1,008
  chars under `"compress"` and **1,008 → 1,008** under `"replace"` — no change, because
  Claude Code had already discarded the block. What it removes is stdio bytes, not context.
  Shipped as an explicit opt-in that `"auto"` never selects, because it is correct for a
  client that forwards *both* fields (which `"compress"` can leave holding a cross-call
  diff in the block contradicting a full envelope in the typed field); no such client has
  been measured. Five independently-tested guards must all hold before a block is dropped
  — explicit `"replace"`, non-empty `tiers`, not an `isError`, exactly one text block, and
  that block's parsed JSON **equal** to `structuredContent`. Whether the tool declared an
  `outputSchema` is deliberately *not* a guard: the new `noschema` probe shows the client
  reads the typed field from a tool that declares none.

### Fixed
- **The savings ledger no longer reports a saving terse did not deliver (#128).** terse
  compresses a result's text block but leaves `structuredContent` untouched, and the
  ledger counted only the block — so a tool emitting both was credited with the block's
  full reduction while the untouched duplicate rode along at full size. `build_record`
  now counts that duplicate on *both* sides, making `raw_chars`/`out_chars` the whole
  result's cost, and records the split as `structured_chars`/`structured_tokens`. On the
  reference fixture the same call now reports **33.9%** where it previously reported
  58.7%. `decision` is unchanged — it names what terse did to the block, and terse did
  compress it. Records predating the field had no duplicate, so a missing value reads
  as 0. Measured against a live client, the honest figure may be lower still: see
  `scripts/probe/structured_content/`, which found that `claude` 2.1.218 reads
  `structuredContent` and discards the compressed block entirely.
- **A broken capture/stats/audit sink now says so, instead of failing silently.** The
  callbacks handed to the `Interceptor` caught their own exceptions behind a `--debug`
  gate, so the `try/except → _warn_sink` around them never saw one and `_warn_sink`'s
  unconditional first-failure warning was dead code. A `--capture-dir` pointing at a
  regular file, or a `--stats-log` pointing at a directory, produced a completely
  normal-looking run — every tool call answered, exit 0 — with zero payloads captured,
  no ledger written, and nothing on stderr; a later `terse measure --corpus` then
  reported a percentage over whatever subset happened to land. The callbacks now own
  I/O only and let failures propagate to the single caller that has the per-sink
  bookkeeping. The fail-open contract is unchanged: a sink failure still never changes
  what the client receives — it is now merely *audible*: once per sink kind, and under
  `multiproxy` (where the `Interceptor` and its bookkeeping are per-peer) once per
  peer, so a dead shared sink is attributed to each downstream that hit it.
- **A server-initiated request no longer silently disables compression for an in-flight
  call.** A server→client request (`roots/list`, `sampling/createMessage`,
  `elicitation/create`) carries a `method` alongside an id, and JSON-RPC gives each
  direction its own id space — both sides conventionally numbering from 1 — so such an id
  routinely collides with an in-flight `tools/call` id. `transform_response` popped
  `pending[id]` unconditionally (deliberately, so an error-shaped reply still frees its
  entry), which also consumed the entry for a server request; the real tool result then
  arrived untracked and was forwarded **uncompressed and absent from the savings ledger**,
  with nothing logged to say so. The `initialize` path had the same exposure — a colliding
  server request consumed `init_id`, so the real reply never received the terse primer.
  Method-bearing messages are now forwarded untouched, using the same predicate
  `multiproxy` already applied one layer up.

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
- **`structured` now defaults to `"auto"`, which decides per connected client (#128).**
  This **changes default wire behavior for Claude Code users**: `structuredContent` will
  carry a terse envelope where it previously carried the server's own object. That is the
  intended effect — with the previous `"leave"` default terse was a measured no-op on
  filesystem, memory and kb — but it is a real behavior change, stated here rather than
  buried under Added.
  The `"leave"` default shipped alongside the knob rested on "terse cannot detect which
  client it sits behind." That was wrong: the MCP `initialize` request carries
  `params.clientInfo`, a name the client *declares*, and the proxy proxies that request.
  `"auto"` compresses the typed field only for clients measured not to validate it
  (`policy.STRUCTURED_SAFE_CLIENTS`, currently `claude-code`, evidenced by the `badtype`
  and `enveloped` probes) and **fails closed** for an unlisted client, a client that omits
  `clientInfo`, and a library caller that never handshakes. Explicit `"compress"`/`"leave"`
  still win. Measured with a stock policy — no `structured` key anywhere — the fixture's
  context cost drops 2,596 → 1,008 chars (61.2%) against `claude-code`, and is untouched
  against anything else.
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
