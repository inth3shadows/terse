# Changelog

All notable changes to terse are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are cut from git tags (`vX.Y.Z`, via hatch-vcs) — an entry moves from
`[Unreleased]` to a versioned section **in the first pull request after its tag is pushed**,
by running `python3 scripts/release/graduate_changelog.py <tag> CHANGELOG.md` (a manual step;
the workflow cannot push to protected `main`). Nobody has to remember:
`tests/test_changelog_covers_every_release.py::test_unreleased_does_not_describe_work_that_already_shipped`
fails that pull request until the section has moved.

## [Unreleased]

_Nothing yet._

## [0.26.0] - 2026-08-18

### Added

- **`cli:<alias>` answerer reaches real Anthropic models through `claude -p` OAuth**
  (`#249`). Every prior "frontier panel" was measured against the loopback LiteLLM
  gateway, whose `claude-sonnet-5` / `claude-fable-5` / `claude-haiku-4-*` ids are
  aliases onto DeepSeek, not real Claude — a published four-model panel was actually
  two DeepSeek models measured twice under Anthropic names. `cli:<alias>` ids mix
  freely with gateway ids in one `--models` list, always pass `--system-prompt`
  (empty string included) so it *replaces* rather than appends Claude Code's default
  preamble, and strip `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY`
  from the child env so a session already routed through the aliasing gateway can't
  route this path back through it. Every failure returns `None`, never a string,
  after killing the process group so orphaned children don't burn subscription quota
  unseen. `--drop-eval` refuses `cli:` models outright rather than mis-scoring their
  prose output as a declined retrieval.

## [0.25.5] - 2026-08-18

### Fixed

- **`dropeval`'s final-accuracy metric is scored against a measured no-drop control,
  not a fixed 100% ideal that was never run** (`#269`). The metric is JSON
  value-equality against 500+ character prose fields, and a model handed the
  un-dropped payload paraphrases rather than reproducing it verbatim — so the old
  fixed ideal billed normal paraphrase loss to the drop. A live reproduction showed
  every mechanism metric at 100% and final-accuracy at 54%, verdict FAIL. The new
  control arm strips only the drop specs (same tiers, codec, and primer) and measures
  76-88% across three models from three labs, not 100% — flipping two of three models
  from FAIL to PASS. No control run excludes the metric rather than defaulting back to
  the fixed ideal, and per-arm trial counts now exclude errored calls.

## [0.25.4] - 2026-08-15

### Fixed

- **A loss that correlates with the arm under test can no longer publish a PASS** (`#280`).
  When a model's unanswered calls track the arm being measured — a token-budget stop kills
  the LONGEST prompt first, and the diff/terse arm's prompt is strictly longer than its
  control's — `_form_stats` divided each arm by its OWN surviving trial count, scoring the
  arm that lost the hard questions over an easier exam than its control. Measured: a real
  **-20% FAIL** rendered as **-3% PASS**, printing "safe to enable `proxy --diff`".

  This is the fourth pass at the same class of bug, and the first to treat it as
  structural. `_form_stats(rows, form)` computes ONE arm, so every gap site called it twice
  and subtracted — and nothing in that shape can enforce that the two arms answered the
  same questions, because pairing is a property of the pair. Each previous pass wired
  pairing into the sites it was looking at; the next site stayed writable by accident. The
  third attempt's commit claimed "every diff-vs-control gap site", and reverting its
  pairing at two of the three left the entire suite green.

  So the shape is gone rather than the sites patched. `arm_gap` / `best_arm_gap` are now
  the only way to turn a form and a control into comparable numbers: they gate
  (`_unmeasured`, then a new `unpaired` refusal), pair the rows, and compute every arm over
  that paired subset — numbers and gate arrive together, so a caller cannot take one
  without the other. All seven sites route through it: the diff markdown table, the HTML
  banner, both terminal forest plots, the soak by-depth table and its deepest-depth
  verdict, and the fluency verdict. `build_fluency_report` also loses its duplicate copy of
  `fluency_gap_rows`' best-of math.

  The new per-depth gate made one branch newly reachable, and it was wrong: with the
  deepest slice withheld while the pooled model still published, `deepest is None` was read
  as "passed" and the soak printed **"No depth-correlated comprehension drift"** about the
  one depth nobody scored. It now prints `NO VERDICT at the deepest tested depth`, and
  withheld depths are named rather than left as an unexplained `n/a`.

  Two things pairing must NOT treat as a loss, both of which it initially did. A row with
  no `attempts` counter comes from `score_pack`, where uneven per-form trial counts are a
  documented collection mode (#91) rather than a failure — `score_pack` never calls a
  backend, so it has no transport to lose calls to, and voiding those rows deleted a
  working feature. And withheld models are now split into **Not measured** (transport) and
  **Not compared** (unpaired), which are different events and no longer share a sentence.

  **No refusal ships with this.** `paired_rows` removes the bias outright — after pairing
  both arms sit the identical exam — so nothing further is needed for correctness. An
  additional bar was attempted, meant to decline when the surviving exam looked too
  *selected* to generalise from, and five review rounds each found it publishing a false
  PASS or refusing healthy runs: a volume share that voided 41% of models at a 2% per-call
  failure rate; an asymmetry statistic that cancelled a total one-sided loss when the
  control hiccupped once on the same questions, that unrelated control failures could buy
  tolerance from, that went inert above `--trials 2` when scaled by trials, and that with
  two form arms let a -23% regression publish as a PASS. It also made one soak case WORSE
  than before the change. It is not required for the fix and is dropped rather than shipped
  half-right; the honest position is that "how selected is the surviving exam" needs a
  measurement nobody here has yet got right, not a fifth attempt at a threshold.

  Every renderer now reads the exclusion vocabulary from one place (`REASON_LABEL` /
  `REASON_HEADING` / `exclusion_note`), and `diff_gap_rows` / `fluency_gap_rows` return the
  reason rather than a bare name list. Five of six renderers had been restating it wrongly:
  the terminal fluency plot called an unpaired model "raw control failed" while its raw
  control read 100%, the HTML page told the reader to check stderr for a `returned no
  content` line about a backend that answered every call, and the soak's NO-VERDICT line
  said "fix the backend(s) and re-run" about a run that would fail identically on re-run.
  One fact restated independently at six sites, drifting at five — the same shape as the
  bug this change exists to fix, in prose instead of numbers. The soak also now names
  models it drops from its own verdict, which it previously discarded silently.

  `_per_transform_table` pools paired rows too. It computes no gap, so it stays on the
  `_form_stats` allowlist, but pooling unpaired rows reintroduced the same bias one section
  below the fix: a partially-lost row contributes only its surviving trials, and the
  surviving trials are the easy ones. Measured on the added fixture, that read 36% where
  the paired truth is 33% — in the table whose own comment says a reader uses it to decide
  which transforms to keep in the policy.

  `_safe_ask`'s handling of a non-str reply is deliberately unchanged: on `main` the
  `AttributeError` from calling `.strip()` inside the `try` is what converts "answerer
  returned garbage" into "unanswered", and that is correct.

## [0.25.3] - 2026-08-14

### Fixed

- **A model that returns HTTP 200 with no content is scored as unanswered, not wrong**
  (`#268`). `#264` gated every fluency/diff report on TRANSPORT failures — the exceptions
  `_safe_ask` turns into `None`. A backend returning `content: null` (or an empty/blank
  string) raises nothing: it flowed through `openai_answerer`'s `content or ""` as a real
  reply of `""`, was scored wrong, and was invisible to `_unmeasured`. Observed live:
  `gemini-3.6-flash` returning null when reasoning consumed the token budget.

  No-content is now normalised to `None` at `_safe_ask`, the choke point every live harness
  funnels through, and in `openai_answerer` and `dropeval.openai_tool_answerer` directly (the
  same `content or ""` bug, one layer down, on the drop-eval side). `_answerable` now
  excludes empty-expected-answer questions on every generation path, not just the
  flat-record one — `score("lookup", "", "")` is `True`, so a blank reply could otherwise
  score as *correct*. All four verdict renderers (markdown, HTML, terminal, the
  excluded-model line) now say a call "went unanswered" instead of "never reached the
  backend," which prescribed the wrong remedy for a token-budget stop.

  Scoped to detection only. Whether a run should also protect against losses that are
  *correlated* with one arm (e.g. the diff arm's longer prompt failing more often than its
  control's) is an open design question, tracked separately in `#280` rather than folded in
  here — a reactive per-site patch for that already went through two review passes without
  closing the class.

## [0.25.2] - 2026-08-13

### Fixed

- **`install-mcp` no longer bakes a launcher that dies with the venv that installed it**
  (`#275`). `terse_invocation` returned `[sys.executable, "-m", "terse"]` — whatever
  interpreter happened to run the install. Invoked with `uv run` from a throwaway git
  worktree, that is `<worktree>/.venv/bin/python3`, so every wrapped MCP server was wired
  to a directory deleted at the end of that session. The config keeps working until then,
  which is what makes it dangerous: the servers fail **silently** days later, showing up
  with no tools, far from the `install-mcp` run that caused it. An isolated-tool install
  (`uv tool`, `pipx`) has the milder version of the same exposure — a versioned venv an
  upgrade can move.

  A second tier now sits between the `$TERSE_MCP_CMD` override and that fallback: the
  first `terse` console script on `PATH` that does **not** live under `sys.prefix`. One
  rule covers every layout — in a worktree it skips the ephemeral venv (which `uv run`
  puts *first* on `PATH`) and finds `~/.local/bin/terse`; on an installed uv tool it takes
  that script directly; in a checkout where terse was never installed, nothing qualifies
  and the interpreter form ships unchanged. Preferred over hardcoding `~/.local/bin`
  because it covers pipx, Homebrew and system-pip the same way.

  Directories are **resolved** before the `sys.prefix` test and the selected script is
  **not**, which is the whole subtlety: `~/.local/bin/terse` is a symlink *into* the uv
  tool venv — on an installed terse, that IS `sys.prefix` — so resolving the script would
  reject the one stable launcher being looked for, while its parent `~/.local/bin` is an
  ordinary directory whose resolution costs nothing and closes the spellings a literal
  comparison misses (`<wt>/../<wt>/.venv/bin`, or a venv reached through a symlinked
  parent), each of which would otherwise select the ephemeral script and reproduce the bug
  with the fix in place. Relative `PATH` entries (`.`, `bin`, the empty string) are
  skipped rather than absolutized, since they mean the *installer's* cwd — the same class
  of path-that-outlives-nothing. And the venv's directories are dropped from the search
  path rather than the first hit being checked and rejected: `shutil.which` stops at its
  first match, which under `uv run` is exactly the ephemeral script.

  Tier 2 selects by NAME off `PATH`, a weaker claim than tier 3's — it can find a shim, or
  an unrelated tool of the same name — so the candidate is now **proven with one
  `--version` call** and rejected in favour of tier 3 if it does not answer as terse.
  `install-mcp` also prints the launcher it chose, because the `after:` line truncates at
  100 chars and a long policy path pushes the launcher off the end of it.

  It **warns when that launcher reports a different version than the terse writing the
  config**: a checkout emits argv in its own grammar, and every flag this repo has added to
  `proxy` (`--server-name`, `--config`, `--no-join-blocks`) makes an older installed terse
  exit 2 at spawn — which the client shows only as a server with no tools. Observed while
  building this, on an editable install that had drifted to `0.23.3` behind a `0.25.2.dev`
  checkout (since upgraded, so that particular reading no longer reproduces).

  Only hatch-vcs's dirty-tree date stamp is normalised away before comparing, so an
  editable install does not warn against a dirty checkout of the same commit. Comparing
  PEP 440 *public* versions instead was the first attempt and is too coarse: hatch-vcs
  stamps every commit at the same distance from a tag as `0.25.2.dev1`, so a branch that
  adds a `proxy` flag and the tool built from `main` agree on the public version and differ
  only in `+g<hash>` — exactly the case the warning exists for. The hash stays in.

  Probing costs **one** subprocess per install, not two (the result is memoized; both the
  selection and the skew check need it), with a 10s timeout — a launcher that hangs used to
  stall a `--print` that changes nothing for a full minute.

  The probe cannot abort an install. `subprocess.run(text=True)` decodes strictly and the
  resulting `UnicodeDecodeError` is a `ValueError` — neither of the exceptions the probe
  catches — so it escaped to the CLI's `except (FileNotFoundError, ValueError)` and exited
  2. A single stray localized binary named `terse` anywhere earlier on `PATH` blocked every
  `install-mcp` run, `--print` included, with a message that named no launcher at all.
  Output is now read as bytes and decoded with `errors="replace"`; an unproven launcher
  falls back to tier 3, which is the whole contract.

  Consequence worth knowing: running `install-mcp` from a terse checkout now wires MCP to
  the **installed** terse, not the checkout. Config written by `install-mcp` has to outlive
  the process that wrote it by weeks, so an ephemeral path is never the right artifact; a
  development build is wired in deliberately with `$TERSE_MCP_CMD`, which still wins over
  everything. `USAGE.md` documents the three tiers in place of the old "after upgrading,
  re-check `mcp-status`" workaround this removes the need for.

### Fixed (found reviewing the above, same area)

- **A Windows console-script entry was invisible to every terse-managed check.**
  `_looks_like_terse_launcher` matched `Path(cmd).name == "terse"`, but `shutil.which`
  returns `...\Scripts\terse.exe` on Windows — never the extension-less name — so with
  tier 2 above, every entry `install-mcp` writes there would have failed the predicate the
  whole managed-server layer keys off. `_detect_routers` would find no router, so a second
  `--multiproxy` install would refuse the router name it wrote itself, and
  `uninstall-mcp --all` would unlink the peers file while leaving the router entry pointing
  at it — verbatim the state that code exists to prevent. `parse_proxy_opts` would return
  None, so `policy autotune` would report no wrapped servers on a fully wrapped config.
  Before tier 2 the bare `terse` token in `-m terse` carried the match, which is why this
  surfaced only alongside it; CI is ubuntu-only, so a test holds it now.

  Matching is by a separator- and `.exe`-tolerant basename rather than `Path`, because a
  config is JSON that may have been written on the other OS — a `PosixPath` reads
  `C:\Users\me\Scripts\terse.exe` as one long filename, so the obvious spelling would
  silently stop recognising Windows entries anywhere else.

### Changed

- **Test isolation: `$PATH` and `$TERSE_MCP_CMD` are now pinned suite-wide**
  (`tests/conftest.py`). Only ~15 of the ~45 `do_install` call sites monkeypatch
  `terse_invocation`, so after the change above the rest asserted against whichever argv
  *shape* the developer's machine produced — one element on a box with a global
  `~/.local/bin/terse`, three in CI. That is precisely how the Windows defect above stayed
  invisible: the shape that breaks detection is the shape CI never produces. Only the
  `PATH` entries that actually provide a `terse` are dropped, so `git` stays reachable for
  the tests that shell out to it.

  `test_the_entrypoint_uses_this_interpreter_not_whatever_is_on_path` asserted
  `argv[0] == sys.executable`, which was stricter than the rationale its own docstring
  gave — an absolute console-script path satisfies "never resolved off `PATH` at launch
  time" equally well. It now asserts absoluteness across **both** tiers and *executes*
  both, since tier 2 is what ships and no test ran one; the console script under test is a
  shim onto this checkout, not the installed terse, for the same reason the `PATH` guard
  exists.

## [0.25.1] - 2026-08-12

### Fixed

- **A transport failure in the fluency eval no longer scores as a wrong answer** (`#263`).
  `_safe_ask` returned `""` on any exception and `score` counted that as incorrect, so an
  unreachable model reported ~0% accuracy — indistinguishable in the report from a model
  that genuinely could not read terse's compressed form. Because the verdict gates on the
  **worst** model, a single rate-limited backend did not dilute a panel, it decided it: a
  live `gemini-3.6-flash` rate limit would have returned FAIL for the whole no-primer
  question (`#249`) on a run that measured nothing, after 66 minutes and zero output.

  `_safe_ask` now returns `None` — distinct from every real reply, including an empty one —
  and `_ask_n` returns `(correct, transport_failures)`, counting a failure separately and
  never scoring it. A call that never happened is not a wrong answer.

  The report already excluded a model whose raw control was exactly 0%. That guard catches
  a **total** outage only, and only after scoring it as if the model had answered; it does
  not catch a **partial** rate limit, which leaves `raw` non-zero while depressing every arm
  — the case that reaches a plausible-looking verdict. A model with any failed call now
  publishes **no accuracy at all** (`n/a`, never a footnoted 0%) and is excluded from the
  gate, and a run in which every model failed states `NO VERDICT — nothing was measured`
  rather than falling silent, because silence is how a run that measured nothing gets read
  as a run that found nothing wrong.

  Secondary hazard closed with it: `""` could *match* a question whose expected answer was
  empty, scoring a total failure as **correct**. `questions.py` excludes such questions
  defensively for exactly that reason; those exclusions are now belt-and-braces rather than
  load-bearing.

  A failed call is removed from **its arm's denominator** (`<form>_trials`, which
  `_form_stats` already preferred) rather than voiding the model, so a handful of transient
  429s no longer depress an accuracy at all; a model is withheld only when an arm has zero
  completed trials or more than `UNMEASURED_FAIL_SHARE` (20%) of its calls were lost. The
  earlier any-failure rule would have discarded an otherwise-complete multi-hour run — the
  same outcome as the bug, by a different route. Models that lost some calls but stayed
  under the bar are listed as *partially degraded* with `fails/attempts`.

  The exclusion now also covers `fluency_gap_rows`, which feeds the **terminal forest
  plot**: `cli` prints it directly below the markdown, so a dead backend previously showed
  `n/a` in the table and "not measured" in the verdict while the chart beneath plotted its
  gap as a red FAIL bar. The per-transform table likewise stops pooling withheld models'
  rows — that table is what a reader uses to decide "restrict the policy to the transforms
  that held".

  Verified against a live outage rather than only in tests: with `gemini-3.5-flash`
  returning 503, the old code named it the worst-case model and printed **PASS** ("terse's
  compressed form preserves comprehension within tolerance") off a backend that was down —
  a false *pass*. The same run now reports `104/104 calls lost` and `NO VERDICT`.

  The same gate now covers the **`proxy --diff` ship gate** (`build_diff_report` /
  `build_text_diff_report`), which had no control of any kind — not even the `raw == 0`
  one the payload report already had. A backend that was entirely down scored 0% on both
  arms, so the gap was exactly 0 and the verdict read *"safe to enable `proxy --diff`"*.
  A false **pass** on a ship gate is worse than the false fail this issue was filed about:
  a false fail blocks someone and gets re-run, while a false pass agrees with whoever ran
  it and is never checked again. The diff harnesses now emit the `attempts` counter the
  gate divides by, `_unmeasured` discovers arm names from the rows instead of hardcoding
  the payload harness's four, and a run with nothing left to score prints `NO VERDICT`
  rather than an empty verdict section.

  The gate reaches every renderer of those rows, not just the first one fixed. Gating the
  diff markdown alone re-created the split it was meant to close: `diff_gap_rows` — whose
  docstring promises "a chart's gap can never read differently than `build_diff_report`'s"
  — kept drawing a FAIL bar for a model the markdown had just declined to score, in the
  forest plot `cli` prints directly beneath it, for all three diff paths. It now returns
  `(gap_rows, excluded)` like `fluency_gap_rows` and names what it dropped.
  `build_diff_soak_report` is gated too: a down backend scored 0% on both arms at every
  depth, which is a gap of exactly 0 reading **PASS**, beneath a by-depth table showing a
  flat, reassuring no-drift line drawn entirely from calls that never happened. And
  `build_html_diff_report`, which builds its own gap rows rather than calling
  `diff_gap_rows`, rendered a green `✓ PASS` banner off the same dead backend — the
  artifact most likely to be screenshotted and quoted, and so the last place a false pass
  should survive. It now renders `NO VERDICT` and names the models it dropped.

  Those were one defect wearing five coats — the gate was added a renderer at a time, and
  each renderer that still lacked it published the verdict the others had just withdrawn.
  `diff_gap_rows` asserted its agreement with the markdown *in a docstring*, which is how
  the split opened without a test going red. The rule is now pinned as an invariance —
  **an unmeasured model must not change any renderer's verdict** — checked across the
  fluency markdown, the diff markdown, the soak markdown, both terminal forest plots and
  the HTML banner from one fixture. It catches all seven gate sites, including the
  soak's deepest-depth drift signal, which was found by writing the invariance test and
  not by review.

## [0.25.0] - 2026-08-11

### Added

- **A `drop-to-retrieve` rule's COST is now recorded and reported, not just its saving**
  (`#251`). `answer_retrieve` has served the synthetic `terse.retrieve` tool since `#10`,
  but `stats.py` contained no reference to it — so the ledger measured only the tokens a
  dropped field never spent, and never the extra tool call the model spent fetching that
  field back. A rule dropping a field the model *always* needs was indistinguishable in the
  data from one dropping a field it never needs, which made every lossy rule look better
  than it was. On a demo ledger the gap is stark: a tool reading 96.4% saved in the per-tool
  table while its drop rule cost 5,628 tokens in round-trips.

  Attribution rides back on `policy.Applied.drop_origins` (`handle -> (tool, rule path)`)
  rather than through a widened `drop_sink`: the staged sink is `dict.__setitem__`, which
  takes exactly two arguments. Origins are staged with the values and published on the
  **same commit**, so a failed recoverability gate leaves no orphan attribution — and the
  map is shared across multiproxy peers exactly where the drop store is, because any peer
  may answer a retrieve for a handle another peer dropped.

  The ledger `server` label is captured at **drop** time and travels in the shared origins
  map, because under multiproxy `_route_call` answers every retrieve through `peers[0]` —
  so the answering Interceptor is almost never the dropping one, and billing the answerer
  filed a `kb` rule's cost under `gh`, where it does not join with that tool's own result
  rows. A miss is unattributable by construction (the origin is discarded with the value)
  and the report says so rather than implying a rule can be named. The cost table honours
  the same tokens-or-chars fallback as the savings table above it.

  **`terse stats --json` gains a top-level `retrieves` key** — one row per
  `(server, tool, rule path)` with `calls` / `hits` / `misses` / `bytes` / `tokens` /
  `untokenized`. Always present (an empty list when nothing was recorded) so a consumer can
  read it unconditionally. A retrieve row is deliberately **never** folded into `total` or
  `tools`: it carries no `raw_chars`/`out_chars`, and `aggregate` skips it on an explicit
  `event` marker as well, so it cannot be miscounted as a compressed block and silently
  move the published savings percentage. Both guards are pinned by tests.

- **The `multiproxy.py` / codec boundary is now enforced by a test rather than described in
  prose** (`#237`). That issue recorded multiproxy as a second product — 1,525 lines of MCP
  router, none of it compression — and ruled that new optimization logic must not land
  there, but left open whether to enforce it or trust the convention, warning that "a test
  that is wrong is worse than a convention that is read".

  Both candidates it named were weighed against a real change (`#246`, which grew the file
  by 109 lines). The **line-count ceiling was rejected**: that growth was routing logic,
  exactly what belongs in a router, so a ceiling would have blocked correct work while
  catching no boundary violation — it measures the proxy, not the property. The **import
  assertion was kept**: the router may depend on `policy`/`stats`/`lossy` to *call* them,
  but reaching for the codec is the signal `#237` says should move the decision out rather
  than widen the file.

  `tests/test_module_boundaries.py` parses imports by AST, not grep — `multiproxy.py` uses
  the word "transform" throughout its prose, so a text search would fail on a comment and
  get deleted as noisy, which is precisely how a wrong test earns less than the convention
  it replaced. A second test guards the guard against passing vacuously, sweeping five
  spellings a violation could arrive in; it caught the detector reducing
  `from terse.transforms import compress` to `terse` and missing it.

## [0.24.1] - 2026-08-11

### Fixed

- **A collision seen once now keeps its tools/list name qualified when the rival peer goes
  silent** (`#178`). multiproxy computed collision naming from one listing alone, so if
  `gh` and `kb` both exported `search` and `kb` later missed a broadcast, gh's copy flipped
  from `gh__search` to bare `search` — and back again when `kb` returned. A client caching
  either spelling got a `-32601` for the other.

  **One ordering, not both.** The ratchet can only fire once a contest has been witnessed,
  so `kb` answering and *then* going silent is closed, while `kb` being silent on the
  *first* listing and then returning still flips the name — and that is the direction
  `#226`'s `-32601` diagnosis cannot explain, since no peer is silent on the listing where
  the rival returns. Closing it needs knowledge of what a peer exports before it has ever
  answered, for which the router has no source short of the two options this issue already
  rejected (carrying routes forward, or unconditional prefixing). The remaining gap is
  pinned by `test_a_rival_silent_on_the_first_listing_still_flips_the_name` rather than
  left as a claim in prose.

  Closed with a **contested-name ratchet** — a per-surface set of bare names seen exported
  by two or more distinct peers in one listing, fed into the next listing's existing
  `reserved` mechanism. Deliberately NOT the route retention withdrawn in `#178` after 11
  defects across 3 review rounds: it holds no peer identity, no liveness and no routes, so
  it cannot resurrect a tool, keep an erroring peer's routes alive, or erase a live peer.
  A peer that answers `error`/`malformed`/`empty`/not-at-all contributes nothing and can
  never contest a name; one peer listing a name twice does not ratchet it (distinct peers,
  not occurrences); and a reserved-only qualification never enters the set, which would
  otherwise self-feed back to the unconditional prefixing `#168` removed. The ratchet is
  intentionally not behind the seq guard — a superseded listing's *table* is discarded, but
  "these two peers both export `search`" is not falsified by arriving late.

  Two limits stated rather than papered over: the ratchet is **never invalidated**, so a
  peer that collides in exactly one listing keeps its rival qualified until the process
  restarts (un-ratcheting would need to distinguish "the contest ended" from "the rival was
  silent", which one listing cannot); and reading the ratchet is **not atomic with
  installing a table**, so concurrent merges can still reproduce the flip for a single
  self-healing listing. Both are narrowings, not proofs.

  Six new tests in `tests/test_multiproxy.py`, each mutation-verified to fail against the
  specific defect it pins.

## [0.24.0] - 2026-08-11

### Added

- **`secret-broker.secret.exa_search` opted into compression tiers** (`#143`). The
  server's default-deny stance held it at `tiers:[]` pending evidence a carve-out was
  worth it — the original synthetic estimate (~59%, from `#52`'s "body arrives as a
  string, not real JSON" theory) turned out wrong once `secret-broker` PR #53 shipped
  the parsed shape: two live measurements against the real payload came back 1.0%
  (`n=3` results) and 2.6% (`n=10`), dragged down by `highlights` free-text excerpts
  (69% of tokens, 0.0% compressible — `tabularize`/`dictionary` win on repeated
  structure, not unique prose). The result metadata alone (id/title/url/author/date)
  hits 8.5%. Small but real, safe (public web-search results, no credential in the
  payload — verified by `#143`'s leak check), and free: `capture:false` keeps it off
  disk exactly as before. Every other `secret-broker.*` tool is unaffected by the
  default-deny rule.

## [0.23.4] - 2026-08-11

### Fixed

- **`policy.example.json` carried no `secret-broker` rules at all**, even though `#243`
  added the `require_server_name` field specifically to protect a rule shaped like this
  server's. A fresh install that copied the example verbatim (or wrapped `secret-broker`
  before ever authoring a custom policy) got zero protection —
  `secret.reveal_credential` would compress and persist under the unmatched-tool
  default. Ported the two rules from the operator's live policy: `list_credentials`
  (metadata-only, compresses 45.5%) gets full tiers, everything else is
  `tiers:[]`/`capture:false`/`require_server_name:true`. Pinned by
  `test_example_policy_guards_secret_broker_crown_jewels`.

## [0.23.3] - 2026-08-11

### Fixed

- **`Policy.select` failed OPEN, not closed, when `--server-name` was omitted.** A
  server-scoped deny-all rule (e.g. `secret-broker.*` — the crown-jewel default-deny
  rule guarding `secret.reveal_credential`) is only reachable via the server-qualified
  match candidate `_match_candidates` synthesizes, and that synthesis only happens when
  `server` is truthy. Drop `--server-name` — a hand-edited config, a future refactor,
  anything outside the normal `install-mcp` path (which always bakes it) — and such a
  rule silently goes unreachable: the credential-returning tool falls through to
  `Policy.select`'s unmatched-tool default, whose `capture` is the dataclass default
  `True`. Fixed with a new opt-in, per-rule `"require_server_name": true` field —
  declarative, matching the existing `capture` pattern (#85), not a heuristic: a fully
  general static detector can't tell a server-scoped glob (`secret-broker.*`, whose
  tools never self-prefix) from a same-server namespaced one (`kb.*`, whose tools do)
  from the glob string alone. `run_proxy` now refuses to start (clear stderr, exit 2)
  rather than silently degrading, whenever a loaded policy marks a rule
  `require_server_name` and no `--server-name` was given. Applied to the operator's own
  live policy (`~/.config/terse/policy.json`, outside this repo) — `policy.example.json`
  carries no `secret-broker` rule at all, so this change alone does not protect a
  fresh install that copies the example; that's the scope of a planned follow-up.

## [0.23.2] - 2026-08-11

### Fixed

- **`docs/POSITIONING.md` and `BENCHMARKS.md` §5 published stale and falsified personal-fleet
  numbers.** The personal-fleet savings figure was two conflicting numbers across the two docs
  (6.8% vs 9.2%, both dated snapshots from early August); republished as one live figure
  (15.1% blended, `terse stats`, 2026-08-11). Retracted `BENCHMARKS.md`'s "prose-heavy records
  ... hard ceiling ... no tier combination changes that" and `POSITIONING.md`'s "nothing
  structural left to remove" — both false as of the day they were re-measured:
  `kb.read.list_principles` reads 15.1% blended in production, driven by large multi-block
  calls routed through multiproxy dominating the token-weighted average (a composition effect,
  not the codec learning to compress prose — a fresh Tier-0-only measurement on the same tool
  still lands at 3.5%, close to the original figure). New
  `tests/test_published_ledger_ceiling_claims.py` pins the retracted phrasing from ever coming
  back.
- **`mcp-status` couldn't see a real ledger-identity split.** A wrapped entry launched by hand
  (not via `install-mcp`, which always bakes `--server-name` per #152) writes ledger records
  under a GUESSED identity — the downstream command's basename — which silently diverges from
  another install of the same logical server launched under a different command name. Found
  live: `runecho` and `runecho-mcp` were two ledger identities for one server. `mcp-status` now
  flags a wrapped entry with no explicit `--server-name`. The identity rule itself
  (`server_name or server_label(cmd)`) is now `stats.resolve_ledger_identity`, called from both
  `proxy.py`'s live write path and `mcp-status`'s detector — a review round caught the two
  independently re-deriving the same fallback, which would have let them silently diverge.

## [0.23.1] - 2026-08-11

### Fixed

- **PyPI publishing was silently broken since v0.23.0.** Hatchling now defaults to
  `Metadata-Version: 2.5`; the pinned `gh-action-pypi-publish@v1.14.0` bundles a twine
  that predates upstream support for it, so every publish failed with `InvalidDistribution:
  Invalid distribution metadata: '2.5' is not a valid metadata version`. v0.23.0 tagged and
  GitHub-released correctly but never reached PyPI. Bumped the pin to v1.14.2, whose release
  notes name 2.5 upload support as the headline fix (twine bumped to v7 internally).

## [0.23.0] - 2026-08-11

### Added

- **`terse stats --recommend` — one wrap/don't-wrap verdict per installed entry.** #175 stated
  the rule ("wrap a server when its session-lifetime savings clear its own primer") and #197
  put both halves of the arithmetic in `terse stats` as two numeric columns, leaving the
  operator to do the comparison. The new mode rolls them into one word — `KEEP` / `TUNE` /
  `UNWRAP` / `INSUFFICIENT` — beside the coverage ratio it was derived from, replacing the
  ledger tables rather than appending to them (the default report's output is byte-identical).
  No new arithmetic: every input is a field `_break_even` and `_cadence` already published, and
  the 1.0 threshold is the one the report already draws when it prints `NET NEGATIVE`.
- **Four new fields on every `primer_liability.servers[]` row in `--json`**, present regardless
  of the `--recommend` flag so the contract has one shape rather than two: `verdict` and
  `verdict_reason` (never null — there is always an answer, even if it is `INSUFFICIENT`),
  `break_even_coverage` (`tokenized_blocks / blocks_to_break_even`, null wherever the ratio is
  undefined — never a fabricated `0` or infinity), and `contributors`.
- **A new nested `primer_liability.servers[].contributors[]` shape** — `label`, `blocks`,
  `tokenized_blocks`, `saved_tokens`, `saved_per_block` — ranking each ledger label an entry
  pools, by tokens saved. It deliberately carries **no** `primer_tokens`,
  `blocks_to_break_even`, `break_even_verdict` or `verdict`: a router pays ONE union primer for
  its whole fleet, so there is no honest per-peer break-even to publish, and the verdict stays
  per installed entry. `tests/test_stats_json_contract.py` pins all of it as exact key sets.

### Changed

- **`release.yml` no longer tries to graduate the changelog itself.** The step pushed a
  `chore(release):` commit to protected `main` after every release and failed 44 out of 44
  times, silently — the push was suffixed `|| echo "::warning::..."`, so a red step never
  surfaced. It is unfixable with the built-in token: this is a user-owned repo, where Ruleset
  bypass actors (org-only) do not exist, and a `GITHUB_TOKEN`-opened PR triggers no workflow
  runs, so under `required_status_checks.strict=true` it would be permanently unmergeable.
  Graduation is now an explicit manual step in the first PR after a release, enforced by the
  existing CI-run-gated detection test rather than by a stored PAT.
- **`graduate_changelog.py` dates a section from its tag instead of from the day it ran.**
  Now that the script runs days after the tag rather than seconds, `date.today()` was
  guaranteed to disagree with `git log -1 --format=%cs <tag>` — the exact comparison
  `test_every_section_carries_the_release_date_git_records` makes. It now prefers an explicit
  `YYYY-MM-DD` third argument, then the tag's own commit date, and falls back to today only
  with a stderr warning naming `git fetch --tags` as the fix.

### Documentation

_Prose and one new test only — no runtime behaviour changed, and no default moved. Every
correction below brings a document into line with code that already shipped._

- **Seven prose sites still announced that cross-call diffing is on by default.** #170
  flipped `Policy.diff` to `False` (2026-07-28) and updated the code, `USAGE.md` and
  `BENCHMARKS.md`, but left "Default-on since its validation program completed" standing in
  `README.md` (×3), `TECHNICAL.md` (×2) and a `USAGE.md` code block that told the reader
  "nothing to enable — a plain proxy diffs" nine lines under a paragraph correctly saying
  **OFF by default**. Two of the README sites sat four lines apart, one saying OPT-IN and
  the next saying default-on. All now say what `src/terse/policy.py` says.
- **`tests/test_published_diff_default.py` pins it.** The contradiction went unnoticed for
  two weeks because nothing checked, so the durable artifact is a test rather than a note:
  it sweeps `README`/`TECHNICAL`/`USAGE`/`BENCHMARKS`/`POSITIONING` for default-ON
  assertions about the diff tier and fails while a default policy's `diff` is `False`. It
  found the `USAGE.md` code block that the hand pass had missed.
- **`README.md` pinned headroom at v0.32.0 in both competitor sections**; `BENCHMARKS.md`
  re-measured v0.34.0 on 2026-08-05. Both now say v0.34.0.
- **The README headline claimed "byte-faithful by default" without qualification.** The
  lossless gate's own words are "byte-faithful **by value**" (`transforms.roundtrip_ok`),
  which is `==` plus a NaN contract rather than equality of the re-serialized bytes. The
  headline now reads "lossless by value". `values_equal` is unchanged — its docstring
  forbids tightening it, and doing so would risk #187's NaN fix.
- **`TECHNICAL.md` gained *The guarantee ladder*** — mechanical reconstruction →
  model comprehension → task success → net economics, each rung naming the shipped artifact
  that proves it (`roundtrip_ok`, `fluency`, `fluency --diff-soak`, `terse stats`
  break-even). All four already existed and were documented separately; the ladder is an
  index, not a new claim, and it makes #170 legible as a tier that cleared rungs 1–3 and
  was still turned off for failing rung 4.

## [0.22.3] - 2026-08-06

### Fixed

- **A bare `terse policy autotune` only ever resolved wiring from user-scope
  `~/.claude.json`, so a project- or local-scope-only install reported "no terse-wrapped
  servers found" and fell through to requiring explicit `--policy`/`--corpus`.** Flagged
  as a follow-up in #167, which closed the loop for the common (user-scope) case only.
  `discover_wrapped_opts_all_scopes` now scans the same three scopes `mcp-status` already
  does — user, project, local — mirroring `scan_scopes`'s target resolution so the two
  commands agree on where a wrapped server can live.

  Review caught two follow-on defects in the first cut: a corrupt/unreadable config file
  in ANY one scope aborted resolution for all three (a broken project-scope `.mcp.json`
  could mask perfectly good user-scope wiring), and user/local scope's shared physical
  `~/.claude.json` was read and parsed twice per invocation. Both fixed — a per-scope
  failure is now absorbed as "no wrapped servers there" rather than propagated, and the
  physical file is loaded once and reused across the scopes that point at it.

## [0.22.2] - 2026-08-06

### Fixed

- **The launch path baked into every wrapped MCP config was never executed by any test.**
  `install_mcp` writes `[sys.executable, "-m", "terse"]` as the command for every wrapped
  entry, so `python -m terse` is *the* production entrypoint — and `src/terse/__main__.py`
  sat at 0% coverage. The existing tests assert the config **string**
  (`entry["args"] == ["-m", "terse", "proxy", ...]`) and never run it.

  Demonstrated rather than asserted: breaking the import inside `__main__.py` leaves
  `python -m terse` raising `ImportError` on every wrapped server on every user's machine
  while **all 1,287 other tests pass**. Four tests now run the launcher's *own* return value
  as a subprocess — not a hand-written copy of the argv, so a test cannot keep passing after
  the launcher changes to something that does not run — and pin that the CLI's exit code
  survives to the shell (a wrapped entry that always exits 0 hides a failed proxy from the
  client supervising it), that the console script and `-m terse` agree, and that the
  absolute interpreter path is preserved.

  Note: `__main__.py` still reports 0% coverage, because the tests spawn a subprocess and
  `coverage` does not instrument it. The line is now tested; the number does not move.

  Review caught the first cut of these tests being **vacuous under `$TERSE_MCP_CMD`**: that
  documented override makes `terse_invocation()` return the operator's console script, so
  with it set the tests passed even with `__main__.py`'s import broken — the subprocess
  never touched `__main__.py`. An autouse fixture now clears it for every test in the file
  rather than leaving each one to remember. The console-script comparison also used
  `shutil.which("terse")`, which finds *a* terse rather than the one belonging to this
  interpreter, and failed on a machine with a global install; it now resolves the script
  beside `sys.executable`.

- **`.gitleaksignore` fingerprints had rotted, so the local secret gate reported three
  false positives on every full-history scan.** gitleaks emits a different fingerprint per
  scan mode — `file:rule:line` for `--staged` (the pre-commit gate), `commit:file:rule:line`
  for a full-history `gitleaks git` — and an entry in one format is silently inert in the
  other. Only the staged-mode entry existed. Each of the three findings was verified by
  reading the line at its commit before suppressing: all are fixtures for the argv/header
  **redactor's own tests**, asserting a fake credential scrubs to `***`, so the "secret" is
  the input to the scrubber under test. A scanner that cries wolf on every run trains its
  reader to ignore it, which is the one thing a secret scanner cannot afford.

  Review then caught the staged-mode entry being rotted too, in **both** directions: it
  pointed at line 193, which today is an ordinary policy fixture with no secret, so the
  pre-commit gate still false-positived on any edit to the real fixtures *and* carried a
  standing blind spot where a genuine credential landing at that line would pass silently.
  Both formats are now re-derived from live runs, with the re-derivation command recorded
  in the file, since line-numbered fingerprints will rot again.

## [0.22.1] - 2026-08-06

### Added

- **`terse stats --json` is now pinned field by field.** USAGE calls it "the raw aggregate,
  for scripts", which makes it a contract with people outside this repo — and it carried
  **37 fields across four nested shapes** with exactly two assertions on any of them
  (`total.blocks` and `total.raw_tokens`).

  The cost showed up immediately: in one day the liability blob gained
  `session_once_tokens`, `session_covered`, `free`, `uncertain` and a per-server `cadence`,
  the tool rows gained `encoded`, and `per_turn_tokens` was **redefined** from "every
  wrapped server" to "the recurring ones only". A consumer reading that key would have seen
  the number drop with nothing failing anywhere; prose in this file was their only warning.

  The manifests are deliberately exact rather than "at least these" — a removal or rename
  is a break, and an addition is a decision worth making on purpose, so the failure message
  says to update the manifest *and* note it here. Types are pinned alongside names (a
  consumer reading `per_turn_tokens` as an int must not one day get a string), `bool` is
  rejected where a count belongs (it is an `int` subclass and would be summed as 0/1), and
  the whole thing is driven through `main(["stats", ...])` because the composition —
  `cli` merging the ledger and the liability into one document — is part of the contract
  that neither function alone can be asked about.

  USAGE now also documents the three things a parser has to know: `versions` is an object
  keyed by version string rather than an array, `null` is a real answer and never means
  zero, and `primer_liability` itself can be `null` when the install could not be sized.

### Fixed

- **A corpus row with no `tool` key silently left the per-tool tables — and one with
  `"tool": None` crashed the report outright.** Found by a logic sweep, not by a diff.

  All three report surfaces (`report.py` markdown, `html_report.py`, `terminal_report.py`)
  built their tool list with `r.get("tool", "?")` and then filtered rows with
  `r.get("tool") == tool`. The two expressions agree on every input except one: a row where
  the key is **absent**. The set substitutes `"?"`; the filter compares `None` against it and
  matches nothing. The row's tokens vanish from the per-tool table while still counting in
  every other total on the page, and a phantom `?` row prints at 0/0/n-a. Executed: two rows
  of 1,000 and 5,000 raw tokens render a per-tool table summing to **1,000 of 6,000** — a 5x
  under-report with no error anywhere.

  A row carrying an explicit `"tool": None` fails differently and worse. The key exists, so
  `None` enters the set and `sorted()` raises `TypeError: '<' not supported between 'str'
  and 'NoneType'`: the whole report dies rather than mis-reporting. That second failure is
  why the fix normalises with `or "?"` on both lines rather than giving the filter a
  matching default — `or` covers absent, `None` and `""` in one expression.

  Not reachable from the CLI today: `measure_corpus` sets `tool` unconditionally and
  `_cmd_measure`/`_cmd_verify` are the only production callers, so no published number was
  ever wrong. These are public functions taking rows as an argument, though, and the `"?"`
  default is itself evidence that someone expected the key to be missable — the code
  intended to handle the case and handled it by dropping data.

- **A ledger record whose `decision` cannot be read no longer under-bills its primer.**
  `aggregate` tolerates a record with no `decision` field, counting it as `"unknown"` — it
  reached `blocks` but not `encoded`, so a server whose every readable block was unknown
  landed in `free` and the report told the operator it "costs nothing at all". `encoded` is
  now derived by excluding the two decisions that *prove* no terse marker shipped
  (`passthrough`, `unchanged`) rather than by including the two that may have
  (`compressed`, `diff`). Identical for any record terse has ever written —
  `classify_decision` returns exactly one of the four, and 0 of 2,115 records in a live
  ledger lack the field — so this only ever decides a hand-written or third-party line,
  where over-billing is the safe direction and under-billing is the one `_cadence` argues
  against.

  A related overclaim is corrected rather than shipped: `encoded == 0` was documented as
  *proving* the primer never attached. It does not. The attach guard fires on `"__terse_`
  appearing anywhere in the final content, and that text can come from the **downstream
  payload** — a code-search tool returning terse's own source, a doubly wrapped peer — so a
  `passthrough` result can attach a primer while classifying as `passthrough`, leaving
  `encoded` at 0 and the server filed under `free`. Reproduced, and pinned as a known gap in
  `test_primer_liability.py` with the wrong answer asserted, so the day it is fixed the test
  fails and says why. Closing it needs the ledger to record whether the attach fired, which
  is a shape change and a separate decision.

## [0.22.0] - 2026-08-05

### Changed

- **A multiproxy `-32601` now says which peers missed the listing, instead of only "unknown
  tool".** This is the half of #178 that does not need the design #178 withdrew.

  When a peer misses a `tools/list` broadcast, its tools are absent from the merged listing
  and a call to one is a clean `-32601` — correct, and indistinguishable from "no such tool
  ever". A non-conformant client holding names from an earlier listing therefore got a
  message that pointed at the wrong problem, for a tool whose peer was alive and merely
  slow. The error now names the peers that contributed nothing to the listing behind the
  current table, why, and to re-read `tools/list`. On a complete listing it says nothing
  extra: a permanent suffix would be noise on the common case and would imply a partial
  listing where there was none.

  **No route is carried forward.** #178 withdrew route *retention* after three review rounds
  found 11 defects, every one in the retention and ordering machinery rather than in the
  naming rule it protected, and closed with the advice that a revisit should *prefer a
  design where the table is derived, not accumulated*. So what is stored is a diagnosis of
  the listing that produced the current table — installed inside `_tool_state` under the
  same lock and the same seq guard as the route itself, replaced wholesale with it, and
  never consulted to resolve a name. A peer named in it is exactly as unroutable as before;
  a listing refused as stale installs neither its table nor its diagnosis.

  #178's other closing requirement was to decide up front what "the peer did not answer"
  means, since conflating those cases is where its round-3 defect came from. There are four
  and they stay four: `no reply` (absent from the broadcast — already named on stderr by the
  timeout path), `error` (a live peer refusing the method), `empty` (a peer exercising its
  right to export nothing — not a fault, so not warned about), and `malformed` (a reply
  whose `result.tools` is not a list). A non-empty list of entries with no usable `name` is
  `empty`, not `malformed`: from the client's side it is indistinguishable from exporting
  nothing, and `malformed` would accuse a peer that answered perfectly well.

  Only reasons a re-read might actually fix reach the client: a peer that legitimately
  exports zero tools is `empty` on every listing, so surfacing it would append "re-read
  tools/list" to every unknown-tool error the install ever produces. It stays in the
  diagnosis; it just isn't actionable.

  `error` and `malformed` also now warn on stderr at merge time, which nothing did before —
  only the timeout path warned, so a peer that *answered* with a refusal disappeared from
  the listing in total silence. `prompts/list` deliberately carries no diagnosis (most MCP
  servers answer it with `-32601`, so "contributed nothing" is the norm there rather than a
  signal) but keeps the same tuple shape, so the two surfaces cannot drift into different
  unpackings.

  Still open in #178, and untouched here: collision naming is computed per listing, so two
  peers exporting the same tool name can see it re-qualify between listings. That needs an
  actual cross-peer collision, which is empty on the fleets this feature targets.

## [0.21.6] - 2026-08-05

### Fixed

- **Four review findings against the primer-cadence split (#222).**

  - **A pre-cadence `--json` blob got the wrong legend.** The prose gated its per-cadence
    lines on `cadence or "per-turn"` and the break-even table's legend on a bare `cadence`,
    so on the exact backward-compat path both were written for — a liability blob from a
    terse that records no cadence — the table suppressed the `/turn` legend and printed the
    standalone one instead, directly under prose declaring the whole figure recurring. Two
    spellings of one default is how they drifted; there is now one helper.
  - **The `blocks` column could overflow its width.** It was narrowed to 11 on the
    reasoning that it only ever holds `N` or `tokenized/N` — but the pair form carried
    thousands separators and the live ledger already rendered `1,790/1,799`, exactly 11.
    One more order of magnitude would have broken the 80-column guarantee the table's own
    comment makes. The pair form drops its separators (it is a ratio to compare, not a
    magnitude to read) and the column is sized against a million-block ledger, pinned by a
    test that fails if any width changes without the others.
  - **A server called but never compressed was billed a primer it could not have paid.**
    The lazy primer attaches to a result carrying a terse wire form, so a standalone entry
    called a thousand times that never produced one — an all-passthrough policy, non-JSON
    payloads, a shape the codec never wins on — paid nothing. `blocks` counts every emitted
    block regardless of decision and cannot see that, which is the same mis-bucketing the
    split exists to fix, in the other direction. `aggregate` now also counts `encoded`
    blocks (`compressed`/`diff` only), and the inference is one-directional by design:
    `encoded == 0` proves the primer could not have attached, while `encoded > 0` does not
    prove it did (a minify-only `compressed` block carries no marker), so a non-zero count
    still bills — the over-billing direction this module argues is the safe one. A row that
    cannot report the counter falls back to `blocks`, the old coarser behaviour.
  - **A mixed install could read two true lines as jointly true.** The recurring and
    one-time lines each credit the same savings in full against their own charge. Not
    netting them is right — the units differ — but silence about it meant an install paying
    `per_turn x turns + once` could see "pays for ~1 turn" beside "covers the one-time
    charge at most ~1x" and read itself as break-even. Said once, explicitly, when both
    cadences are present.

## [0.21.5] - 2026-08-05

### Fixed

- **`terse stats`'s primer liability no longer charges a lazily-primed server as if it
  primed every turn.** This is the re-derivation #211 left as a follow-up, and until now the
  report gave standalone installs a headline that was wrong in the direction that costs the
  operator money.

  There are two cadences and the old model had one. A multiproxy router still primes
  eagerly, once, into its own merged `initialize.instructions`, which the client re-reads
  every turn as `cache_read` — genuinely recurring. A standalone `terse proxy` entry has
  been lazy since #211: its primer attaches to the first result carrying a terse wire form,
  so it is paid **once per session** if that result comes and **not at all** if it never
  does. Summing those into one `tok/turn` figure overstated a standalone install by the
  whole session's turn count.

  Worse than the headline was the advice under it. A never-called standalone entry was
  listed as *"paying but never called here — pure cost until they handle a compressible
  result"*, which is exactly inverted: those are the servers #211 made free. An operator
  acting on that line would unwrap the servers costing them least. Those entries now appear
  under `installed but not triggered this window, so costing nothing at all (#211)`, and a
  server whose ledger label could not be recovered is reported as *unknown* rather than
  being quietly filed as free — the same `None`-is-not-`0` discipline the rest of this
  report already keeps.

  The break-even table's `blocks/turn to break even` header was true of routers only, so it
  overstated the bar for every standalone entry by the session's turn count. The header is
  now `blocks to break even` beside a new `cadence` column that carries the unit
  (`per-turn`, `once/session`, `once/session (unpaid)`, `once/session (?)`), because the
  arithmetic is identical and only the unit differs.

  `--json` shape: `per_turn_tokens` is **redefined** to the recurring (eagerly-primed)
  entries only — a consumer comparing across versions will see it drop, and that drop is the
  correction. New alongside it: `session_once_tokens`, `session_covered`, `free`,
  `uncertain`, and a per-server `cadence`. `turns_covered` now settles the one-time charge
  out of the same savings before the remainder buys turns, and a standalone-only install —
  which has no recurring charge and previously got no bottom line at all — now gets a
  per-session verdict. A sub-1.0 ratio prints the shortfall in tokens instead of rounding to
  a `~0x` that reads as a measurement.

  Not modelled, deliberately: a session whose every compressible result also carried
  `structuredContent` never gets the lazy attach (the accepted gap at the attach guard), so
  it was called and still paid nothing. The ledger cannot observe that, and discounting for
  it would be the #144/#186/#188 defect family again. Over-billing by an unobservable
  exception is the safe direction.

## [0.21.4] - 2026-08-05

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix(docs): correct lazy-primer economics and undated ledger in POSITIONING.md (#220)

### Changed

- docs: positioning doc + resolve router primer economics (#219)
- chore: gitignore serena-mcp's stray project index (#218)

## [0.21.3] - 2026-08-05

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix: close the corrupt-sidecar-warning gap and the run_capture.sh TOCTOU; share the sidecar predicate (#217)

## [0.21.2] - 2026-08-05

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix: harden the _calls.json sidecar write and close the same directory-perms bug at its second call site (#216)

## [0.21.1] - 2026-08-05

### Added

- **`scripts/bench/mcp_servers/mcp_probe.py` now persists a probe's exact call arguments
  beside its corpus** (`<corpus_dir>/_calls.json`, #138 step 0). A saved §6 corpus dir
  previously carried no record of what it was called with — `{tool, shape, bytes, sha,
  captured_at, server, result_id, raw}`, no `arguments` — which is exactly what blocked a
  faithful cold re-probe of `memory`/`serena` after the 2026-07-30 round (documented as a
  blocker in that round's notes). The sidecar is `_`-prefixed so it reads as auxiliary, not
  a capture envelope; `toon_column.py`'s envelope glob now skips it by name instead of
  falling through to its (still-correct, just noisier) generic SKIP path.

### Fixed

- **§6 re-measured against the merge of #202, and a genuinely cold, argument-recording
  round settled the two-round-old memory/serena discrepancy (#138 phase 2).** Fresh probes
  of all 10 servers plus the repo-size sweep, using the new `_calls.json` sidecar so this
  round's numbers are reproducible in a way earlier ones weren't. Net effect of #202 on the
  codec column is small: `filesystem/directory_tree` 58.0% → 57.3% (within noise, not the
  direction #202 usually pushes); every other reproducing row is unchanged.

  Two fixture bugs found and fixed along the way, unrelated to #202: `express` and
  `fastapi` under `/tmp/mcp-fixtures` had silently become **1-commit shallow clones**
  despite being pinned at the right tag, degenerating the `git_log` row to a single commit
  — `git fetch --unshallow` recovered full history (6,103 / 7,521 commits) without a
  re-clone. And `memory`'s graph was confirmed empty (`MEMORY_FILE_PATH` pointed at a path
  verified not to exist) before this round's cold probe, closing the "was it warm?"
  question the 2026-07-30 round couldn't answer for lack of recorded arguments.

  **serena's published 22–37% never reproduces at either edge under two independent
  re-measures** (the 2026-07-30 reprocessing: 22.2–28.5%; this round's fresh cold probe:
  21.4–21.7%) — serena carries no session state, so this isn't a warm-state artifact; the
  most likely explanation is the original high end came from different, unrecorded
  arguments. §6 now publishes the reproducible range. **memory's gap is re-scoped, not
  resolved**: a genuinely cold graph this round measured 35.1–41.1%, inside the originally
  published 27–52% but below the 2026-07-30 reprocessed-capture's 43.9–54.1% — expected,
  since the percentage is a function of what's actually in the graph, which is
  round-specific by construction.

  Separately, and unrelated to #202: **cross-call diffing has been opt-in since #170
  (2026-07-28)**, which predates even the 2026-07-30 §6 round, so §6's "diff tier is
  universal, zero-config" framing was already stale and nobody had caught it. A plain
  `terse proxy` today reports `diff_off` for every row in §6's table; the repeat column
  now documents that it requires `--diff` explicitly (`mcp_probe.py` has no CLI hook to add
  a proxy flag, so this round used a small `TERSE_BIN` shim — see the updated
  `scripts/bench/mcp_servers/README.md`). The underlying mechanism is unchanged: re-run
  with `--diff` and every row reproduces its originally-published diff/text-diff behavior
  exactly.

  TOON column re-measured on the same fresh captures — **terse 48.0%, TOON 47.5%** over
  6,243 raw cl100k tokens across 14 JSON-encodable payloads: both serena rows flipped from
  TOON's best case in the 2026-07-30 round to terse's this round (`get_symbols_overview`
  21.4% vs 7.1%, `find_symbol` 21.7% vs 4.3%) — plausibly
  #202's union-schema tabularize, whose fold targets exactly serena's array-of-symbol-record
  shape, though the original round's unrecorded arguments mean this isn't provable, only
  plausible. TOON still wins the smallest, most uniform rows, same as before.

## [0.21.0] - 2026-08-04

### Added

- **Lazy primer (#168 phase 2): the format primer no longer rides on every `initialize`
  reply, paid every turn for the life of the connection.** It now attaches to the first
  `tools/call` result that actually carries a terse wire form — once per session, as a
  new leading content block ahead of the compressed data — instead of costing
  `servers x turns` of cached context regardless of whether a wrapped server is ever
  called. A session that never touches a wrapped tool now pays zero primer bytes; a
  session with real tool use pays it once. This is the runtime implementation of the fix
  the `inline_ok` fluency arm (below, #210) measured safe before it shipped.

  `lazy_primer=True` is the new `Interceptor`/`run_proxy` default (not a CLI flag —
  ships the same way #204's union-schema tabularize did). Multiproxy peers are
  explicitly excluded (`_build_peers` passes `lazy_primer=False`): the router already
  primes eagerly, once, via `union_primer` at its own merged `initialize` — a peer going
  lazy too would just attach a second, redundant explanation on top of that.

  A result carrying `structuredContent` never gets the lazy-attach treatment, even when
  a terse marker landed there: measured (`scripts/probe/structured_content/`) that
  Claude Code discards the text block entirely whenever `structuredContent` is present,
  so a primer block sitting next to it would be silently thrown away. The proxy waits
  for a text-only compressible result instead. A session whose *every* wrapped-tool call
  happens to carry `structuredContent` under a rewriting-eligible client never finds that
  later call — a known, narrow, accepted gap (comprehension degrades to the un-primed
  level for that traffic, not to broken output), not a silent one.

  `terse stats`'s primer-liability figure was left by this change as a worst-case upper
  bound for a standalone `terse proxy` entry — reflecting the old always-eager cost rather
  than the new one-time one — and stayed live and accurate for a multiproxy router entry.
  That re-derivation has since landed: see the `Fixed` entry above, which splits the two
  cadences and stops reporting the servers this change made free as "pure cost".

## [0.20.0] - 2026-08-04

### Added

- **A fourth fluency arm, `inline_ok`, measures #168's "lazy primer" proposal before it
  ships.** `run_payload`/`run_fluency` now also ask each question with the format primer
  riding inline in the user message (no system prompt at all) — the exact delivery mode a
  primer attached to the first compressed result would use, instead of `initialize`'s
  `instructions` field. `build_fluency_report` renders it as a `terse+inline` column,
  falling back to `n/a` (never `0%`) for older result files that predate the arm.

  Measured against two live models over the synthetic stress corpus
  (`scripts/gen_stress_corpus.py`): `glm-5.2` scored raw 100% / terse 96% / terse+primer
  100% / **terse+inline 100%**; `deepseek-v4-flash` scored raw 100% / terse 92% /
  terse+primer 96% / **terse+inline 100%** — inline delivery matched or beat the
  system-level primer on both. Answers the open question #168 left explicitly blocking a
  lazy-primer implementation: comprehension does not depend on the primer riding in the
  system slot.

## [0.19.1] - 2026-08-04

### Fixed

- **The shape classifier was narrower than the codec, so the measurement stack under-fired
  on union-schema traffic (#204).** `capture._find_record_list` still used the strict
  identical-keyset rule after #202 widened what the tabularizer folds. A payload where two
  thirds of the rows carry `line` bucketed as `compact-json` with **no record list** while
  the codec compressed it **55.8%** — so `classify_shape`'s buckets, `measure`/`report`
  coverage, `policy_gen`'s auto drop-path generation, `dropeval` and `fluency.questions` all
  skipped exactly the traffic union-schema tabularize was built for. Deferred out of #202 on
  purpose, because moving the measurement stack in the same commit as the codec leaves no
  clean before/after; this is that change, with the numbers.

  `transforms.is_tabularizable` is now the single canonical "would the codec fold this"
  rule, and capture uses it. On synthetic runecho-shaped payloads: `compact-json` →
  `array-of-records`, `extract_records` 0/3 → 3/3, drop-path 0/3 → 3/3. End to end,
  `terse policy generate` over a runecho-shaped corpus now emits
  `↳ drop-candidate symbols[].body (~96% of tokens, 100% unique)` where it previously
  emitted nothing — the same shape the regression test carries, so the claim is auditable
  from the branch.

  On the tracked bench corpus exactly **one** payload moves — `gh_issues`, from
  no-record-list to `array-of-records` — which is the same payload, and the same nested
  non-uniform array, that #202 moved from 32.7% to 38.8%. The classifier and the codec now
  agree on all nine; a test pins that equivalence over the corpus plus the shapes that broke
  it before, so the two cannot drift again silently.

  **The uniform-keys guarantee is gone, and that was the real work.** `extract_records` used
  to promise callers could index every record by `records[0].keys()`. Three consumers relied
  on it and would now raise `KeyError` on a record missing a key: `fluency`'s column pickers,
  `gen_questions`, and `dropeval`. They now take the intersection via the
  `_intersection_cols` helper the nested-record path already had — which is also the
  semantically right answer, since a column some records lack cannot be the subject of "for
  the record whose id is X, what is Y?". `probes` needed no change: it walks `rec.items()`
  and already counts per-field presence.

  That helper changed from sorted to **first-seen** order, because the pickers take the
  first column matching their predicate and sorting would re-pick id/target columns. For a
  uniform list first-seen returns exactly `records[0].keys()`, so no pre-existing payload's
  questions move — verified across the whole bench corpus, where only `gh_issues` changes at
  all. It is **not** a no-op for the helper's original caller: `_nested_record_group` (#71)
  only ever sees non-uniform child lists, so a structure payload whose keys run `sym` before
  `path` now enumerates `sym` where it enumerated `path`. Same information, different
  question — the alternative was leaving the flat path picking columns alphabetically.

- **README and BENCHMARKS published the pre-#202 numbers, and nothing was checking (#206).**
  Union-schema tabularize moved the bench corpus and both documents went on claiming the old
  figures. §1: weighted total **58.3% → 59.1%**, `gh_issues` **32.7% → 38.8%** — every other
  row and the entire TOON column byte-identical, the whole delta being one payload's
  tabularize tier (+0 → +4,718) where union-schema reaches a nested non-uniform array. §3
  moved with it, because its "full re-send" column *is* that same single-shot codec:
  `gh_issues` 32,608 → 29,611 tok (**86.4% → 85.0%**) and the total 152,837 → 149,840
  (**73.7% → 73.2%**). Both re-measured from live `benchmark.py` / `diff_demo.py` runs rather
  than hand-patched. §2's width sweep re-ran byte-identical, as expected for synthetic
  uniform tables.

  §3 is worth calling out separately: it was missed on the first pass at this very issue, by
  the same mechanism — a table nobody thought to re-run because nothing failed when it went
  stale. Review caught it.

  §4's headroom comparison inherited the same stale cell, and fixing it **flipped a
  conclusion**: at 38.8% terse now beats headroom's lossy 33.1% on `gh_issues`, so the honest
  reading goes from "on two of four files headroom's lossy number beats terse's lossless one"
  to one file. That paragraph now also states what it is comparing — terse at 2026-08-04
  against headroom at 2026-07-30, since re-running headroom means standing its proxy back up.

  **§6 is explicitly NOT re-measured** and now says so in a banner: it needs three pinned repo
  clones and live `npx`/`uvx` servers, and whether #202 moves it depends on whether those
  servers emit non-uniform record arrays — plausible, unverified, not assumed either way.

  The durable half is `tests/test_published_benchmarks.py`. Both files are hand-maintained
  prose, so "remember to update the table" was the only thing standing between a codec change
  and a stale published claim — principle #134's argument exactly. Now asserted against live
  output on the tracked corpus: §1's rows (raw tokens, record count, terse %) and total, §3's
  diff table (by importing `diff_demo.py` rather than reimplementing its churn model), §4's
  terse column (which reprints §1's numbers in a different table shape and so drifts on its
  own), that no corpus payload is missing from a table, and that the two documents agree
  where they overlap. Percentages are checked by `round(measured, 1) == published`, not a
  tolerance window — at 80.3477 a `< 0.05` band left `gh_workflow_runs` 0.002pp from failing
  while correct. The weighted total mirrors `benchmark.py`'s rule of dropping a payload that
  fails its round-trip, so the two cannot diverge if one ever goes lossy.

  Scoped to what runs keylessly in CI: §1's TOON column comes from a pinned npm encoder there
  is no node for and cannot move without a visible dependency bump; §4's headroom columns need
  a live proxy; §6 needs three pinned repo clones and live servers. Those are dated in the
  documents instead. Verified by mutation — 10 hand-built regressions (stale cell in each
  section, wrong raw count, wrong record count, deleted row, 0.1pp drift, one document updated
  without the other) and all 10 fail the suite.

## [0.19.0] - 2026-08-04

### Added

- **Union-schema tabularize — non-uniform record arrays now fold.** `_uniform_dict_list`
  required an identical key set across every row, so `runecho structure` (where `import`/
  `export` rows lack the `line` that `class`/`function` rows carry) fell straight through
  the tabularizer: **10% of real fleet traffic, 190 blocks, 181K tokens, compressing at
  0.6%**. The header is now the union of all keys in first-seen order; a row that omits a
  key gets `null`, or the `__terse_absent__` sentinel in a column that also carries an
  explicit `null`, so "key omitted" and "key present, value null" stay distinguishable.
  `absent_cols` / `sentinel_cols` in the table header tell the decoder which is which, and
  are emitted only when a hole exists. A 50%-fill density gate refuses payloads too sparse
  to amortize the header, with `compress_with`'s emit-only-if-smaller (#154) as the
  backstop. Measured: 14–20% on runecho-shaped payloads, 62.7% on the GitHub bench corpus
  (no regression against the 58.3% baseline).

  Three things this needed beyond the codec:
  - **The primer had to learn the vocabulary, and it is load-bearing.** A wire token the
    codec emits but the paragraph never names is invisible to the reader — a fidelity gap
    the round-trip gate structurally cannot see, since it only proves terse decodes its own
    output. Measured on 24 absent-vs-null questions over a non-uniform table, three models
    (Haiku 4, DeepSeek-V4-Flash, GLM-5.2) scored **54.2% / 54.2% / 79.2%** against the
    unextended paragraph — a binary question answered near chance — and **95.8% / 100% /
    100%** with the shipped paragraph (single trial, n=24). A cheaper encoding exists (one sentinel in
    EVERY absent cell, no index arrays: 87 tokens total, also 100% on the same probe),
    traded away because it costs more on the wire at scale (31.8% vs 33.5% saved on a
    200-record table) — the primer is paid per turn, the wire per payload.

    Making that coupling a test rather than a promise then surfaced a **pre-existing** gap:
    `subcols` has been on the wire since nested key folding shipped and the paragraph never
    named it, so a model received a nested positional tuple with no rule for reading it. Now
    explained. `PRIMER_TABLE` goes 55 → 155 cl100k tokens and the whole primer 402 → 555,
    which makes this the section to attack first if #168's per-server tax reopens.
  - **The offline eval's primer was brought back in step.** `fluency.pack.PRIMER` is served
    against the same `compress()` output; left on the old rule, every future `terse fluency`
    run would have read as a codec fidelity regression caused by this change, when the stale
    preamble belonged to the harness doing the measuring. A test now pins both primers to a
    real emission.
  - **The sentinel needed a real collision guard.** `ABSENT_MARKER` is the one marker the
    codec writes into a *cell*, so it collides as a string VALUE and never as a key —
    listing it beside the envelope keys in `_RESERVED_MARKERS` was a silent no-op. A record
    whose own value is `"__terse_absent__"` in a sentinel column decoded as "key absent" and
    lost the field; `_lossless_stage`'s verify-before-emit caught it, but only by discarding
    the whole payload's compression and recording a `gate_fail` that
    `policy_gen._tool_decision` reads as "this tool's shape defeats the codec", marking it
    passthrough permanently. `has_terse_marker` now screens reserved string values too, so
    `apply` passes such a payload through one layer earlier.

  Known and deliberately deferred: `capture._find_record_list` still uses the strict
  `_uniform_dict_list` rule, so it is now NARROWER than what the codec folds — a payload
  where two thirds of the rows carry `line` classifies as `compact-json` with no record
  list while the codec tabularizes it at 30.6%. Everything downstream (`classify_shape`
  buckets, `policy_gen`'s auto drop-path generation, `dropeval`, `measure` coverage,
  `fluency.questions`) therefore under-fires on exactly the traffic this change targets.
  Not widened here on purpose: that function feeds the measurement stack, and moving it in
  the same commit as the codec leaves no clean before/after. The comments that claimed the
  two could "never drift" have been corrected to describe the gap.

## [0.18.1] - 2026-08-04

### Fixed

- **Three `policy.py` soundness gaps the #198 review parked as one issue (#199).**
  1. `_glob_covers_server` decided cover by string equality against three literal forms
     while `select` matches by `fnmatch`, so a server literally named `kb[1]` counted
     `kb[1].*` as covering — `[1]` is a character class and that rule matches none of its
     tools. Cover is now proved structurally: the whole-world glob, or a metacharacter-free
     literal prefix no longer than `{server}.` followed by `*`. Deliberately NOT a probe
     against a representative name (`fnmatch(f"{server}.x", glob)`): that is unsound the
     other way, calling `kb.?` and `kb.*x` covering because both match the literal `kb.x`
     while matching no real tool — which terminates `has_drop`'s walk early and drops the
     dropped-field paragraph **and** the `terse.retrieve` tool from a server that still
     reaches the drop rule, leaving the model an unretrievable `__terse_dropped__`. Refusing
     an unproven cover is the safe direction: the walk continues and the caller
     over-approximates, which is already `reachable_tiers`' contract.
  2. `_match_candidates`' reported gap — that a peer-qualified `gh__gh.api.items` leaves a
     server-scoped `gh.*` rule with nothing to match, because candidate[0] keeps the `__` —
     **was investigated and is not a bug; no change shipped.** `select` also tries the BARE
     candidate `gh.api.items`, which `gh.*` fnmatches, so the rule was never missing
     (verified across `gh__gh.api.items`, `gh__gh.rate_limit`, `mcp__gh.search`). The
     proposed fix was actively harmful: candidate order is major over rule order, so
     synthesizing `gh.gh.rate_limit` at position 0 lets `gh.*` win one candidate *before* a
     specific rule can match the bare name. On terse's own `policy.example.json` that turned
     `{"tool": "*.rate_limit", "tiers": []}` — an explicit passthrough — into `gh.*` with
     `result[].body` truncation (2,125 bytes lossless became 626 with the body cut), and
     turned USAGE.md's documented `{"tiers": [], "capture": false}` recipe for keeping a
     credential tool's output off disk into a rule with `capture: true`, making the payload
     eligible for the `--capture-dir` corpus and the `--debug-log` replay trace. The guard
     is unchanged and now carries that measurement; a regression test pins that a specific
     passthrough rule keeps its `tiers: ()`, its `capture: false` and its empty field map.
  3. `has_drop` ignored `server_never_lossy`. A server that structurally forbids every drop
     still paid the 64-token dropped-field paragraph and advertised a `terse.retrieve` it
     could never mint a handle for.

## [0.18.0] - 2026-07-31

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(stats): stamp the writing version and canonicalize tool identity (#200)

## [0.17.1] - 2026-07-31

### Fixed

- **`has_drop` was the last primer gate ignoring the server (#168).** The other four
  (`emits_table`/`emits_dict`/`emits_embedded`/`emits_diff`) all take a server name and walk
  rules the way `select` does; `has_drop` scanned every rule in the file unconditionally. Peers
  commonly share ONE policy file, so it effectively answered *"does this file contain a drop
  rule"* — and a server whose own rule totally covers it, and can therefore never reach the drop
  rule at all, still paid the 64-token dropped-field paragraph **and** advertised a
  `terse.retrieve` tool it could never mint a handle for.

  Measured on this operator's live policy: **192 tok/turn** across six separately-wrapped
  servers (`secret-broker`, `gh`, `runecho` each 212 -> 148; `codegraph`, `kb`, `shot-mcp`
  unchanged because they genuinely reach the drop rule). That is the six-separate-proxies
  configuration #168 measured at **+23.1% RAW**. A single `multiproxy` router is **unchanged at
  212** — one peer that can drop is enough for the union primer, since the client sees one
  server and cannot be told per-peer. On an install that is one router plus a standalone
  proxy launched without `--server-name` (`server=None` -> whole-file scan, unchanged), the
  realized saving is **0 tok/turn**; 192 is the fan-out configuration, not a number anyone
  banks by upgrading.

  Answering `terse.retrieve` is deliberately left **ungated** while advertising it is gated:
  answer >= advertise, matching `multiproxy`. A retrieve call reaching a server this build
  believes cannot drop is the symptom of the `_glob_covers_server` cases in #199, and
  forwarding it downstream would turn one wasted paragraph into an unredeemable handle plus
  a `-32601` from a server that never had the tool.

  The narrowing taken is only the sound one, the same `reachable_tiers` uses: terminate the
  walk at a rule that totally covers the server, because `select` returns the first match.
  A rule whose glob merely *looks* scoped elsewhere still counts — `_match_candidates`' second
  candidate is the tool's own unqualified name. Pinned by an under-inclusion invariant test
  (`select` drops for a server => `has_drop(server)` is True), because the failure directions
  are not symmetric: a surplus paragraph costs tokens, a missing retrieve tool costs a handle
  nobody can redeem.

## [0.17.0] - 2026-07-31

### Added

- **Per-server break-even in `terse stats` (#175).** The primer-liability block gained a
  per-server table: `primer`, `blocks`, `saved/block`, and `blocks/turn to break even`. #175
  established the rule — *wrap a server when its typical payload saves more than
  `primer x turns-per-call` tokens* — and then computed the evidence for it by hand from the
  ledger; this makes it self-service. `terse stats --json` carries `saved_per_block`,
  `blocks_to_break_even`, `tokenized_blocks`, and `break_even_verdict` per server under
  `primer_liability.servers`.

  Stated per **block**, not per call: a block is what the ledger counts — one record per
  emitted tool-result text block, which is `>= 1` per call and moves with join behaviour by
  design (#141). A `/call` label over that counter would silently overstate the break-even
  by the blocks-per-call factor, so the reported bar is deliberately the conservative one.

  A rate is a number **or** a verdict naming why there isn't one — never a `0` standing in
  for a missing measurement, because each of these accuses the install of something
  different and `None` alone cannot tell them apart: `no ledger label` (the entry matched no
  ledger rows, so we cannot even say it went uncalled), `never called`, `no token data`
  (recorded without tiktoken — savings in *tokens* are unknown, not zero, and dividing a
  cl100k primer by a char-derived rate would silently mix units), `primer unknown` (the rate
  is real but the policy could not be read), `no primer` (a default-deny policy emits none,
  so there is nothing to earn back), and `never` (a known non-positive rate, which no call
  volume earns back — the one verdict here that should stop an operator, so it is a word
  rather than a large number).

  The denominator is the **tokenized** block count, not every block: `aggregate` counts
  every record in `blocks` but only tokenized ones in the token sums, so a ledger spanning
  an offline session (`count_cl100k` returns `None`) and later online ones would divide a
  partial numerator by a full denominator — always understating the rate, i.e. always
  arguing to unwrap a server that is in fact paying for itself. The table prints
  `tokenized/blocks` when they differ, so the contamination shows in the column whose number
  it changed. A router's rate pools every peer it fronts, matching the single union primer
  it actually pays.

## [0.16.2] - 2026-07-31

### Fixed

- **`mcp-status` labelled every router row's diff before the peers policy resolved (#191).**
  A router entry carries `--config`, never `--policy`, so `policy` was still `None` when the
  diff label was computed and `_peers_policy` only ran eight lines later. Every `router` /
  `router-ambiguous` row therefore printed `default (off)` even when the shared peers policy
  set `"diff": true` — the proxy diffs, status said it does not. Same label-vs-reality
  divergence as #181, in the one branch #188/#190 didn't reach. A fleet whose peers carry
  *different* policy paths now reports the answer they agree on (`policy (on)`) rather than
  the dataclass default, and `peers (mixed)` only when they genuinely disagree.

- **`install-mcp --diff` help text claimed diffing was already the default.** #170 reverted
  #75's default-on, so the flag's own help sent operators to omit the one flag that turns
  diffing on — the same divergence one layer up, in the text.

## [0.16.1] - 2026-07-31

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix(codec): give the lossless gate a NaN-aware equality (#187) (#195)

## [0.16.0] - 2026-07-30

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(docker): Dockerfile + demo downstream so the Glama listing builds (#194)

### Changed

- docs: add the Glama MCP server score badge to the README

## [0.15.0] - 2026-07-30

### Added

- **`terse stats` now shows what the primer costs (#168).** The ledger charges terse for the
  payloads it compresses and never for the context it adds, so `terse stats` could report a
  win in a session that was a net loss — measured from outside terse as a **14.0% win at one
  wrapped server and a 2.1% loss at three**. The report gains a `primer liability` block:
  tokens per turn across the installed wrapped servers, how many turns the window's savings
  cover, a `NET NEGATIVE` call-out when they cover less than one, and a list of wrapped
  servers that pay every turn but were never called. `terse stats --json` carries the same
  under `primer_liability`.

  **It does not charge a per-turn cost into the ledger, deliberately.** `turns` is not
  observable: a `terse proxy` is a stdio process that sees one `initialize` per process
  lifetime and then `tools/call` requests, several of which can share a turn — nothing in
  MCP reports the client's turn count. Inventing one would be the #144/#186/#188 defect
  family again, a number describing something the code never measured. A break-even
  statement gives the operator the same decision with no fabricated denominator.

  Sized from the **install**, not from the ledger, because a wrapped server nobody called
  still ships its primer every turn and contributes zero ledger rows — sizing it from the
  ledger would hide exactly the worst case. Each server is measured from its own policy via
  `build_primer`, so a default-deny server correctly pays 0 rather than a shared constant; a
  router is sized as one `union_primer` over its **peers'** names, since gating the union on
  the router's own name tests rules like `kb.*` against the router and over-reports. A server
  whose policy cannot be read is excluded and the total is labelled a lower bound, rather
  than substituting the built-in default and overstating.

## [0.14.3] - 2026-07-30

### Fixed

- **`mcp-status` resolved a RELATIVE `--policy` path against the scanner's own cwd (#188).**
  `_default_diff_label` read the file and reported its `diff` setting, but a relative path
  resolves against the MCP *launcher's* cwd — which a status scan cannot know, which is
  exactly why the `policy_missing` check two lines away already skipped relative paths. So
  the label could confidently report the setting of whatever `policy.json` happened to sit in
  the scanner's directory. It now reports `policy (relative path — unknown)`.

  Falling through to the dataclass default (`default (off)`) was the first fix and was
  rejected in review: the file *does* state a value, the scanner simply cannot reach it, so
  naming the built-in default is the same label-vs-reality divergence #181 exists to kill —
  just pointing the other way, and with nothing to warn the reader that the value is a guess.
  An unreachable value is now reported as unreachable. (`do_install` always writes an
  absolute `--policy`, so only a hand-edited entry reaches this branch.)

## [0.14.2] - 2026-07-30

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

- **`measure` gated a different pipeline than the one it scored — in both functions, and the
  first fix over-reached (#186, completed by #188).** `embedded`/`tier_total` are computed
  from `compress_with(..., embedded=True)`, but the round-trip gate ran the default
  combination, so a failure that appeared only with the tier enabled would have kept its
  savings banked and fed them to `policy generate`.

  The first pass fixed `measure_payload` and left `measure_joined` untouched — which is the
  path that actually matters: `policy_gen._tool_decision` calls `measure_joined` FIRST for
  every result group and falls back to `measure_payload` only when the join refuses, so on a
  multi-block fleet the new gate never ran. It also folded both checks into ONE flag, so an
  embedded-only failure zeroed `minify`/`tabularize`/`dictionary` as well and the generator
  returned `tiers: []` — full passthrough for a tool whose default pipeline had just
  round-tripped perfectly, plus a report claiming the codec was not lossless for it.

  Both gates now run in both functions and are reported **separately**. `roundtrip_ok` covers
  the default pipeline and still disqualifies the tool; the new `embedded_ok` covers the
  opt-in fold and costs only that tier — `tier_total` falls back to the default pipeline's
  own `raw - compressed` rather than to 0, `policy generate` drops just `embedded` and says
  why (`embedded dropped — 1/4 result(s) failed the embedded round-trip`) instead of hiding a
  losslessness failure behind "below threshold", and `terse verify --json` / the measurement
  report carry an `embedded_gate` verdict so the split cannot make the failure invisible to
  readers that filter on `roundtrip_ok` — the markdown report, `terse verify --json`, and
  the `--html` banner all carry it. (The runtime was never at risk in any of this —
  `_lossless_stage` independently self-checks the actually-applied combination.)

  Two consequences of the split, both caught in review before merge and both about a
  published number rather than the codec:

  - **A dropped tier must not be counted in the savings it advertises.** `emb_fail` drops
    `embedded` for the whole tool — the policy matches on tool *name*, so there is no
    enabling it for only the results that round-tripped — yet `tier_total` still carried the
    embedded saving of every row that *passed*. Measured on crafted rows: a tool reported
    `22.5% saved` on tiers delivering `0.0%`, and cleared the passthrough threshold solely on
    savings the generator had just refused to enable. `total` now falls back to the surviving
    tiers' own `raw - compressed`, skipping rows the default gate already zeroed.
  - **"Not evaluated" is not "failed".** When the default gate fails the embedded pipeline
    never runs, and `embedded_ok` stays `False` rather than claim a pipeline is good on no
    evidence — so every reader qualifies it with `roundtrip_ok`. Without that, the report
    printed *"The default pipeline passed, so the savings below stand"* directly beneath
    *"INVALID — 1/1 payloads FAILED the round-trip gate"*, listing the same sha twice, and a
    CI job gating on `embedded_gate.ok` could not tell an opt-in-tier defect from total codec
    failure.

- **`_default_diff_label` now survives a pathologically deep policy file** (`RecursionError`
  joins the caught set — it is the *only* `json.loads`-on-file site that catches it; an
  earlier draft of this entry claimed it "matches every other" such site, which is false:
  `capture.py` catches `JSONDecodeError` alone and `policy.py` / `multiproxy.py` /
  `install_mcp.py` catch nothing).
  Its truthiness check is deliberately UNCHANGED and now pinned by test: `load_policy` builds
  the policy with `bool(doc.get("diff", False))`, so `"diff": "false"` genuinely diffs at
  runtime, and reporting `policy (on)` is correct. A review flagged the truthiness as a bug;
  tightening it to `is True` would have made the label print "off" while the proxy diffs —
  reintroducing the label-vs-reality divergence #181 was filed to kill.

## [0.14.1] - 2026-07-30

### Fixed

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

## [0.14.0] - 2026-07-30

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

## [0.13.0] - 2026-07-30

### Added

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

### Fixed

- **`policy.example.json` disabled `dictionary` on `kb.*` from a measurement that predates
  the #116 multi-block join (#144).** The original call (+2.6% total, not worth the tier)
  was made before that join gave `dictionary` a multi-block record array to fold into.
  Re-measured on 1,657 real captured payloads with the join in the path: fleet-wide 7.5% →
  8.0%, and per-tool up to `lodestone_search` 10.9% → 44.0%. `kb.*` now ships with
  `["minify", "tabularize", "dictionary"]`. The underlying `policy autotune` generator was
  never stale — it re-derives the marginal-savings threshold from fresh measurements every
  run — only this hand-authored example had gone stale after a codec change landed under it.

## [0.12.0] - 2026-07-29

### Added

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

## [0.11.0] - 2026-07-29

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

## [0.10.0] - 2026-07-28

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(policy): autotune reads the savings ledger, and install-mcp points at it (#136) (#176)

## [0.9.0] - 2026-07-28

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(bench): multi-run arms with modal-turn outlier control (ab_session) (#174)

## [0.8.1] - 2026-07-28

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix(install-mcp): classify wrapped-ness from the config, not stash membership (#172) (#173)

## [0.8.0] - 2026-07-28

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(policy)!: flip `diff` to default-off — its primer paragraph costs ~900-2,700x what the tier saves (#170) (#171)

## [0.7.0] - 2026-07-28

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat(proxy): assemble the primer per-policy so a server documents only what it can emit (#168) (#169)

## [0.6.0] - 2026-07-24

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat: bare `terse policy autotune` resolves --policy/--corpus from the install-mcp wiring (#136) (#167)

## [0.5.2] - 2026-07-24

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix: ledger counts BLOCKS, not "results" — rename the mislabelled field (#141 part 2) (#166)

## [0.5.1] - 2026-07-24

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Fixed

- fix: partial multi-block join — fold the record run, keep the rest per-block (#140) (#165)

## [0.5.0] - 2026-07-24

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

### Fixed

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

## [0.1.0] - 2026-07-17

_Reconstructed from the commit log — this release shipped without a hand-written
entry. The lines below are the commit subjects verbatim, not a summary._

### Added

- feat: fluency --bars terminal forest plot (#56)
- feat: terminal-bar mode for measure/verify (#51 fast-follow) (#54)
- feat: charted HTML report mode for measure/verify (#52)
- feat: policy generate auto-detects drop-to-retrieve candidates (#47) (#48)
- feat: Tier-1 lossy drop-to-retrieve (#10) (#45)
- feat: content-defined-chunking diff for non-JSON tool output (#44)
- feat(policy): `terse policy generate` — auto-author a lossless policy from a corpus (#24) (#40)
- feat(proxy): fail-fast on non-stdio downstream + document stdio-only wiring (#19) (#39)
- feat(proxy): --debug-log replay trace of raw→decision→emitted (#23) (#38)
- feat(install-mcp): --capture-dir passthrough for live-traffic measurement (#36)
- feat(proxy): --capture-dir to tee live tool results into a corpus (#32) (#34)
- feat: VERIFY.md + terse verify (self-contained verification report) (#29)
- feat: install-mcp / uninstall-mcp — Claude Code integration installer (#27)

### Fixed

- fix: address code-review findings on HTTP/SSE transport + multi-peer fan-out (#57)
- fix: restrict fluency-pack permissions and redact secrets in install-mcp output (#50)
- fix: restrict fluency-pack permissions and redact secrets in install-mcp output (#49)
- fix: close TOCTOU chmod window, harden capture.py, dedupe report.py, split lint job (#43)
- fix(proxy): reliability — reap orphaned child, bound pending map, guard diff desync (#33)

### Changed

- release: tag-based versioning (hatch-vcs) + GitHub Release CI (#88)
- version: single-source from __init__.py + add `terse --version` (#87)
- policy: per-rule "capture": false — never persist a tool's payloads (#85) (#86)
- policy: make server-scoped rules match servers that don't self-prefix (#83) (#84)
- live savings ledger + terse stats: payload-free per-result records, default-on (#82)
- docs: drop stale Phase-0/spike framing, acknowledge TOON as prior art (#81)
- codec hot-path fix (#79) + fluency package split (#78) (#80)
- audit fixes: checked types + curated lint; install-mcp drift guard; strict policy keys; transport credential guard (#77)
- flip cross-call diffing to default-on; add --no-diff opt-outs (#76)
- diff drift soak: mechanical long-chain pytest + fluency --diff-soak depth eval (#75)
- install-mcp: opt-in --diff wiring; docs: diff fluency is now validated (#74)
- fluency: cover nested-record tools (structure) so proxy --diff is validated (#71) (#72)
- chore: remove all Anthropic API integration (never used) (#73)
- security: fix the three remaining red-team lows (#70)
- security: restrict downstream URL schemes to http/https; prune config backups (#69)
- #64 Phase 2: broadcast/merge resources|prompts|ping + route reads (#68)
- #64: measurement infra — capture-order replay + probe value-overlap gate (#67)
- Add terse probe --cross-server: #64 Phase 0 cross-peer redundancy gate (#65)
- Add terse fluency --text-diff-eval: behavioral gate for text-diff codec (#63)
- Add test coverage for _secure_io restricted-permission helpers (#62)
- add measure --history: track token savings across runs, not just one snapshot (#61)
- add mcp-status: read-only enumeration of terse-wrapped servers across all scopes (#60)
- install-mcp: support project (.mcp.json) and local (nested projects.<path>) scope (#59)
- HTTP/SSE transport, multi-peer fan-out, and drop-to-retrieve eval (#5, #10) (#55)
- chore: harden install-mcp backups, dedupe report verdict math, add CLI/lint gate (#42)
- chore: add .runechoguardignore to silence a runecho-guard false positive (#41)
- ci: run pytest on push + PR across Python 3.11-3.13
- policy: guard reserved-marker collisions (#6)
- proxy: keyframe diff-anchoring + recursive record-list detection (#15)
- proxy: inject one-time terse/diff format primer via initialize.instructions (#13) (#14)
- Cross-call diffing, multi-trial fluency, and Tier-1 truncate (#12)
- fluency: validate object-valued aliases (the #4 follow-up) + fix unhashable-legend bug
- transforms: whole-subtree aliasing (Tier 0.5) — fold repeated objects, not just strings
- fluency: harden scoring from the code-review pass
- fluency: measure whether a model reads the compressed form as well as raw JSON
- tier0: row-count hint in table header — close the enumeration recall gap
- docs: document the MCP proxy (README/TECHNICAL/USAGE)
- proxy: MCP stdio middleware — compress downstream tool results per policy
- docs: README + TECHNICAL + USAGE (/docs gate before first push)
- policy: selective per-tool compression shell + `terse compress`
- validate: cross-tokenizer invariance check (cl100k vs o200k) — keyless ground truth
- cleanup: drop stray runecho .ai/ index dirs, gitignore them
- report: per-tool savings table + fix trailing-newline shape misclassification
- tier0: nested key folding — hoist uniform-dict columns into a shared subcols header
- tier0.5: lossless dictionary coder — fold repeated string values via inline legend
- probe: value-redundancy + cross-call-overlap ceiling probes (Tier 0.5 go/no-go)
- measure: corpus capture + per-tier/per-bucket token measurement
- scaffold: terse Phase-0 spike — lossless Tier-0 spine + round-trip gate
