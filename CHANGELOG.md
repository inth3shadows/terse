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

### Fixed

- `tune --drop-eval` evaluated a drop suggestion attached to a `tiers: []` rule to
  **zero questions, with no disclosure** (#375). `generate_policy` attaches
  `_suggested_fields` to a tool that scored under `--threshold`, whose `tiers` is
  therefore `[]`; `activate_suggestions` promoted the fields but left the tiers, and
  `policy.apply` reads `tiers: []` as an explicit hands-off passthrough and returns
  before the drop step. The tool appeared nowhere in the report — a run that never
  tested it read exactly like a run it passed. `activate_suggestions` now restores the
  doc's `defaults.tiers` on an entry it promotes a suggestion onto (the in-memory eval
  copy only — the disk path's `_keep_lossy_inert` refusal is unchanged), and the run
  names the lifted rules before printing its verdict.
- The SHIP directive no longer prescribes the rename that #375 proved is inert. It said
  "enable the verified fields by renaming `_suggested_fields` -> `fields`"; for a rule
  whose `tiers: []` the eval had just lifted, that rename alone leaves the passthrough in
  place and the drop never fires. `dropeval_next_step_line` now takes the lifted rules and
  says the rename must be accompanied by setting tiers.

### Added

- The drop-eval discloses what it could not measure. Every "nothing to test" exit now
  carries a `DROP_SKIP_REASONS` key, `tiers: []` is checked at the source instead of
  hiding inside `policy.apply`'s `skipped` flag alongside a dozen unrelated conditions,
  and a new **Not evaluated** report section lists the skipped payloads per tool with
  the reason — the same principle as `_unmeasured` (#352). `run_drop_fluency` and the
  new `drop_eval_coverage` share one `_probe_envelope`, so they cannot disagree about
  which payloads were evaluated.

### Changed

- A `_suggested_fields_note` on a passthrough rule now says that renaming the block
  alone enables nothing, and points at `terse stats` — a tool with real live traffic may
  already be compressing under the deployed policy (#274's cross-check, #375).

## [0.30.6] - 2026-09-03

### Fixed

- **`_unmeasured` saw dropeval's control arm but not its treatment arm, so the same
  transport loss withheld a run on one side and published a gap on the other** (`#352`).
  The gate discovers arms by scanning rows for `<arm>_trials` keys, and dropeval emits
  exactly one of them. Its treatment arm deliberately has none — errored trials must stay
  in the accuracy denominator, because scoring them as misses makes the drop rule look
  *worse*, which is the conservative direction — so the gate was structurally blind to it
  at every loss level. The only remaining cover was `inconclusive_models`' arm-blind
  50%-of-pooled-calls threshold, which means the treatment arm could lose 49% of its own
  calls and still have a final-accuracy gap published, while a control losing 21% was
  withheld. That arm runs two turns to the control's one, so it is the arm that fails
  first under a token-budget stop, and the questions it loses are not a random sample.

  `_unmeasured` gains a fourth trigger reading an explicit `<arm>_errors` counter, at the
  same `UNMEASURED_FAIL_SHARE` against that arm's own calls. Two smaller changes were
  rejected and the reasons are recorded in the code: emitting `fails` reaches only the
  pooled trigger, where a treatment-only loss would need 40% to fire against the control's
  20% — the pooled-denominator defect `#339` removed, one harness over; and emitting
  `answer_trials` would pull errored trials out of the recall denominator, the measured
  33%-FAIL-to-100%-PASS regression `dropeval.py` exists to prevent. On the control arm the
  new trigger is arithmetically identical to the existing one, so symmetry follows from the
  shape rather than from a second threshold.

  **Not closed, and now asserted rather than assumed:** recall and no-overfetch gate
  against a fixed 100% ideal and never reach `_unmeasured` at all, so a treatment error is
  still scored as a behavioural miss there. The run-level directive therefore remains
  asymmetric — BLOCK for a treatment loss, NOT_CONCLUDED for the same loss on the control.
  That is strictly conservative and never a gained ship authorization, and
  `test_the_run_level_verdict_is_still_asymmetric_and_that_is_recorded_not_fixed` pins the
  direction so it cannot drift.


- **Four evasion classes left open in the `paired_rows` disclosure guard, and two false
  claims it made about itself** (`#363`, following `#361`). No renderer emitted anything
  wrong — what was wrong is the test meant to keep it that way. Its load-bearing claim is
  that a `build_*` function reaching `paired_rows` but not `attrition_block` fails CI; each
  item below was a way to be that function and pass.

  **Graph keys are now qualified by enclosing scope** (`report.py:Outer.build_x`). `ast.walk`
  is breadth-first, so a nested def OVERWROTE a top-level namesake in the same module and
  destroyed the renderer's entry — a class method and a `try/except ImportError` pair both
  demonstrated it. This was already live: **16 of 585 definitions were silently missing from
  the graph**, none a `build_*` yet, which is one same-named helper away from wrong. All 585
  are present now, pinned by a test that counts them. Two defs at the *same* scope keep
  separate nodes and their shared name is marked ambiguous, because which one a caller
  reaches depends on which ran last — merging their edges would route a silent renderer
  through its compliant twin.

  Qualification breaks name resolution unless three sites change together, and the other two
  fail silently: `_reaches` matched `endswith(":" + name)`, and `_paired_and_silent` selected
  on `startswith("build_")` over the whole qualified tail. Both now match the last segment,
  each pinned by a synthetic renderer that reaches the pairing through a method and through
  a same-scope duplicate.

  **Bare decorators and argument defaults now draw edges.** `@paired_rows` is an `ast.Name`
  in `decorator_list` and `_pair=paired_rows` one in `args.defaults`; neither is ever inside
  a `Call`, so an edge-scan keyed on `Call` drew nothing for either — while the parenthesised
  `@deco()` form was caught. Keyword-only defaults are a separate list and are covered too.

  **The codec check covers both spellings and the function the comment already named.**
  `_payload_tokens` is a separate top-level function, so its body is not inside
  `inspect.getsource(run_codec_fluency)` — the comment beside the assertion named it as the
  vector while the check did not reach it. And `tags["terse_attempts"] = v`, the subscript
  form `codeceval.py` already uses two lines from the emitter, cannot match a colon-anchored
  pattern. The predicate is now testable against synthetic sources, since widening a guard
  nothing currently trips is otherwise unfalsifiable.

  **`codeceval`'s hand-built fixture is pinned to its emitter**, the counterpart `dropeval`
  already had. It immediately found the drift it exists to catch: the fixture was missing
  `transform`.

  **Two false self-claims removed.** `_call_graph`'s docstring said reverting the async and
  attribute branches "leaves every test green" — true when written, false by the end of the
  same commit, since the tests that falsify it were added alongside it (measured: **4**
  fail — async alone 1, attribute alone 3). And "the idiom in 11 modules here" was called
  unreproducible and removed; 11 is exactly what the natural rule gives (modules containing
  a bare `from . import X`), so the rule is now stated alongside the number.

  An adversarial review found the blind spot had **moved rather than closed**: `_defs`
  split what was one node into several, but resolving a call by bare name traversed the
  UNION of their edges, so a renderer `main` reported as silent became
  silent-and-unreported. Ambiguity is now counted over graph NODES rather than modules,
  which makes different-scope namesakes, same-scope duplicates and cross-module collisions
  one rule instead of three. The same review found the mirror image of the decorator fix —
  every bound name became a traversable disclosure edge, so `_unused=attrition_block`
  scored a silent renderer compliant — so a binding now counts for "drawn over the pairing"
  and not for "discloses", neither as the target nor onward. The attribute spellings
  (`@report.paired_rows`) are covered too; closing only the `ast.Name` half had left the
  same asymmetry one level up. And the drift guard did not guard the fixture it named: it
  re-declared its own copy of the dict, so deleting `transform` from the real fixture left
  every test green. Both now come from one builder.

  A 22-mutation sweep leaves one survivor, recorded in the module docstring as an
  equivalent mutant with the reason it cannot be killed.

## [0.30.5] - 2026-09-03

### Fixed

- **A scope flag that does not apply to the active scope is now refused, not silently
  ignored** (`#366`). `resolve_target` read `--file` only for `--scope project` and
  `--repo-path` only for `--scope local`, dropping either one elsewhere without a word. So
  `install-mcp kb --file /tmp/x.json` — at the **default** scope, which is `user` — named
  one file, created no `/tmp/x.json`, and rewrote `~/.claude.json` instead. The behaviour
  was documented (`--file` was helped as "--scope project: …"), which is what made it
  dangerous rather than merely surprising: a flag that accepts a path, ignores it, and
  writes somewhere else reads as correct right up until it isn't.

  Not theoretical. It fired while developing `#277`'s own tests, rewriting a live router's
  `command` to a pytest temp binary that pytest then deleted — killing that router and
  every peer behind it until it was noticed and repaired by hand. The symptom is a server
  with no tools, days later: precisely the failure class `#277` exists to disclose.

  Both flags now raise at `resolve_target`, naming the scope they belong to **and the
  concrete consequence of ignoring them**, which is the half that conveys the danger:

  ```
  install-mcp: --file applies only to --scope project, but this is --scope user.
  Ignoring it would have written /home/u/.claude.json instead of the file --file names —
  pass --scope project, or drop --file.
  ```

  The consequence is spelled per flag, because the two do not name the same kind of thing:
  `--file` names a write **target**, while `--repo-path` names a **key inside** one — user
  and local scope write the same physical file, differing only in whether the entry lands
  at the top level or under `projects.<key>`. A single shared template got that wrong in
  half the four mismatch pairs, printing a literal `?` at project scope and, at user scope,
  asserting a difference between two identical paths.

  `install-mcp` and `uninstall-mcp` both exit 2. An empty value counts as passed —
  `--file "$CFG"` with an unset `$CFG` is the shell foot-gun that produces this — so the
  check is `is not None`, not truthiness. `mcp-status` is unaffected: it has no `--scope`
  and scans every scope, calling `resolve_target` once per scope with only that scope's
  own flag, which is why the check lives where scope and flag are both known rather than
  in argument parsing. A 9-mutation sweep leaves no survivors, including one per message
  branch — the placeholder arm had survived the first sweep because only the
  `--file`/user pair was ever exercised.

## [0.30.4] - 2026-09-03

### Fixed

- **`install-mcp --print` did not disclose that it was CHANGING an existing command**
  (`#277`, the second ask of `#275`). The dry-run rendered a `before:`/`after:` pair per
  server, but three shapes of change were invisible in it. An **already-folded peer**
  showed `before: (absent)` — its entry legitimately no longer exists standalone, it lives
  in the peers file — so its prior command was not shown at all, and `(absent)` is also
  what a genuinely new server renders, leaving the two indistinguishable. The peers file
  is now read as that peer's before-state, labelled `(from peers file)` so the provenance
  is stated rather than implied; a malformed record still renders `(absent)` **without**
  that label, rather than claiming a provenance for a value never recovered. (The
  launcher rewrite itself is disclosed in the router block, which is where a folded peer's
  launcher actually lives.) The **router's own
  command** had no `before:` at all (`router: <name> -> <command>` printed the new value
  only), and the router is the single entry every folded peer is reached through, so a
  rewrite there breaks the whole fleet at once; it gets a `before:`/`after:` pair like any
  other change. And `_short_cmd` **truncated at 100 chars with no marker**, so a long
  policy path pushed the launcher off the end and a reader could not tell a short command
  from a cut one — the cut is now marked, and the field that changed is stated outright:

  ```
  command CHANGED — the client spawns a different binary than before:
    from: /home/u/.local/bin/terse
    to:   /home/u/.pyenv/shims/terse
  ```

  printed from the raw field rather than the truncated render, and only when the value
  actually differs. **`args` is on that list too**, against the issue's own framing that
  `command` is "the one field whose change is both invisible and fatal": a moved
  `--policy` path exits 2 at spawn with the identical symptom, and is *more* likely to be
  hidden, since the launcher sits at the head of the rendered line and the policy path
  sits past the 100-char cut. A before/after pair can be byte-identical on screen while
  the policy moved. This matters because the failure is silent by construction: the MCP
  client cannot spawn a bad entry, so the server appears with no tools and nothing says
  why, days later. The distance between the config change and the symptom is the whole
  reason `#275` was hard to diagnose.

  Pinned by `tests/test_install_print_discloses_changes.py`, whose fixtures run at
  **project** scope under a redirected `HOME`, with an autouse canary asserting the real
  `~/.claude.json` did not move. An earlier draft did not, and a non-dry-run setup call
  rewrote the developer's live router to a pytest temp binary — the exact failure this
  issue exists to disclose. That hazard is filed separately as `#366`. An 18-mutation
  sweep leaves no survivors, and `--print` no longer aborts on a hand-edited entry whose
  `command` or `args` hold a non-string — a `TypeError` out of the renderer that killed
  the whole disclosure on exactly the malformed state where it is most needed.

## [0.30.3] - 2026-09-03

### Fixed

- **`fetch_corpus.sh` overwrote the committed snapshot in place, so a measurement taken
  after a re-fetch was unreproducible** (`#341`). The script's own header states the
  invariant — "the committed corpus/ snapshot is what the published numbers were measured
  on" — and then broke it silently: no cleanliness check, and a closing `wc -c` that
  reported byte sizes without ever saying whether anything had MOVED relative to `HEAD`.
  Eight of the nine payloads come from live endpoints (`pulls`, `issues`, `commits`,
  `actions/runs`, `labels`, `contents`, the repo object, `rate_limit`) that return
  different content week to week, and a run left the working tree holding un-committed API
  output that `terse measure --corpus`, `benchmark.py` and
  `tests/test_published_benchmarks.py` would all happily measure — the last of those
  compares the docs against a live re-measurement, so on a dirty tree it measured content
  that never entered git. That is not hypothetical: `#293` was investigated off figures —
  `#249`'s — recording 491 tokens for `gh_rate_limit.json`, a file byte-identical since
  `267af9e` (2026-07-17) that measures 357 today. Identical bytes cannot produce different
  counts. A dirty tree explains the **eight** payloads this script writes; it does not
  explain `gh_commits_flat.json`, which moved in the same table and has no producer in the
  repo at all. Something else is also in play there — this closes the half that is ours.

  The script now refuses to run when `corpus/` is dirty relative to `HEAD` — untracked
  payloads included, since anything sitting in that directory is measured by everything
  that globs it — and names what is dirty. `--force` overrides and says that it did. The
  check runs **before** the first API call, so a refusal is a guard rather than a report.
  After a fetch the run states its own outcome: `identical to HEAD` when nothing moved, or
  the changed files (porcelain status **and** diffstat, because `--stat` cannot see an
  untracked file) plus the reason it matters. Neither guard changes what is fetched.

  A `gh` that dies partway — expired token, 403 rate limit — used to abort under `set -e`
  before any report, leaving some payloads on today's API content, one truncated to zero
  bytes by its own redirect, and the rest on the committed content: a tree in a state that
  never existed at any point in time, and the ONE path where the script said nothing. The
  report now runs from an `EXIT` trap, so the failure path is the loudest rather than the
  quietest. The refusal's remedy also prints both `restore --source=HEAD --staged
  --worktree` and `clean -fd`, because `restore` alone fixes neither a staged change nor
  an untracked payload and exits 0 regardless — the guard used to repeat itself verbatim
  with `--force` as the only way out.

  Pinned by `tests/test_fetch_corpus_guard.py` — the first test in this repo to execute a
  shell script, running it against a throwaway git checkout with stubbed `gh`/`jq`, so
  nothing reaches the network. One test runs the printed remedy verbatim and requires the
  next invocation to proceed. A 19-mutation sweep over the script leaves no survivors.

## [0.30.2] - 2026-09-03

### Fixed

- **The `regressions` and `recovered` columns divided by a denominator none of the
  accuracies beside them use** (`#353`). Both compared `<arm>_ok` to the row's shared
  `trials`, which for a `score_pack` row is `max(...)` across forms (`fluency/pack.py`) —
  the documented `#91` uneven-collection mode, where a hand-built pack may carry 3 raw
  replies and 2 terse ones for the same question. An arm that answered every reply it was
  actually given then read as incomplete: 24 questions, every arm full, printed **24
  regressions out of 24 beside a 100% terse accuracy**. `regressions` is the column a
  reader scans to decide whether the compressed form costs comprehension, so the row
  argued against itself in the one place it is read. Each arm is now counted against its
  own `<arm>_trials`, the denominator `_form_stats` and (since `#283`) `paired_rows`
  already use, falling back to the shared count so result files predating the per-arm
  counters read as they did. An arm with **zero** collected trials is explicitly not
  complete: `0 == 0` would report every question as a regression against a control that
  never ran, which `fluency/scoring.py` produces for a responses file missing a form
  entirely. The issue named the fluency table; the diff report carried a **second
  hardcoded copy** of the same arithmetic (`terse_ok`/`diff_ok`) with the identical
  defect, and it is fixed too — two copies of one rule is the shape that produced `#299`.
  The suffix-swap derivation, which `_trials_keys` already claimed to be "the one place
  that mapping is written" while `_form_stats` and `arm_measured` both spelled it inline,
  now lives only in `_trials_key`, and an AST guard fails if a second copy appears in any
  spelling — a substring count, the first attempt, both accused docstrings that quote the
  derivation and missed `f.replace("_ok", "_trials")`.

  Scope of "reads as before": exact for files predating `#263` (no per-arm counters at
  all). A file written **between** `#263` and `#283` — per-arm counters but no `attempts`
  key — does change, because `_paired_partition` keeps every such row unconditionally; an
  arm that lost a call there previously read as a regression and now does not. That is the
  corrected reading, not a regression: a call that never happened is not a wrong answer.

## [0.30.1] - 2026-09-02

### Fixed

- **The `#299` disclosure guard was evadable by two idioms this package already uses, and
  its own fix enumerated renderers by hand** (`#361`). No renderer emitted anything wrong —
  what was wrong is the test meant to keep it that way, and three claims made about it.
  `_call_graph` matched only `ast.FunctionDef` and only `ast.Name` callees, so a renderer
  written as `from . import report` + `report.arm_gap(...)` (the idiom in 11 modules here,
  including `cli.py`'s own `dropeval.run_drop_fluency(...)`) or as an `async def` was
  invisible to it: both pass the entire suite as silent paired renderers, and both are now
  caught. What it still cannot see is recorded rather than implied away — a renderer not
  named `build_*`, and one defined outside `src/terse`. The diff-SOAK's `is_diff_run` guard
  was pinned by nothing: dropping it survived all 1852 tests and is not an equivalent
  mutant — with a fluency-shaped model in the same result set it renders a selection-bias
  clause about a `diff_ok`/`terse_ok` pairing that was never performed. That test had
  listed three renderers by hand and missed the fourth, which is the failure mode `#360`'s
  entry claims to have ended, recurring inside the fix for it. `_CANNOT_EXCLUDE`'s premise
  was guarded in one direction only: `_arm_attempts` prefers `<arm>_attempts` over
  `trials`, so `codeceval` emitting that key would make exclusion possible with the
  `_trials` lines untouched and leave `build_codec_verdict_report` a genuinely silent
  paired renderer still sitting in the exemption. Finally `deref 15/15` was quoted as fact
  in a comment, a test docstring and this changelog while nothing in the tree produced it;
  it is true (measured `excluded 15/75 — by arm: diff_ok 15; by kind: deref 15/15,
  count 0/60`) and is now produced by a fixture instead of remembered.

  The hardening itself is pinned on a **synthetic package**, via `_call_graph(root=)`.
  It cannot be pinned on `src/terse`: there is not one `async def` in the package and no
  renderer that pairs through an attribute call, so reverting both branches leaves all
  1853 tests green — the fix was correct and completely inert against the live tree, and
  the only thing that had exercised it was a throwaway probe that writes a module into
  `src/` and deletes it. A probe that runs nowhere is not a test. Keying the graph on the
  relative path turned out to be necessary and NOT sufficient: the check still collapsed
  to bare function names, so a silent `build_diff_report` in `src/terse/fluency/report.py`
  was scored compliant by `report.py`'s disclosing one of the same name. It now works in
  qualified keys, and the check lives in one helper both the live and synthetic tests
  drive rather than a copy each. **What the guard still cannot see is now written down**
  rather than implied away — a symbol alias (`from .report import paired_rows as _pr`,
  which `lossy.py` already does for another name; a MODULE alias is caught, which is the
  surprising asymmetry), `build_x = _impl`, a module-level lambda, `functools.partial`,
  `getattr` dispatch, a renderer outside `src/terse`, and one not named `build_*`. Each
  was demonstrated evading the full suite. Closing the first three needs a name-resolving
  import graph, which is a bigger tool than this test; the honest claim is that it catches
  the mistakes people actually make here, not that it is airtight.

## [0.30.0] - 2026-09-02

### Added

- **The fluency and dropeval reports publish the attrition of their paired exam, per arm and
  per question kind** (`#299`). `paired_rows` (`#280`) drops a question from every gated arm
  unless all of them completed all of its trials. That makes the surviving arms comparable; it
  does not make the SELECTION ignorable, because which questions survive is decided by the arm
  most likely to fail — and that arm is not random. dropeval's treatment runs a two-turn
  retrieve protocol against the control's one turn, and fluency's longest prompts are also its
  hardest questions, so under a token-budget stop the arm under test thins out first and on
  exactly the questions that discriminate. A run that loses its five hardest questions from one
  arm and none from the other still produces a perfectly paired comparison over the twenty easy
  ones, reports a tiny gap with a tight interval, and is wrong. It was excluded **silently**:
  the reports printed a pooled `errors`/`attempts` and an INCONCLUSIVE gate, neither of them
  per-arm and neither per-question-kind. dropeval already reported the arm split and the
  surviving-question count (`#300`); what NO report had, on either harness, is the
  per-question-KIND axis — whether the removed questions were the hard ones — and the fluency
  report had none of the three. New `report.attrition` reports all three, and it
  reads the excluded set out of `paired_rows`' own body (`_paired_partition`) rather than
  re-deriving the rule — a reporter that re-implemented the escapes for an absent `attempts`,
  an absent `<arm>_trials`, and the run-level `collected` fact would be free to contradict the
  pairing it annotates. Kinds print worst-share-first and carry their denominator, because
  `deref 5/5` beside `count 0/25` is the concentration signal and a bare `deref 5` is not.
  `inline_ok` is deliberately excluded: it is display-only and outside the pairing by design,
  so counting its losses would report an exclusion that never happened (that arm's lack of
  protection is `#292`). This is Option 1 of the issue and nothing more — the bias is made
  visible, not corrected, because no run in the repo can currently say whether it is large or
  negligible, and a threshold picked before looking at one would be invented rather than
  measured. **dropeval's arms are asymmetric by design and its note says so**: the treatment
  carries no `answer_trials`, because removing a failed treatment call from its own
  denominator turned a 33% recall FAIL into a 100% PASS at an 11% error rate (`#300`), so a
  failed treatment call is scored a MISS and `_paired_arm` is unconditionally true for
  `answer_ok` — the pairing can only ever exclude on the CONTROL side. Both consequences are
  executed, and both were review findings against the first cut of this change: a run whose
  two-turn treatment lost every call while the one-turn control lost none reported
  `excluded 0` and rendered nothing at all (the motivating case, reading as "the exam was not
  selected"), and a run where both arms lost everything attributed 100% of the exclusion to
  the control while the generic note invited that to be read as bias in the control's favour.
  The dropeval line now carries the treatment's own loss from `treatment_errors` — labelled
  as misses, not exclusions, because those questions are IN the paired exam depressing the
  treatment's accuracy rather than removed from it — and its note states the asymmetry
  instead of the generic reading rule.

  **Every renderer drawn over the paired subset carries it**, through one
  `attrition_block`: the fluency, dropeval and diff-family markdown reports, all three
  terminal forest-plot charts, and the HTML page. A chart drawn over the paired subset that
  does not say what left it is the same silent exclusion the markdown stopped printing — and
  the chart is the artifact people quote. The diff family is why that sentence had to be
  taken literally: `cli` prints the diff markdown and its terminal chart on EVERY diff path
  and writes the HTML page only under `--html`, so wiring the page alone would have put the
  disclosure exactly where it is least read — an operator running `--diff --bars` would see
  `PASS` over 20 questions and never learn the diff arm removed all five `deref` ones. The
  diff renderers annotate their OWN pairing (`diff_ok` vs `terse_ok`, the arms their
  `arm_gap` is given), not fluency's four — including the diff-SOAK markdown, which does
  not route through the shared diff body and is the artifact `--out` keeps, so leaving it
  silent had its own chart disclosing `deref 15/15` beside a file that said nothing. One
  helper rather than eight copies because
  `#335`'s review rounds are the record of what happens otherwise: two renderers deciding the
  same thing separately disagreed three times. The heading is a single constant with a
  markdown and a plain spelling — the markdown sites first bolted the bold on afterwards
  with a `.replace()` keyed on the heading text, a second copy of the literal that silently
  no-ops (emitting an unclosed `**` into both reports) the moment the heading is edited, and
  that survived 226 tests. The helper spells the block three ways — markdown, HTML and
  plain — because block markup and INLINE markup are separate questions: folding them into
  one flag cost the HTML page its `<code>` spans, leaving one paragraph in plain text beside
  a verdict banner two lines up that still rendered them.

  **The enumeration is computed, not written down.** "Every renderer carries it" was
  claimed three times and was false three times — first the HTML page while `cli` printed
  two other renderers, then the diff family, then the diff soak — because each fix listed
  the renderers by hand and the next one was missed the same way.
  `test_every_renderer_drawn_over_a_paired_subset_discloses_its_attrition` walks the call
  graph of the whole package instead: any `build_*` function that transitively
  reaches `paired_rows` must transitively reach `attrition_block`. A renderer added later
  inherits the requirement, and adding one without a disclosure fails CI rather than waiting
  for a reviewer. It found a tenth renderer nobody had named — `build_codec_verdict_report`
  — which is exempt because `codeceval` pins `raw_trials`/`terse_trials` to `trials` on
  purpose, so its pairing can never exclude anything; that exemption's premise is proven by
  execution, so it goes red if `codeceval` ever starts emitting real per-arm denominators.
  `DIFF_ARMS` is threaded through every diff-family `arm_gap`/`paired_rows` call, not only
  the `attrition` ones — for one round the constant was read by four sites while seven
  others still spelled the pair literally, so its own comment claimed a sharing that did
  not exist. A dropeval run started with `--no-control` gets its own note: the standard one
  explains why exclusion can only land on the control side and points at a **Where they
  failed** split, both meaningless without a control arm and the second a markdown-only
  section absent from the two-line terminal chart.

## [0.29.1] - 2026-09-02

### Fixed

- **An envelope's `shape` is re-classified at read time, so two tables in one report cannot
  disagree** (`#355`). `shape` is a pure function of `raw`, but it was stored at capture time
  and read back as ground truth — so `7be9d41` (`#208`/`#204`), which relaxed
  `_find_record_list` from an identical-keyset rule to the codec's union-schema
  `is_tabularizable`, silently left every pre-existing envelope in its old bucket. Measured on
  the live 1524-envelope corpus: `terse measure`'s **Coverage** table said 56 `array-of-records`
  / 77 `compact-json` while its **Tier-0 savings by shape bucket** table, a few lines below in
  the same report, said 92 / 41 — 36 payloads apart, all drifting one way, all qualifying only
  under the union-schema rule. New `capture.envelope_shape` classifies from `raw` and is the
  one accessor every consumer reads the bucket through (`capture.coverage`,
  `codeceval.run_codec_fluency`'s per-`(tool, shape)` verdict key, `text_alias_ceiling`). The
  read is the whole mechanism: `load_corpus` deliberately leaves the stored field alone, so it
  stays readable as evidence that a corpus predates a classifier change. The stored value is
  now a cache, used only when there is no `raw` to classify. Both tables now read 92 / 41.
  Cost is one `classify_shape` — a `json.loads` plus a record-list walk, not just the parse —
  at each read: for `terse measure` that is one added pass over the corpus, since `coverage`
  used to read the stored field and `measure_payload` already classified live. Measured 0.021s
  for a pass over those 1524 payloads. **Correction to `#358`'s own description:** it claimed
  the codec verdict was the one consumer the drift could not reach in practice, on the ground
  that a non-uniform payload yields no `deref` question. That is wrong — `gen_questions` picks
  its `blobcol` out of the key INTERSECTION, so a drifted payload can and does produce one.
  Two of the 36, both `kb.read.list_nodes`, were being filed under `compact-json` for a payload
  the codec tabularizes. The fix was worth more than the pull request said it was.

## [0.29.0] - 2026-09-02

### Added

- **Codec-tier token savings are reported beside the verdict, never inside it** (`#303`).
  `build_codec_verdict_report` gains a sibling `## Savings by tool and shape` section over
  the same `(tool, shape)` groups as the SAFE/UNSAFE/UNRESOLVED table, so the two read
  against each other with no combined score anywhere — the ordering is the argument, since
  a cell that folds a saving into a verdict is how a savings number ends up licensing a
  correctness loss (`#295` DoD 4). An UNSAFE group still prints what it saves; suppressing
  that would be its own editorialising. `run_codec_fluency` now stamps each payload's
  `raw_tokens`/`terse_tokens` (cl100k, over exactly the two forms the model was fed) onto
  its rows, and the renderer de-duplicates by `sha` before summing. A payload with no token
  counts — no tokenizer at run time, or a result file predating this — is excluded from the
  sums and reported as uncounted, never as a zero saving.

## [0.28.11] - 2026-09-01

### Fixed

- **A non-answer was scored as a wrong answer** (`#279`, `#283`). `fluency`'s scorer divided
  by `len(replies)`, so a reply the backend never produced — `None`, a non-string, or a
  blank — landed in the denominator as a miss, and a model that answered correctly every
  time it answered at all reported degraded accuracy. `_score_form` now leaves a non-answer
  out of both the numerator and the denominator (the discipline `harnesses._ask_n` already
  followed), and a new `MISSING` sentinel separates "this form was never collected" from
  "the reply came back empty". Emitted rows carry per-arm `<arm>_attempts` plus pooled
  `fails`/`attempts`, so `--responses` runs are gated by the same transport check as every
  live harness instead of publishing a verdict off whatever survived (`#283`).

## [0.28.10] - 2026-09-01

### Fixed

- **`_unmeasured` divided transport loss across the pooled sample, not per arm** (`#339`).
  The 20% call-loss gate's comment claimed it withheld a verdict when an arm lost more than
  a fifth of its calls; the arithmetic divided by BOTH arms' attempts, so the real threshold
  was 40% of one arm — a factor of the arm count away from the number the comment argued
  for, and it had already been mis-cited in that comment. `_unmeasured` now reads each arm's
  own `<arm>_attempts` (falling back to the shared `trials` where a row does not carry
  them, so every legacy result file renders byte-identically), and `paired_rows` gates on
  the same per-arm counts.

## [0.28.9] - 2026-08-27

### Fixed

- **`build_diff_soak_report` hardcoded a transport cause the reason didn't license** (`#338`).
  It was the seventh renderer with its own hardcoded phrasing: two sites asserted "too many
  calls went unanswered" as settled fact for the `"unmeasured"` reason, which since `#332`
  also covers a backend that answered every call but simply couldn't pair. Both sites now
  route through the same hedge every other renderer already uses. Adds
  `test_every_renderer_names_the_right_exclusion_reason`, the test `REASON_LABEL`'s note in
  `report.py` cited but which did not exist.

## [0.28.8] - 2026-08-27

### Fixed

- **A fixed-ideal metric can no longer print PASS off a handful of questions** (`#335`).
  Recall and no-overfetch gate against a fixed 100% ideal, so they never pair — which meant
  `_MIN_PAIRED_QUESTIONS`, whose whole job is to stop a thin sample buying a PASS, counts
  PAIRED questions and so had never applied to either of them. A one-question recall run
  printed `100% ±0 pts **PASS**` and `safe to enable drop-to-retrieve`: maximum confidence
  off a single question, with the `±0` not a rounding artifact but the exact SE of a sample
  that is all-right or all-wrong.
  `_FIXED_IDEAL_MIN_QUESTIONS = 5` is a **disclosure threshold, not a statistical floor**,
  and the distinction is deliberate: the metric has never run on real data (zero of 1,524
  captured payloads have a drop rule selected), so there is no distribution to calibrate a
  Clopper-Pearson bound against, and inventing one would be a fabricated justification
  rather than an admitted convention. Nothing is withheld — the percentage and its question
  count are both published, and the model stays in the chart. What a thin sample cannot buy
  is the word PASS. A FAIL still publishes at any `n`, because an exclusion must never be
  able to improve a verdict.

## [0.28.7] - 2026-08-27

### Fixed

- **Withholding a model from a gate can no longer authorize a ship** (`#342`, `#344`).
  Excluding a model removed it from `_worst_case_gap` entirely, so the verdict computed
  over what remained came back cleaner than the verdict over everything. Stripping one
  model's no-drop control arm — strictly *less* evidence, no other change — turned
  `keep drop-to-retrieve off` into `safe to enable drop-to-retrieve`, with the failing
  model named nowhere. Two more routes around the same gate are closed with it: a model
  present in `results` with no rows was skipped rather than withheld (22 of 144
  non-shipping two-model fleets started shipping when one model's rows were emptied), and
  a model with no rows of a `kind` was scored at a fabricated `0%` against the fixed 100%
  ideal, publishing a `-100%` **FAIL** and `keep drop-to-retrieve off` beside a `recall q`
  column of literally `0`.
  `terse tune --drop-eval` printed `If the worst-case model PASSES, enable the verified
  fields` under the report — a rule the reader applied by eye to lines that report the
  worst *scored* model, so a fleet with one model withheld showed three `**PASS**`
  headlines above an instruction to enable what the verdict had just declined. It now
  reads the directive.

### Changed

- **The dropeval verdict is computed, not branched** (`#342`). Four review rounds on one
  ~200-line change to this path produced 7, 6, 5 then 9 findings and never converged; every
  round found a defect inside the previous round's fix. `ArmGap.excluded` is now a closed
  `Literal` rather than `str | None`, so a new reason with an unhandled consumer is a mypy
  error at the consumer; the directive is `max()` over a `SHIP < INSUFFICIENT <
  NOT_CONCLUDED < BLOCK` lattice, so branch-precedence bugs are unconstructible; and
  `dropeval_verdict()` decides once for both renderers, so the markdown and the terminal
  chart cannot reach different conclusions — a docstring promise they had broken three
  times. `tests/test_dropeval_monotonicity.py` sweeps the metamorphic invariant over the
  full cross product in seconds, and reproduces 20 violating input pairs against the code
  it replaces.

## [0.28.6] - 2026-08-25

### Fixed

- **A PASS now requires at least 20 paired questions; a FAIL still publishes at any
  number** (`#334`). `_gap`'s previous gate fired only when *nothing* survived pairing, so
  a run that kept a single question still printed a confident verdict: at 15% call loss,
  nine of ten questions voided, the report rendered `✓ PASS — diff-form 100% vs full-terse
  100% (gap +0% ±0pt)` off that one question, and nothing in the diff-family output
  disclosed the other nine. The `±0` is not a rounding artifact — at n=1 the SE is exactly
  zero whenever the single question is all-right or all-wrong, which is the normal case at
  temperature 0.
  The floor is **asymmetric**, and that is what makes it safe: it withholds only a gap that
  would have PASSED (`best >= control - tolerance`), so what it removes from the worst case
  is never the failing model. An exclusion therefore cannot improve a run's verdict — the
  defect that sank a symmetric survival floor in `#332`. The cut is the tolerance line and
  not exact equality: drawn at equality, the whole `[-5%, 0)` band escaped both gates at
  once — behind its control so the floor let it through, inside tolerance so the verdict
  passed it — and one paired question at 19/20 vs 20/20 still printed
  `gap -5% ±10 pts **PASS** … safe to enable proxy --diff`.
  20 comes from `_CODEC_MIN_TRIALS`'s existing Clopper-Pearson framing rather than a new
  judgment call: n zero-regression questions bound the true rate below `1 - 0.05 ** (1/n)`,
  which is ~14pp at 20 and ~63pp at 3. Counted in QUESTIONS, not trials, because `#297`
  established that trials within a question are correlated.
  **Operationally this needs a corpus of at least five captured results**: `gen_questions`
  yields a fixed 4 questions per payload regardless of its size, so a single-tool corpus
  produces 4 paired questions and will now report `Not concluded` rather than a verdict.
  `codec_verdict` opts out — it gates its own sample size in trials, the unit its verdict
  actually counts in, and layering a second floor would silently re-calibrate that tier.
  That opt-out is keyword-only and AST-pinned to a one-name allowlist.
  **The soak's per-depth slices inherit this floor**, and `--soak-windows` (default 6) caps
  a slice at roughly 24 questions, so a deepest depth with few available windows will report
  `NO VERDICT at the deepest tested depth` rather than a passing one. Raise `--soak-windows`
  to restore it. A depth slice that shows real drift still publishes at any question count.
- **`passes_tolerance` is now the single definition of "within tolerance".** The floor and
  the verdicts each spelled the comparison out, as `facc >= cacc - tol` and
  `gap >= -tol - 1e-9`, and binary float makes those disagree on 122 exact-boundary
  accuracy pairs — `0.40 - 0.05` is `0.35000000000000003`. Those pairs landed in neither
  set: not withheld (the floor read the arm as behind its control) and not failed (the
  verdict read the gap as inside tolerance). Measured on `35% vs 40%` over 10 paired
  questions — a green "safe to enable `proxy --diff`" with `±0 pts`, the exact symptom
  `#334` was filed on, surviving inside its own fix. Five sites across the three renderers
  now share one function.
- **The soak's deepest-depth verdict no longer depends on the pooled one.** It was nested
  inside the pooled `if worst:`, so a withheld pooled gap skipped the depth analysis
  entirely — and `#334` made that reachable for a run that lost zero calls. Measured: a
  fully-paired −100% collapse at the deepest depth disappeared from the verdict, under a
  line promising "depth slices that pair cleanly are still scored below". An exclusion must
  never remove a demonstrated regression from a verdict.
- **An underpowered model's rows still pool into the per-transform table.** Dropping them
  moved that table's `table` row from 72% to 90% — an exclusion improving the figure the
  verdict tells the reader to use when restricting policy by transform. Only the per-model
  conclusion is unsupported; the rows themselves are fully paired.
- **Withheld models are no longer all described as transport failures.** `"underpowered"`
  is a distinct exclusion reason with its own label and heading, because nothing failed in
  that case — reusing `"unmeasured"` would have printed "too few calls to compare" about a
  run that lost no calls at all. `_not_measured_lines` now groups by reason and reports
  paired-question counts for the new one instead of a "calls lost" figure that would read
  `0/N`.

## [0.28.5] - 2026-08-25

### Fixed

- **`install-mcp`/`uninstall-mcp` write recovery data before the destructive write, on
  both the single-server install and uninstall paths** (`#329`). A crash between the two
  writes (SIGKILL, OOM, disk full) previously left a wrapped config with no matching
  stash entry after install — `wrap()`'s fallback branch then silently treated the
  already-wrapped entry as the original on the next run, nesting a double wrap
  (`terse proxy ... -- terse proxy ... -- <server>`). Install now writes stash before
  config, matching `_install_multiproxy`'s existing order. Uninstall writes the OPPOSITE
  order (config before stash): `unwrap()` moves the original from stash into config in
  memory before either write lands, so config is the recovery write and stash is the
  destructive one there — copying install's order onto uninstall (an intermediate
  version of this fix) would have erased the on-disk stash record while the config was
  still wrapped, a worse crash window than either original order.
- **`load_policy` validates the `fields` rule key's shape**; a malformed spec (a bare
  string instead of `{"lossy": ...}`) now fails to load with a clear error instead of
  silently never firing, or crashing `Rule.lossy_fields()` later on a never-lossy/
  credential server (`#328`). **If your policy file has a `fields` entry shaped like
  `{"path": "drop-to-retrieve"}` instead of `{"path": {"lossy": "drop-to-retrieve"}}`,
  the proxy will now refuse to start until it's corrected** — this closes a case that
  previously ran with the field silently uncompressed.
- **A drop-to-retrieve recall question is no longer generated when the dropped value is
  also visible elsewhere in the compressed payload** (`#327`), which made the question
  answerable without a retrieve call and inflated recall-accuracy scores. Applies to
  both scalar and dict/list dropped values (the needle now uses the same compact
  separators the wire format itself uses).
- **`terse fluency --diff --html` no longer checks retired `"unpaired"`/`"exam too
  small"` exclusion reasons** that `report.py`'s `_gap` never actually produces — the
  distinction was designed (see `_gap`'s docstring) but never implemented, so the more
  specific verdict/tail text could never render, and every exclusion reason got a
  generic tail that was sometimes factually wrong (e.g. claiming "no model returned
  enough calls" for a model excluded because its rows had no diff arm at all) (`#330`).
- **`dict_encode` no longer overestimates a repeated value's alias savings when its
  occurrences are swallowed by a bigger repeated-subtree alias chosen in the same pass**
  (`#326`) — applies to both a repeated string and a repeated subtree nested inside
  another aliased subtree. Compression-quality only (round-trip losslessness was never
  at risk); leaves a small amount of achievable compression on the table in the affected
  cases, now recovered.
- **A verdict is withheld when NO question survived pairing, instead of being published as
  a confident zero** (`#332`). `_gap` — the shared gate behind `arm_gap` and `best_arm_gap`,
  and so behind the diff, fluency, dropeval and soak reports in markdown, terminal and HTML
  alike — gated on transport failure but never on whether anything was left to compare
  afterwards. Because `paired_rows` voids a whole question when either arm loses one trial
  of it, loss is amplified from the call level to the question level: at three trials an
  arm, one lost call per question is 16.7% of the calls (under `UNMEASURED_FAIL_SHARE`) and
  100% of the questions. Nothing survived, `_form_stats` scored both arms a flat 0.0 with an
  SE of 0.0, and the HTML banner rendered a green **✓ PASS** reading `+0% ±0pt` — maximum
  confidence from no evidence. Such a run is now withheld as `unmeasured`.
  **The gate is deliberately narrow: it fires only on an EMPTY paired subset, never on a
  small one.** The rows that survive pairing are the strongest evidence a run produces
  (every arm completed every trial of them), and because an exclusion drops a model from the
  gate entirely, withholding a small subset can *improve* a run's verdict — an earlier
  version of this fix used a 50%-survival floor and was measured turning a demonstrated
  −100% regression into "safe to enable `proxy --diff`" at 10% call loss.
  **This therefore does not cover a small-but-nonempty subset**: a run where one question
  of ten survives still publishes a `±0` PASS off that single question. Tracked as `#334`.
- **The reports no longer tell you your backend was unreachable when it answered almost
  every call** (`#332`). The `unmeasured` exclusion now covers two causes — a dead backend,
  and a live one whose losses left no question complete on both arms — so the prose that
  named only the first was false for the second, and pointed at the wrong remedy. The shared
  `REASON_LABEL` is now "too few calls to compare"; the "**Not measured**" paragraph reads
  from that label (it was the one exclusion site that did not, and had drifted from the
  other renderers), states both causes, and suggests lowering `--trials` — each extra trial
  is another chance for a question to lose one and be dropped from both arms. Also fixes a
  by-depth soak lead that branched on `why == "x"`, a reason string nothing produces, so the
  specific wording it guarded had been unreachable since `#284`.

## [0.28.4] - 2026-08-21

### Fixed

- **`terse fluency`'s reported `±` no longer collapses to a false `±0` at realistic trial
  counts** (`#297`). The old pooled-binomial SE measured within-question consistency, not
  question-sampling variance: at temperature 0 a question is nearly always all-right or
  all-wrong across its own trials, so the SE collapsed to ≈0 at ANY accuracy regardless of
  how much the *questions* disagreed with each other run to run — two runs at identical
  accuracy could report `±0%` and `±3%`, with the more volatile one printing as more
  certain. Replaced with a cluster-robust (sandwich) SE that clusters on the question:
  more questions tighten it without limit, while more trials tighten it only down to a
  between-question floor they can't cross. Every report's `±` column, the `--trials` help
  text, and a stale "raise `--trials` to tighten" remediation line are updated to match.

## [0.28.3] - 2026-08-20

### Fixed

- **`terse tune` cross-checks passthrough tools against the live ledger before
  recommending they stay uncompressed** (`#274`). `tune`'s corpus is idempotent by sha
  (holds each payload's first sighting, not every call) and capped at 200 samples/tool —
  structurally blind to call frequency. Measured case: `kb.read.list_principles` scored
  4.5% (passthrough) from the corpus while the live ledger measured 15.1% over 881 blocks
  and 2.19M raw tokens; applying `tune`'s own `--out` policy would have silently disabled
  compression on the fleet's single largest source of savings. Added `--ledger` to `tune`,
  reusing the same `_resolve_ledger`/`_ledger_traffic` cross-check `policy autotune`
  already applies to its downgrade warning — any passthrough row whose ledger traffic
  clears `--threshold` is flagged with block counts and raw tokens before the policy
  prints or writes. Advisory only: `--out` still writes, the operator decides.

## [0.28.2] - 2026-08-20

### Fixed

- **A primer the proxy declines to send is now recorded as costing zero, instead of being
  billed in full** (`#286`, `#317`). `terse stats` sized every wrapped server's primer from
  its installed policy and billed it to anyone the ledger showed was *called*. A server
  whose results always carry `structuredContent` never reaches the lazy attach — the client
  would discard the text block unread — so it pays **nothing, forever**, and was billed
  anyway. `searxng-mcp` was charged 312 tok/session for a primer it cannot send, which made
  a zero-cost wrap look marginal.

  The proxy now writes a ledger row at the moment it *declines* to attach, carrying
  `attached: false` and zero tokens. A server with such a row and no attach reports
  `primer_tokens: 0` with `primer_source: "recorded"`, lands in the `free` list, and is
  labelled `once/session (unpaid)` rather than "pays once per session".

  **Absence of a row still means nothing**, deliberately. The primer decision happens once,
  at a session's first compressible result, while result rows accrue for hours afterwards —
  so any `--since` window or ledger rotation starting mid-session drops it and keeps the
  rest. A window with no primer row therefore falls back to the labelled policy estimate. An
  earlier attempt inferred non-payment from that absence and produced two different
  "measurements" from one ledger; a false zero published as a measurement is worse than an
  honest estimate.

- **The liability report no longer contradicts its own table.** Three rendered strings
  still described the `free` list by its only previous cause. A `structuredContent`-only
  server is triggered heavily and pays nothing, so "installed but not triggered this
  window", the `1x-` legend, and "the proxy recorded the emission" were each false for the
  very server this change is about — printed directly above a `blocks` column reading 500.

### Added

- **`terse stats --json`: `primers[].attached`** (`#286`). `true` = the primer was emitted
  and cost its `tokens`; `false` = the proxy declined to send it, costing nothing. Rows
  written before this field have no `attached` key and are read as `true`, since the
  suppression row did not exist then. Additive.

## [0.28.1] - 2026-08-20

### Fixed

- **The primer charge is measured where it can be, instead of always inferred**
  (`#311`, `#286`). `terse stats` sized every wrapped server's primer from its installed
  policy and then billed it to anyone the ledger showed had been *called*. For a server
  whose results always carry `structuredContent`, the lazy attach never fires — so it
  paid nothing and was billed anyway. `#286` caught this in production: `searxng-mcp`
  charged 312 tok/session for a primer it is structurally incapable of sending, which
  made a free wrap look marginal.

  The proxy now writes a ledger row at the moment it attaches a lazy primer, so the cost
  is recorded rather than reconstructed. Servers with a recorded row report the measured
  figure; the rest keep the old estimate and are **labelled as estimates** rather than
  blended in. Deliberately no session id, epoch id, or cross-process correlation — that
  design was tried and rejected in `#312`.

  Reading absence of a row as proof of non-payment still needs a ledger-version floor
  (an old ledger has no primer rows either), so `#286`'s phantom bill is corrected only
  once the server has attached at least one primer in the window. Tracked on `#311`.

### Added

- **`terse stats --json`: two additions** (`#311`). Top-level `primers` lists primers
  actually emitted this window, by label and cadence — **empty means "this ledger cannot
  say", never "no primer was sent"**. Per-server `primer_source` is `"recorded"` or
  `"estimated"`, so a consumer never has to guess which of its numbers is a measurement.
  Both additive; the eager `initialize` sites are deliberately not recorded, since they
  emit unconditionally and inference there is already exact.

## [0.28.0] - 2026-08-19

### Added

- **`terse stats --json`: two additions to the break-even vocabulary** (`#285`). A new
  per-server `superseded_labels` (list) names ledger labels an entry wrote under before
  `--server-name` was baked in, and `break_even_verdict` gains `ambiguous ledger label` to
  its documented closed set. Both are additive, but a consumer switching exhaustively on
  that set will see a value it does not know — which is why this releases as a minor.

### Fixed

- **`terse stats` billed a wrapped server's break-even against another server's ledger
  rows** (`#285`). The per-server break-even table derived each standalone entry's ledger
  label from its downstream *command*, ignoring `--server-name` — the one flag that
  overrides what the proxy writes to `server`. Both failure directions were live in a real
  fleet and neither was distinguishable from a measurement: `searxng-mcp`
  (`.venv/bin/python -m searxng_mcp` → label `python`) was billed against unrelated `python`
  rows, manufacturing a cleared verdict out of another server's 3,390 saved tokens, while
  `secret-broker` (`… python3 …` → label `python3`) matched nothing and was published as
  `never called` in the same report whose per-tool table showed its 21 blocks at 59.6%. The
  existing guards could not catch either: their vocabulary (`no ledger label`, `never
  called`, `no token data`) all describes a *missing* label, and `python` is a real label
  with real rows. Break-even now reads the scan's `ledger_identity` —
  `resolve_ledger_identity`, the same rule `proxy.py`'s write path uses — whenever
  `--server-name` was explicit. Where two installed entries bake no flag and *guess* the
  same launcher basename (`python`, `python3.12`, `node`, `npx`, `uv`, …), that label's rows
  belong to both and to neither, so it now reports a new `ambiguous ledger label` verdict —
  distinct from `no ledger label`, which is documented as "matched no ledger rows" — naming
  the entries and the `--server-name` fix. A *lone* launcher wrap owns its rows outright and
  keeps its measurement.
- **`parse_proxy_opts` did not recognise `--flag=value`** (`#285`). `--server-name` is a
  plain argparse optional, so `--server-name=kb` is accepted by the proxy and written to the
  ledger as `kb`, but the config scan saw only the two-token form `wrap` emits. A
  hand-edited entry using the `=` spelling was therefore invisible to both `terse mcp-status`
  and break-even, reproducing the same silent `never called` on an entry that had already
  baked the flag that fixes it. Both spellings are now read, for `--policy` and
  `--capture-dir` too; a value containing `=` survives (split on the first one only). The
  config scan now sources `policy` from the same parser instead of its own two-token-only
  scan, so `--policy=/p.json` no longer leaves the row's policy unset — which had
  `terse stats` size that entry's primer from `default_policy()`, billing a primer to an
  entry whose real policy emits a different one, and stopped `policy_missing` from ever
  firing for a deleted file.
- **`--server-name=` (empty value) was reported as an explicit identity** (`#285` review).
  `resolve_ledger_identity` is `name or server_label(cmd)`, so an empty value correctly
  falls back to the command basename — but the scan flagged explicitness with `is not None`,
  telling break-even to trust that guess and skipping the ambiguity guard on exactly the
  hand-edited entries it exists for. Both now use the same truthiness.
- **A peerless multiproxy router was billed a primer for a phantom peer** (`#285` review).
  `scan_scopes` writes the literal `(no peers)` into a router row's `wraps` when its peers
  file is empty; break-even read that sentinel as a peer NAME, so the router published it as
  a ledger label and — worse — sized its union primer against it, charging a real per-turn
  cost into the headline `recurring tok/turn` figure for instructions it cannot emit. A
  peerless router now reports a known zero on both sides.
- **History stranded by baking `--server-name` is now reported** (`#285` review). The ledger
  records the identity in force at write time, so adding the flag to an entry that had been
  guessing renames the server from that moment and silently splits its history. The earlier
  label is surfaced per-server as `superseded_labels` (and named in the report) rather than
  merged: merging would be the guessing this release removed, and two labels can equally be
  two servers. Suppressed for launcher basenames, where "these rows are probably yours"
  cannot be said, and for any label another installed entry still answers to — a sibling
  wrap that never baked a name, or a router peer — so one server's live traffic is never
  printed as another's past.
- **The cadence legend re-collapsed the two causes of an unknown label** (`#285` review).
  It attributed every `1x?` to "no ledger label" one line below the table that had just
  distinguished it from `ambiguous ledger label`.

## [0.27.0] - 2026-08-19

### Added

- **`terse fluency --codec-verdict`: a downstream-outcome verdict for the codec tier**
  (`#295`). The existing fluency harness renders a single PASS/FAIL against a fixed 5%
  comprehension-accuracy tolerance — a budget for how much semantic damage the codec's
  *unconditionally lossless* tier is allowed to cause at the reader, which contradicts its
  own round-trip-proven losslessness claim. Comprehension failures also concentrate in
  `deref` (reconstructing terse's compressed form back into JSON), which is exactly what an
  agent does when it feeds a value into the next tool call — a `deref` miss is a malformed
  downstream tool argument, not "a wrong answer".

  `--codec-verdict` asks each `deref` question over raw vs terse via a real tool-calling
  model, scoring the emitted tool-call ARGUMENT by structural equality rather than a
  free-text comprehension score, and renders **SAFE / UNSAFE / UNRESOLVED per (tool,
  shape)** — never a global percentage. Any PAIRED excess of terse-arm misses beyond what
  raw also missed is UNSAFE, full stop, regardless of sample size; a clean run needs enough
  zero-failure trials (Clopper-Pearson bounded) to claim SAFE rather than UNRESOLVED. Drop
  tier is unaffected — `dropeval.py` already has this shape there.

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
