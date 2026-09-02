"""Measurement report: token delta per shape bucket with tier attribution.

Honesty requirements (plan Section 7, principle #24):
  - every shape bucket is shown, including near-zero / negative ones — never
    averaged away into a single headline number
  - coverage (which tools, how many payloads) is a first-class section, so a thin
    sample cannot read as "nothing to compress"
  - the lossless gate result gates the whole report: any round-trip failure prints
    an INVALID banner, because savings on top of lost data are meaningless
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, TypeGuard, assert_never

# Shared pass/fail tolerance for both worst-case verdict gates below — the number behind
# "safe to enable `proxy --diff`". Pinned to its VALUE, in both directions, by
# `tests/test_ship_policy_constants.py`: every other test reads it out of the source and
# asserts a relative inequality, so raising it to 0.06 left all 1706 green (#337).
_GAP_TOLERANCE = 0.05


def _form_stats(rows: list[dict[str, Any]], form: str) -> tuple[float, float]:
    """(accuracy, standard_error) for one form over rows carrying success COUNTS.

    accuracy = Σsuccesses / Σtrials. SE is the cluster-robust (sandwich) SE of that
    ratio estimator, clustering on the question — each row is one question, worth
    t trials of a Bernoulli:

        SE = sqrt( n/(n-1) · Σ(kᵢ − acc·tᵢ)² ) / Σt

    A pooled *within-question* SE (Σt·p̂(1−p̂)) looks stable but is measuring the wrong
    thing: at temperature 0 a question is nearly always all-right or all-wrong across
    its trials, so p̂(1−p̂)≈0 for every row and that estimator collapses to ≈0 SE at ANY
    accuracy, regardless of how much the *set of questions* disagreed with each other
    (#297). This form instead treats the question as the sampling unit — the axis that
    actually varies run to run — and reduces to 0 only when every row's per-question
    rate already equals the overall accuracy.

    n<2 clusters can't estimate a BETWEEN-question spread at all (that needs at least
    two questions to compare), but reporting a flat SE=0 there would print false
    certainty on exactly one draw — the same failure #297 filed against the old
    estimator, just relocated. So a single surviving question instead falls back to
    that question's own within-question binomial SE (√(t·p̂(1−p̂))/t), which is the
    pre-#297 formula restricted to n=1 — the best estimate available with one cluster,
    not a confident zero.
    """
    ks: list[int] = []
    ts: list[int] = []
    for r in rows:
        # Prefer a per-form trial count ("terse_ok" -> "terse_trials") when the row
        # carries one — an uneven hand-built pack can collect fewer replies for one form,
        # and dividing that form's successes by the shared per-row `trials` would
        # understate it. Falls back to the shared count, so the live/uniform path (no
        # per-form keys) reports exactly as before.
        t_key = form[:-3] + "_trials" if form.endswith("_ok") else ""
        t = r.get(t_key, r.get("trials", 1))
        k = int(r[form])
        # Every emitter is responsible for its own k<=t invariant (dropeval's is pinned
        # by test_no_success_count_can_exceed_its_own_trial_count) — this is the
        # last-line check so a future violation fails loud here, at its source, instead
        # of silently publishing an impossible accuracy. `0 <= k <= t` alone already
        # rejects every t<0 row (no k>=0 can satisfy k<=t<0) as well as the t==0/k>0
        # corner (a row can score a "success" on zero trials only by the same class of
        # bug this guard exists to catch) — not gated on `t > 0` the way an earlier
        # revision was, which let exactly that corner through.
        if not (0 <= k <= t):
            raise ValueError(f"{form}: {k} successes over {t} trials is not a valid count")
        if t > 0:
            ks.append(k)
            ts.append(t)
    if not ks:
        return 0.0, 0.0
    tot_t, tot_k = sum(ts), sum(ks)
    acc = tot_k / tot_t
    n = len(ks)
    if n < 2:
        k, t = ks[0], ts[0]
        p = k / t
        return acc, math.sqrt(t * p * (1 - p)) / t
    resid_sq_sum = sum((k - acc * t) ** 2 for k, t in zip(ks, ts, strict=True))
    se = math.sqrt(n / (n - 1) * resid_sq_sum) / tot_t
    return acc, se


def _ci(se: float) -> float:
    """95% half-width in accuracy units."""
    return 1.96 * se




def _arm_attempts(r: dict[str, Any], trials_key: str, default: int) -> int:
    """How many calls THIS arm was asked to make on this row.

    `<arm>_trials` says how many it completed — except in `codeceval.py`, whose docstring
    (`src/terse/codeceval.py`) records that it keeps the key FIXED at `trials` on purpose and
    reports its loss only through `fails`/`attempts`, which is why `_unmeasured` keeps a
    pooled trigger. The attempt count `<arm>_trials` should be read against is the row's
    shared `trials` for every live harness — every arm is put the same question
    the same number of times — but NOT for `score_pack`, whose per-form counts differ by
    collection design and whose `trials` is a `max(...)` across forms (#91). Those rows state
    `<arm>_attempts` explicitly, and both `paired_rows` and `_unmeasured` read it here so
    that a design-uneven form is not counted as a lost call (#283)."""
    arm = trials_key[:-len("_trials")] if trials_key.endswith("_trials") else trials_key
    return int(r.get(arm + "_attempts", default))


def arm_measured(rows: list[dict[str, Any]], form: str) -> bool:
    """Did this run collect ANY calls for `form`? False means the arm is absent, not failing.

    `_form_stats` returns a flat `(0.0, 0.0)` for an arm with no completed trials, which
    renders as a confident `0% ±0` — indistinguishable from "comprehension collapsed" and
    exactly the unknown-is-not-zero mistake the inline arm's `n/a` already avoids one
    function down. Only `<arm>_attempts` can tell the two apart, so a row set that does not
    carry it (every live harness, where all arms are always attempted) answers True.

    NOT filtered to rows that carry the key: a row set predating the counters has no
    `<arm>_trials` at all, and skipping those rows would leave nothing to check and report
    a fully-measured arm as absent — which is how the first cut of this rendered `n/a` for
    the primer column of every legacy result file."""
    key = form[:-3] + "_trials" if form.endswith("_ok") else form
    return any(_arm_attempts(r, key, int(r.get("trials", 1))) for r in rows)


def paired_rows(rows: list[dict[str, Any]], *forms: str) -> list[dict[str, Any]]:
    """The subset of `rows` on which every one of `forms` completed ALL of its trials.

    Scoring is PAIRED — `harnesses`' module docstring calls that load-bearing: the same
    questions, in the same order, put to every arm. `_form_stats` divides each arm by its
    OWN `<form>_trials`, so dropping an unanswered call from one arm silently re-bases that
    arm onto a DIFFERENT question set, and the "gap" stops comparing like with like.

    That is the real defect behind #268's near-miss, and a per-arm loss THRESHOLD does not
    close it — it only raises the bar. Measured: a model answering the control perfectly and
    returning no content on every `deref` question (the longest prompt, so the first to hit
    a token-budget stop) loses exactly 1 of 5 question types. That is 20.0% of the arm — on
    the threshold, and the comparison is strictly `>` — so the gate stayed quiet while a
    real **-20% FAIL** rendered as **PASS / "safe to enable `proxy --diff`"**. Question
    difficulty varies far more than trial-to-trial noise, so losing the HARD questions from
    one arm flatters it without bound, at any share.

    Excluding the row from BOTH arms restores the pairing. Coarser than dropping the
    individual trial — the row counts say how many trials survived, not WHICH — so a row is
    comparable only when both arms answered every trial of it.

    TWO kinds of row are always kept, because in neither is an uneven trial count evidence
    of a LOSS — and pairing only defends against loss:

      - rows with no `<form>_trials` key at all (result files predating #263). `.get(k, t)`
        reads those as complete, which is the same "absent is not evidence of failure" rule
        `_unmeasured` applies to its own counters;
      - rows with no `attempts` key: legacy result files, and hand-built #91 rows. HISTORICAL
        NOTE, because this bullet used to read "`score_pack` never calls a backend, so it has
        no transport to lose calls to" and #283 falsified that: `score_pack` now DOES emit
        `fails`/`attempts` (a stored empty reply is a call that produced no answer, and
        without the counters `--responses` had no transport gate at all). It no longer takes
        this escape. What remains true is the reason the escape existed — an uneven per-form
        count is COLLECTION DESIGN, not loss: an uneven hand-built pack may carry 3 raw
        replies and 2 terse ones for the same question, and #91 added those counters
        precisely so the sparser form is scored over its own denominator instead of being
        understated. Its `trials` is `max(...)` of the forms, not an attempt count.

    So that reason is now served by a counter rather than by an absence. An arm's attempt
    count is `<arm>_attempts` WHERE THE ROW STATES ONE, and the shared `trials` only where it
    does not (#283) — inferring one arm's attempts from the row `max(...)` would read the
    design as loss. `<arm>_attempts` is a fact the emitter knows and this function cannot
    derive. Absent, as in every live-harness row, the fallback is the previous behaviour
    exactly.

    ZERO attempts for an arm on a row is decided at the RUN level, not the row level,
    because within a single row the two cases are indistinguishable and they are opposites:

      - an arm collected on NO row (a responses file that ran raw and terse but never the
        primer arm) is an ABSENT arm. Voiding every row for it would empty the paired subset
        and withhold a perfectly good pack — the same "absent is not evidence of failure"
        rule as the two bullets above, and the same distinction `build_fluency_report` draws
        with `has_inline`;
      - an arm collected on SOME rows is a LOST question wherever it is missing. Keeping
        those rows re-bases that arm onto the subset it happens to have been collected on
        while the control keeps every question — which is precisely the failure this
        function's first paragraph exists to prevent, arriving through a different door.

    That second case is not hypothetical: measured on a 24-question pack whose terse arm was
    collected only for the six `count` questions (the easiest type) and answered them
    perfectly, the report rendered `best terse-form 100% vs raw 100% ... **PASS**` with `18
    regressions` printed in the same row. Scoring an uncollected form as a miss — the #279
    defect — had been hiding it behind a terse arm of 6/24 = 25% and a loud FAIL, so fixing
    #279 without this is a strict regression from a conservative wrong answer to a false
    green. Both numbers are executed, not estimated.

    Pinned by `test_rows_without_per_form_counters_are_treated_as_fully_paired`,
    `test_an_uneven_score_pack_still_publishes`,
    `test_a_real_uneven_score_pack_with_counters_still_publishes` and
    `test_an_arm_collected_for_only_some_questions_cannot_publish_against_a_full_control`."""
    keys = [f[:-3] + "_trials" if f.endswith("_ok") else f for f in forms]
    # Which of the requested arms this run collected anything for at all. Computed over the
    # whole row set because that is the only place the fact exists — see the docstring.
    collected = {k for k in keys if arm_measured(rows, k)}
    out = []
    for r in rows:
        t = int(r.get("trials", 1))
        if "attempts" not in r:
            out.append(r)
            continue
        if all(_paired_arm(r, k, t, k in collected) for k in keys):
            out.append(r)
    return out


def _paired_arm(r: dict[str, Any], trials_key: str, trials: int, collected: bool) -> bool:
    """Did this row's arm complete every call it was asked to make?

    `collected` is the run-level fact from `paired_rows`: True when SOME row of this run has
    attempts for the arm. An arm with zero attempts on this row is then a question it did not
    answer while its neighbours did — unpaired. When `collected` is False the arm is absent
    from the whole run, which is not a loss, and the row stands."""
    a = _arm_attempts(r, trials_key, trials)
    if a == 0:
        return not collected
    return int(r.get(trials_key, trials)) == a


# Share of a model's calls that may fail before its numbers stop meaning anything (#263).
# NOT zero: a failed call is already excluded from its arm's denominator (see the
# `<form>_trials` keys `run_payload` emits), so a handful of transient 429s no longer
# depress an accuracy at all — voiding a whole model for one of them would discard an
# otherwise-complete multi-hour run, which is its own kind of wrong answer. What a
# threshold still has to catch is the backend that was substantially down, where the
# surviving sample is small and self-selected rather than merely smaller.
#
# THE DENOMINATOR IS PER ARM (#339), NOT POOLED ACROSS EVERY ARM. `_unmeasured` finds each
# arm's own `<arm>_trials` counter and sums, over the rows that carry that key, its loss as
# `<arm>_attempts - <arm>_trials` and its attempts as `<arm>_attempts` — where
# `<arm>_attempts` is the row's shared `trials` for every live harness and an explicit
# per-arm count for `score_pack` (#283, see `_arm_attempts`) — then fires if ANY single arm
# exceeds `UNMEASURED_FAIL_SHARE` of its own calls. Before
# #339 the denominator was pooled `attempts` (= trials * arm_count), which let a single
# arm lose over 40% of ITS calls in a two-arm run, or over 80% in a four-arm one, before
# this fired — the threshold read as "20% of calls" and behaved as "20% of all arms'
# calls pooled". `paired_rows`' docstring below works through "one of five question types
# lost" and calls it "20.0% of the arm — on the threshold": that arithmetic is now what
# this gate itself computes, not a separate concern it stays quiet about.
#
# Value and the strictness of the comparison are pinned by
# `test_a_model_exactly_at_the_loss_share_is_still_measured` (#337): `>` -> `>=` survived
# a full-suite mutation, so nothing observed which side of the line a model fell on.
UNMEASURED_FAIL_SHARE = 0.20

# WHY THE PAIRING-LOSS GATE IS `not pr` AND NOT A SHARE (#332).
#
# The bug: `paired_rows` voids a whole ROW when either arm loses one trial of it, so loss
# is amplified from the call level to the question level. At three trials an arm, one lost
# call per question is 16.7% of the calls pooled — under `UNMEASURED_FAIL_SHARE` at the time
# — and 100% of the questions. `_unmeasured` stayed quiet, nothing was left to compare, both
# arms scored a flat 0.0 with an SE of 0.0, and the HTML banner rendered a green PASS reading
# `+0% ±0pt`: maximum confidence from no evidence at all.
#
# NOTE post-#339: if every question loses one of the SAME arm's trials (as in this
# paragraph's example, repeated across the whole set), that arm's own share is 33.3%, not
# 16.7%, and the per-arm `_unmeasured` trigger now catches it before pairing does — this
# paragraph is describing the pre-#339 gate that made the bug reachable, not today's. The
# case this file's fixtures still need (and still reach the pairing floor without tripping
# `_unmeasured`) is more trials per arm with the SAME one-lost-trial-per-row shape, which
# dilutes the per-arm share under the line while still voiding every row — see
# `_pairing_wipeout_rows`.
#
# The first fix withheld any model under a `PAIRED_KEEP_SHARE = 0.50` survival floor. Review
# killed it, and the reason is worth keeping because it inverts the intuition: the rows that
# SURVIVE pairing are not a degraded remnant, they are the strongest evidence the harness
# produces — every arm completed every trial of them. A floor discards that evidence, and
# because an exclusion removes a model from the gate entirely, discarding it can IMPROVE the
# run's verdict. Reproduced at 10% call loss (half the existing threshold): six voided rows
# plus four fully-paired rows showing a real -100% regression rendered `**FAIL** ... keep
# proxy --diff off` before the floor and `**PASS** ... safe to enable proxy --diff` after it.
# A gate against a false green that manufactures a false green is not a smaller version of
# the right fix; it is the same defect. `codec_verdict` states the principle one screen
# down: any paired excess of misses is unsafe "regardless of how small a fraction of the
# sample it is".
#
# So the gate fires only where there is provably nothing to discard. An empty paired subset
# cannot be hiding a demonstrated regression, which makes this narrow form incapable of the
# failure the share-based one had. A minority-but-nonempty subset still publishes; its small
# `n` is the reader's to weigh, and `build_dropeval_report` prints the surviving count.
#
# Pinned by `test_a_demonstrated_regression_is_never_withheld_as_unmeasured` — the guard
# against re-introducing the floor. Do not add a survival threshold here without first
# making an exclusion unable to improve a verdict; those are one change, not two.


def _unmeasured(rows: list[dict]) -> bool:
    """True when transport failures make this model's numbers untrustworthy.

    THREE independent triggers, because they fail differently:
      1. any arm with ZERO completed trials THAT WAS ACTUALLY ATTEMPTED — that arm cannot
         be computed at all, and `_form_stats` would report it as a flat 0.0 indistinguishable
         from real failure. An arm nobody collected (a `score_pack` responses file with no
         primer replies) is absence, not failure, and does not fire it (#283);
      2. more than `UNMEASURED_FAIL_SHARE` of ONE arm's own calls lost (#339) — the sample
         that survived is both small and selected by which calls happened to get through,
         and the share has to be read against that arm's own denominator or a loss
         concentrated on one arm hides behind the others' clean numbers;
      3. the pooled fallback: more than `UNMEASURED_FAIL_SHARE` of TOTAL `attempts` lost.
         Needed because not every harness's `<arm>_trials` shrinks on a failed call —
         `codeceval.py` deliberately keeps it FIXED at `trials` and tracks loss only
         through `fails`/`attempts` (see its module docstring), so trigger 2 alone is
         permanently blind to that harness and a substantially-down codec backend would
         publish a confident SAFE. Review finding on #339 (verified by execution:
         `codec_verdict` returned SAFE at 68% call-loss without this trigger).
    """
    if not rows:
        return False
    attempts = sum(int(r.get("attempts", 0)) for r in rows)
    if not attempts:
        # Rows predating the counters (older result files) carry neither key. Absent is
        # not zero-failures, but it is also not evidence of failure — treat as measured,
        # exactly as this report did before the counters existed.
        return False
    # Arms are DISCOVERED from the rows, not hardcoded. The payload harness emits
    # raw/terse/primer/inline; the diff harnesses emit terse/diff. A fixed list would
    # silently skip every arm it did not name — which is how the diff-side report kept
    # publishing a verdict off a dead backend after the payload side stopped.
    arm_keys = sorted({k for r in rows for k in r if k.endswith("_trials")})
    for key in arm_keys:
        if sum(int(r.get(key, 0)) for r in rows) != 0:
            continue
        # Zero completed trials is only a FAILURE where the arm was actually attempted.
        # `score_pack` emits a full set of arm keys whether or not the responses file
        # collected that form, so a pack that simply never ran the primer arm would
        # otherwise withhold the whole model on the strength of a question nobody asked
        # (#283). `<arm>_attempts` is how a row says which it is; absent — every live
        # harness — the shared `trials` stands in and this reads exactly as before.
        if any(_arm_attempts(r, key, int(r.get("trials", 1))) for r in rows if key in r):
            return True
    fails = sum(int(r.get("fails", 0)) for r in rows)
    if fails / attempts > UNMEASURED_FAIL_SHARE:
        return True
    # #339: the threshold is documented and tested as a share of ONE arm's own calls, not
    # of every arm's calls pooled. Dividing by pooled `attempts` (= trials * arm_count) let
    # a single arm lose over 40% of its own calls in a two-arm run, or over 80% in a
    # four-arm one, before this fired — see `UNMEASURED_FAIL_SHARE`'s comment. Only rows
    # that CARRY both an arm's key AND `attempts` count toward that arm's loss/attempts —
    # matching `paired_rows`' own per-row `"attempts" not in r` escape — so a row from a
    # collection mode carrying no `attempts` key at all (a legacy result file, or a
    # hand-built #91 row) contributes nothing either way, even when merged into a row set
    # that does have them (review finding on #339: checking only `key in r` let one such
    # merged row swing the whole model to withheld with zero calls actually lost).
    for key in arm_keys:
        carrying = [r for r in rows if key in r and "attempts" in r]
        if not carrying:
            continue
        # Per row, this arm's OWN attempt count — `<arm>_attempts` where the emitter states
        # one, else the shared `trials` (#283, see `_arm_attempts`). Without that, a
        # `score_pack` row's `trials` (a `max(...)` across forms) would make an
        # uneven-BY-DESIGN form look like a lost call, which is the same class of error as
        # the pooled denominator #339 removed — just one level down.
        per_row = [(_arm_attempts(r, key, int(r.get("trials", 1))), int(r[key]))
                   for r in carrying]
        arm_attempts = sum(a for a, _ in per_row)
        if not arm_attempts:
            continue
        lost = sum(max(0, a - t) for a, t in per_row)
        if lost / arm_attempts > UNMEASURED_FAIL_SHARE:
            return True
    return False


# Why a gap is never computed from two bare `_form_stats` calls again (#280).
#
# `_form_stats(rows, form)` computes ONE arm. Every gap site therefore called it twice and
# subtracted, and nothing in that shape can enforce that the two arms answered the SAME
# questions — pairing is a property of the pair, and `_form_stats` never sees a pair. That
# is why the same false-PASS survived three fixes: each pass wired pairing into the sites
# it was looking at, and the next site was still writable by accident. The third attempt's
# own commit claimed "every diff-vs-control gap site"; reverting its pairing at two of the
# three left the entire suite green.
#
# So the shape is removed rather than the sites patched. `arm_gap` is the only place a
# form/control pair is turned into comparable numbers, and
# `tests/test_gap_gate_boundary.py` asserts by AST that `_form_stats` is called from
# nowhere else but an explicit allowlist. Adding a seventh gap site cannot skip the gate
# silently; it has to edit that list, in a diff a reviewer sees.
# The CLOSED set of reasons a model can be withheld from a gate. A `Literal`, not `str`,
# and that is the single most load-bearing type in this file.
#
# `excluded: str | None` said "some string". Nothing anywhere knew there were exactly
# eight, so every consumer was a site where a programmer had to REMEMBER the list — and
# across four review rounds on #335, four separate sites forgot: a reason was added and the
# next consumer over kept its old fallthrough. Ten of that batch's 27 findings are that one
# shape. With a `Literal`, a ninth reason plus a consumer that does not handle it is a mypy
# error at the consumer, at edit time, instead of a review finding four rounds later.
#
# What the `Literal` actually buys, precisely — the looser claim ("mypy rejects a consumer
# that misses a reason") is only true of some of them:
#   - PRODUCERS: every dict that carries a reason is annotated `ExclusionReason | None`
#     (`_gap`, `_accuracy_gate`, `dropeval_verdict`, `diff_gap_rows`, `fluency_gap_rows`,
#     `_not_measured_lines`, `html_report`'s `excluded`), so a typo'd or invented reason is
#     a mypy error at the assignment. Verified by injecting one.
#   - CONSUMERS that switch on the reason (`_reason_directive`, `_exclusion_remedy`) end in
#     `assert_never`, so a new member with no arm is a mypy error at the switch.
#   - CONSUMERS that look the reason up with a FALLBACK (`REASON_LABEL.get(why, why)` and
#     friends, six sites) cannot be checked by mypy at all — a dict is not exhaustive-
#     checkable. `test_every_exclusion_reason_has_a_label_a_heading_a_directive_and_a_remedy`
#     covers those by asserting both maps are total over `typing.get_args(ExclusionReason)`.
ExclusionReason = Literal[
    "unmeasured",
    "broken control",
    "not a diff run",
    "empty",
    "no control arm",
    "partial control coverage",
    "underpowered",
    # Set by `build_diff_soak_report`'s per-depth table via `ArmGap._replace`, splitting
    # the transport half of "unmeasured" from the pairing half for that table only. It is a
    # real inhabitant of this type even though `_gap` never returns it, so leaving it out
    # would make the Literal a lie in exactly the direction it exists to prevent.
    "unpaired",
]


class ArmGap(NamedTuple):
    """One model's form-vs-control numbers, or why it was withheld.

    `excluded` is a REASON, not a bool: sites 1/2/3 each render withheld models in their
    own prose, and `test_the_renderers_agree_on_which_models_were_dropped` pins that they
    agree about WHICH models. Carrying the reason from the one place that decides it keeps
    that true by construction rather than by three copies of the same condition."""
    form_acc: float
    form_se: float
    control_acc: float
    control_se: float
    rows: list[dict[str, Any]]  # the PAIRED subset the numbers above were computed over
    excluded: ExclusionReason | None
    # Per-arm (accuracy, se) over that same paired subset, for the table columns beside the
    # verdict. Carried here rather than left to callers so a report cannot print a column
    # computed over a different question set than the gap printed next to it — which is the
    # #280 defect in miniature, and would also put `_form_stats` back in every renderer.
    #
    # A NamedTuple default is a single instance shared by every defaulted `ArmGap`, so a
    # plain `{}` here would let one stray `g.arms[k] = v` corrupt every future excluded
    # ArmGap process-wide. Read-only today; the proxy makes that structural rather than a
    # convention nobody checks.
    arms: Mapping[str, tuple[float, float]] = MappingProxyType({})


# How many PAIRED questions a PASS needs before "no regression observed" means anything
# (#334). A floor on the evidence behind a NEGATIVE result, not a tolerance for a positive
# one — the same shape and the same statistics as `_CODEC_MIN_TRIALS` below, and cited from
# there rather than invented again: Clopper-Pearson bounds the true regression rate below
# `1 - 0.05 ** (1/n)` at 95% confidence when n questions show zero regressions. At n=20
# that is ~14pp; at n=3 it is 63pp, which is why the reported #334 case (a `+0% ±0pt` PASS
# off ONE surviving question, at 15% call loss) could print maximum confidence off nothing.
#
# QUESTIONS, not trials, unlike `_CODEC_MIN_TRIALS`. #297 established that trials within a
# question are correlated — at temperature 0 a question is nearly always all-right or
# all-wrong — which is why `_form_stats` clusters its SE on the question. Counting trials
# here would multiply the apparent evidence by `--trials` without adding any.
#
# ASYMMETRIC, and that is the whole design. #332's first attempt was a symmetric survival
# floor that withheld models regardless of what they showed; because an exclusion drops a
# model from the gate entirely, it was measured turning a demonstrated -100% regression
# into "safe to enable `proxy --diff`". This floor can only ever withhold a model whose
# form arm is NOT behind its control, so what it removes from `_worst_case_gap` is always a
# non-negative gap — which cannot be the worst case unless every model is non-negative, and
# then the run is a PASS or a NO VERDICT either way. An exclusion can never improve a
# verdict here. Pinned by `test_a_demonstrated_regression_is_never_withheld_as_unmeasured`
# and `test_a_small_but_failing_arm_still_publishes_its_FAIL`.
_MIN_PAIRED_QUESTIONS = 20


def passes_tolerance(gap: float, tol: float = _GAP_TOLERANCE) -> bool:
    """Is `gap` within `tol`? THE definition — every verdict and the #334 floor share it.

    The epsilon is not decoration. `0.40 - 0.05` is `0.35000000000000003` in binary float,
    so `facc >= cacc - tol` and `gap >= -tol` disagree on 122 distinct exact-boundary
    accuracy pairs. #334's floor was written the first way and the verdicts the second,
    which put those pairs in neither set: not withheld (the floor thought the arm was
    behind), not failed (the verdict thought it was inside tolerance). Measured on
    `35% vs 40%` over 10 paired questions — a green "safe to enable `proxy --diff`" with
    `±0 pts`, the exact symptom #334 was filed on, surviving inside its own fix.

    Callers must pass a GAP (form minus control), never compare accuracies directly."""
    return gap >= -tol - 1e-9


def _gap(rows: list[dict[str, Any]], gating: list[str], control: str,
         display: tuple[str, ...] = (), *, min_paired: int | None = None) -> ArmGap:
    """Shared body of `arm_gap`/`best_arm_gap`. Gates, pairs, then computes every arm.

    Order matters: `_unmeasured` first (the backend was down — nothing here is measurable),
    then pair, then the pairing-loss floor (it was up, but too little of the question set
    survived on one side to compare), then compute. A caller that wants the numbers must
    accept the gate, because they arrive together.

    Both gates report the SAME reason, `"unmeasured"` — one vocabulary for the reader, as
    `REASON_LABEL`'s note explains. #332 planned a distinct `"unpaired"` reason and it is
    NOT built here: the two causes reach the reader as the same fact ("calls went
    unanswered, so there is no verdict") and a second vocabulary is one more thing six
    renderers can disagree about. (`"unpaired"` does exist as an `ExclusionReason`, set by
    the diff-soak per-depth table via `_replace` for its own prose — that table draws the
    distinction locally, at one site, which is a different thing from `_gap` minting it for
    every renderer. An earlier version of this paragraph said the reason was "retired",
    which contradicted the `ExclusionReason` list 200 lines above it.)

    `display` arms are computed over the paired subset but do NOT participate in pairing.
    That is deliberate: `run_payload`'s `inline` arm carries the longest prompt of the four
    and so truncates first under a token-budget stop, while gating nothing — pairing on it
    would void otherwise-complete runs over an arm no verdict consumes."""
    if not rows:
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "empty")
    if _unmeasured(rows):
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "unmeasured")
    pr = paired_rows(rows, *gating, control)
    if not pr:
        # Nothing completed every trial on every arm, so there is no comparison to make.
        # Falling through would compute `_form_stats([], f)` == (0.0, 0.0) for BOTH arms:
        # a gap of exactly zero carrying an SE of exactly zero, which every renderer reads
        # as a confident PASS. See the note on this gate above for why it is `not pr` and
        # not a survival share.
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "unmeasured")
    arms = {f: _form_stats(pr, f) for f in (*gating, control, *display)}
    cacc, cse = arms[control]
    if control in ("raw_ok", "control_ok") and cacc == 0:
        # A control at exactly 0% is a backend/config error, not a comprehension result —
        # every form would "beat" it for free. True for `raw_ok` (fluency's raw-payload
        # control) AND `control_ok` (dropeval's no-drop control, #269): the control arm
        # answers a question with the value sitting verbatim in its payload, and #269's own
        # live reproduction measured that arm at 88%, not 0% — a 0% control means the
        # grader or backend produced no signal, not that the drop is blameless (review
        # finding 3 on #300). Kept here rather than at the call sites so the markdown
        # verdict and the forest plot cannot disagree about it.
        return ArmGap(0.0, 0.0, cacc, cse, pr, "broken control", arms)
    best = max((arms[f] for f in gating), key=lambda s: s[0])
    floor = _MIN_PAIRED_QUESTIONS if min_paired is None else min_paired
    # The cut is the TOLERANCE line, not exact equality. Withholding only `best >= cacc`
    # left the whole `[-_GAP_TOLERANCE, 0)` band escaping both gates at once: such a gap is
    # behind its control, so the floor let it through, and inside tolerance, so the verdict
    # passed it. Measured: one paired question at 19/20 vs 20/20 published
    # `gap -5% ±10 pts  **PASS** ... safe to enable proxy --diff` at 90% pairing loss.
    #
    # Drawn here, the withheld set is exactly the gaps that would have PASSED — literally
    # so, via the shared `passes_tolerance`, because writing the comparison twice is what
    # leaked a false green through the first version of this floor. That is what
    # what keeps the asymmetry argument in `_MIN_PAIRED_QUESTIONS` intact: removing a
    # would-be PASS from `_worst_case_gap` cannot turn a FAIL into a PASS, because the
    # failing model is still in the set. A form arm genuinely BEHIND its control by more
    # than tolerance publishes at any question count.
    if len(pr) < floor and passes_tolerance(best[0] - cacc):
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "underpowered")
    return ArmGap(best[0], best[1], cacc, cse, pr, None, arms)


def arm_gap(rows: list[dict[str, Any]], form: str, control: str, *,
            min_paired: int | None = None) -> ArmGap:
    """`form` vs `control` over the rows BOTH arms completed, or an exclusion reason.

    The single chokepoint for every diff-vs-control verdict in terse."""
    return _gap(rows, [form], control, min_paired=min_paired)


def best_arm_gap(rows: list[dict[str, Any]], forms: list[str], control: str,
                 display: tuple[str, ...] = ()) -> ArmGap:
    """`arm_gap` for the BEST of several forms against one control (the fluency shape).

    Pairs across every named arm before picking a winner, not after: choosing the best arm
    first and pairing second would let an arm win on a question set the others never
    answered — the same defect one level up. `build_fluency_report` and `fluency_gap_rows`
    both route here, which also collapses the duplicate copy of this math they carried."""
    return _gap(rows, forms, control, display)


# The ONE vocabulary for why a model was withheld. Every renderer reads it from here.
#
# Three exclusions reach the reports and they are different events: the backend never
# answered, the backend answered but the arms cannot be compared, and the control itself
# failed. Each renderer used to hardcode a single phrase for whatever landed in its
# exclusion list — so the terminal plot called an unpaired model "raw control failed" while
# its raw control read 100%, and the HTML page told a reader to check stderr for a
# `returned no content` line about a backend that answered 90% of its calls.
#
# That is the #280 defect wearing different clothes: one fact, restated independently at
# six sites, drifting at five of them. `exclusion_note` is to the prose what `arm_gap` is
# to the numbers, and `test_every_renderer_names_the_right_exclusion_reason` loops the
# invariant over all of them so a seventh renderer cannot invent a seventh phrasing.
REASON_LABEL = {
    # NOT "calls went unanswered": since #332 this reason also covers a backend that
    # answered almost everything, where the losses landed so as to leave no question
    # complete on both arms. Reproduced at 5% loss. The label has to be true of both.
    "unmeasured": "too few calls to compare",
    "broken control": "control arm failed",
    "not a diff run": "no diff arm in these rows",
    "empty": "no rows",
    "no control arm": "no control arm was run",
    "partial control coverage": "control ran on only some rows",
    # Distinct from "unmeasured" ON PURPOSE. Nothing failed here: the backend answered, the
    # arms paired, there were simply too few questions to conclude anything from an absence
    # of regressions. Folding it into "unmeasured" would print "too few calls to compare"
    # about a run that lost no calls at all — the #332 mistake, one reason over.
    "underpowered": f"fewer than {_MIN_PAIRED_QUESTIONS} paired questions",
    # Set only by the diff-soak per-depth table, whose own `lead` dict already carries a
    # phrasing for it — so as of today this entry renders NOWHERE, and replacing its value
    # with garbage leaves the suite green. It is here for TOTALITY, not to fix a live
    # misrender: every other reason is reachable through `exclusion_note` and
    # `REASON_HEADING.get(why, ...)`, both of which fall back silently on a missing key, so
    # the next site that hands a reason to either has no protection unless the map covers
    # the whole `Literal`. (An earlier version of this comment claimed the raw token was
    # already leaking to a reader. It was not — the one producer's value reaches only
    # `withheld_depths`. Recorded rather than deleted because a false comment written while
    # closing a real gap is the exact failure #342 counts five of.)
    "unpaired": "no question completed on both arms",
}

# The heading each reason gets in prose renderers (markdown, HTML). "Not measured" and
# "Not compared" are different claims and the distinction is the whole point: one says the
# backend did not answer, the other says it did.
REASON_HEADING = {
    "unmeasured": "Not measured",
    "broken control": "Excluded",
    "not a diff run": "Not applicable",
    "empty": "Not measured",
    "no control arm": "Not run",
    "partial control coverage": "Excluded",
    # Not "Not measured": it WAS measured, and the measurement simply does not reach.
    "underpowered": "Not concluded",
    # Unrendered today for the same reason as `REASON_LABEL["unpaired"]` — see the note
    # there. The wording follows the distinction the diff-soak table already draws in its
    # own prose: the backend ANSWERED, the two arms simply share no question.
    "unpaired": "Not compared",
}


def exclusion_note(reasons: Mapping[str, ExclusionReason | None]) -> str:
    """`(excluded — <why>: a, b; <why>: c)`, grouped by reason. "" when nothing was cut.

    One line, because its consumers are a terminal plot footer and an HTML paragraph that
    both sat on a single hardcoded phrase before."""
    if not reasons:
        return ""
    by: dict[str, list[str]] = {}
    for model, why in sorted(reasons.items()):
        by.setdefault(REASON_LABEL.get(why or "", str(why)), []).append(model)
    return "excluded — " + "; ".join(
        f"{label}: {', '.join(models)}" for label, models in sorted(by.items()))


def unmeasured_cause(fails: int) -> str:
    """The cause-and-remedy sentence for an `"unmeasured"` exclusion, CHOSEN BY THE COUNTS.

    `"unmeasured"` has FOUR producers now (#339 split the old trigger 2 in two), not the
    two every prose site named before #338:

      1. `_unmeasured` trigger 1 — an arm whose `<form>_trials` sum to ZERO. It never ran,
         or a merged/replayed pack carries rows that lack its key. Fires at `fails == 0`.
      2. `_unmeasured` trigger 2 — pooled loss share over `UNMEASURED_FAIL_SHARE`. Always
         `fails > 0` (it IS `fails / attempts`).
      3. `_unmeasured` trigger 3 (#339) — ONE arm's own loss share over
         `UNMEASURED_FAIL_SHARE`, read from `trials - <arm>_trials`, independent of
         `fails`. NOT guaranteed `fails > 0`: a harness whose `<arm>_trials` reflects real
         loss but whose `fails` counter undercounts it (a coupling nothing enforces — see
         the caution on `_unmeasured` above) would fire this trigger while `fails == 0`,
         which would wrongly render this function's zero-loss branch below. No known
         harness does this today; flagged as a maintenance trap, not a live bug.
      4. `_gap`'s `not pr` — the calls landed but left no question complete on both arms.

    #332 hedged the prose across (2)/(3) and (4) — "either too many calls went unanswered, or
    enough of them did that no question completed on BOTH arms". Both disjuncts name lost
    calls, so the hedge is FALSE for (1), which is reachable with nothing lost at all: the
    #338 reproduction renders `(0/36 calls lost)` and a sentence asserting calls went
    unanswered six words later, plus a remedy pointing at a backend that answered
    everything. A disjunction of two false claims is not a smaller version of one false
    claim.

    So the sentence is chosen by the only evidence that separates them — whether any call
    was actually lost — and at zero loss no transport vocabulary is emitted at all. That is
    what `test_every_renderer_names_the_right_exclusion_reason` greps for: at `fails == 0`
    no renderer may say "unanswered", "unreachable", or "fix the backend"."""
    if fails:
        return (
            " Either too many calls went unanswered, or enough of them did that no "
            "question completed every trial on BOTH arms, leaving nothing comparable — "
            "the counts above say which, and a low one means the second. An unanswered "
            "call is not a wrong answer. Check stderr for a `returned no content` line "
            "naming a `finish_reason` — `length` means raise max_tokens, "
            "`content_filter` means the payload tripped a filter. If most calls were "
            "answered, lower `--trials`: each extra trial is another chance for a "
            "question to lose one and be dropped from both arms.")
    return (
        " No calls were lost, so transport is not the cause: either an arm completed zero "
        "trials — it was never run, or a merged pack carries rows without its "
        "`<form>_trials` key — or the trials that did run left no question complete on "
        "both arms. Check that every arm named above actually ran, and lower `--trials` "
        "if the rows show one arm short of the others.")


def _not_measured_lines(
        withheld: dict[str, tuple[ExclusionReason | None, int, int, int]]) -> list[str]:
    """The withheld-models paragraph(s), with the counts that justify each one.

    Grouped BY REASON, and this is the first version where that is more than a promise.
    The signature has always carried the exclusion reason and the body has always thrown it
    away, back when `"unmeasured"` was the only reason that reached a report — the docstring
    said the wording was reason-specific while the code hardcoded one sentence. #334 adds
    `"underpowered"`, whose counts and whose remedy are both different: no calls were lost,
    so "calls lost" is the wrong number and "check stderr" is the wrong advice.

    Each model arrives as `(why, fails, attempts, paired)`."""
    if not withheld:
        return []
    by: dict[ExclusionReason | None, list[tuple[str, int, int, int]]] = {}
    for m, (why, f, a, paired) in sorted(withheld.items()):
        by.setdefault(why, []).append((m, f, a, paired))
    out: list[str] = []
    for why, models in sorted(by.items(), key=lambda kv: str(kv[0])):
        if why == "underpowered":
            out.append("") if out else None
            out.append(
                f"**{REASON_HEADING['underpowered']}** — "
                # Carries the call-loss count too when there is one. `unmeasured` is True
                # for an underpowered model, and the "Partially degraded" paragraph is
                # gated on `not unmeasured`, so without this a run that lost calls AND
                # fell short of the floor disclosed its losses in no line of the report.
                + ", ".join(
                    f"`{m}` ({paired} paired question(s)"
                    + (f", {f}/{a} calls lost)" if f else ")")
                    for m, f, a, paired in models)
                + f". These arms are not failing, but "
                  f"{_MIN_PAIRED_QUESTIONS} paired questions are needed before an ABSENCE "
                  "of regressions supports a PASS, and short of that the run is not "
                  "evidence either way. Generate more questions, or — if the counts "
                  "above show calls were lost — recover the ones pairing dropped by "
                  "lowering `--trials`. A form arm that IS failing publishes at any "
                  "question count.")
        else:
            # Opens with the shared `REASON_LABEL` rather than a hand-written phrase: this
            # paragraph was the one exclusion site that did not read from the shared
            # vocabulary, which is how it came to tell readers "too many calls went
            # unanswered" about backends that had answered every call (#332).
            out.append(
                f"**{REASON_HEADING.get(why or '', 'Not measured')}** — "
                f"{REASON_LABEL.get(why or '', str(why))}, so no accuracy is published for: "
                + ", ".join(f"`{m}` ({f}/{a} calls lost)" for m, f, a, _ in models)
                + "."
                # Chosen by the counts, not hardcoded: this paragraph is reachable with
                # ZERO calls lost (`_unmeasured` trigger 1), where every transport claim
                # in the #332 hedge is false. See `unmeasured_cause`.
                + unmeasured_cause(sum(f for _, f, _, _ in models)))
    out.append("")
    return out


# --------------------------------------------------------------------------- #
# Codec-tier material-preservation verdict (#295) — replaces a floating accuracy
# tolerance with a demonstrated-corruption gate. See `codeceval.py`'s module docstring
# for the full argument; the short version: any percentage tolerance is a budget for how
# much structural damage terse's *lossless* codec tier is allowed to cause at the reader,
# which contradicts the round-trip-proven losslessness claim one layer down. So this gate
# asks a different question than `_GAP_TOLERANCE` does — not "how big is the gap" but "was
# any corruption demonstrated, and if not, was there enough evidence to say so."
# --------------------------------------------------------------------------- #

# A sample-size floor for trusting an observed ZERO failures, not a tolerance for a
# nonzero one. Clopper-Pearson: n zero-failure trials bounds the true failure rate below
# `1 - 0.05 ** (1/n)` at 95% confidence. At n=20 that is ~14pp — loose, but this is the
# single most contestable number in this module; it wants explicit sign-off before it is
# trusted at scale, not a mechanical tuning pass. Raise it once real panels show it holding
# up, rather than lowering the bar to make an early run print SAFE. That sign-off is
# `test_the_codec_trial_floor_is_twenty` (#337), which pins the value; its siblings
# `test_nineteen_zero_failure_trials_are_UNRESOLVED` / `..._twenty_..._are_SAFE` bracket
# the floor with literal counts, and `test_the_quoted_clopper_pearson_bound_still_computes`
# reads the "~14pp" above back out of this file and recomputes it, so the number and the
# argument for it cannot drift apart. (The inclusive `>=` was already pinned, by
# `tests/test_codec_verdict.py`; #337 added the VALUE, which nothing held.)
_CODEC_MIN_TRIALS = 20

# Below this many questions, a fixed-ideal metric prints INSUFFICIENT instead of PASS
# (#335). A DISCLOSURE THRESHOLD, NOT A STATISTICAL FLOOR — deliberately, and the
# distinction is the whole design:
#
# `_MIN_PAIRED_QUESTIONS` and `_CODEC_MIN_TRIALS` are Clopper-Pearson numbers; each says
# what a zero-failure run bounds the true failure rate to. This one cannot be, because
# there is nothing to calibrate against. Measured 2026-08-26 on a live 1,524-payload
# capture corpus across 39 tools: **zero** payloads had a drop rule selected, so
# `gen_drop_questions` produced zero recall questions and this metric has never run on
# real data. Deriving a bound from a distribution that does not exist would be a
# fabricated justification, which is worse than an admitted convention (#337's lesson).
#
# So this does not WITHHOLD. The measured percentage and its question count are both
# published; what a thin sample cannot buy is the word PASS. A FAIL always publishes at
# any n — the asymmetry `_MIN_PAIRED_QUESTIONS`' comment argues for at length, because an
# exclusion must never be able to improve a verdict.
#
# Calibrate this into a real floor once a live policy actually configures a drop rule
# (#271, #273) and the metric produces its first real question count.
_FIXED_IDEAL_MIN_QUESTIONS = 5


def fixed_ideal_sufficient(n: int | None) -> bool:
    """Is `n` questions enough for a fixed-ideal metric to publish a PASS? (#335)

    THE predicate, in one place. The first cut of #335 downgraded only the BADGE, inside
    `_format_worst_case_line`, and review found the decision four lines below still reading
    `verdict.passed`: a 2-question run printed `**INSUFFICIENT**` and then "safe to enable
    drop-to-retrieve". A cosmetic fix that leaves the authorization intact is worse than
    none — it reads as handled. On the lattice there is only one decision point, so the
    predicate now feeds `Directive.INSUFFICIENT` and every renderer inherits it.

    `None` is NOT sufficient: a metric whose sample size could not be determined has not
    demonstrated one."""
    return n is not None and n >= _FIXED_IDEAL_MIN_QUESTIONS


_VERDICT_RANK = {"SAFE": 0, "UNRESOLVED": 1, "UNSAFE": 2}  # worst wins when grouping models


def codec_verdict(rows: list[dict[str, Any]]) -> tuple[str, ArmGap]:
    """SAFE / UNSAFE / UNRESOLVED for one model's codec-eval rows, already scoped to one
    `(tool, shape)` group (`codeceval.run_codec_fluency`'s row tags).

    Unlike every other verdict in this file, this one does NOT compare a gap to
    `_GAP_TOLERANCE`. `arm_gap` still does the pairing and exclusion-gating work (so a
    dead backend or an unpaired question set reports UNRESOLVED via the same `_unmeasured`/
    `paired_rows` machinery every other renderer uses) — but the pass/fail decision itself
    is a demonstrated-corruption gate: ANY paired excess of terse-arm misses beyond what the
    raw arm ALSO missed is UNSAFE, full stop, regardless of how small a fraction of the
    sample it is. Zero observed excess is SAFE only once `_CODEC_MIN_TRIALS` zero-failure
    trials have accumulated; short of that it is UNRESOLVED.

    Deliberately NOT `terse_ok < terse_trials` (raw code review, PR #302 F1) — that counts
    every terse miss as codec-caused, including one the model would have missed on raw too
    (a hard reconstruction, a prose-reply habit, anything about the MODEL rather than the
    FORM). Measured: a model scoring 80% on BOTH raw and terse — identical behavior, no
    demonstrated difference — printed UNSAFE under that definition. `max(0, raw_ok -
    terse_ok)` per row is this file's own established "regression" shape (`harnesses.py`'s
    module docstring: "a regression is a question the model got right on raw and wrong on
    terse — a stronger signal than two independent accuracy rates"), just kept at
    trial-count granularity instead of a per-row boolean, so it stays comparable to `n` for
    the Clopper-Pearson framing above."""
    # `min_paired=0` opts OUT of #334's paired-QUESTION floor, which would otherwise make
    # SAFE unreachable for this tier: a codec group is characteristically one question run
    # many times (`test_identical_partial_failure_on_both_arms_at_the_trial_floor_is_SAFE`
    # is a single row at 25 trials), so it would never reach 20 paired questions. This is
    # not an exemption from sample-size gating — `_CODEC_MIN_TRIALS` below is the same
    # Clopper-Pearson floor in the unit this verdict actually counts in. Layering a second
    # floor in a different unit would silently re-calibrate a tier that already decided
    # this question, without saying so anywhere near `_CODEC_MIN_TRIALS`.
    g = arm_gap(rows, "terse_ok", "raw_ok", min_paired=0)
    if g.excluded:
        return "UNRESOLVED", g
    n = sum(r.get("terse_trials", r.get("trials", 1)) for r in g.rows)
    excess_terse_misses = sum(max(0, int(r["raw_ok"]) - int(r["terse_ok"])) for r in g.rows)
    if excess_terse_misses > 0:
        return "UNSAFE", g
    if n >= _CODEC_MIN_TRIALS:
        return "SAFE", g
    return "UNRESOLVED", g


def _codec_savings_section(
    groups: dict[tuple[str, str], dict[str, list[dict]]],
) -> list[str]:
    """The economics, rendered BESIDE the verdict and never inside it (#303, #295 DoD 4).

    A SIBLING section over the same `(tool, shape)` groups — a peer `##` heading, not a
    `###` nested under the verdict, which is as close to "folded into it" as markdown gets
    (`test_the_savings_heading_is_a_sibling_not_a_subsection` pins the level, not just the
    text). Deliberately not a column of the verdict table and deliberately not combined with
    it by any arithmetic. The ordering is the argument: correctness is decided first, on its
    own table, and the savings number is what you consult AFTER it — because a cell that
    multiplies a saving by a verdict is how a savings argument ends up licensing a
    correctness loss.

    An UNSAFE group still prints its savings. Withholding it would be its own editorialising
    in the other direction: "this shape saves 61% and is UNSAFE" is the true and useful
    statement, and suppressing half of it does not make the tier safer, only less legible.
    The two facts are independent measurements of the same group; neither gates the other.

    Two ways a row can fail to contribute, and BOTH are disclosed rather than silently
    dropped — the sums must never be a subset presented as a total:

    - **No token counts** (`raw_tokens`/`terse_tokens` absent, `None`, a string, or a bool —
      `isinstance(True, int)` is `True`, so bools are excluded explicitly). A run with no
      tokenizer, or a stored result predating #303. Read as `0/0` these would print a
      perfect saving off a measurement that never happened.
    - **No usable `sha`.** De-duplication is BY `sha`, because `run_codec_fluency` stamps the
      same per-payload counts onto every question row a payload produces, for every model
      that answered it. A row without one cannot be attributed to a payload at all: summing
      it risks multiplying a payload's tokens by its question count, and defaulting it to a
      shared placeholder key (the first cut of this function used `str(r.get("sha", "?"))`)
      silently COLLAPSES every such payload in a group into whichever one was seen first —
      wrong count, wrong sums, wrong percentage, and no disclosure. Two rounds of review
      found this: the first found the placeholder here, the second found `run_codec_fluency`
      still emitting its own `env.get("sha", "?")` one call upstream, which walked straight
      past this guard because `"?"` is a non-empty `str`. Both are fixed. `capture.record`
      always writes `sha`, but `capture.load_corpus` does not REQUIRE it, so the reachable
      path is a foreign or hand-built corpus, not only a merged result file.

    The counted/uncounted split is per `(tool, shape)` group, not global, because the sums it
    qualifies are per-group: the same payload can be counted in one group and uncounted in
    another (reachable via the stored-`shape` drift of `#355`), and the note below says so."""
    out = [
        "## Savings by tool and shape",
        "",
        "Reported BESIDE the verdict above, never folded into it: no figure here is",
        "weighted by, multiplied into, or gated on a SAFE/UNSAFE/UNRESOLVED result, and an",
        "UNSAFE group still prints what it saves. Decide correctness from the table above",
        "first; this one only says what the tier costs or buys once that is settled.",
        "cl100k tokens, over the payloads the verdict above was computed on — raw payload vs",
        "the same compressed form the terse arm was fed. Whatever the sums could not cover is",
        "disclosed beneath the table, never quietly dropped — as payloads where the rows",
        "identify one, and as rows where they do not.",
        "",
        "| Tool | Shape | payloads | raw tok | terse tok | saved | % |",
        "|---|---|---|---|---|---|---|",
    ]
    uncounted_total = 0
    unattributable_total = 0
    # `sorted` on BOTH loops, and it is the same expression as the verdict table's (#303
    # review): two tables a reader is asked to line up row-for-row must not order their rows
    # independently. `test_both_tables_list_their_groups_in_the_same_order` holds the pair.
    for (tool, shape), by_model in sorted(groups.items()):
        counted: dict[str, tuple[int, int]] = {}
        uncounted: set[str] = set()
        for mrows in by_model.values():
            for r in mrows:
                sha = r.get("sha")
                if not isinstance(sha, str) or not sha:
                    unattributable_total += 1   # counted in ROWS: without a sha there is no
                    continue                    # payload identity to count in
                raw_t, terse_t = r.get("raw_tokens"), r.get("terse_tokens")
                if _is_token_count(raw_t) and _is_token_count(terse_t):
                    # First-wins. Every row of a payload carries the same two counts, so the
                    # choice is only visible for a merged result set whose runs disagree —
                    # in which case the EARLIER run's measurement is the one already cited
                    # in whatever report it produced, and re-reading it keeps the two
                    # agreeing rather than silently superseding one with the other.
                    counted.setdefault(sha, (raw_t, terse_t))
                else:
                    uncounted.add(sha)
        uncounted -= set(counted)
        uncounted_total += len(uncounted)
        if not counted:
            out.append(f"| `{tool}` | {shape} | 0 | n/a | n/a | n/a | n/a |")
            continue
        raw = sum(v[0] for v in counted.values())
        cmp_ = sum(v[1] for v in counted.values())
        out.append(f"| `{tool}` | {shape} | {len(counted)} | {raw} | {cmp_} | "
                   f"{raw - cmp_:+d} | {_pct(raw - cmp_, raw)} |")
    out.append("")
    if uncounted_total:
        out += [
            f"{uncounted_total} payload(s) — counted once per `(tool, shape)` group, so a "
            "payload measured in one",
            "group can still be uncounted in another — carry no token counts and are "
            "excluded from the",
            "sums above (no tokenizer available at run time, or a result file predating "
            "`#303`).",
            "Excluded, not counted as zero.",
            "",
        ]
    if unattributable_total:
        out += [
            f"{unattributable_total} row(s) carry no `sha` and are excluded from the sums "
            "above: without a payload",
            "identity they cannot be de-duplicated, and counting them would multiply a "
            "payload's tokens by",
            "its question count. `capture.record` always writes one, so this indicates a "
            "merged or foreign",
            "result file rather than a corpus this build captured.",
            "",
        ]
    return out


def _is_token_count(v: object) -> TypeGuard[int]:
    """A usable token count: a real, non-negative `int`. Never a `bool`, a string spelling of
    one, or a negative sentinel.

    Three exclusions, each for a different reason, and none reachable from `_payload_tokens`
    (which emits `int` or nothing) — all three are reachable from a hand-written or foreign
    result file:

    - **`bool`**: `isinstance(True, int)` is `True` in Python, so a JSON `true` would sum as
      one token and render a saving. A surviving mutation in #303's second review.
    - **`str`**: `"1000"` would reach `sum()` and `{:+d}` and raise. The first cut already
      excluded this via a bare `isinstance(v, int)`; it is pinned here so a later widening
      of the check (`v is not None`) cannot reintroduce it, NOT because it ever shipped
      broken. An earlier version of this docstring claimed it had, which was wrong.
    - **negative**: a token count cannot be negative, and `_pct` has no guard for a negative
      BASE — only for a zero one. Executed: `raw=-1, terse=1` renders `-2` saved at
      `+200.0%`, a saving reported for a payload that expanded. Excluding it here is the
      targeted fix; `_pct`'s zero-guard is shared with six other renderers and is not
      widened from this call site."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def build_codec_verdict_report(results: dict[str, list[dict]]) -> str:
    """Render the codec-tier material-preservation eval, grouped by `(tool, shape)` — never
    as one global number (#295's explicit non-goal). `results` is
    `{model: [row, ...]}` from `codeceval.run_codec_fluency`; each row carries `tool` and
    `shape` tags plus the same `<form>_ok`/`<form>_trials` convention every other harness
    here uses.

    No existing renderer groups by `(tool, shape)` — `build_dropeval_report` and
    `build_fluency_report` both pool a model's rows across the whole corpus. Per-shape
    grouping matters here because the product ships per-tool policy; a single global verdict
    can't answer "compress THIS shape, for THIS tool, by default?", which is the question
    #295 says the eval must be able to answer.

    Token savings render as a SIBLING section over the same groups (`_codec_savings_section`,
    #303), never as a column of the verdict table and never combined with a verdict by any
    arithmetic — see that function for why the ordering is the argument."""
    out = ["# terse codec-tier material-preservation eval", ""]
    out += [
        "Does a real tool-calling model's downstream tool-call argument stay structurally",
        "identical whether it read raw JSON or terse's compressed form? Scored on `deref`",
        "questions only — reconstructing an aliased or table-encoded value back into the",
        "original structure, which is what an agent does when it feeds a result into the",
        "next tool call, PAIRED against the same question answered from raw. No percentage",
        "tolerance: any trial where raw succeeded and terse did not is UNSAFE; a clean run",
        "needs enough trials to trust the zero, or it is UNRESOLVED.",
        "",
    ]
    if not results or not any(results.values()):
        out += [
            "No tool-capable model answered, or no `deref`-eligible payloads in the corpus",
            "(needs a record-shaped payload with a whole object/array column). Configure a",
            "model and re-run `terse fluency --codec-verdict`.",
            "",
        ]
        return "\n".join(out)

    groups: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for model, rows in results.items():
        for r in rows:
            key = (str(r.get("tool", "?")), str(r.get("shape", "unknown")))
            groups.setdefault(key, {}).setdefault(model, []).append(r)

    out += [
        "## Verdict by tool and shape",
        "",
        "| Tool | Shape | n | Verdict | Worst model | Why |",
        "|---|---|---|---|---|---|",
    ]
    for (tool, shape), by_model in sorted(groups.items()):
        # Gates on the worst model, not the mean — the same principle every other verdict
        # in this file follows (#24): a shape that's unsafe for one model in the fleet is
        # unsafe, full stop. `_VERDICT_RANK` orders UNSAFE worst, SAFE best; ties keep the
        # first model encountered in sorted order rather than an arbitrary last-wins.
        worst_verdict, worst_model, worst_gap = "SAFE", "", None
        for model, mrows in sorted(by_model.items()):
            v, g = codec_verdict(mrows)
            if worst_gap is None or _VERDICT_RANK[v] > _VERDICT_RANK[worst_verdict]:
                worst_verdict, worst_model, worst_gap = v, model, g
        assert worst_gap is not None  # by_model is never empty — every group has >=1 model
        n = sum(r.get("terse_trials", r.get("trials", 1)) for r in worst_gap.rows)
        if worst_verdict == "UNSAFE":
            excess = sum(max(0, int(r["raw_ok"]) - int(r["terse_ok"]))
                        for r in worst_gap.rows)
            why = (f"{excess} trial(s) where raw succeeded and terse did not "
                  f"(raw {worst_gap.control_acc:.0%}, terse {worst_gap.form_acc:.0%})")
        elif worst_gap.excluded:
            why = REASON_LABEL.get(worst_gap.excluded, worst_gap.excluded)
        elif worst_verdict == "UNRESOLVED":
            why = f"only {n} zero-failure trial(s), need {_CODEC_MIN_TRIALS}"
        else:
            why = f"{n} zero-failure trials"
        out.append(f"| `{tool}` | {shape} | {n} | **{worst_verdict}** | `{worst_model}` | "
                   f"{why} |")
    out.append("")
    out += _codec_savings_section(groups)
    return "\n".join(out)


class GapVerdict(NamedTuple):
    model: str
    gap: float
    form_acc: float
    control_acc: float
    gap_ci: float
    passed: bool


def _worst_case_gap(
    rows: dict[str, tuple[float, float, float, float]], tol: float = _GAP_TOLERANCE
) -> GapVerdict | None:
    """Shared verdict-gating math for both fluency-style reports — principle #24, gate on
    the worst model, never the mean. `rows` maps model to a 4-tuple of form_acc, form_se,
    control_acc, control_se. Returns the model with the lowest gap as a GapVerdict, or
    None if `rows` is empty. gap = form_acc minus control_acc; gap_ci is the 95%
    half-width of the independence-combined question-clustered (sandwich) SE of the two
    arms; passed iff gap is at least -tol, inclusive of the boundary. Callers access
    fields by name, e.g. verdict.form_acc, never by position, so a future field reorder
    can't silently swap values."""
    worst = None  # (model, gap, facc, cacc, gap_ci) — cheapest to track positionally here;
    for model, (facc, fse, cacc, cse) in rows.items():  # this is a private local, not the
        gap = facc - cacc                               # public interface callers rely on.
        # sqrt(fse^2+cse^2) assumes the two arms' SEs are independent, which overstates
        # the true paired-on-question variance FOR ANY PAIR OF ARMS THAT ARE POSITIVELY
        # CORRELATED — the normal case, since question difficulty is shared across arms.
        # (It is not a universal upper bound: two arms that are anti-correlated on which
        # questions they get right would make this an UNDER-estimate instead — just not a
        # shape real comprehension arms take.) `passed` below does NOT read gap_ci (it
        # gates on the point estimate `gap` alone), so this looseness cannot flip the
        # merge gate itself; every consumer of `gap_ci` (`build_fluency_report`'s worst-
        # case prose here, `_format_worst_case_line`'s "±N pts" headline shared by the
        # diff/diff-soak/dropeval/fluency reports, the diff-soak deepest-slice line, and
        # `build_html_diff_report`'s verdict banner) renders it as a width, never as a
        # significance test — note this is NOT the HTML forest plot's whiskers, which read
        # per-arm `form_ci`/`control_ci`, not `gap_ci`, at all —
        # which is why `build_fluency_report` is written to never let a wide gap_ci argue
        # AWAY a verdict: a bound that is only reliably an over-estimate can rule a gap
        # "not yet tightly measured", never "not real". #297 made both SEs cluster-robust
        # instead of ~0, so this bound is now wide enough to matter in practice, not just
        # a formality. A tighter, correct fix is a paired cluster-robust SE on the
        # per-question DIFFERENCE rather than combining two independent arm SEs; tracked
        # as a follow-up rather than folded into #297's scope.
        gap_ci = _ci(math.sqrt(fse ** 2 + cse ** 2))
        if worst is None or gap < worst[1]:
            worst = (model, gap, facc, cacc, gap_ci)
    if worst is None:
        return None
    model, gap, facc, cacc, gap_ci = worst
    passed = passes_tolerance(gap, tol)
    return GapVerdict(model, gap, facc, cacc, gap_ci, passed)


def _format_worst_case_line(verdict: GapVerdict, tol: float, form_label: str, control_label: str) -> str:
    return (f"- Worst-case model `{verdict.model}`: {form_label} {verdict.form_acc:.0%} vs "
            f"{control_label} {verdict.control_acc:.0%} (gap {verdict.gap:+.0%} "
            f"±{verdict.gap_ci * 100:.0f} pts). **{'PASS' if verdict.passed else 'FAIL'}** "
            f"at {tol:.0%} tolerance.")


def diff_gap_rows(results: dict) -> tuple[dict[str, tuple[float, float, float, float]],
                                          dict[str, ExclusionReason | None]]:
    """(form=diff_ok, control=terse_ok) gap-row tuples per model — the same shape
    `_worst_case_gap` and the bar-chart renderers (html/terminal) consume, computed
    once here so a chart's gap can never read differently than build_diff_report's.
    Returns (gap_rows, excluded_model_names), mirroring `fluency_gap_rows`.

    "Can never read differently" only holds if this applies the SAME exclusions the
    markdown does. It did not: when `_build_diff_style_report` gained the transport-
    failure gate (#264), a dead backend rendered `n/a` in the table and `NO VERDICT` in
    the markdown while the forest plot printed directly beneath it still drew a FAIL bar
    off the same rows. `cli` prints both together for all three diff paths (`--diff`,
    `--diff-soak`, `--text-diff-eval`), and the chart is what a reader sees first."""
    out: dict[str, tuple[float, float, float, float]] = {}
    excluded: dict[str, ExclusionReason | None] = {}
    for model, rows in results.items():
        if not rows:
            continue
        g = arm_gap(rows, "diff_ok", "terse_ok")
        if g.excluded:
            # The REASON, not just the name. The terminal plot used to render every
            # exclusion as "calls went unanswered", which is false for an `unpaired` model.
            excluded[model] = g.excluded
            continue
        out[model] = (g.form_acc, g.form_se, g.control_acc, g.control_se)
    return out, excluded


def fluency_gap_rows(results: dict) -> tuple[dict[str, tuple[float, float, float, float]],
                                             dict[str, ExclusionReason | None]]:
    """(form=best of terse/primer, control=raw) gap-row tuples per model, for the bar-
    chart renderers. Excludes any model whose raw control failed (0% — a backend/config
    error, not a comprehension result) AND any model too degraded by transport failures to
    measure, matching `build_fluency_report`'s gate. Returns (gap_rows, excluded_names).

    The second exclusion is not decoration. `cli` prints the markdown report and the
    terminal forest plot together, and this function feeds only the plot: without it a
    rate-limited model rendered `n/a` in the table and "not measured" in the verdict while
    the chart immediately below plotted its depressed gap as a red FAIL bar — the exact
    false verdict #263 exists to kill, surviving in the renderer the reader looks at
    first."""
    out: dict[str, tuple[float, float, float, float]] = {}
    broken: dict[str, ExclusionReason | None] = {}
    for model, rows in results.items():
        if not rows:
            continue
        g = best_arm_gap(rows, ["terse_ok", "primer_ok"], "raw_ok")
        if g.excluded:
            broken[model] = g.excluded
            continue
        out[model] = (g.form_acc, g.form_se, g.control_acc, g.control_se)
    return out, broken


def inconclusive_models(results: dict) -> dict[str, tuple[int, int]]:
    """{model: (failed_calls, attempts)} for models whose calls mostly did not reach the
    backend. A failed call scores identically to a model that declined to retrieve, so past
    this threshold the accuracy columns are counting transport errors and no verdict may be
    rendered from them. Lives beside `dropeval_gap_rows` and for the same reason: the
    markdown verdict and the terminal chart must never disagree about whether a run counts.
    """
    out: dict[str, tuple[int, int]] = {}
    for model, rows in results.items():
        errs = sum(r.get("errors", 0) for r in rows)
        # `attempts` where the row carries one: with a control arm a question costs two
        # calls, and dividing two arms' failures by one arm's trial count would report a
        # doubled failure rate and withhold verdicts from runs that are fine (#269).
        attempts = sum(r.get("attempts", r.get("trials", 1)) for r in rows)
        if errs and attempts and errs * 2 >= attempts:
            out[model] = (errs, attempts)
    return out


def _accuracy_gate(rows: list[dict[str, Any]]) -> ArmGap:
    """final-accuracy as a gap against the MEASURED no-drop control arm (#269).

    Recall and no-overfetch are correctly gated against a fixed 100%: a tool call either
    happens or it does not, so 100% IS the target. final-accuracy is not like that — it is
    JSON value-equality against the full original value, and the fields it runs on are
    500+ character prose. A model handed the UN-dropped payload does not reproduce those
    verbatim either; it paraphrases. Gating that against a perfect ideal measured
    verbatim-reproduction ability and billed the shortfall to the drop, which is what
    blocked drop-to-retrieve on a 54% that had little to do with dropping (#269).

    So when rows carry `control_ok` — the same questions asked against the same payload
    with the drop rule stripped — the control becomes that measured arm. `paired_rows`
    restricts both arms to the questions that completed every trial on BOTH sides, for the
    same reason the fluency path does it (#280): dropping an incomplete question from one
    arm's denominator only is safe while losses are uncorrelated with the arm, and here
    they are not — the treatment arm runs two turns to the control's one, so it fails
    first.

    When no control ran the metric is **excluded, not defaulted back to the fixed ideal**.
    Gating a verbatim-reproduction score against an unrun 100% is the defect; silently
    reproducing it for older packs would keep emitting the false FAIL this exists to
    remove. Recall and no-overfetch still gate normally — their fixed ideal is correct —
    so a `--no-control` run still produces a verdict, just not one about final accuracy.

    Routing through `arm_gap` (rather than two `_form_stats` calls) is what
    `tests/test_gap_gate_boundary.py` requires of every form-vs-control gap, and its
    allowlist comment names this issue as the reason dropeval was temporarily exempt."""
    with_control = sum("control_ok" in r for r in rows)
    if with_control == 0:
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "no control arm")
    if with_control != len(rows):
        # `any()` would activate the metric on a mixed result set and let `paired_rows`
        # silently discard every control-less row — a verdict quietly computed over a
        # subset the reader was never told about. A merged/legacy/partially-failed run is
        # exactly when that is most likely and least visible.
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "partial control coverage")
    return arm_gap(rows, "answer_ok", "control_ok")


def dropeval_gap_rows(results: dict) -> tuple[dict[str, dict[Metric, tuple[float, float, float, float]]],
                                              dict[str, ExclusionReason]]:
    """(gap_rows, accuracy_excluded) — the `diff_gap_rows`/`fluency_gap_rows` shape, now a
    projection of `dropeval_verdict` rather than a second copy of its math.

    It used to compute the per-model gates itself, alongside a third copy inside
    `build_dropeval_report`'s table loop, with a docstring promising the two verdicts "can
    never disagree". Kept as a function because it is the shape the other two gap-row
    helpers expose and `tests/test_gap_gate_boundary.py` allowlists by name; the verdict
    itself now travels as a `DropevalVerdict`, which carries the exclusion reasons the
    renderers used to re-derive.

    `out` is never itself empty for an excluded model — recall and precision still render;
    only the "accuracy" key is missing, and `accuracy_excluded` names why."""
    v = dropeval_verdict(results)
    return v.gates, v.metrics["accuracy"].excluded


# --------------------------------------------------------------------------- #
# The dropeval verdict: decided once, rendered twice.
# --------------------------------------------------------------------------- #


class Directive(IntEnum):
    """What the dropeval report tells an operator to DO, as a lattice (#342).

    ORDERED BY STRICTNESS, and the order is the whole mechanism: the verdict is
    `max()` over the per-metric outcomes, never an `if/elif` chain over them. Four review
    rounds on this path produced four separate precedence bugs — an absent arm outranking
    a measured `-100%` FAIL, a branch added in one round shadowing the branch below it —
    and every one of them is a question about which arm of a chain runs first. `max()` has
    no arms. `BLOCK` is the top, so a demonstrated regression is never displaced by a
    missing measurement, whichever order the metrics happen to be visited in.

    `SHIP` is the bottom, which is the safety-relevant half: it is reached only when NO
    metric contributed anything else, so a model that left a gate — for want of a control
    arm, for a control that failed, for too few paired questions — raises the verdict off
    `SHIP` by construction instead of silently vanishing from `_worst_case_gap` and leaving
    the survivors to authorize the ship. That is #344's critical finding, and
    `tests/test_dropeval_monotonicity.py` is the property that pins it.
    """

    SHIP = 0
    INSUFFICIENT = 1
    NOT_CONCLUDED = 2
    BLOCK = 3


# The three gates. `Metric` is a `Literal` for the same reason `ExclusionReason` is: these
# keys index the gate dicts every renderer reads, and a typo'd key used to be a silent
# missing bar. The label and control-label ride along here rather than being re-typed in
# each renderer — "vs ideal (100%)" and "vs no-drop control" are DIFFERENT CLAIMS, and #269
# exists because a reader could not tell which one the verdict was making.
Metric = Literal["recall", "precision", "accuracy"]

DROPEVAL_METRICS: tuple[tuple[Metric, str, str], ...] = (
    ("recall", "retrieve-recall", "ideal (100%)"),
    ("precision", "no-overfetch", "ideal (100%)"),
    ("accuracy", "final-accuracy", "no-drop control"),
)


def _reason_directive(reason: ExclusionReason) -> Directive:
    """Where a withheld model lands on the lattice. TOTAL over `ExclusionReason`.

    `assert_never` is the point: add a ninth reason and mypy fails HERE, at the consumer,
    rather than letting it fall through to whatever the last `elif` happened to be.

    The `INSUFFICIENT`/`NOT_CONCLUDED` split is the one #334 drew: "underpowered" means the
    backend answered, both arms paired, and there were simply too few questions for an
    absence of regressions to mean anything — a measurement that does not reach. Every
    other reason means the comparison was never made at all. Both refuse to authorize;
    they differ in what the operator has to go fix, and `_accuracy_remedy` is where that
    difference is spent."""
    match reason:
        case "underpowered":
            return Directive.INSUFFICIENT
        case ("unmeasured" | "unpaired" | "broken control" | "empty"
              | "no control arm" | "partial control coverage" | "not a diff run"):
            return Directive.NOT_CONCLUDED
    assert_never(reason)


def _exclusion_remedy(reason: ExclusionReason) -> str:
    """What the operator has to do to turn a withheld metric into a gated one.

    Keyed on the REASON, not the metric — but every reason here is about a paired
    form-vs-control comparison, and final-accuracy is dropeval's only metric that has one.
    `dropeval_exclusion_bullets` nevertheless loops all three, so a recall exclusion (which
    `test_only_accuracy_is_ever_withheld` says cannot happen) would render a control-arm
    remedy under a metric with no control arm. That is the lesser of two evils on purpose:
    the alternative — looping only "accuracy" — would make a future recall exclusion vanish
    from the report entirely, and a wrong-but-visible sentence is recoverable where a
    silent omission is the #300-finding-5 defect this whole path exists to prevent.

    TOTAL over `ExclusionReason` via `assert_never`, and that totality is load-bearing
    rather than tidy: the verdict used to pick its remedy sentence with a set-equality test
    over the exclusion reasons present, so a MIXED set fell through to whichever sentence
    the test did not match — telling an operator to switch on a control arm that was
    already running (#300 finding 2), and, once both halves were handled, printing both
    sentences joined into a single self-contradicting bullet. Rendering one bullet PER
    REASON, each naming its own models, is what makes that unconstructible; this function
    is the per-reason half of it."""
    match reason:
        case "no control arm":
            return ("It is scored by JSON value-equality against the full original value, "
                    "and a model given the UN-dropped payload does not reproduce a long "
                    "prose field verbatim either — gating that against an unrun 100% "
                    "measures verbatim reproduction and bills it to the drop (#269). "
                    "Re-run without `--no-control` to gate it.")
        case "underpowered":
            return ("The control arm ran; there were simply too few paired questions for "
                    "an absence of drop-caused loss to mean anything. Generate more "
                    "questions and re-run — this is not evidence either way.")
        case "broken control":
            return ("The control answers a question with the value sitting verbatim in "
                    "its payload, so a 0% control is a grader or backend fault, not a "
                    "blameless drop (#300). Fix the control arm and re-run.")
        case "partial control coverage":
            return ("Only some rows carry a control arm, so scoring the metric would "
                    "compute it over a subset the reader was never told about. Re-run the "
                    "whole question set with the control on rather than merging packs.")
        case "unmeasured" | "unpaired":
            # This sentence asserts NOTHING about the present run, and that is the whole
            # design. The other five exclusion sites pick their cause from the loss count
            # (`unmeasured_cause`); this one is handed a reason and no counts —
            # `dropeval_exclusion_bullets` receives a `DropevalVerdict`, which carries no
            # per-arm failure totals — so it teaches the reader to read the split that is
            # printed directly above instead. It previously said "Fix the backend and
            # re-run" unconditionally, which neither reason licenses: `unpaired` means
            # literally "no question completed on both arms" (so the backend answered),
            # and `unmeasured` fires at zero calls lost whenever an arm completed no
            # trials (#338). Threading the counts in here is the better end state and is
            # deliberately NOT done in this change: it means widening `DropevalVerdict`,
            # which is a wider blast radius than the false sentence justifies.
            return ("Too few calls completed on BOTH arms to compare. Read the per-arm "
                    "failure split above: a non-zero loss there is a transport problem "
                    "and the run needs repeating; a zero means an arm completed no trials "
                    "at all, so check that every arm named actually ran.")
        case "empty":
            return ("No rows of this kind were scored for this model — the pack carries "
                    "none, or a merged run lost them. Re-generate the question set for "
                    "this model before reading anything into the metrics that did run.")
        case "not a diff run":
            return "These rows carry no diff arm, so this gate does not apply to them."
    assert_never(reason)


class MetricVerdict(NamedTuple):
    """One gate's outcome: the worst SCORED model, who was withheld and why, who was
    measured on too thin a sample to buy a PASS, and where all three put the metric on the
    lattice."""
    metric: Metric
    worst: GapVerdict | None
    excluded: dict[str, ExclusionReason]
    # model -> question count, for models scored on fewer than `_FIXED_IDEAL_MIN_QUESTIONS`
    # questions whose gap nonetheless CLEARS tolerance (#335). Deliberately NOT an
    # exclusion: the measured percentage and its question count are both published, and the
    # model stays in `gates` so the table and the chart still show it. What a thin sample
    # cannot buy is the word PASS.
    #
    # Only ever populated for a would-be PASS, which is what keeps the #335 floor from
    # being able to improve a verdict — the same asymmetry `_MIN_PAIRED_QUESTIONS` argues
    # for, and the reason a demonstrated FAIL publishes at any n.
    thin: dict[str, int]
    directive: Directive


class DropevalVerdict(NamedTuple):
    """Everything both renderers need, decided once.

    `dropeval_gap_rows`' docstring has always promised that the markdown and the terminal
    chart "can never disagree". Before #342 that was a promise about two functions reading
    the same NUMBERS while deciding their own verdicts, and they disagreed three times
    across four review rounds — on badge scope, on exclusion notes, and on whether a
    demonstrated FAIL deserved a thin-sample caveat. Here the decision itself is the shared
    value, so a disagreement is not caught by a test, it is unconstructible."""
    directive: Directive
    metrics: dict[Metric, MetricVerdict]
    # model -> metric -> (form_acc, form_se, control_acc, control_se), for the table and
    # the forest plots. "accuracy" is ABSENT rather than zeroed for a withheld model, so a
    # renderer that iterates what it finds cannot draw a bar for a gap nobody measured.
    gates: dict[str, dict[Metric, tuple[float, float, float, float]]]
    # Transport failure past the INCONCLUSIVE threshold. Non-empty means the run measures
    # the harness, not the model, and no behavioral claim survives.
    inconclusive: dict[str, tuple[int, int]]
    # The same models, when `--accept-degraded` moved them out of `inconclusive`. Recorded
    # rather than silently honoured: the operator's "the cause was model-independent" is a
    # claim the harness cannot verify.
    degraded_accepted: dict[str, tuple[int, int]]


def dropeval_verdict(results: dict, accept_degraded: bool = False) -> DropevalVerdict:
    """Decide the drop-to-retrieve verdict. Pure: no strings, no formatting, no I/O.

    Split out from rendering (#342) for two reasons. The first is that markdown and chart
    now consume one decision instead of re-deriving two. The second is that a pure function
    over a small input space can be swept EXHAUSTIVELY in seconds — which is how
    `tests/test_dropeval_monotonicity.py` covers a cross product that 29 hand-written
    examples left a hole in, and how the same sweep can be a gate rather than a review."""
    gates: dict[str, dict[Metric, tuple[float, float, float, float]]] = {}
    excluded_by_metric: dict[Metric, dict[str, ExclusionReason]] = {
        m: {} for m, _, _ in DROPEVAL_METRICS}
    thin_by_metric: dict[Metric, dict[str, int]] = {m: {} for m, _, _ in DROPEVAL_METRICS}
    for model, rows in results.items():
        if not rows:
            # WITHHELD, not skipped. `continue` here dropped the model out of `gates` AND
            # out of every `excluded` dict, so `max()` never saw it and the survivors
            # decided the verdict alone — #344's critical shape, routing around the lattice
            # instead of through it. Measured: of the 144 two-model fleets that do not ship,
            # 22 started shipping when one model's rows were emptied, and the CLI's policy
            # instruction said "the verdict authorizes it". `"empty"` was already in
            # `ExclusionReason` with a directive and a remedy; nothing could reach them.
            for m, _, _ in DROPEVAL_METRICS:
                excluded_by_metric[m][model] = "empty"
            continue
        # Recall/precision keep the fixed 100% ideal — for them it IS the right control, a
        # tool call either happens or it doesn't. Accuracy pairs against the measured
        # control arm when one ran (#269); `_accuracy_gate` owns that choice so the table
        # and the verdict cannot disagree about which control they used.
        #
        # A model with NO rows of a kind is WITHHELD from that gate, not scored at 0%.
        # `_form_stats([], f)` is `(0.0, 0.0)`, which against the fixed 100% ideal reads as
        # a `-100%` gap and publishes `**FAIL** ... keep drop-to-retrieve off` — a BLOCK
        # whose failing evidence does not exist, printed on the same line as `recall q` of
        # literally 0. Reached by a precision-only pack, a merged run, or a generator that
        # emitted one kind. It is the mirror of the empty-`rows` bug above: both substituted
        # a number for an absence and routed around the lattice rather than through it.
        gates[model] = {}
        by_kind: dict[Metric, list[dict[str, Any]]] = {
            "recall": [r for r in rows if r["kind"] == "recall"],
            "precision": [r for r in rows if r["kind"] == "precision"],
        }
        for mech in ("recall", "precision"):
            kind_rows = by_kind[mech]
            if not kind_rows:
                excluded_by_metric[mech][model] = "empty"
                continue
            acc, se = _form_stats(kind_rows, "retrieve_ok")
            gates[model][mech] = (acc, se, 1.0, 0.0)
            # #335. Recall and no-overfetch gate against a FIXED 100% ideal, so they never
            # pair, so `_MIN_PAIRED_QUESTIONS` — which counts PAIRED questions — never
            # applied to them at all. A one-question recall run printed `100% ±0 pts
            # **PASS**` and `safe to enable drop-to-retrieve`: maximum confidence off a
            # single question, with the `±0` not a rounding artifact but the exact SE of a
            # sample that is all-right or all-wrong.
            if not fixed_ideal_sufficient(len(kind_rows)) and passes_tolerance(acc - 1.0):
                thin_by_metric[mech][model] = len(kind_rows)
        g = _accuracy_gate(rows)
        if g.excluded:
            excluded_by_metric["accuracy"][model] = g.excluded
        else:
            gates[model]["accuracy"] = (g.form_acc, g.form_se, g.control_acc, g.control_se)

    inconclusive = inconclusive_models(results)
    degraded_accepted: dict[str, tuple[int, int]] = {}
    if inconclusive and accept_degraded:
        inconclusive, degraded_accepted = {}, inconclusive

    metrics: dict[Metric, MetricVerdict] = {}
    for metric, _, _ in DROPEVAL_METRICS:
        scored = {model: g[metric] for model, g in gates.items() if metric in g}
        worst = _worst_case_gap(scored)
        excluded = excluded_by_metric[metric]
        # The lattice. A withheld model contributes its reason's directive; the worst
        # SCORED model contributes SHIP or BLOCK. `max()` over both — so a withheld model
        # can only ever make the verdict stricter, and a demonstrated FAIL is never
        # displaced by one. `default` covers a metric no model reached at all.
        thin = thin_by_metric[metric]
        outcomes = [_reason_directive(r) for r in excluded.values()]
        # A thin would-be PASS joins the lattice as INSUFFICIENT rather than being removed
        # from the gate. `max()` then does the asymmetry for free: if any model actually
        # FAILS, `worst` contributes BLOCK and outranks every one of these.
        outcomes += [Directive.INSUFFICIENT] * len(thin)
        if worst is not None:
            outcomes.append(Directive.SHIP if worst.passed else Directive.BLOCK)
        # `default` fires for an empty fleet — no model scored, none withheld, nothing
        # known. It must not be SHIP, and it is the one place in this function where that
        # is a choice rather than a consequence, so
        # `test_an_empty_fleet_is_not_concluded_rather_than_shipped` pins it: both renderers
        # stop before reading it today, and a value nothing reads today is read by the next
        # consumer.
        metrics[metric] = MetricVerdict(
            metric, worst, excluded, thin, max(outcomes, default=Directive.NOT_CONCLUDED))

    # A dead backend is NOT_CONCLUDED outright rather than `max()`-ed with the metrics: the
    # numbers a half-failed run produces are counting transport errors, so promoting one of
    # them to BLOCK would be asserting a behavioral conclusion from a broken harness — the
    # exact over-claim `test_report_refuses_a_verdict_when_the_calls_failed` pins. It is
    # still non-authorizing, which is what the monotonicity property requires.
    # No `default=`: `metrics` has exactly one entry per `DROPEVAL_METRICS`, always, so an
    # empty `max()` here is unreachable — and a `default` on an unreachable path is a value
    # no test can pin and no reader can check. If that loop ever stops running, the
    # `ValueError` is the correct outcome.
    directive = (Directive.NOT_CONCLUDED if inconclusive
                 else max(mv.directive for mv in metrics.values()))
    return DropevalVerdict(directive, metrics, gates, inconclusive, degraded_accepted)


def _by_reason(excluded: dict[str, ExclusionReason]) -> list[tuple[ExclusionReason, list[str]]]:
    """Models grouped by why they were withheld, so one bullet can be emitted per reason
    NAMING ITS OWN MODELS. Joining remedies across reasons instead is how the verdict came
    to tell an operator, in one sentence, both to switch the control arm on and that it was
    already on."""
    by: dict[ExclusionReason, list[str]] = {}
    for model, reason in sorted(excluded.items()):
        by.setdefault(reason, []).append(model)
    return sorted(by.items())


def dropeval_exclusion_bullets(v: DropevalVerdict) -> list[str]:
    """The `not gated` / `not concluded` bullets, one per (metric, reason), shared by both
    renderers. Thin samples get their own bullet because they are a different fact: the
    model WAS measured and the number IS published — see `MetricVerdict.thin`."""
    out = []
    for metric, label, _ in DROPEVAL_METRICS:
        for reason, models in _by_reason(v.metrics[metric].excluded):
            names = ", ".join(f"`{m}`" for m in models)
            out.append(f"- **{label}: not gated for {names}** — {REASON_LABEL[reason]}. "
                       + _exclusion_remedy(reason))
        thin = v.metrics[metric].thin
        if thin:
            names = ", ".join(f"`{m}` ({n} question{'' if n == 1 else 's'})"
                              for m, n in sorted(thin.items()))
            out.append(
                f"- **{label}: measured, not concluded for {names}** — fewer than "
                f"{_FIXED_IDEAL_MIN_QUESTIONS} questions (#335). The percentage above is "
                "real and is not withheld; what this many questions cannot buy is the word "
                "PASS, because a fixed-ideal metric never pairs and so never met the "
                "paired-question floor the other gates use. Generate more questions and "
                "re-run. A FAIL still publishes at any n.")
    return out


def dropeval_next_step_line(v: DropevalVerdict) -> str:
    """What `terse tune --drop-eval` should tell the operator to do with the policy.

    A separate sentence from `dropeval_directive_line` because it is about the POLICY FILE,
    not the measurement — but it reads the same `directive`, and that is the point. `cli`
    used to print "If the worst-case model PASSES, enable the verified fields" and leave the
    reader to apply that rule by eye to the three worst-case lines. Those lines report the
    worst SCORED model, so a fleet with one model scored and one withheld prints three
    `**PASS**` headlines under a verdict that authorizes nothing — and the reader following
    the instruction enables the drops the report just refused. That is #344's critical
    finding wearing a renderer's clothes, and no property could catch it here, because the
    sentence was not derived from the verdict at all."""
    if v.directive is Directive.SHIP:
        return ("The verdict authorizes it: enable the verified fields by renaming that "
                "tool's '_suggested_fields' -> 'fields' in the policy.")
    return ("The verdict does NOT authorize enabling these drops — see the last bullet "
            "above for what is missing. Leave '_suggested_fields' as it is until it does.")


def dropeval_directive_line(v: DropevalVerdict) -> str:
    """The one sentence that says what to do. TOTAL over `Directive` via `assert_never`.

    Both renderers print this string, so the chart and the markdown cannot reach opposite
    conclusions about the same run — which they did three times across #335's review
    rounds, back when each of them decided for itself."""
    if v.inconclusive:
        # Handled HERE, not left to the callers. `dropeval_verdict` forces NOT_CONCLUDED
        # for a dead backend regardless of what the metrics say, so a caller that printed
        # this line without its own early return would have got "recall and no-overfetch
        # clear tolerance for the worst model" over a fleet whose recall metric is BLOCK.
        # Both renderers do early-return today, which made the old comment ("reachable ONLY
        # with recall and precision passing") true of the CALLERS and false of the
        # function — and `dropeval_next_step_line` is already a third consumer.
        return ("**INCONCLUSIVE** — " + ", ".join(
            f"`{m}` failed {e}/{a} model calls" for m, (e, a) in sorted(v.inconclusive.items()))
            + ". Fix the backend and re-run; no behavioral claim can be made from this.")
    if v.directive is Directive.SHIP:
        return ("Recall, precision, and final accuracy all clear tolerance for the worst "
                "model — safe to enable drop-to-retrieve.")
    if v.directive is Directive.BLOCK:
        return ("At least one metric misses tolerance for its worst model — keep "
                "drop-to-retrieve off until this improves.")
    if v.directive in (Directive.INSUFFICIENT, Directive.NOT_CONCLUDED):
        # WHICH metrics failed to conclude, read off the lattice — not assumed. The first
        # version of this branch hardcoded "recall and no-overfetch clear tolerance for the
        # worst model, so the mechanism works", licensed by the claim that only accuracy
        # could ever be withheld. That stopped being true the moment a model with no rows
        # of a kind was withheld from that kind's gate instead of scored at a fabricated
        # 0%, and the sentence would then have asserted a pass for a metric that was never
        # measured — the same over-claim, in the metric it was written to avoid.
        withheld = {label: sorted(set(v.metrics[m].excluded) | set(v.metrics[m].thin))
                    for m, label, _ in DROPEVAL_METRICS
                    if v.metrics[m].directive is not Directive.SHIP}
        if not withheld:
            # No metric reached its gate at all — only `dropeval_verdict({})`, since both
            # renderers stop before this on an empty fleet. Spelled out rather than folded
            # into the sentence below with an `or "this fleet"`, which would have claimed
            # models were withheld when none existed.
            return ("**INCONCLUSIVE for enabling** — no model reached a gate, so there is "
                    "nothing to conclude about enabling the drop.")
        # INSUFFICIENT is the max only when EVERY withheld model is underpowered — the
        # arms paired and showed no loss, over too few questions to mean anything. A
        # different headline because it is a different next action: generate questions,
        # versus go find out why the comparison never happened. Before this the two
        # rendered identically, which made `INSUFFICIENT` a lattice member with no
        # observable behaviour — a distinction the code drew and the reader never saw.
        head = ("**INSUFFICIENT for enabling**" if v.directive is Directive.INSUFFICIENT
                else "**INCONCLUSIVE for enabling**")
        detail = "; ".join(
            f"{label} for " + (", ".join(f"`{m}`" for m in models) if models else "this fleet")
            for label, models in withheld.items())
        mech_ok = all(v.metrics[m].directive is Directive.SHIP
                      for m in ("recall", "precision"))
        if mech_ok:
            lead = ("recall and no-overfetch clear tolerance for the worst model, so the "
                    "mechanism works, but the OUTCOME impact of dropping is unmeasured")
        elif any(v.metrics[m].thin and not v.metrics[m].excluded
                 for m in ("recall", "precision")):
            # Measured, not missing. Saying "not gated" here would be the #335 defect
            # inverted: the numbers ARE published and the reader can see them, so the
            # sentence has to be about their weight, not their absence.
            lead = ("the drop-to-retrieve mechanism was measured on too few questions to "
                    "conclude anything from a clean result — not concluded")
        else:
            lead = ("the drop-to-retrieve MECHANISM itself was not gated, so nothing "
                    "downstream of it means anything — not gated")
        # SCOPED to the withheld models. The unscoped form of this sentence ("the OUTCOME
        # impact of dropping is unmeasured") was true on the old code, where it could only
        # fire when NO model was scored on accuracy. The lattice reaches here whenever ANY
        # model is withheld, so the unscoped claim now prints two lines under a measured
        # `**PASS**` for the models that were gated — a false statement about the report's
        # own contents.
        acc = v.metrics["accuracy"]
        scoped = ""
        if mech_ok and acc.worst is not None:
            scoped = (f" It WAS gated for the rest of the fleet — worst-case "
                      f"`{acc.worst.model}` at {acc.worst.gap:+.0%} — but a policy that is "
                      "unsafe for one model in the fleet is unsafe (#24), so the verdict "
                      "cannot rest on the models that happened to be measurable.")
        return (f"{head} — {lead}: {detail}.{scoped} Each model above is named with the "
                "reason its metric did not conclude.")
    assert_never(v.directive)

def _pct(saved: int, base: int) -> str:
    return f"{(saved / base * 100):+.1f}%" if base else "n/a"


def _sum(rows: list[dict[str, Any]], *path: str) -> int:
    total = 0
    for r in rows:
        v: Any = r
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            total += int(v)
    return total


def build_probe_report(
    vr_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]
) -> str:
    """Render the Tier-0.5 ceiling probes — value redundancy + cross-call overlap.

    These are UPPER BOUNDS on what a dictionary coder / diff encoder could save,
    measured ON TOP of what tabularize already achieves. They inform the go/no-go
    on building Tier 0.5; they do not compress anything.
    """
    out: list[str] = ["# terse ceiling probes (Tier 0.5)", ""]

    out += [
        "## Value redundancy — dictionary-coding headroom",
        "",
        "Repeated VALUE tokens across cells, beyond the repeated KEYS tabularize folds.",
        "`est dict saving` is a conservative upper bound (first occurrence kept as legend).",
        "",
        "| Tool | sha | cells | redundancy | redundant tok | est dict saving |",
        "|---|---|---|---|---|---|",
    ]
    for r in vr_rows:
        out.append(
            f"| `{r['tool']}` | `{r['sha']}` | {r['cells']} | {r['redundancy_ratio']:.1%} "
            f"| {r['redundant_value_tokens']} | {r['est_dict_saving_tokens']} |"
        )
    if vr_rows:
        ratios = sorted(r["redundancy_ratio"] for r in vr_rows)
        median = ratios[len(ratios) // 2]
        verdict = "worth a Tier 0.5 build" if median >= 0.15 else "thin — likely skip Tier 0.5"
        out += ["", f"Median value-redundancy: **{median:.1%}** → {verdict}.", ""]
    else:
        out += ["", "_No record-shaped payloads in corpus to probe._", ""]

    out += [
        "## Cross-call overlap — diffing headroom",
        "",
        "Token overlap between successive same-tool payloads. `est delta saving` is the",
        "fraction of the current payload already present in the prior one (upper bound).",
        "",
    ]
    if overlap_rows:
        out += [
            "| Tool | prev | curr | curr tok | shared | overlap |",
            "|---|---|---|---|---|---|",
        ]
        for r in overlap_rows:
            out.append(
                f"| `{r['tool']}` | `{r['prev_sha']}` | `{r['curr_sha']}` | {r['curr_tokens']} "
                f"| {r['shared_tokens']} | {r['overlap_ratio']:.1%} |"
            )
        out.append("")
    else:
        out += [
            "_No same-tool payload pairs in corpus — capture a tool 2+ times in an agent",
            "loop to measure diffing headroom._",
            "",
        ]
    return "\n".join(out)


def build_cross_server_probe_report(
    redundancy: dict[str, Any], overlap: dict[str, Any],
    corpus_servers: list[str] | None = None,
) -> str:
    """Render the #64 Phase 0 cross-server redundancy probe with an explicit go/no-go.

    Primary gate = the record-value dictionary increment (shared whole VALUES across peers) —
    the ONLY quantity a value dictionary can actually elide. Lever B's idf token overlap is
    corroborating context ONLY: it measures shared SUBWORD tokens, which do NOT imply shared
    elidable values and must never on their own trigger a BUILD. (Verified the hard way in
    #64 Phase 1: a 20.9% Lever-B token overlap coincided with ZERO cross-server value overlap
    and a −17.7% realized result — token overlap is not value-elision headroom.) Thresholds
    on Lever A mirror the plan: <3% of corpus -> close; >=10% -> build; between -> lean close.
    When Lever A is blind, the verdict is INCONCLUSIVE — never a Lever-B BUILD.
    """
    out: list[str] = ["# terse #64 Phase 0 — cross-server redundancy probe", ""]

    out += [
        "## Lever A — cross-server value dictionary (primary gate)",
        "",
        "Does one legend shared across peers fold more repeated record VALUES than",
        "independent per-peer legends? `increment` = pooled saving − Σ per-peer saving,",
        "i.e. values repeated across ≥2 distinct servers. Upper bound.",
        "",
        "| Server | record-lists | cells | redundancy | est per-peer dict saving |",
        "|---|---|---|---|---|",
    ]
    for r in redundancy["per_server"]:
        out.append(
            f"| `{r['server']}` | {r['record_lists_folded']} | {r['cells']} "
            f"| {r['redundancy_ratio']:.1%} | {r['est_dict_saving_tokens']} |"
        )
    inc = redundancy["cross_server_increment_tokens"]
    frac_corpus = redundancy["increment_frac_of_corpus"]
    frac_peer = redundancy["increment_frac_over_per_peer"]
    out += [
        "",
        f"- per-peer saving (today): **{redundancy['per_peer_saving_tokens']} tok**",
        f"- pooled shared-legend saving: **{redundancy['pooled_saving_tokens']} tok**",
        f"- **cross-server increment: {inc} tok** "
        f"({frac_corpus:+.1%} of corpus value-tokens; {frac_peer:+.1%} over per-peer)",
        "",
    ]

    content_median = overlap.get("median_content_overlap", 0.0)
    out += [
        "## Lever B — cross-server overlap, framing-normalized (spans all servers)",
        "",
        "Token overlap between payloads of *different* servers. `raw` is inflated by shared",
        "JSON framing; `content` re-weights by idf so ubiquitous framing tokens drop out.",
        "Unlike Lever A this sees text/source servers. **CAVEAT (do not gate BUILD on this):**",
        "these are shared SUBWORD tokens, not shared whole VALUES. A value dictionary can only",
        "elide entire repeated values; two servers can share high token mass (both mention",
        "'Interceptor', 'proxy', 'src/terse') while sharing ZERO complete values. Lever B is a",
        "loose upper bound / corroboration signal, never realizable headroom on its own (#64).",
        "",
    ]
    if overlap["rows"]:
        cap_note = " (capped per server-pair)" if overlap["capped"] else ""
        out += [
            f"Median overlap — raw **{overlap['median_overlap']:.1%}** vs "
            f"content **{content_median:.1%}** (framing-netted) "
            f"across {overlap['pairs']} payload pairs{cap_note} (cap {overlap['cap_per_pair']}/pair).",
            "",
            "| A | B | curr tok | shared | raw overlap | content overlap |",
            "|---|---|---|---|---|---|",
        ]
        for r in overlap["rows"][:15]:
            out.append(
                f"| `{r['server_a']}` | `{r['server_b']}` | {r['curr_tokens']} "
                f"| {r['shared_tokens']} | {r['overlap_ratio']:.1%} "
                f"| {r.get('content_overlap_ratio', 0.0):.1%} |"
            )
        if len(overlap["rows"]) > 15:
            out.append(f"| … | | | | | _{len(overlap['rows']) - 15} more rows omitted_ |")
        out.append("")
    else:
        out += ["_Fewer than two servers with record/raw payloads — nothing to cross._", ""]

    # Coverage guard: Lever A only sees record-shaped payloads. A server whose output is
    # text/source (codegraph) or pre-compressed contributes nothing, so a low increment may
    # just mean the lever is blind — not that redundancy is absent. Surface this BEFORE the
    # verdict so a thin sample can't read as a confident "close".
    lever_a_servers = {r["server"] for r in redundancy["per_server"]}
    seen = set(corpus_servers or lever_a_servers)
    missing = sorted(seen - lever_a_servers)
    thin = any(r["record_lists_folded"] < 30 for r in redundancy["per_server"])
    inconclusive = len(lever_a_servers) < 2 or bool(missing)

    if missing and not lever_a_servers:
        out += [
            f"> ⚠ **Coverage gap:** no server produced record-shaped payloads "
            f"({', '.join(f'`{m}`' for m in missing)} all emit text/source or pre-compressed "
            "output), so **Lever A is empty** — the verdict rests entirely on Lever B's "
            "framing-netted content overlap.",
            "",
        ]
    elif missing:
        out += [
            f"> ⚠ **Coverage gap:** {', '.join(f'`{m}`' for m in missing)} present in the "
            "corpus but produced **no record-shaped payloads** (text/source or pre-compressed "
            "output), so Lever A cannot see them. Its increment above is measured only across "
            f"{', '.join(f'`{s}`' for s in sorted(lever_a_servers))}; Lever B spans all servers.",
            "",
        ]

    # Verdict. Lever A's shared whole-VALUE increment is the ONLY gate that can say BUILD,
    # because it is the only quantity a value dictionary can elide. When Lever A is blind (a
    # server emits text/source or non-record payloads), the verdict is INCONCLUSIVE — NOT a
    # fallback BUILD on Lever B. Lever B is idf-weighted SUBWORD-token overlap; #64 Phase 1
    # proved it does not track value overlap (20.9% Lever B → 0 cross-server value → −17.7%
    # realized). A high Lever B with a blind Lever A means "capture a corpus where whole-value
    # overlap is measurable," never "build."
    blind = ", ".join(f"`{m}`" for m in missing)
    if inconclusive:
        verdict = (f"**INCONCLUSIVE — do NOT build on Lever B.** Lever A (shared whole-value "
                   f"increment, the only realizable gate) is blind to {blind}. Lever B's "
                   f"framing-netted content overlap is **{content_median:.1%}**, but that is shared "
                   "SUBWORD-token mass, not shared elidable VALUES — #64 Phase 1 confirmed a high "
                   "Lever B can sit atop ZERO value overlap and realize NEGATIVE savings. To decide, "
                   "capture a corpus where Lever A can see whole-value overlap across peers (record- "
                   "or scalar-value payloads from ≥2 servers about the same entities) and re-run. "
                   "Do not greenlight a value dictionary on token overlap.")
    elif thin:
        verdict = ("**WEAK — lean CLOSE, re-run to confirm.** Increment is "
                   f"{frac_corpus:+.1%} of corpus, but <30 record-lists on a peer makes it "
                   "noisy. Close only if a fuller capture holds the same near-zero increment.")
    elif frac_corpus < 0.03:
        verdict = ("**CLOSE #64 Phase 1** — cross-server redundancy is negligible "
                   "(<3% of corpus). The gateway stays pure ergonomics; ship only Phase 2 "
                   "(fan-out gaps).")
    elif frac_corpus >= 0.10:
        verdict = ("**BUILD Phase 1** — a shared cross-peer legend has real headroom "
                   "(≥10% over per-peer). Proceed to design the shared session dictionary.")
    else:
        verdict = ("**MARGINAL (3–10%)** — lean CLOSE: the increment is unlikely to pay for "
                   "the shared-legend complexity (cross-stream diff-desync #20, marker-collision "
                   "#6 across peers). Revisit only with an evenly-captured corpus.")
    out += ["## Verdict", "", verdict, ""]
    return "\n".join(out)


def build_tokenizer_report(rows: list[dict[str, Any]]) -> str:
    """Render cross-tokenizer invariance — cl100k vs o200k savings % per tool.

    Claude has no public local tokenizer, so there is no ground-truth token count to
    check against. Invariance across two different vocabs is the substitute: if the
    savings % barely moves, it's robust to Claude's.
    """
    from .tokenize import CL100K, O200K

    out: list[str] = [
        "# terse cross-tokenizer invariance",
        "",
        "No public Claude tokenizer exists, so cl100k is only an estimate.",
        "Substitute: savings % under two different BPE vocabularies. Stability => robust",
        "to Claude's tokenizer, because structural folding removes tokens in any vocab.",
        "",
        "| Tool | cl100k % | o200k % | Δ (pts) |",
        "|---|---|---|---|",
    ]
    deltas = []
    for r in sorted(rows, key=lambda r: -(r[CL100K]["pct"] or 0)):
        a = r[CL100K]["pct"]
        b = r[O200K]["pct"]
        if a is None or b is None:
            out.append(f"| `{r['tool']}` | n/a | n/a | n/a |")
            continue
        d = abs(a - b)
        deltas.append(d)
        out.append(f"| `{r['tool']}` | {a:+.1f}% | {b:+.1f}% | {d:.1f} |")
    out.append("")
    if deltas:
        worst = max(deltas)
        mean = sum(deltas) / len(deltas)
        verdict = "savings are tokenizer-invariant" if worst <= 3.0 else "savings vary by tokenizer — investigate"
        out += [
            f"Max divergence: **{worst:.1f} pts**, mean **{mean:.1f} pts** → {verdict}.",
            "",
        ]
    return "\n".join(out)


def build_verify_header(corpus_label: str, n_payloads: int) -> str:
    """Attestation header for `terse verify` — states what the report proves and what an
    adopter must still verify themselves (tests, no-egress, fail-open). Self-contained so
    the markdown stands alone as a shareable proof. No timestamp, so the artifact stays
    reproducible (principle #31)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("terse")
    except PackageNotFoundError:
        v = "(editable/dev)"
    return "\n".join([
        "# terse — verification report",
        "",
        f"- terse `{v}`  ·  python `{platform.python_version()}`  ·  os `{platform.system()}`",
        f"- corpus: {corpus_label} — {n_payloads} payloads",
        "",
        "## What the tables below prove",
        "",
        "- **Lossless** — every payload round-trips byte-faithfully through terse. The "
        "lossless gate INVALIDATES the whole report if any payload fails, because token "
        "savings on top of corrupted data are meaningless.",
        "- **Savings** — measured cl100k-token reduction per shape bucket and per tool on "
        "this corpus. terse's win is shape-dependent, so it is never averaged into one "
        "headline number.",
        "",
        "## What this does NOT replace — verify these yourself",
        "",
        "- **Correctness suite:** `pytest` — the full lossless / diff / proxy test set "
        "(runs in CI on Python 3.11–3.13).",
        '- **No UNEXPECTED egress:** `grep -rE "requests|urllib|socket" src/terse` finds '
        "real network code in three places — `fluency/answerers.py` and `dropeval.py` (each an "
        "explicit, opt-in model eval) and `transport.py` (the proxy's own downstream "
        "connection). A stdio-only downstream makes zero network calls; an HTTP/SSE "
        "downstream (opt-in via `--config`/a `url`-configured server) talks only to the "
        "target you configured — never a third party. The same grep also flags a few "
        "incidental, non-networking hits (the word \"requests\" in a comment or docstring) "
        "elsewhere in the tree; read the actual matches rather than trusting a count. The "
        "proxy persists nothing beyond what `--capture-dir`/`--debug-log` explicitly ask for.",
        "- **Fail-open:** read `src/terse/proxy.py` — any parse/compress error forwards the "
        "ORIGINAL tool result unchanged; terse never drops or blocks a tool call.",
        "",
        "---",
        "",
        "",
    ])


def verify_summary(rows: list[dict[str, Any]], coverage: dict[str, Any],
                   corpus_label: str) -> dict[str, Any]:
    """Machine-readable counterpart to build_verify_header + build_report: the same
    lossless-gate verdict and cl100k savings the markdown shows, as a dict for
    `terse verify --json` (scriptable / CI-checkable). Numbers come from the same
    `_sum` path as the report, so the JSON can never disagree with it."""
    failures = [r for r in rows if not r.get("roundtrip_ok", False)]
    # Reported as its own verdict, not folded into `lossless_gate` (#188). It does NOT
    # invalidate the savings below — those come from the default pipeline, which passed —
    # but an opt-in tier that loses data is still a codec defect, and splitting the gates
    # would otherwise make it vanish from every report that only reads `roundtrip_ok`.
    #
    # Qualified by `roundtrip_ok`, and that qualification is load-bearing: `measure` leaves
    # `embedded_ok` False when the DEFAULT gate failed, because the embedded pipeline was
    # never evaluated there. Counting those rows would put every total-codec-failure sha in
    # the embedded list too, telling a CI job gating on `embedded_gate.ok` that an opt-in
    # tier broke when in fact the codec failed outright.
    emb_evaluated = [r for r in rows if r.get("roundtrip_ok", False)]
    emb_failures = [r for r in emb_evaluated if not r.get("embedded_ok", True)]
    total = len(rows)

    def _bucket(sub: list[dict[str, Any]]) -> dict[str, Any]:
        raw = _sum(sub, "cl100k", "raw")
        cmp_ = _sum(sub, "cl100k", "compressed")
        return {"n": len(sub), "raw_tokens": raw, "terse_tokens": cmp_,
                "saved_tokens": raw - cmp_,
                "saved_pct": round((raw - cmp_) / raw * 100, 1) if raw else 0.0}

    return {
        "corpus": corpus_label,
        "payloads": total,
        "lossless_gate": {
            "ok": not failures,
            "passed": total - len(failures),
            "total": total,
            "failures": [{"tool": r.get("tool"), "sha": r.get("sha"),
                          "shape": r.get("shape")} for r in failures],
        },
        "embedded_gate": {
            "ok": not emb_failures,
            # Same key names as `lossless_gate` so a scripted reader can treat the two
            # verdict blocks symmetrically — plus `evaluated`, which the sibling does not
            # need: the embedded gate only runs on payloads the default gate cleared, so
            # `passed + failed` is `evaluated`, not `total`.
            "passed": len(emb_evaluated) - len(emb_failures),
            "evaluated": len(emb_evaluated),
            "total": total,
            "failures": [{"tool": r.get("tool"), "sha": r.get("sha"),
                          "shape": r.get("shape")} for r in emb_failures],
        },
        "tokens_cl100k": _bucket(rows),
        "by_shape": {s: _bucket([r for r in rows if r["shape"] == s])
                     for s in sorted({r["shape"] for r in rows})},
        "coverage": {"total": coverage.get("total", 0),
                     "by_tool": dict(coverage.get("by_tool", {})),
                     "by_shape": dict(coverage.get("by_shape", {}))},
    }


def build_report(rows: list[dict[str, Any]], coverage: dict[str, Any]) -> str:
    out: list[str] = ["# terse measurement report", ""]

    # --- Lossless gate (gates everything) ---
    failures = [r for r in rows if not r.get("roundtrip_ok", False)]
    total = len(rows)
    passed = total - len(failures)
    out += ["## Lossless gate", ""]
    if failures:
        out += [
            f"**INVALID — {len(failures)}/{total} payloads FAILED the round-trip gate.**",
            "Savings below are meaningless until this is 0. Failing shas:",
            "",
            *[f"- `{r.get('tool')}` / `{r.get('sha')}` ({r.get('shape')})" for r in failures],
            "",
        ]
    else:
        out += [f"All {passed}/{total} payloads round-trip losslessly. ✅", ""]

    # The opt-in `embedded` fold gets its own verdict (#188): it costs only its own tier,
    # so it never marks the report INVALID — but it must still be visible, or splitting the
    # gates would hide a real losslessness failure from every reader of this section.
    # Only among rows the default gate CLEARED — see `verify_summary` for why. Printing an
    # embedded failure for a row already listed as INVALID above put "The default pipeline
    # passed, so the savings below stand" directly beneath "INVALID — n/n payloads FAILED",
    # with the same sha in both lists.
    emb_failures = [r for r in rows
                    if r.get("roundtrip_ok", False) and not r.get("embedded_ok", True)]
    if emb_failures:
        out += [
            f"**`embedded` tier: {len(emb_failures)}/{total} payloads FAILED its round-trip.**",
            "The default pipeline passed, so the savings below stand and the tool still "
            "compresses — but `policy generate` will not offer `embedded` for it. "
            "Failing shas:",
            "",
            *[f"- `{r.get('tool')}` / `{r.get('sha')}` ({r.get('shape')})"
              for r in emb_failures],
            "",
        ]

    # --- Coverage ---
    out += ["## Coverage", "", f"Total payloads captured: **{coverage.get('total', 0)}**", ""]
    out += ["| Tool | Payloads |", "|---|---|"]
    for tool, n in sorted(coverage.get("by_tool", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| `{tool}` | {n} |")
    out += ["", "| Shape bucket | Payloads |", "|---|---|"]
    for shape, n in sorted(coverage.get("by_shape", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| {shape} | {n} |")
    out.append("")

    # --- Savings per shape bucket (Tier-0 total, cl100k) ---
    shapes = sorted({r["shape"] for r in rows})
    out += [
        "## Tier-0 savings by shape bucket (cl100k)",
        "",
        "Headline = full Tier-0 (minify + tabularize) vs the raw bytes the model would see.",
        "",
        "| Shape | n | raw tok | terse tok | saved | % |",
        "|---|---|---|---|---|---|",
    ]
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        raw = _sum(sub, "cl100k", "raw")
        cmp_ = _sum(sub, "cl100k", "compressed")
        saved = raw - cmp_
        out.append(f"| {shape} | {len(sub)} | {raw} | {cmp_} | {saved:+d} | {_pct(saved, raw)} |")
    raw_all = _sum(rows, "cl100k", "raw")
    cmp_all = _sum(rows, "cl100k", "compressed")
    out.append(
        f"| **ALL** | {len(rows)} | {raw_all} | {cmp_all} | {raw_all - cmp_all:+d} | "
        f"{_pct(raw_all - cmp_all, raw_all)} |"
    )
    out.append("")

    # --- Per-tool savings (the proxy decision is per-tool, not per-shape) ---
    # Shape buckets can hide a deep-nested win next to a true no-op (e.g. runecho's
    # nested symbol lists vs a single compact object both land in 'compact-json').
    # `or "?"` on BOTH lines, and the same expression on each. They used to differ
    # — `.get("tool", "?")` here, a bare `.get("tool")` in the filter below — which
    # agreed on every input except a row whose `tool` key is ABSENT: the set
    # substituted "?" and the filter compared None to it, so that row matched
    # nothing and its tokens left this table while still counting in every other
    # total on the page. A row with an explicit `"tool": None` was worse: the key
    # exists, so `None` entered the set and `sorted()` raised TypeError on the
    # whole report. `or` normalises absent, None and "" alike, once, in one place.
    tools = sorted({r.get("tool") or "?" for r in rows})
    out += [
        "## Tier-0 savings by tool (cl100k)",
        "",
        "Per-tool, because terse's value is shape-dependent and a blanket average hides it.",
        "",
        "| Tool | shape | raw tok | terse tok | saved | % |",
        "|---|---|---|---|---|---|",
    ]
    tool_rows = []
    for tool in tools:
        sub = [r for r in rows if (r.get("tool") or "?") == tool]
        raw = _sum(sub, "cl100k", "raw")
        cmp_ = _sum(sub, "cl100k", "compressed")
        shape = sub[0]["shape"] if sub else "?"
        tool_rows.append((raw - cmp_, raw, cmp_, tool, shape))
    for saved, raw, cmp_, tool, shape in sorted(tool_rows, reverse=True):
        out.append(f"| `{tool}` | {shape} | {raw} | {cmp_} | {saved:+d} | {_pct(saved, raw)} |")
    out.append("")

    # --- Tier attribution (where the saving came from) ---
    out += [
        "## Tier attribution by shape (cl100k tokens saved)",
        "",
        "minify = whitespace + \\uXXXX unescaping · tabularize = repeated keys folded ·",
        "dictionary = repeated values folded into an inline legend (Tier 0.5).",
        "A ~0 minify column means the payload arrived already-compact (the headroom no-op).",
        "",
        "| Shape | minify | tabularize | dictionary | total |",
        "|---|---|---|---|---|",
    ]
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        m = _sum(sub, "saved_cl100k", "minify")
        t = _sum(sub, "saved_cl100k", "tabularize")
        d = _sum(sub, "saved_cl100k", "dictionary")
        out.append(f"| {shape} | {m:+d} | {t:+d} | {d:+d} | {m + t + d:+d} |")
    out.append("")

    return "\n".join(out)


def build_trend_report(runs: list[dict[str, Any]]) -> str:
    """Render the `measure --history` trend (#51 fast-follow) — one row per past run,
    oldest first, so a reader sees whether the win is improving, flat, or regressing
    as the corpus grows/changes over time. `runs` is `history.load_history()`'s output
    WITH the current run already appended by the caller — this function only ever
    displays the persisted summary numbers, never re-derives them, so a rendered trend
    can never drift from what was actually written to the history file."""
    out: list[str] = ["## Trend across runs", ""]
    if len(runs) < 2:
        out += ["_Only one run recorded so far — trend needs at least two "
                "`--history` runs to show a delta._", ""]
        return "\n".join(out)
    out += [
        "| # | timestamp | label | payloads | lossless | raw tok | terse tok | saved % | Δ pts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    prev_pct: float | None = None
    for i, r in enumerate(runs, start=1):
        pct = r.get("saved_pct")
        pct_s = f"{pct:+.1f}%" if pct is not None else "n/a"
        delta = f"{pct - prev_pct:+.1f}" if pct is not None and prev_pct is not None else "—"
        gate = f"{r.get('lossless_pass', '?')}/{r.get('n_payloads', '?')}"
        out.append(
            f"| {i} | {r.get('ts', '?')} | {r.get('label') or '—'} | {r.get('n_payloads', '?')} "
            f"| {gate} | {r.get('raw_tok', '?')} | {r.get('compressed_tok', '?')} "
            f"| {pct_s} | {delta} |")
        prev_pct = pct
    out.append("")
    return "\n".join(out)


def _build_diff_style_report(results: dict, title: str, intro: list[str],
                             empty_hint: list[str], control_label: str = "full-terse") -> str:
    """Shared body for build_diff_report and build_text_diff_report — the row shape
    ({qid, qtype, transform, trials, terse_ok, diff_ok}) and verdict math are identical
    for both; only the title/intro/empty-hint copy and the control column's label
    differ. `empty_hint` is pre-split into lines (not a single string) so each caller
    controls its own line-wrapping exactly, the same way `intro` already does."""
    out: list[str] = [title, ""]
    out += intro
    if not results or not any(results.values()):
        out += [*empty_hint, ""]
        return "\n".join(out)

    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a "
        "question-clustered (sandwich) bound.",
        "",
        f"| Model | q | {control_label} | diff | regressions |",
        "|---|---|---|---|---|",
    ]
    gap_rows: dict[str, tuple[float, float, float, float]] = {}
    unmeasured: dict[str, tuple[ExclusionReason | None, int, int, int]] = {}  # (why, fails, attempts, paired)
    for model, rows in results.items():
        n = len(rows)
        if not n:
            continue
        # Same gate as build_fluency_report (#263/#264). This report had NO control of any
        # kind — not even the raw==0 one — so a backend that was entirely down scored 0%
        # on both arms, produced a gap of exactly 0, and printed "safe to enable
        # `proxy --diff`". A false PASS on a ship gate is strictly worse than the false
        # FAIL #263 was filed about: nobody re-checks a result that agrees with them.
        # Same rule the forest plot applies (`diff_gap_rows`), from the same function, or
        # the chart printed beneath this table draws a bar for a model the table just
        # declined to score.
        g = arm_gap(rows, "diff_ok", "terse_ok")
        if g.excluded:
            unmeasured[model] = (g.excluded,
                                 sum(int(r.get("fails", 0)) for r in rows),
                                 sum(int(r.get("attempts", 0)) for r in rows),
                                 len(paired_rows(rows, "diff_ok", "terse_ok")))
            out.append(f"| `{model}` | {n} | n/a | n/a | n/a |")
            continue
        facc, fse = g.control_acc, g.control_se
        dacc, dse = g.form_acc, g.form_se
        # Counted over the PAIRED subset, not `rows` (#280 F5): a row the diff arm never
        # answered is not a regression, and counting it as one made the column contradict
        # the gap printed beside it.
        regr = sum(1 for r in g.rows if int(r["terse_ok"]) == r.get("trials", 1)
                   and int(r["diff_ok"]) < r.get("trials", 1))
        gap_rows[model] = (dacc, dse, facc, fse)  # form=diff, control=control_label
        # `len(g.rows)`, not `len(rows)`: every accuracy on this line is computed over the
        # PAIRED subset, so printing the full question count beside them states a
        # denominator the numbers do not use. `q` is the exam that was actually sat.
        out.append(f"| `{model}` | {len(g.rows)} | {facc:.0%} ±{_ci(fse) * 100:.0f} "
                   f"| {dacc:.0%} ±{_ci(dse) * 100:.0f} | {regr} |")
    out.append("")
    out += _not_measured_lines(unmeasured)

    out += ["## Verdict", ""]
    worst = _worst_case_gap(gap_rows)
    if worst:
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "diff-form", control_label))
        if worst.passed:
            out.append("- Reading the diff costs no comprehension beyond tolerance — safe to "
                       "enable `proxy --diff` for the tested models.")
        else:
            out.append("- The diff form regresses comprehension beyond tolerance — keep "
                       "`proxy --diff` off, or restrict it to tools whose diffs stay legible.")
    else:
        # Nothing survived the gate. Say so instead of falling silent — this verdict is
        # read as a go/no-go on `proxy --diff`, and an empty one reads as "no objection".
        # Reason-specific advice: "fix the backend" is wrong guidance for a run whose
        # backend answered every call and whose arms merely could not be paired.
        # Not "no model left a question both arms completed": `_unmeasured` gates on
        # transport BEFORE pairing, so a model withheld by it can still have questions
        # that pair cleanly. Says what is true of every withheld model instead.
        # It POINTS at the paragraph rather than re-listing the reasons. The old wording
        # enumerated three ("an unreachable backend, calls lost until nothing paired, or
        # too few questions"), which made this a FOURTH copy of the reason vocabulary —
        # already drifted, since it had no entry for the producer that loses no calls at
        # all (an arm with zero completed trials). The paragraph above states the reason
        # and its remedy together, chosen from the counts; restating it here can only
        # disagree with it (#338).
        out.append("- **NO VERDICT — nothing was scored.** Every model was withheld, so "
                   "this run says nothing about the diff form either way. The paragraph "
                   "above names each model, its reason, and the remedy that reason "
                   "licenses; they are not the same remedy, so read it before re-running.")
    out.append("")
    return "\n".join(out)


def build_diff_report(results: dict) -> str:
    """Render the cross-call diff fluency eval: does a model read a diff against the
    prior result as accurately as the full current result?

    `results` is {model: [row,...]} from fluency.run_diff_fluency; each row carries
    full-terse (`terse_ok`) and diff-form (`diff_ok`) success counts over the same
    questions. The verdict gates on the worst model (principle #24): the proxy emits a
    diff only when smaller, so this bounds the comprehension cost of enabling it.
    """
    return _build_diff_style_report(
        results,
        "# terse cross-call diff fluency",
        ["Does a model read a diff against the prior same-tool result as accurately as the",
         "full current result? Same questions, paired per question; ground truth is",
         "deterministic. Risk-item check for `proxy --diff` before turning it on.", ""],
        ["No model answers, or no same-tool payload PAIRS in the corpus. Capture a tool",
         "2+ times (an agent loop) and configure a backend, then re-run "
         "`terse fluency --diff`."],
    )


def build_text_diff_report(results: dict) -> str:
    """Render the text-diff fluency eval: does a model reconstruct the current TEXT as
    accurately from (previous text + text-diff) as from the full current text?

    `results` is {model: [row,...]} from fluency.run_text_diff_fluency. Unlike
    build_diff_report, the control form is raw text, not full-terse — Tier 0 doesn't
    compress non-JSON text at all — so the control column is labeled accordingly.
    """
    return _build_diff_style_report(
        results,
        "# terse text-diff fluency",
        ["Does a model reconstruct the current text as accurately from (previous text +",
         "text-diff) as from the full current text? Tier 0 doesn't compress non-JSON text",
         "at all, so the control form here is the raw text, not a compressed one. Risk-item",
         "check before enabling `proxy --diff` for text-heavy tools.", ""],
        ["No model answers, or no same-tool TEXT payload PAIRS in the corpus (JSON pairs "
         "are `--diff`'s domain, not this one's). Capture a text-producing tool 2+ times, "
         "then re-run `terse fluency --text-diff-eval`."],
        control_label="raw text",
    )


def build_diff_soak_report(results: dict) -> str:
    """Render the diff-chain soak: does comprehension DRIFT as a model reads deeper
    chains of consecutive diffs off one full anchor (#8/#20 follow-up)?

    `results` is {model: [row,...]} from fluency.run_diff_soak; rows carry the
    run_diff_payload shape plus `depth` (how many diffs were chained). The overall
    verdict gates on the worst model across ALL depths (principle #24); the by-depth
    table is the drift signal itself — production caps chains at the keyframe
    interval (default 5), so the deepest tested depth is the deployed worst case."""
    out: list[str] = ["# terse diff-chain soak", ""]
    out += [
        "Does a model answer questions about the LATEST state as accurately from",
        "(one full result + k consecutive diffs, applied in order) as from the full",
        "current result — and does accuracy drift as k grows? Chains are real",
        "consecutive same-tool corpus payloads (chronological), exactly what",
        "`proxy --diff` emits between keyframes. No system primer (production",
        "condition).", "",
    ]
    if not results or not any(results.values()):
        out += [
            "No model answers, or no same-tool diffable RUNS in the corpus. Capture a",
            "tool 3+ times with small changes between calls, then re-run",
            "`terse fluency --diff-soak`.", "",
        ]
        return "\n".join(out)

    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    depths = sorted({r["depth"] for rows in results.values() for r in rows})
    out += [
        "## Accuracy by chain depth",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a "
        "question-clustered (sandwich) bound. depth = diffs chained after the full anchor.",
        "",
        "| Model | depth | q | full-terse | chain | gap |",
        "|---|---|---|---|---|---|",
    ]
    # Same transport-failure gate as the other two diff reports (#264). Without it a
    # backend that was down scores 0% on both arms at every depth, which is a gap of
    # exactly 0 and reads as PASS — and the by-depth table shows a flat, reassuring
    # no-drift line drawn entirely from calls that never happened.
    unmeasured = {m: rows for m, rows in results.items() if rows and _unmeasured(rows)}
    # model -> reason -> depths withheld by the per-depth gate. Keyed by reason because
    # "the backend answered, but..." is false for a slice withheld by `_unmeasured`.
    withheld_depths: dict[str, dict[str, set[int]]] = {}
    for model, rows in results.items():
        for depth in depths:
            drows = [r for r in rows if r["depth"] == depth]
            if not drows:
                continue
            # Per-DEPTH pairing, not just the per-model gate above: a depth slice is its
            # own gap, and an arm that lost calls only at depth 5 was invisible to a gate
            # computed over the whole model. This table sits outside `## Verdict`, so the
            # invariance test's `_gate_signature` never saw it either (#280 F2).
            dg = arm_gap(drows, "diff_ok", "terse_ok")
            if model in unmeasured or dg.excluded:
                # Both causes share the `"unmeasured"` reason since #332, so `dg.excluded`
                # cannot tell them apart — but `_unmeasured` can, and it is the transport
                # half by definition. Keyed on that instead so the heading stays true:
                # "not measured" for a slice whose calls failed, "not compared" for one
                # whose calls landed but left no shared question.
                # Keep the reason `_gap` decided; only SPLIT the one it cannot tell apart.
                # Substituting a string here discarded `"underpowered"` and rendered it as
                # "the backend answered, but one arm did not complete enough of the same
                # questions" — about slices where both arms completed everything.
                if dg.excluded in (None, "unmeasured"):
                    dg = dg._replace(
                        excluded=("unmeasured" if _unmeasured(drows) else "unpaired"))
                out.append(f"| `{model}` | {depth} | {len(drows)} | n/a | n/a | n/a |")
                # Recorded, because an `n/a` in this table with no prose anywhere is how a
                # withheld DEEPEST depth turned into a green "no drift" conclusion below.
                if model not in unmeasured:
                    withheld_depths.setdefault(model, {}).setdefault(
                        dg.excluded or "unmeasured", set()).add(depth)
                continue
            facc, fse = dg.control_acc, dg.control_se
            dacc, dse = dg.form_acc, dg.form_se
            out.append(f"| `{model}` | {depth} | {len(drows)} "
                       f"| {facc:.0%} ±{_ci(fse) * 100:.0f} "
                       f"| {dacc:.0%} ±{_ci(dse) * 100:.0f} | {dacc - facc:+.0%} |")
    out.append("")
    if unmeasured:
        out += [
            f"**{REASON_HEADING['unmeasured']}** — {REASON_LABEL['unmeasured']}, so no "
            "accuracy is published for: "
            + ", ".join(
                f"`{m}` ({sum(int(r.get('fails', 0)) for r in rs)}/"
                f"{sum(int(r.get('attempts', 0)) for r in rs)} calls lost)"
                for m, rs in sorted(unmeasured.items()))
            + "."
            + unmeasured_cause(
                sum(int(r.get("fails", 0)) for rs in unmeasured.values() for r in rs)),
            "",
        ]
    if withheld_depths:
        # An `n/a` row with no explanation is not a disclosure. Named here because the
        # deepest depth is exactly the one a soak exists to measure, and losing it silently
        # is how "no depth-correlated drift" gets printed about a depth nobody scored.
        # Grouped by reason: a slice withheld because its calls FAILED must not be
        # described as one where "the backend answered".
        for why in sorted({w for d in withheld_depths.values() for w in d}):
            # Counts PER WITHHELD SLICE, not per model. They are the same disclosure the
            # pooled paragraph makes, and they are what settles the reason for the reader:
            # a slice at `0/20 calls lost` was not withheld by an unreachable backend, so
            # the remedy `unmeasured_cause` picks below is checkable against the number
            # printed beside it rather than taken on trust (#338).
            per_model = []
            slice_fails = slice_attempts = 0
            for m, ds in sorted(withheld_depths.items()):
                if why not in ds:
                    continue
                sl = [r for r in results[m] if r["depth"] in ds[why]]
                f = sum(int(r.get("fails", 0)) for r in sl)
                a = sum(int(r.get("attempts", 0)) for r in sl)
                slice_fails, slice_attempts = slice_fails + f, slice_attempts + a
                depths_txt = ", ".join(str(d) for d in sorted(ds[why]))
                per_model.append(f"`{m}` (depth {depths_txt}; {f}/{a} calls lost)")
            at = ", ".join(per_model)
            # The old branch tested `why == "x"`, a reason string nothing produces, so
            # the specific wording it guarded had been unreachable since #284 and every
            # withheld depth got the generic one. `"unpaired"` is set just above from
            # `_unmeasured(drows)` — local to this table, not an `ArmGap` reason — which
            # makes the precise branch reachable for the first time.
            lead = {
                "unpaired": "**Depths not compared** — the backend answered, but one arm "
                            "did not complete enough of the same questions at: ",
                "unmeasured": f"**Depths not measured** — {REASON_LABEL['unmeasured']} at: ",
            }.get(why, f"**{REASON_HEADING.get(why, 'Excluded')}** — "
                       f"{REASON_LABEL.get(why, why)} at: ")
            tail = (". Those depths are excluded from the verdict below rather than "
                    "scored on a question set the two arms did not share (#280)."
                    if why in ("unpaired", "unmeasured") else
                    ". Too few questions survived pairing at that depth for an absence "
                    "of drift to mean anything — check the `q` column against the "
                    "generated count before assuming none were lost.")
            if why == "unmeasured":
                # Over exactly the withheld slices, not the whole model: a depth withheld
                # by a zero-trial arm can sit beside depths that lost calls, and the two
                # licence different sentences. Before #338 this site asserted the
                # transport cause unconditionally, and no test reached it — the pooled
                # gate short-circuits `withheld_depths` (`model not in unmeasured` above),
                # so a fixture that trips `_unmeasured` renders no by-depth prose at all.
                tail += unmeasured_cause(slice_fails)
            out += [lead + at + tail, ""]

    gap_rows: dict[str, tuple[float, float, float, float]] = {}
    pooled_out: dict[str, ExclusionReason | None] = {}
    for model, rows in results.items():
        if not rows or model in unmeasured:
            continue
        g = arm_gap(rows, "diff_ok", "terse_ok")
        if g.excluded:
            # Recorded and named below. Dropping a model from the ship gate with no trace
            # anywhere let a soak print "Worst-case model `N` ... PASS" while `M`'s pooled
            # gap had been withheld entirely — the other two report families name their
            # exclusions and this one did not.
            pooled_out[model] = g.excluded
            continue
        gap_rows[model] = (g.form_acc, g.form_se, g.control_acc, g.control_se)
    if pooled_out:
        # "the pooled verdict", precisely: a model withheld here may still appear in the
        # by-depth table and in the deepest-depth line, because those are scored per depth
        # slice and a slice can be sound while the pooled comparison is not.
        note = exclusion_note(pooled_out).removeprefix("excluded — ")
        out += [f"**Excluded from the pooled verdict** — {note}. Depth slices that pair "
                f"cleanly are still scored below.", ""]

    out += ["## Verdict", ""]
    worst = _worst_case_gap(gap_rows)
    if not worst:
        # Nothing survived the gate. An empty verdict on a drift soak reads as "no drift
        # found", which is the opposite of "nothing was looked at".
        # The advice has to match the cause. "Fix the backend(s) and re-run" sends a reader
        # to re-run something that will fail identically when the backend answered fine and
        # the arms simply could not be paired.
        out.append("- **No pooled verdict** — every model's POOLED gap was withheld, so "
                   "the pooled line says nothing about drift either way. Scoped to the "
                   "pooled comparison on purpose: a depth slice can still be scored below, "
                   "and since #334 often is, so claiming the run measured nothing would "
                   "contradict a verdict printed two lines further down. The paragraphs "
                   "above name each model, its reason, and the remedy that reason "
                   "licenses — see the note on the sibling line in `build_diff_report` for "
                   "why they are not re-listed here.")
    if worst:
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "chain-form",
                                           "full-terse"))
    # DELIBERATELY NOT nested under `if worst:`. This block's own comments below insist it
    # is independent of the POOLED exclusion — and it was not: being inside that branch
    # meant a withheld pooled gap skipped the deepest-depth analysis entirely. #334 made
    # that state reachable for a model that lost zero calls (a pooled subset under the
    # question floor), and it was measured dropping a fully-paired -100% collapse at the
    # deepest depth out of the verdict, under a line promising "depth slices that pair
    # cleanly are still scored below". An exclusion must never remove a demonstrated
    # regression from a verdict; nesting made the pooled gate decide the depth question.
    if True:
        # The soak-specific signal on top of the overall gate: the worst model's gap
        # at the DEEPEST depth, since a clean average can hide a depth-correlated slide.
        deep = depths[-1] if depths else 0
        # `unmeasured` applies here too. Gating only the overall gap above left this
        # depth-specific signal computing off withheld models, so a down backend still
        # decided the drift conclusion — the same defect, one paragraph further down the
        # same function. Found by the invariance test, not by review.
        # `unmeasured` only. NOT `pooled_out`: that is the POOLED exclusion, and a model
        # excluded there because its shallow depths were one-sided can still have a
        # complete, scorable deepest slice — which the per-depth `arm_gap` below is the
        # right judge of, and which is real evidence about drift. Filtering on the pooled
        # list dropped a fully-paired -80% depth-5 failure out of the depth verdict and
        # printed "No depth-correlated comprehension drift" beside a table showing it; when
        # that model was the only one at the deepest depth it emptied `deep_rows`, so even
        # the NO-VERDICT branch was skipped. That is the "absence reads as a pass" bug this
        # function already fixed once, re-entered through a different door.
        deep_rows = {m: r for m, r in
                     ((m, [x for x in rs if x["depth"] == deep])
                      for m, rs in results.items() if m not in unmeasured) if r}
        deep_gaps = {}
        for m, rs in deep_rows.items():
            dg = arm_gap(rs, "diff_ok", "terse_ok")
            if dg.excluded:
                continue
            deep_gaps[m] = (dg.form_acc, dg.form_se, dg.control_acc, dg.control_se)
        deepest = _worst_case_gap(deep_gaps)
        # The deepest slice can now be WITHHELD while the overall gap still publishes —
        # the per-depth gate above is per-slice, so a depth-5 arm that lost its hard
        # questions is excluded there while the pooled model sails through. Before that
        # gate existed, `deep_rows` empty implied `worst` was None too, so `deepest is
        # None` was unreachable here and reading it as "passed" was harmless. It is not
        # harmless now: it printed "No depth-correlated comprehension drift" about the one
        # depth nobody scored. Absence of a measurement is not a passing measurement.
        deep_withheld = bool(deep_rows) and not deep_gaps
        if deepest:
            out.append(f"- At the deepest tested depth ({deep}): worst model "
                       f"`{deepest.model}` chain {deepest.form_acc:.0%} vs full "
                       f"{deepest.control_acc:.0%} (gap {deepest.gap:+.0%} "
                       f"±{deepest.gap_ci * 100:.0f} pts). "
                       f"**{'PASS' if deepest.passed else 'FAIL'}** at "
                       f"{_GAP_TOLERANCE:.0%} tolerance.")
        elif deep_withheld:
            out.append(f"- **NO VERDICT at the deepest tested depth ({deep})** — every "
                       f"model's deepest slice was withheld, so the depth this soak exists "
                       f"to probe was not scored."
                       + (f" The overall line above is pooled across shallower depths and "
                          f"says nothing about drift at {deep}." if worst else
                          " The pooled gap was withheld too, so this run has no verdict at "
                          "any depth."))
        if worst and worst.passed and not deep_withheld and (
                deepest is None or deepest.passed):
            out.append("- No depth-correlated comprehension drift within tolerance — "
                       "chained diffs up to the tested depth read as well as fulls.")
        elif not deep_withheld and (worst or deepest):
            # `worst or deepest` is the guard un-nesting this block made necessary: with
            # neither, NOTHING was scored, and "comprehension drifts beyond tolerance" is a
            # finding rather than the absence of one. The NO VERDICT line above already
            # said what happened; a conclusion here would contradict it.
            passing = [d for d in depths if d != deep] if worst and worst.passed else []
            out.append("- Comprehension drifts beyond tolerance somewhere in the chain — "
                       + ("keep the keyframe interval at or below the deepest PASSING "
                          "depth." if passing else
                          "no tested depth published a passing gap, so this run does not "
                          "identify a safe keyframe interval."))
    out.append("")
    return "\n".join(out)


def _per_transform_table(results: dict, summary: dict[str, dict[str, float]]) -> list[str]:
    """terse-form accuracy pooled by stressed transform, across models.

    A DISPLAY table, not a gate: it pools one arm at a time and computes no form-vs-control
    gap, which is why it is on `test_gap_gate_boundary`'s `_form_stats` allowlist. It lives
    in its own function so that allowlist entry covers these two pooled columns and not the
    whole of `build_fluency_report`, whose verdict must stay behind `best_arm_gap`."""
    by_tf: dict[str, list[dict]] = {}
    for model, rows in results.items():
        # Skip the models whose per-model numbers were withheld. Publishing them here,
        # pooled and unannotated, would reprint the same corrupt counts the table above
        # just refused to show — and this is the table a reader uses to decide "restrict
        # the policy to the transforms that held", so a depressed row here becomes a
        # policy change made on a backend outage.
        if summary.get(model, {}).get("rows_untrustworthy"):
            continue
        # PAIRED rows only. These columns pool one arm at a time — so they are not a gap
        # and stay on the `_form_stats` allowlist — but pooling UNPAIRED rows reintroduced
        # exactly the bias this change removes elsewhere: an arm that lost the hard
        # questions of one transform is divided by its own smaller denominator and reads
        # inflated. This is the table whose own comment says a reader uses it "to decide
        # 'restrict the policy to the transforms that held'", so a flattered row here
        # becomes a policy decision.
        for r in paired_rows(rows, "terse_ok", "primer_ok", "raw_ok"):
            by_tf.setdefault(r["transform"], []).append(r)
    if not by_tf:
        return []
    out = [
        "## terse-form accuracy by stressed transform",
        "",
        "Which transform, if any, costs comprehension. `table+dict` rows resolve a "
        "`~N` alias; `table` rows map a column position to a value.",
        "",
        "| Transform | n | terse | terse+primer |",
        "|---|---|---|---|",
    ]
    for tf, rs in sorted(by_tf.items()):
        tacc, _ = _form_stats(rs, "terse_ok")
        pacc, _ = _form_stats(rs, "primer_ok")
        # `n/a`, not 0%, for an arm this responses file never collected (#283) — the same
        # rule the model table applies to its own primer/inline cells. This is the table a
        # reader uses to "restrict the policy to the transforms that held", so a 0% here for
        # a question nobody asked is a policy change made on an absence.
        p_cell = f"{pacc:.0%}" if arm_measured(rs, "primer_ok") else "n/a"
        out.append(f"| {tf} | {len(rs)} | {tacc:.0%} | {p_cell} |")
    out.append("")
    return out


def build_dropeval_report(results: dict, accept_degraded: bool = False) -> str:
    """Render the drop-to-retrieve behavioral eval: does a real tool-calling model call
    `terse.retrieve` when a dropped field is needed (recall), and leave it alone when it
    isn't (precision / no-overfetch)?

    `results` is {model: [row,...]} from dropeval.run_drop_fluency; each row carries
    `kind` ("recall"|"precision") plus retrieve_ok/answer_ok/handle_ok success counts
    over `trials`. The verdict gates on the WORST model across all three metrics
    (principle #24) — a policy that's unsafe for one model in the fleet is unsafe,
    full stop — reusing the same worst-case-gap machinery as build_diff_report/
    build_fluency_report, with a 100%-ideal control (a real tool call either happens
    correctly or it doesn't; there's no "raw form" to compare against here).
    """
    out: list[str] = ["# terse drop-to-retrieve behavioral eval", ""]
    out += [
        "Does a real tool-calling model call `terse.retrieve` when a `__terse_dropped__`",
        "marker matters (recall), and leave it alone when it doesn't (precision /",
        "no-overfetch)? Ground truth is deterministic; the loop mirrors the proxy's real",
        "2-turn retrieve protocol exactly (same primer, same tool, same miss string).",
        "",
    ]
    if not results or not any(results.values()):
        out += [
            "No tool-capable model answers, or no drop-marked record payloads in the",
            "corpus — set a policy with a `drop-to-retrieve` field and configure a model",
            "(TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS), then re-run",
            "`terse fluency --drop-eval --policy <file>`.",
            "",
        ]
        return "\n".join(out)

    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a "
        "question-clustered (sandwich) bound.",
        "",
        "| Model | recall q | retrieve-recall | precision (no-overfetch) | final-accuracy "
        "| control (no drop) | handle-accuracy | failed calls |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # A call that never reached the model scores identically to a model that declined to
    # retrieve, so the error count is reported next to the accuracy it can counterfeit and
    # suppresses the verdict outright past a threshold — a broken harness must not be
    # readable as a behavioral result (this is exactly how the `terse.retrieve` tool-name
    # 400 produced a clean 0%-recall FAIL for every model, see dropeval._oai_name).
    err_by_model: dict[str, tuple[int, int]] = {}
    # The table READS the decision; it does not recompute it. It used to hold a third copy
    # of the per-model gate math (`dropeval_gap_rows` had the second), under a comment
    # promising "the table and the verdict cannot disagree about which control they used" —
    # a promise two independent copies were in no position to keep.
    #
    # Two things are still computed here rather than read. `handle_ok` is a display column
    # no gate reads. `errs`/`attempts` are NOT: the same two sums are computed again inside
    # `inconclusive_models`, which is a gate, so the "failed calls" column and the
    # INCONCLUSIVE threshold rest on two independent copies of one expression. Pre-existing
    # on `main` and left alone here rather than claimed away — folding it in means the
    # verdict carrying a per-model error count, which is a wider change than #342's scope.
    v = dropeval_verdict(results, accept_degraded=accept_degraded)
    for model, rows in results.items():
        if not rows:
            # A row of `n/a`, not a skipped line. The verdict now WITHHOLDS this model
            # rather than dropping it, so omitting it from the table would leave a model
            # named in the verdict bullets and absent from the numbers above them — the
            # #300-finding-5 defect, one cause over.
            out.append(f"| `{model}` | 0 | n/a | n/a | n/a "
                       f"| {REASON_LABEL['empty']} | n/a | 0/0 |")
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        hacc, hse = _form_stats(recall_rows, "handle_ok") if recall_rows else (0.0, 0.0)
        # `.get`: a model with no rows of a kind is withheld from that gate, not scored at
        # a fabricated 0% (see `dropeval_verdict`). The cell says so rather than printing a
        # number the verdict declined to publish.
        cells = {}
        for mech in ("recall", "precision"):
            gate = v.gates[model].get(mech)
            cells[mech] = (f"{gate[0]:.0%} ±{_ci(gate[1]) * 100:.0f}" if gate
                           else "not gated")
        errs = sum(r.get("errors", 0) for r in rows)
        attempts = sum(r.get("attempts", r.get("trials", 1)) for r in rows)
        err_by_model[model] = (errs, attempts)
        # Both cells go to "not gated" together, but the control cell names the ACTUAL
        # reason rather than assuming "not run": without a control there is no paired
        # subset, so the final-accuracy number would be computed over a different question
        # set than the one the column header implies. Printing a bare 100% under "control
        # (no drop)" is precisely the misreading #269 is about — and printing "not run" for
        # a control that ran and errored out is a different misreading #300 is about.
        scored = v.gates[model].get("accuracy")
        if scored is None:
            reason = v.metrics["accuracy"].excluded[model]
            acc_cell = "not gated"
            ctl_cell = "not run" if reason == "no control arm" else REASON_LABEL[reason]
        else:
            aacc, ase, cacc, _ = scored
            acc_cell = f"{aacc:.0%} ±{_ci(ase) * 100:.0f}"
            ctl_cell = f"{cacc:.0%}"
        out.append(f"| `{model}` | {len(recall_rows)} | {cells['recall']} "
                   f"| {cells['precision']} | {acc_cell} "
                   f"| {ctl_cell} | {hacc:.0%} ±{_ci(hse) * 100:.0f} "
                   f"| {errs}/{attempts} |")
    out.append("")
    broken = {m: (e, a) for m, (e, a) in err_by_model.items() if e}
    if broken:
        out += ["> **Model calls failed** — these rows measure the harness, not the model: "
                + ", ".join(f"`{m}` {e}/{a}" for m, (e, a) in sorted(broken.items())) + ".",
                ""]
        # WHICH ARM lost the calls, not just how many. #299: the treatment runs a two-turn
        # retrieve protocol against the control's one turn, so it should fail first under a
        # token-budget stop — and that skews the gap toward "the drop is harmless" or
        # "harmful" depending on which side thins out. Reporting only the total invites the
        # cause to be guessed, which is exactly what happened to a real run.
        split = []
        for model, rows in sorted(results.items()):
            t = sum(r.get("treatment_errors", 0) for r in rows)
            c = sum(r.get("control_errors", 0) for r in rows)
            if t or c:
                split.append(f"`{model}` treatment {t} / control {c}")
        if split:
            out += ["> **Where they failed** (per arm — the attrition #299 is about): "
                    + ", ".join(split) + ". A large imbalance means the paired subset is "
                    "selected by which arm survived, and the gap above is biased in that "
                    "arm's favour; roughly equal counts are what infrastructure failure "
                    "looks like and leave the survivors an approximately random sample.", ""]
        # How many questions actually SURVIVED pairing. This, not the failure rate, is what
        # decides whether a degraded run still means anything: `paired_rows` needs every
        # trial complete on BOTH arms, so a 50%-loss run can leave 15 usable questions or
        # 2, and the failure percentage alone cannot tell them apart. Printed whenever any
        # call failed, because that is exactly when a reader needs it.
        surviving = []
        for model, rows in sorted(results.items()):
            g = _accuracy_gate(rows)
            if not g.excluded:
                surviving.append(f"`{model}` {len(g.rows)}/{len(rows)}")
            elif g.excluded == "underpowered":
                # `g.rows` is empty for every exclusion, so recompute: this is the one
                # reason whose entire remedy is a NUMBER, and the reader was given none.
                pr = paired_rows(rows, "answer_ok", "control_ok")
                surviving.append(f"`{model}` {len(pr)}/{len(rows)}")
        if surviving:
            out += ["> **Questions surviving the pairing**: " + ", ".join(surviving)
                    + ". The gap is computed over these only — read it as accuracy among "
                    "jointly-completed questions, and note that at small n one question "
                    "moves it a long way (#297).", ""]

    out += ["## Verdict", ""]
    # ONE decision, rendered here and by `build_terminal_dropeval_report` — see
    # `DropevalVerdict`. This block chooses words; it does not choose a verdict.
    if v.degraded_accepted:
        # The operator has asserted the losses have a known, model-independent cause (a
        # gateway restart, a local rate limit). That is a claim the harness cannot verify,
        # so it is recorded in the verdict rather than silently honoured — and the arm
        # split above is the evidence that decides whether it is credible: symmetric
        # losses leave the survivors an approximately random sample, a skew does not.
        out += ["> **Degraded run accepted** (`--accept-degraded`) — "
                + ", ".join(f"`{m}` failed {e}/{a} calls" for m, (e, a) in
                            sorted(v.degraded_accepted.items()))
                + ". The verdict below is computed over the surviving questions only. It is "
                "valid ONLY if the failures were independent of the model and of the arm; "
                "check the per-arm split and the surviving-question counts above before "
                "citing it.", ""]
    if v.inconclusive:
        # Half of a model's calls failing means its accuracy columns are mostly counting
        # transport errors. Refuse to render a pass/fail rather than let the run be cited.
        # The sentence comes from the shared `dropeval_directive_line`, like every other
        # verdict sentence — the chart used to word this one itself, so the one case an
        # operator reaches on a broken backend was the one case the two renderers did not
        # share a string.
        out += ["- " + dropeval_directive_line(v), ""]
        return "\n".join(out)
    for metric, label, control_label in DROPEVAL_METRICS:
        worst = v.metrics[metric].worst
        if worst is not None:
            out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, label, control_label))
    # One bullet per (metric, reason), each naming its own models. A withheld model is
    # named here AND has already raised `v.directive` off SHIP, so the two halves of
    # "excluded" — the prose and the gate — cannot drift apart.
    out += dropeval_exclusion_bullets(v)
    out.append("- " + dropeval_directive_line(v))
    out.append("")
    return "\n".join(out)


def build_fluency_report(results: dict, token_rows: list[dict[str, Any]]) -> str:
    """Render the format-fluency eval: does the model read terse as well as raw JSON?

    `results` is {model: [scored_row,...]} from fluency.run_fluency / score_pack. Each
    row has raw_ok / terse_ok / primer_ok plus qtype/transform. Scoring is PAIRED, so a
    regression (raw right, terse wrong) is a first-class column. The verdict gates on
    the worst model, not the mean (principle #24): a format that helps one model but
    breaks the consumer is a regression, not a wash.
    """
    out: list[str] = ["# terse format-fluency eval", ""]
    out += [
        "Can a model read terse's compressed form as accurately as raw JSON?",
        "Ground truth is deterministic (no LLM judge); scoring is paired per question.",
        "",
    ]
    if not results or not any(results.values()):
        out += [
            "No model answers provided. Configure a backend and re-run:",
            "  - broker pool / loopback gateway: set TERSE_FLUENCY_BASE_URL / TERSE_FLUENCY_API_KEY / TERSE_FLUENCY_MODELS",
            "  - offline: `terse fluency` writes an eval pack you can drive by hand and score later.",
            "",
        ]
        return "\n".join(out)

    if token_rows:
        rt = sum(r["raw_tok"] for r in token_rows if r.get("raw_tok"))
        tt = sum(r["terse_tok"] for r in token_rows if r.get("terse_tok"))
        if rt:
            out += [
                f"Token cost over {len(token_rows)} record-shaped payloads: "
                f"raw {rt} -> terse {tt} ({_pct(tt - rt, rt)}). "
                "Comprehension is the price of that saving — measured below.",
                "",
            ]

    # --- per-model accuracy by form ---
    # Trial count is read from the rows (multi-trial via `--trials`); a `±` column shows
    # the 95% half-width so the verdict is a bound, not a single noisy point. A question
    # "regresses" when raw is fully right across its trials but terse is not.
    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model and form",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a "
        "question-clustered (sandwich) bound on the accuracy.",
        "",
        "| Model | q | raw | terse | terse+primer | terse+inline | regressions | "
        "primer recovers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summary: dict[str, dict[str, float]] = {}
    reasons: dict[str, ExclusionReason | None] = {}  # model -> `ArmGap.excluded`, for the verdict split
    for model, rows in results.items():
        n = len(rows)
        if not n:
            continue
        # #168: the same primer delivered with the RESULT instead of at `initialize`. Absent
        # from older result files, which predate the arm — rendered as `n/a`, never as 0%,
        # which would read as "inline comprehension collapsed" rather than "not measured".
        # ALL rows, not any: a result file that MIXES pre-arm rows (no `inline_ok`) with
        # post-arm ones would otherwise pass this gate and then crash in `_form_stats`,
        # which indexes `r[form]` unconditionally — n/a is the correct degrade for a
        # partially-measured model, not a KeyError.
        has_inline = bool(rows) and all("inline_ok" in r for r in rows)
        # One paired subset for the whole row: the columns and the gap are computed over
        # the same questions, or the table argues against its own verdict (#280 F3).
        # `inline` is display-only — see `_gap`.
        g = best_arm_gap(rows, ["terse_ok", "primer_ok"], "raw_ok",
                         ("inline_ok",) if has_inline else ())
        racc, rse = g.arms.get("raw_ok", (0.0, 0.0))
        tacc, tse = g.arms.get("terse_ok", (0.0, 0.0))
        pacc, pse = g.arms.get("primer_ok", (0.0, 0.0))
        iacc, ise = g.arms.get("inline_ok", (0.0, 0.0))
        # Counted over the PAIRED subset (#280 F5): a question one arm never answered is
        # neither a regression nor a primer recovery, and counting it as one made these
        # columns disagree with the gap beside them.
        regr = sum(1 for r in g.rows if int(r["raw_ok"]) == r.get("trials", 1)
                   and int(r["terse_ok"]) < r.get("trials", 1))
        rec = sum(1 for r in g.rows if int(r["terse_ok"]) < r.get("trials", 1)
                  and int(r["primer_ok"]) == r.get("trials", 1))
        # Transport failures (#263): calls that never reached the model. Distinct from a
        # wrong answer, and NOT the same test as the raw==0 control below — a partial
        # rate limit lets some calls through, so it depresses every arm's accuracy while
        # leaving `raw` non-zero, and the control never fires. That is the case that
        # silently publishes a comprehension number for a backend that was half down.
        fails = sum(int(r.get("fails", 0)) for r in rows)
        attempts = sum(int(r.get("attempts", 0)) for r in rows)
        # "unmeasured" now covers `unpaired` too — a run where the backend was up but one
        # arm lost too much of the question set is equally unpublishable, and it reaches
        # the reader through the same `n/a` row rather than a second vocabulary.
        # Any reason that withholds this model's numbers, not just the transport one:
        # #334's `"underpowered"` must render the same `n/a` row, or the table prints
        # percentages the verdict below refuses to use. `reasons[model]` keeps the
        # distinction for the prose that follows.
        # Two different questions, and conflating them let an exclusion flatter a pooled
        # number. `unmeasured` decides whether this model gets percentages of its own;
        # `rows_untrustworthy` decides whether its ROWS may be pooled with other models'
        # in `_per_transform_table`. An underpowered model's rows are fully paired and
        # perfectly good evidence — it is the per-model CONCLUSION that is unsupported,
        # not the measurements, and that table is not a conclusion. Dropping it moved a
        # published, decision-bearing figure from 72% to 90% in the flattering direction.
        unmeasured = g.excluded in ("unmeasured", "underpowered")
        reasons[model] = g.excluded
        # "n" is the GENERATED question count; nothing in `src/` reads it (the table
        # prints `len(g.rows)`), kept only because result files carry the shape.
        summary[model] = {"n": n, "raw": racc, "raw_se": rse,
                          "terse": tacc, "terse_se": tse, "primer": pacc, "primer_se": pse,
                          "fails": fails, "attempts": attempts, "unmeasured": unmeasured,
                          "rows_untrustworthy": g.excluded == "unmeasured",
                          # NOT `len(g.rows)`: `_gap` returns `rows=[]` for every
                          # exclusion, and only excluded models reach the paragraph that
                          # prints this, so that read was a dead constant 0 — which reads
                          # as "nothing paired at all", the OTHER reason.
                          "paired": len(paired_rows(rows, "terse_ok", "primer_ok", "raw_ok")),
                          "gap_form": g.form_acc, "gap_form_se": g.form_se}
        if has_inline:
            summary[model].update({"inline": iacc, "inline_se": ise})
        inline_cell = (f"{iacc:.0%} ±{_ci(ise) * 100:.0f}" if has_inline else "n/a")
        # Same rule for the primer arm, which `score_pack` can now report as never collected
        # (#283): `_form_stats` would render its empty sample as a confident `0% ±0`, and
        # "primer recovers" would silently mean "recovered none of them" for an arm that was
        # never asked. Live-harness rows always answer True here, so this is `n/a` only where
        # a responses file really did omit the form.
        has_primer = arm_measured(rows, "primer_ok")
        primer_cell = (f"{pacc:.0%} ±{_ci(pse) * 100:.0f}" if has_primer else "n/a")
        rec_cell = str(rec) if has_primer else "n/a"
        if unmeasured:
            # No percentages at all for a model that did not answer. A number here would
            # be read as comprehension no matter how it is footnoted — the same reason the
            # inline arm renders `n/a` rather than 0%, and the same unknown-is-not-zero
            # rule the ledger applies to untokenized records.
            out.append(f"| `{model}` | {n} | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            out.append(
                # `len(g.rows)` — the paired exam these percentages are actually over.
                f"| `{model}` | {len(g.rows)} | {racc:.0%} ±{_ci(rse) * 100:.0f} "
                f"| {tacc:.0%} ±{_ci(tse) * 100:.0f} "
                f"| {primer_cell} | {inline_cell} | {regr} | {rec_cell} |"
            )
    out.append("")
    unreachable = {m: s for m, s in summary.items() if s.get("unmeasured")}
    out += _not_measured_lines({
        m: (reasons.get(m), int(s["fails"]), int(s["attempts"]), int(s["paired"]))
        for m, s in unreachable.items()})
    # A model that lost SOME calls but stayed under the bar still publishes numbers — say
    # so, because those percentages are over a smaller denominator than the run intended
    # and a reader comparing across models deserves to know which ones were degraded.
    degraded = {m: s for m, s in summary.items()
                if s.get("fails") and not s.get("unmeasured")}
    if degraded:
        out += [
            "Partially degraded (a question is dropped from every GATED arm unless all of "
            "them completed all of its trials, so the `q` column is the paired exam, not "
            "the questions generated; `terse+inline` is display-only and is not part of "
            "that pairing. The losses below are calls, "
            "not scored as wrong): "
            + ", ".join(f"`{m}` ({s['fails']}/{s['attempts']})"
                        for m, s in sorted(degraded.items()))
            + ".",
            "",
        ]

    out += _per_transform_table(results, summary)

    # --- verdict: gate on the worst model ---
    out += ["## Verdict", ""]
    # Raw JSON is the control: a model that can't read RAW (0%) is a backend/config
    # failure (bad model id, refusals), not a terse-comprehension result — exclude it
    # from the gate, but say so, so a broken run can't masquerade as a verdict.
    broken = [m for m, r in reasons.items() if r == "broken control"]
    # Excluded for the same reason, by a different and EARLIER signal: the raw==0 control
    # only catches a TOTAL failure, and only once it has already been scored as if the
    # model answered wrongly. A counted unanswered call catches the partial case too, which
    # is the one that reaches a plausible-looking verdict (#263) — including, since #268,
    # a model that WAS reached and produced no content.
    # Split by REASON, not lumped. This line said "calls went unanswered" for every
    # withheld model, which contradicted the "**Not compared** — the backend answered"
    # paragraph printed ten lines above it in the same document.
    unmeasured_models = [m for m, r in reasons.items() if r == "unmeasured"]
    underpowered_models = [m for m, r in reasons.items() if r == "underpowered"]
    gated = {m: s for m, s in summary.items() if reasons.get(m) is None}
    if broken:
        out.append(f"- Excluded (raw control failed — backend/config error, not comprehension): "
                   f"{', '.join(f'`{m}`' for m in broken)}.")
    if unmeasured_models:
        # Reads the shared label rather than restating it: this line asserted "calls
        # went unanswered" as the settled cause, which #332 made false — the reason now
        # also covers a backend that answered nearly everything but whose losses left no
        # question complete on both arms. Pinned by
        # `test_the_fluency_verdict_does_not_assert_a_dead_backend`.
        out.append(f"- Excluded ({REASON_LABEL['unmeasured']} — not measured): "
                   f"{', '.join(f'`{m}`' for m in unmeasured_models)}.")
    if underpowered_models:
        # Its own line. Folded into the one above it would tell a reader whose backend
        # answered every call that its calls went unanswered; omitted — which is what
        # happened until this review — the model vanishes from the verdict unnamed.
        out.append(f"- Excluded ({REASON_LABEL['underpowered']} — not concluded): "
                   f"{', '.join(f'`{m}`' for m in underpowered_models)}.")
    # best terse-side form per model, carrying its own SE for the gap's confidence interval.
    # gap CI: raw and the best form are over the same questions (not independent), so
    # √(se_raw²+se_best²) is a conservative over-estimate of the gap's SE — the honest
    # direction for a bound that gates a ship decision.
    # Straight from `best_arm_gap`, which already picked the best arm over the paired
    # subset. This used to re-derive the best-of math here, a second copy of
    # `fluency_gap_rows`' body that could (and did) drift from it.
    gap_rows = {m: (s["gap_form"], s["gap_form_se"], s["raw"], s["raw_se"])
                for m, s in gated.items()}
    worst = _worst_case_gap(gap_rows)
    if worst:
        helps = sum(1 for s in gated.values() if s["primer"] > s["terse"] + 1e-9)
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "best terse-form", "raw"))
        # `gap_ci` combines the two arms' SEs as if independent (√(fse²+cse²)), which
        # overstates the true variance of a PAIRED gap whenever the arms are positively
        # correlated (the normal case — question difficulty is shared) — before #297 that
        # didn't matter because both SEs were ~0 by default, but the cluster-robust SE is
        # routinely several points wide, and this bound is wide enough in practice to call
        # a genuine regression "noise" (the correct paired CI on the same data can be
        # several times tighter and not cross the gap at all). A bound that is only ever
        # an OVER-estimate cannot support a "not distinguishable" conclusion on EITHER
        # side of the tolerance line, so neither branch below claims one — only that the
        # gap is smaller than a loose ceiling on its own noise. (A proper paired
        # cluster-robust SE on the per-question difference would let this section make a
        # real significance claim — tracked as a follow-up, not folded into #297's scope.)
        if worst.gap_ci > 1e-9 and abs(worst.gap) < worst.gap_ci:
            # More questions tighten this bound without limit; more trials also tighten
            # it, but only down to a between-question floor they can't cross (they
            # resample k, not the question set) — so this is neither "raise `--trials`"
            # (#297's own estimator makes that false as the SOLE lever) nor "trials never
            # help" (also false).
            verdict_word = "This passing gap" if worst.passed else "Note: this gap"
            out.append(f"- {verdict_word} is smaller than a (loose, conservative) "
                       "upper-bound estimate of its own noise floor — treat the verdict "
                       "above as real, not yet as a tightly measured margin; more "
                       "questions would tighten this bound further than more trials would.")
        out.append(f"- The primer improves terse-form accuracy for {helps}/{len(gated)} model(s).")
        if worst.passed:
            out.append("- terse's compressed form preserves comprehension within tolerance — "
                       "the proxy's in-place rewrite holds for the tested models.")
        else:
            out.append("- Comprehension regresses beyond tolerance — the proxy's in-place rewrite "
                       "is not safe to ship as-is for the worst model; prefer the primer or restrict "
                       "the policy to the transforms that held.")
    elif not gated:
        # Nothing to gate on. Say so loudly: silence here is how a run that measured
        # NOTHING gets read as a run that found nothing wrong (#263). Absence of a
        # regression is not evidence of comprehension.
        #
        # Two different causes, named separately — claiming models were "excluded above"
        # when the exclusion lists are empty (every model simply produced zero rows, e.g.
        # a corpus that generated no questions) sends the reader hunting for a backend
        # failure that never happened.
        # `underpowered_models` belongs in this test: without it, a run whose models were
        # all fully scored and fully paired — merely short of the question floor — printed
        # "did the corpus generate questions?" about a corpus that generated them and
        # answered every one, contradicting the paragraph four lines above.
        why = ("Every model was excluded above"
               if (broken or unmeasured_models or underpowered_models)
               else "No model produced any scored rows (did the corpus generate questions?)")
        out.append(f"- **NO VERDICT — nothing was measured.** {why}, so this run says "
                   "nothing about comprehension either way. Fix that and re-run before "
                   "drawing any conclusion from it.")
    out.append("")
    return "\n".join(out)
