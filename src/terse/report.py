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
from types import MappingProxyType
from typing import Any, NamedTuple

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
      - rows with no `attempts` key. `score_pack` (`fluency/pack.py`) emits per-form counts
        that differ by COLLECTION DESIGN: an uneven hand-built pack may carry 3 raw replies
        and 2 terse ones for the same question, and #91 added those counters precisely so
        the sparser form is scored over its own denominator instead of being understated.
        Its `trials` is `max(...)` of the forms, not an attempt count, so every uneven row
        looks like a loss here. Voiding them would delete a documented collection mode to
        defend against a failure it cannot have — `score_pack` never calls a backend, so it
        has no transport to lose calls to.

    Pinned by `test_rows_without_per_form_counters_are_treated_as_fully_paired` and
    `test_an_uneven_score_pack_still_publishes`."""
    keys = [f[:-3] + "_trials" if f.endswith("_ok") else f for f in forms]
    out = []
    for r in rows:
        t = int(r.get("trials", 1))
        if "attempts" not in r or all(int(r.get(k, t)) == t for k in keys):
            out.append(r)
    return out


# Share of a model's calls that may fail before its numbers stop meaning anything (#263).
# NOT zero: a failed call is already excluded from its arm's denominator (see the
# `<form>_trials` keys `run_payload` emits), so a handful of transient 429s no longer
# depress an accuracy at all — voiding a whole model for one of them would discard an
# otherwise-complete multi-hour run, which is its own kind of wrong answer. What a
# threshold still has to catch is the backend that was substantially down, where the
# surviving sample is small and self-selected rather than merely smaller.
#
# THE DENOMINATOR IS TOTAL `attempts`, ACROSS EVERY ARM — not one arm's own calls. Every
# emitter sets `attempts = trials * <arm count>` (2 for the diff/codec harnesses, 4 for
# the payload one), so a single arm can lose over 40% of ITS calls in a two-arm run, or
# over 80% in a four-arm one, before this fires. `paired_rows`' docstring below works
# through "one of five question types lost" and calls it "20.0% of the arm — on the
# threshold": true of the arm, but it reaches this gate as 10% of attempts, nowhere near
# the line. The conclusion there still holds (the gate stays quiet, and `paired_rows` is
# what catches it) — the arithmetic offered for it does not. Tracked separately; do not
# reason about this gate's permissiveness from that paragraph.
#
# Value and the strictness of the comparison are pinned by
# `test_a_model_exactly_at_the_loss_share_is_still_measured` (#337): `>` -> `>=` survived
# a full-suite mutation, so nothing observed which side of the line a model fell on.
UNMEASURED_FAIL_SHARE = 0.20

# WHY THE PAIRING-LOSS GATE IS `not pr` AND NOT A SHARE (#332).
#
# The bug: `paired_rows` voids a whole ROW when either arm loses one trial of it, so loss
# is amplified from the call level to the question level. At three trials an arm, one lost
# call per question is 16.7% of the calls — under `UNMEASURED_FAIL_SHARE` — and 100% of the
# questions. `_unmeasured` stayed quiet, nothing was left to compare, both arms scored a
# flat 0.0 with an SE of 0.0, and the HTML banner rendered a green PASS reading `+0% ±0pt`:
# maximum confidence from no evidence at all.
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

    Two independent triggers, because they fail differently:
      1. any arm with ZERO completed trials — that arm cannot be computed at all, and
         `_form_stats` would report it as a flat 0.0 indistinguishable from real failure;
      2. more than `UNMEASURED_FAIL_SHARE` of calls lost — the sample that survived is
         both small and selected by which calls happened to get through.
    """
    if not rows:
        return False
    attempts = sum(int(r.get("attempts", 0)) for r in rows)
    fails = sum(int(r.get("fails", 0)) for r in rows)
    if not attempts:
        # Rows predating the counters (older result files) carry neither key. Absent is
        # not zero-failures, but it is also not evidence of failure — treat as measured,
        # exactly as this report did before the counters existed.
        return False
    # Arms are DISCOVERED from the rows, not hardcoded. The payload harness emits
    # raw/terse/primer/inline; the diff harnesses emit terse/diff. A fixed list would
    # silently skip every arm it did not name — which is how the diff-side report kept
    # publishing a verdict off a dead backend after the payload side stopped.
    for key in sorted({k for r in rows for k in r if k.endswith("_trials")}):
        if sum(int(r.get(key, 0)) for r in rows) == 0:
            return True
    return fails / attempts > UNMEASURED_FAIL_SHARE


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
    excluded: str | None
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
    `REASON_LABEL`'s note explains. #332 originally planned a distinct `"unpaired"` reason
    and never built it; that plan is retired rather than finished, because the two causes
    reach the reader as the same fact ("calls went unanswered, so there is no verdict")
    and a second vocabulary is one more thing six renderers can disagree about.

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
}


def exclusion_note(reasons: dict[str, str | None]) -> str:
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


def _not_measured_lines(withheld: dict[str, tuple[str | None, int, int, int]]) -> list[str]:
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
    by: dict[str | None, list[tuple[str, int, int, int]]] = {}
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
                + ". Either too many calls went unanswered, or enough of them did that no "
                  "question completed every trial on BOTH arms, leaving nothing comparable "
                  "— the counts above say which, and a low one means the second. An "
                  "unanswered call is not a wrong answer. Check stderr for a `returned no "
                  "content` line naming a `finish_reason` — `length` means raise "
                  "max_tokens, `content_filter` means the payload tripped a filter. If the "
                  "backend answered most calls, lower `--trials`: each extra trial is "
                  "another chance for a question to lose one and be dropped from both "
                  "arms.")
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
    #295 says the eval must be able to answer."""
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
                                          dict[str, str | None]]:
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
    excluded: dict[str, str | None] = {}
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
                                             dict[str, str | None]]:
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
    broken: dict[str, str | None] = {}
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


def dropeval_gap_rows(results: dict) -> tuple[dict[str, dict[str, tuple[float, float, float, float]]],
                                              dict[str, str | None]]:
    """Per-model (recall, precision, accuracy) gap-row tuples for build_dropeval_report
    and its terminal-bar companion. Recall and precision keep a fixed 100% ideal (se=0),
    which is correct for them — a tool call either happens or it doesn't, so neither is
    ever excluded. Accuracy routes through `_accuracy_gate`, which pairs against a
    measured control arm when one ran (#269). Same per-model math build_dropeval_report's
    own table loop uses, kept in one place so the two verdicts (markdown table, terminal
    chart) can never disagree.

    Returns (gap_rows, accuracy_excluded) — mirroring `diff_gap_rows`/`fluency_gap_rows`,
    unlike which this function's `out` is never itself empty for an excluded model (recall
    and precision still render); only the "accuracy" key is missing, and
    `accuracy_excluded` names why. `build_terminal_dropeval_report` used to swallow that
    reason — a model excluded from the accuracy gate vanished from the accuracy plot with
    no note, while the verdict text still said "the worst model" as if every model had
    been considered for every metric (review finding 5 on #300)."""
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    excluded: dict[str, str | None] = {}
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        out[model] = {
            "recall": (racc, rse, 1.0, 0.0),
            "precision": (pacc, pse, 1.0, 0.0),
        }
        # "accuracy" is ABSENT rather than zeroed when no control ran — a renderer that
        # iterates the metrics it finds then cannot draw a bar for a gap nobody measured.
        g = _accuracy_gate(rows)
        if not g.excluded:
            out[model]["accuracy"] = (g.form_acc, g.form_se, g.control_acc, g.control_se)
        else:
            excluded[model] = g.excluded
    return out, excluded


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
    unmeasured: dict[str, tuple[str | None, int, int, int]] = {}  # -> (why, fails, attempts, paired)
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
        out.append("- **NO VERDICT — nothing was scored.** Every model was withheld, so "
                   "this run says nothing about the diff form either way. The paragraph "
                   "above names each model and its reason — an unreachable backend, calls "
                   "lost until nothing paired, or too few questions to conclude from. The "
                   "remedies differ, and only the first is a backend problem.")
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
            "**Not measured** — too many calls went unanswered, so no accuracy "
            "is published for: "
            + ", ".join(
                f"`{m}` ({sum(int(r.get('fails', 0)) for r in rs)}/"
                f"{sum(int(r.get('attempts', 0)) for r in rs)} calls lost)"
                for m, rs in sorted(unmeasured.items()))
            + ". An unanswered call is not a wrong answer. Check stderr for a "
              "`returned no content` line naming a `finish_reason` — `length` means "
              "raise max_tokens, `content_filter` means the payload tripped a filter; "
              "otherwise re-run once the backend is reachable.",
            "",
        ]
    if withheld_depths:
        # An `n/a` row with no explanation is not a disclosure. Named here because the
        # deepest depth is exactly the one a soak exists to measure, and losing it silently
        # is how "no depth-correlated drift" gets printed about a depth nobody scored.
        # Grouped by reason: a slice withheld because its calls FAILED must not be
        # described as one where "the backend answered".
        for why in sorted({w for d in withheld_depths.values() for w in d}):
            at = ", ".join(
                f"`{m}` (depth {', '.join(str(d) for d in sorted(ds[why]))})"
                for m, ds in sorted(withheld_depths.items()) if why in ds)
            # The old branch tested `why == "x"`, a reason string nothing produces, so
            # the specific wording it guarded had been unreachable since #284 and every
            # withheld depth got the generic one. `"unpaired"` is set just above from
            # `_unmeasured(drows)` — local to this table, not an `ArmGap` reason — which
            # makes the precise branch reachable for the first time.
            lead = {
                "unpaired": "**Depths not compared** — the backend answered, but one arm "
                            "did not complete enough of the same questions at: ",
                "unmeasured": "**Depths not measured** — too many calls went unanswered "
                              "at: ",
            }.get(why, f"**{REASON_HEADING.get(why, 'Excluded')}** — "
                       f"{REASON_LABEL.get(why, why)} at: ")
            tail = (". Those depths are excluded from the verdict below rather than "
                    "scored on a question set the two arms did not share (#280)."
                    if why in ("unpaired", "unmeasured") else
                    ". Too few questions survived pairing at that depth for an absence "
                    "of drift to mean anything — check the `q` column against the "
                    "generated count before assuming none were lost.")
            out += [lead + at + tail, ""]

    gap_rows: dict[str, tuple[float, float, float, float]] = {}
    pooled_out: dict[str, str | None] = {}
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
                   "above name each model and its reason — an unreachable backend, calls "
                   "lost until nothing paired, or too few questions to conclude from.")
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
        out.append(f"| {tf} | {len(rs)} | {tacc:.0%} | {pacc:.0%} |")
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
    recall_gate: dict[str, tuple[float, float, float, float]] = {}
    precision_gate: dict[str, tuple[float, float, float, float]] = {}
    accuracy_gate: dict[str, tuple[float, float, float, float]] = {}
    # Populated only for EXCLUDED models, and only with the reason `_accuracy_gate`
    # actually returned. The verdict's "not gated" fallback used to say "no no-drop
    # control arm was run" unconditionally — true for "no control arm", false for e.g.
    # "broken control" (a control that ran, and whose entire arm errored out under
    # `--accept-degraded`), which made the report assert a control was never run in the
    # same breath as reporting how many of its calls it lost (review finding 2 on #300).
    accuracy_excluded: dict[str, str | None] = {}
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        hacc, hse = _form_stats(recall_rows, "handle_ok") if recall_rows else (0.0, 0.0)
        # Recall/precision keep the fixed 100% ideal — for them it is the right control, a
        # tool call either happens or it doesn't. Accuracy pairs against the measured
        # control arm when one ran (#269); `_accuracy_gate` owns that choice so the table
        # and the verdict cannot disagree about which control they used.
        recall_gate[model] = (racc, rse, 1.0, 0.0)
        precision_gate[model] = (pacc, pse, 1.0, 0.0)
        g = _accuracy_gate(rows)
        if not g.excluded:
            accuracy_gate[model] = (g.form_acc, g.form_se, g.control_acc, g.control_se)
        else:
            accuracy_excluded[model] = g.excluded
        aacc, ase, cacc = g.form_acc, g.form_se, g.control_acc
        errs = sum(r.get("errors", 0) for r in rows)
        attempts = sum(r.get("attempts", r.get("trials", 1)) for r in rows)
        err_by_model[model] = (errs, attempts)
        # Both cells go to "not gated" together, but the control cell names the ACTUAL
        # reason rather than assuming "not run": without a control there is no paired
        # subset, so the final-accuracy number would be computed over a different question
        # set than the one the column header implies. Printing a bare 100% under "control
        # (no drop)" is precisely the misreading #269 is about — and printing "not run" for
        # a control that ran and errored out is a different misreading #300 is about.
        if g.excluded:
            acc_cell = "not gated"
            ctl_cell = "not run" if g.excluded == "no control arm" \
                else REASON_LABEL.get(g.excluded, g.excluded)
        else:
            acc_cell = f"{aacc:.0%} ±{_ci(ase) * 100:.0f}"
            ctl_cell = f"{cacc:.0%}"
        out.append(f"| `{model}` | {len(recall_rows)} | {racc:.0%} ±{_ci(rse) * 100:.0f} "
                   f"| {pacc:.0%} ±{_ci(pse) * 100:.0f} | {acc_cell} "
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
    # Half of a model's calls failing means its accuracy columns are mostly counting
    # transport errors. Refuse to render a pass/fail rather than let the run be cited.
    inconclusive = inconclusive_models(results)
    if inconclusive and accept_degraded:
        # The operator has asserted the losses have a known, model-independent cause (a
        # gateway restart, a local rate limit). That is a claim the harness cannot verify,
        # so it is recorded in the verdict rather than silently honoured — and the arm
        # split above is the evidence that decides whether it is credible: symmetric
        # losses leave the survivors an approximately random sample, a skew does not.
        out += ["> **Degraded run accepted** (`--accept-degraded`) — "
                + ", ".join(f"`{m}` failed {e}/{a} calls" for m, (e, a) in
                            sorted(inconclusive.items()))
                + ". The verdict below is computed over the surviving questions only. It is "
                "valid ONLY if the failures were independent of the model and of the arm; "
                "check the per-arm split and the surviving-question counts above before "
                "citing it.", ""]
        inconclusive = {}
    if inconclusive:
        out += ["- **INCONCLUSIVE** — "
                + ", ".join(f"`{m}` failed {e}/{a} model calls" for m, (e, a) in
                            sorted(inconclusive.items()))
                + ". Fix the backend and re-run; no behavioral claim can be made from this.",
                ""]
        return "\n".join(out)
    recall_worst = _worst_case_gap(recall_gate)
    precision_worst = _worst_case_gap(precision_gate)
    accuracy_worst = _worst_case_gap(accuracy_gate)
    if recall_worst and precision_worst:
        out.append(_format_worst_case_line(recall_worst, _GAP_TOLERANCE, "retrieve-recall",
                                           "ideal (100%)"))
        out.append(_format_worst_case_line(precision_worst, _GAP_TOLERANCE, "no-overfetch",
                                           "ideal (100%)"))
        if accuracy_worst:
            # The control label is not cosmetic: "vs ideal (100%)" and "vs no-drop control"
            # are different claims, and #269 exists because a reader could not tell which
            # one the verdict was making.
            out.append(_format_worst_case_line(accuracy_worst, _GAP_TOLERANCE,
                                               "final-accuracy", "no-drop control"))
        elif accuracy_excluded and set(accuracy_excluded.values()) != {"no control arm"}:
            # At least one model was excluded for a reason OTHER than "a control never
            # ran" — e.g. "broken control", when `--accept-degraded` accepted a run whose
            # control arm errored out entirely. The old fallback said "no no-drop control
            # arm was run" unconditionally here, which is false in that case: the control
            # DID run, and the "Where they failed" paragraph above already says how many
            # of its calls it lost. Name the real reason instead of contradicting that
            # paragraph two sentences later (review finding 2 on #300).
            # The tail has to match the reason too, not just the opening. #300's finding 2
            # stopped this branch claiming "no control was run" — and the sentence it kept
            # still asserts an "unrun 100%", which is the same claim, and is false whenever
            # the control ran and the exclusion was about sample size instead.
            underpowered = set(accuracy_excluded.values()) == {"underpowered"}
            out.append("- **final-accuracy: not gated** — " + exclusion_note(accuracy_excluded)
                       + (". The control arm ran; there were simply too few paired "
                          "questions for an absence of drop-caused loss to mean anything. "
                          "Generate more questions and re-run — this is not evidence "
                          "either way." if underpowered else
                          ". It is scored by JSON value-equality against the full original "
                          "value, and a model given the UN-dropped payload does not "
                          "reproduce a long prose field verbatim either — gating that "
                          "against an unrun 100% measures verbatim reproduction and bills "
                          "it to the drop (#269)."))
        else:
            out.append("- **final-accuracy: not gated** — no no-drop control arm was run, "
                       "so there is nothing to compare the drop against. It is scored by "
                       "JSON value-equality against the full original value, and a model "
                       "given the UN-dropped payload does not reproduce a long prose field "
                       "verbatim either — gating that against an unrun 100% measures "
                       "verbatim reproduction and bills it to the drop (#269). Re-run "
                       "without `--no-control` to gate it.")
        if recall_worst.passed and precision_worst.passed and (
                accuracy_worst is None or accuracy_worst.passed):
            if accuracy_worst:
                out.append("- Recall, precision, and final accuracy all clear tolerance for "
                           "the worst model — safe to enable drop-to-retrieve.")
            else:
                # "safe to enable" is not supported by mechanism metrics alone. Recall and
                # no-overfetch say the model OPERATES the protocol correctly; they say
                # nothing about whether the answer it ends up with is right. Calling that
                # "safe" is the same over-claim in the opposite direction to the one #269
                # opened for.
                # Reason-aware for the same reason the line above it is: telling an
                # operator to add a control arm that is already running is #300's finding
                # 2, one paragraph further down. Membership, not set-equality: a MIXED
                # exclusion set still contains an underpowered model whose control ran,
                # so the blanket "re-run with a control" is false for that half.
                has_underpowered = "underpowered" in set(accuracy_excluded.values())
                out.append(
                    "- **INCONCLUSIVE for enabling** — recall and no-overfetch clear "
                    "tolerance for the worst model, so the mechanism works, but final "
                    "accuracy was not gated: the OUTCOME impact of dropping is "
                    "unmeasured. "
                    + ("Generate more questions and re-run — the control arm is "
                       "already on." if has_underpowered else
                       "Re-run with the no-drop control arm before enabling "
                       "drop-to-retrieve on this evidence."))
        else:
            out.append("- At least one metric misses tolerance for its worst model — keep "
                       "drop-to-retrieve off until this improves.")
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
    reasons: dict[str, str | None] = {}  # model -> `ArmGap.excluded`, for the verdict split
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
                f"| {pacc:.0%} ±{_ci(pse) * 100:.0f} | {inline_cell} | {regr} | {rec} |"
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
