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

_GAP_TOLERANCE = 0.05  # shared pass/fail tolerance for both worst-case verdict gates below


def _form_stats(rows: list[dict[str, Any]], form: str) -> tuple[float, float]:
    """(accuracy, standard_error) for one form over rows carrying success COUNTS.

    accuracy = Σsuccesses / Σtrials. SE is the pooled binomial SE of that estimator:
    each row is t trials of a Bernoulli with p̂=k/t, so Var(total successes)=Σ t·p̂(1-p̂)
    and SE(acc)=√Var / Σt. This is stable at the realistic small trial count (N=2–3),
    where an empirical std across N whole-eval runs would be pure noise. At trials=1
    every p̂∈{0,1} → SE=0, so single-trial runs report exactly as before.
    """
    tot_t = tot_k = 0
    var = 0.0
    for r in rows:
        # Prefer a per-form trial count ("terse_ok" -> "terse_trials") when the row
        # carries one — an uneven hand-built pack can collect fewer replies for one form,
        # and dividing that form's successes by the shared per-row `trials` would
        # understate it. Falls back to the shared count, so the live/uniform path (no
        # per-form keys) reports exactly as before.
        t_key = form[:-3] + "_trials" if form.endswith("_ok") else ""
        t = r.get(t_key, r.get("trials", 1))
        k = int(r[form])
        tot_t += t
        tot_k += k
        if t > 0:
            p = k / t
            var += t * p * (1 - p)
    if tot_t == 0:
        return 0.0, 0.0
    return tot_k / tot_t, math.sqrt(var) / tot_t


def _ci(se: float) -> float:
    """95% half-width in accuracy units."""
    return 1.96 * se


# How ONE-SIDED the pairing losses may be before a gap stops being publishable.
#
# The refusal measures asymmetry, not volume, because asymmetry is what causes the harm.
# `paired_rows` already removes the BIAS outright — after pairing, both arms sit the
# identical exam whatever fraction was dropped. What is left to guard against is the
# survivors being SELECTED, and the mechanism that selects them is loss correlated with an
# arm: a token-budget stop kills the longest prompt first, and one arm's prompt is
# systematically longer than the other's. WHICH arm depends on the family — the diff form
# in a `--diff` run (`PREVIOUS RESULT … UPDATE …` versus the compressed payload alone), the
# uncompressed `raw` control in a fluency run, since shrinking that is the product. So both
# directions are counted; see `loss_asymmetry`.
#
# A volume bar cannot tell those apart. It counted a run where both arms flaked equally the
# same as one where only one arm truncated — so uncorrelated 429 background voided runs it
# had no reason to. Simulated over 1000 runs x 20 questions x 4 arms at `--trials 3`, a 1%
# per-call failure rate withheld 9% of models under the volume rule, 2% withheld 41%, and 5%
# withheld 98%, while `_unmeasured` never fired at any of them. The rule below withholds
# 0.6% / 6.7% / 58% on the same inputs — better than the volume bar at every rate, and the
# 5% figure is a backend failing one call in twenty, which is not a healthy run.
#
# Calibration is unchanged where it was actually measured. The #268 case — a model
# answering its control perfectly and returning no content on every `deref` question — is
# one question type of five lost by one arm and not the other: 20.0%, so it still refuses.
# `>=`, because that boundary IS the measured case and on a ship gate the boundary belongs
# on the refusing side. The statistic is in QUESTIONS, so this figure does not move with
# `--trials`.
UNPAIRED_ASYMMETRY_SHARE = 0.20

# The backstop asymmetry cannot provide: losses can be perfectly symmetric and still gut
# the exam, and a gap measured on the handful of easy questions that survived generalises
# to nothing. Deliberately loose — this is "the exam is mostly gone", not a tuning knob.
UNPAIRED_VOLUME_SHARE = 0.50


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


def _short_rows(rows: list[dict[str, Any]], form: str) -> list[tuple[int, int]]:
    """Trials `form` did NOT complete, per row, and the trials attempted — (lost, attempted).

    Magnitudes, not a boolean. A per-row "was this arm short at all" predicate cannot tell
    "answered 0 of 5" from "answered 4 of 5", and when both arms are short on the same row
    those two cancel — which let a TOTAL one-sided loss read as perfectly symmetric as soon
    as the control hiccuped once on the same questions. That is not a corner case: a
    token-budget stop that kills the form arm's longer prompt on a hard question is MORE
    likely, not less, to occasionally kill the control's prompt on that same question.

    Rows with no `attempts` counter contribute nothing to either total: their uneven
    per-form counts are a `score_pack` collection choice, not a loss (see `paired_rows`)."""
    key = form[:-3] + "_trials" if form.endswith("_ok") else form
    out = []
    for r in rows:
        t = int(r.get("trials", 1))
        if "attempts" not in r:
            out.append((0, 0))
            continue
        # `max(0, …)`: a hand-built row whose `<form>_trials` exceeds `trials` would
        # otherwise leak a negative loss and inflate the OPPOSITE direction.
        out.append((max(0, t - int(r.get(key, t))), t))
    return out


def loss_asymmetry(rows: list[dict[str, Any]], forms: list[str], control: str) -> float:
    """How ONE-SIDEDLY the pairing losses fall on one arm rather than the other.

    Per row, the excess of one arm's lost trials over the other's; summed in each direction
    separately, and the larger direction reported, as a share of the trials attempted. The
    worst form wins, because the verdict gates on the worst arm.

    Three properties this shape has and a naive count of short rows does not:

    MAGNITUDE DECIDES ATTRIBUTION, NOT SCALE. A row where the form answered none of its 5
    trials and the control answered 4 is one question lost by the FORM — not "both were
    short, so they cancel", and not "four trials of one-sidedness" either. The unit is the
    QUESTION because that is the unit of harm: `paired_rows` voids the whole question when
    any arm falls short, so a one-trial clip and a total blackout do identical damage to
    the exam. Dividing lost TRIALS by attempted trials instead diluted the statistic by
    `--trials`: at `--trials 3` the ceiling reachable before the volume backstop fires is
    0.50/3 = 0.17, below the 0.20 bar, so the asymmetry gate could never fire at all and
    was strictly dominated by the backstop it is supposed to precede. Measured on that
    version: a form arm clipping one trial on 9 of 20 questions voided 45% of the exam
    one-sidedly, scored 0.09, and published "safe to enable `proxy --diff`".

    MONOTONE IN THE CONTROL'S FAILURES. The two directions are tallied separately rather
    than subtracted, so a control-side loss on some OTHER question cannot buy back
    tolerance for a form-side loss here. Subtracting them meant three stray 429s on the
    control flipped a correct refusal into "safe to enable `proxy --diff`" — a gate that
    rewards a worse backend.

    DIRECTIONLESS. Both directions refuse, because which arm carries the longest prompt —
    and so truncates first — depends on the family. In the diff family the FORM is longer
    (`PREVIOUS RESULT … UPDATE …` versus the compressed payload alone). In the payload
    family it is the CONTROL: `run_payload` sends uncompressed `raw_text` against
    `compress(obj)`, because shrinking that is terse's entire purpose. So a control-side
    excess there drops the biggest payloads — exactly where terse's saving is largest and
    its comprehension risk highest — and removing them FLATTERS terse. An earlier version
    floored this at zero on the argument that a control-side excess is "the safe error";
    that argument holds for the diff family and is backwards for the fluency one.

    Per form separately, not "any form": the payload family gates two forms against one
    control, so an any-test gives the form side two chances against the control's one and
    reads symmetric flake as a one-sided loss."""
    if not rows:
        return 0.0
    c_loss = _short_rows(rows, control)
    worst = 0.0
    for f in forms:
        f_loss = _short_rows(rows, f)
        scorable = [i for i, (_, t) in enumerate(f_loss) if t]
        if not scorable:
            continue
        # Each VOIDED question is attributed to whichever arm lost more of it, and the two
        # sides are tallied separately. Trial magnitudes decide the attribution; they do not
        # scale the result.
        form_side = sum(1 for i in scorable if f_loss[i][0] > c_loss[i][0])
        ctrl_side = sum(1 for i in scorable if c_loss[i][0] > f_loss[i][0])
        worst = max(worst, max(form_side, ctrl_side) / len(scorable))
    return worst


def unpaired(rows: list[dict[str, Any]], forms: list[str], control: str) -> bool:
    """True when the surviving question set is too SELECTED, or too small, to compare on.

    `paired_rows` makes the surviving gap honest — after pairing both arms sit the identical
    exam — but it cannot make that exam REPRESENTATIVE. Two different ways it stops being:

      1. the losses are one-sided (`loss_asymmetry`), which is the #268 signature and the
         mechanism that actually selects the survivors by difficulty;
      2. the exam is mostly gone (`UNPAIRED_VOLUME_SHARE`), where even symmetric loss
         leaves a gap measured on whichever handful of questions happened to survive.

    Trigger 1 replaced a plain volume bar, which could not tell a run where both arms flaked
    equally from one where only the arm under test truncated — see `UNPAIRED_ASYMMETRY_SHARE`
    for the measured cost of that confusion."""
    return unpaired_reason(rows, forms, control) is not None


def unpaired_reason(rows: list[dict[str, Any]], forms: list[str],
                    control: str) -> str | None:
    """Which of the two refusals fired, or None. `"unpaired"` | `"exam too small"`.

    Separate strings because the two are separate findings and the reports say different
    things about them. A single label made the "Not compared" paragraph assert that "the
    losses fell ONE-SIDEDLY on an arm under test" and that "even losses do not trigger it"
    for a run refused at exactly 0.0 asymmetry by the volume backstop — every clause false,
    and it advised re-running at a lower `--trials` when the finding was "half the exam is
    gone"."""
    if not rows:
        return None
    if loss_asymmetry(rows, forms, control) >= UNPAIRED_ASYMMETRY_SHARE:
        return "unpaired"
    dropped = len(rows) - len(paired_rows(rows, *forms, control))
    if dropped / len(rows) >= UNPAIRED_VOLUME_SHARE:
        return "exam too small"
    return None


# Share of a model's calls that may fail before its numbers stop meaning anything (#263).
# NOT zero: a failed call is already excluded from its arm's denominator (see the
# `<form>_trials` keys `run_payload` emits), so a handful of transient 429s no longer
# depress an accuracy at all — voiding a whole model for one of them would discard an
# otherwise-complete multi-hour run, which is its own kind of wrong answer. What a
# threshold still has to catch is the backend that was substantially down, where the
# surviving sample is small and self-selected rather than merely smaller.
UNMEASURED_FAIL_SHARE = 0.20


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


def _gap(rows: list[dict[str, Any]], gating: list[str], control: str,
         display: tuple[str, ...] = ()) -> ArmGap:
    """Shared body of `arm_gap`/`best_arm_gap`. Gates, pairs, then computes every arm.

    Order matters: `_unmeasured` first (the backend was down — nothing here is measurable),
    then `unpaired` (it was up, but too little of the question set survived on one side to
    compare), then pair and compute. A caller that wants the numbers must accept the gate,
    because they arrive together.

    `display` arms are computed over the paired subset but do NOT participate in pairing.
    That is deliberate: `run_payload`'s `inline` arm carries the longest prompt of the four
    and so truncates first under a token-budget stop, while gating nothing — pairing on it
    would void otherwise-complete runs over an arm no verdict consumes."""
    if not rows:
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "empty")
    if _unmeasured(rows):
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], "unmeasured")
    why = unpaired_reason(rows, gating, control)
    if why:
        return ArmGap(0.0, 0.0, 0.0, 0.0, [], why)
    pr = paired_rows(rows, *gating, control)
    arms = {f: _form_stats(pr, f) for f in (*gating, control, *display)}
    cacc, cse = arms[control]
    if control == "raw_ok" and cacc == 0:
        # A raw control at exactly 0% is a backend/config error, not a comprehension
        # result — every form would "beat" it. Kept here rather than at the call sites so
        # the markdown verdict and the forest plot cannot disagree about it.
        return ArmGap(0.0, 0.0, cacc, cse, pr, "broken control", arms)
    best = max((arms[f] for f in gating), key=lambda s: s[0])
    return ArmGap(best[0], best[1], cacc, cse, pr, None, arms)


def arm_gap(rows: list[dict[str, Any]], form: str, control: str) -> ArmGap:
    """`form` vs `control` over the rows BOTH arms completed, or an exclusion reason.

    The single chokepoint for every diff-vs-control verdict in terse."""
    return _gap(rows, [form], control)


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
    "unmeasured": "calls went unanswered",
    "unpaired": "one-sided losses left the arms not comparable",
    "exam too small": "too little of the question set survived pairing to compare",
    "broken control": "control arm failed",
    "not a diff run": "no diff arm in these rows",
    "empty": "no rows",
}

# The heading each reason gets in prose renderers (markdown, HTML). "Not measured" and
# "Not compared" are different claims and the distinction is the whole point: one says the
# backend did not answer, the other says it did.
REASON_HEADING = {
    "unmeasured": "Not measured",
    "unpaired": "Not compared",
    "exam too small": "Not compared",
    "broken control": "Excluded",
    "not a diff run": "Not applicable",
    "empty": "Not measured",
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


def _not_measured_lines(withheld: dict[str, tuple[str | None, int, int]]) -> list[str]:
    """The "**Not measured**" paragraph, split by WHY each model was withheld.

    Two exclusions reach here and they are not the same event, so they cannot share a
    sentence. "Too many calls went unanswered" is about transport; an `unpaired` exclusion
    is about a backend that answered fine while one arm lost too much of the question set
    to compare. Printing the transport wording for an unpaired model contradicted the call
    count printed in the same breath (e.g. "too many calls went unanswered — `m` (4/240
    calls lost)"), which is a report arguing with itself."""
    if not withheld:
        return []
    out: list[str] = []
    lost = {m: (f, a) for m, (why, f, a) in withheld.items()
            if why not in ("unpaired", "exam too small")}
    unpair = sorted(m for m, (why, _, _) in withheld.items() if why == "unpaired")
    small = sorted(m for m, (why, _, _) in withheld.items() if why == "exam too small")
    if lost:
        out.append(
            "**Not measured** — too many calls went unanswered (connection error, rate "
            "limit, bad model id, or the model returning no content), so no accuracy "
            "is published for: "
            + ", ".join(f"`{m}` ({f}/{a} calls lost)" for m, (f, a) in sorted(lost.items()))
            + ". An unanswered call is not a wrong answer. Check stderr for a "
              "`returned no content` line naming a `finish_reason` — `length` means "
              "raise max_tokens, `content_filter` means the payload tripped a filter; "
              "otherwise re-run once the backend is reachable.")
    if unpair:
        out.append(
            "**Not compared** — the backend answered, but the questions it failed fell "
            "one-sidedly on one arm, so the survivors are selected by difficulty rather "
            "than merely fewer, and no gap is published for: "
            + ", ".join(f"`{m}`" for m in unpair)
            + f". A question counts only when every arm answered all of its trials, so one "
              f"lost trial withholds the whole question — and past "
              f"{UNPAIRED_ASYMMETRY_SHARE:.0%} of the question set withheld because ONE arm "
              f"failed it and the other did not, the two arms are no longer sitting the "
              f"same exam (#280). Either arm counts: the longest prompt belongs to the "
              f"diff form in a `--diff` run and to the uncompressed raw control in a "
              f"fluency run, so either can be the one a token-budget stop truncates. "
              f"Evenly-failing questions do not trigger it; re-run at a lower `--trials` so "
              f"one failure costs less of a question, or once the backend is steadier.")
    if small:
        # Its own paragraph. The one above asserts one-sidedness, which is exactly what did
        # NOT happen here — this is the symmetric case, refused on volume alone.
        out.append(
            "**Not compared** — the backend answered and its losses were evenly spread, but "
            "pairing left too little of the question set to publish a gap for: "
            + ", ".join(f"`{m}`" for m in small)
            + f". A question is comparable only when every arm answered all of its trials, "
              f"so one lost trial withholds the whole question; past "
              f"{UNPAIRED_VOLUME_SHARE:.0%} of the exam gone, what survives is too small to "
              f"generalise from however evenly it was lost (#280). Lower `--trials` so a "
              f"single failure costs less of the question, or re-run once the backend is "
              f"steadier.")
    out.append("")
    return out


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
    half-width of the pooled standard error; passed iff gap is at least -tol, inclusive
    of the boundary. Callers access fields by name, e.g. verdict.form_acc, never by
    position, so a future field reorder can't silently swap values."""
    worst = None  # (model, gap, facc, cacc, gap_ci) — cheapest to track positionally here;
    for model, (facc, fse, cacc, cse) in rows.items():  # this is a private local, not the
        gap = facc - cacc                               # public interface callers rely on.
        gap_ci = _ci(math.sqrt(fse ** 2 + cse ** 2))
        if worst is None or gap < worst[1]:
            worst = (model, gap, facc, cacc, gap_ci)
    if worst is None:
        return None
    model, gap, facc, cacc, gap_ci = worst
    passed = gap >= -tol - 1e-9
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
        attempts = sum(r.get("trials", 1) for r in rows)
        if errs and attempts and errs * 2 >= attempts:
            out[model] = (errs, attempts)
    return out


def dropeval_gap_rows(results: dict) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """Per-model (recall, precision, accuracy) gap-row tuples for build_dropeval_report
    and its terminal-bar companion. Control is always a fixed 100% ideal (se=0) — there's
    no raw/full-terse form to compare against here, only "did the model do the right
    thing." Same per-model math build_dropeval_report's own table loop uses, kept in one
    place so the two verdicts (markdown table, terminal chart) can never disagree."""
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        aacc, ase = _form_stats(rows, "answer_ok")
        out[model] = {
            "recall": (racc, rse, 1.0, 0.0),
            "precision": (pacc, pse, 1.0, 0.0),
            "accuracy": (aacc, ase, 1.0, 0.0),
        }
    return out


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
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound.",
        "",
        f"| Model | q | {control_label} | diff | regressions |",
        "|---|---|---|---|---|",
    ]
    gap_rows: dict[str, tuple[float, float, float, float]] = {}
    unmeasured: dict[str, tuple[str | None, int, int]] = {}  # model -> (why, fails, attempts)
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
                                 sum(int(r.get("attempts", 0)) for r in rows))
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
        if unmeasured and all(w in ("unpaired", "exam too small") for w, _, _ in unmeasured.values()):
            out.append("- **NO VERDICT — nothing could be compared.** The backend answered, "
                       "but no model had enough questions completed by both arms to score a "
                       "gap, so this run says nothing about the diff form either way. Lower "
                       "`--trials`, or re-run once the backend stops truncating, before "
                       "enabling `proxy --diff`.")
        else:
            out.append("- **NO VERDICT — nothing was measured.** No model returned enough "
                       "calls to score, so this run says nothing about the diff form either "
                       "way. Fix the backend(s) and re-run before enabling `proxy --diff`.")
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
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound. depth = diffs chained after the full anchor.",
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
            lead = ("**Depths not compared** — the backend answered, but one arm did not "
                    "complete enough of the same questions at: "
                    if why in ("unpaired", "exam too small") else
                    "**Depths not measured** — too many calls went unanswered at: ")
            out += [lead + at + ". Those depths are excluded from the verdict below rather "
                    "than scored on a question set the two arms did not share (#280).", ""]

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
        if pooled_out and all(w in ("unpaired", "exam too small") for w in pooled_out.values()):
            out.append("- **NO VERDICT — nothing could be compared.** The backend answered, "
                       "but no model had enough questions completed by both arms to score a "
                       "gap, so this run says nothing about diff-chain drift either way. "
                       "Lower `--trials`, or re-run once the backend stops truncating.")
        else:
            out.append("- **NO VERDICT — nothing was measured.** No model returned enough "
                       "calls to score, so this run says nothing about diff-chain drift "
                       "either way. Fix the backend(s) and re-run.")
    if worst:
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "chain-form",
                                           "full-terse"))
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
                       f"to probe was not scored. The overall line above is pooled across "
                       f"shallower depths and says nothing about drift at {deep}.")
        if worst.passed and not deep_withheld and (deepest is None or deepest.passed):
            out.append("- No depth-correlated comprehension drift within tolerance — "
                       "chained diffs up to the tested depth read as well as fulls.")
        elif not deep_withheld:
            out.append("- Comprehension drifts beyond tolerance somewhere in the chain — "
                       "keep the keyframe interval at or below the deepest PASSING depth.")
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
        if summary.get(model, {}).get("unmeasured"):
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


def build_dropeval_report(results: dict) -> str:
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
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound.",
        "",
        "| Model | recall q | retrieve-recall | precision (no-overfetch) | final-accuracy "
        "| handle-accuracy | failed calls |",
        "|---|---|---|---|---|---|---|",
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
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        aacc, ase = _form_stats(rows, "answer_ok")
        hacc, hse = _form_stats(recall_rows, "handle_ok") if recall_rows else (0.0, 0.0)
        # control is a fixed 100% ideal (se=0) — there's no raw/full-terse form to pair
        # against here, only "did the model do the right thing."
        recall_gate[model] = (racc, rse, 1.0, 0.0)
        precision_gate[model] = (pacc, pse, 1.0, 0.0)
        accuracy_gate[model] = (aacc, ase, 1.0, 0.0)
        errs = sum(r.get("errors", 0) for r in rows)
        attempts = sum(r.get("trials", 1) for r in rows)
        err_by_model[model] = (errs, attempts)
        out.append(f"| `{model}` | {len(recall_rows)} | {racc:.0%} ±{_ci(rse) * 100:.0f} "
                   f"| {pacc:.0%} ±{_ci(pse) * 100:.0f} | {aacc:.0%} ±{_ci(ase) * 100:.0f} "
                   f"| {hacc:.0%} ±{_ci(hse) * 100:.0f} "
                   f"| {errs}/{attempts} |")
    out.append("")
    broken = {m: (e, a) for m, (e, a) in err_by_model.items() if e}
    if broken:
        out += ["> **Model calls failed** — these rows measure the harness, not the model: "
                + ", ".join(f"`{m}` {e}/{a}" for m, (e, a) in sorted(broken.items())) + ".",
                ""]

    out += ["## Verdict", ""]
    # Half of a model's calls failing means its accuracy columns are mostly counting
    # transport errors. Refuse to render a pass/fail rather than let the run be cited.
    inconclusive = inconclusive_models(results)
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
    if recall_worst and precision_worst and accuracy_worst:
        out.append(_format_worst_case_line(recall_worst, _GAP_TOLERANCE, "retrieve-recall",
                                           "ideal (100%)"))
        out.append(_format_worst_case_line(precision_worst, _GAP_TOLERANCE, "no-overfetch",
                                           "ideal (100%)"))
        out.append(_format_worst_case_line(accuracy_worst, _GAP_TOLERANCE, "final-accuracy",
                                           "ideal (100%)"))
        if recall_worst.passed and precision_worst.passed and accuracy_worst.passed:
            out.append("- Recall, precision, and final accuracy all clear tolerance for the "
                       "worst model — safe to enable drop-to-retrieve.")
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
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound on the accuracy.",
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
        unmeasured = g.excluded in ("unmeasured", "unpaired", "exam too small")
        reasons[model] = g.excluded
        # "n" is the GENERATED question count; nothing in `src/` reads it (the table
        # prints `len(g.rows)`), kept only because result files carry the shape.
        summary[model] = {"n": n, "raw": racc, "raw_se": rse,
                          "terse": tacc, "terse_se": tse, "primer": pacc, "primer_se": pse,
                          "fails": fails, "attempts": attempts, "unmeasured": unmeasured,
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
        m: (reasons.get(m), int(s["fails"]), int(s["attempts"]))
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
    unpaired_models = [m for m, r in reasons.items() if r in ("unpaired", "exam too small")]
    gated = {m: s for m, s in summary.items() if reasons.get(m) is None}
    if broken:
        out.append(f"- Excluded (raw control failed — backend/config error, not comprehension): "
                   f"{', '.join(f'`{m}`' for m in broken)}.")
    if unmeasured_models:
        out.append(f"- Excluded (calls went unanswered — not measured): "
                   f"{', '.join(f'`{m}`' for m in unmeasured_models)}.")
    if unpaired_models:
        out.append(f"- Excluded (the backend answered, but too little of the question set "
                   f"could be compared across arms — not compared): "
                   f"{', '.join(f'`{m}`' for m in unpaired_models)}.")
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
        if worst.gap_ci > 1e-9 and abs(worst.gap) < worst.gap_ci:
            out.append("- The gap is within its own confidence interval — terse and raw are "
                       "indistinguishable at this trial count (raise `--trials` to tighten).")
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
        why = ("Every model was excluded above" if (broken or unmeasured_models)
               else "No model produced any scored rows (did the corpus generate questions?)")
        out.append(f"- **NO VERDICT — nothing was measured.** {why}, so this run says "
                   "nothing about comprehension either way. Fix that and re-run before "
                   "drawing any conclusion from it.")
    out.append("")
    return "\n".join(out)
