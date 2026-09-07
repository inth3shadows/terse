"""`_unmeasured` must see BOTH of dropeval's arms, on the same terms (#352).

`_unmeasured` discovers arms by scanning for `<arm>_trials` keys, and dropeval emits
exactly one of them — `control_trials`. Its treatment arm deliberately has no
`answer_trials`: `dropeval.py`'s row build argues errored trials must stay in the accuracy
denominator, because scoring them as misses makes the drop rule look *worse*, which is the
conservative direction. Measured there: removing them turned a 33% recall FAIL into a 100%
PASS at an 11% error rate.

The side effect was a one-sided gate. A control arm losing 21% of its calls withheld
final-accuracy; the treatment arm losing the same share was invisible at every level, and
the only remaining cover was `inconclusive_models`' arm-blind 50%-of-pooled-calls
threshold. So the treatment could lose 49% of its own calls — the arm that runs two turns
to the control's one, and therefore the arm that fails first under a token-budget stop —
and the report would still publish a gap.

The invariant these tests pin is the one the issue names: **the same loss on either arm
produces the same verdict.** They are written against the row shape `dropeval.py` actually
emits, not a synthetic one — in particular the treatment arm carries NO `<arm>_trials` key
here, because a fixture that gives it one tests a harness that does not exist.
"""

from __future__ import annotations

import pytest

from terse.report import (
    REASON_LABEL,
    UNMEASURED_FAIL_SHARE,
    Directive,
    _accuracy_gate,
    _arm_loss_share,
    _credited_loss_share,
    _distrust_loss_share,
    _unmeasured,
    build_dropeval_report,
    dropeval_verdict,
    inconclusive_models,
    passes_tolerance,
)

# --------------------------------------------------------------------------- #
# Fixtures in dropeval's real row shape.
# --------------------------------------------------------------------------- #


def _row(qid, *, kind, trials, t_err, c_err, answer, control):
    """One dropeval row, internally coherent with `dropeval.py`'s row build.

    Note what is ABSENT: `answer_trials` / `retrieve_trials` / `handle_trials`. The
    treatment arm keeps every trial in its denominator by design, which is exactly why
    `_unmeasured`'s `<arm>_trials` triggers cannot see it and why trigger 4 exists.
    """
    return {
        "qid": qid, "kind": kind, "trials": trials,
        "retrieve_ok": trials - t_err, "handle_ok": trials - t_err, "answer_ok": answer,
        "errors": t_err + c_err, "treatment_errors": t_err, "control_errors": c_err,
        "attempts": trials * 2,
        "control_ok": control, "control_trials": trials - c_err,
    }


def _run(*, trials=10, t_err=0, c_err=0, n=24, kind="recall"):
    """`n` questions, each losing `t_err` treatment calls and `c_err` control calls.

    Both arms score every call they complete, so nothing here is a behavioural
    difference — the only thing varying between the two directions is which arm lost.
    """
    return [_row(f"q{i}", kind=kind, trials=trials, t_err=t_err, c_err=c_err,
                 answer=trials - t_err, control=trials - c_err) for i in range(n)]


def _both_kinds(**kw):
    return _run(kind="recall", **kw) + [dict(r, qid="p" + r["qid"])
                                        for r in _run(kind="precision", **kw)]


# --------------------------------------------------------------------------- #
# The invariant.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lost", range(0, 11))
def test_the_same_loss_on_either_arm_produces_the_same_verdict(lost):
    """The whole of #352 in one assertion, swept across the threshold rather than sampled
    at one convenient point.

    Before the fix this held only for `lost == 0`: every non-zero share past the line
    withheld the run when the CONTROL lost it and published when the TREATMENT lost the
    identical number of identical calls."""
    treatment_lost = _unmeasured(_run(trials=10, t_err=lost))
    control_lost = _unmeasured(_run(trials=10, c_err=lost))
    assert treatment_lost == control_lost, (
        f"{lost}/10 calls lost is withheld on one arm and published on the other: "
        f"treatment={treatment_lost}, control={control_lost}")


def test_the_issue_reproduction_now_withholds_the_treatment_arm():
    """Verbatim from #352, executed against `main` where it printed
    `treatment 80% lost, _unmeasured = False`."""
    rows = [{"qid": f"q{i}", "qtype": "lookup", "transform": "t", "trials": 10,
             "retrieve_ok": 10, "answer_ok": 2, "handle_ok": 10,
             "control_ok": 10, "control_trials": 10,
             "errors": 8, "treatment_errors": 8, "control_errors": 0, "attempts": 20}
            for i in range(20)]
    assert _unmeasured(rows), "80% of the treatment arm's calls lost is not a measurement"


def test_a_treatment_arm_exactly_at_the_loss_share_is_still_measured():
    """The mirror of `test_a_model_exactly_at_the_loss_share_is_still_measured` (#337) for
    the new trigger: `>` -> `>=` must be an observed mutation on this side too, or the
    treatment arm gets a boundary nothing watches.

    `trials=10, t_err=2` lands the ratio exactly on `UNMEASURED_FAIL_SHARE`."""
    assert UNMEASURED_FAIL_SHARE == 0.20, "fixture arithmetic is keyed to the constant"
    assert not _unmeasured(_run(trials=10, t_err=2)), "exactly on the line is measured"
    assert _unmeasured(_run(trials=10, t_err=3)), "one call past the line is not"


def test_the_treatment_trigger_is_not_the_pooled_one_in_disguise():
    """Why emitting `fails = errors` would not have closed this.

    The pooled trigger divides by `attempts`, which counts BOTH arms — so a treatment-only
    loss would have to reach 40% of its own calls to fire while the control still fires at
    20%. That is the pooled-denominator defect #339 removed, and this fixture sits in the
    window where the two answers differ: 30% of the treatment arm, 15% of pooled calls."""
    rows = _run(trials=10, t_err=3)
    pooled = sum(r["errors"] for r in rows) / sum(r["attempts"] for r in rows)
    assert pooled == pytest.approx(0.15), "fixture must sit under the pooled threshold"
    assert _unmeasured(rows), "read against its own arm, 30% is past the line"


def test_the_denominator_is_the_arms_own_calls_not_the_rows_that_happened_to_lose_one():
    """A loss concentrated on some questions is still a share of the whole arm.

    Counting only the rows carrying a non-zero counter would read this run at 100% and
    withhold it. `dropeval.py` emits `treatment_errors` on every row, zero included; that
    emitter contract is pinned separately, against the harness, by
    `test_dropeval_emits_both_per_arm_counters_on_every_row`."""
    rows = _run(trials=10, t_err=10, n=4) + _run(trials=10, t_err=0, n=20)
    assert all("treatment_errors" in r for r in rows)
    # 40 lost of 240 = 16.7%, under the line despite four questions being wiped out.
    assert not _unmeasured(rows)


def test_a_row_that_carries_no_counter_is_unknown_loss_not_zero_loss():
    """The row-set rule, which is the trigger's denominator and was unobserved.

    Mutating `err_key in r` to "read an absent counter as zero" left the whole suite green,
    because the only two fixtures that distinguished the rules were the two this change
    already had to repair. So the choice gets its own witness.

    An absent counter is not evidence of a clean call. Reading it as zero would put rows
    whose loss is unknown into the denominator of a share computed from rows whose loss is
    known — the same class of error as #339's pooled denominator, and it dilutes toward
    publishing. One current row reporting a lost call, merged with rows from a producer
    that counted nothing, is 1 of 3 known calls; it is NOT 1 of 144."""
    legacy = [{"qid": f"old{i}", "kind": "recall", "trials": 3, "attempts": 6,
               "answer_ok": 3, "control_ok": 3, "control_trials": 3} for i in range(47)]
    current = [{"qid": "new", "kind": "recall", "trials": 3, "attempts": 6,
                "answer_ok": 2, "control_ok": 2, "control_trials": 2,
                "errors": 1, "treatment_errors": 0, "control_errors": 1}]
    # AMENDED by #383, and the amendment is the point of the issue. This asserted
    # `_unmeasured(legacy + current)` on the reasoning "1 lost of the 3 calls anyone
    # counted is 33%, not 0.7% of 144 calls nobody did". That reasoning proves too much:
    # under it legacy run size is mathematically irrelevant, so ONE sufficiently degraded
    # row carrying a counter withholds a merged run of ANY size. Nor does it preserve the
    # uncertainty it claims to — skipping the unknown rows does not hold them unknown, it
    # implicitly assigns them the carrying subset's own loss rate, which is a STRONGER
    # assumption than reading them as zero.
    #
    # What survives, and is what this test was really witnessing, is the ROW-SET rule
    # inside `_arm_loss_share`. That helper still skips rows carrying no counter, because
    # its question is "how degraded is this arm" and a caller wanting the degraded slice's
    # own undiluted rate needs exactly that. `_unmeasured` is simply no longer such a
    # caller.
    assert _arm_loss_share(legacy + current, "control") == pytest.approx(1 / 3), (
        "the row-set rule itself is unchanged: rows carrying no counter are skipped, "
        "not read as a measured zero")
    assert _distrust_loss_share(legacy + current, "control") == pytest.approx(1 / 144)
    assert not _unmeasured(legacy + current), (
        "one lost call in 144 is not grounds to withhold the run (#383)")
    # The same 47 legacy rows with nothing lost anywhere still publish, so the assertion
    # above is about the missing counter and not about merging packs at all.
    assert not _unmeasured(legacy + [dict(current[0], answer_ok=3, control_ok=3,
                                          control_trials=3, errors=0, control_errors=0)])


def test_an_arm_reporting_more_errors_than_calls_is_withheld_not_rounded_down():
    """The refused clamp (`min(attempts, errors)`), which the code argues for in a comment
    and nothing observed — adding the clamp left the full suite green.

    A row claiming more failures than the calls it was given is an emitter bug, and there
    is no benign form of it. Clamping would let one such row hide behind a healthy majority
    at exactly the moment the counters stopped being trustworthy; the unclamped share goes
    over 1.0 and withholds, which is the direction that asks a human to look."""
    rows = [{"qid": "a", "kind": "recall", "trials": 10, "attempts": 20,
             "answer_ok": 10, "treatment_errors": 0},
            {"qid": "b", "kind": "recall", "trials": 1, "attempts": 2,
             "answer_ok": 0, "treatment_errors": 10}]
    # Unclamped: 10 lost of 11 calls. Clamped: 1 of 11, which publishes.
    assert _unmeasured(rows)


def test_dropeval_emits_both_per_arm_counters_on_every_row():
    """The coupling the trigger above rests on, asserted against the harness rather than
    against a fixture: a row that omits its zero would halve the denominator."""
    from terse import dropeval
    from terse import policy as policy_mod

    class _Answerer:
        """Fails the first two calls only, so the earliest question carries a non-zero
        counter and every later one carries an explicit zero — which is the case the
        denominator depends on."""

        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            self.calls += 1
            return dropeval.Turn(text="x", tool_calls=[], error=self.calls <= 2)

    obj = {"rows": [{"id": i, "evidence": f"{i}" + "E" * 300} for i in range(4)]}
    rule = policy_mod.Rule(tool_glob="t", tiers=("minify", "table"),
                           fields={"rows[].evidence": {"lossy": "drop-to-retrieve",
                                                       "min": 10}})
    rows = dropeval.run_drop_payload(obj, "", rule, "t", _Answerer(), trials=3,
                                     control=True)
    assert rows
    for r in rows:
        assert "treatment_errors" in r and "control_errors" in r
    # ...and a zero is written EXPLICITLY, not left out: "always present" must not be
    # satisfied only by the rows that happened to fail.
    assert any(r["treatment_errors"] for r in rows), "fixture lost no treatment calls"
    assert any(r["treatment_errors"] == 0 for r in rows)
    assert any(r["control_errors"] == 0 for r in rows)


# --------------------------------------------------------------------------- #
# Three mutations that survived the first sweep of the new trigger. Each one is a
# behavioural difference nothing observed, so each gets the fixture that observes it.
# --------------------------------------------------------------------------- #


def test_the_pooled_errors_total_is_not_read_as_a_third_arm():
    """`errors` is `treatment_errors + control_errors`, so matching it would count every
    failure twice and divide by ONE arm's trials — a hidden third threshold at half the
    documented share.

    Mutation `k.endswith("_errors")` -> `k.endswith("errors")` survived the first sweep:
    no fixture had both arms on the right side of the line while their sum was not.
    This one does — 2 lost of 10 on each arm is exactly `UNMEASURED_FAIL_SHARE`, and the
    pooled 4-of-10 is double it."""
    rows = _run(trials=10, t_err=2, c_err=2)
    assert all(r["errors"] == r["treatment_errors"] + r["control_errors"] for r in rows)
    assert not _unmeasured(rows), (
        "both arms are exactly on the line; only the double-counted total is past it")


def test_a_counter_on_an_arm_that_made_no_calls_is_absence_not_failure():
    """Zero attempts is the #283 distinction one counter over: an arm nobody called cannot
    have a loss SHARE, and reading it as a withheld run would void the model on the
    strength of a question that was never asked. It is also the division guard, so the
    alternative to `continue` is a `ZeroDivisionError`, not a different verdict.

    Mutation `continue` -> `return True` survived TWICE. The first fixture merged
    zero-trial rows into healthy ones that ALSO carried the counter, so the arm's attempts
    summed to 120 and the guard was never reached — a fixture that cannot fail. Reaching it
    needs every row carrying the counter to have zero attempts, which is a merged pack: one
    half predates the per-arm counters entirely, the other ran no trials."""
    predates_the_counters = [
        {"qid": f"old{i}", "kind": "recall", "trials": 10, "attempts": 20,
         "answer_ok": 10, "control_ok": 10, "control_trials": 10} for i in range(12)]
    counted_but_never_run = [
        {"qid": f"new{i}", "kind": "recall", "trials": 0, "attempts": 0,
         "answer_ok": 0, "treatment_errors": 0, "control_errors": 0} for i in range(12)]
    rows = predates_the_counters + counted_but_never_run
    # The precondition the first fixture missed: every row carrying a counter has no calls.
    assert all(r["trials"] == 0 for r in rows if "treatment_errors" in r)
    assert not _unmeasured(rows)


def test_an_explicitly_stated_arm_attempt_count_is_the_denominator():
    """`<arm>_attempts` overrides the shared `trials` — the `score_pack` idiom from #283,
    applied to this trigger by deriving the bare arm name from the counter key.

    Mutation `err_key[:-len("_errors")]` -> `err_key` survived: `_arm_attempts` then looks
    up `treatment_errors_attempts`, which no row carries, so it silently fell back to the
    shared `trials` and every existing fixture agreed. Here they disagree — 1 lost of a
    stated 4 is past the line, 1 of the shared 10 is not."""
    rows = [dict(r, treatment_attempts=4, treatment_errors=1)
            for r in _run(trials=10, n=24)]
    assert _unmeasured(rows), (
        "the share must be read against the 4 calls the row says this arm was given, "
        "not the 10 the other arm was")
    # The same row set without the explicit count is measured, so the assertion above is
    # about the denominator and not about the loss.
    assert not _unmeasured([dict(r, treatment_errors=1) for r in _run(trials=10, n=24)])


# --------------------------------------------------------------------------- #
# End to end: the same symmetry, through the report the operator reads.
# --------------------------------------------------------------------------- #


def test_a_degraded_treatment_arm_publishes_no_final_accuracy():
    """`_unmeasured` returning True is only worth something if the report acts on it.

    40% treatment loss, and — the point of the fixture — a total error rate of 20% of
    pooled calls, well under `inconclusive_models`' 50% threshold. So this run has no
    other gate: before #352 it published a final-accuracy gap computed over whichever
    questions the two-turn arm happened to survive."""
    rows = _both_kinds(trials=10, t_err=4)
    assert inconclusive_models({"m": rows}) == {}, "no other gate may be doing the work"
    report = build_dropeval_report({"m": rows})
    assert "no usable comparison" in report
    assert "**Where they failed** (per arm" in report


def test_the_accuracy_gate_reaches_the_same_verdict_whichever_arm_lost_the_calls():
    """The symmetry, scoped to the gate `_unmeasured` actually controls.

    NAMED FOR THE GATE, NOT THE RUN, and that is a correction rather than a nicety. An
    earlier revision called itself
    `test_the_report_reaches_the_same_verdict_whichever_arm_lost_the_calls` and asserted
    only that both renderings contain "no usable comparison" and "not gated" — both of
    which are true of both reports while their run-level `Directive`s differ. A test named
    for an invariant it cannot observe failing is worse than no test: it is the #352
    blind spot re-created inside #352's own fix. The residual asymmetry it could not see is
    pinned deliberately by the test below."""
    treatment = build_dropeval_report({"m": _both_kinds(trials=10, t_err=4)})
    control = build_dropeval_report({"m": _both_kinds(trials=10, c_err=4)})
    for report in (treatment, control):
        assert "no usable comparison" in report
        assert "not gated" in report


def test_the_run_level_verdict_is_symmetric_across_the_arms_and_says_which():
    """The inverse of the pin this replaces (#371), and the reason that one was written.

    Its predecessor asserted the residual ASYMMETRY as a recorded defect: `_unmeasured`
    gates only `_accuracy_gate` -> `arm_gap`, while recall and no-overfetch score against a
    FIXED 100% ideal, never pair, and so were never gated on transport loss at all. Identical
    loss produced BLOCK on the treatment arm and NOT_CONCLUDED on the control. That test
    ended "if someone makes it, this test goes red and tells them to delete it." #371 made
    it; this is the replacement, not the deletion.

    What must now hold is the symmetry #352 asked for at the RUN level: the same loss on
    either arm produces the same directive, for a stated reason, on every metric.

    The direction still matters and is asserted separately below: withholding may never
    manufacture authority. NOT_CONCLUDED is not SHIP.
    """
    treatment = dropeval_verdict({"m": _both_kinds(trials=10, t_err=4)})
    control = dropeval_verdict({"m": _both_kinds(trials=10, c_err=4)})

    assert treatment.directive is control.directive is Directive.NOT_CONCLUDED
    # Not merely equal directives — equal for a STATED reason. Two runs could agree on
    # NOT_CONCLUDED while disagreeing about which metrics were measured, which is the
    # contradiction #371 is about: a number in the table the prose above it disowns.
    for metric in ("accuracy", "recall", "precision"):
        assert treatment.metrics[metric].excluded == {"m": "unmeasured"}, metric
    # The control arm's loss withholds accuracy (it pairs) but NOT the mechanism metrics,
    # which are computed from the treatment loop's `retrieve_ok` and are unharmed by a
    # control-arm failure. Symmetry is a property of the run-level directive, not a claim
    # that the two losses damage the same columns — gating the mechanism metrics on control
    # loss would withhold a measurement that actually succeeded.
    assert control.metrics["accuracy"].excluded == {"m": "unmeasured"}
    assert control.metrics["recall"].excluded == {}

    # The load-bearing direction: neither arm's loss may produce a SHIP authorization.
    for v in (treatment, control):
        assert v.directive is not Directive.SHIP
    # And no metric survives as a behavioural FAIL built from lost calls. `worst is None`
    # is what distinguishes "withheld" from "scored and failed" — the old BLOCK carried a
    # `worst` gap of -40% computed from rows the report itself called unmeasurable.
    assert treatment.metrics["recall"].worst is None


def test_a_withheld_mechanism_metric_says_transport_in_the_rendered_report():
    """#371 at the renderer, not just the verdict object. The defect an operator actually
    met was a rendered line — `retrieve-recall 60% vs ideal (100%) ... **FAIL**` printed
    under a paragraph declaring those same rows unmeasurable — so a verdict-only assertion
    would leave the thing that was wrong on screen unpinned."""
    report = build_dropeval_report({"m": _both_kinds(trials=10, t_err=4)})
    # The line the issue quoted is gone: no behavioural FAIL anywhere, and no verdict
    # authorizing or refusing policy on the strength of one.
    assert "**FAIL**" not in report
    # The two mechanism metrics now say they were not gated, in the verdict prose that
    # used to carry `retrieve-recall 60% vs ideal (100%) ... keep drop-to-retrieve off`.
    for metric in ("retrieve-recall", "no-overfetch"):
        assert f"**{metric}: not gated for `m`**" in report, metric
    assert "INCONCLUSIVE for enabling" in report
    # The REMEDY under a one-arm metric must not be the two-arm sentence. `_exclusion_remedy`
    # is keyed on the reason, and its "unmeasured" prose was written for a paired metric:
    # "Too few calls completed on BOTH arms to compare ... a zero means an arm completed no
    # trials at all." Recall and no-overfetch have one arm, and this gate cannot fire at zero
    # loss, so both halves of that sentence are false here — it was rendered to operators on
    # every degraded run (review finding on #379).
    bullet = next(ln for ln in report.splitlines()
                  if ln.startswith("- **retrieve-recall: not gated"))
    assert "BOTH arms" not in bullet, bullet
    assert "withheld, not failed" in bullet, bullet
    # And the table cell is the withheld marker, not a number. Asserting on the ROW is what
    # separates this from the verdict-object test: the defect an operator met was a
    # percentage printed in a column, under a paragraph disowning the rows behind it.
    # Read the cells by the HEADER's own column order rather than by a literal index.
    # Indexing 3/4 was correct but could not detect a reorder: both expected values are the
    # identical string, so swapping the two emitted cells survived the whole dropeval suite
    # (review finding on #379). Locating them by name makes the assertion about the named
    # columns, which is what it always claimed to be.
    header = next(ln for ln in report.splitlines() if ln.startswith("| Model |"))
    cols = [c.strip() for c in header.split("|")[1:-1]]
    row = next(ln for ln in report.splitlines() if ln.startswith("| `m` |"))
    cells = dict(zip(cols, [c.strip() for c in row.split("|")[1:-1]], strict=True))
    assert cells["retrieve-recall"] == "not gated", row
    assert cells["precision (no-overfetch)"] == "not gated", row

    # Both cells above hold the SAME string, so they cannot detect the two being emitted in
    # the wrong order — swapping them at the format string survived this file (review
    # finding on #379). One asymmetric run fixes that: recall loses enough calls to be
    # withheld, precision loses none and is scored, so the two cells are distinguishable and
    # a swap moves each into the other's column.
    mixed = build_dropeval_report({"m": _perfect_on_landed("recall", 24, 10, 4)
                                   + _perfect_on_landed("precision", 24, 10, 0)})
    header = next(ln for ln in mixed.splitlines() if ln.startswith("| Model |"))
    cols = [c.strip() for c in header.split("|")[1:-1]]
    row = next(ln for ln in mixed.splitlines() if ln.startswith("| `m` |"))
    cells = dict(zip(cols, [c.strip() for c in row.split("|")[1:-1]], strict=True))
    assert cells["retrieve-recall"] == "not gated", row
    assert cells["precision (no-overfetch)"].startswith("100%"), row
    # handle-accuracy is deliberately NOT covered here. It is the display-only column no
    # gate reads (see `test_gap_gate_boundary.py`'s allowlist note), so it still renders a
    # percentage on this run — an inconsistency worth its own decision, not a silent
    # widening of this fix's scope.


def _perfect_on_landed(kind, n, trials, t_err):
    """`n` questions where every call that LANDED was answered correctly. `retrieve_ok`
    counts the survivors only, which is what a transport loss looks like from the scorer's
    side: dropeval emits no `retrieve_trials`, so each lost call is scored a MISS."""
    return [{"qid": f"{kind}{i}", "kind": kind, "trials": trials,
             "retrieve_ok": trials - t_err, "handle_ok": trials - t_err,
             "answer_ok": trials - t_err, "errors": t_err, "treatment_errors": t_err,
             "control_errors": 0, "attempts": trials * 2,
             "control_ok": trials, "control_trials": trials} for i in range(n)]


def _flat(kind, n, trials, t_err, r_ok):
    """`n` questions scoring exactly `r_ok` of `trials`, whatever the loss."""
    return [{"qid": f"{kind}{i}", "kind": kind, "trials": trials,
             "retrieve_ok": r_ok, "handle_ok": r_ok, "answer_ok": r_ok,
             "errors": t_err, "treatment_errors": t_err, "control_errors": 0,
             "attempts": trials * 2, "control_ok": trials, "control_trials": trials}
            for i in range(n)]


def test_a_demonstrated_mechanism_failure_is_scored_even_at_a_large_transport_loss():
    """The defect the FIRST cut of #371 shipped, and the reason the gate asks whether the
    loss could EXPLAIN the miss rather than whether the loss is large.

    A loss-share threshold withheld this run: 21% of the treatment arm lost, which is past
    `UNMEASURED_FAIL_SHARE`. But the model operated the retrieve protocol correctly on ZERO
    of the 79 calls that landed. The loss bounds recall at 79%; the observed 0% is 79 points
    below anything transport can account for. Withholding it turned a BLOCK into
    NOT_CONCLUDED — and `NOT_CONCLUDED (2) < BLOCK (3)`, so for that model the exclusion
    IMPROVED the verdict. `UNMEASURED_FAIL_SHARE`'s own comment refuses exactly this: "do
    not add a survival threshold here without first making an exclusion unable to improve a
    verdict; those are one change, not two."

    Measured on the first cut: a sweep of 1,440 two-model fleets turned 240 BLOCKs into
    NOT_CONCLUDED this way."""
    rows = _flat("recall", 24, 100, 21, 0) + _flat("precision", 24, 100, 21, 0)
    assert _arm_loss_share(rows, "treatment") > UNMEASURED_FAIL_SHARE, (
        "fixture must sit past the old threshold, or it proves nothing")
    v = dropeval_verdict({"m": rows})
    assert v.metrics["recall"].excluded == {}, "a demonstrated failure is never withheld"
    assert v.metrics["recall"].worst is not None
    assert v.metrics["recall"].worst.gap == pytest.approx(-1.0)
    assert v.directive is Directive.BLOCK


@pytest.mark.parametrize("t_err", [1, 2, 3, 4])
def test_a_loss_that_fully_explains_the_miss_withholds_at_every_size(t_err):
    """The other half of the same first cut: the band BELOW the old threshold kept
    publishing the very line #371 is about.

    At 10% loss with the model perfect on every landed call, the report printed
    `retrieve-recall 90% ... **FAIL** at 5% tolerance` under its own "these rows measure the
    harness, not the model". Tolerance is 5% and every lost call scores a miss, so the
    defect was live across the whole `(5%, 20%]` band — while #371's body asserted that
    band was harmless because "any loss past `UNMEASURED_FAIL_SHARE` forced the column
    under tolerance". It does not; the loss share and the tolerance are different numbers.

    The predicate has no threshold to sit under: what matters is that crediting the lost
    calls clears tolerance."""
    v = dropeval_verdict({"m": _perfect_on_landed("recall", 24, 10, t_err)
                          + _perfect_on_landed("precision", 24, 10, t_err)})
    for mech in ("recall", "precision"):
        assert v.metrics[mech].excluded == {"m": "unmeasured"}, (
            f"{t_err}/10 lost fully explains the miss and must be withheld, not failed")


def test_a_metric_that_still_PASSES_under_loss_is_scored_not_withheld():
    """The gate withholds a FAIL the loss explains — never a PASS.

    Dropping the `not passes_tolerance(acc - 1.0)` clause left the suite green: every other
    fixture here either fails or has no loss, so nothing observed the case where both are
    true. The direction is conservative (a withheld PASS cannot authorize anything), which
    is exactly why it would have gone unnoticed — and a SHIP silently downgraded to
    NOT_CONCLUDED on a run that measured fine is still the harness lying about what it
    knows, in the cheaper direction.

    97 of 100 with 2 lost: inside tolerance as scored, so there is nothing for the loss to
    explain."""
    v = dropeval_verdict({"m": _flat("recall", 24, 100, 2, 97)
                          + _flat("precision", 24, 100, 2, 97)})
    for mech in ("recall", "precision"):
        assert v.metrics[mech].excluded == {}, f"{mech}: a passing metric is never withheld"
        assert v.metrics[mech].worst is not None and v.metrics[mech].worst.passed


def test_the_predicate_boundary_is_whether_the_credited_loss_clears_TOLERANCE():
    """The boundary itself, which under the old threshold was unpinned — mutating
    `> UNMEASURED_FAIL_SHARE` to `>=` SURVIVED the whole suite (review finding on #379).

    100 trials, 10 lost. Crediting all 10 lifts a 91% score to 101% (clears) and an 85%
    score to 95% — exactly 5% down, which `passes_tolerance` accepts at its epsilon. One
    point lower cannot be explained by the loss and must be scored."""
    withheld = dropeval_verdict({"m": _flat("recall", 24, 100, 10, 90)
                                 + _flat("precision", 24, 100, 10, 90)})
    assert withheld.metrics["recall"].excluded == {"m": "unmeasured"}
    # 85 + 10 = 95: on the tolerance line, still explained by the loss.
    on_line = dropeval_verdict({"m": _flat("recall", 24, 100, 10, 85)
                                + _flat("precision", 24, 100, 10, 85)})
    assert on_line.metrics["recall"].excluded == {"m": "unmeasured"}
    # 84 + 10 = 94: one point past what the loss can account for, so it is behaviour.
    scored = dropeval_verdict({"m": _flat("recall", 24, 100, 10, 84)
                               + _flat("precision", 24, 100, 10, 84)})
    assert scored.metrics["recall"].excluded == {}
    assert scored.metrics["recall"].worst is not None


def test_an_arm_that_attempted_nothing_is_unknown_loss_not_zero_loss():
    """`_arm_loss_share` returns None, never 0.0, when the arm carried no attempts.

    The docstring calls this load-bearing and nothing pinned it: both call sites spell
    `share is not None and share > ...` / `if loss and ...`, and `0.0` is falsey and fails
    every comparison, so `return 0.0` is a provable no-op at both. The commit that added
    the helper claimed a mutation killed it; that mutation had rewritten `_worst_case_gap`
    by accident (review finding on #379). An absence and a measured zero are different
    facts, and the type is the only thing that says so."""
    assert _arm_loss_share([{"qid": "a", "kind": "recall", "trials": 0,
                             "treatment_errors": 0}], "treatment") is None
    assert _arm_loss_share([{"qid": "a", "kind": "recall", "trials": 10}],
                           "treatment") is None, "no counter at all is not a zero loss"
    assert _arm_loss_share([{"qid": "a", "kind": "recall", "trials": 10,
                             "treatment_errors": 0}], "treatment") == 0.0


def test_a_row_stating_no_trial_count_is_read_as_one_call_not_zero():
    """The `trials` fallback, inherited verbatim from trigger 2 and untested on both.

    Mutating the default from 1 to 0 left the suite green. It decides what a row carrying a
    loss counter but no trial count means: one call (so the loss is real and counts) or no
    calls (so the arm drops out of the denominator and the loss vanishes). The second reads
    a reported failure as no evidence of failure, which is the direction this whole
    function exists to refuse."""
    rows = [{"qid": "a", "kind": "recall", "trials": 10, "attempts": 20, "answer_ok": 10},
            {"qid": "b", "kind": "recall", "attempts": 2, "answer_ok": 0,
             "treatment_errors": 1}]
    assert "trials" not in rows[1]
    # Asserted through `_arm_loss_share` rather than through `_unmeasured` since #383.
    # The observable had to move because trigger 4 now divides by every row's attempts,
    # under which BOTH defaults (1 -> 1/11, 0 -> 1/10) sit below the 0.20 threshold and
    # the gate can no longer see which one is in force. The helper still can, and it is
    # where the fallback actually lives:
    #   default 1 -> the row contributes 1 attempt, share 1/1 = 1.0
    #   default 0 -> the row contributes nothing, no attempts at all, share None
    # None is the reading this function exists to refuse: a reported failure turned into
    # no evidence of failure.
    assert _arm_loss_share(rows, "treatment") == pytest.approx(1.0), (
        "a stated loss on a row with no trial count is still a loss")


# --------------------------------------------------------------------------- #
# #381 — the same predicate, on the one metric that PAIRS.
#
# `final-accuracy` never got #371's fix. It pairs against a measured no-drop control (#269),
# so it routes `_accuracy_gate` -> `arm_gap` -> `_gap`, whose only transport gate is
# `_unmeasured`'s loss SHARE. Below `UNMEASURED_FAIL_SHARE` that gate is silent and every
# lost treatment call scores a MISS, so the gap IS the loss — the identical arithmetic and
# the identical rendered contradiction #371 was filed on, one metric over.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("t_err", [1, 2])
def test_a_treatment_loss_that_fully_EXPLAINS_the_accuracy_gap_withholds_it(t_err):
    """The issue reproduction, executed on `main` @ 22eea92 before the fix as:

        t_err=1   loss=10%   accuracy BLOCK   excluded={}   gap=-0.100
        t_err=2   loss=20%   accuracy BLOCK   excluded={}   gap=-0.200

    The model is correct on every call that LANDED and the control lost nothing, so every
    point of that gap is transport. Both sizes sit in the live band: tolerance is 5% and
    `_unmeasured` is a strict `>` against 0.20, so 10% and 20% clear neither.
    """
    rows = (_perfect_on_landed("recall", 24, 10, t_err)
            + _perfect_on_landed("precision", 24, 10, t_err))
    assert _arm_loss_share(rows, "treatment") <= UNMEASURED_FAIL_SHARE, (
        "fixture must sit inside the band the old gate never reached")
    assert not _unmeasured(rows), "no other gate may be doing the work"
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {"m": "unmeasured"}, (
        f"{t_err}/10 lost fully explains the gap and must be withheld, not failed")


def test_a_demonstrated_accuracy_regression_inside_the_band_is_still_scored():
    """The invariant the whole design rests on: an exclusion must never IMPROVE a verdict.

    10% of the treatment arm lost and 50% scored on a control that scored 100%. Crediting
    every lost call still leaves the arm 40 points behind — far past anything transport can
    account for — so this is behaviour and is published. Without this, the predicate is
    indistinguishable from `UNMEASURED_FAIL_SHARE`'s refused survival threshold.
    """
    rows = _flat("recall", 24, 100, 10, 50) + _flat("precision", 24, 100, 10, 50)
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {}, "a demonstrated regression is never withheld"
    assert v.metrics["accuracy"].worst is not None
    assert v.metrics["accuracy"].worst.gap == pytest.approx(-0.5)
    assert v.directive is Directive.BLOCK


def test_an_accuracy_gap_that_still_PASSES_under_loss_is_scored_not_withheld():
    """The `not passes_tolerance(gap)` clause, which is otherwise a pure no-op to delete.

    97 of 100 against a 100% control is inside the 5% tolerance as scored, so there is
    nothing for the 2% loss to explain. Withholding it would downgrade a measured SHIP to
    NOT_CONCLUDED — the cheap direction, and still the harness lying about what it knows.
    """
    rows = _flat("recall", 24, 100, 2, 97) + _flat("precision", 24, 100, 2, 97)
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {}, "a passing metric is never withheld"
    assert v.metrics["accuracy"].worst is not None and v.metrics["accuracy"].worst.passed


def test_the_accuracy_predicate_boundary_is_the_TOLERANCE_line_not_a_loss_SHARE():
    """The boundary itself. 100 trials, 10 lost, control perfect throughout.

    85 + 10 - 100 = -5%, exactly the tolerance line, which `passes_tolerance` accepts at
    its epsilon — still explained. 84 is one point past what the loss can account for, so
    it is behaviour. Nothing about either number is near `UNMEASURED_FAIL_SHARE`, which is
    the point: the loss share and the tolerance are different numbers.
    """
    on_line = dropeval_verdict({"m": _flat("recall", 24, 100, 10, 85)
                                + _flat("precision", 24, 100, 10, 85)})
    assert on_line.metrics["accuracy"].excluded == {"m": "unmeasured"}
    scored = dropeval_verdict({"m": _flat("recall", 24, 100, 10, 84)
                               + _flat("precision", 24, 100, 10, 84)})
    assert scored.metrics["accuracy"].excluded == {}
    assert scored.metrics["accuracy"].worst is not None


def test_the_credited_loss_is_read_over_the_PAIRED_subset_not_every_row():
    """`_arm_loss_share(g.rows, ...)` — mutating `g.rows` to `rows` survives every fixture
    above, because in all of them pairing drops nothing.

    It cannot survive here. `paired_rows` discards the 30 questions whose CONTROL lost a
    call, and those questions lost no treatment calls at all — so they dilute the treatment
    share from the 10% actually behind the published accuracy down to 4%, which no longer
    explains the gap and publishes a FAIL the harness manufactured.

    The rule is that the credited loss and the accuracy must share a denominator: 90% was
    computed over the paired 20, so the loss must be too.
    """
    paired = [{"qid": f"a{i}", "kind": "recall", "trials": 10, "attempts": 20,
               "answer_ok": 9, "retrieve_ok": 9, "handle_ok": 9,
               "errors": 1, "treatment_errors": 1, "control_errors": 0,
               "control_ok": 10, "control_trials": 10} for i in range(20)]
    dropped_by_pairing = [{"qid": f"b{i}", "kind": "recall", "trials": 10, "attempts": 20,
                           "answer_ok": 10, "retrieve_ok": 10, "handle_ok": 10,
                           "errors": 1, "treatment_errors": 0, "control_errors": 1,
                           "control_ok": 9, "control_trials": 9} for i in range(30)]
    rows = paired + dropped_by_pairing
    assert not _unmeasured(rows), "no other gate may be doing the work"
    assert _arm_loss_share(rows, "treatment") == pytest.approx(0.04)
    assert _arm_loss_share(paired, "treatment") == pytest.approx(0.10)
    g = _accuracy_gate(rows)
    assert g.excluded == "unmeasured", (
        "the 10% behind the paired 90% explains the gap; the diluted 4% does not")


def test_only_the_TREATMENT_arms_loss_is_credited_to_the_accuracy_gap():
    """Why this is one-armed even though the metric has two arms.

    `dropeval.py` emits `control_trials = trials - control_errors` but deliberately emits no
    `answer_trials`, so a control loss leaves its own denominator AND takes the whole row
    out via `paired_rows`, while a treatment loss stays in scoring a MISS. Crediting the
    control could only push the gap further negative, so it can never rescue a FAIL.

    Here the treatment arm lost nothing and the control lost 10% of its calls on rows that
    still pair (`control_trials` intact, `control_errors` set — the shape a merged pack
    reaches). The gap is real behaviour and must be published; reading the CONTROL's share
    instead would credit 10% and withhold it.
    """
    ANSWER_OK = 9  # NOT 8, and the difference is the whole test — see below.
    rows = [{"qid": f"{k}{i}", "kind": k, "trials": 10, "attempts": 20,
             "answer_ok": ANSWER_OK, "retrieve_ok": 10, "handle_ok": 10,
             "errors": 1, "treatment_errors": 0, "control_errors": 1,
             "control_ok": 10, "control_trials": 10}
            for k in ("recall", "precision") for i in range(24)]
    assert _credited_loss_share(rows, "treatment") == 0.0
    assert _credited_loss_share(rows, "control") == pytest.approx(0.10)
    assert not _unmeasured(rows), "no other gate may be doing the work"
    # THE FIXTURE HAS TO PUT THE MUTANT INSIDE TOLERANCE OR IT PINS NOTHING. This scored
    # `answer_ok: 8` when written, giving a -20% gap: crediting the control's 10% reaches
    # -10%, still outside the 5% tolerance, so the swapped-arm mutant did not withhold
    # either and the assertion below held with the arm swapped AND with the whole #381
    # block deleted. Adversarial review of #382 caught it. At 9 the gap is -10%, the
    # mutant's credit lands on 0.0 and withholds, and only the real code publishes.
    assert passes_tolerance(ANSWER_OK / 10 + 0.10 - 1.0), (
        "crediting the CONTROL's loss must bring this gap inside tolerance, or the "
        "swapped-arm mutant behaves identically and this test cannot fail")
    g = _accuracy_gate(rows)
    assert g.excluded is None, "the treatment arm lost nothing, so nothing is explained"
    assert g.form_acc - g.control_acc == pytest.approx(-0.1)


def test_a_withheld_final_accuracy_does_not_claim_the_arms_failed_to_PAIR():
    """#381 at the renderer. The verdict object is not what an operator meets.

    Every question here completed all 10 trials on BOTH arms — `control_trials == trials`
    on every row, so `paired_rows` drops nothing and all 24 pair. The old first sentence,
    "Too few calls completed on BOTH arms to compare", was written for `_unmeasured` and
    the empty-pairing gate and is simply false on this new path; printing it would send the
    operator looking for a pairing failure that did not happen.

    Asserted on the RENDERED bullet rather than the reason string, because the reason is
    `"unmeasured"` on all three paths by design — the sentence is the only place the
    difference is visible to a reader, so it is the only place it can be pinned.
    """
    rows = (_perfect_on_landed("recall", 24, 10, 1)
            + _perfect_on_landed("precision", 24, 10, 1))
    assert all(r["control_trials"] == r["trials"] for r in rows), "every row must pair"
    report = build_dropeval_report({"m": rows})
    assert "**FAIL**" not in report, "no behavioural verdict may survive a transport loss"
    assert "**final-accuracy: not gated for `m`**" in report
    bullet = next(ln for ln in report.splitlines()
                  if ln.startswith("- **final-accuracy: not gated"))
    assert "Too few calls completed on BOTH arms" not in bullet, bullet
    assert "lost to account for the gap on their own" in bullet, bullet
    # The SECOND false opening clause, caught by review of the first fix: "Not enough of
    # this comparison survived to read" is equally false here -- all 48 questions survived
    # pairing, at a clean -10% gap. Banning the old clause and requiring the shared
    # disjunction (both asserted above) left this mutable: reverting ONLY the opening
    # sentence, with the disjunction left untouched, passed both of those asserts. Pinned
    # directly.
    assert "survived to read" not in bullet, bullet
    assert "not trusted as evidence" in bullet, bullet
    # The table cell is the withheld marker, not a percentage — the defect an operator met
    # on the mechanism half was a number printed under a paragraph disowning its rows.
    header = next(ln for ln in report.splitlines() if ln.startswith("| Model |"))
    cols = [c.strip() for c in header.split("|")[1:-1]]
    row = next(ln for ln in report.splitlines() if ln.startswith("| `m` |"))
    cells = dict(zip(cols, [c.strip() for c in row.split("|")[1:-1]], strict=True))
    assert cells["final-accuracy"] == "not gated", row


# --------------------------------------------------------------------------- #
# Review of #382. The credit and the accuracy must share a ROW SET, not merely a
# per-row denominator — `_arm_loss_share` skips rows carrying no counter, and a
# credit is subtracted from a FAIL.
# --------------------------------------------------------------------------- #


def _merged_pack(kind, n_legacy, n_current, *, ok_legacy, ok_current, t_err, trials=10):
    """A pack merged from one producer predating `treatment_errors` (#299) and one
    emitting it. `_accuracy_gate` admits this: it requires `control_ok` on every row, which
    a post-#269 legacy pack carries, and nothing requires the error counters."""
    legacy = [{"qid": f"{kind}L{i}", "kind": kind, "trials": trials, "attempts": trials * 2,
               "answer_ok": ok_legacy, "retrieve_ok": ok_legacy, "handle_ok": ok_legacy,
               "control_ok": trials, "control_trials": trials} for i in range(n_legacy)]
    current = [{"qid": f"{kind}C{i}", "kind": kind, "trials": trials,
                "attempts": trials * 2, "answer_ok": ok_current,
                "retrieve_ok": ok_current, "handle_ok": ok_current,
                "errors": t_err, "treatment_errors": t_err, "control_errors": 0,
                "control_ok": trials, "control_trials": trials} for i in range(n_current)]
    return legacy + current


def test_a_merged_pack_cannot_credit_a_loss_the_scored_rows_never_paid():
    """The finding both reviewers of #382 reached independently, and the invariant it
    breaks is the one this whole gate is built around.

    `_form_stats` scores EVERY paired row; `_arm_loss_share` skips rows carrying no
    counter. On a merged set that makes the credit a fraction of a SUBSET applied to the
    WHOLE. Executed on the branch before the fix: 40 legacy rows at 7/10 with no loss at
    all, plus 40 current rows at 8/10 losing 2 each, against a 100% control —

        gap                       -0.25
        credited (subset, 400)     0.20  ->  -0.05  ->  WITHHELD
        honest   (all rows, 800)   0.10  ->  -0.15  ->  three times tolerance

    A 15-point regression that survives crediting every genuinely lost call was withheld.
    `NOT_CONCLUDED (2) < BLOCK (3)`, so the exclusion IMPROVED the model's verdict — what
    `UNMEASURED_FAIL_SHARE`'s comment refuses and what #379's review found 240 of.
    """
    rows = (_merged_pack("recall", 40, 40, ok_legacy=7, ok_current=8, t_err=2)
            + _merged_pack("precision", 40, 40, ok_legacy=7, ok_current=8, t_err=2))
    assert not _unmeasured(rows), "no other gate may be doing the work"
    assert _arm_loss_share(rows, "treatment") == pytest.approx(0.20), (
        "the SIBLING helper still reports the subset share — that is its job")
    assert _credited_loss_share(rows, "treatment") == pytest.approx(0.10), (
        "the credited share is read over the same rows the accuracy was")
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {}, (
        "a 15-point gap that survives crediting every lost call is behaviour")
    assert v.metrics["accuracy"].worst is not None
    assert v.directive is Directive.BLOCK


def test_a_merged_pack_cannot_credit_the_MECHANISM_metrics_either():
    """The same defect at #371's site (`dropeval_verdict`'s fixed-ideal loop), which
    shipped with it and which #382's fix inherits rather than introduces.

    Kept as its own test because the two sites reach `_credited_loss_share` by different
    routes — one through `_accuracy_gate` -> `arm_gap` -> `_gap`, one directly off
    `by_kind` — so a fix applied to only one leaves the other live.
    """
    rows = (_merged_pack("recall", 40, 40, ok_legacy=7, ok_current=8, t_err=2)
            + _merged_pack("precision", 40, 40, ok_legacy=7, ok_current=8, t_err=2))
    v = dropeval_verdict({"m": rows})
    for mech in ("recall", "precision"):
        assert v.metrics[mech].excluded == {}, (
            f"{mech}: a 25-point miss with a 10% honest loss is behaviour, not transport")


def test_the_credit_never_exceeds_what_the_arm_could_have_scored():
    """The sharpened form: an inflated credit does not merely withhold too much, it claims
    an accuracy above 100%.

    5 carrying rows losing 2 of 10, 95 legacy rows at 9/10, control perfect. The subset
    share is 0.20, so `acc + loss` reaches 1.095 — crediting the arm with more successes
    than it had trials, under a comment asserting the ceiling is real.
    """
    rows = (_merged_pack("recall", 95, 5, ok_legacy=9, ok_current=8, t_err=2)
            + _merged_pack("precision", 95, 5, ok_legacy=9, ok_current=8, t_err=2))
    acc = 0.895
    assert _arm_loss_share(rows, "treatment") + acc > 1.0, (
        "fixture must reach an impossible ceiling under the subset share, or it is not "
        "reproducing the finding")
    assert _credited_loss_share(rows, "treatment") + acc <= 1.0
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {}, "a -9.5pt gap is not explained by 1% loss"


def test_a_credited_share_over_one_is_REFUSED_where_its_sibling_fires():
    """The one place `_credited_loss_share` deliberately inverts `_arm_loss_share`.

    An arm reporting more errors than calls is an emitter bug with no benign form. Its
    sibling lets the share go over 1.0 so `_unmeasured` FIRES — withholding asks a human to
    look, which is the safe direction there. Here the same share would BUY a withheld FAIL
    off that bug, so it is refused and the metric is scored instead.
    """
    rows = [{"qid": "a", "kind": "recall", "trials": 1, "attempts": 2,
             "answer_ok": 0, "treatment_errors": 10}]
    assert _arm_loss_share(rows, "treatment") == 10.0, "the sibling still goes over 1.0"
    assert _credited_loss_share(rows, "treatment") is None
    # ...and a share of exactly 1.0 is a real measurement, not the bug: it is refused only
    # ABOVE the line, so the boundary is observed rather than assumed.
    assert _credited_loss_share([dict(rows[0], treatment_errors=1)], "treatment") == 1.0


def test_a_row_set_where_NOBODY_counted_credits_nothing():
    """`None`, never 0.0 — the distinction `_arm_loss_share` documents, one helper over.

    Absent counters mean the loss is unknown. Returning 0.0 would be indistinguishable at
    the call sites (`if loss and ...` treats both as falsey today), but the two are
    different facts and a future site spelling `is not None` would silently credit a
    measured zero to a pack that measured nothing.
    """
    legacy = _merged_pack("recall", 20, 0, ok_legacy=8, ok_current=0, t_err=0)
    assert not any("treatment_errors" in r for r in legacy)
    assert _credited_loss_share(legacy, "treatment") is None
    assert _credited_loss_share([{"qid": "z", "trials": 0, "treatment_errors": 0}],
                                "treatment") is None, "no calls is not a zero loss"


def test_the_survivor_COUNT_still_renders_under_the_new_exclusion():
    """Review finding 2 of #382: the fix deleted the evidence its own remedy cites.

    `> **Questions surviving the pairing**` is listed only for models that are scored or
    `underpowered`, and the block is guarded by `if broken:` — a condition the #381 cause
    always satisfies. So the report claimed the comparison had not survived while
    suppressing the `48/48` proving pairing lost nothing. `origin/main` printed it for this
    exact fixture.
    """
    rows = (_perfect_on_landed("recall", 24, 10, 1)
            + _perfect_on_landed("precision", 24, 10, 1))
    report = build_dropeval_report({"m": rows})
    assert "**Questions surviving the pairing**" in report
    line = next(ln for ln in report.splitlines() if "surviving the pairing" in ln)
    assert "`m` 48/48" in line, line


def test_no_renderer_claims_a_CALL_COUNT_about_a_fully_paired_withheld_run():
    """The other half of finding 2, and it pre-dates #382: `REASON_LABEL["unmeasured"]` read
    "too few calls to compare", which the mechanism bullets have printed on 48/48-paired
    runs since #371 shipped. #382 extended it to the accuracy column, so it is fixed here.

    The label now names the CONSEQUENCE the four routes to `"unmeasured"` share -- no
    verdict about the model follows from this run -- rather than a count or a cause, both
    of which are false of at least one route. A second round of review on #382 found the
    first replacement, "transport loss", was a false CAUSE (see
    `test_a_zero_loss_exclusion_does_not_claim_transport_lost_the_calls` below); this test
    only ever pinned the false-COUNT defect and would not have caught that one.
    """
    rows = (_perfect_on_landed("recall", 24, 10, 1)
            + _perfect_on_landed("precision", 24, 10, 1))
    report = build_dropeval_report({"m": rows})
    assert "too few calls to compare" not in report, (
        "480 of 480 control calls landed and 48 of 48 questions paired")
    for metric in ("retrieve-recall", "final-accuracy"):
        bullet = next(ln for ln in report.splitlines()
                      if ln.startswith(f"- **{metric}: not gated"))
        assert "no usable comparison" in bullet, bullet


def test_a_zero_loss_exclusion_does_not_claim_transport_lost_the_calls():
    """Review finding 1 of the #382 remediation itself: "transport loss" fixed the
    call-count falsehood and introduced a cause falsehood in its place, one route over.

    `_unmeasured` trigger 1 fires at ZERO calls lost -- an arm that was simply never run.
    Executed against "transport loss": `build_diff_report` rendered "**Not measured** --
    transport loss, so no accuracy is published for: `m` (0/240 calls lost). No calls were
    lost, so transport is not the cause: ..." -- the claim and its own refutation six words
    apart, in the same sentence. This is `#338`'s defect re-entering through the shared
    vocabulary: `REASON_LABEL["unmeasured"]` must be true at EVERY route, including the one
    where nothing was lost at all.
    """
    from terse.report import build_diff_report

    diff_rows = [{
        "qid": f"q{i}", "qtype": "lookup", "trials": 20, "attempts": 20, "fails": 0,
        "diff_ok": 0, "diff_trials": 0, "terse_ok": 20, "terse_trials": 20,
    } for i in range(12)]
    assert _unmeasured(diff_rows), "fixture must trip trigger 1 (a zero-trial arm)"
    assert sum(r["fails"] for r in diff_rows) == 0, "fixture must lose zero calls"
    report = build_diff_report({"m": diff_rows})
    line = next(ln for ln in report.splitlines() if "Not measured" in ln)
    assert "0/240 calls lost" in line, line
    # Scoped to the LABEL itself, not the sentence that follows it -- that sentence's whole
    # job is to say "transport is not the cause" at this route, and banning the word from
    # the line would forbid the correct rebuttal along with the defect.
    label = line.split("\u2014", 1)[1].split(", so no accuracy", 1)[0]
    for word in ("transport", "unanswered", "unreachable"):
        assert word not in label.casefold(), (
            f"the label {label!r} claims a cause ({word!r}) this route's own count "
            f"(0/240 calls lost) refutes")


def test_a_treatment_only_loss_is_not_attributed_to_the_control_column():
    """Review finding 2: the dropeval table's `control (no drop)` cell rendered
    `REASON_LABEL[reason]` unconditionally, so a #381-route exclusion -- loss entirely on
    the TREATMENT arm -- printed "transport loss" under a column headed by the arm that
    lost nothing.

    A cause-bearing label has an arm to get wrong; a consequence-only one does not, so this
    is pinned by asserting the CURRENT label carries no arm-specific word rather than by
    asserting a specific arm name is absent -- the same argument `REASON_LABEL`'s own note
    makes for why route 4 forced a consequence label in the first place.
    """
    rows = (_perfect_on_landed("recall", 24, 10, 1)
            + _perfect_on_landed("precision", 24, 10, 1))
    report = build_dropeval_report({"m": rows})
    row = next(ln for ln in report.splitlines() if ln.startswith("| `m` |"))
    assert "0 lost" not in row or "control" not in row.casefold(), row  # sanity: no stray count
    for word in ("transport", "treatment", "control"):
        assert word not in REASON_LABEL["unmeasured"].casefold(), (
            f"REASON_LABEL['unmeasured'] is {REASON_LABEL['unmeasured']!r}, which names an "
            f"arm -- and the table prints it under 'control (no drop)' regardless of "
            f"which arm actually lost the calls")


# --------------------------------------------------------------------------- #
# #383. The same subset-denominator defect #382 fixed at its two CREDIT sites,
# one gate upstream of them: `_unmeasured`'s trigger 4. `_gap` calls
# `_unmeasured` first, so firing there withholds the run before `_accuracy_gate`
# or `dropeval_verdict` score anything at all.
# --------------------------------------------------------------------------- #


def _pack_383():
    """#383's fixture: a merged pack whose SUBSET share crosses UNMEASURED_FAIL_SHARE
    while its whole-run share is nowhere near it, and whose honest gap is a demonstrated
    regression far past `_GAP_TOLERANCE`."""
    return (_merged_pack("recall", 95, 5, ok_legacy=9, ok_current=5, t_err=3)
            + _merged_pack("precision", 95, 5, ok_legacy=9, ok_current=5, t_err=3))


def test_trigger_4_reads_the_run_denominator_not_the_carrying_subset():
    """Executed on `origin/main` @ 1138572 before the fix: `_unmeasured` returned True, the
    model was excluded as `unmeasured`, and `worst` came back None -- so a 12-point
    regression against a 100% control was withheld on the strength of 15 lost calls in
    1,000.

    `NOT_CONCLUDED (2) < BLOCK (3)`, so that exclusion IMPROVES the model's verdict, which
    is exactly what `UNMEASURED_FAIL_SHARE`'s own comment forbids.
    """
    rows = _pack_383()
    assert _arm_loss_share(rows, "treatment") == pytest.approx(0.30), (
        "the subset share is unchanged -- reporting the degraded slice's own rate is "
        "still that helper's job, and #383 is about which caller may use it")
    assert _distrust_loss_share(rows, "treatment") == pytest.approx(0.015)
    assert 0.015 < UNMEASURED_FAIL_SHARE < 0.30, (
        "the fixture is only meaningful while the threshold sits between the two shares")

    assert not _unmeasured(rows), "1.5% whole-run loss must not withhold the run"
    v = dropeval_verdict({"m": rows})
    assert v.metrics["accuracy"].excluded == {}
    worst = v.metrics["accuracy"].worst
    assert worst is not None, "the regression must reach the verdict, not be withheld"
    assert worst.gap == pytest.approx(-0.12)
    assert worst.passed is False
    assert v.directive is Directive.BLOCK


def test_trigger_4_still_fires_on_a_loss_that_is_real_across_the_whole_run():
    """The other direction, so the fix cannot be mistaken for disabling the trigger: when
    every row carries the counter, the two denominators coincide and a 30% loss withholds
    exactly as it did before."""
    rows = _merged_pack("recall", 0, 100, ok_legacy=9, ok_current=5, t_err=3)
    assert _distrust_loss_share(rows, "treatment") == pytest.approx(0.30)
    assert _unmeasured(rows)


def test_an_emitter_reporting_more_errors_than_attempts_still_withholds():
    """The guard that a swap to `_credited_loss_share` would have silently dropped.

    An arm reporting more errors than it had attempts is an emitter bug with no benign
    form. `_credited_loss_share` REFUSES an over-1.0 share on purpose -- there it would buy
    a withholding off a bug -- and a refused share is a share that cannot fire. Trigger 4
    needs the opposite: firing withholds and asks a human to look, which is the right
    response to a bug. This is why #383 added a third helper instead of reusing the
    sibling.
    """
    rows = _merged_pack("recall", 0, 10, ok_legacy=9, ok_current=5, t_err=11, trials=10)
    assert _distrust_loss_share(rows, "treatment") == pytest.approx(1.1)
    assert _credited_loss_share(rows, "treatment") is None, (
        "the credit helper refuses it -- unchanged by #383")
    assert _unmeasured(rows), "trigger 4 must still fire on an impossible share"


def test_the_two_whole_run_helpers_agree_wherever_the_share_is_sane():
    """`_credited_loss_share` is a thin policy wrapper over `_distrust_loss_share`, so the
    only input on which they may differ is one over 1.0. Pinned because a future edit that
    re-copies the derivation into the wrapper would pass every other test here."""
    for rows in (_pack_383(),
                 _merged_pack("recall", 0, 100, ok_legacy=9, ok_current=5, t_err=3),
                 _merged_pack("recall", 50, 50, ok_legacy=7, ok_current=8, t_err=2)):
        assert _credited_loss_share(rows, "treatment") == _distrust_loss_share(
            rows, "treatment")


def test_absence_is_not_a_zero_share_for_the_distrust_helper():
    """Same contract as its siblings: no row carrying the counter is nothing known, which
    must not be compared to a threshold as though it were a clean run."""
    rows = _merged_pack("recall", 40, 0, ok_legacy=9, ok_current=5, t_err=3)
    assert _distrust_loss_share(rows, "treatment") is None
    assert not _unmeasured(rows)


# --------------------------------------------------------------------------- #
# #383, second half. Trigger 2 kept the carrying-rows denominator trigger 4 shed,
# so the withhold did not disappear -- it relocated one trigger up. Found by
# review of the first half.
# --------------------------------------------------------------------------- #


def _attempts_key_fleet(*, attempts_on_legacy):
    """Two fleets differing in ONE key: whether the loss-free legacy rows carry `attempts`.

    Every row here RAN the control arm and carries `control_trials`; the legacy rows simply
    come from a producer that emitted no pooled `attempts`. The control loss is identical
    in both -- 30 calls of 2,000."""
    rows = []
    for kind in ("recall", "precision"):
        for i in range(95):
            r = {"qid": f"{kind}L{i}", "kind": kind, "trials": 10, "answer_ok": 9,
                 "retrieve_ok": 9, "handle_ok": 9, "control_ok": 10, "control_trials": 10}
            if attempts_on_legacy:
                r["attempts"] = 20
            rows.append(r)
        for i in range(5):
            rows.append({"qid": f"{kind}C{i}", "kind": kind, "trials": 10, "attempts": 20,
                         "answer_ok": 5, "retrieve_ok": 5, "handle_ok": 5,
                         "control_ok": 7, "control_trials": 7,
                         "errors": 3, "control_errors": 3, "treatment_errors": 0})
    return rows


def test_trigger_2_does_not_decide_a_verdict_on_the_presence_of_an_attempts_key():
    """Executed on the first half of #383 before this fix:

        legacy rows WITHOUT `attempts`:  _unmeasured True,  excluded {'m': 'unmeasured'},
                                         worst None
        legacy rows WITH    `attempts`:  _unmeasured False, excluded {}, worst gap -0.10

    Same arm, same 30 lost calls in 2,000 (1.5%), and the verdict turned on one unrelated
    key. Trigger 2 read 30/100 = 0.30 because its denominator was the carrying rows only --
    the identical subset-against-a-whole-run-threshold defect the first half removed from
    trigger 4, one trigger up.
    """
    without = _attempts_key_fleet(attempts_on_legacy=False)
    with_ = _attempts_key_fleet(attempts_on_legacy=True)
    assert not _unmeasured(without), "1.5% control loss must not withhold the run"
    assert not _unmeasured(with_)

    a, b = dropeval_verdict({"m": without}), dropeval_verdict({"m": with_})
    assert a.metrics["accuracy"].excluded == b.metrics["accuracy"].excluded == {}
    assert a.metrics["accuracy"].worst is not None
    assert a.metrics["accuracy"].worst.gap == pytest.approx(
        b.metrics["accuracy"].worst.gap), (
        "an unrelated key must not move the measured gap")
    assert a.metrics["accuracy"].worst.gap == pytest.approx(-0.10)


def test_trigger_2_keeps_a_not_attempted_arm_out_of_its_own_denominator():
    """The reason the denominator widened to `key in r` and not to `rows` wholesale.

    A `--no-control` pack emits no `control_trials` at all on the rows that ran without
    one. Dividing a real control loss by calls those rows never made would understate it
    and let a substantially dead control arm publish. Rows carrying the key ran the arm;
    rows without it did not, and stay out.
    """
    ran = [{"qid": f"c{i}", "kind": "recall", "trials": 10, "attempts": 20, "answer_ok": 9,
            "control_ok": 2, "control_trials": 2} for i in range(5)]
    never_ran = [{"qid": f"n{i}", "kind": "recall", "trials": 10, "attempts": 10,
                  "answer_ok": 9} for i in range(95)]
    assert _unmeasured(ran + never_ran), (
        "8 of 10 control calls lost on every row that ran it is a withhold, and 95 rows "
        "that never ran the arm must not dilute it")


def test_trigger_2_still_refuses_phantom_loss_from_a_row_with_no_attempts_key():
    """The #339 guard the widened denominator had to preserve: a row carrying the arm key
    but no `attempts` may enter the DENOMINATOR, never the numerator. Its `trials` may
    disagree with the arm count for benign reasons, and reading that as loss is what let
    one merged row swing a whole model to withheld with nothing actually lost."""
    rows = [{"qid": f"a{i}", "kind": "recall", "trials": 10, "attempts": 20,
             "answer_ok": 10, "control_ok": 10, "control_trials": 10} for i in range(10)]
    rows.append({"qid": "legacy", "kind": "recall", "trials": 10, "answer_ok": 10,
                 "control_ok": 0, "control_trials": 0})
    assert not _unmeasured(rows), (
        "the row with no `attempts` contributes its attempts, not a 10-call loss")
