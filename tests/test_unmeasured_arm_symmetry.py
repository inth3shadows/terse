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
    UNMEASURED_FAIL_SHARE,
    Directive,
    _unmeasured,
    build_dropeval_report,
    dropeval_verdict,
    inconclusive_models,
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
    assert _unmeasured(legacy + current), (
        "1 lost of the 3 calls anyone counted is 33%, not 0.7% of 144 calls nobody did")
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
    assert "too few calls to compare" in report
    assert "**Where they failed** (per arm" in report


def test_the_accuracy_gate_reaches_the_same_verdict_whichever_arm_lost_the_calls():
    """The symmetry, scoped to the gate `_unmeasured` actually controls.

    NAMED FOR THE GATE, NOT THE RUN, and that is a correction rather than a nicety. An
    earlier revision called itself
    `test_the_report_reaches_the_same_verdict_whichever_arm_lost_the_calls` and asserted
    only that both renderings contain "too few calls to compare" and "not gated" — both of
    which are true of both reports while their run-level `Directive`s differ. A test named
    for an invariant it cannot observe failing is worse than no test: it is the #352
    blind spot re-created inside #352's own fix. The residual asymmetry it could not see is
    pinned deliberately by the test below."""
    treatment = build_dropeval_report({"m": _both_kinds(trials=10, t_err=4)})
    control = build_dropeval_report({"m": _both_kinds(trials=10, c_err=4)})
    for report in (treatment, control):
        assert "too few calls to compare" in report
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
    assert "**PASS**" not in report
    # The two mechanism metrics now say they were not gated, in the verdict prose that
    # used to carry `retrieve-recall 60% vs ideal (100%) ... keep drop-to-retrieve off`.
    for metric in ("retrieve-recall", "no-overfetch"):
        assert f"**{metric}: not gated for `m`**" in report, metric
    assert "INCONCLUSIVE for enabling" in report
    # And the table cell is the withheld marker, not a number. Asserting on the ROW is what
    # separates this from the verdict-object test: the defect an operator met was a
    # percentage printed in a column, under a paragraph disowning the rows behind it.
    row = next(ln for ln in report.splitlines() if ln.startswith("| `m` |"))
    recall_cell, precision_cell = row.split("|")[3].strip(), row.split("|")[4].strip()
    assert recall_cell == "not gated" and precision_cell == "not gated", row
    # handle-accuracy is deliberately NOT covered here. It is the display-only column no
    # gate reads (see `test_gap_gate_boundary.py`'s allowlist note), so it still renders a
    # percentage on this run — an inconsistency worth its own decision, not a silent
    # widening of this fix's scope.


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
    assert _unmeasured(rows), "a stated loss on a row with no trial count is still a loss"
