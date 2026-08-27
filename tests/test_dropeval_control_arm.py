"""dropeval's no-drop CONTROL arm (#269).

final-accuracy used to be scored against a fixed 100% ideal that was never run, and that
metric alone gated the "keep drop-to-retrieve off" verdict. Because the metric is JSON
value-equality against a 500+ character prose field, and a model handed the UN-dropped
payload paraphrases it too, the gate was largely measuring verbatim-reproduction ability
and billing the shortfall to the drop. A real run scored every mechanism metric at 100%
and final-accuracy at 54%, and emitted FAIL.

These tests pin the three things that make the new arm honest: the control differs from
the treatment by *only* the drop, a missing control withholds the verdict instead of
falling back to the unrun ideal, and neither arm can score a failed call as a wrong answer.
"""

from __future__ import annotations

from terse import dropeval
from terse import policy as policy_mod
from terse.report import _accuracy_gate, build_dropeval_report, dropeval_gap_rows

# --------------------------------------------------------------------------- #
# The control policy: same rule, minus the drop.
# --------------------------------------------------------------------------- #


def _rule(**fields):
    return policy_mod.Rule(tool_glob="t", tiers=("minify", "table"), fields=fields)


def test_control_rule_strips_only_the_drop_specs():
    """A control that also loses the codec would confound the drop with an encoding
    change — the same defect one level over."""
    rule = _rule(**{
        "rows[].evidence": {"lossy": "drop-to-retrieve", "min": 10},
        "rows[].id": {"critical": True},
        "rows[].note": {"max": 40},
    })
    ctl = dropeval._control_rule(rule)
    assert "rows[].evidence" not in ctl.fields
    assert ctl.fields["rows[].id"] == {"critical": True}
    assert ctl.fields["rows[].note"] == {"max": 40}
    # Everything outside `fields` must survive verbatim: tiers ARE the codec.
    assert ctl.tiers == rule.tiers
    assert ctl.tool_glob == rule.tool_glob


def test_control_rule_strips_text_span_drops_too():
    """`$text.*` selectors are the drop under test on the non-JSON path; leaving one in
    would hand the control the very cut it is supposed to lack."""
    rule = _rule(**{"$text.code_blocks": {"lossy": "drop-to-retrieve", "min": 10}})
    assert dropeval._control_rule(rule).fields == {}


def test_control_rule_keeps_a_critical_field_that_is_also_drop_marked():
    """`critical` already exempts a field from dropping, so it is not part of the
    treatment and must not be removed from the control either."""
    rule = _rule(**{"rows[].secret": {"lossy": "drop-to-retrieve", "critical": True}})
    assert "rows[].secret" in dropeval._control_rule(rule).fields


def test_control_text_carries_no_drop_markers_and_the_treatment_does():
    obj = {"rows": [{"id": i, "evidence": "E" * 200} for i in range(4)]}
    rule = _rule(**{"rows[].evidence": {"lossy": "drop-to-retrieve", "min": 10}})
    applied, staging = dropeval._staged_apply(obj, rule, "t")
    ctl = dropeval._control_text(obj, rule, "t", is_json=True)
    assert staging, "fixture must actually drop something or it tests nothing"
    assert "__terse_dropped__" in applied.text
    assert "__terse_dropped__" not in ctl
    # The value the treatment hid is present in full in the control.
    assert "E" * 200 in ctl


def test_control_text_on_the_non_json_path_also_carries_no_drop_markers():
    """The is_json=False branch of `_control_text` — `$text.code_blocks`, the drop under
    test on the non-JSON path — had zero coverage: only the rule object was tested
    (`test_control_rule_strips_text_span_drops_too` above), never the text the control
    arm actually sends to the model (review finding 6 on #300)."""
    raw = "intro\n```\n" + "x" * 200 + "\n```\ntail"
    rule = _rule(**{"$text.code_blocks": {"lossy": "drop-to-retrieve", "min": 10}})
    applied, staging = dropeval._staged_apply_text(raw, rule, "t")
    ctl = dropeval._control_text(raw, rule, "t", is_json=False)
    assert staging, "fixture must actually drop something or it tests nothing"
    assert "__terse_dropped__" in applied.text
    assert "__terse_dropped__" not in ctl
    assert "x" * 200 in ctl


# --------------------------------------------------------------------------- #
# The gate: a control that did not run withholds the verdict.
# --------------------------------------------------------------------------- #


# n=24, not 6: #334 withholds a non-failing arm under `_MIN_PAIRED_QUESTIONS` paired
# questions, and every ratio these fixtures pin is a proportion, not a count.
def _rows(n=24, *, answer, control=None, kind="recall", trials=1, errors=0):
    row = {"kind": kind, "trials": trials, "retrieve_ok": trials, "handle_ok": trials,
           "answer_ok": answer, "answer_trials": trials - errors,
           "retrieve_trials": trials - errors, "handle_trials": trials - errors,
           "errors": errors, "attempts": trials * (1 if control is None else 2)}
    if control is not None:
        row |= {"control_ok": control, "control_trials": trials}
    return [dict(row, qid=f"q{i}") for i in range(n)]


def test_no_control_arm_excludes_the_metric_rather_than_defaulting_to_the_ideal():
    """The whole defect was gating against an unrun 100%. Silently reproducing it for
    older packs would keep emitting the false FAIL."""
    g = _accuracy_gate(_rows(answer=0))
    assert g.excluded == "no control arm"


def test_a_control_arm_makes_final_accuracy_a_gap_between_two_measured_arms():
    """The control here ties the treatment at a measured, non-degenerate 50% — NOT
    all-zero. An all-zero control is a different, and more suspicious, case: see
    `test_an_all_zero_control_is_excluded_as_broken_not_scored_as_a_free_pass` below
    (review finding 3 on #300)."""
    rows = _rows(n=12, answer=1, control=1) + _rows(n=12, answer=0, control=0)
    g = _accuracy_gate(rows)
    assert not g.excluded
    # Both arms tie at 50% -> the DROP costs nothing, which the old fixed-100% control
    # reported as a total failure.
    assert g.form_acc == 0.5 and g.control_acc == 0.5
    assert g.form_acc - g.control_acc == 0.0


def test_an_all_zero_control_is_excluded_as_broken_not_scored_as_a_free_pass():
    """A no-drop control that reproduces a value sitting verbatim in its own payload
    scoring exactly 0% across every row is not "the drop is blameless" — #269's own live
    reproduction measured the un-dropped arm at 88%. 0% means the grader or backend
    produced no signal, and treating it as a legitimate tie let a dead control arm PASS
    with "safe to enable drop-to-retrieve" (review finding 3 on #300). Mirrors the
    identical guard fluency's `raw_ok` control already had."""
    g = _accuracy_gate(_rows(answer=0, control=0))
    assert g.excluded == "broken control"


def _both_kinds(**kw):
    """A verdict needs recall AND precision rows.

    A missing kind used to score that gate at a fabricated 0% — `_form_stats([], f)` is
    `(0.0, 0.0)`, which against the fixed 100% ideal published a `-100%` **FAIL**. Since
    #342 it withholds the metric as `"empty"` instead, so a one-kind fixture now produces
    NO verdict rather than a wrong one."""
    return _rows(kind="recall", **kw) + _rows(kind="precision", **kw)


def test_the_metric_that_used_to_fail_now_passes_when_the_drop_is_blameless():
    """#269's live reproduction in miniature: perfect mechanism metrics, a final-accuracy
    well under 100%, and a control that is exactly as low. Old gate: FAIL (the drop was
    charged for a verbatim-reproduction limit). New gate: PASS, because the drop changed
    nothing — which is the entire point of running a control. Mixed 50/50 rather than an
    all-zero control, which is excluded as broken rather than scored as a free pass (see
    `test_an_all_zero_control_is_excluded_as_broken_not_scored_as_a_free_pass`)."""
    rows = (_both_kinds(n=12, answer=1, control=1, trials=1)
            + _both_kinds(n=12, answer=0, control=0, trials=1))
    report = build_dropeval_report({"m": rows})
    assert "no-drop control" in report
    assert "safe to enable drop-to-retrieve" in report


def test_a_dead_control_arm_does_not_pass_the_verdict_end_to_end():
    """The end-to-end form of `test_an_all_zero_control_is_excluded_as_broken_...`: an
    all-zero control must not let `build_dropeval_report` print "safe to enable
    drop-to-retrieve" off a control that measured nothing."""
    rows = _both_kinds(n=5, answer=0, control=0, trials=1)
    report = build_dropeval_report({"m": rows})
    assert "safe to enable drop-to-retrieve" not in report
    assert "not gated" in report


def test_the_report_names_the_real_exclusion_reason_not_a_hardcoded_one():
    """When a control ran and failed (excluded as "broken control"), the report must not
    say "no no-drop control arm was run" — that is a different, false claim. Old code
    hardcoded that phrasing (and the table's "not run" cell) for every exclusion reason
    alike, so a run under `--accept-degraded` whose control errored out on every call
    printed both "the control lost N calls" and "no control arm was run" two sentences
    apart (review finding 2 on #300)."""
    rows = _both_kinds(n=5, answer=0, control=0, trials=1)
    report = build_dropeval_report({"m": rows})
    assert "control arm failed" in report
    assert "no no-drop control arm was run" not in report
    # The table cell must match: "not run" is specifically for "no control arm", and this
    # run's control DID run — it just failed every trial.
    assert "not run" not in report


def test_a_drop_that_really_hurts_still_fails():
    """The fix must not simply make the gate unfalsifiable."""
    rows = _both_kinds(n=5, answer=0, control=1, trials=1)
    report = build_dropeval_report({"m": rows})
    assert "keep drop-to-retrieve off" in report


def test_the_report_says_final_accuracy_is_not_gated_when_no_control_ran():
    report = build_dropeval_report({"m": _both_kinds(n=5, answer=0)})
    assert "not gated" in report
    # It must never claim a comparison it did not make.
    assert "vs no-drop control" not in report


def test_partial_control_coverage_excludes_the_metric_rather_than_scoring_a_subset():
    """A mixed result set (merged runs, a legacy pack, a partially-failed arm) would
    otherwise activate the metric and let `paired_rows` silently discard the control-less
    rows — a verdict computed over a subset nobody was told about."""
    mixed = _rows(n=3, answer=1, control=1) + _rows(n=3, answer=1)
    assert _accuracy_gate(mixed).excluded == "partial control coverage"


def test_mechanism_metrics_alone_do_not_license_enabling_the_drop():
    """Recall/no-overfetch say the model OPERATES the protocol; they say nothing about
    whether the answer it lands on is right. Calling that "safe to enable" is the same
    over-claim as #269's, pointing the other way."""
    report = build_dropeval_report({"m": _both_kinds(n=5, answer=1)})
    assert "INCONCLUSIVE for enabling" in report
    assert "safe to enable drop-to-retrieve" not in report


def test_gap_rows_omit_accuracy_entirely_without_a_control():
    gaps, excluded = dropeval_gap_rows({"m": _rows(answer=1)})
    assert "accuracy" not in gaps["m"]
    assert excluded == {"m": "no control arm"}
    gaps2, excluded2 = dropeval_gap_rows({"m": _rows(answer=1, control=1)})
    assert "accuracy" in gaps2["m"]
    assert excluded2 == {}


# --------------------------------------------------------------------------- #
# The k <= t invariant. Every row this module emits is later divided by its own
# `<form>_trials` inside `_form_stats`, which now raises ValueError on a k>t
# violation rather than silently computing an impossible accuracy (#297) —
# still killing the whole report at render time, just loudly and by design
# instead of via a stray `math.sqrt(negative)`. This crashed a live 3-model run
# and no unit test saw it, because every hand-built fixture happened to satisfy
# the invariant by construction.
# --------------------------------------------------------------------------- #


class _ErroringAnswerer:
    """Fails the first `n_fail` calls, then answers. Mimics a rate limit mid-run."""

    def __init__(self, n_fail: int):
        self.n_fail, self.calls = n_fail, 0

    def __call__(self, messages):
        self.calls += 1
        if self.calls <= self.n_fail:
            return dropeval.Turn(text="", tool_calls=[], error=True)
        return dropeval.Turn(text="whatever", tool_calls=[])


def _live_rows(n_fail: int, trials: int = 3, control: bool = True):
    obj = {"rows": [{"id": i, "evidence": f"{i}" + "E" * 300} for i in range(4)]}
    rule = _rule(**{"rows[].evidence": {"lossy": "drop-to-retrieve", "min": 10}})
    return dropeval.run_drop_payload(obj, "", rule, "t", _ErroringAnswerer(n_fail),
                                     trials=trials, control=control)


def test_no_success_count_can_exceed_its_own_trial_count():
    """The bug: an errored call still satisfied `retrieved == needs_retrieve` for a
    precision question, so it scored +1 while being removed from the denominator."""
    for n_fail in (0, 1, 3, 5, 12):
        for row in _live_rows(n_fail):
            for form in ("retrieve", "answer", "handle", "control"):
                ok = row.get(f"{form}_ok")
                if ok is None:
                    continue
                # Mirror `_form_stats`' own lookup: a per-form `<form>_trials` when the row
                # carries one, else the shared `trials`. The treatment arm deliberately
                # carries none, so it divides by `trials` — checking a key that is absent
                # by design would test the test, not the invariant.
                t = row.get(f"{form}_trials", row["trials"])
                assert 0 <= ok <= t, f"{form}: {ok} successes over {t} trials (n_fail={n_fail})"


def test_the_report_renders_rather_than_crashing_when_calls_fail():
    """End-to-end guard on the same invariant: `_form_stats` raises ValueError on k>t."""
    rows = _live_rows(n_fail=4)
    build_dropeval_report({"m": rows})  # must not raise


def test_a_treatment_error_stays_IN_the_recall_denominator_and_is_scored_a_miss():
    """The regression this test exists for: adding `retrieve_trials = trials - errors`
    removed errored trials from the recall/precision/handle denominators, turning a 33%
    recall FAIL into a 100% PASS at an 11% error rate — under the INCONCLUSIVE gate, so
    nothing withheld it.

    Direction is what decides this. gap = treatment - control, so scoring a treatment
    error as a MISS pushes the gap negative (the drop looks worse — conservative), while
    excluding it flatters the treatment. final-accuracy is protected by `paired_rows`;
    recall/precision/handle are NOT, so they must keep every trial in the denominator.
    `openai_tool_answerer`'s own comment says so."""
    rows = _live_rows(n_fail=4, trials=3)
    assert rows
    for r in rows:
        # The treatment arm must NOT carry its own denominator...
        for form in ("retrieve", "answer", "handle"):
            assert f"{form}_trials" not in r, (
                f"{form}_trials removes errored trials from the denominator — the "
                f"documented dangerous direction")
        # ...so _form_stats divides by the shared, full trial count.
        assert r["trials"] == 3
    # ...while the CONTROL does, because deflating the control would flatter the drop.
    assert all("control_trials" in r for r in rows)


def test_a_degraded_recall_arm_reports_a_low_number_not_a_perfect_one():
    """End-to-end form of the same regression: two of three trials return nothing, the
    surviving one is correct. That is 33% recall, not 100%."""
    rows = _live_rows(n_fail=8, trials=3)
    recall = [r for r in rows if r["kind"] == "recall"]
    assert recall, "fixture must produce recall rows"
    acc = sum(r["retrieve_ok"] for r in recall) / sum(r["trials"] for r in recall)
    assert acc < 1.0, "a model that failed most of its calls must not read as 100% recall"


def test_the_control_denominator_EXCLUDES_the_controls_own_failures():
    """The mirror of the test above, and the dangerous direction. gap = treatment -
    control, so scoring a failed CONTROL call as a miss deflates the control, shrinks the
    gap, and flatters the drop — #268 on the control side, with no k>t crash to expose it.
    The control therefore carries its own denominator where the treatment does not."""
    rows = _live_rows(n_fail=6, trials=3)
    assert rows
    lost = [r for r in rows if r["control_errors"] > 0]
    assert lost, "fixture must lose at least one control call or it tests nothing"
    for r in lost:
        assert r["control_trials"] == r["trials"] - r["control_errors"], (
            "control_trials must remove the control's own failures, else a failed control "
            "call is scored as a wrong answer and the gap flatters the drop")


def test_errors_are_recorded_PER_ARM_not_only_as_a_total():
    """A collapsed count cannot answer #299's question — which arm lost the calls. A real
    run's 12/48 failures were attributed to arm-correlated attrition with no evidence
    either way, because the split was not recorded."""
    rows = _live_rows(n_fail=3, trials=3)
    assert rows
    for r in rows:
        assert "treatment_errors" in r and "control_errors" in r
        assert r["errors"] == r["treatment_errors"] + r["control_errors"]
    assert sum(r["treatment_errors"] for r in rows) > 0


def test_treatment_and_control_errors_are_attributed_to_the_right_arm():
    """The test above only checks the SUM equals `errors` — a swap of the two fields at
    the assignment site (`row["treatment_errors"] = errors` / `row["control_errors"] =
    control_errors`) would satisfy it identically and survived review (finding 6 on #300).
    `n_fail=1` fails only the very first call the harness makes — the treatment's first
    turn of the first question — so the ground truth here is unambiguous: exactly one
    treatment error, zero control errors."""
    rows = _live_rows(n_fail=1, trials=1)
    assert rows
    assert sum(r["treatment_errors"] for r in rows) == 1
    assert sum(r["control_errors"] for r in rows) == 0


def test_a_degraded_run_is_inconclusive_by_default_and_renderable_on_request():
    """The operator may know the cause is model-independent (a gateway restart). That is a
    claim the harness cannot verify, so it is RECORDED in the verdict rather than silently
    honoured — and the arm split is the evidence that decides whether to believe it."""
    rows = _live_rows(n_fail=40, trials=3)
    default = build_dropeval_report({"m": rows})
    assert "INCONCLUSIVE" in default

    accepted = build_dropeval_report({"m": rows}, accept_degraded=True)
    assert "Degraded run accepted" in accepted
    # It must not quietly become a normal-looking verdict.
    assert "valid ONLY if the failures were independent" in accepted


def test_the_report_says_how_many_questions_survived_the_pairing():
    """The failure RATE cannot distinguish a 50%-loss run with 15 usable questions from
    one with 2. The surviving count can, and it is what the gap is computed over.

    Built synthetically rather than via `_live_rows`: that fixture's dummy answerer
    always replies "whatever", which never matches a real expected answer and so scores
    the control at a uniform 0% — exactly the degenerate control
    `test_an_all_zero_control_is_excluded_as_broken_not_scored_as_a_free_pass` now
    excludes (review finding 3 on #300), which would make the accuracy gate excluded here
    too and this message never print."""
    # 48 rows, half of them damaged: the same 50% loss the docstring is about, over a
    # surviving half that clears `_MIN_PAIRED_QUESTIONS` (#334) so the model is scored and
    # the line under test actually renders.
    rows = _both_kinds(n=24, answer=1, control=1, trials=3, errors=0)
    for r in rows[:24]:
        r["control_trials"], r["control_errors"], r["errors"] = 2, 1, 1
    report = build_dropeval_report({"m": rows}, accept_degraded=True)
    assert "Questions surviving the pairing" in report


def test_the_surviving_count_uses_the_paired_subset_not_the_full_row_count():
    """The test above only checks the LABEL appears, not that the two numbers in "N/M" are
    the right ones — `len(g.rows)/len(rows)` swapped for `len(rows)/len(rows)` (mutation:
    reading the raw count for the numerator too) would still print a "Questions surviving"
    line and survived review (finding 6 on #300). One row is forced to miss a control
    trial (`control_trials=2` of `trials=3`), which drops it from `paired_rows` while it
    still counts toward the raw `len(rows)` — so the paired subset (47) is strictly smaller
    than the raw count (48), and only the correct denominator prints "47/48"."""
    rows = _both_kinds(n=24, answer=1, control=1, trials=3, errors=0)
    rows[0] = dict(rows[0], control_trials=2, control_errors=1, errors=1)
    report = build_dropeval_report({"m": rows})
    assert "47/48" in report


def test_the_report_names_which_arm_lost_the_calls():
    rows = _live_rows(n_fail=4, trials=3)
    report = build_dropeval_report({"m": rows})
    assert "Where they failed" in report
    assert "treatment" in report and "control" in report


def test_the_where_they_failed_line_binds_the_counts_to_the_right_arm():
    """The test above only checks that the words "treatment" and "control" appear
    somewhere in the header, which is true regardless of which number sits next to which
    word — swapping `t`/`c` in the f-string survived review (finding 6 on #300). A row
    with a deliberately asymmetric split (5 treatment errors, 1 control error) makes the
    two numbers distinguishable."""
    rows = _both_kinds(n=5, answer=1, control=1, trials=3, errors=0)
    rows[0] = dict(rows[0], errors=5, treatment_errors=5, control_errors=1)
    report = build_dropeval_report({"m": rows})
    assert "treatment 5 / control 1" in report
    assert "treatment 1 / control 5" not in report


def test_the_failed_calls_column_uses_the_attempts_field_not_bare_trials():
    """`attempts` counts calls across BOTH arms when a control ran (double the trial
    count); falling back to `trials` alone would understate it by half and survived
    review (finding 6 on #300): every prior fixture happened to have `attempts ==
    trials` or never checked the column's value at all."""
    rows = _both_kinds(n=1, answer=1, control=1, trials=4, errors=0)
    report = build_dropeval_report({"m": rows})
    # 2 rows (recall + precision) x (trials=4, doubled for the control) = 16 attempts.
    assert "0/16" in report


def test_the_control_arm_still_runs_every_trial_when_the_treatment_errors():
    """Skipping the control after a treatment failure would leave `control_trials`
    counting attempts it never made — the same bug one arm over."""
    rows = _live_rows(n_fail=3, trials=3)
    assert rows, "fixture must generate questions"
    # The first question's treatment calls all failed, yet its control denominator is
    # intact: the arms are independent measurements.
    assert any(r["control_trials"] > 0 for r in rows)


# --------------------------------------------------------------------------- #
# Neither arm may score a failed call as a wrong answer (#268 on this path).
# --------------------------------------------------------------------------- #


def test_a_question_whose_call_failed_is_dropped_from_BOTH_arms_not_scored_wrong():
    """#269's reproduction showed failed-call count rank-ordering final-accuracy exactly
    (2 fails -> 54%, 0 -> 88%) — the arm's own errors were being counted as misses.

    The fix is `paired_rows`' rule (#280), which is stricter than re-basing the denominator:
    a question that did not complete every trial on BOTH arms is excluded outright, because
    the row counts say how many trials survived, not WHICH."""
    incomplete = _rows(n=20, answer=0, control=4, trials=4, errors=2)
    complete = [dict(r, qid=f"ok{i}") for i, r in
                enumerate(_rows(n=20, answer=4, control=4, trials=4))]
    g = _accuracy_gate(incomplete + complete)
    assert [r["qid"] for r in g.rows] == [f"ok{i}" for i in range(20)], (
        "incomplete rows must not be scored")
    # Scored over the surviving pair only: both arms right -> no gap, and crucially the
    # errored rows did NOT drag the treatment arm to 0%.
    assert g.form_acc == 1.0 and g.control_acc == 1.0


def test_attempts_counts_both_arms_so_the_inconclusive_ratio_is_not_doubled():
    from terse.report import inconclusive_models
    # 1 error out of 2 real calls per trial: under the half-of-all-calls threshold, so the
    # run stands. Dividing by one arm's trials would read 100% failed and withhold it.
    rows = _rows(n=4, answer=2, control=2, trials=2, errors=1)
    # 4 failures across 16 real calls (2 arms x 2 trials x 4 questions) = 25%, under the
    # half-of-all-calls threshold. Dividing by one arm's trials would read 50% and withhold.
    assert inconclusive_models({"m": rows}) == {}
