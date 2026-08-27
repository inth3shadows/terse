"""dropeval's fixed-ideal metrics disclose their sample size instead of hiding it (#335).

`retrieve-recall` and `no-overfetch` are gated against a fixed 100% ideal, correctly: a
tool call either happened or it did not, so 100% IS the target and there is no second arm
to pair against. That is why they never went through `_gap` — and `_gap` is where every
evidence gate in this module lives, so they inherited none. One question at 100% published

    (gap +0% ±0 pts). **PASS** at 5% tolerance

which is the strongest thing this report can say, off a single observation.

WHY DISCLOSURE AND NOT A FLOOR. Both #334 (`_MIN_PAIRED_QUESTIONS`) and #295
(`_CODEC_MIN_TRIALS`) answered the same question with a Clopper-Pearson floor, and copying
that here was the obvious move. It was measured and rejected: `gen_drop_questions` emits
exactly ONE recall and ONE precision question per drop-marked payload, so a
20-question floor needs 20 drop-marked payloads — and on a live 1,524-payload capture
corpus across 39 tools, **zero** payloads had a drop rule selected at all (2026-08-26).
This metric has never run on real data. Any threshold picked today is calibrated against a
distribution that does not exist, and a fabricated derivation is worse than an admitted
convention.

So the number and its sample size are both published; what a thin sample cannot buy is the
word PASS. Calibrate a real floor once a live policy configures a drop rule (#271, #273).
"""
from __future__ import annotations

from terse.report import (
    _FIXED_IDEAL_MIN_QUESTIONS,
    _fixed_ideal_gate,
    build_dropeval_report,
    dropeval_gap_rows,
)


def _rows(n: int, *, retrieve_ok: int = 3, trials: int = 3) -> list[dict]:
    """`n` recall + `n` precision questions. Recall's success count is the lever."""
    return [{"qid": f"q{i}", "kind": kind, "trials": trials,
             "retrieve_ok": retrieve_ok if kind == "recall" else trials,
             "answer_ok": trials, "handle_ok": trials,
             "errors": 0, "treatment_errors": 0, "control_errors": 0,
             "attempts": trials}
            for i in range(n) for kind in ("recall", "precision")]


def _recall_line(md: str) -> str:
    return next(line for line in md.splitlines()
                if "Worst-case" in line and "retrieve-recall" in line)


def test_the_fixed_ideal_disclosure_threshold_is_five():
    """Pinned to its VALUE, because nothing else can pin it (#337).

    It is a convention, not a derivation — there is no Clopper-Pearson bound to recompute
    against it the way `test_the_quoted_clopper_pearson_bound_still_computes` does for
    `_CODEC_MIN_TRIALS`. That makes the literal the only guard available, and it is why
    the constant's comment says at length what it is not."""
    assert _FIXED_IDEAL_MIN_QUESTIONS == 5, (
        f"the disclosure threshold is {_FIXED_IDEAL_MIN_QUESTIONS}, not 5. Below it a "
        f"fixed-ideal metric prints INSUFFICIENT rather than PASS; changing it changes "
        f"what a dropeval run is allowed to claim")


def test_one_question_at_a_hundred_percent_does_not_publish_a_PASS():
    """The #335 report, verbatim. This is the line that used to read `**PASS**`."""
    line = _recall_line(build_dropeval_report({"m": _rows(1)}))
    assert "n=1 question" in line, line
    assert "**INSUFFICIENT**" in line, line
    assert "**PASS**" not in line, line


def test_four_questions_still_cannot_publish_a_PASS_and_five_can():
    """The boundary, with literal counts on both sides — not `_FIXED_IDEAL_MIN_QUESTIONS`
    and `... - 1`, which would slide with the constant and pin nothing (#337)."""
    assert "**INSUFFICIENT**" in _recall_line(build_dropeval_report({"m": _rows(4)}))
    five = _recall_line(build_dropeval_report({"m": _rows(5)}))
    assert "**PASS**" in five, five
    assert "n=5 questions" in five, five


def test_a_demonstrated_failure_publishes_its_FAIL_at_any_sample_size():
    """The asymmetry, and the reason this is a disclosure rather than an exclusion.

    `_MIN_PAIRED_QUESTIONS`' comment argues it at length: an evidence gate may only ever
    withhold a metric that is NOT behind its ideal, because an exclusion drops a model from
    the verdict entirely and #332 measured that turning a demonstrated -100% regression
    into "safe to enable". A downgrade that also silenced FAILs would reintroduce it."""
    line = _recall_line(build_dropeval_report({"m": _rows(1, retrieve_ok=0)}))
    assert "**FAIL**" in line, line
    assert "-100%" in line, line
    assert "**INSUFFICIENT**" not in line, line


def test_the_sample_size_is_disclosed_even_when_it_is_ample():
    """`n=` is not an error badge — it is on every fixed-ideal line, so a reader never has
    to infer that its absence means "enough". `final-accuracy` carries no `n=`: it has a
    real paired floor (#334) and its own exclusion prose."""
    md = build_dropeval_report({"m": _rows(20)})
    assert "n=20 questions" in _recall_line(md)
    assert "n=20 questions" in next(
        line for line in md.splitlines()
        if "Worst-case" in line and "no-overfetch" in line)


def test_a_metric_with_no_rows_is_absent_rather_than_a_zero_percent_bar():
    """`_fixed_ideal_gate` reports `"empty"` instead of publishing `(0.0, 0.0, 1.0, 0.0)`.

    The old inline construction emitted that tuple unconditionally, so a run carrying only
    recall questions drew a no-overfetch bar at 0% — a metric that never ran, rendered as
    a total failure of it."""
    assert _fixed_ideal_gate([], "retrieve_ok").excluded == "empty"
    recall_only = [r for r in _rows(6) if r["kind"] == "recall"]
    gaps, excluded = dropeval_gap_rows({"m": recall_only})
    assert "precision" not in gaps["m"]
    assert excluded["precision"] == {"m": "empty"}
    assert "recall" in gaps["m"]


def test_a_withheld_mechanism_metric_does_not_silence_the_final_accuracy_verdict():
    """final-accuracy decides "safe to enable"; the mechanism metrics do not.

    Its verdict line used to be nested inside `if recall_worst and precision_worst:`,
    unreachable while both fixed-ideal gates were always populated. `"empty"` makes it
    reachable, and a dropeval report whose headline verdict is simply absent reads as
    "nothing was wrong" — the silent-withholding defect this family keeps re-fixing."""
    rows = [dict(r, control_ok=r["trials"], control_trials=r["trials"])
            for r in _rows(20) if r["kind"] == "recall"]
    md = build_dropeval_report({"m": rows})
    # The precondition, asserted directly. An earlier spelling was
    #   `"no-overfetch" not in md or "Worst-case" not in md.split("no-overfetch")[0][-200:]`
    # which can never be False: `split` cuts at the FIRST occurrence, the report preamble
    # ~30 lines above any verdict, so `[0][-200:]` is always prose. Review found it.
    assert not any("Worst-case" in line and "no-overfetch" in line
                   for line in md.splitlines()), "no-overfetch should not have been scored"
    assert any("final-accuracy" in line and "Worst-case" in line
               for line in md.splitlines()), (
        f"the final-accuracy verdict vanished when no-overfetch had no rows:\n{md}")


# --------------------------------------------------------------------------------------
# Regressions from the #335 review. Every one of these was live in the first cut, and the
# first three are the defect the fix itself introduced or failed to remove.
# --------------------------------------------------------------------------------------

def _mixed(n_recall: int, n_precision: int, *, recall_ok: int = 3, trials: int = 3,
           control: bool = True) -> list[dict]:
    out = []
    for kind, count in (("recall", n_recall), ("precision", n_precision)):
        for i in range(count):
            r = {"qid": f"{kind}{i}", "kind": kind, "trials": trials,
                 "retrieve_ok": recall_ok if kind == "recall" else trials,
                 "answer_ok": trials, "handle_ok": trials, "errors": 0,
                 "treatment_errors": 0, "control_errors": 0, "attempts": trials}
            if control:
                r |= {"control_ok": trials, "control_trials": trials}
            out.append(r)
    return out


def test_an_insufficient_metric_cannot_authorize_enabling_the_drop():
    """The badge is not the decision. The first cut downgraded only the rendered word.

    `_format_worst_case_line` never mutates `verdict.passed`, so the "safe to enable"
    summary four lines below still read a 2-question sample as a pass: the report printed
    `**INSUFFICIENT**` and then "safe to enable drop-to-retrieve". A cosmetic fix that
    leaves the authorization intact is worse than none — it reads as handled."""
    md = build_dropeval_report({"m": _mixed(2, 22)})
    assert "**INSUFFICIENT**" in md
    assert "safe to enable drop-to-retrieve" not in md, md.split("## Verdict", 1)[1]
    # ...and it must not claim a failure that did not happen. Both arms are at 100%.
    assert "misses tolerance" not in md
    assert "INSUFFICIENT for enabling" in md


def test_a_failing_metric_publishes_even_when_its_partner_has_no_rows():
    """25 questions at 0% recall published NOWHERE in the first cut.

    The two fixed-ideal lines were jointly gated on `if recall_worst and precision_worst:`,
    which was safe only while neither could be None. `_fixed_ideal_gate`'s `"empty"` broke
    that, so a run with no precision rows silently dropped a demonstrated -100% recall
    failure — strictly worse than the false PASS #335 was filed about."""
    md = build_dropeval_report({"m": _mixed(25, 0, recall_ok=0)})
    line = _recall_line(md)
    assert "**FAIL**" in line, line
    assert "-100%" in line, line
    assert "n=25 questions" in line, line


def test_a_model_the_gate_could_not_score_is_named_in_the_markdown():
    """`dropeval_gap_rows` recorded the reason for the chart; the markdown discarded it.

    A model dropped from the recall gate left no trace while the verdict still said "the
    worst model" as if every model had been considered — review finding 5 on #300, whose
    fix landed only on the chart path."""
    md = build_dropeval_report({"A": _mixed(20, 20), "B": _mixed(0, 20)})
    assert "not scored" in md, md.split("## Verdict", 1)[1]
    assert "B" in md.split("## Verdict", 1)[1].split("not scored")[1][:120]


def test_an_unscored_metric_is_n_a_in_the_table_not_zero_percent():
    """An excluded `ArmGap` carries `form_acc == 0.0`; printing it renders a metric that
    never ran as a total failure of it."""
    md = build_dropeval_report({"m": _mixed(20, 0)})
    row = next(line for line in md.splitlines() if line.startswith("| `m`"))
    assert "| n/a |" in row, row
    assert "| 0% ±0 |" not in row, row


def test_a_thin_model_cannot_hide_behind_a_gap_tie():
    """`n` is the fleet minimum, not the worst-gap model's own count.

    `_worst_case_gap` breaks ties by insertion order, and a 100% tie is the normal outcome
    for a fixed-ideal metric — so keying the disclosure on the winner let model `A` with 11
    questions publish `n=11 **PASS**` while `B`'s single observation went undisclosed."""
    md = build_dropeval_report({"A": _mixed(11, 11), "B": _mixed(1, 11)})
    line = _recall_line(md)
    assert "n=1 question" in line, line
    assert "**INSUFFICIENT**" in line, line


def test_the_chart_carries_the_same_disclosure_as_the_markdown():
    """`dropeval_gap_rows`' docstring promises the two "can never disagree".

    The first cut put the disclosure only in `_format_worst_case_line`, which the terminal
    chart never calls — so the markdown printed **INSUFFICIENT** for a one-question metric
    while the chart printed a green PASS beside it, on the surface a reader sees first."""
    from terse.terminal_report import build_terminal_dropeval_report
    md = build_dropeval_report({"m": _mixed(1, 1, control=False)})
    chart = build_terminal_dropeval_report({"m": _mixed(1, 1, control=False)}, color=False)
    assert "**INSUFFICIENT**" in md
    assert "too few to publish a PASS" in chart, chart


def test_a_failing_accuracy_is_not_reported_as_a_thin_sample():
    """The `INSUFFICIENT for enabling` branch must not swallow a demonstrated failure.

    Mechanism metrics thin but passing, final-accuracy at -100%: the run cannot ship
    because the drop HURT, not because the sample was small, and saying otherwise asserts
    a cause the run does not show (#332's defect, one report over)."""
    rows = [dict(r, answer_ok=0, control_ok=r["trials"], control_trials=r["trials"])
            for r in _mixed(2, 2)]
    md = build_dropeval_report({"m": rows})
    assert "INSUFFICIENT for enabling" not in md, md.split("## Verdict", 1)[1]
    assert "keep drop-to-retrieve off" in md


def test_a_measured_failure_outranks_a_missing_arm_in_the_SUMMARY():
    """Not just the per-metric line — the headline directive too.

    `test_a_failing_metric_publishes_even_when_its_partner_has_no_rows` asserts the
    `**FAIL**` line and passes straight over the summary, where the real defect lived: the
    "Not concluded" branch sat above the failure branch, so an ABSENT no-overfetch arm
    improved the headline from "keep drop-to-retrieve off" to "not supported either way".
    An exclusion improving a verdict is the one thing #332 established may never happen."""
    # The mirror case is built by hand, NOT `_mixed(0, 25, recall_ok=0)`: `recall_ok` only
    # touches recall rows, so that fixture has a PASSING no-overfetch arm and cannot
    # express the failure it claims to. A fixture unable to fail is this repo's most
    # frequent way of shipping a test that pins nothing.
    failing_precision = [dict(r, retrieve_ok=0) for r in _mixed(0, 25)]
    assert all(r["retrieve_ok"] == 0 for r in failing_precision), "fixture is not failing"
    for rows, what in (
            (_mixed(25, 0, recall_ok=0), "recall failing, no-overfetch absent"),
            (failing_precision, "no-overfetch failing, recall absent")):
        md = build_dropeval_report({"m": rows})
        verdict = md.split("## Verdict", 1)[1]
        assert "keep drop-to-retrieve off" in verdict, f"{what}:\n{verdict}"
        assert "not supported either way" not in verdict, f"{what}:\n{verdict}"


def test_a_failing_accuracy_outranks_a_missing_mechanism_arm():
    """final-accuracy owns the "safe to enable" decision, so its FAIL outranks hardest."""
    rows = [dict(r, answer_ok=0, control_ok=r["trials"], control_trials=r["trials"])
            for r in _mixed(25, 0)]
    verdict = build_dropeval_report({"m": rows}).split("## Verdict", 1)[1]
    assert "**FAIL**" in verdict
    assert "keep drop-to-retrieve off" in verdict, verdict
    assert "not supported either way" not in verdict, verdict


def test_insufficient_for_enabling_names_an_ungated_accuracy_and_its_real_remedy():
    """Thin mechanism metrics AND no control arm — two causes, both must survive.

    This branch took over a state that previously reached "Re-run with the no-drop control
    arm before enabling". Saying "every metric cleared tolerance" is false (final-accuracy
    cleared nothing; it was never gated), and "generate more questions" alone points an
    operator at work that can never make the run shippable — without a control, accuracy
    stays ungated at any n."""
    md = build_dropeval_report({"m": _mixed(2, 2, control=False)})
    verdict = md.split("## Verdict", 1)[1]
    assert "INSUFFICIENT for enabling" in verdict, verdict
    assert "every metric cleared tolerance" not in verdict, verdict
    assert "never gated" in verdict, verdict
    # The REMEDY sentence specifically, not the phrase "no-drop control arm" — that also
    # appears in the "final-accuracy: not gated" line above, so asserting it alone passed
    # even with the remedy hardcoded to the wrong branch. Caught by mutation.
    assert "more questions alone leaves final-accuracy ungated" in verdict, verdict


def test_insufficient_for_enabling_keeps_the_simple_remedy_when_accuracy_was_gated():
    """The mirror: a control DID run and accuracy passed, so questions are the only gap.

    ASYMMETRIC on purpose. `_mixed(2, 2)` cannot reach this branch: `_accuracy_gate` pairs
    over ALL rows and needs `_MIN_PAIRED_QUESTIONS`, so four rows leave accuracy
    underpowered and it lands in the other branch instead. 2 recall + 30 precision keeps
    recall thin (the mechanism gap) while giving accuracy the 20 paired rows it needs."""
    verdict = build_dropeval_report(
        {"m": _mixed(2, 30)}).split("## Verdict", 1)[1]
    assert "INSUFFICIENT for enabling" in verdict, verdict
    assert "every metric cleared tolerance" in verdict, verdict
    assert "more questions alone leaves final-accuracy ungated" not in verdict, verdict


def test_a_fleet_minimum_from_another_model_says_so():
    """`n` is the fleet minimum, but the sentence names the worst-GAP model.

    With `A` at 11 questions and `B` at 1, both 100%, the line reads `Worst-case model
    \\`A\\` ... n=1 question` — and a reader attributes B's count to A. It has to say which."""
    line = _recall_line(build_dropeval_report({"A": _mixed(11, 11), "B": _mixed(1, 11)}))
    assert "n=1 question for the thinnest model" in line, line
    # ...and when the minimum IS the named model's own count, no qualifier is added.
    solo = _recall_line(build_dropeval_report({"A": _mixed(2, 11)}))
    assert "n=2 questions)" in solo, solo
    assert "thinnest model" not in solo, solo


def test_the_chart_names_a_metric_no_model_could_score():
    """A metric where EVERY model is excluded vanished from the chart entirely.

    The per-metric note ran only after `if not plot_rows: continue`, so it fired only when
    some OTHER model still drew a bar for that metric. With no recall questions at all, the
    chart rendered an all-green run for a mechanism nobody measured — while the markdown
    said `**retrieve-recall: not scored**` and `**Not concluded**`. The chart is the
    surface a reader sees first."""
    from terse.terminal_report import build_terminal_dropeval_report
    rows = _mixed(0, 20)
    chart = build_terminal_dropeval_report({"m": rows}, color=False)
    assert "retrieve-recall" in chart, chart
    assert "no rows" in chart, chart
    assert "**Not concluded**" in build_dropeval_report({"m": rows})


def test_an_unscorable_run_says_no_data_rather_than_nothing():
    """Every model now gets a dict from `dropeval_gap_rows`, empty when nothing scored, so
    a truthy `gaps` stopped meaning "something was measured" and the chart returned ""."""
    from terse.terminal_report import build_terminal_dropeval_report
    nothing = [{"qid": "q0", "kind": "other", "trials": 3, "retrieve_ok": 3,
                "answer_ok": 3, "handle_ok": 3, "errors": 0, "treatment_errors": 0,
                "control_errors": 0, "attempts": 3}]
    assert "no data" in build_terminal_dropeval_report({"m": nothing}, color=False)
