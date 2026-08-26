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
    assert "no-overfetch" not in md or "Worst-case" not in md.split("no-overfetch")[0][-200:]
    assert any("final-accuracy" in line and "Worst-case" in line
               for line in md.splitlines()), (
        f"the final-accuracy verdict vanished when no-overfetch had no rows:\n{md}")
