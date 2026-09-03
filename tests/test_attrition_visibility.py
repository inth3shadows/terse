"""`paired_rows` excludes silently, and #299 is that the silence hides a BIAS.

Pairing makes the surviving arms comparable. It does not make the SELECTION ignorable:
which questions survive is decided by the arm most likely to fail, and that arm is not
random — dropeval's treatment runs two turns to the control's one, and fluency's longest
prompts are also its hardest questions. A run that loses its five hardest questions from
one arm and none from the other still produces a perfectly paired comparison over the
twenty easy ones, reports a tiny gap with a tight interval, and is wrong.

Nothing here fixes that. #299's own "suggested first step" is Option 1 — make the
attrition VISIBLE, per arm and per question kind, then look at a real run before
designing anything further, because no run in the repo can currently say whether the bias
is large or negligible.

WHY THE FIXTURES CARRY `attempts` AND `<arm>_trials`. `paired_rows` never excludes a row
that lacks `attempts` (legacy files, #91 hand-built packs), so the obvious fixture — the
`_rows` helper in `test_fluency.py`, booleans and no counters — takes that escape and
CANNOT be excluded by anything. Every fixture below is guarded by an assertion that the
pairing really did drop rows, so a later edit cannot re-base these tests onto a row shape
that tests nothing.
"""
from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Mapping
from pathlib import Path

from terse import codeceval, dropeval
from terse import policy as policy_mod
from terse.html_report import build_html_diff_report
from terse.report import (
    ATTRITION_HEADING,
    ATTRITION_NOTE,
    DIFF_ARMS,
    DROPEVAL_ATTRITION_NOTE,
    FLUENCY_CONTROL,
    FLUENCY_GATING,
    NO_CONTROL_ATTRITION_NOTE,
    attrition,
    attrition_block,
    attrition_line,
    build_diff_report,
    build_diff_soak_report,
    build_dropeval_report,
    build_fluency_report,
    build_text_diff_report,
    is_diff_run,
    paired_rows,
    strip_markup,
)
from terse.terminal_report import (
    build_terminal_diff_report,
    build_terminal_dropeval_report,
    build_terminal_fluency_report,
)

GATED = ("terse_ok", "primer_ok", "raw_ok")
TRIALS = 3


def _frow(qid: str, qtype: str, *, lost: tuple[str, ...] = ()) -> dict:
    """One live-shaped fluency row; `lost` names the arms that dropped a call of it."""
    row = {"tool": "t", "sha": "s", "qid": qid, "qtype": qtype, "transform": "table",
           "trials": TRIALS, "fails": len(lost), "attempts": TRIALS * 4}
    for arm in ("raw", "terse", "primer", "inline"):
        n = TRIALS - (1 if f"{arm}_ok" in lost else 0)
        row[f"{arm}_ok"] = n
        row[f"{arm}_trials"] = n
    return row


def _clean(n: int, qtype: str = "count") -> list[dict]:
    return [_frow(f"c{i}", qtype) for i in range(n)]


# --------------------------------------------------------------------------- #
# The helper agrees with the pairing it describes.
# --------------------------------------------------------------------------- #


def test_attrition_excludes_exactly_what_pairing_drops():
    """The whole value of this report is that it describes the REAL paired subset. A
    second implementation of the exclusion rule — the escapes for an absent `attempts`,
    an absent `<arm>_trials`, and the run-level `collected` fact — would be free to
    disagree with `paired_rows`, and a report that contradicts the pairing it annotates
    is worse than no report. Hence `_paired_partition`, one body, two callers."""
    cases = [
        _clean(6),
        _clean(6) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)],
        _clean(3) + [_frow("x", "lookup", lost=("raw_ok", "primer_ok"))],
        # No counters at all: `paired_rows`' legacy escape. Excludes nothing.
        [{"qid": f"q{i}", "qtype": "count", "raw_ok": 1, "terse_ok": 0, "primer_ok": 0}
         for i in range(4)],
    ]
    assert any(attrition(rows, *GATED).excluded for rows in cases), (
        "no case drops a row — the fixture would pass against a hard-coded 0")
    for rows in cases:
        a = attrition(rows, *GATED)
        assert a.total == len(rows)
        assert a.excluded == len(rows) - len(paired_rows(rows, *GATED))


def test_every_excluded_row_is_attributed_to_at_least_one_arm():
    """`attrition_line` renders `by_arm` with no fallback string, on the ground that a
    dropped row always has a losing arm — `_paired_partition` drops only when some arm's
    `_paired_arm` is False, and `attrition` asks the same question of the same arms.
    That is an invariant, so it is asserted rather than guarded by a branch no run can
    reach."""
    rows = _clean(4) + [_frow("d", "deref", lost=("terse_ok",)),
                        _frow("e", "deref", lost=("raw_ok",))]
    a = attrition(rows, *GATED)
    assert a.excluded == 2
    assert a.by_arm and sum(a.by_arm.values()) >= a.excluded


def test_attrition_names_the_arm_that_actually_lost_the_question():
    """A swap of the attribution would satisfy a count-only assertion identically — the
    same class of defect as #300's finding 6 on `treatment_errors`/`control_errors`. Only
    the terse arm loses here, so the ground truth is unambiguous."""
    rows = _clean(5) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(3)]
    a = attrition(rows, *GATED)
    assert a.by_arm == {"terse_ok": 3}


def test_an_arm_loss_on_a_row_the_pairing_KEPT_is_not_counted_as_attrition():
    """`by_arm` answers "who removed these questions", not "who lost a call somewhere".

    The two come apart exactly at `paired_rows`' escapes. A row predating #263 carries no
    `attempts`, so the pairing keeps it no matter what its per-arm counters say — and
    attributing over every row rather than the dropped ones would then report a loss for
    a question that is sitting in the paired exam, inflating `by_arm` past `excluded` for
    a reason the note ascribes to multi-arm loss. Found by mutation: attributing over
    `rows` instead of `dropped` survived the first version of this file, because every
    fixture in it carried `attempts` and the two quantities could not disagree."""
    legacy = _frow("legacy", "count", lost=("terse_ok",))
    del legacy["attempts"]
    rows = _clean(3) + [legacy, _frow("d", "deref", lost=("terse_ok",))]
    assert legacy in paired_rows(rows, *GATED), "fixture: the escape must keep this row"
    a = attrition(rows, *GATED)
    assert a.excluded == 1
    assert a.by_arm == {"terse_ok": 1}


def test_per_arm_counts_can_exceed_the_excluded_total():
    """`ATTRITION_NOTE` tells the reader so, which makes it a claim this file has to
    pin: `by_arm` is a per-arm attribution, not a partition of `excluded`."""
    rows = _clean(5) + [_frow("d", "deref", lost=("terse_ok", "raw_ok"))]
    a = attrition(rows, *GATED)
    assert a.excluded == 1
    assert a.by_arm == {"terse_ok": 1, "raw_ok": 1}
    assert sum(a.by_arm.values()) > a.excluded
    assert "can exceed the excluded total" in ATTRITION_NOTE


def test_by_kind_carries_its_denominator_and_leads_with_the_concentration():
    """The whole signal. "deref 5/5" beside "count 0/6" reads instantly; a bare "deref
    5" does not say whether that is all of them or a fifth of them, and a corpus with a
    dozen kinds would bury the finding if the list were alphabetical."""
    rows = _clean(6) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]
    a = attrition(rows, *GATED)
    assert a.by_kind == {"count": (0, 6), "deref": (5, 5)}
    line = attrition_line("m", a)
    assert "deref 5/5" in line and "count 0/6" in line
    assert line.index("deref 5/5") < line.index("count 0/6"), "worst kind must lead"


def test_a_row_with_no_kind_field_is_still_counted():
    """Filing it under `?` rather than dropping it: an unlabelled question still left the
    paired subset, and omitting it would understate the total this exists to publish."""
    rows = _clean(3) + [dict(_frow("d", "deref", lost=("terse_ok",)), qtype=None)]
    del rows[-1]["qtype"]
    a = attrition(rows, *GATED)
    assert a.excluded == 1 and a.by_kind["?"] == (1, 1)


def test_a_clean_run_renders_no_attrition_line():
    """A report does not grow a section that always says zero."""
    assert attrition_line("m", attrition(_clean(20), *GATED)) == ""


# --------------------------------------------------------------------------- #
# It reaches the reader.
# --------------------------------------------------------------------------- #


def test_the_fluency_report_publishes_the_attrition_of_its_paired_exam():
    rows = _clean(25) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]
    assert len(paired_rows(rows, *GATED)) == 25, "fixture must actually lose 5 questions"
    report = build_fluency_report({"m": rows}, [])
    assert "Attrition of the paired exam" in report
    assert "excluded 5/30 question(s)" in report
    assert "terse_ok 5" in report
    assert "deref 5/5" in report and "count 0/25" in report


def test_the_fluency_report_stays_quiet_when_nothing_was_excluded():
    assert "Attrition of the paired exam" not in build_fluency_report(
        {"m": _clean(25)}, [])


def test_the_inline_arm_is_outside_this_report_because_it_is_outside_the_pairing():
    """`inline_ok` is display-only — `_gap` passes it as a `display` arm precisely because
    it carries the longest prompt and so truncates first while gating nothing. Counting
    its losses here would report an exclusion the pairing never performed. That the inline
    arm has no protection at all is #292, a different issue, and folding it in would be
    scope drift dressed as thoroughness."""
    rows = _clean(20) + [_frow(f"i{i}", "deref", lost=("inline_ok",)) for i in range(5)]
    assert len(paired_rows(rows, *GATED)) == 25, "inline loss must not unpair a row"
    assert "Attrition of the paired exam" not in build_fluency_report({"m": rows}, [])


def test_a_withheld_model_still_publishes_its_attrition():
    """Its numbers are unpublishable; its attrition is the EVIDENCE for why, and is the
    one thing a reader can act on. Losing almost every question renders `n/a` in the
    table — the attrition line must survive that."""
    rows = _clean(2) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(18)]
    report = build_fluency_report({"m": rows}, [])
    assert "| `m` | 20 | n/a |" in report, "fixture must be a withheld model"
    assert "deref 18/18" in report


class _ErroringAnswerer:
    """Returns nothing for the first `n` calls — an unanswered call, not a wrong one."""

    def __init__(self, n: int) -> None:
        self.left = n

    def __call__(self, system: str, user: str) -> str:
        if self.left > 0:
            self.left -= 1
            return ""
        return "E" * 300


def _live_drop_rows(n_fail: int, trials: int = 3) -> list[dict]:
    obj = {"rows": [{"id": i, "evidence": f"{i}" + "E" * 300} for i in range(4)]}
    rule = policy_mod.Rule(tool_glob="t", tiers=("minify", "table"),
                           fields={"rows[].evidence":
                                   {"lossy": "drop-to-retrieve", "min": 10}})
    return dropeval.run_drop_payload(obj, "", rule, "t", _ErroringAnswerer(n_fail),
                                     trials=trials, control=True)


def _drow(qid: str, *, t_err: int, c_err: int, trials: int = 3) -> dict:
    """A dropeval row in the shape `run_drop_payload` emits — key set pinned by
    `test_the_hand_built_dropeval_row_matches_the_emitter`."""
    return {"qid": qid, "kind": "recall", "trials": trials,
            "retrieve_ok": trials, "answer_ok": 0, "handle_ok": trials,
            "errors": t_err + c_err, "treatment_errors": t_err, "control_errors": c_err,
            "attempts": trials * 2, "control_ok": trials - c_err,
            "control_trials": trials - c_err}


def test_the_hand_built_dropeval_row_matches_the_emitter():
    """A fixture that has drifted from the emitter tests the fixture. The two rows below
    are built by different code and must carry the same keys."""
    assert set(_drow("q", t_err=1, c_err=1)) == set(_live_drop_rows(n_fail=3)[0])


def test_a_treatment_that_lost_everything_is_not_reported_as_an_unselected_exam():
    """THE motivating case of #299, and the first cut of this change rendered NOTHING for
    it. `run_drop_payload` deliberately emits no `answer_trials` — a failed treatment call
    is scored a MISS, not excluded, because excluding it turned a 33% recall FAIL into a
    100% PASS (#300) — so `_paired_arm` is unconditionally True for `answer_ok` and the
    pairing CANNOT exclude on the treatment side. The two-turn treatment thinning out
    under a token-budget stop while the one-turn control survives therefore produced
    `excluded 0/N` and a silent report, which reads as "the paired exam is unselected":
    the exact opposite of the truth."""
    rows = [_drow(f"q{i}", t_err=3, c_err=0) for i in range(4)]
    assert attrition(rows, "answer_ok", "control_ok", kind_key="kind").excluded == 0
    report = build_dropeval_report({"m": rows})
    assert "Attrition of the paired exam" in report
    assert "treatment lost 12 call(s), scored as misses and never excluded" in report


def test_a_flat_two_arm_loss_is_not_rendered_as_control_side_concentration():
    """Both arms lost every call — the note's own "flat spread is what infrastructure
    failure looks like" case. `by_arm` can only ever say `control_ok`, so the generic note
    would steer the reader to the opposite conclusion (selection bias in the control's
    favour). The dropeval note states the asymmetry instead."""
    rows = [_drow(f"q{i}", t_err=3, c_err=3) for i in range(2)]
    a = attrition(rows, "answer_ok", "control_ok", kind_key="kind")
    assert a.by_arm == {"control_ok": 2}, "the structural fact this note exists to explain"
    report = build_dropeval_report({"m": rows})
    assert "treatment lost 6 call(s)" in report
    assert DROPEVAL_ATTRITION_NOTE in report
    assert ATTRITION_NOTE not in report, "the generic note is read backwards here"


def test_the_dropeval_note_names_the_reason_the_treatment_cannot_be_excluded():
    for claim in ("no per-arm denominator", "scored a MISS, not excluded",
                  "only ever exclude on the CONTROL side"):
        assert claim in DROPEVAL_ATTRITION_NOTE


def test_the_dropeval_report_publishes_the_attrition_of_a_real_run():
    """Live harness rows, not hand-built ones: dropeval's row shape is emitted in one
    place and this proves the report reads THAT shape. `kind` here is recall/precision,
    not fluency's `qtype` — a report that looked for the wrong key would silently file
    every question under `?` and still render."""
    rows = _live_drop_rows(n_fail=6)
    assert rows and len(paired_rows(rows, "answer_ok", "control_ok")) < len(rows), (
        "fixture must lose at least one question to the pairing")
    report = build_dropeval_report(rows and {"m": rows})
    assert "Attrition of the paired exam" in report
    kinds = {r["kind"] for r in rows}
    assert kinds <= {"recall", "precision"} and kinds
    assert any(f"{k} " in report.split("Attrition of the paired exam", 1)[1]
               for k in kinds)


# --------------------------------------------------------------------------- #
# Pins added after review — each one caught a decision that 383 tests ignored.
# --------------------------------------------------------------------------- #


def test_the_attrition_pairs_on_exactly_the_arms_the_gap_pairs_on():
    """A second, hardcoded copy of the arm list let the attrition drift from the pairing
    it annotates, and drift SILENCES it: with `raw_ok` dropped from the copy, a row set
    whose only loss is on `raw_ok` is excluded from the gap and reported as nothing at
    all. That is the silent exclusion this whole change exists to end, reintroduced by
    the change itself. Both now read `FLUENCY_GATING`/`FLUENCY_CONTROL`."""
    assert set(GATED) == set(FLUENCY_GATING) | {FLUENCY_CONTROL}
    rows = _clean(20) + [_frow(f"r{i}", "deref", lost=("raw_ok",)) for i in range(6)]
    report = build_fluency_report({"m": rows}, [])
    assert "excluded 6/26 question(s)" in report and "raw_ok 6" in report


def test_attrition_and_the_pairing_share_one_collected_arms_fact():
    """`by_arm` is non-empty whenever `excluded` is — but only while `attrition` and
    `_paired_partition` agree about which arms the run collected at all. Two copies of
    that computation could disagree only on a row set carrying a ZERO-attempt arm, which
    no earlier fixture had: replacing `attrition`'s copy with `set()` survived 383 tests.
    One `_collected_arms` now serves both, and this is the row shape that separates them.
    """
    rows = _clean(4)
    for r in rows[:2]:  # primer collected for SOME questions only -> a lost question
        r["primer_trials"] = r["primer_attempts"] = 0
    a = attrition(rows, *GATED)
    assert a.excluded == 2, "fixture must exercise the collected-arms rule"
    assert a.by_arm == {"primer_ok": 2}
    assert attrition_line("m", a), "an excluded row with no named arm renders a broken line"


def test_the_note_says_the_arm_it_NAMES_is_the_one_the_exclusion_flatters():
    """The direction, executed. The note inherited a sentence from dropeval's
    error-count line, where the listed quantity is misses and the SURVIVOR is flattered.
    On an exclusion the reverse holds, and the rendered list names the loser: terse loses
    every `deref`, so terse is scored over the easy questions only and its 83% true
    accuracy renders as 100%. A reader following the old sentence's antecedent concluded
    the bias ran the other way."""
    rows = _clean(25) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]
    unpaired_terse = sum(r["terse_ok"] for r in rows) / sum(r["trials"] for r in rows)
    assert unpaired_terse < 1.0, "fixture: terse is genuinely below raw over ALL questions"
    report = build_fluency_report({"m": rows}, [])
    assert "| 100% ±0 | 100% ±0" in report, "the paired exam hides the gap — the premise"
    assert "terse_ok 5" in report
    assert "The arm NAMED is the arm that LOST" in ATTRITION_NOTE
    assert "in the named arm's favour" in ATTRITION_NOTE


def test_the_excluded_set_is_checked_against_an_INDEPENDENT_recomputation():
    """`excluded == len(rows) - len(paired_rows(...))` is near-tautological: both sides
    call `_paired_partition`, so it reduces to `len(dropped) == len(rows) - len(kept)`
    and passes against a rule that keeps everything. This re-derives the expected set from
    the row fields directly, which is the check the other test's docstring claimed."""
    rows = _clean(6) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]
    expected = {r["qid"] for r in rows
                if any(r[f"{arm[:-3]}_trials"] < r["trials"] for arm in GATED)}
    assert expected, "fixture must expect a non-empty exclusion"
    kept = {r["qid"] for r in paired_rows(rows, *GATED)}
    assert {r["qid"] for r in rows} - kept == expected
    assert attrition(rows, *GATED).excluded == len(expected)


def test_the_fluency_report_carries_the_GENERIC_note_not_dropevals():
    """The two notes state opposite reading rules, because the two harnesses' arms are
    asymmetric in opposite ways — dropeval's treatment cannot be excluded at all, fluency's
    every gated arm can. Swapping them at the fluency site went unnoticed by 383 tests, so
    the report could have told a fluency reader that its exclusions were a property of the
    emitter rather than a selected exam."""
    rows = _clean(25) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]
    report = build_fluency_report({"m": rows}, [])
    assert ATTRITION_NOTE in report
    assert DROPEVAL_ATTRITION_NOTE not in report


# --------------------------------------------------------------------------- #
# Every renderer drawn over the paired subset discloses what left it.
# --------------------------------------------------------------------------- #


def _lossy_fluency_rows():
    return _clean(25) + [_frow(f"d{i}", "deref", lost=("terse_ok",)) for i in range(5)]


def test_the_terminal_fluency_chart_discloses_its_own_attrition():
    """The forest bars are drawn over the paired subset, so a chart without this says
    less than the markdown beside it — and the chart is the artifact people quote."""
    text = build_terminal_fluency_report({"m": _lossy_fluency_rows()}, color=False)
    assert "Attrition of the paired exam" in text
    assert "excluded 5/30" in text and "terse_ok 5" in text and "deref 5/5" in text


def test_the_terminal_dropeval_chart_discloses_its_own_attrition():
    rows = [_drow(f"q{i}", t_err=3, c_err=0) for i in range(4)]
    text = build_terminal_dropeval_report({"m": rows}, color=False,
                                          accept_degraded=True)
    assert "treatment lost 12 call(s)" in text
    assert "only ever exclude on the CONTROL side" in text


def test_the_html_page_discloses_the_attrition_of_ITS_OWN_pairing():
    """The HTML forest plot pairs `diff_ok` against `terse_ok` — not fluency's arms.
    Annotating a chart with the attrition of a different pairing is worse than annotating
    nothing, so the arms here must be the ones `arm_gap` was given on this page."""
    rows = [{"qid": f"c{i}", "qtype": "count", "trials": 3, "attempts": 6,
             "diff_ok": 3, "terse_ok": 3, "diff_trials": 3, "terse_trials": 3}
            for i in range(20)]
    rows += [{"qid": f"d{i}", "qtype": "deref", "trials": 3, "attempts": 6,
              "diff_ok": 2, "terse_ok": 3, "diff_trials": 2, "terse_trials": 3}
             for i in range(5)]
    html = build_html_diff_report({"m": rows})
    assert "Attrition of the paired exam" in html
    assert "diff_ok 5" in html and "deref 5/5" in html


def test_every_renderer_states_the_attrition_in_the_SAME_sentence():
    """Four renderers, one `attrition_block`. #335's review rounds are the record of what
    happens when two renderers decide the same thing separately — they disagreed three
    times — so the joining, the model separator and the note choice live in one function
    and this asserts all four carry its output."""
    rows = _lossy_fluency_rows()
    md = build_fluency_report({"m": rows}, [])
    term = build_terminal_fluency_report({"m": rows}, color=False)
    body = attrition_block({"m": attrition(rows, *FLUENCY_GATING, FLUENCY_CONTROL)},
                           ATTRITION_NOTE, style="markdown").strip()
    core = body.split("— ", 1)[1]
    assert core in md, "the markdown report carries the shared sentence verbatim"
    # The terminal carries the SAME sentence with the markup stripped — `**` and
    # backticks are noise in a chart, and `_plain` is applied to the shared directive one
    # function down for exactly that reason. Both spellings come from one string.
    assert strip_markup(core) in strip_markup(term)


def test_two_models_are_separated_by_something_other_than_the_clause_separator():
    """`; ` already separates the arm, kind and treatment clauses WITHIN one model's
    line, so reusing it between models left no way to see where one model ended except
    by spotting a backtick. Four levels of nesting on one delimiter."""
    a = attrition(_lossy_fluency_rows(), *FLUENCY_GATING, FLUENCY_CONTROL)
    text = attrition_block({"alpha": a, "beta": a}, ATTRITION_NOTE, style="markdown")
    assert "`alpha`" in text and "`beta`" in text
    assert " · `beta`" in text, "models must not be joined by the clause separator"


# --------------------------------------------------------------------------- #
# Round 3 of review: the diff family, and the heading that was a second literal.
# --------------------------------------------------------------------------- #


def _diff_rows(n_clean: int = 20, n_lost: int = 5) -> list[dict]:
    rows = [{"qid": f"c{i}", "qtype": "count", "trials": 3, "attempts": 6,
             "diff_ok": 3, "terse_ok": 3, "diff_trials": 3, "terse_trials": 3}
            for i in range(n_clean)]
    rows += [{"qid": f"d{i}", "qtype": "deref", "trials": 3, "attempts": 6,
              "diff_ok": 2, "terse_ok": 3, "diff_trials": 2, "terse_trials": 3}
             for i in range(n_lost)]
    return rows


def test_the_diff_family_discloses_its_attrition_in_EVERY_renderer_cli_prints():
    """`cli` prints the diff markdown and the terminal chart on every diff path and
    writes the HTML page only under `--html`. Wiring the HTML alone put the disclosure
    exactly where it was least read: an operator running `--diff --bars` saw a `PASS` over
    20 questions and was never told the diff arm removed all five `deref` questions.
    All three diff modes route through these two renderers."""
    rows = _diff_rows()
    for text in (build_diff_report({"m": rows}),
                 build_text_diff_report({"m": rows}),
                 build_terminal_diff_report({"m": rows}, color=False)):
        assert "Attrition of the paired exam" in text
        assert "excluded 5/25" in text and "diff_ok 5" in text and "deref 5/5" in text


def test_the_terminal_dropeval_chart_discloses_on_the_INCONCLUSIVE_path_too():
    """That early return fires on the runs with the MOST attrition — 12 of 24 calls lost
    is exactly when a reader needs to know which arm lost them — and it used to drop the
    disclosure while the markdown printed it. The first version of the sibling test only
    passed because it set `accept_degraded=True` and so never took this branch."""
    rows = [_drow(f"q{i}", t_err=3, c_err=0) for i in range(4)]
    text = build_terminal_dropeval_report({"m": rows}, color=False)
    assert "INCONCLUSIVE" in text, "fixture must take the early-return path"
    assert "treatment lost 12 call(s)" in text


def test_the_terminal_charts_strip_markdown_markup():
    """`_plain` exists because `**` and backticks are noise in a terminal, and the shared
    directive one function down is already routed through it. These blocks were not."""
    term = build_terminal_fluency_report({"m": _lossy_fluency_rows()}, color=False)
    block = term.split("Attrition of the paired exam", 1)[1]
    assert "`" not in block and "**" not in block


def test_the_heading_is_ONE_string_and_both_spellings_come_from_it():
    """The markdown sites used to bolt the bold on afterwards with a `.replace()` keyed on
    the heading text — a second copy of the literal 200 lines away. Editing the heading
    made that no-op and emitted an UNCLOSED `**` into both reports, and 226 tests stayed
    green. Now one string, two spellings, both produced here."""
    a = attrition(_lossy_fluency_rows(), *FLUENCY_GATING, FLUENCY_CONTROL)
    md = attrition_block({"m": a}, ATTRITION_NOTE, style="markdown")
    txt = attrition_block({"m": a}, ATTRITION_NOTE)
    htm = attrition_block({"m": a}, ATTRITION_NOTE, style="html")
    assert ATTRITION_HEADING == "**Attrition of the paired exam** (#299)", (
        "asserted as a literal: `md.startswith(f'> {ATTRITION_HEADING}')` reads the "
        "constant on both sides and passes for ANY heading, bold or not")
    assert md.startswith("> **Attrition of the paired exam** (#299) — ")
    assert strip_markup(ATTRITION_HEADING) in txt and "**" not in txt
    assert md.count("**") % 2 == 0, "an unclosed bold run"
    # `html` is a THIRD spelling, not an alias for `text`: the page renders backtick spans
    # as `<code>`, so stripping them left one paragraph in plain text beside a verdict
    # banner that still rendered them. Block markup and inline markup are separate.
    assert "`m`" in htm and "**" not in htm
    assert build_fluency_report({"m": _lossy_fluency_rows()}, []).count("**") % 2 == 0


def test_the_html_page_reads_the_loops_own_exclusion_rather_than_retesting_row_shape():
    """A second copy of `all("diff_ok" in r ...)`, 80 lines from the original, that
    nothing could catch disagreeing — deleting its guard left 226 tests green. The page
    now reads back the loop's verdict, so a non-diff run is excluded by one decision."""
    # `notdiff` carries a real terse-arm loss, so a renderer that stopped consulting the
    # loop's verdict WOULD print a line for it. A row set that could not produce one
    # either way tests nothing — the first version of this test used exactly that.
    notdiff = [{"qid": f"n{i}", "qtype": "count", "trials": 3, "attempts": 3,
                "terse_ok": 2, "terse_trials": 2} for i in range(4)]
    assert attrition(notdiff, *DIFF_ARMS).excluded == 4, "fixture must be renderable"
    html = build_html_diff_report({"m": _diff_rows(), "notdiff": notdiff})
    assert "diff_ok 5" in html
    assert "notdiff" not in html.split("Attrition of the paired exam", 1)[1]


# --------------------------------------------------------------------------- #
# Round 4 of review.
# --------------------------------------------------------------------------- #


def test_the_diff_soak_MARKDOWN_discloses_and_not_only_its_chart():
    """The soak's markdown is the artifact that gets KEPT — `cli` writes it to `--out`,
    and under `--bars` prints the terminal chart beside it. Wiring only
    `_build_diff_style_report` left that chart saying `deref 15/15` (a number now produced
    by `test_the_soaks_worked_example_is_produced_by_a_fixture_not_asserted_in_prose`, not quoted) while the file on disk said nothing; without `--bars` the soak was silent in both renderers, the same defect
    with no witness.
    `build_diff_soak_report` does NOT route through `_build_diff_style_report`."""
    rows = [dict(r, depth=1) for r in _diff_rows(20, 5)]
    md = build_diff_soak_report({"m": rows})
    chart = build_terminal_diff_report({"m": rows}, color=False)
    assert "Attrition of the paired exam" in chart, "the chart already spoke"
    assert "Attrition of the paired exam" in md
    assert "diff_ok 5" in md and "deref 5/5" in md


def test_the_html_page_keeps_its_code_spans():
    """`_esc_md` exists to render backtick spans as `<code>`, so handing it a string with
    the backticks already stripped left this paragraph in plain text beside a verdict
    banner two lines up that still rendered them. Block markup (the `> ` and the bold
    heading) and inline markup (backticks) are separate questions; one flag conflated
    them."""
    html = build_html_diff_report({"m": _diff_rows()})
    block = html.split("Attrition of the paired exam", 1)[1]
    assert "<code>m</code>" in block
    assert "`" not in block and "**" not in block


def test_all_three_diff_renderers_share_ONE_shape_predicate():
    """The HTML tested `diff_ok` AND `terse_ok`; the two sites added with the diff wiring
    tested only `diff_ok`. On a row set carrying `diff_ok` but not `terse_ok` the HTML
    withheld the model while the other two raised `KeyError: 'terse_ok'`. A weaker third
    copy is how "one decision, read back rather than recomputed" stays true only in the
    comment."""
    assert not is_diff_run([{"qid": "x", "diff_ok": 1}]), "missing terse_ok is not a diff run"
    assert not is_diff_run([]) and is_diff_run(_diff_rows())
    # Spelled ONCE: every diff-family site must reach the predicate through this function,
    # not re-write it. Grepped rather than executed because the markdown and terminal
    # renderers raise `KeyError: 'terse_ok'` on a mixed-shape row set BEFORE reaching
    # their attrition block — a pre-existing crash from the unguarded `arm_gap` in their
    # own loops, out of scope here and deliberately not papered over by this change.
    src = Path(inspect.getfile(is_diff_run)).parent
    for f in ("report.py", "terminal_report.py", "html_report.py"):
        text = (src / f).read_text()
        assert 'all("diff_ok" in r' not in text.replace(
            inspect.getsource(is_diff_run), ""), f"{f} re-spells the predicate"
    mixed = [{"qid": "m", "qtype": "count", "trials": 3, "attempts": 6, "diff_ok": 3}]
    assert "Attrition of the paired exam" not in build_html_diff_report({"m": mixed})


def test_the_docstring_call_site_count_matches_the_source():
    """This number was wrong twice — `FOUR` when it was five, then `SIX` when it was
    seven — in the header of the helper whose whole thesis is that a restated fact drifts.
    A comment that asserts a countable fact is a test, so it is counted."""
    import re
    src = Path(inspect.getfile(attrition_block)).parent
    sites = sum(len(re.findall(r"attrition_block\(", (src / f).read_text()))
                for f in ("report.py", "terminal_report.py", "html_report.py"))
    sites -= 1  # the definition itself
    words = {"FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}
    doc = attrition_block.__doc__ or ""
    claimed = next(v for k, v in words.items() if f"{k} call sites" in doc)
    assert claimed == sites, f"docstring says {claimed} call sites; source has {sites}"


# --------------------------------------------------------------------------- #
# The enumeration itself, so the next missed renderer fails CI, not review.
# --------------------------------------------------------------------------- #


# Renderers that consume the pairing but CANNOT disclose attrition, because their arms
# can never be excluded by it. Not an allowlist of forgiven omissions: each entry's
# premise is proven by execution in the test below this one, so an entry whose emitter
# later starts carrying real per-arm denominators goes red instead of staying quiet.
_CANNOT_EXCLUDE = {"report.py:build_codec_verdict_report"}

# Every renderer drawn over a paired subset today. `_build_diff_style_report` is absent
# on purpose: it is the shared BODY of `build_diff_report` and `build_text_diff_report`,
# both of which are listed, and the scan keys on the `build_*` prefix. Not the source of
# the requirement —
# that is computed — but the record of what is in scope, so a renderer that STOPS being
# paired shows up as a deliberate edit instead of quietly leaving the set.
_PAIRED_RENDERERS = {
    "report.py:build_fluency_report", "report.py:build_dropeval_report",
    "report.py:build_diff_report", "report.py:build_text_diff_report",
    "report.py:build_diff_soak_report", "report.py:build_codec_verdict_report",
    "terminal_report.py:build_terminal_fluency_report",
    "terminal_report.py:build_terminal_dropeval_report",
    "terminal_report.py:build_terminal_diff_report",
    "html_report.py:build_html_diff_report",
}


def _package() -> Path:
    return Path(inspect.getfile(attrition_block)).parent


def _call_graph(root: Path | None = None) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    """`(graph, attribute-only edges, ambiguous names)` over the WHOLE package.

    Every `.py` under `src/terse`, not a hand-written file list: a renderer added in a new
    module would never have entered a three-file scan, so "a renderer added later inherits
    the requirement" would have been false for exactly the case nobody thinks of.

    `root` exists so the parsing rules can be pinned on a SYNTHETIC package. They cannot
    be pinned on `src/terse`: it contains zero `async def` today, so reverting the async
    and attribute branches leaves every test green and the hardening is inert against the
    live tree (measured, #361 review). A throwaway probe that writes a module into `src/`
    and deletes it is not a test — it runs nowhere.

    `async def` counts, and so does an ATTRIBUTE callee (`report.arm_gap(...)`). Matching
    only `ast.FunctionDef` and only `ast.Name` callees let two silent paired renderers pass
    the entire suite (#361) — both written in idioms this package already uses.

    WHAT THIS STILL DOES NOT SEE. Recorded rather than implied away, because "adding a
    renderer without a disclosure fails CI" is only true within these limits. Each was
    demonstrated evading the full suite (#361 review):

      - a renderer defined outside `src/terse`, or one not named `build_*`;
      - `from .report import paired_rows as _pr` — a SYMBOL alias. The callee name is
        `_pr`, so the edge is never drawn. (A MODULE alias, `from . import report as rp`
        then `rp.paired_rows(...)`, IS caught: the attribute is still `paired_rows`. The
        asymmetry is the surprising part.) `lossy.py:33` already aliases a symbol this way;
      - `build_x = _impl`, a module-level lambda, `functools.partial`, or `getattr`
        dispatch — the `FunctionDef` behind the exported `build_*` name is privately named,
        so the prefix scan misses it. Arguably the "not named `build_*`" gap, but a reader
        of that gap would not predict them.

    A name-resolving import graph would close the first three. That is a bigger tool than
    this test, and the honest position is that this catches the mistakes people actually
    make here — not that it is airtight.

    Calls resolve BY NAME, with no import binding, so a name defined in two modules gets
    edges to both. That over-approximation is the UNSAFE direction for this test's
    load-bearing assertion — the check is "does this renderer FAIL to reach
    `attrition_block`", and a spurious edge can only make a silent renderer look
    compliant. (An earlier version of this docstring claimed the opposite, reasoning about
    "never reported as not reaching something" — the wrong half of the check.) So
    ambiguous names are returned and the caller refuses to traverse one.
    """
    graph: dict[str, set[str]] = {}
    attr_only: dict[str, set[str]] = {}
    defined: dict[str, set[str]] = {}
    pkg = root or _package()
    for f in sorted(pkg.rglob("*.py")):
        # The RELATIVE PATH, not `f.name`. Keying on the basename let two modules with the
        # same filename collide: the later-sorted one silently overwrote the earlier one's
        # entries, and same-named functions in same-basename modules were never marked
        # ambiguous. Demonstrated — a silent `build_diff_report` in
        # `src/terse/fluency/report.py` passed the whole suite, the third evasion of this
        # class (#361 review). `src/terse/fluency/__init__.py` and `src/terse/__init__.py`
        # already collide today, harmlessly only because neither defines a function.
        mod = str(f.relative_to(pkg))
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            key = f"{mod}:{node.name}"
            calls, attrs = set(), set()
            for c in ast.walk(node):
                if not isinstance(c, ast.Call):
                    continue
                if isinstance(c.func, ast.Name):
                    calls.add(c.func.id)
                elif isinstance(c.func, ast.Attribute):
                    # `report.arm_gap(...)` after `from . import report` — the idiom this
                    # package already uses. Dropping attribute callees made a renderer
                    # written that way invisible to this whole test (#361). Kept SEPARATE
                    # because `.attr` also matches every `x.get`/`x.append`/`re.sub`, and
                    # a stdlib method name colliding with a single src function is not
                    # `ambiguous` and would be traversed — see `_reaches`.
                    attrs.add(c.func.attr)
            graph[key] = calls | attrs
            attr_only[key] = attrs - calls
            defined.setdefault(node.name, set()).add(mod)
    return graph, attr_only, {n for n, mods in defined.items() if len(mods) > 1}


def _reaches(graph: dict[str, set[str]], start: str, target: str,
             ambiguous: set[str] = frozenset(),
             attr_only: Mapping[str, set[str]] | None = None) -> bool:
    """Does `start` transitively call `target`? Names in `ambiguous` are NOT traversed,
    and neither are attribute-derived edges when `attr_only` is given.

    Which approximation is safe depends on the question, and this is the half the first
    version of this file got backwards:

      - "is this renderer drawn over the pairing?" — over-approximate (traverse
        everything). A spurious edge can only ADD a renderer to the set that must
        disclose, which is conservative.
      - "does this renderer reach `attrition_block`?" — UNDER-approximate (skip ambiguous
        names). Here a spurious edge routes a SILENT renderer through a compliant namesake
        and scores it compliant, so over-approximation would hide the exact defect this
        test exists to find. Skipping can only report a compliant renderer as silent,
        which is loud and fixable.
    """
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        calls = graph.get(cur, set())
        if target in calls:
            return True
        skip = ambiguous | ((attr_only or {}).get(cur, set()))
        for name in calls - skip:
            stack.extend(k for k in graph if k.endswith(f":{name}"))
    return False


def _paired_and_silent(graph, attr_only, ambiguous, exempt=frozenset()):
    """`(paired renderers, the silent ones)` — THE check, in one place.

    Extracted so the synthetic-package tests exercise this body rather than a copy of it.
    A test that re-implements the logic it covers is the defect this file has already been
    bitten by twice; mutating the live check then leaves the synthetic test green."""
    paired = {k for k in graph if k.split(":", 1)[1].startswith("build_")
              and _reaches(graph, k, "paired_rows")}
    silent = {k for k in paired
              if not _reaches(graph, k, "attrition_block", ambiguous, attr_only)
              and k not in exempt}
    return paired, silent


def test_every_renderer_drawn_over_a_paired_subset_discloses_its_attrition():
    """The claim "every renderer carries it" was made three times and was false three
    times — the HTML page while `cli` printed two other renderers, then the diff family,
    then the diff SOAK whose markdown does not route through the shared diff body. Each
    was found by a reviewer reading call sites one at a time, and each fix enumerated the
    renderers BY HAND, which is why the next one was missed the same way.

    So the enumeration is computed: any `build_*` function anywhere in the package that
    transitively reaches `paired_rows` is drawn over a paired subset and must transitively
    reach `attrition_block`. A renderer added later — in any module — inherits the
    requirement, and adding one without a disclosure fails CI rather than waiting for a
    fifth reviewer."""
    graph, attr_only, ambiguous = _call_graph()
    # QUALIFIED keys (`report.py:build_diff_report`), never bare function names. Collapsing
    # to names let a silent renderer hide behind a compliant NAMESAKE: a
    # `build_diff_report` in `src/terse/fluency/report.py` was already in the expected set
    # under that name, and the check scored it disclosing. Keying `_call_graph` on the
    # relative path was necessary and NOT sufficient — measured, it still evaded.
    renderers, silent_set = _paired_and_silent(graph, attr_only, ambiguous, _CANNOT_EXCLUDE)
    assert renderers == _PAIRED_RENDERERS, (
        f"paired renderers changed: +{sorted(renderers - _PAIRED_RENDERERS)} "
        f"-{sorted(_PAIRED_RENDERERS - renderers)}. A new one must disclose; a departing "
        "one must be understood, not just removed from this set.")
    assert "report.py:build_diff_soak_report" in renderers, (
        "the soak was the renderer round 4 caught — it must stay in scope")
    silent = sorted(silent_set)
    assert not silent, (
        f"{silent} are drawn over a paired subset and disclose nothing about what left "
        "it. Wire attrition_block, or add to _CANNOT_EXCLUDE and prove the exemption in "
        "test_the_exempt_renderers_genuinely_cannot_be_excluded_by_the_pairing.")


def test_the_exempt_renderers_genuinely_cannot_be_excluded_by_the_pairing():
    """`_CANNOT_EXCLUDE` is only honest if its premise is executed. `codeceval` emits
    `raw_trials`/`terse_trials` FIXED at `trials` on purpose — it reports its loss through
    `fails`/`attempts` instead — so `_paired_arm` is unconditionally true for both arms
    and the pairing can never remove a codec question. An attrition block there would be a
    permanent `excluded 0`, which reads as "nothing was removed" rather than "removal is
    not measurable here".

    If `codeceval` ever starts emitting real per-arm denominators, this goes red and the
    exemption has to be revisited — which is the whole point of proving it rather than
    asserting it in a comment."""
    assert {"report.py:build_codec_verdict_report"} == _CANNOT_EXCLUDE
    # The codec row shape, with the terse arm losing every one of its calls.
    rows = [{"qid": f"q{i}", "qtype": "deref", "trials": 3, "raw_ok": 3, "terse_ok": 0,
             "raw_trials": 3, "terse_trials": 3, "fails": 3, "attempts": 6}
            for i in range(5)]
    src = inspect.getsource(codeceval.run_codec_payload)
    assert '"raw_trials": trials,' in src and '"terse_trials": trials,' in src, (
        "codeceval no longer pins its per-arm denominators to `trials`; the exemption's "
        "premise is gone and build_codec_verdict_report must now disclose")
    # The OTHER drift direction, and the one the grep above cannot see (#361):
    # `_arm_attempts` reads `<arm>_attempts` IN PREFERENCE TO `trials`, so emitting that
    # key makes exclusion possible with the two `_trials` lines untouched. Measured — with
    # `"terse_attempts": trials - terse_fail` added, the same row set goes from 5 kept to
    # 5 excluded while the whole suite stayed green.
    # An emitted KEY, not a bare substring: `run_codec_payload`'s docstring already
    # discusses attempts, and a future comment explaining why codeceval deliberately does
    # NOT emit this counter would redden the test with a message asserting a drift that
    # did not happen. `run_codec_fluency` is checked too — the row is assembled one frame
    # up as `{**tags, **toks, **row}`, so a counter introduced through `tags` or
    # `_payload_tokens` would restore excludability without touching the emitter below.
    for fn in (codeceval.run_codec_payload, codeceval.run_codec_fluency):
        assert not re.search(r'"\w+_attempts":', inspect.getsource(fn)), (
            f"{fn.__name__} now emits a per-arm attempts counter, which `_arm_attempts` "
            "prefers over `trials`; exclusion is possible again and the exemption is stale")
    assert len(paired_rows(rows, "terse_ok", "raw_ok")) == len(rows)
    assert attrition(rows, "terse_ok", "raw_ok").excluded == 0


def test_the_disclosure_check_refuses_a_spurious_route_through_an_ambiguous_name():
    """`_reaches` resolves calls by NAME, so a name defined in two modules gets edges to
    both. For the "does this renderer disclose?" question that is the dangerous direction:
    a silent renderer calling a helper whose namesake elsewhere reaches `attrition_block`
    would be scored compliant.

    Exercised on a synthetic graph because no real collision sits on a disclosure path
    today — which is exactly why the live source cannot pin this. Removing the `ambiguous`
    skip is invisible to every other test in this file; it is not invisible here."""
    graph = {
        "a.py:build_silent_report": {"helper", "paired_rows"},
        "a.py:helper": set(),                          # the one it really calls
        "b.py:helper": {"attrition_block"},            # an unrelated namesake
    }
    ambiguous = {"helper"}
    assert _reaches(graph, "a.py:build_silent_report", "paired_rows")
    assert _reaches(graph, "a.py:build_silent_report", "attrition_block"), (
        "premise: traversing the ambiguous name DOES find a spurious route")
    assert not _reaches(graph, "a.py:build_silent_report", "attrition_block", ambiguous), (
        "a silent renderer must not be scored compliant via a namesake in another module")


# --------------------------------------------------------------------------- #
# Round 5, second reviewer: guards that were reachable and pinned by nothing.
# --------------------------------------------------------------------------- #


def _not_a_diff_run_rows() -> list[dict]:
    """Fluency-shaped rows with a real terse-arm loss, and no `diff_ok` anywhere."""
    return [{"qid": f"n{i}", "qtype": "count", "trials": 3, "attempts": 3,
             "terse_ok": 2, "terse_trials": 2} for i in range(4)]


def test_the_diff_markdown_and_chart_withhold_a_model_that_is_not_a_diff_run():
    """`is_diff_run` was pinned for the HTML page and for NOTHING else — dropping the
    guard at the two renderers `cli` prints on every diff path survived all 1847 tests.

    It is not an equivalent mutant. A merged result file carrying one diff-eval model and
    one fluency-shaped model prints an attrition clause for a model the table already
    withheld as "not a diff run", attributing its loss to `terse_ok` under a note that
    says the named arm is the one the exclusion flatters — a selection-bias claim about a
    pairing that was never performed."""
    results = {"m": _diff_rows(), "notdiff": _not_a_diff_run_rows()}
    assert attrition(_not_a_diff_run_rows(), *DIFF_ARMS).excluded == 4, (
        "fixture: the unguarded renderer WOULD print a clause for this model")
    # The SOAK is here because the first version of this test listed three renderers by
    # hand and missed the fourth (#361) — the same by-hand enumeration this whole file was
    # written to end, recurring inside the fix for it. `build_diff_soak_report` needs
    # `depth`, so it gets its own copies.
    soak = {m: [dict(r, depth=1) for r in rows] for m, rows in results.items()}
    for text in (build_diff_report(results),
                 build_text_diff_report(results),
                 build_diff_soak_report(soak),
                 build_terminal_diff_report(results, color=False)):
        block = text.split("Attrition of the paired exam", 1)[1]
        assert "diff_ok 5" in block
        assert "notdiff" not in block


def test_the_diff_soak_attrition_pairs_on_BOTH_arms():
    """The defect `test_the_attrition_pairs_on_exactly_the_arms_the_gap_pairs_on` exists
    to prevent, re-entered in the renderer round 4 added: dropping the control arm from
    the soak's `attrition(...)` survived, because the other soak fixture loses only on
    `diff_ok`. A soak whose TERSE anchor arm loses the deep questions would then report
    `excluded 0` and render nothing."""
    rows = [{"qid": f"c{i}", "qtype": "count", "trials": 3, "attempts": 6, "depth": 1,
             "diff_ok": 3, "terse_ok": 3, "diff_trials": 3, "terse_trials": 3}
            for i in range(20)]
    rows += [{"qid": f"d{i}", "qtype": "deref", "trials": 3, "attempts": 6, "depth": 5,
              "diff_ok": 3, "terse_ok": 2, "diff_trials": 3, "terse_trials": 2}
             for i in range(5)]
    md = build_diff_soak_report({"m": rows})
    assert "Attrition of the paired exam" in md, "a terse-arm loss must not be invisible"
    assert "terse_ok 5" in md and "deref 5/5" in md


def test_no_diff_site_spells_the_arm_pair_literally():
    """`DIFF_ARMS` was read by the four `attrition(...)` calls while all seven
    `arm_gap`/`paired_rows` calls still wrote the pair out, so the constant's own comment
    claimed a sharing that did not exist. Threaded now, the way `FLUENCY_GATING` was."""
    src = _package()
    for f in ("report.py", "html_report.py", "terminal_report.py"):
        text = (src / f).read_text()
        body = text.replace('DIFF_ARMS = ("diff_ok", "terse_ok")', "")
        assert '"diff_ok", "terse_ok"' not in body, (
            f"{f} spells the diff arm pair literally; read DIFF_ARMS instead")


def test_the_terminal_dropeval_block_is_separated_by_exactly_one_blank_line():
    """`attr` carries a leading `"\\n\\n"` and the sections are joined with `"\\n\\n"`, so
    dropping the `.strip()` yields four consecutive newlines. Cosmetic, but it was a
    decision in the delta with nothing behind it."""
    rows = [_drow(f"q{i}", t_err=3, c_err=0) for i in range(4)]
    text = build_terminal_dropeval_report({"m": rows}, color=False, accept_degraded=True)
    assert "\n\n\n" not in text


def test_a_run_with_no_control_arm_does_not_explain_the_control_arms_role():
    """`--no-control` is a live flag and emits no `control_ok`/`control_trials` at all.
    The dropeval note explains why exclusion can only land on the control side and points
    at a **Where they failed** split — meaningless without a control arm, and that section
    does not exist in the two-line terminal chart at all. Moving the attrition above the
    terminal's `v.inconclusive` return made the prose newly reachable there."""
    rows = [{"qid": f"q{i}", "kind": "recall", "trials": 3, "retrieve_ok": 3,
             "answer_ok": 0, "handle_ok": 3, "errors": 3, "treatment_errors": 3,
             "control_errors": 0, "attempts": 3} for i in range(4)]
    assert not any("control_ok" in r for r in rows), "fixture: no control arm ran"
    # `accept_degraded`: at 12/12 lost calls the markdown returns INCONCLUSIVE before its
    # attrition block. The note choice is what is under test, not the degradation gate.
    for text in (build_dropeval_report({"m": rows}, accept_degraded=True),
                 build_terminal_dropeval_report({"m": rows}, color=False),
                 build_terminal_dropeval_report({"m": rows}, color=False,
                                                accept_degraded=True)):
        assert "treatment lost 12 call(s)" in text
        # stripped on both sides: the terminal spelling drops backticks (see `_reaches`'
        # sibling rule in `attrition_block`), so the raw constant is absent there by design.
        assert strip_markup(NO_CONTROL_ATTRITION_NOTE) in strip_markup(text)
        assert "only ever exclude on the CONTROL side" not in text
        assert "Where they failed** above" not in text


def test_the_soaks_worked_example_is_produced_by_a_fixture_not_asserted_in_prose():
    """`deref 15/15` is quoted as fact in `build_diff_soak_report`'s comment, in a test
    docstring and in the CHANGELOG — and nothing in the tree produced it (#361). A comment
    asserting a countable fact is a test, so it is executed here: 3 depths x 5 `deref`
    questions, all lost by the diff arm, against 3 x 20 clean `count` questions."""
    rows = []
    for depth in (1, 3, 5):
        rows += [{"qid": f"c{depth}_{i}", "qtype": "count", "trials": 3, "attempts": 6,
                  "depth": depth, "diff_ok": 3, "terse_ok": 3,
                  "diff_trials": 3, "terse_trials": 3} for i in range(20)]
        rows += [{"qid": f"d{depth}_{i}", "qtype": "deref", "trials": 3, "attempts": 6,
                  "depth": depth, "diff_ok": 2, "terse_ok": 3,
                  "diff_trials": 2, "terse_trials": 3} for i in range(5)]
    md = build_diff_soak_report({"m": rows})
    assert "excluded 15/75 question(s)" in md
    assert "by arm: diff_ok 15" in md
    assert "deref 15/15, count 0/60" in md


# --------------------------------------------------------------------------- #
# The PARSING RULES, pinned on a synthetic package. `src/terse` cannot pin them:
# it has zero `async def` and no attribute-paired renderer, so every rule below
# can be reverted with all 1853 tests green (measured). See `_call_graph(root=)`.
# --------------------------------------------------------------------------- #


def _synthetic(tmp_path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return tmp_path


def test_call_graph_sees_async_defs(tmp_path):
    """`ast.AsyncFunctionDef` is not an `ast.FunctionDef`. Matching only the latter made an
    `async def` renderer absent from the graph as BOTH a node and a traversable callee."""
    root = _synthetic(tmp_path, {"m.py": "async def build_x_report(r):\n    return paired_rows(r)\n"})
    graph, _, _ = _call_graph(root)
    assert "m.py:build_x_report" in graph
    assert "paired_rows" in graph["m.py:build_x_report"]


def test_call_graph_sees_attribute_callees(tmp_path):
    """`report.paired_rows(...)` after `from . import report` — the idiom 11 modules here
    use. Matching only `ast.Name` callees dropped the edge entirely."""
    root = _synthetic(tmp_path, {"m.py": "def build_x_report(r):\n    return report.paired_rows(r)\n"})
    graph, attr_only, _ = _call_graph(root)
    assert "paired_rows" in graph["m.py:build_x_report"]
    assert "paired_rows" in attr_only["m.py:build_x_report"], (
        "attribute edges must be tracked separately — see `_reaches`")


def test_call_graph_keys_on_the_relative_path_not_the_basename(tmp_path):
    """Two modules sharing a basename collided: the later-sorted one silently overwrote
    the earlier one's entries. `src/terse/fluency/report.py` vs `src/terse/report.py` is
    the live shape of this, and `__init__.py` already collides today (harmlessly, only
    because neither defines a function)."""
    root = _synthetic(tmp_path, {
        "report.py": "def build_x_report(r):\n    return paired_rows(r)\n",
        "sub/report.py": "def build_x_report(r):\n    return 1\n"})
    graph, _, ambiguous = _call_graph(root)
    assert {"report.py:build_x_report", "sub/report.py:build_x_report"} <= set(graph)
    assert graph["report.py:build_x_report"] == {"paired_rows"}
    assert graph["sub/report.py:build_x_report"] == set()
    assert "build_x_report" in ambiguous, "same name in two modules is ambiguous"


def test_a_renderer_cannot_hide_behind_a_COMPLIANT_NAMESAKE(tmp_path):
    """The check works in qualified keys. Collapsing to bare names let a silent
    `build_diff_report` in `fluency/report.py` be scored compliant by `report.py`'s
    disclosing one of the same name — keying `_call_graph` on the relative path was
    necessary and NOT sufficient, and this is the assertion that caught the difference."""
    root = _synthetic(tmp_path, {
        "report.py": ("def build_x_report(r):\n"
                      "    attrition_block(attrition(r))\n"
                      "    return paired_rows(r)\n"),
        "sub/report.py": "def build_x_report(r):\n    return paired_rows(r)\n"})
    _, silent = _paired_and_silent(*_call_graph(root))
    assert silent == {"sub/report.py:build_x_report"}, (
        "the silent namesake must be named, not absorbed by the compliant one")


def test_attribute_edges_are_not_traversed_in_the_disclosure_direction(tmp_path):
    """`.attr` matches every `x.get`/`re.sub` too, and a stdlib method name colliding with
    a SINGLE src function is not `ambiguous`, so it would be traversed. Here `helper.write`
    is a method call whose name happens to match a module-level `write` that discloses:
    traversing it would score a silent renderer compliant. 118 such collisions exist in
    `src/terse` today; none currently reaches `attrition_block`, so only a synthetic
    package can pin the rule."""
    root = _synthetic(tmp_path, {
        "m.py": ("def build_x_report(r):\n"
                 "    helper.write(r)\n"          # attribute edge -> `write`
                 "    return paired_rows(r)\n"
                 "def write(r):\n"
                 "    return attrition_block(r)\n")})
    graph, attr_only, ambiguous = _call_graph(root)
    assert "write" in attr_only["m.py:build_x_report"], "fixture: the edge must exist"
    _, silent = _paired_and_silent(graph, attr_only, ambiguous)
    assert silent == {"m.py:build_x_report"}, (
        "a method call must not route a silent renderer to a disclosing namesake")
