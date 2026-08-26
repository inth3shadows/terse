"""A correlated loss cannot publish a PASS — at every renderer (#280).

The bug: when a model's lost calls correlate with the arm under test (a token-budget stop
kills the LONGEST prompt first, and the diff/terse arm's prompt is strictly longer than its
control's), `_form_stats` divides each arm by its OWN surviving trial count, so the arm that
lost the hard questions is scored over an easier question set than its control. The gap then
compares two different exams and flatters the arm being tested.

WHY THE FIXTURE LOOKS THE WAY IT DOES — this is the part three previous attempts got wrong.

A gate already existed before pairing, and a fixture that trips it proves nothing about
pairing: the report would refuse to publish anyway, the test would pass, and the pairing
could be deleted without going red. That is exactly what happened to the last attempt's
headline test, which passed unmodified against the un-fixed code.

So `_correlated_loss_rows` is built to sit UNDER it:

  - pooled loss `_unmeasured`: 3 lost calls of 36 = 8.3%, under `UNMEASURED_FAIL_SHARE`
    (0.20, strictly `>`).
`test_the_fixture_stays_under_the_pre_existing_gate` asserts that margin directly, so a
later edit cannot quietly re-base these tests onto a gate they are not about.

What is left is pure pairing, and the numbers are unambiguous:

    unpaired  form 80.0%  control 83.3%  gap  -3.3%  -> PASS at 5% tolerance
    paired    form 80.0%  control 100.0% gap -20.0%  -> FAIL

Every test below asserts the FAIL reaches the reader. Each one is verified by reverting its
own site's wiring and confirming THIS test goes red — see the PR body for the results.
`test_gap_gate_boundary.py` is the other half: it stops an eighth site being written.
"""
from __future__ import annotations

import pytest

from terse.html_report import build_html_diff_report
from terse.report import (
    _GAP_TOLERANCE,
    _MIN_PAIRED_QUESTIONS,
    UNMEASURED_FAIL_SHARE,
    _form_stats,
    _unmeasured,
    arm_gap,
    build_diff_report,
    build_diff_soak_report,
    build_fluency_report,
    diff_gap_rows,
    fluency_gap_rows,
    paired_rows,
)
from terse.terminal_report import (
    build_terminal_diff_report,
    build_terminal_fluency_report,
)

TRIALS = 3
# Clean-row fixtures must also clear #334's floor to stay SCORED rather than withheld.
_CLEAN_N = 20
# Per-row form successes across the five questions BOTH arms answered: 12 of 15 = 80%. The
# control answers all fifteen. A real, ordinary regression — the point is that it is
# invisible until the arms are paired.
# Repeated x4 so the paired subset clears `_MIN_PAIRED_QUESTIONS` (#334) while every ratio
# these tests assert is untouched: 4x[3,3,3,2,1] = 48/60, the same 80%.
FORM_OK = [3, 3, 3, 2, 1] * 4


def _correlated_loss_rows(depth: int | None = None) -> list[dict]:
    """Six questions. The control answers all of them; the form arm loses every trial of
    the one question the control also finds hard, and is genuinely worse on the rest."""
    rows = [{
        # The hard question. The control scores 0 on it — so dropping it from the control's
        # denominator is what inflates the control-relative comparison — and the form arm
        # never answered it at all (`diff_trials` 0). One per five paired rows: the whole
        # file rests on "unpaired reads -3.3%, paired reads -20%", and that ratio is a
        # property of hard:paired, not of either count, so #334's x4 applies to both.
        "qid": f"hard{h}", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "terse_ok": 0, "terse_trials": TRIALS, "diff_ok": 0, "diff_trials": 0,
        "fails": TRIALS, "attempts": TRIALS * 2,
    } for h in range(len(FORM_OK) // 5)]
    for qid, ok in zip([f"p{i}" for i in range(len(FORM_OK))], FORM_OK, strict=True):
        rows.append({
            "qid": qid, "qtype": "lookup", "transform": "table", "trials": TRIALS,
            "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": ok, "diff_trials": TRIALS,
            "fails": 0, "attempts": TRIALS * 2,
        })
    if depth is not None:
        for r in rows:
            r["depth"] = depth
    return rows


def _clean_rows(depth: int | None = None) -> list[dict]:
    """Six questions both arms answered perfectly — the shallow, undamaged half of a soak."""
    rows = [{
        "qid": f"clean{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": TRIALS, "diff_trials": TRIALS,
        "fails": 0, "attempts": TRIALS * 2,
    } for i in range(_CLEAN_N)]
    if depth is not None:
        for r in rows:
            r["depth"] = depth
    return rows


def _payload_correlated_loss_rows() -> list[dict]:
    """The same correlated loss in the payload family: control `raw`, forms terse/primer.

    No `inline_ok`: that arm is display-only (it gates nothing and carries the longest
    prompt of the four), and leaving it out keeps this fixture about the gated arms.
    """
    rows = [{
        "qid": f"hard{h}", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "raw_ok": 0, "raw_trials": TRIALS,
        "terse_ok": 0, "terse_trials": 0, "primer_ok": 0, "primer_trials": 0,
        "fails": TRIALS * 2, "attempts": TRIALS * 3,
    } for h in range(len(FORM_OK) // 5)]
    for qid, ok in zip([f"p{i}" for i in range(len(FORM_OK))], FORM_OK, strict=True):
        rows.append({
            "qid": qid, "qtype": "lookup", "transform": "table", "trials": TRIALS,
            "raw_ok": TRIALS, "raw_trials": TRIALS,
            "terse_ok": ok, "terse_trials": TRIALS,
            # Strictly below terse, so "best of terse/primer" is unambiguously terse and
            # the verdict cannot be rescued by the primer arm.
            "primer_ok": max(ok - 1, 0), "primer_trials": TRIALS,
            "fails": 0, "attempts": TRIALS * 3,
        })
    return rows


# --------------------------------------------------------------------------------------
# The fixture's own preconditions. If these drift, every test below silently stops
# testing pairing and starts testing a gate that was already there.
# --------------------------------------------------------------------------------------

def test_the_fixture_stays_under_the_pre_existing_gate():
    rows = _correlated_loss_rows()
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE, (
        f"pooled loss {lost:.1%} would trip `_unmeasured` on its own, so these tests would "
        f"pass with pairing deleted")
    assert not _unmeasured(rows)



def test_without_pairing_this_fixture_reads_as_a_pass():
    """The false PASS itself, stated as arithmetic rather than trusted as a premise."""
    from terse.report import _form_stats
    rows = _correlated_loss_rows()
    facc, _ = _form_stats(rows, "diff_ok")
    cacc, _ = _form_stats(rows, "terse_ok")
    assert facc - cacc == pytest.approx(-0.0333, abs=1e-3)
    assert facc - cacc > -0.05  # inside tolerance -> would publish PASS

    pr = paired_rows(rows, "diff_ok", "terse_ok")
    pfacc, _ = _form_stats(pr, "diff_ok")
    pcacc, _ = _form_stats(pr, "terse_ok")
    assert pfacc - pcacc == pytest.approx(-0.20, abs=1e-9)  # the truth: a -20% regression


# --------------------------------------------------------------------------------------
# Site 1 — the diff markdown table and its ship-gate verdict.
# --------------------------------------------------------------------------------------

def test_a_correlated_loss_cannot_publish_PASS_in_the_diff_markdown():
    md = build_diff_report({"m": _correlated_loss_rows()})
    assert "**FAIL**" in md
    assert "**PASS**" not in md
    assert "safe to enable" not in md


# --------------------------------------------------------------------------------------
# Site 2 — the HTML banner. The most-quoted artifact of a run.
# --------------------------------------------------------------------------------------

def test_a_correlated_loss_cannot_publish_PASS_in_the_html_banner():
    html = build_html_diff_report({"m": _correlated_loss_rows()})
    assert "✕ FAIL" in html
    assert "✓ PASS" not in html
    assert "banner critical" in html


def test_the_html_banner_gives_no_verdict_for_rows_that_are_not_a_diff_run():
    """The old `rows[0]`-key heuristic set control := form for a payload-shaped row set,
    making the gap identically 0 and the banner unconditionally green."""
    html = build_html_diff_report({"m": _payload_correlated_loss_rows()})
    assert "✓ PASS" not in html


# --------------------------------------------------------------------------------------
# Sites 3a / 3b — the terminal forest plots. `terminal_report` computes no accuracy of
# its own, so these pin `diff_gap_rows` / `fluency_gap_rows` through their real consumer.
# --------------------------------------------------------------------------------------

def test_a_correlated_loss_cannot_publish_PASS_in_the_diff_forest_plot():
    gap_rows, _ = diff_gap_rows({"m": _correlated_loss_rows()})
    facc, _, cacc, _ = gap_rows["m"]
    assert facc - cacc == pytest.approx(-0.20, abs=1e-9)
    assert "FAIL" in build_terminal_diff_report({"m": _correlated_loss_rows()}, color=False)


def test_a_correlated_loss_cannot_publish_PASS_in_the_fluency_forest_plot():
    gap_rows, _ = fluency_gap_rows({"m": _payload_correlated_loss_rows()})
    facc, _, cacc, _ = gap_rows["m"]
    assert facc - cacc == pytest.approx(-0.20, abs=1e-9)
    plot = build_terminal_fluency_report({"m": _payload_correlated_loss_rows()}, color=False)
    assert "FAIL" in plot


# --------------------------------------------------------------------------------------
# Sites 4a / 4b — the soak by-depth table and the deepest-depth verdict. Both were
# unpaired, and the by-depth table sits OUTSIDE `## Verdict`, where the existing
# invariance test's `_gate_signature` cannot see it.
# --------------------------------------------------------------------------------------

def _soak_results() -> dict:
    """Clean at depth 1, correlated loss at depth 5 — drift that appears only with depth."""
    return {"m": _clean_rows(depth=1) + _correlated_loss_rows(depth=5)}


def test_a_correlated_loss_cannot_publish_PASS_in_the_soak_by_depth_table():
    md = build_diff_soak_report(_soak_results())
    depth5 = [ln for ln in md.splitlines() if ln.startswith("| `m` | 5 |")]
    assert depth5, f"no depth-5 row in:\n{md}"
    # -20%, the paired truth — not the -3% an unpaired slice reports.
    assert "-20%" in depth5[0], depth5[0]


def test_a_correlated_loss_cannot_publish_PASS_in_the_deepest_depth_verdict():
    md = build_diff_soak_report(_soak_results())
    deepest = [ln for ln in md.splitlines() if "deepest tested depth" in ln]
    assert deepest, f"no deepest-depth verdict in:\n{md}"
    assert "**FAIL**" in deepest[0], deepest[0]
    assert "No depth-correlated comprehension drift" not in md


# --------------------------------------------------------------------------------------
# Site 5 — the fluency markdown verdict, the main terse-vs-raw gate.
# --------------------------------------------------------------------------------------

def test_a_correlated_loss_cannot_publish_PASS_in_the_fluency_verdict():
    md = build_fluency_report({"m": _payload_correlated_loss_rows()}, [])
    assert "**FAIL**" in md
    assert "**PASS**" not in md
    assert "preserves comprehension within tolerance" not in md


def test_the_fluency_table_counts_regressions_over_the_paired_subset():
    """F5: a question the terse arm never ANSWERED is not a question it got wrong.

    Its own fixture, because the correlated-loss one cannot show this: there the lost row
    also has `raw_ok` 0, so the `raw_ok == trials` half of the regression predicate rejects
    it either way and paired/unpaired agree by accident. The discriminating shape is a row
    the CONTROL answered perfectly and the form arm never answered at all — unpaired that
    is a regression, paired it is not there.
    """
    rows = [{
        "qid": "never-answered", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": 0, "terse_trials": 0, "primer_ok": 0, "primer_trials": 0,
        "fails": TRIALS * 2, "attempts": TRIALS * 3,
    }] + [{
        "qid": f"ok{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "primer_ok": TRIALS, "primer_trials": TRIALS,
        "fails": 0, "attempts": TRIALS * 3,
    } for i in range(20)]
    # Same preconditions as the main fixture: neither pre-existing gate may fire.
    assert not _unmeasured(rows)

    md = build_fluency_report({"m": rows}, [])
    row = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert row, md
    cells = [c.strip() for c in row[0].split("|")]
    assert cells[7] == "0", (
        f"regressions cell was {cells[7]!r} — the unanswered question is being counted as "
        f"a wrong answer: {row[0]!r}")


# --------------------------------------------------------------------------------------
# The pairing predicate's own contract.
# --------------------------------------------------------------------------------------

def test_an_uneven_score_pack_still_publishes():
    """A hand-built pack with fewer replies for one form is a COLLECTION mode, not a loss.

    `score_pack` emits per-form trial counts precisely so a sparser form is scored over its
    own denominator (#91). Those rows carry no `attempts`, because nothing was ever sent to
    a backend — so an uneven count there is not evidence of a lost call, and pairing on it
    would delete a documented feature to defend against a failure that cannot occur.
    """
    rows = [{
        "tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "lookup", "transform": "table",
        "trials": 3,
        "raw_ok": 3, "raw_trials": 3,
        "terse_ok": 2, "terse_trials": 2,   # only two replies collected for this form
        "primer_ok": 3, "primer_trials": 3,
    } for i in range(_CLEAN_N)]
    assert paired_rows(rows, "terse_ok", "raw_ok") == rows

    md = build_fluency_report({"m": rows}, [])
    assert "n/a | n/a | n/a" not in md, "an uneven pack was withheld as if calls were lost"
    assert "100%" in md  # the raw column still publishes


def test_a_withheld_deepest_depth_is_not_reported_as_no_drift():
    """Absence of a measurement is not a passing measurement.

    The per-depth gate can withhold the deepest slice while the pooled model still
    publishes, which made `deepest is None` reachable inside `if worst:` — where it was
    read as "passed" and printed "No depth-correlated comprehension drift" about the one
    depth the soak exists to probe.
    """
    def q(depth: int, *, dead: bool) -> dict:
        return {
            "qid": f"d{depth}", "qtype": "lookup", "transform": "table", "trials": 1,
            "terse_ok": 0 if dead else 1, "terse_trials": 0 if dead else 1,
            "diff_ok": 0 if dead else 1, "diff_trials": 0 if dead else 1,
            "fails": 2 if dead else 0, "attempts": 2, "depth": depth,
        }

    # Depths 1-4 clean; every call at the deepest depth failed, so that SLICE is withheld
    # by `_unmeasured` while the model as a whole (10 of 100 calls lost) is not. That
    # combination is what made the bug reachable: the overall gap still publishes a PASS
    # while the depth the soak exists to probe was never scored.
    rows = [q(d, dead=False) for d in (1, 2, 3, 4) for _ in range(10)]
    rows += [q(5, dead=True) for _ in range(5)]

    assert not _unmeasured(rows), "the per-model gate fired; this test would prove nothing"
    deep = [r for r in rows if r["depth"] == 5]
    assert _unmeasured(deep), "the deepest slice must be withheld on its own"

    md = build_diff_soak_report({"m": rows})
    assert "**PASS**" in md, f"the overall gap should still publish a PASS:\n{md}"
    assert "No depth-correlated comprehension drift" not in md, md
    assert "NO VERDICT at the deepest tested depth" in md, md
    # And the withheld depth is NAMED, rather than left as an unexplained `n/a`.
    # Dead backend at this depth, so the TRANSPORT heading is the true one. #332 briefly
    # collapsed both causes into "not compared" here; the per-depth `_unmeasured` split
    # restored the distinction, and this is the fixture that proves the right half fires.
    assert "Depths not measured" in md
    assert "depth 5" in md


def test_the_per_transform_table_pools_only_paired_rows():
    """The by-transform columns are display-only, but they were still biased.

    A row with a PARTIAL loss is the discriminating shape: `_form_stats` already drops
    lost trials from the denominator, so a row an arm never touched contributes nothing
    either way — but a row where the arm answered 1 of 3 trials contributes that one
    surviving trial, and the surviving trials are the easy ones. Pooling unpaired rows
    therefore reads HIGH. This is the table whose own comment says a reader uses it to
    decide which transforms to keep in the policy.
    """
    rows = [{
        "qid": "partial", "qtype": "lookup", "transform": "table", "trials": 3,
        "raw_ok": 3, "raw_trials": 3,
        "terse_ok": 1, "terse_trials": 1,     # 2 of 3 trials lost; the survivor was right
        "primer_ok": 3, "primer_trials": 3,
        "fails": 2, "attempts": 9,
    }] + [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": 3,
        "raw_ok": 3, "raw_trials": 3,
        "terse_ok": 1, "terse_trials": 3,     # genuinely poor: 1 of 3
        "primer_ok": 3, "primer_trials": 3,
        "fails": 0, "attempts": 9,
    } for i in range(22)]
    # The model itself must still publish, or the table never renders.
    assert not _unmeasured(rows)

    md = build_fluency_report({"m": rows}, [])
    line = [ln for ln in md.splitlines() if ln.startswith("| table |")]
    assert line, md
    terse_cell = [c.strip() for c in line[0].split("|")][3]
    # paired: 7/21 = 33%. unpaired: 8/22 = 36%, flattered by the one surviving easy trial.
    assert terse_cell == "33%", (
        f"per-transform terse column was {terse_cell!r} — expected the paired 33%, not the "
        f"unpaired 36%: {line[0]!r}")


def test_the_diff_table_counts_regressions_over_the_paired_subset():
    """The diff-side twin of the fluency F5 test, with its own discriminating fixture.

    `_correlated_loss_rows` cannot show this: its withheld row has `terse_ok == 0`, so the
    predicate's `terse_ok == trials` clause rejects it paired or unpaired and the counts
    agree by accident. The shape that separates them is a question the CONTROL answered
    perfectly and the form arm never answered at all.
    """
    rows = [{
        "qid": "never-answered", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": 0, "diff_trials": 0,
        "fails": TRIALS, "attempts": TRIALS * 2,
    }] + [{
        "qid": f"ok{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": TRIALS, "diff_trials": TRIALS,
        "fails": 0, "attempts": TRIALS * 2,
    } for i in range(20)]
    assert not _unmeasured(rows)

    md = build_diff_report({"m": rows})
    row = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert row, md
    cells = [c.strip() for c in row[0].split("|")]
    assert cells[5] == "0", (
        f"regressions cell was {cells[5]!r} — the unanswered question is being counted as "
        f"a wrong answer: {row[0]!r}")


def test_the_fluency_table_counts_primer_recoveries_over_the_paired_subset():
    """Same for the `primer recovers` column, which had no discriminating fixture either.

    Needs a row the terse arm never answered but the PRIMER arm answered perfectly —
    unpaired that reads as "the primer rescued a question terse got wrong", which is a
    claim about a question terse was never asked.
    """
    rows = [{
        "qid": "never-answered", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": 0, "terse_trials": 0,
        "primer_ok": TRIALS, "primer_trials": TRIALS,
        "fails": TRIALS, "attempts": TRIALS * 3,
    }] + [{
        "qid": f"ok{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "primer_ok": TRIALS, "primer_trials": TRIALS,
        "fails": 0, "attempts": TRIALS * 3,
    } for i in range(20)]
    assert not _unmeasured(rows)

    md = build_fluency_report({"m": rows}, [])
    row = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert row, md
    cells = [c.strip() for c in row[0].split("|")]
    assert cells[8] == "0", (
        f"primer-recovers cell was {cells[8]!r} — a question terse never answered is being "
        f"counted as one the primer rescued: {row[0]!r}")


def test_the_payload_fixture_also_stays_under_both_pre_existing_gates():
    """Sites 3b and 5 rest on this fixture; its margins deserve the same assertion."""
    rows = _payload_correlated_loss_rows()
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE, f"pooled loss {lost:.1%} would trip `_unmeasured`"
    assert not _unmeasured(rows)


def test_rows_without_per_form_counters_are_treated_as_fully_paired():
    """Legacy result files and `score_pack` output carry no `<form>_trials` key at all.

    Reading an absent counter as a loss would void every one of them wholesale, so
    `paired_rows` defaults it to the row's `trials`. Load-bearing and easy to mistake for
    an accident, hence pinned.
    """
    legacy = [{"qid": "a", "qtype": "lookup", "transform": "table", "trials": 1,
               "terse_ok": 1, "diff_ok": 1}]
    assert paired_rows(legacy, "diff_ok", "terse_ok") == legacy


def test_the_question_column_reports_the_exam_that_was_actually_sat():
    """`q` must be the paired count, not the questions generated.

    Every accuracy on the line comes from the paired subset, so printing the full count
    beside them states a denominator the numbers do not use — and the by-transform table
    one section below already pools the paired rows, so one document showed two exam sizes
    for one model with no explanation.
    """
    # 6 of 30 questions lost outright — 3 by each arm — so 24 are comparable
    # (and 24 clears `_MIN_PAIRED_QUESTIONS`, so the row is scored, not withheld).
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": 5,
        "terse_ok": 0 if 3 <= i < 6 else 5, "terse_trials": 0 if 3 <= i < 6 else 5,
        "diff_ok": 0 if i < 3 else 5, "diff_trials": 0 if i < 3 else 5,
        "fails": 5 if i < 6 else 0, "attempts": 10,
    } for i in range(30)]
    assert not _unmeasured(rows), "the transport gate must not be what withholds this"
    scored = len(paired_rows(rows, "diff_ok", "terse_ok"))
    assert scored == 24

    md = build_diff_report({"m": rows})
    row = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert row and f"| `m` | {scored} |" in row[0], (
        f"q column should be the paired {scored}, not 30: {row}")

# --------------------------------------------------------------------------------------
# Site 6 — the pairing-loss floor itself (#332).
#
# The gate above voids a row when either arm lost a trial of it. Nothing then checked how
# much of the question set that left. `_gap`'s docstring promised a second stage for
# exactly this ("too little of the question set survived on one side to compare") and it
# was never implemented, so a form arm that lost every paired trial reached the reader as
# a green PASS.
#
# Why the fail-share gate cannot catch it: `paired_rows` voids a WHOLE ROW for ONE lost
# trial, so loss is amplified from the call level to the question level. At 3 trials per
# arm, one lost call per row is 1-in-6 of the calls (16.7%, under `UNMEASURED_FAIL_SHARE`)
# and 6-in-6 of the questions. The two gates measure different quantities; the second one
# is not redundant with the first at any threshold.
# --------------------------------------------------------------------------------------

def _pairing_wipeout_rows(n: int = 10, lost: int | None = None) -> list[dict]:
    """`lost` of `n` questions lose one diff trial apiece — enough to void them entirely.

    Verbatim shape from #332: the control answers all three trials of every question, and
    the form arm is one trial short on the damaged ones. Overall fail share stays at
    16.7%, so `_unmeasured` never fires."""
    lost = n if lost is None else lost
    return [{
        "qid": f"q{i}", "qtype": "count", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "diff_ok": 0 if i < lost else TRIALS,
        "diff_trials": TRIALS - 1 if i < lost else TRIALS,
        "attempts": TRIALS * 2, "fails": 1 if i < lost else 0,
    } for i in range(n)]


def test_the_wipeout_fixture_stays_under_the_pre_existing_gate():
    """Same precondition the rest of this file lives by: if `_unmeasured` already fired,
    the tests below would pass with the pairing floor deleted."""
    rows = _pairing_wipeout_rows()
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE, f"pooled loss {lost:.1%} trips `_unmeasured` alone"
    assert not _unmeasured(rows)
    assert paired_rows(rows, "diff_ok", "terse_ok") == []


def test_a_total_pairing_loss_is_withheld_rather_than_scored():
    g = arm_gap(_pairing_wipeout_rows(), "diff_ok", "terse_ok")
    assert g.excluded == "unmeasured", (
        "nothing survived pairing, so there is no comparison to publish")


def test_a_total_pairing_loss_cannot_publish_PASS_in_the_html_banner():
    html = build_html_diff_report({"m": _pairing_wipeout_rows()}, "diff-form", "full-terse")
    assert "✓ PASS" not in html


def test_a_total_pairing_loss_cannot_publish_PASS_in_the_diff_markdown():
    md = build_diff_report({"m": _pairing_wipeout_rows()})
    assert "**PASS**" not in md
    assert "safe to enable" not in md


def test_a_total_pairing_loss_cannot_publish_PASS_in_the_diff_forest_plot():
    plot = build_terminal_diff_report({"m": _pairing_wipeout_rows()}, color=False)
    assert "PASS" not in plot


def test_a_demonstrated_regression_is_never_withheld_as_unmeasured():
    """The guard against re-introducing a survival FLOOR. Read this before adding one.

    The first fix for #332 withheld any model whose paired subset fell under half the
    question set. That is worse than the bug: an excluded model leaves the gate entirely,
    so withholding one can IMPROVE a run's verdict. Here six rows are voided by pairing and
    four SURVIVE it showing a real -100% regression — at 10% call loss, half of
    `UNMEASURED_FAIL_SHARE`. The survivors are fully paired: the strongest evidence the
    harness produces, not a degraded remnant.

    Under the floor this rendered `**PASS** ... safe to enable proxy --diff`, off the other
    model alone. The rows that survive pairing must always be scored."""
    bad = [{
        "qid": f"b{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": 0,
        # First six lose a trial and are voided; last four are complete and score 0/3.
        "diff_trials": TRIALS - 1 if i < 6 else TRIALS,
        "attempts": TRIALS * 2, "fails": 1 if i < 6 else 0,
    } for i in range(10)]
    lost = sum(r["fails"] for r in bad) / sum(r["attempts"] for r in bad)
    assert lost < UNMEASURED_FAIL_SHARE, f"pooled loss {lost:.1%} trips `_unmeasured` alone"

    g = arm_gap(bad, "diff_ok", "terse_ok")
    assert g.excluded is None, "a demonstrated regression must never be withheld"
    assert len(g.rows) == 4
    assert g.form_acc == 0.0 and g.control_acc == 1.0

    # End to end, alongside a healthy model: the exclusion must not be able to rescue it.
    md = build_diff_report({"good": _clean_rows(), "bad": bad})
    assert "**FAIL**" in md
    assert "safe to enable" not in md


def test_a_small_but_failing_arm_still_publishes_its_FAIL():
    """The asymmetry, on the side that makes #334's floor safe (see `_MIN_PAIRED_QUESTIONS`).

    #332's first attempt was a SYMMETRIC survival floor and was measured turning a
    demonstrated -100% regression into "safe to enable `proxy --diff`", because an exclusion
    drops a model from the gate entirely. #334's floor may only ever withhold a form arm
    that is NOT behind its control, so the removed gap is always non-negative and cannot be
    the worst case. Two paired questions, both lost by the form arm, must still FAIL."""
    # Eight questions voided by pairing; the two that survive are complete on BOTH arms and
    # the form arm got neither of them right. Written out rather than reusing
    # `_pairing_wipeout_rows`, whose survivors are clean by construction.
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "diff_ok": 0, "diff_trials": TRIALS - 1 if i < 8 else TRIALS,
        "attempts": TRIALS * 2, "fails": 1 if i < 8 else 0,
    } for i in range(10)]
    assert len(paired_rows(rows, "diff_ok", "terse_ok")) == 2 < _MIN_PAIRED_QUESTIONS

    g = arm_gap(rows, "diff_ok", "terse_ok")
    assert g.excluded is None, "a form arm behind its control publishes at any question count"
    assert g.form_acc < g.control_acc

    md = build_diff_report({"m": rows})
    assert "**FAIL**" in md
    assert "safe to enable" not in md


def test_a_small_and_passing_arm_is_withheld_as_underpowered():
    """The other side: the same two questions, both arms perfect. Nothing failed, and that
    is exactly the problem — two questions cannot support "no regression". Withheld under
    its OWN reason, not `"unmeasured"`, whose label would claim calls were lost."""
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "diff_ok": TRIALS, "diff_trials": TRIALS - 1 if i < 8 else TRIALS,
        "attempts": TRIALS * 2, "fails": 1 if i < 8 else 0,
    } for i in range(10)]
    assert len(paired_rows(rows, "diff_ok", "terse_ok")) == 2 < _MIN_PAIRED_QUESTIONS

    g = arm_gap(rows, "diff_ok", "terse_ok")
    assert g.excluded == "underpowered"

    md = build_diff_report({"m": rows})
    assert "**PASS**" not in md and "safe to enable" not in md
    # Withheld is not the same as unmentioned, and the reason must not blame the backend.
    assert "`m`" in md
    assert "too few calls to compare" not in md, (
        "no calls were lost here; the transport wording would be a fabricated cause")


def test_the_canonical_correlated_loss_fixture_is_still_scored():
    """The rest of this file asserts a FAIL reaches the reader for `_correlated_loss_rows`.
    That only means anything while the fixture is still SCORED — a gate that withheld it
    would turn fifteen tests green for the wrong reason."""
    rows = _correlated_loss_rows()
    assert paired_rows(rows, "diff_ok", "terse_ok")
    assert arm_gap(rows, "diff_ok", "terse_ok").excluded is None


def test_a_withheld_model_is_not_told_its_backend_was_unreachable():
    """The reason string is shared, but its WORDING has to be true of both causes.

    `test_every_renderer_describes_an_unanswered_call_the_same_way` reads
    `REASON_LABEL["unmeasured"]` out of the source, so it pins that the renderers AGREE and
    is silent on whether they are right — reverting the label to its pre-#332 wording
    ("calls went unanswered") leaves it green. That is a real mutation this file has to
    catch, because the second cause reaches a reader whose backend answered almost
    everything: here 16.7% of calls are lost and the losses simply land so that no question
    survives on both arms.

    So this asserts the CLAIM, not the vocabulary: nothing may state that the calls went
    unanswered as the settled cause, and the pairing cause must be offered."""
    rows = _pairing_wipeout_rows()
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE

    # The shared label is the html and terminal renderers' ENTIRE explanation — they carry
    # no second sentence to hedge in — so its wording has to hold for both causes on its
    # own. Asserted against the constant directly: going through rendered output cannot
    # distinguish this, because the markdown's (legitimate) hedge "either too many calls
    # went unanswered, or..." contains the bad phrase as a substring.
    from terse.report import REASON_LABEL
    assert REASON_LABEL["unmeasured"] == "too few calls to compare", (
        f"the shared label is {REASON_LABEL['unmeasured']!r}; the renderers spell this "
        f"phrase literally in their own tests, so changing it here alone splits them")
    assert "unanswered" not in REASON_LABEL["unmeasured"], (
        f"REASON_LABEL['unmeasured'] is {REASON_LABEL['unmeasured']!r}, which asserts a "
        f"cause that is false whenever the arms merely failed to pair")

    md = build_diff_report({"m": rows})
    html = build_html_diff_report({"m": rows}, "diff-form", "full-terse")
    term = build_terminal_diff_report({"m": rows}, color=False)
    for name, text in (("markdown", md), ("html", html), ("terminal", term)):
        assert "too many calls went unanswered (" not in text, (
            f"the {name} renderer states an unreachable backend as the settled cause for a "
            f"model whose backend answered {1 - lost:.0%} of its calls")
    # The remedy must not send the reader at a backend that is working.
    assert "Fix the backend(s) and re-run" not in md
    # ...and the cause that DID apply has to be named where the reader is looking.
    assert "no question completed every trial on BOTH arms" in md


def test_the_fluency_verdict_does_not_assert_a_dead_backend():
    """The seventh renderer. `REASON_LABEL`'s note claims a seventh phrasing is impossible
    and cites `test_every_renderer_names_the_right_exclusion_reason` as the loop that makes
    it so — that test does not exist anywhere in the repo, which is how
    `build_fluency_report`'s verdict bullet kept hardcoding "calls went unanswered" through
    the #332 sweep that hedged every other site.

    Here the backend answers 88.9% of its calls; the one it loses per question is a
    `primer` trial, so `paired_rows` voids every row and the model is withheld. The
    document must not contain a bullet asserting its calls went unanswered."""
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "primer_ok": 0, "primer_trials": TRIALS - 1,
        "attempts": TRIALS * 3, "fails": 1,
    } for i in range(10)]
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE, f"pooled loss {lost:.1%} trips `_unmeasured` alone"
    assert paired_rows(rows, "terse_ok", "primer_ok", "raw_ok") == []

    md = build_fluency_report({"m": rows}, [])
    assert "`m`" in md, "the withheld model must still be named"
    assert "Excluded (calls went unanswered" not in md, (
        f"the fluency verdict asserts an unreachable backend for a model whose backend "
        f"answered {1 - lost:.1%} of its calls")


# --------------------------------------------------------------------------------------
# Site 7 — `"underpowered"` through every consumer (#334).
#
# The first version of this reason was exercised end to end through `build_diff_report`
# ALONE, and four separate defects lived in the renderers it did not reach: the soak
# relabelled it as a pairing failure, the fluency verdict never named the model at all and
# then blamed the corpus, and the fluency paragraph printed a dead-constant "0 paired
# question(s)". One renderer's worth of coverage is what let a shared reason drift again.
# --------------------------------------------------------------------------------------

def _underpowered_rows(n: int = 10) -> list[dict]:
    """Fully paired, nothing lost, both arms perfect — just too few questions."""
    return [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS, "diff_ok": TRIALS, "diff_trials": TRIALS,
        "attempts": TRIALS * 2, "fails": 0,
    } for i in range(n)]


def test_an_underpowered_model_is_named_in_every_renderer():
    rows = _underpowered_rows()
    assert len(paired_rows(rows, "diff_ok", "terse_ok")) == 10 < _MIN_PAIRED_QUESTIONS
    rendered = {
        "markdown": build_diff_report({"m": rows}),
        "html": build_html_diff_report({"m": rows}, "diff-form", "full-terse"),
        "terminal": build_terminal_diff_report({"m": rows}, color=False),
    }
    for name, text in rendered.items():
        assert "m" in text, name
        assert "✓ PASS" not in text and "**PASS**" not in text, name
        # Nothing was lost. No renderer may say otherwise.
        assert "too few calls to compare" not in text, name
        assert "went unanswered" not in text, name


def test_the_fluency_verdict_names_an_underpowered_model_and_counts_its_questions():
    """Findings 3 and 4 of #334's review, in one fixture: the count was a dead constant 0
    (read from `g.rows`, which `_gap` empties on every exclusion), and the model appeared
    in no exclusion line — so a run of fully-answered, fully-paired questions reported
    "did the corpus generate questions?" about a corpus that generated fifteen."""
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "raw_ok": TRIALS, "raw_trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "primer_ok": TRIALS, "primer_trials": TRIALS,
        "attempts": TRIALS * 3, "fails": 0,
    } for i in range(15)]
    md = build_fluency_report({"m": rows}, [])
    assert "15 paired question(s)" in md, (
        "the paragraph's only actionable content is how many questions are short")
    assert "0 paired question(s)" not in md
    assert "did the corpus generate questions?" not in md, (
        "the corpus generated 15 questions and every one of them was answered")
    assert "not concluded" in md.lower()


def test_the_soak_reports_a_deepest_depth_FAIL_even_when_the_pooled_gap_is_withheld():
    """The severest finding of #334's review, and the one the asymmetry proof missed.

    The proof covers `_worst_case_gap`. The deepest-depth analysis is a SECOND verdict, and
    it was nested inside `if worst:` — the pooled result — so withholding the pooled gap
    silently skipped it. #334 made that state reachable for a model that lost zero calls.

    Depth 1 runs +50 (chain ahead), depth 5 runs -100 (chain collapsed). Pooled they cancel
    to exactly 0 over 15 paired questions, which is under the floor and not behind control,
    so the pooled gap is withheld — and the -100% collapse at the depth a soak exists to
    probe must still reach the reader."""
    rows = ([{"qid": f"d1q{i}", "qtype": "lookup", "transform": "table", "depth": 1,
              "trials": 1, "terse_ok": 1 if i < 4 else 0, "terse_trials": 1,
              "diff_ok": 1 if i < 9 else 0, "diff_trials": 1,
              "attempts": 2, "fails": 0} for i in range(10)]
            + [{"qid": f"d5q{i}", "qtype": "lookup", "transform": "table", "depth": 5,
                "trials": 1, "terse_ok": 1, "terse_trials": 1,
                "diff_ok": 0, "diff_trials": 1,
                "attempts": 2, "fails": 0} for i in range(5)])
    assert arm_gap(rows, "diff_ok", "terse_ok").excluded == "underpowered", (
        "fixture must exercise the withheld-pooled path, or this proves nothing")

    md = build_diff_soak_report({"m": rows})
    assert "deepest tested depth (5)" in md, f"the deepest-depth verdict vanished:\n{md}"
    assert "-100%" in md
    assert "**FAIL**" in md
    assert "No depth-correlated comprehension drift" not in md


def test_a_depth_slice_that_paired_cleanly_is_not_called_unpaired():
    """The by-depth table substituted one of two hardcoded reasons for whatever `_gap`
    decided, so `"underpowered"` rendered as "one arm did not complete enough of the same
    questions" — about a slice where both arms completed every question."""
    rows = [{"qid": f"d{d}q{i}", "qtype": "lookup", "transform": "table", "depth": d,
             "trials": 1, "terse_ok": 1, "terse_trials": 1, "diff_ok": 1, "diff_trials": 1,
             "attempts": 2, "fails": 0}
            for d in (1, 2) for i in range(25 if d == 1 else 10)]
    md = build_diff_soak_report({"m": rows})
    assert "one arm did not complete enough of the same questions" not in md, (
        "every arm completed every question at both depths")
    assert "too many calls went unanswered" not in md


def test_an_underpowered_models_rows_still_pool_into_the_per_transform_table():
    """An exclusion moved a published number in the flattering direction: dropping the
    underpowered model took its bad `table` rows out of the pooled average with it, and the
    verdict tells the reader to use that table to 'restrict the policy to the transforms
    that held'. The per-model CONCLUSION is unsupported; the rows are fully paired."""
    big = [{"qid": f"a{i}", "qtype": "lookup", "transform": "table", "trials": 1,
            "raw_ok": 1, "raw_trials": 1, "terse_ok": 1, "terse_trials": 1,
            "primer_ok": 1, "primer_trials": 1, "attempts": 3, "fails": 0}
           for i in range(30)]
    small = [{"qid": f"b{i}", "qtype": "lookup", "transform": "table", "trials": 1,
              "raw_ok": 1, "raw_trials": 1, "terse_ok": 0, "terse_trials": 1,
              "primer_ok": 1, "primer_trials": 1, "attempts": 3, "fails": 0}
             for i in range(10)]
    md = build_fluency_report({"A": big, "B": small}, [])
    section = md.split("by stressed transform")[1].split("## Verdict")[0]
    assert "| table | 40 |" in section, (
        f"B's 10 paired rows must still pool; excluding them flatters the figure:\n{section}")


def test_a_gap_inside_tolerance_is_withheld_not_published_as_a_PASS():
    """The band between "behind its control" and "failing" — where #334's first cut leaked.

    The floor withheld only `best >= cacc`, but the verdict PASSES at
    `gap >= -_GAP_TOLERANCE`. A gap of -3% is therefore behind its control (so the floor
    let it through) AND inside tolerance (so the verdict passed it) — a green PASS off one
    paired question, which is the exact symptom #334 was filed about. Reverting the cut to
    exact equality must redden this test; it is the only thing pinning that boundary.

    One question, 100 trials, 97/100 against 100/100. `_form_stats` clusters on the
    QUESTION, so a hundred trials of one question is still n=1 evidence."""
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": 100,
        "terse_ok": 100, "terse_trials": 100,
        "diff_ok": 97 if i == 9 else 0, "diff_trials": 100 if i == 9 else 99,
        "attempts": 200, "fails": 0 if i == 9 else 1,
    } for i in range(10)]
    pr = paired_rows(rows, "diff_ok", "terse_ok")
    assert len(pr) == 1
    assert _form_stats(pr, "diff_ok")[0] == pytest.approx(0.97)
    assert _form_stats(pr, "terse_ok")[0] == pytest.approx(1.0)
    # Behind its control, but by less than tolerance — so nothing else would withhold it.
    assert -_GAP_TOLERANCE < 0.97 - 1.0 < 0

    assert arm_gap(rows, "diff_ok", "terse_ok").excluded == "underpowered"
    md = build_diff_report({"m": rows})
    assert "**PASS**" not in md, f"a -3% gap off ONE question published as a PASS:\n{md}"
    assert "safe to enable" not in md
