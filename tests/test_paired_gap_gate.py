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
    UNMEASURED_FAIL_SHARE,
    _unmeasured,
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
# Per-row form successes across the five questions BOTH arms answered: 12 of 15 = 80%. The
# control answers all fifteen. A real, ordinary regression — the point is that it is
# invisible until the arms are paired.
FORM_OK = [3, 3, 3, 2, 1]


def _correlated_loss_rows(depth: int | None = None) -> list[dict]:
    """Six questions. The control answers all of them; the form arm loses every trial of
    the one question the control also finds hard, and is genuinely worse on the rest."""
    rows = [{
        # The hard question. The control scores 0 on it — so dropping it from the control's
        # denominator is what inflates the control-relative comparison — and the form arm
        # never answered it at all (`diff_trials` 0).
        "qid": "hard", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "terse_ok": 0, "terse_trials": TRIALS, "diff_ok": 0, "diff_trials": 0,
        "fails": TRIALS, "attempts": TRIALS * 2,
    }]
    for qid, ok in zip("BCDEF", FORM_OK, strict=True):
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
    } for i in range(6)]
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
        "qid": "hard", "qtype": "deref", "transform": "table", "trials": TRIALS,
        "raw_ok": 0, "raw_trials": TRIALS,
        "terse_ok": 0, "terse_trials": 0, "primer_ok": 0, "primer_trials": 0,
        "fails": TRIALS * 2, "attempts": TRIALS * 3,
    }]
    for qid, ok in zip("BCDEF", FORM_OK, strict=True):
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
    } for i in range(5)]
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
    } for i in range(6)]
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
    } for i in range(7)]
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
    } for i in range(5)]
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
    } for i in range(5)]
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
    # 6 of 20 questions lost outright — 3 by each arm — so 14 are comparable.
    rows = [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": 5,
        "terse_ok": 0 if 3 <= i < 6 else 5, "terse_trials": 0 if 3 <= i < 6 else 5,
        "diff_ok": 0 if i < 3 else 5, "diff_trials": 0 if i < 3 else 5,
        "fails": 5 if i < 6 else 0, "attempts": 10,
    } for i in range(20)]
    assert not _unmeasured(rows), "the transport gate must not be what withholds this"
    scored = len(paired_rows(rows, "diff_ok", "terse_ok"))
    assert scored == 14

    md = build_diff_report({"m": rows})
    row = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert row and f"| `m` | {scored} |" in row[0], (
        f"q column should be the paired {scored}, not 20: {row}")