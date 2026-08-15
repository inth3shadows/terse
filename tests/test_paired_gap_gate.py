"""A correlated loss cannot publish a PASS — at every renderer (#280).

The bug: when a model's lost calls correlate with the arm under test (a token-budget stop
kills the LONGEST prompt first, and the diff/terse arm's prompt is strictly longer than its
control's), `_form_stats` divides each arm by its OWN surviving trial count, so the arm that
lost the hard questions is scored over an easier question set than its control. The gap then
compares two different exams and flatters the arm being tested.

WHY THE FIXTURE LOOKS THE WAY IT DOES — this is the part three previous attempts got wrong.

Two gates already existed before pairing, and a fixture that trips either of them proves
nothing about pairing: the report would refuse to publish anyway, the test would pass, and
the pairing could be deleted without going red. That is exactly what happened to the last
attempt's headline test, which passed unmodified against the un-fixed code.

So `_correlated_loss_rows` is built to sit UNDER both:

  - pooled loss `_unmeasured`: 3 lost calls of 36 = 8.3%, under `UNMEASURED_FAIL_SHARE`
    (0.20, strictly `>`);
  - `unpaired` refusal: 1 unusable row of 6 = 16.7%, under the same 0.20 (`>=`).

`test_the_fixture_stays_under_both_pre_existing_gates` asserts both margins directly, so a
later edit cannot quietly re-base these tests onto a gate they are not about.

What is left is pure pairing, and the numbers are unambiguous:

    unpaired  form 80.0%  control 83.3%  gap  -3.3%  -> PASS at 5% tolerance
    paired    form 80.0%  control 100.0% gap -20.0%  -> FAIL

Every test below asserts the FAIL reaches the reader. Each one is verified by reverting its
own site's wiring and confirming THIS test goes red — see the PR body for the seven results.
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
    unpaired,
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

def test_the_fixture_stays_under_both_pre_existing_gates():
    rows = _correlated_loss_rows()
    lost = sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows)
    assert lost < UNMEASURED_FAIL_SHARE, (
        f"pooled loss {lost:.1%} would trip `_unmeasured` on its own, so these tests would "
        f"pass with pairing deleted")
    assert not _unmeasured(rows)

    dropped = (len(rows) - len(paired_rows(rows, "diff_ok", "terse_ok"))) / len(rows)
    assert dropped < UNMEASURED_FAIL_SHARE, (
        f"unpaired share {dropped:.1%} would trip the `unpaired` refusal, which is a "
        f"different mechanism than the pairing these tests are about")
    assert not unpaired(rows, "diff_ok", "terse_ok")


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
    assert not unpaired(rows, "terse_ok", "primer_ok", "raw_ok")

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

def test_rows_without_per_form_counters_are_treated_as_fully_paired():
    """Legacy result files and `score_pack` output carry no `<form>_trials` key at all.

    Reading an absent counter as a loss would void every one of them wholesale, so
    `paired_rows` defaults it to the row's `trials`. Load-bearing and easy to mistake for
    an accident, hence pinned.
    """
    legacy = [{"qid": "a", "qtype": "lookup", "transform": "table", "trials": 1,
               "terse_ok": 1, "diff_ok": 1}]
    assert paired_rows(legacy, "diff_ok", "terse_ok") == legacy
    assert not unpaired(legacy, "diff_ok", "terse_ok")


def test_unpaired_refuses_at_the_boundary_share():
    """`>=`, not `>`. One question type of five is exactly 20.0%, and that is the measured
    case in which a real -20% FAIL published as PASS. On a ship gate the boundary belongs
    on the refusing side."""
    rows = [{"qid": str(i), "trials": 1, "terse_ok": 1, "terse_trials": 1,
             "diff_ok": 1, "diff_trials": 1} for i in range(5)]
    rows[0]["diff_trials"] = 0
    assert (len(rows) - len(paired_rows(rows, "diff_ok", "terse_ok"))) / len(rows) == 0.20
    assert unpaired(rows, "diff_ok", "terse_ok")
