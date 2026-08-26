"""The three numbers that decide whether terse ships are pinned to their VALUES (#337).

Every other guard in this suite is written to survive a change to these constants, and
that is correct: a test that hardcodes `0.05` where the source says `_GAP_TOLERANCE`
breaks on a deliberate re-calibration and teaches the next reader to update it without
thinking. So the suite reads them from the source — and a mutation pass found the cost.
`_GAP_TOLERANCE = 0.05` -> `0.06`, `_CODEC_MIN_TRIALS = 20` -> `19`, and `_unmeasured`'s
`>` -> `>=` each left all 1706 tests green. The ship threshold could drift by a fifth and
nothing would say so.

The fix is not to hardcode the constants everywhere. It is to hardcode them in exactly
ONE place — here — so a change to a ship policy is an edit to a file whose whole purpose
is to make that edit visible in a diff, next to the argument for the current value.

This mirrors what the suite already does for the shared PROSE:
`test_a_withheld_model_is_not_told_its_backend_was_unreachable` asserts
`REASON_LABEL["unmeasured"] == "too few calls to compare"` literally, because a
source-read assertion "pins that the renderers AGREE and is silent on whether they are
right". The numeric policy never got the literal counterpart. It has one now.

Each constant gets a PAIR: the literal value, and a behavioural test at the boundary that
says what the number buys. The literal alone would be ceremony — trivially true, and
mechanically updatable by anyone who reddens it. The boundary test is what makes the
literal mean something: it fails in the same commit, and it fails by rendering a verdict
that changed.

WHAT THIS DOES NOT CLAIM. It does not argue that any of these values is RIGHT. Each is a
judgement call with its rationale written beside its definition in `report.py`
(`_CODEC_MIN_TRIALS`'s says outright that it "wants explicit sign-off before it is trusted
at scale"). This file only ensures that changing one is a decision rather than an
accident.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from terse.report import (
    _CODEC_MIN_TRIALS,
    _GAP_TOLERANCE,
    UNMEASURED_FAIL_SHARE,
    _unmeasured,
    arm_gap,
    build_diff_report,
    codec_verdict,
)

# 25 questions, 8 trials each: above #334's 20-paired-question floor, so the verdict below
# is decided by the tolerance and nothing else. 200 trials divides cleanly into the two
# accuracies this file needs to bracket 5%.
QUESTIONS, TRIALS = 25, 8


def _diff_rows(misses: int) -> list[dict]:
    """25 fully-paired questions; the diff arm loses one trial on `misses` of them.

    The control arm is perfect, so the gap is exactly `-misses / 200`. Losses are spread
    one-per-question rather than concentrated because `_form_stats` clusters its SE on the
    question — piling every miss into one question would leave the gap unchanged but widen
    `gap_ci`, and `gap_ci` is rendered in the same line this test reads.
    """
    return [{
        "qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": TRIALS,
        "terse_ok": TRIALS, "terse_trials": TRIALS,
        "diff_ok": TRIALS - 1 if i < misses else TRIALS, "diff_trials": TRIALS,
        "attempts": TRIALS * 2, "fails": 0,
    } for i in range(QUESTIONS)]


def test_the_ship_tolerance_is_five_points():
    """`_GAP_TOLERANCE` is the number behind "safe to enable `proxy --diff`"."""
    assert _GAP_TOLERANCE == 0.05, (
        f"the ship tolerance is {_GAP_TOLERANCE}, not 0.05. This is the gap at which a "
        f"comprehension regression stops blocking a release — if the change is intended, "
        f"update this test and say why in the commit. Drift past half a point reddens "
        f"the boundary tests below too; anything smaller reaches only this line, which "
        f"is why the literal exists")


def test_a_gap_inside_the_ship_tolerance_publishes_a_PASS():
    """-4.5%: inside 5%, outside 4%. Reddens if the tolerance is tightened."""
    rows = _diff_rows(9)
    g = arm_gap(rows, "diff_ok", "terse_ok")
    # Fixture integrity, asserted before the verdict so a broken fixture reads as a
    # broken fixture rather than as a tolerance regression. The gap sits half a point
    # INSIDE the line rather than on it: an exactly--5% gap is decided by
    # `passes_tolerance`'s 1e-9 epsilon, which `test_gap_gate_boundary.py` and #332 own.
    # This pair brackets the tolerance to +/-0.005, so it catches a re-calibration and
    # leaves the epsilon's boundary to the tests written for it.
    assert g.excluded is None
    assert g.form_acc - g.control_acc == pytest.approx(-0.045)

    md = build_diff_report({"m": rows})
    assert "**PASS**" in md, f"a -4.5% gap did not pass a 5% tolerance:\n{md}"


def test_a_gap_outside_the_ship_tolerance_publishes_a_FAIL():
    """-5.5%: outside 5%, inside 6%. Reddens if the tolerance is loosened.

    This is the direction that matters. A loosened tolerance does not break anything
    visibly — it publishes PASS on runs that used to FAIL, which is the failure mode
    nobody notices until a regression ships."""
    rows = _diff_rows(11)
    g = arm_gap(rows, "diff_ok", "terse_ok")
    assert g.excluded is None
    assert g.form_acc - g.control_acc == pytest.approx(-0.055)

    md = build_diff_report({"m": rows})
    assert "**FAIL**" in md, f"a -5.5% gap passed a 5% tolerance:\n{md}"
    assert "**PASS**" not in md, f"a -5.5% gap published a PASS somewhere:\n{md}"


def _codec_row(n: int) -> list[dict]:
    """One question, `n` trials, both arms perfect — zero demonstrated corruption.

    `codec_verdict` counts TRIALS, not questions (`min_paired=0` opts out of #334's
    question floor), so this is the shape its sample-size gate actually sees."""
    return [{"qid": "q0", "qtype": "lookup", "transform": "table", "trials": n,
             "raw_ok": n, "raw_trials": n, "terse_ok": n, "terse_trials": n,
             "attempts": n * 2, "fails": 0}]


def test_the_codec_trial_floor_is_twenty():
    """`_CODEC_MIN_TRIALS` is how many zero-failure trials buy a SAFE."""
    assert _CODEC_MIN_TRIALS == 20, (
        f"the codec trial floor is {_CODEC_MIN_TRIALS}, not 20. Its definition calls it "
        f"'the single most contestable number in this module' and asks for explicit "
        f"sign-off before it is trusted at scale — this test is that sign-off")



def test_the_quoted_clopper_pearson_bound_still_computes():
    """The constant's comment quotes the bound its value buys. This reads that number
    back out of the source and recomputes it.

    An earlier draft asserted `1 - 0.05 ** (1 / _CODEC_MIN_TRIALS) == approx(0.139)`
    inside the test above, which was ceremony twice over: the literal assert eight lines
    earlier aborts first, so it never ran on the mutation it claimed to guard, and with
    the constant pinned it is an arithmetic identity over a literal. Neither half touched
    the COMMENT, which is the thing that drifts — changing `~14pp` to `~99pp` left the
    whole file green.

    So the bound is parsed out of `report.py`. Now a re-calibration that updates the
    constant and forgets the sentence justifying it reddens here."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "terse" / "report.py").read_text()
    quoted = re.search(r"At n=(\d+) that is ~(\d+)pp", src)
    assert quoted, "the Clopper-Pearson sentence next to `_CODEC_MIN_TRIALS` is gone"
    n, pp = int(quoted.group(1)), int(quoted.group(2))
    assert n == _CODEC_MIN_TRIALS, (
        f"the comment reasons about n={n} but the floor is {_CODEC_MIN_TRIALS}")
    assert round((1 - 0.05 ** (1 / n)) * 100) == pp, (
        f"the comment claims ~{pp}pp at n={n}; Clopper-Pearson gives "
        f"{(1 - 0.05 ** (1 / n)) * 100:.1f}pp")


def test_nineteen_zero_failure_trials_are_UNRESOLVED():
    """19 and 20 are written out, NOT as `_CODEC_MIN_TRIALS - 1` / `_CODEC_MIN_TRIALS`.

    Spelling them relative to the constant is what makes a boundary test useless as a pin:
    the fixture slides with the value it is supposed to hold still. Verified by mutation —
    the relative form left both of these green at a floor of 19 and at 21.

    Together the pair also fixes the comparison as INCLUSIVE (`n >= _CODEC_MIN_TRIALS`):
    a `>` would turn the 20-trial case below UNRESOLVED."""
    assert codec_verdict(_codec_row(19))[0] == "UNRESOLVED"


def test_twenty_zero_failure_trials_are_SAFE():
    """Why the existing SAFE coverage did not stop the floor from moving.

    `tests/test_codec_verdict.py` has two. `test_identical_partial_failure_on_both_arms_
    at_the_trial_floor_is_SAFE` is one row at 25 trials, which clears a floor of 19, 20 or
    21 alike. `test_zero_failures_at_exactly_the_trial_floor_is_SAFE` sits exactly ON the
    floor — and is spelled `range(_CODEC_MIN_TRIALS)`, so the fixture slid with the
    mutation and stayed green. That one is the more direct reason 20 -> 19 survived, and
    it is the anti-pattern this module's docstring describes, living in the file this test
    is meant to reinforce.

    The inclusive `>=` is NOT this test's contribution: mutating `n >= _CODEC_MIN_TRIALS`
    to `n >` reddens both `test_codec_verdict.py` tests as well. What is new here is the
    floor's VALUE, held by a literal 20 that cannot slide."""
    assert codec_verdict(_codec_row(20))[0] == "SAFE"


def test_the_unmeasured_loss_share_is_twenty_percent():
    assert UNMEASURED_FAIL_SHARE == 0.20, (
        f"the transport-failure share is {UNMEASURED_FAIL_SHARE}, not 0.20 — the point at "
        f"which a model's numbers stop being published at all")


def test_a_model_exactly_at_the_loss_share_is_still_measured():
    """The comparison is strictly `>`, and that is deliberate — pinning the OPERATOR.

    The threshold is deliberately permissive: a failed call is already excluded from its
    arm's denominator, so voiding a whole multi-hour run over a handful of transient 429s
    would be its own kind of wrong answer. `paired_rows`, not this threshold, is the
    control that catches arm-correlated loss.

    The fixture is built to land the RATIO exactly on the line, which is the only thing
    the operator can be observed through. It is deliberately not the scenario
    `paired_rows`' docstring describes — that one does not reach the boundary at all,
    because `_unmeasured` divides by total `attempts` across every arm while the docstring
    reasons about one arm's own calls. Filed separately; see the note beside
    `UNMEASURED_FAIL_SHARE` in `report.py`.

    So this test does not assert the boundary is in the right place. It asserts that which
    side of it a model falls on is observed by something, because `>` -> `>=` survived a
    full-suite mutation of all 1706 tests.
    """
    # Internally coherent, not just arithmetically convenient: each row runs 10 trials on
    # each of two arms (20 attempts), loses 2 per arm, and reports the surviving counts.
    # A row whose `attempts` contradicts its `*_trials` would still exercise `_unmeasured`
    # — it reads only the totals — but it could not occur, and a fixture that cannot occur
    # is how a boundary test ends up pinning arithmetic rather than behaviour.
    on_the_line = [{"qid": f"q{i}", "trials": 10, "attempts": 20, "fails": 4,
                    "terse_ok": 8, "terse_trials": 8,
                    "diff_ok": 8, "diff_trials": 8} for i in range(5)]
    assert sum(r["fails"] for r in on_the_line) / sum(
        r["attempts"] for r in on_the_line) == UNMEASURED_FAIL_SHARE
    assert not _unmeasured(on_the_line), (
        "a model at exactly UNMEASURED_FAIL_SHARE was withheld; the comparison in "
        "`_unmeasured` is documented as strictly `>` and `paired_rows`' docstring "
        "reasons from that")

    # One more lost call on the last row: 21/100, coherent the same way.
    over_the_line = [*on_the_line[:-1],
                     {**on_the_line[-1], "fails": 5, "diff_ok": 7, "diff_trials": 7}]
    assert sum(r["fails"] for r in over_the_line) / sum(
        r["attempts"] for r in over_the_line) > UNMEASURED_FAIL_SHARE
    assert _unmeasured(over_the_line), (
        "a model past UNMEASURED_FAIL_SHARE was still published")
