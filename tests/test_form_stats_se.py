"""`_form_stats`'s SE must measure question-sampling variance, not within-question
consistency (#297) -- and must not trade the old false-certainty failure for a new one.

The pooled binomial SE it used to compute (Σt·p̂(1−p̂)) collapses to 0 whenever every row
is internally consistent (all-right or all-wrong across its own trials) — which is the
common case at temperature 0 — regardless of how much the questions disagreed with each
other. That let two runs at IDENTICAL accuracy report `±0` and `±3`, with the more
volatile one (one question wholly wrong) printing as the more certain one. The fix
clusters on the question instead; these are the two worked examples from the issue,
pinned so a future edit can't silently reintroduce the collapse.

A first cut of the fix (fixed in code review, before this ever shipped) returned a flat
`SE=0` whenever fewer than 2 questions survived -- reintroducing #297's own failure mode
(false certainty on a single draw) through a different door, in the more dangerous
direction (over-, not under-, confidence). `_form_stats` must instead fall back to the
pre-#297 within-question binomial SE for a single surviving question -- the best estimate
available with one cluster, never a confident zero.

Likewise a first cut silently accepted an impossible `k>t` count (`Σ(kᵢ−acc·tᵢ)²` is a sum
of squares and can never go negative, unlike the old `p̂(1−p̂)`, so it can no longer crash
loud on that invariant violation the way the pre-#297 code did). `_form_stats` must raise
rather than silently publish a >100% (or negative) accuracy.
"""

from __future__ import annotations

import math

import pytest

from terse.report import _ci, _form_stats, build_fluency_report


def test_one_wholly_wrong_question_is_not_reported_as_more_certain_than_five_slightly_wrong():
    # 24 questions 5/5 right, 1 question 0/5 -- 96% accuracy, self-consistent rows.
    one_bad_question = [{"terse_ok": 5, "trials": 5}] * 24 + [{"terse_ok": 0, "trials": 5}]
    # 20 questions 5/5, 5 questions 4/5 -- also 96% accuracy, but spread across more rows.
    five_slightly_wrong = [{"terse_ok": 5, "trials": 5}] * 20 + [{"terse_ok": 4, "trials": 5}] * 5

    acc_a, se_a = _form_stats(one_bad_question, "terse_ok")
    acc_b, se_b = _form_stats(five_slightly_wrong, "terse_ok")

    assert acc_a == acc_b == pytest.approx(0.96)
    # The old within-question estimator reported these as ±0.00% and ±3.14% -- backwards.
    assert _ci(se_a) * 100 == pytest.approx(7.84, abs=0.01)
    assert _ci(se_b) * 100 == pytest.approx(3.20, abs=0.01)
    assert se_a > se_b, "one wholly-wrong question is MORE uncertain, not less"


def test_a_single_surviving_question_falls_back_to_the_within_question_se_not_a_confident_zero():
    acc, se = _form_stats([{"terse_ok": 3, "trials": 5}], "terse_ok")
    assert acc == pytest.approx(0.6)
    # The pre-#297 formula's value for this exact row: sqrt(t*p*(1-p))/t.
    assert se == pytest.approx(math.sqrt(5 * 0.6 * 0.4) / 5)
    assert se > 0, "one question answered 3/5 is NOT perfectly certain"


def test_a_single_surviving_question_answered_uniformly_reports_zero_se():
    # p=1 (or p=0) genuinely has zero within-question variance -- this is the one case
    # where a single cluster legitimately reports ±0, not a fallback bug.
    acc, se = _form_stats([{"terse_ok": 5, "trials": 5}], "terse_ok")
    assert acc == 1.0
    assert se == 0.0


def test_perfectly_uniform_rows_still_report_zero_se():
    rows = [{"terse_ok": 4, "trials": 5}] * 10
    acc, se = _form_stats(rows, "terse_ok")
    assert acc == pytest.approx(0.8)
    assert se == 0.0


def test_a_success_count_exceeding_its_trial_count_raises_rather_than_publishing_impossible_accuracy():
    with pytest.raises(ValueError):
        _form_stats([{"terse_ok": 7, "trials": 5}, {"terse_ok": 5, "trials": 5}], "terse_ok")


def test_a_negative_success_count_also_raises():
    with pytest.raises(ValueError):
        _form_stats([{"terse_ok": -1, "trials": 5}, {"terse_ok": 5, "trials": 5}], "terse_ok")


def test_a_success_scored_on_zero_trials_also_raises():
    # A first cut of the k<=t guard was gated on `t > 0`, which let exactly this shape
    # through -- and it is not hypothetical: it is the coordinate the guard's own comment
    # cites (a `<form>_trials = trials - errors` bug scoring a "success" on an errored,
    # zero-remaining-trial row).
    with pytest.raises(ValueError):
        _form_stats([{"terse_ok": 1, "trials": 0}, {"terse_ok": 1, "trials": 5}], "terse_ok")


def test_a_negative_trial_count_raises_rather_than_indexerrors():
    with pytest.raises(ValueError):
        _form_stats([{"terse_ok": 0, "trials": -1}], "terse_ok")


def _lopsided_regression_rows() -> list[dict]:
    """25 questions, trials=1: raw 68% vs terse 52% -- a real 16pt regression whose
    independence-combined gap_ci (±27pts, since #297 made both arm SEs cluster-robust
    rather than ~0) is wide enough to exceed the gap itself. Round-2 review's own repro."""
    rows = []
    for i in range(17):
        ok = 1 if i < 13 else 0
        rows.append({"qid": f"r{i}", "qtype": "lookup", "transform": "table", "trials": 1,
                     "raw_ok": 1, "terse_ok": ok, "primer_ok": ok, "fails": 0, "attempts": 3})
    for i in range(8):
        rows.append({"qid": f"w{i}", "qtype": "lookup", "transform": "table", "trials": 1,
                     "raw_ok": 0, "terse_ok": 0, "primer_ok": 0, "fails": 0, "attempts": 3})
    return rows


def test_a_wide_conservative_gap_ci_never_argues_a_real_fail_is_noise():
    """gap_ci here is only ever an OVER-estimate of the gap's true spread (it discards the
    positive correlation between paired arms), so it can rule a gap "not yet tightly
    measured" but must never rule a genuine FAIL "indistinguishable from noise" -- round-2
    review's repro of exactly that contradiction. Round 2's first fix only gated the FAIL
    side; round 3 removed the "not distinguishable from noise" verdict-language entirely
    (it was still reachable on a PASS, off the same over-loose bound) in favor of wording
    that never claims significance either way."""
    md = build_fluency_report({"m1": _lopsided_regression_rows()}, [])
    assert "**FAIL**" in md
    assert "not distinguishable from noise" not in md
    assert "indistinguishable" not in md
    assert "treat the verdict above as real" in md


def _wide_ci_pass_rows() -> list[dict]:
    """25 questions, trials=5: a real ~5pt regression that PASSES the tolerance gate, whose
    gap_ci (±17pts) is wide enough that the old wording would have called it "not
    distinguishable from noise" -- round-3 review's repro that the PASS side of the same
    branch had the identical bug the FAIL side was fixed for in round 2."""
    raw_terse = [(2, 2), (4, 4), (2, 1), (1, 1), (2, 2), (5, 5), (2, 2), (4, 4), (5, 5),
                 (4, 4), (0, 0), (3, 3), (1, 0), (1, 1), (2, 1), (1, 1), (1, 1), (4, 4),
                 (1, 0), (4, 4), (2, 1), (3, 2), (3, 3), (5, 5), (3, 3)]
    return [{"qid": f"q{i}", "qtype": "lookup", "transform": "table", "trials": 5,
             "raw_ok": r, "terse_ok": t, "primer_ok": t, "fails": 0, "attempts": 15}
            for i, (r, t) in enumerate(raw_terse)]


def test_a_wide_conservative_gap_ci_also_never_argues_a_pass_is_noise():
    md = build_fluency_report({"m1": _wide_ci_pass_rows()}, [])
    assert "**PASS**" in md
    assert "not distinguishable from noise" not in md
    assert "indistinguishable" not in md
    assert "This PASS is smaller than a" in md
