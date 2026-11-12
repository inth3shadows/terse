"""#582 item 7: a comprehension PASS needs its interval, not just its point estimate,
to clear the tolerance; the interval is the PAIRED SE of the per-question difference;
and a withheld model blocks PASS."""
from __future__ import annotations

import pytest

from terse.report import (
    _paired_gap_se,
    _worst_case_gap,
    build_diff_report,
    gate_word,
)


def _q(i, form, control, trials=1):
    return {"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
            "trials": trials, "diff_ok": form, "terse_ok": control}


def test_identical_arms_have_zero_paired_se():
    rows = [_q(i, i % 2, i % 2) for i in range(24)]    # 50% on both, same questions
    assert _paired_gap_se(rows, "diff_ok", "terse_ok") == pytest.approx(0.0)


def test_paired_se_is_positive_when_arms_disagree():
    rows = [_q(i, 0 if i == 0 else 1, 1) for i in range(20)]
    assert _paired_gap_se(rows, "diff_ok", "terse_ok") > 0


def test_point_inside_but_interval_outside_is_inconclusive():
    v = _worst_case_gap({"m": (0.95, 0.05, 1.0, 0.0, 0.05)})
    assert v is not None and v.inconclusive and not v.passed
    assert gate_word(v) == "INCONCLUSIVE"


def test_interval_inside_passes_and_point_outside_fails():
    assert _worst_case_gap({"m": (0.99, 0.0, 1.0, 0.0, 0.005)}).passed
    v = _worst_case_gap({"m": (0.90, 0.0, 1.0, 0.0, 0.001)})
    assert not v.passed and not v.inconclusive and gate_word(v) == "FAIL"


def test_inconclusive_reports_the_model_whose_interval_reaches_furthest():
    v = _worst_case_gap({"tight": (0.97, 0, 1.0, 0, 0.001), "wide": (0.99, 0, 1.0, 0, 0.05)})
    assert v.inconclusive and v.model == "wide"


def test_any_fail_outranks_an_inconclusive_model():
    v = _worst_case_gap({"bad": (0.80, 0, 1.0, 0, 0.001), "wide": (0.99, 0, 1.0, 0, 0.05)})
    assert gate_word(v) == "FAIL" and v.model == "bad"


def test_withheld_blocks_pass_but_never_a_fail():
    ok = _worst_case_gap({"m": (1.0, 0, 1.0, 0, 0.0)})
    assert gate_word(ok, withheld=True) == "INCONCLUSIVE"
    bad = _worst_case_gap({"m": (0.5, 0, 1.0, 0, 0.0)})
    assert gate_word(bad, withheld=True) == "FAIL"


def test_diff_report_one_miss_in_twenty_is_not_safe_to_enable():
    # The audit's case: 19/20 vs 20/20 printed PASS "safe to enable" at ±10 pts.
    md = build_diff_report({"m": [_q(i, 0 if i == 0 else 1, 1) for i in range(20)]})
    assert "**INCONCLUSIVE**" in md and "safe to enable" not in md


def test_diff_report_identical_arms_pass():
    md = build_diff_report({"m": [_q(i, 1, 1) for i in range(20)]})
    assert "**PASS**" in md and "safe to enable" in md


def test_html_banner_shows_inconclusive_as_a_warning():
    from terse.html_report import build_html_diff_report
    html = build_html_diff_report({"m": [_q(i, 0 if i == 0 else 1, 1) for i in range(20)]})
    assert 'class="banner warn"' in html and "INCONCLUSIVE" in html
