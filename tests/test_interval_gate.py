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


def _nineteen_of_twenty():
    return {"m": [_q(i, 0 if i == 0 else 1, 1) for i in range(20)]}


def test_chart_badges_agree_with_the_markdown_verdict():
    # Review of #582: the forest plots kept badging PASS from the point estimate.
    from terse.html_report import build_html_diff_report
    from terse.terminal_report import build_terminal_diff_report
    results = _nineteen_of_twenty()
    assert "**INCONCLUSIVE**" in build_diff_report(results)
    plot = build_terminal_diff_report(results, color=False)
    assert "INCONCLUSIVE" in plot and "PASS" not in plot
    html = build_html_diff_report(results)
    assert ">PASS</text>" not in html and ">INCONCLUSIVE</text>" in html


def test_a_gap_inside_tolerance_with_a_tight_interval_reaches_pass():
    # 12 misses in 400 questions: gap -3%, interval about ±1.7 pts, clear of -5%.
    md = build_diff_report({"m": [_q(i, 0 if i < 12 else 1, 1) for i in range(400)]})
    assert "**PASS**" in md and "safe to enable" in md


def test_paired_se_counts_a_row_only_one_arm_scored():
    # A legacy row with no `attempts` key can pair while one arm has zero trials;
    # dropping it from the SE made the interval cover fewer questions than the gap.
    rows = [_q(i, 1, 1) for i in range(20)]
    rows.append({**_q(99, 0, 0), "terse_trials": 0})
    assert _paired_gap_se(rows, "diff_ok", "terse_ok") > 0


def test_an_arm_scored_on_one_row_falls_back_to_the_independent_bound():
    # Review round 2 of #582: a one-row control had residual 0 and published ±0 PASS.
    rows = [{**_q(i, 23, 0 if i else 24, trials=25), "diff_trials": 25,
             "terse_trials": 25 if i == 0 else 0} for i in range(20)]
    assert _paired_gap_se(rows, "diff_ok", "terse_ok") is None
    md = build_diff_report({"m": rows})
    assert "±0 pts" not in md and "safe to enable" not in md


def test_a_model_whose_deepest_slice_is_withheld_blocks_the_soak_pass():
    from terse.report import _unmeasured, build_diff_soak_report

    def clean(depth):
        return [{**_q(f"{depth}-{i}", 1, 1), "depth": depth} for i in range(20)]

    dead = [{**_q(f"5-{i}", 0, 0), "depth": 5, "fails": 2, "attempts": 2,
             "diff_trials": 0, "terse_trials": 0} for i in range(5)]
    assert _unmeasured(dead)
    results = {"good": [r for d in range(1, 6) for r in clean(d)],
               "bad": [r for d in range(1, 5) for r in clean(d)] + dead}
    md = build_diff_soak_report(results)
    assert "No depth-correlated comprehension drift" not in md
    assert "**PASS** at" not in md.split("deepest tested depth", 1)[1].split("\n", 1)[0]
