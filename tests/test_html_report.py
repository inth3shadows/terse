"""Tests for html_report.py — the charted HTML companion to report.py's markdown."""
from __future__ import annotations

from terse.html_report import (
    build_html_diff_report,
    build_html_report,
    diverging_bar_chart,
    forest_plot,
    stacked_bar_chart,
)

_ROW = {
    "tool": "demo.tool", "shape": "array-of-records", "sha": "abc123",
    "roundtrip_ok": True,
    "cl100k": {"raw": 100, "compressed": 60},
    "saved_cl100k": {"minify": 5, "tabularize": 30, "dictionary": 5},
}
_COVERAGE = {"total": 1, "by_tool": {"demo.tool": 1}, "by_shape": {"array-of-records": 1}}


def test_no_javascript_or_network_anywhere():
    # The whole point of the HTML mode is zero JS / zero CDN — regression-guard it.
    # (the SVG xmlns is a namespace URI, not a network fetch, so it's not checked here)
    html = build_html_report([_ROW], _COVERAGE)
    assert "<script" not in html
    assert "cdn." not in html
    assert "fetch(" not in html


def test_build_html_report_renders_gate_coverage_and_charts():
    html = build_html_report([_ROW], _COVERAGE)
    assert "All 1/1 payloads round-trip losslessly" in html
    assert "demo.tool" in html
    assert "array-of-records" in html
    assert "<svg" in html


def test_build_html_report_gate_failure_shows_invalid_banner():
    bad_row = {**_ROW, "roundtrip_ok": False}
    html = build_html_report([bad_row], _COVERAGE)
    assert "INVALID" in html
    assert "1/1 payloads FAILED" in html


def test_build_html_report_attestation_card():
    html = build_html_report([_ROW], _COVERAGE, attestation=("your captured traffic", 1))
    assert "your captured traffic" in html
    assert "1 payloads" in html


def test_diverging_bar_chart_colors_by_sign():
    svg = diverging_bar_chart([("wins", 38.1), ("regresses", -12.4)])
    assert "var(--diverging-pos)" in svg
    assert "var(--diverging-neg)" in svg
    assert "+38.1%" in svg and "-12.4%" in svg


def test_diverging_bar_chart_empty():
    assert "No data" in diverging_bar_chart([])


def test_stacked_bar_chart_totals_and_legend():
    svg = stacked_bar_chart([("array-of-records", [0, 874, 134])],
                             ("minify", "tabularize", "dictionary"))
    assert "minify" in svg and "tabularize" in svg and "dictionary" in svg
    assert "+1008" in svg  # sum of the three tiers


def test_stacked_bar_chart_negative_tier_not_clamped_to_zero():
    # measure.py: "at tiny N the table envelope can cost more than the keys it
    # folds" — a negative tier must still show up, not silently disappear.
    svg = stacked_bar_chart([("tiny-payload", [-3, 40, 2])],
                             ("minify", "tabularize", "dictionary"))
    assert "(-3)" in svg  # direct label on the negative step, not just a hover title
    assert "minify: -3" in svg  # hover title still carries the exact signed value
    assert "var(--diverging-neg)" in svg  # anomaly outline present
    assert "+39" in svg  # printed total matches sum(vals), not the clamped geometry


def test_stacked_bar_chart_all_negative_tiers():
    # Degenerate case: every tier regressed. Must not divide by zero or crash,
    # and the total must still be the true (negative) sum.
    svg = stacked_bar_chart([("worst-case", [-1, -2, -3])],
                             ("minify", "tabularize", "dictionary"))
    assert "-6" in svg
    assert "(-1)" in svg and "(-2)" in svg and "(-3)" in svg


def test_forest_plot_pass_fail_badges():
    rows = [
        {"model": "a", "form_acc": 0.9, "form_ci": 0.05, "control_acc": 0.92,
         "control_ci": 0.04, "passed": True},
        {"model": "b", "form_acc": 0.5, "form_ci": 0.1, "control_acc": 0.9,
         "control_ci": 0.05, "passed": False},
    ]
    svg = forest_plot(rows, "diff-form", "full-terse")
    assert "PASS" in svg and "FAIL" in svg
    assert svg.count("<circle") >= 4  # 2 models x 2 series


def test_build_html_diff_report_renders_verdict_and_forest():
    # Paired diff-family results (terse_ok = control, diff_ok = form) -> a forest plot
    # + a PASS/FAIL verdict gated on the worst model. This is what `fluency --diff --html`
    # writes; it was built but had no wiring or test until now.
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 1, "terse_ok": 1, "diff_ok": 1} for i in range(10)]
    html = build_html_diff_report({"m": rows}, "diff-form", "full-terse")
    assert "<!doctype html>" in html
    assert "diff-form" in html and "full-terse" in html
    assert "PASS" in html and "<svg" in html
    # no scripts / external fetches (the SVG xmlns is not egress) — matches the
    # no-network contract the main report is held to.
    assert "<script" not in html and "fetch(" not in html
