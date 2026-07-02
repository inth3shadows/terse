"""Tests for terminal_report.py — the ANSI bar-chart companion to report.py's markdown."""
from __future__ import annotations

import re

from terse.terminal_report import (
    build_terminal_diff_report,
    build_terminal_fluency_report,
    build_terminal_report,
    diverging_bar_lines,
    forest_bar_lines,
    stacked_bar_lines,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_ROW = {
    "tool": "demo.tool", "shape": "array-of-records", "sha": "abc123",
    "roundtrip_ok": True,
    "cl100k": {"raw": 100, "compressed": 60},
    "saved_cl100k": {"minify": 5, "tabularize": 30, "dictionary": 5},
}


def test_diverging_bar_lines_shows_signed_values_and_labels():
    text = diverging_bar_lines([("wins", 38.1), ("regresses", -12.4)], color=False)
    assert "wins" in text and "+38.1%" in text
    assert "regresses" in text and "-12.4%" in text


def test_diverging_bar_lines_empty():
    assert "no data" in diverging_bar_lines([], color=False)


def test_diverging_bar_lines_no_ansi_when_color_disabled():
    text = diverging_bar_lines([("a", 10.0)], color=False)
    assert not _ANSI.search(text)


def test_diverging_bar_lines_ansi_when_color_enabled():
    text = diverging_bar_lines([("a", 10.0)], color=True)
    assert _ANSI.search(text)


def test_diverging_bar_lines_rows_stay_aligned_with_color_on():
    # Padding must happen BEFORE coloring, or ANSI codes get counted as visible
    # width and rows drift out of alignment. Same label width + same value-string
    # width (both single-digit-percent) should still yield equal-length stripped
    # lines regardless of how much of the bar each row's magnitude fills.
    text = diverging_bar_lines([("small", 1.0), ("big!!", 9.9)], color=True)
    lines = [_ANSI.sub("", ln) for ln in text.splitlines()]
    assert len(lines[0]) == len(lines[1])


def test_stacked_bar_lines_totals_and_legend():
    text = stacked_bar_lines([("array-of-records", [0, 874, 134])],
                              ("minify", "tabularize", "dictionary"), color=False)
    assert "minify" in text and "tabularize" in text and "dictionary" in text
    assert "+1008" in text


def test_stacked_bar_lines_negative_tier_not_hidden():
    text = stacked_bar_lines([("tiny-payload", [-3, 40, 2])],
                              ("minify", "tabularize", "dictionary"), color=False)
    assert "+39" in text  # true signed total, not a clamped geometry


def test_stacked_bar_lines_all_negative_does_not_crash():
    text = stacked_bar_lines([("worst-case", [-1, -2, -3])],
                              ("minify", "tabularize", "dictionary"), color=False)
    assert "-6" in text


def test_stacked_bar_lines_empty():
    assert "no data" in stacked_bar_lines([], ("minify", "tabularize", "dictionary"), color=False)


def test_build_terminal_report_renders_all_three_sections():
    text = build_terminal_report([_ROW], color=False)
    assert "Tier-0 savings by shape bucket" in text
    assert "Tier-0 savings by tool" in text
    assert "Tier attribution by shape" in text
    assert "demo.tool" in text
    assert "array-of-records" in text


def test_build_terminal_report_no_ansi_when_color_disabled():
    text = build_terminal_report([_ROW], color=False)
    assert not _ANSI.search(text)


_FOREST_ROWS = [
    {"model": "a", "form_acc": 0.9, "form_ci": 0.05, "control_acc": 0.92,
     "control_ci": 0.04, "passed": True},
    {"model": "b", "form_acc": 0.5, "form_ci": 0.1, "control_acc": 0.9,
     "control_ci": 0.05, "passed": False},
]


def test_forest_bar_lines_pass_fail_badges():
    text = forest_bar_lines(_FOREST_ROWS, "diff-form", "full-terse", color=False)
    assert "PASS" in text and "FAIL" in text
    assert "a" in text and "b" in text
    assert "diff-form" in text and "full-terse" in text


def test_forest_bar_lines_empty():
    assert "no data" in forest_bar_lines([], "form", "control", color=False)


def test_forest_bar_lines_no_ansi_when_color_disabled():
    text = forest_bar_lines(_FOREST_ROWS, "diff-form", "full-terse", color=False)
    assert not _ANSI.search(text)


def test_forest_bar_lines_ansi_when_color_enabled():
    text = forest_bar_lines(_FOREST_ROWS, "diff-form", "full-terse", color=True)
    assert _ANSI.search(text)


def test_track_marker_position_reflects_accuracy():
    # 0% -> marker at the track's first column; 100% -> marker at the last column;
    # regression-guard the accuracy->column mapping the forest plot renders from.
    from terse.terminal_report import _TRACK_WIDTH, _track

    assert _track(0.0, 0.0, "●")[0] == "●"
    assert _track(1.0, 0.0, "●")[_TRACK_WIDTH] == "●"


def test_track_whisker_spans_the_confidence_interval():
    from terse.terminal_report import _track

    track = _track(0.5, 0.2, "●")
    assert "─" in track  # a nonzero CI draws a whisker, not just the point marker


def test_build_terminal_diff_report_matches_markdown_verdict():
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 1, "terse_ok": 1, "diff_ok": 1} for i in range(10)]
    text = build_terminal_diff_report({"m": rows}, color=False)
    assert "PASS" in text and "FAIL" not in text
    assert "diff-form" in text and "full-terse" in text


def test_build_terminal_diff_report_empty():
    assert "no data" in build_terminal_diff_report({}, color=False)


def test_build_terminal_fluency_report_excludes_broken_model():
    rows_good = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
                  "raw_ok": True, "terse_ok": True, "primer_ok": True} for i in range(20)]
    rows_broken = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
                    "raw_ok": False, "terse_ok": False, "primer_ok": False} for i in range(20)]
    text = build_terminal_fluency_report({"good": rows_good, "broken": rows_broken}, color=False)
    assert "good" in text and "PASS" in text
    assert "excluded" in text and "broken" in text
