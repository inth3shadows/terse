"""Tests for terminal_report.py — the ANSI bar-chart companion to report.py's markdown."""
from __future__ import annotations

import re

from terse.terminal_report import (
    build_terminal_report,
    diverging_bar_lines,
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
