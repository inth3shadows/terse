"""Tests for report.py's build_trend_report (#51 fast-follow: historical trend across
`measure --history` runs). Scoped to this one addition, not a full backfill of
report.py's existing markdown builders."""
from __future__ import annotations

from terse.report import build_trend_report

_RUN_A = {"ts": "t1", "label": "corpus", "n_payloads": 3, "lossless_pass": 3,
          "raw_tok": 300, "compressed_tok": 180, "saved_tok": 120, "saved_pct": 40.0}
_RUN_B = {"ts": "t2", "label": "corpus", "n_payloads": 4, "lossless_pass": 4,
          "raw_tok": 400, "compressed_tok": 200, "saved_tok": 200, "saved_pct": 50.0}


def test_build_trend_report_single_run_says_not_enough_data():
    text = build_trend_report([_RUN_A])
    assert "at least two" in text
    assert "|" not in text  # no table rendered for a single run


def test_build_trend_report_two_runs_shows_delta():
    text = build_trend_report([_RUN_A, _RUN_B])
    assert "+40.0%" in text and "+50.0%" in text
    assert "+10.0" in text  # delta pts between the two runs
    assert "t1" in text and "t2" in text
    assert "corpus" in text


def test_build_trend_report_first_row_has_no_delta():
    text = build_trend_report([_RUN_A, _RUN_B])
    lines = [line for line in text.splitlines() if line.startswith("| 1 ")]
    assert lines and lines[0].rstrip().endswith("| — |")


def test_build_trend_report_handles_none_saved_pct():
    zero_raw = {"ts": "t0", "label": None, "n_payloads": 0, "lossless_pass": 0,
                "raw_tok": 0, "compressed_tok": 0, "saved_tok": 0, "saved_pct": None}
    text = build_trend_report([zero_raw, _RUN_A])
    assert "n/a" in text
    assert "—" in text  # no label, no prior pct to delta against
