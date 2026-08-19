"""Tests for the codec-tier material-preservation verdict (#295) — `report.codec_verdict`
and `build_codec_verdict_report`.

This gate is deliberately NOT a tolerance: any demonstrated structural mismatch is UNSAFE
regardless of sample size, and a clean run needs `_CODEC_MIN_TRIALS` zero-failure trials
before it can print SAFE rather than UNRESOLVED. Every boundary below is the kind of edge a
tolerance-shaped gate would get wrong by construction — this file exists so an accidental
reintroduction of a ratio/threshold shows up as a red test, not a silent regression.
"""
from __future__ import annotations

from terse.report import (
    _CODEC_MIN_TRIALS,
    build_codec_verdict_report,
    codec_verdict,
)


def _row(qid: str, raw_ok: int, terse_ok: int, trials: int = 1) -> dict:
    return {
        "qid": qid, "qtype": "deref", "transform": "table", "trials": trials,
        "raw_ok": raw_ok, "terse_ok": terse_ok,
        "raw_trials": trials, "terse_trials": trials,
        "fails": 0, "attempts": trials * 2,
    }


# --------------------------------------------------------------------------- #
# codec_verdict
# --------------------------------------------------------------------------- #
def test_a_single_structural_failure_is_UNSAFE_regardless_of_sample_size():
    # 1 failure out of a large, otherwise-clean sample — a ratio-shaped gate would round
    # this down to "safe enough"; the demonstrated-corruption gate must not.
    rows = [_row(f"q{i}", 1, 1) for i in range(50)]
    rows.append(_row("q-bad", 1, 0))  # one paired miss
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNSAFE"


def test_zero_failures_below_the_trial_floor_is_UNRESOLVED_not_SAFE():
    rows = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS - 1)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNRESOLVED"


def test_zero_failures_at_exactly_the_trial_floor_is_SAFE():
    rows = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "SAFE"


def test_an_unmeasured_backend_reports_UNRESOLVED_not_a_confident_verdict():
    # Every row carries zero completed trials -> `_unmeasured` fires inside `arm_gap`.
    rows = [{
        "qid": "q1", "qtype": "deref", "transform": "table", "trials": 1,
        "raw_ok": 0, "terse_ok": 0, "raw_trials": 0, "terse_trials": 0,
        "fails": 2, "attempts": 2,
    }]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNRESOLVED"
    assert gap.excluded == "unmeasured"


def test_empty_rows_is_UNRESOLVED():
    verdict, gap = codec_verdict([])
    assert verdict == "UNRESOLVED"


def test_a_single_failure_beats_an_otherwise_unresolved_thin_sample():
    # A thin sample (below the floor) that ALSO shows a failure must still be UNSAFE, not
    # UNRESOLVED — the failure is dispositive; sample size only gates a clean run.
    rows = [_row("q1", 1, 0)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNSAFE"


# --------------------------------------------------------------------------- #
# build_codec_verdict_report — grouped by (tool, shape), gates on the worst model
# --------------------------------------------------------------------------- #
def _tagged(rows: list[dict], tool: str, shape: str) -> list[dict]:
    return [{"tool": tool, "shape": shape, **r} for r in rows]


def test_report_groups_by_tool_and_shape_not_globally():
    clean = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    results = {
        "m1": (_tagged(clean, "tool-a", "array-of-records")
              + _tagged(clean, "tool-b", "array-of-records")),
    }
    report = build_codec_verdict_report(results)
    assert "`tool-a`" in report
    assert "`tool-b`" in report
    # Two distinct group rows, not one pooled global line.
    assert report.count("| **SAFE**") == 2


def test_report_gates_on_the_worst_model_within_a_group():
    clean = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    broken = [_row("q-bad", 1, 0)]
    results = {
        "good-model": _tagged(clean, "tool-a", "array-of-records"),
        "bad-model": _tagged(broken, "tool-a", "array-of-records"),
    }
    report = build_codec_verdict_report(results)
    assert "**UNSAFE**" in report
    assert "`bad-model`" in report


def test_report_with_no_results_says_so_rather_than_rendering_an_empty_table():
    report = build_codec_verdict_report({})
    assert "## Verdict by tool and shape" not in report
    assert "| **" not in report  # no rendered verdict table row
    assert "No tool-capable model answered" in report
