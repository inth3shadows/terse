"""Tests for report.py's build_trend_report (#51 fast-follow: historical trend across
`measure --history` runs). Scoped to this one addition, not a full backfill of
report.py's existing markdown builders."""
from __future__ import annotations

from terse.report import build_report, build_trend_report, verify_summary


def test_verify_summary_passing_corpus_totals_and_gate():
    rows = [
        {"tool": "t", "sha": "a", "shape": "array-of-records", "roundtrip_ok": True,
         "cl100k": {"raw": 100, "compressed": 40}},
        {"tool": "t", "sha": "b", "shape": "array-of-records", "roundtrip_ok": True,
         "cl100k": {"raw": 100, "compressed": 60}},
    ]
    cov = {"total": 2, "by_tool": {"t": 2}, "by_shape": {"array-of-records": 2}}
    s = verify_summary(rows, cov, "my corpus")
    assert s["corpus"] == "my corpus" and s["payloads"] == 2
    assert s["lossless_gate"] == {"ok": True, "passed": 2, "total": 2, "failures": []}
    assert s["tokens_cl100k"]["raw_tokens"] == 200
    assert s["tokens_cl100k"]["saved_tokens"] == 100
    assert s["tokens_cl100k"]["saved_pct"] == 50.0
    assert s["by_shape"]["array-of-records"]["n"] == 2
    assert s["coverage"]["by_tool"] == {"t": 2}


def test_verify_summary_flags_gate_failures():
    rows = [
        {"tool": "t", "sha": "ok", "shape": "s", "roundtrip_ok": True,
         "cl100k": {"raw": 10, "compressed": 5}},
        {"tool": "t", "sha": "bad", "shape": "s", "roundtrip_ok": False,
         "cl100k": {"raw": 10, "compressed": 5}},
    ]
    s = verify_summary(rows, {"total": 2}, "c")
    assert s["lossless_gate"]["ok"] is False
    assert s["lossless_gate"]["passed"] == 1
    assert s["lossless_gate"]["failures"] == [{"tool": "t", "sha": "bad", "shape": "s"}]


def test_embedded_gate_failure_is_reported_without_invalidating_the_run():
    """#188 split the one gate into two. `embedded_ok` costs only its own tier, so it must
    NOT flip `lossless_gate.ok` — but it must still appear, or every reader that filters on
    `roundtrip_ok` (this summary, `build_report`, `history`, `html_report`) would show a
    clean run while an opt-in tier was silently losing data."""
    rows = [
        {"tool": "t", "sha": "ok", "shape": "s", "roundtrip_ok": True, "embedded_ok": True,
         "cl100k": {"raw": 10, "compressed": 5}},
        {"tool": "t", "sha": "emb", "shape": "s", "roundtrip_ok": True, "embedded_ok": False,
         "cl100k": {"raw": 10, "compressed": 5}},
    ]
    s = verify_summary(rows, {"total": 2}, "c")
    assert s["lossless_gate"]["ok"] is True          # the default pipeline round-tripped
    assert s["embedded_gate"]["ok"] is False
    assert s["embedded_gate"]["failures"] == [{"tool": "t", "sha": "emb", "shape": "s"}]

    md = build_report(rows, {"total": 2, "by_tool": {"t": 2}, "by_shape": {"s": 2}})
    assert "round-trip losslessly" in md                      # not marked INVALID
    assert "`embedded` tier: 1/2 payloads FAILED" in md
    assert "`t` / `emb` (s)" in md


def test_rows_predating_the_embedded_gate_read_as_clean():
    """A corpus measured before #188 has no `embedded_ok` key. It must default to True —
    absence of the field is not evidence of a failure, and defaulting the other way would
    print an alarming (and unfounded) tier failure for every historical run."""
    rows = [{"tool": "t", "sha": "a", "shape": "s", "roundtrip_ok": True,
             "cl100k": {"raw": 10, "compressed": 5}}]
    assert verify_summary(rows, {"total": 1}, "c")["embedded_gate"]["ok"] is True
    assert "`embedded` tier" not in build_report(rows, {"total": 1, "by_tool": {"t": 1},
                                                        "by_shape": {"s": 1}})

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
