"""Tests for history.py — run-history persistence for `measure --history` (#51
fast-follow)."""
from __future__ import annotations

import json

from terse.history import append_run, load_history, summarize_run

_ROW_OK = {"roundtrip_ok": True, "cl100k": {"raw": 100, "compressed": 60}}
_ROW_FAIL = {"roundtrip_ok": False, "cl100k": {"raw": 50, "compressed": 50}}


def test_summarize_run_aggregates_tokens_and_gate():
    run = summarize_run([_ROW_OK, dict(_ROW_OK)], "2026-07-03T00:00:00+00:00", label="corpus")
    assert run["ts"] == "2026-07-03T00:00:00+00:00"
    assert run["label"] == "corpus"
    assert run["n_payloads"] == 2
    assert run["lossless_pass"] == 2
    assert run["raw_tok"] == 200
    assert run["compressed_tok"] == 120
    assert run["saved_tok"] == 80
    assert run["saved_pct"] == 40.0


def test_summarize_run_counts_gate_failures():
    run = summarize_run([_ROW_OK, _ROW_FAIL], "ts", label=None)
    assert run["n_payloads"] == 2
    assert run["lossless_pass"] == 1
    assert run["label"] is None


def test_summarize_run_empty_rows_saved_pct_none_not_zero_division():
    run = summarize_run([], "ts")
    assert run["raw_tok"] == 0
    assert run["saved_pct"] is None  # not a ZeroDivisionError, not a misleading 0.0


def test_summarize_run_never_reads_the_clock_itself():
    # principle #31: ts must be an explicit param, never read internally — pass two
    # different timestamps for otherwise-identical rows and confirm both come back
    # verbatim, i.e. the function has no clock of its own to override.
    a = summarize_run([_ROW_OK], "ts-a")
    b = summarize_run([_ROW_OK], "ts-b")
    assert a["ts"] == "ts-a" and b["ts"] == "ts-b"


def test_append_and_load_history_roundtrips(tmp_path):
    path = tmp_path / "history.jsonl"
    run1 = summarize_run([_ROW_OK], "ts-1", label="corpus")
    run2 = summarize_run([_ROW_OK, _ROW_OK], "ts-2", label="corpus")
    append_run(path, run1)
    append_run(path, run2)
    loaded = load_history(path)
    assert loaded == [run1, run2]  # order preserved, oldest first


def test_append_run_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "history.jsonl"
    append_run(path, summarize_run([_ROW_OK], "ts"))
    assert path.exists()


def test_load_history_missing_file_returns_empty(tmp_path):
    assert load_history(tmp_path / "nope.jsonl") == []


def test_load_history_skips_malformed_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"ts": "ok", "n_payloads": 1}\nnot json at all\n', encoding="utf-8")
    loaded = load_history(path)
    assert loaded == [{"ts": "ok", "n_payloads": 1}]


def test_history_file_is_one_json_object_per_line(tmp_path):
    # the whole point of jsonl over a single JSON array: append-only, no read-modify-
    # write of the whole file needed on each run.
    path = tmp_path / "history.jsonl"
    append_run(path, summarize_run([_ROW_OK], "ts-1"))
    append_run(path, summarize_run([_ROW_OK], "ts-2"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)
