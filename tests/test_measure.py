"""Measure + capture behavior, including the per-tier decomposition invariant."""

from __future__ import annotations

import json

from terse import capture
from terse.measure import measure_corpus, measure_payload


def test_decomposition_sums_to_total():
    raw = json.dumps({"result": [{"id": i, "name": f"n{i}", "ok": True} for i in range(20)]})
    row = measure_payload(raw)
    s = row["saved_cl100k"]
    assert s["minify"] + s["tabularize"] + s["dictionary"] == s["tier_total"]
    assert row["roundtrip_ok"] is True


def test_already_compact_has_zero_minify_saving():
    raw = '{"a":1,"b":2,"c":"x"}'  # single compact object, no record list
    row = measure_payload(raw)
    assert row["saved_cl100k"]["minify"] == 0
    assert row["saved_cl100k"]["tabularize"] == 0


def test_pretty_records_save_via_both_tiers():
    raw = json.dumps([{"id": i, "name": "x", "status": "active"} for i in range(30)], indent=2)
    row = measure_payload(raw)
    assert row["shape"] == capture.ARRAY_OF_RECORDS
    assert row["saved_cl100k"]["minify"] > 0       # had whitespace to strip
    assert row["saved_cl100k"]["tabularize"] > 0    # had repeated keys to fold
    assert row["roundtrip_ok"] is True


def test_non_json_is_passthrough_and_lossless():
    raw = "this is not json, just prose " * 50
    row = measure_payload(raw)
    assert row["applicable"] is False
    assert row["roundtrip_ok"] is True
    assert row["saved_cl100k"]["tier_total"] == 0


def test_minified_json_with_trailing_newline_is_not_pretty():
    # `jq -c` emits one line + a trailing newline; that must not read as pretty-printed.
    compact = json.dumps({"a": 1, "b": [1, 2, 3]}) + "\n"
    assert capture.classify_shape(compact) == capture.COMPACT_JSON
    indented = json.dumps({"a": 1, "b": [1, 2, 3]}, indent=2)
    assert capture.classify_shape(indented) == capture.PRETTY_JSON


def test_capture_load_coverage_roundtrip(tmp_path):
    capture.capture_payload("gh.issues", json.dumps([{"n": 1}, {"n": 2}]), tmp_path)
    capture.capture_payload("gh.user", json.dumps({"login": "x"}), tmp_path)
    envs = capture.load_corpus(tmp_path)
    assert len(envs) == 2
    cov = capture.coverage(envs)
    assert cov["total"] == 2
    assert cov["by_tool"]["gh.issues"] == 1


def test_capture_is_idempotent_by_sha(tmp_path):
    raw = json.dumps({"a": 1})
    p1 = capture.capture_payload("t", raw, tmp_path)
    p2 = capture.capture_payload("t", raw, tmp_path)
    assert p1 == p2
    assert len(capture.load_corpus(tmp_path)) == 1


def test_measure_corpus_attaches_provenance(tmp_path):
    capture.capture_payload("gh.issues", json.dumps([{"n": 1}, {"n": 2}]), tmp_path)
    rows = measure_corpus(capture.load_corpus(tmp_path))
    assert rows[0]["tool"] == "gh.issues"
    assert "sha" in rows[0]
