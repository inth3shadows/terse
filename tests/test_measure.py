"""Measure + capture behavior, including the per-tier decomposition invariant."""

from __future__ import annotations

import json

from terse import capture
from terse.measure import cross_tokenizer_savings, measure_corpus, measure_payload
from terse.tokenize import CL100K, O200K


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


def test_deeply_nested_record_list_is_array_of_records():
    # A record list two levels deep still tabularizes, so the classifier must recurse
    # and not understate it as compact-json (#4).
    nested = json.dumps({"data": {"results": [{"id": 1, "s": "a"}, {"id": 2, "s": "b"}]}})
    assert capture.classify_shape(nested) == capture.ARRAY_OF_RECORDS
    # a record list inside a list-of-non-records is also reached
    in_list = json.dumps([1, {"q": [{"id": 1}, {"id": 2}]}])
    assert capture.classify_shape(in_list) == capture.ARRAY_OF_RECORDS
    # no record list anywhere -> still compact-json
    plain = json.dumps({"data": {"results": {"id": 1}}})
    assert capture.classify_shape(plain) == capture.COMPACT_JSON


def test_non_uniform_dict_list_is_not_array_of_records():
    # The tabularizer only folds dict lists that share one key set; a non-uniform list
    # is a measured no-op for tabularize, so it must NOT bucket as array-of-records and
    # overstate coverage (matches transforms._uniform_dict_list, the canonical rule).
    nonuniform = json.dumps([{"a": 1}, {"b": 2}])
    assert capture.classify_shape(nonuniform) == capture.COMPACT_JSON


def test_classify_shape_survives_pathological_nesting():
    # Depth that exceeds json.loads's own recursion tolerance on the 3.11 floor: the
    # classifier must catch it and still return a bucket, not crash (#4). The exact
    # bucket is version-dependent (3.11 can't parse this; 3.12+ can), so assert only
    # that it survives with a valid bucket.
    deep = "[" * 1000 + "1" + "]" * 1000
    assert capture.classify_shape(deep) in {
        capture.COMPACT_JSON, capture.LONG_TEXT, capture.OTHER}


def test_classify_shape_caps_deep_recursion_without_crashing():
    # Parseable on every supported Python (well under json.loads's floor-version limit),
    # but deeper than the classifier's own record-search cap: it must return a record-free
    # bucket rather than RecursionError inside the walk (#4).
    deep = "[" * 300 + "1" + "]" * 300
    assert capture.classify_shape(deep) == capture.COMPACT_JSON


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


def test_cross_tokenizer_savings_are_close():
    raw = json.dumps({"result": [{"id": i, "status": "active", "u": "https://x.example/api/v1/items"}
                                  for i in range(25)]})
    rows = cross_tokenizer_savings([{"tool": "t", "raw": raw}])
    cl = rows[0][CL100K]["pct"]
    o2 = rows[0][O200K]["pct"]
    assert cl is not None and cl > 0
    if o2 is not None:  # o200k may be unavailable offline
        assert abs(cl - o2) < 8.0  # structural savings track across vocabularies


def test_measure_corpus_attaches_provenance(tmp_path):
    capture.capture_payload("gh.issues", json.dumps([{"n": 1}, {"n": 2}]), tmp_path)
    rows = measure_corpus(capture.load_corpus(tmp_path))
    assert rows[0]["tool"] == "gh.issues"
    assert "sha" in rows[0]


def test_failed_lossless_gate_zeroes_banked_savings(monkeypatch):
    # "You cannot bank tokens you lost data to": if the round-trip gate fails, the row's
    # saved_cl100k must be zeroed at the source so a downstream aggregator that forgot to
    # filter on roundtrip_ok can't inflate the headline % with a broken payload's savings.
    raw = json.dumps({"result": [{"id": i, "name": f"n{i}", "ok": True} for i in range(20)]})
    monkeypatch.setattr("terse.measure.transforms.roundtrip_ok", lambda _obj: False)
    row = measure_payload(raw)
    assert row["roundtrip_ok"] is False
    assert set(row["saved_cl100k"].values()) == {0}
    # raw token counts stay for transparency — only the *banked savings* are zeroed.
    assert row["cl100k"]["raw"] > 0
