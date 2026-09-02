"""A stored envelope `shape` is a cache of `classify_shape(raw)`, never a fact (#355).

`7be9d41` (#208/#204) relaxed `_find_record_list` from an identical-keyset rule to the
codec's union-schema `is_tabularizable`. Every envelope captured before it and never
re-recorded kept the old bucket, so `terse measure` printed two shape tables 36 payloads
apart in ONE report — Coverage read the stored field, the savings table re-classified —
and the codec verdict filed groups under a shape the codec no longer assigns.

These pin the invariant the fix buys: a stored shape that disagrees with
`classify_shape(raw)` cannot reach a consumer.
"""

from __future__ import annotations

import json

from terse import capture, codeceval, measure
from terse.dropeval import Turn

# The exact #355 shape: keysets are NON-uniform, so this list qualifies as records only
# under the post-#208 union-schema rule. Under the old identical-keyset rule it was
# `compact-json` — which is what a pre-#208 corpus still has stored on it.
DRIFTED = {"result": [{"id": 1, "name": "a"}, {"id": 2, "extra": True}]}
DRIFTED_RAW = json.dumps(DRIFTED)
STALE = "compact-json"
LIVE = "array-of-records"


def test_the_fixture_is_actually_drifted():
    """Guards the other tests: if `classify_shape` ever stops calling this payload
    records, every assertion below would pass for the wrong reason."""
    assert capture.classify_shape(DRIFTED_RAW) == LIVE != STALE


def test_envelope_shape_ignores_a_stale_stored_bucket():
    env = {"tool": "t", "shape": STALE, "raw": DRIFTED_RAW}
    assert capture.envelope_shape(env) == LIVE


def test_envelope_shape_falls_back_to_the_stored_value_with_no_raw_to_classify():
    """A hand-built envelope or a foreign corpus may carry no `raw`. There is nothing to
    re-derive from, so the stored label is the best available — not the default."""
    assert capture.envelope_shape({"tool": "t", "shape": STALE}) == STALE
    assert capture.envelope_shape({"tool": "t", "shape": STALE, "raw": {"not": "a str"}}) == STALE


def test_envelope_shape_uses_the_default_when_there_is_neither():
    assert capture.envelope_shape({"tool": "t"}) == "?"
    assert capture.envelope_shape({"tool": "t", "shape": ""}, "unknown") == "unknown"


def test_coverage_counts_the_live_bucket_not_the_stored_one():
    envs = [{"tool": "t", "shape": STALE, "raw": DRIFTED_RAW}]
    assert capture.coverage(envs)["by_shape"] == {LIVE: 1}


def test_load_corpus_leaves_the_stale_stored_field_alone(tmp_path):
    """Re-classification happens at the READ, and ONLY there. `load_corpus` deliberately
    does not rewrite the field: it would mutate a dict the caller owns, and it would erase
    the only signal that a corpus predates a classifier change — which a future "N
    envelopes carry a stale bucket" diagnostic has to be able to read.

    So the drift stays visible on the envelope while every consumer still reports live."""
    (tmp_path / "t__abc.json").write_text(json.dumps(
        {"tool": "t", "shape": STALE, "bytes": len(DRIFTED_RAW), "sha": "abc",
         "captured_at": 1, "raw": DRIFTED_RAW}))
    (loaded,) = capture.load_corpus(tmp_path)
    assert loaded["shape"] == STALE                        # evidence preserved
    assert capture.envelope_shape(loaded) == LIVE          # consumer sees live
    assert capture.coverage([loaded])["by_shape"] == {LIVE: 1}


def test_run_codec_fluency_stamps_the_live_shape():
    """The codec verdict is per `(tool, shape)`. A group headed by a shape the codec does
    not assign answers a question nobody asked.

    The fixture is a UNIFORM record list with a lying stored bucket rather than a #355
    payload, because `fluency.gen_questions` emits a `deref` only for a column present on
    every record — so a non-uniform payload yields no question and never reaches this
    tagging at all. Measured: the drifted list above returns `[]` questions. That makes the
    codec verdict the one consumer #355 could not reach in practice today; the tag is still
    read live, so the NEXT classifier change cannot mislabel a group either."""
    payload = {"result": [{"id": 1, "meta": {"o": "a"}}, {"id": 2, "meta": {"o": "b"}}]}
    raw = json.dumps(payload)
    assert capture.classify_shape(raw) == LIVE
    assert codeceval.gen_codec_questions(DRIFTED) == []   # pins the claim above
    envs = [{"tool": "demo.get", "shape": STALE, "sha": "abc", "raw": raw}]
    rows = codeceval.run_codec_fluency(envs, {"m1": lambda m: Turn(text="", tool_calls=[])})
    assert rows["m1"]
    assert {r["shape"] for r in rows["m1"]} == {LIVE}


def test_the_two_shape_tables_of_one_measure_report_agree(tmp_path):
    """The reported symptom, end to end: `build_report`'s Coverage table reads
    `coverage(envelopes)["by_shape"]`, and its savings table groups measured rows on
    `measure_payload`'s own `classify_shape`. Both sides must bucket a drifted payload
    identically."""
    for i, (raw, stored) in enumerate([(DRIFTED_RAW, STALE),
                                       (json.dumps({"a": 1}), LIVE),      # drifted the other way
                                       ("plain text", "pretty-json")]):   # not JSON at all
        (tmp_path / f"t__{i}.json").write_text(json.dumps(
            {"tool": "t", "shape": stored, "sha": str(i), "captured_at": i, "raw": raw}))

    envelopes = capture.load_corpus(tmp_path)
    from_coverage = capture.coverage(envelopes)["by_shape"]

    from_rows: dict[str, int] = {}
    for env in envelopes:
        shape = measure.measure_payload(env["raw"])["shape"]
        from_rows[shape] = from_rows.get(shape, 0) + 1

    assert from_coverage == from_rows
