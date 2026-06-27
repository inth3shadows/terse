"""Cross-call diff codec (Tier 0.7): lossless delta of a result against the prior
same-tool result. Every accepted diff must rebuild curr exactly; the codec returns
None when no representable diff applies (the caller then sends the full form)."""

from __future__ import annotations

import json

from terse import transforms as T


def _recs(n, status="active", extra=None):
    out = [{"id": i, "status": status, "n": i * 10} for i in range(n)]
    if extra:
        for r in out:
            r.update(extra)
    return out


def test_unchanged_payload_yields_empty_diff_that_roundtrips():
    prev = {"result": _recs(20)}
    curr = {"result": _recs(20)}
    diff = T.diff_encode(prev, curr)
    assert diff is not None and diff["shape"] == "rows"
    assert diff["set"] == [] and diff["del"] == [] and diff["new"] == []
    assert T.diff_decode(prev, diff) == curr


def test_changed_field_in_one_row_roundtrips_and_is_minimal():
    prev = {"result": _recs(20)}
    curr = json.loads(json.dumps(prev))
    curr["result"][5]["status"] = "closed"
    diff = T.diff_encode(prev, curr)
    assert [r["id"] for r in diff["set"]] == [5]   # only the changed row carried
    assert diff["new"] == [] and diff["del"] == []
    assert T.diff_decode(prev, diff) == curr


def test_appended_rows_roundtrip():
    prev = {"result": _recs(10)}
    curr = {"result": _recs(13)}            # 3 rows appended
    diff = T.diff_encode(prev, curr)
    assert diff["new"] == [10, 11, 12]
    assert diff["n"] == 13
    assert T.diff_decode(prev, diff) == curr


def test_removed_rows_roundtrip():
    prev = {"result": _recs(10)}
    curr = {"result": [r for r in _recs(10) if r["id"] not in (3, 7)]}
    diff = T.diff_encode(prev, curr)
    assert set(diff["del"]) == {3, 7}
    assert T.diff_decode(prev, diff) == curr


def test_top_level_list_roundtrips():
    prev = _recs(8)
    curr = _recs(9)
    diff = T.diff_encode(prev, curr)
    assert diff["at"] is None
    assert T.diff_decode(prev, diff) == curr


def test_reorder_is_not_representable_falls_back_to_keys_or_none():
    prev = {"result": _recs(6)}
    curr = {"result": list(reversed(_recs(6)))}
    diff = T.diff_encode(prev, curr)
    # row strategy bows out on reorder; key strategy may still represent it losslessly.
    if diff is not None:
        assert T.diff_decode(prev, diff) == curr


def test_dict_key_diff_roundtrips():
    prev = {"a": 1, "b": 2, "c": 3}
    curr = {"a": 1, "b": 99, "d": 4}        # b changed, c removed, d added
    diff = T.diff_encode(prev, curr)
    assert diff["shape"] == "keys"
    assert diff["set"] == {"b": 99, "d": 4} and diff["del"] == ["c"]
    assert T.diff_decode(prev, diff) == curr


def test_no_id_column_is_not_diffable_via_rows():
    # records with no unique scalar column the row diff can key on
    prev = {"result": [{"tags": ["x"]}, {"tags": ["y"]}]}
    curr = {"result": [{"tags": ["x"]}, {"tags": ["z"]}]}
    diff = T.diff_encode(prev, curr)
    # not row-shaped-diffable; key diff handles it (whole list as one changed value)
    assert diff is None or T.diff_decode(prev, diff) == curr


def test_incompatible_shapes_return_none():
    assert T.diff_encode([1, 2, 3], {"a": 1}) is None
    assert T.diff_encode("a string", "another") is None


def test_roundtrip_gate_helper():
    prev = {"result": _recs(20)}
    curr = json.loads(json.dumps(prev))
    curr["result"][0]["status"] = "x"
    assert T.diff_roundtrip_ok(prev, curr)
    assert not T.diff_roundtrip_ok("scalar", 12345)  # nothing representable
