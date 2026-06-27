"""Ceiling-probe behavior: value redundancy + cross-call overlap."""

from __future__ import annotations

import json

from terse.capture import extract_records
from terse.probes import cross_call_overlap, value_redundancy


def test_value_redundancy_flags_repeated_values():
    # 'status' is identical across rows; 'id' is unique -> partial redundancy.
    records = [{"id": i, "status": "active", "region": "us-east-1"} for i in range(10)]
    res = value_redundancy(records)
    assert res["cells"] == 30
    assert res["redundancy_ratio"] > 0.0
    assert res["redundant_value_tokens"] > 0
    assert res["est_dict_saving_tokens"] >= 0


def test_value_redundancy_zero_when_all_unique():
    records = [{"a": f"unique-{i}", "b": f"other-{i}"} for i in range(5)]
    res = value_redundancy(records)
    assert res["redundant_value_tokens"] == 0
    assert res["redundancy_ratio"] == 0.0


def test_cross_call_overlap_high_for_near_identical():
    a = json.dumps([{"id": i, "name": "x"} for i in range(20)])
    b = json.dumps([{"id": i, "name": "x"} for i in range(20)] + [{"id": 99, "name": "y"}])
    res = cross_call_overlap(a, b)
    assert res["available"] is True
    assert res["overlap_ratio"] > 0.8  # b is mostly a


def test_cross_call_overlap_lower_for_disjoint_content():
    # Content-disjoint payloads still share JSON framing tokens, so overlap is not
    # zero — but it must be clearly below the near-identical case.
    base = [{"id": i, "name": f"alpha-payload-{i}-xxxxx"} for i in range(30)]
    near = base + [{"id": 99, "name": "alpha-payload-99-xxxxx"}]
    disjoint = [{"uid": f"zzz-{i}-qqq", "tag": f"omega-{i}-www"} for i in range(30)]
    a = json.dumps(base)
    near_ratio = cross_call_overlap(a, json.dumps(near))["overlap_ratio"]
    disjoint_ratio = cross_call_overlap(a, json.dumps(disjoint))["overlap_ratio"]
    # Relative ordering is the real invariant; absolute overlap is data-dependent
    # (shared framing + integer ids inflate it) and not worth pinning.
    assert near_ratio > 0.9
    assert disjoint_ratio < near_ratio


def test_extract_records_top_level_and_wrapped():
    assert extract_records([{"a": 1}, {"a": 2}]) is not None
    assert extract_records({"result": [{"a": 1}, {"a": 2}]}) is not None
    assert extract_records({"a": 1}) is None
    assert extract_records([{"a": 1}]) is None  # single record, not a list to fold
