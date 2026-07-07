"""Ceiling-probe behavior: value redundancy + cross-call overlap."""

from __future__ import annotations

import json

from terse.capture import extract_records
from terse.probes import (
    cross_call_overlap,
    cross_server_overlap,
    cross_server_redundancy,
    field_profiles,
    server_of_tool,
    token_idf,
    value_redundancy,
)


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


def test_extract_records_recurses_to_match_the_tabularizer():
    # The tabularizer folds a uniform record list at ANY depth, so extract_records must
    # find it there too — else the probes/fluency silently skip nested record payloads
    # the coverage report counts as record-shaped (#4).
    nested = {"data": {"results": [{"id": 1, "s": "a"}, {"id": 2, "s": "b"}]}}
    assert extract_records(nested) == [{"id": 1, "s": "a"}, {"id": 2, "s": "b"}]


def test_extract_records_requires_uniform_keys():
    # A non-uniform dict list is NOT what the tabularizer folds (it needs one shared key
    # set), so it must not be returned — callers index every record by the first record's
    # columns and would KeyError otherwise.
    assert extract_records([{"a": 1}, {"b": 2}]) is None
    assert extract_records({"result": [{"id": 1, "x": 0}, {"id": 2}]}) is None


def test_server_of_tool_maps_the_three_known_servers():
    assert server_of_tool("kb.read.search") == "kb"
    assert server_of_tool("codegraph_explore") == "codegraph"
    assert server_of_tool("locate") == "runecho"
    assert server_of_tool("structure") == "runecho"
    # Unknown server degrades to its leading token, never crashes.
    assert server_of_tool("weather.forecast") == "weather"


def test_cross_server_redundancy_positive_when_value_shared_across_peers():
    # "us-east-1" appears in BOTH servers -> a shared legend folds it once; two
    # per-peer legends each keep their own copy. So pooled > per-peer.
    by_server = {
        "kb": [{"id": i, "region": "us-east-1"} for i in range(5)],
        "codegraph": [{"node": i, "region": "us-east-1"} for i in range(5)],
    }
    res = cross_server_redundancy(by_server)
    assert res["cross_server_increment_tokens"] > 0
    assert res["increment_frac_of_corpus"] > 0
    assert len(res["per_server"]) == 2


def test_cross_server_redundancy_zero_when_no_value_shared_across_peers():
    # Disjoint values between servers -> a shared legend buys nothing over per-peer.
    by_server = {
        "kb": [{"id": i, "tag": f"kb-only-{i}"} for i in range(5)],
        "codegraph": [{"id": i, "tag": f"cg-only-{i}"} for i in range(5)],
    }
    res = cross_server_redundancy(by_server)
    assert res["cross_server_increment_tokens"] == 0


def test_token_idf_zeroes_ubiquitous_tokens():
    # A token in EVERY payload (framing) gets idf 0; a token in one payload gets idf > 0.
    raws = [json.dumps({"framing": "here", "uniq": f"only-{i}-zzz"}) for i in range(8)]
    idf = token_idf(raws)
    # No token should have negative idf; at least one rare content token must be positive.
    assert all(v >= 0 for v in idf.values())
    assert max(idf.values()) > 0


def test_content_overlap_nets_out_framing():
    # Two payloads that share ONLY framing/structure but no content values: idf-weighted
    # content overlap must be far below the raw overlap (which framing inflates).
    idf = token_idf([
        json.dumps([{"k": f"aaa-{i}"} for i in range(20)]),
        json.dumps([{"k": f"bbb-{i}"} for i in range(20)]),
    ])
    a = json.dumps([{"k": f"aaa-{i}"} for i in range(20)])
    b = json.dumps([{"k": f"bbb-{i}"} for i in range(20)])
    res = cross_call_overlap(a, b, idf=idf)
    assert res["content_overlap_ratio"] < res["overlap_ratio"]


def test_content_overlap_high_when_real_content_shared():
    # Same rare content token present in both -> content overlap should be clearly positive.
    corpus = [json.dumps({"sym": "SharedSymbolXYZ", "n": i}) for i in range(6)]
    idf = token_idf(corpus + [json.dumps({"other": "unrelated"})])
    a = json.dumps({"sym": "SharedSymbolXYZ", "n": 1})
    b = json.dumps({"sym": "SharedSymbolXYZ", "n": 2})
    res = cross_call_overlap(a, b, idf=idf)
    assert res["content_overlap_ratio"] > 0


def test_cross_server_overlap_pairs_across_servers_and_caps():
    raws = {
        "kb": [(f"{i:02x}", json.dumps([{"id": i, "v": "x"}])) for i in range(50)],
        "codegraph": [(f"{i:02x}", json.dumps([{"id": i, "v": "x"}])) for i in range(50)],
    }
    res = cross_server_overlap(raws, cap_per_pair=10)
    assert res["capped"] is True
    assert res["pairs"] == 10          # one server-pair, capped to 10 positional pairs
    assert 0.0 <= res["median_overlap"] <= 1.0
    assert 0.0 <= res["median_content_overlap"] <= 1.0


def test_field_profiles_size_and_cardinality():
    # 'blob' is identical across rows (low cardinality, large); 'uniq' differs every row;
    # 'id' is small. This is the size x cardinality split drop-candidate detection keys on.
    recs = [{"id": i, "blob": "z" * 400, "uniq": f"u{i}"} for i in range(10)]
    p = field_profiles(recs)
    assert p["uniq"]["uniq_ratio"] == 1.0
    assert p["blob"]["uniq_ratio"] == 0.1                 # 1 distinct / 10 present
    assert p["blob"]["mean_tok"] > p["id"]["mean_tok"]    # the blob dominates size
    assert p["blob"]["n"] == 10
    # tok_share is a fraction of the record list's total tokens
    assert abs(sum(f["tok_share"] for f in p.values()) - 1.0) < 1e-6
