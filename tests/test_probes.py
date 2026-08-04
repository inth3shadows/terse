"""Ceiling-probe behavior: value redundancy + cross-call overlap."""

from __future__ import annotations

import json

from terse import probes, transforms
from terse.capture import (
    ARRAY_OF_RECORDS,
    classify_shape,
    extract_records,
    find_record_list_with_path,
)
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


def test_extract_records_follows_the_tabularizer_not_a_stricter_rule():
    """The extractor's job is to agree with the codec about what is record-shaped (#204).

    It used to require an identical key set, which was right until union-schema tabularize
    widened the codec — after which a payload the codec folded at 55.8% classified as
    `compact-json` with no record list, and the probes, `policy_gen`'s drop-path
    generation, `dropeval` and coverage all skipped it.
    """
    non_uniform = {"result": [{"id": 1, "x": 0}, {"id": 2}]}
    assert transforms.is_tabularizable(non_uniform["result"])
    assert extract_records(non_uniform) == non_uniform["result"]

    # Still bounded by the codec's own density gate, not by uniformity: 2 keys x 2 rows
    # with 2 cells filled is 50%, which the tabularizer refuses, so this is not a record
    # list for either of them.
    assert not transforms.is_tabularizable([{"a": 1}, {"b": 2}])
    assert extract_records([{"a": 1}, {"b": 2}]) is None


def test_records_from_a_widened_extract_are_safe_for_the_probes():
    """`probes` is the consumer that needed NO change: it walks `rec.items()` and counts
    per-field presence, so an absent key is data, not a KeyError."""
    records = [{"name": "a", "kind": "fn", "line": 1}, {"name": "b", "kind": "fn"},
               {"name": "c", "kind": "var", "line": 3}]
    profiles = probes.field_profiles(records)
    assert profiles["line"]["n"] == 2      # present in 2 of 3, counted as such
    assert profiles["name"]["n"] == 3
    probes.value_redundancy(records)       # must not raise


def test_server_of_tool_maps_the_three_known_servers():
    assert server_of_tool("kb.read.search") == "kb"
    assert server_of_tool("codegraph_explore") == "codegraph"
    assert server_of_tool("locate") == "runecho"
    assert server_of_tool("structure") == "runecho"
    # Unknown server degrades to its leading token, never crashes.
    assert server_of_tool("weather.forecast") == "weather"


def test_server_of_tool_prefers_the_stated_server_over_the_heuristic():
    # #158: since the envelope records `server`, the stated value is returned verbatim —
    # even when the name-based heuristic would guess otherwise. `structure` on a server
    # that ISN'T runecho must attribute to that server, not to the stale hardcoded set.
    assert server_of_tool("structure", "my-fork") == "my-fork"
    assert server_of_tool("kb.read.search", "kb-mirror") == "kb-mirror"


def test_server_of_tool_falls_back_to_the_heuristic_for_legacy_envelopes():
    # No server (pre-#156 corpus), or an empty-string server, still resolves by name.
    assert server_of_tool("structure", None) == "runecho"
    assert server_of_tool("structure", "") == "runecho"


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


def test_cross_server_redundancy_handles_empty_records():
    # Text/source-only corpus yields no record-shaped payloads -> Lever A is empty. It must
    # degrade to zeros without dividing by zero. (Lever B does NOT then carry a BUILD verdict —
    # see test_blind_lever_a_never_builds_on_token_overlap.)
    res = cross_server_redundancy({})
    assert res["per_server"] == []
    assert res["cross_server_increment_tokens"] == 0
    assert res["increment_frac_of_corpus"] == 0.0
    assert res["increment_frac_over_per_peer"] == 0.0


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


def test_blind_lever_a_never_builds_on_token_overlap():
    # The #64 regression guard: with Lever A blind (no record-shaped payloads) and a HIGH
    # Lever B token overlap, the verdict must be INCONCLUSIVE, never a BUILD. Token (subword)
    # overlap is not value-elision headroom — 20.9% Lever B once sat atop 0 value overlap.
    from terse.report import build_cross_server_probe_report
    redundancy = {
        "per_server": [],  # Lever A blind
        "per_peer_saving_tokens": 0, "pooled_saving_tokens": 0,
        "cross_server_increment_tokens": 0, "increment_frac_of_corpus": 0.0,
        "increment_frac_over_per_peer": 0.0,
    }
    overlap = {
        "rows": [{"server_a": "kb", "server_b": "runecho", "curr_tokens": 100,
                  "shared_tokens": 40, "overlap_ratio": 0.4, "content_overlap_ratio": 0.30}],
        "median_overlap": 0.4, "median_content_overlap": 0.30,  # a HIGH Lever B
        "pairs": 1, "capped": False, "cap_per_pair": 10,
    }
    report = build_cross_server_probe_report(redundancy, overlap,
                                             corpus_servers=["kb", "runecho"])
    verdict = report.split("## Verdict", 1)[1]
    assert "INCONCLUSIVE" in verdict
    assert "BUILD" not in verdict.replace("do NOT build", "").replace("Do not", "")


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


# --- #204: the shape classifier must agree with the codec ---


def test_classify_shape_buckets_a_non_uniform_record_list_as_records():
    """The reported symptom: a payload two thirds of whose rows carry `line` bucketed as
    `compact-json` with no record list, while the codec compressed it 55.8%. Coverage
    reporting then said there was nothing record-shaped to compress on a tool that
    compresses well."""
    obj = {"symbols": [{"name": f"s{i}", "kind": "fn", **({"line": i} if i % 3 else {})}
                       for i in range(30)]}
    raw = json.dumps(obj)
    assert not transforms._uniform_dict_list(obj["symbols"])   # the old rule said no
    assert transforms.is_tabularizable(obj["symbols"])         # the codec says yes
    assert classify_shape(raw) == ARRAY_OF_RECORDS
    assert extract_records(obj) == obj["symbols"]


def test_drop_path_generation_reaches_a_non_uniform_record_list():
    """`policy_gen` builds auto drop-path suggestions from `find_record_list_with_path`, so
    the narrow rule meant no record list, no suggestion — on exactly the traffic
    union-schema tabularize was built for."""
    from terse import policy_gen
    # Bodies must be DISTINCT: an identical value across records is dictionary-folded, so
    # it is correctly not a drop candidate. High-cardinality bulk is what drop targets.
    recs = [{"name": f"s{i}", "kind": "fn", "body": "x" * 4000 + str(i),
             **({"line": i} if i % 3 else {})} for i in range(20)]
    payload = json.dumps({"symbols": recs})
    records, path = find_record_list_with_path(json.loads(payload))
    assert path == "symbols[]" and records is not None
    suggestion, rows = policy_gen._drop_candidates([payload])
    assert "symbols[].body" in suggestion, "the big field should now be a drop candidate"
    assert rows


def test_shape_classifier_and_codec_agree_on_every_bench_payload():
    """The invariant #4 established and #204 restored: `_has_record_list` is true exactly
    when the tabularizer folds something in the payload.

    The oracle is `compress_structure`'s actual OUTPUT — does a `__terse_table__` appear —
    not a second call to `is_tabularizable`. Asking the predicate about itself is a
    tautology that stays green through the very regression this test is named for: with the
    classifier reverted to the strict rule it still passed, and with the codec's union
    branch disabled while the classifier stayed correct it still passed.

    Deliberately at the `compress_structure` layer, not the wire: `compress_with`'s
    emit-only-if-smaller can ship plain minify for a payload that DID tabularize, so the
    equivalence is about what the codec folds, not about what survives the size guard.
    """
    from pathlib import Path

    from terse.capture import _has_record_list

    def codec_folds(o):
        """True if compressing `o` actually produces a table anywhere in the result."""
        def has_table(node):
            if isinstance(node, dict):
                if node.get(transforms.TABLE_MARKER) == 1:
                    return True
                return any(has_table(v) for v in node.values())
            if isinstance(node, list):
                return any(has_table(x) for x in node)
            return False
        return has_table(transforms.compress_structure(o))

    corpus = Path(__file__).resolve().parent.parent / "scripts" / "bench" / "corpus"
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(corpus.glob("*.json"))]
    payloads += [
        {"symbols": [{"a": i, "b": "x", **({"c": i} if i % 3 else {})} for i in range(30)]},
        [{"a": 1}, {"b": 2}],                       # too sparse for either
        {"n": 1},                                   # no list at all
        [{"id": i, "v": "x"} for i in range(5)],    # plain uniform
    ]
    for obj in payloads:
        assert _has_record_list(obj) == codec_folds(obj), obj if len(str(obj)) < 200 else "…"
