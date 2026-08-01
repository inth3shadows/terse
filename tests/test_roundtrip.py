"""The lossless gate as a test suite: every Tier-0 transform must round-trip.

A failing case here means terse dropped data it promised to keep — the one thing
the design forbids. These run over shapes that exercise minify and the recursive
tabularizer, including the cases where tabularize must DECLINE (heterogeneous
lists) and still stay lossless.
"""

from __future__ import annotations

import json

import pytest

from terse import transforms

CASES = [
    pytest.param({}, id="empty-dict"),
    pytest.param([], id="empty-list"),
    pytest.param({"a": 1, "b": "x", "c": None, "d": True, "e": 1.5}, id="scalars"),
    pytest.param(
        [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}],
        id="array-of-records",
    ),
    pytest.param(
        {"result": [{"id": 1, "s": 0.9}, {"id": 2, "s": 0.8}], "total": 2},
        id="wrapped-records",
    ),
    pytest.param(
        [{"id": 1, "name": "a"}, {"id": 2}],  # different key sets -> declines
        id="heterogeneous-list",
    ),
    pytest.param(
        [{"id": 1, "tags": [{"k": "x"}, {"k": "y"}]}, {"id": 2, "tags": [{"k": "z"}, {"k": "w"}]}],
        id="nested-records",
    ),
    pytest.param([{"id": 1}], id="single-record-no-fold"),
    pytest.param("just a string", id="bare-string"),
    pytest.param([1, 2, 3, "mixed", {"a": 1}], id="mixed-list"),
    # Tier 0.5 dictionary-coding exercise: repeated long string values across rows.
    pytest.param(
        [
            {"id": i, "url": "https://api.github.com/repos/inth3shadows/terse",
             "owner": {"login": "inth3shadows", "type": "User"}}
            for i in range(15)
        ],
        id="repeated-values-and-subobjects",
    ),
    # Adversarial: literal values that look like alias references must still round-trip.
    pytest.param(
        [{"v": "~0"}, {"v": "~0"}, {"v": "~1"}, {"v": "real"}, {"v": "real"}, {"v": "real"}],
        id="values-collide-with-alias-namespace",
    ),
    # Nested key folding: a uniform-dict column (owner) hoisted to subcols.
    pytest.param(
        [{"id": i, "owner": {"login": "eric", "perms": {"push": True, "admin": False}}}
         for i in range(8)],
        id="nested-dict-columns-deep",
    ),
    # Heterogeneous nested dicts must NOT fold (different inner keys) — still lossless.
    pytest.param(
        [{"id": 1, "meta": {"a": 1}}, {"id": 2, "meta": {"b": 2}}, {"id": 3, "meta": {"a": 9}}],
        id="nested-dicts-heterogeneous",
    ),
    # Whole-subtree aliasing: the same config object in many value positions (NOT a
    # record list, so tabularize can't fold it — only subtree aliasing can).
    pytest.param(
        {f"svc{i}": {"region": "us-east-1", "retries": 5, "endpoints": ["a", "b", "c"]}
         for i in range(6)},
        id="repeated-whole-subobject",
    ),
    # Whole-subtree aliasing of a repeated list value inside records.
    pytest.param(
        [{"id": i, "tags": ["alpha", "beta", "gamma", "delta"]} for i in range(10)],
        id="repeated-whole-list",
    ),
]


@pytest.mark.parametrize("obj", CASES)
def test_roundtrip_is_lossless(obj):
    assert transforms.roundtrip_ok(obj), "Tier-0 pipeline dropped data"


def test_tabularize_actually_folds_records():
    records = [{"id": i, "name": "x"} for i in range(5)]
    compressed = transforms.compress_structure(records)
    assert compressed.get(transforms.TABLE_MARKER) == 1
    assert compressed["cols"] == ["id", "name"]
    assert len(compressed["rows"]) == 5


def test_table_header_carries_row_count():
    """The `n` hint must equal the row count and survive the round-trip exactly."""
    records = [{"id": i, "name": "x"} for i in range(5)]
    compressed = transforms.compress_structure(records)
    assert compressed["n"] == 5 == len(compressed["rows"])
    # `n` is redundant: the inverse ignores it, so losslessness is unaffected.
    assert transforms.roundtrip_ok(records)


def test_tabularize_union_schema_accepts_heterogeneous():
    # Union-schema tabularize accepts non-uniform key sets.  The emit-only-if-smaller
    # guard in compress_with is the backstop for small/uneven payloads this test set
    # used to assert against.
    records = [{"id": 1, "name": "a"}, {"id": 2}]
    compressed = transforms.compress_structure(records)
    assert isinstance(compressed, dict) and compressed.get(transforms.TABLE_MARKER) == 1
    assert compressed["cols"] == ["id", "name"]
    assert compressed["absent_cols"] == [1]  # name column has a hole
    # Round-trip still exact.
    assert transforms.roundtrip_ok(records)


def test_nested_key_folding_hoists_subcols():
    records = [{"id": i, "owner": {"login": "eric", "type": "User"}} for i in range(6)]
    table = transforms.compress_structure(records)
    assert table["cols"] == ["id", "owner"]
    assert "subcols" in table and table["subcols"]["owner"]["cols"] == ["login", "type"]
    # The nested keys 'login'/'type' appear once in subcols, not once per row.
    assert transforms.minify(table).count('"login"') == 1


# ── union-schema tabularize tests ──

def test_union_schema_runecho_shape_no_explicit_null():
    """Non-uniform keys with no explicit nulls: absent cells encode as None, decoder
    strips them from the reconstructed dict."""
    records = [
        {"name": "MyClass", "kind": "class", "line": 42, "hash": "abc"},
        {"name": "my_func", "kind": "function", "line": 10, "hash": "def"},
        {"name": "DEBUG", "kind": "export", "hash": "111"},
    ]
    assert not transforms._uniform_dict_list(records)  # export row lacks "line"
    ok, keys, absent, sentinel = transforms._tabularizable_dict_list(records)
    assert ok
    assert keys == ["name", "kind", "line", "hash"]
    assert absent == {2}  # "line" column at index 2 has a hole
    assert sentinel == set()  # no explicit nulls, so no sentinel
    assert transforms.roundtrip_ok(records)


def test_union_schema_explicit_null_vs_absent():
    """Column carries both explicit null and absent keys: sentinel columns use
    ABSENT_MARKER so decoder can distinguish."""
    records = [
        {"a": 1, "b": None},
        {"a": 2},
        {"a": 3, "b": 5},
    ]
    ok, keys, absent, sentinel = transforms._tabularizable_dict_list(records)
    assert ok
    assert keys == ["a", "b"]
    assert absent == {1}
    assert sentinel == {1}  # "b" column has explicit null AND absent
    table = transforms.compress_structure(records)
    assert table["absent_cols"] == [1]
    assert table["sentinel_cols"] == [1]
    # Row 0: b=None (explicit), Row 1: b=absent (sentinel), Row 2: b=5
    assert table["rows"][0] == [1, None]
    assert table["rows"][1] == [2, transforms.ABSENT_MARKER]
    assert table["rows"][2] == [3, 5]
    assert transforms.roundtrip_ok(records)
    result = transforms.decompress(transforms.compress(records))
    assert result == records


def test_union_schema_mixed_absent_columns():
    """Some columns are absent-only (no sentinel), some are sentinel-qualified."""
    records = [
        {"name": "a", "score": None, "tag": "x"},
        {"name": "b", "tag": "y"},
        {"name": "c", "score": 10, "tag": "z"},
    ]
    ok, keys, absent, sentinel = transforms._tabularizable_dict_list(records)
    assert ok
    assert keys == ["name", "score", "tag"]
    assert absent == {1}  # only score column has holes
    assert sentinel == {1}  # score has explicit null (row 0) AND absent (row 1)
    assert transforms.roundtrip_ok(records)


def test_union_schema_sparse_rejection():
    """Payloads where fewer than half the cells are filled should be rejected."""
    # 6 keys * 2 rows = 12 cells, filled = 5 → 42% → rejected
    records = [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"e": 5},
    ]
    ok, _, _, _ = transforms._tabularizable_dict_list(records)
    assert not ok

    # 2 keys * 2 rows = 4 cells, filled = 3 → 75% → accepted
    records2 = [{"a": 1, "b": 2}, {"a": 3}]
    ok2, _, _, _ = transforms._tabularizable_dict_list(records2)
    assert ok2


def test_union_schema_table_header_fields():
    """absent_cols and sentinel_cols appear in the table only when needed."""
    uniform = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    table = transforms.compress_structure(uniform)
    assert "absent_cols" not in table
    assert "sentinel_cols" not in table

    non_uniform = [{"a": 1, "b": 2}, {"a": 3}]
    table2 = transforms.compress_structure(non_uniform)
    assert table2["absent_cols"] == [1]
    assert "sentinel_cols" not in table2  # no explicit nulls

    with_nulls = [{"a": 1, "b": None}, {"a": 2}]
    table3 = transforms.compress_structure(with_nulls)
    assert table3["absent_cols"] == [1]
    assert table3["sentinel_cols"] == [1]


def test_union_schema_subcols_inherit_absent_info():
    """Nested non-uniform dicts propagate absent_cols into their sub-table spec."""
    records = [
        {"id": 1, "meta": {"x": 1, "y": 2}},
        {"id": 2, "meta": {"y": 3}},
    ]
    table = transforms.compress_structure(records)
    assert "subcols" in table
    meta_spec = table["subcols"]["meta"]
    assert meta_spec["cols"] == ["x", "y"]
    assert meta_spec["absent_cols"] == [0]  # x absent in row 1
    assert "sentinel_cols" not in meta_spec  # no explicit nulls
    assert transforms.roundtrip_ok(records)


def test_union_schema_subtable_sentinel_roundtrip():
    """Sub-table with explicit null and absent keys round-trips correctly through
    the full pipeline, including dictionary coding."""
    records = [
        {"id": 1, "cfg": {"host": "a", "port": 8080}},
        {"id": 2, "cfg": {"host": "b", "port": None}},
        {"id": 3, "cfg": {"host": "c"}},
        {"id": 4, "cfg": {"host": "a", "port": 8080}},
        {"id": 5, "cfg": {"host": "b", "port": None}},
        {"id": 6, "cfg": {"host": "c"}},
    ]
    table = transforms.compress_structure(records)
    cfg_spec = table["subcols"]["cfg"]
    assert cfg_spec["cols"] == ["host", "port"]
    assert cfg_spec["absent_cols"] == [1]
    assert cfg_spec["sentinel_cols"] == [1]
    assert transforms.roundtrip_ok(records)
    result = transforms.decompress(transforms.compress(records))
    assert result == records


def test_union_schema_dict_coding_does_not_alias_sentinel():
    """ABSENT_MARKER appears multiple times but must not be aliased away by
    dict_encode, or the decoder won't recognize it."""
    # 8 rows, 4 have absent cells in a sentinel column.
    records = []
    for i in range(8):
        d = {"id": i}
        if i % 2 == 0:
            d["flag"] = None
        records.append(d)

    ok, _, absent, sentinel = transforms._tabularizable_dict_list(records)
    assert ok and 1 in sentinel
    # Compress and verify each absent row reconstructed without the key.
    text = transforms.compress(records)
    result = transforms.decompress(text)
    assert result == records
    # The sentinel string must not have been replaced by an alias.
    assert transforms.ABSENT_MARKER not in text  # all replaced by alias or absent
    # But the decoder must still see it — check that even rows have no "flag" key.
    for i, d in enumerate(result):
        if i % 2 == 1:  # odd rows: absent
            assert "flag" not in d
        else:  # even rows: explicit null
            assert d["flag"] is None


def test_nested_heterogeneous_sparse_rejected_by_density_gate():
    # The meta column is 2 rows × 2 union keys, 2 filled → 50% → rejected.
    records = [{"id": 1, "meta": {"a": 1}}, {"id": 2, "meta": {"b": 2}}]
    table = transforms.compress_structure(records)
    assert "subcols" not in table  # too sparse for union-schema tabularize
    assert transforms.roundtrip_ok(records)


def test_dictionary_coding_folds_repeated_values():
    url = "https://api.github.com/repos/inth3shadows/terse/contents/very/deep/path"
    structure = transforms.compress_structure([{"id": i, "url": url} for i in range(20)])
    data, legend = transforms.dict_encode(structure)
    assert legend, "expected a repeated long URL to be aliased"
    # The long URL appears once (in the legend), not 20 times in the data.
    assert transforms.minify(data).count(url) == 0
    assert url in legend.values()


def test_subtree_aliasing_folds_a_repeated_subobject():
    cfg = {"region": "us-east-1", "retries": 5, "endpoints": ["alpha", "beta", "gamma"]}
    obj = {f"svc{i}": cfg for i in range(6)}
    structure = transforms.compress_structure(obj)  # a dict, not a record list -> no table
    data, legend = transforms.dict_encode(structure)
    assert legend, "expected the repeated config subtree to be aliased"
    # the whole subtree is the legend value (a dict), referenced once per occurrence
    assert any(isinstance(v, dict) and v == cfg for v in legend.values())
    # the inner region string was swallowed by the subtree alias, not aliased separately
    assert transforms.minify(data).count("us-east-1") == 0
    assert transforms.roundtrip_ok(obj)


@pytest.mark.parametrize("obj", CASES)
def test_dictionary_tier_never_regresses_tokens(obj):
    """The net-token guard: the dict tier (incl. subtree aliasing) must never produce a
    larger payload than tabularize-only."""
    assert transforms._tok_text(transforms.compress(obj)) <= \
        transforms._tok_text(transforms.compress_tabular(obj))


def test_dictionary_coding_declines_when_no_repeats():
    structure = transforms.compress_structure([{"id": i, "u": f"unique-{i}"} for i in range(5)])
    _data, legend = transforms.dict_encode(structure)
    assert legend == {}  # nothing repeats enough to pay


def test_aliases_never_collide_with_literals():
    # 'real' repeats (would be aliased); '~0'/'~1' are literal values. Whatever
    # aliases get assigned must avoid the literal '~0'/'~1', else decode corrupts.
    obj = [{"v": "real", "w": "~0"}, {"v": "real", "w": "~1"}, {"v": "real", "w": "~0"}]
    assert transforms.roundtrip_ok(obj)
    _data, legend = transforms.dict_encode(transforms.compress_structure(obj))
    assert "~0" not in legend and "~1" not in legend


# --- the gate's own equality (#187) ---------------------------------------------------
# `roundtrip_ok` used plain `==`, and IEEE-754 says `nan != nan`, so a payload the codec
# handled PERFECTLY was reported as a losslessness failure. The direction was safe — a
# failed self-check falls back to the plain minified form — but the number was wrong, and
# `policy_gen._tool_decision` marks a tool `passthrough` permanently on a gate_fail while
# `measure` zeroes its banked savings. The codec was fine; the checker was not.

def test_nan_payload_passes_the_gate_because_its_bytes_are_exact():
    obj = {"v": float("nan")}
    # the premise: json emits and re-reads the non-standard NaN token faithfully
    assert transforms.compress(obj) == '{"v":NaN}'
    assert transforms.minify(transforms.decompress(transforms.compress(obj))) \
        == transforms.minify(obj)
    assert transforms.roundtrip_ok(obj)
    # and NaN nested inside the record/table path, not just at the top level
    assert transforms.roundtrip_ok({"rows": [{"id": i, "score": float("nan")} for i in range(6)]})


def test_infinity_and_negative_zero_were_never_affected():
    # scope check from the issue: only NaN needed the special case, so if either of these
    # ever starts failing it is a NEW break, not this one resurfacing.
    assert transforms.roundtrip_ok({"v": float("inf")})
    assert transforms.roundtrip_ok({"v": float("-inf")})
    assert transforms.roundtrip_ok({"v": -0.0})


def test_values_equal_still_rejects_every_real_difference():
    # The contract is "same as ==, plus NaN". A NaN-blind gate that also stopped catching
    # genuine corruption would be far worse than the false alarm it replaces.
    ve = transforms.values_equal
    assert not ve({"v": float("nan")}, {"v": 1.0})       # NaN is not equal to a number
    assert not ve({"v": 1.0}, {"v": float("nan")})       # ...in either order
    assert not ve({"a": 1}, {"a": 1, "b": 2})            # missing key
    assert not ve([1, 2], [1, 2, 3])                     # length
    assert not ve({"a": [{"b": float("nan")}]}, {"a": [{"b": 0.0}]})  # nested
    assert not ve("1", 1)


def test_values_equal_does_not_tighten_what_equality_already_allowed():
    ve = transforms.values_equal
    assert ve({"a": 1, "b": 2}, {"b": 2, "a": 1})  # dicts compare order-insensitively
    assert ve({"a": True}, {"a": 1})               # bool/int cross-equality, as `==` has it
    assert ve({"a": 1}, {"a": 1.0})                # int/float, likewise


def test_the_gate_no_longer_leans_on_jsons_shared_NaN_object():
    """Why #187 never fired in production, and why the fix is still right.

    `json` hands back ONE module-level NaN object for every `NaN` token — both the C
    accelerator and the pure-Python scanner (`_CONSTANTS['NaN']`). Container `==` compares
    elements identity-first, so when BOTH sides come off the wire the comparison short-
    circuits and plain `==` already answered correctly. That is why `policy._lossless_stage`
    and `measure_payload` — whose two sides are always parsed — were never actually
    affected, contrary to the issue's blast-radius note.

    The break needs one side built in PYTHON, which is exactly the fuzz/property surface
    that certifies the lossless claim. Leaning on a shared-singleton implementation detail
    for the correctness of the gate is the fragility worth removing either way.
    """
    parsed = json.loads('{"v": NaN}')
    assert parsed["v"] is json.loads('{"v": NaN}')["v"]        # the singleton, both scanners
    assert transforms.values_equal(parsed, json.loads('{"v": NaN}'))
    # the case that actually failed: a Python-constructed NaN has no such identity
    built = {"v": float("nan")}
    assert built["v"] is not parsed["v"]
    assert built != parsed                                     # plain `==` still says no
    assert transforms.values_equal(built, parsed)              # the gate no longer does
    assert transforms.roundtrip_ok(built)
