"""The lossless gate as a test suite: every Tier-0 transform must round-trip.

A failing case here means terse dropped data it promised to keep — the one thing
the design forbids. These run over shapes that exercise minify and the recursive
tabularizer, including the cases where tabularize must DECLINE (heterogeneous
lists) and still stay lossless.
"""

from __future__ import annotations

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


def test_tabularize_declines_heterogeneous():
    records = [{"id": 1, "name": "a"}, {"id": 2}]
    compressed = transforms.compress_structure(records)
    assert isinstance(compressed, list)  # left untouched, not wrapped


def test_nested_key_folding_hoists_subcols():
    records = [{"id": i, "owner": {"login": "eric", "type": "User"}} for i in range(6)]
    table = transforms.compress_structure(records)
    assert table["cols"] == ["id", "owner"]
    assert "subcols" in table and table["subcols"]["owner"]["cols"] == ["login", "type"]
    # The nested keys 'login'/'type' appear once in subcols, not once per row.
    assert transforms.minify(table).count('"login"') == 1


def test_nested_heterogeneous_columns_not_folded():
    records = [{"id": 1, "meta": {"a": 1}}, {"id": 2, "meta": {"b": 2}}]
    table = transforms.compress_structure(records)
    assert "subcols" not in table  # differing inner keys -> left as dicts in cells


def test_dictionary_coding_folds_repeated_values():
    url = "https://api.github.com/repos/inth3shadows/terse/contents/very/deep/path"
    structure = transforms.compress_structure([{"id": i, "url": url} for i in range(20)])
    data, legend = transforms.dict_encode(structure)
    assert legend, "expected a repeated long URL to be aliased"
    # The long URL appears once (in the legend), not 20 times in the data.
    assert transforms.minify(data).count(url) == 0
    assert url in legend.values()


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
