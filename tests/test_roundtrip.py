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
