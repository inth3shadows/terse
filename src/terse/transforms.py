"""Tier-0 lossless transforms: minify + tabularize, with a round-trip gate.

These are the differentiated, fully-lossless core. Each transform is paired with
an exact inverse, and `roundtrip_ok` asserts decompress(compress(x)) == x over
any JSON-native value. A failing round-trip is a bug, not a tuning knob.

Not yet implemented (deferred until the spike justifies them, per the plan):
  - Tier 0.5 dictionary coding (repeated VALUES + tokenizer-aware delimiters)
  - Tier 1 lossy modes (truncate / drop-to-retrieve)
"""

from __future__ import annotations

import json
from typing import Any

# Marker for a tabularized list. Chosen to be vanishingly unlikely in real tool
# output; detable keys off it. (A production version would also guard against a
# genuine payload that happens to contain this key.)
TABLE_MARKER = "__terse_table__"


def minify(obj: Any) -> str:
    """Serialize with no insignificant whitespace. Lossless for JSON-native data.

    Inverse: json.loads. Round-trips because JSON scalar/containers survive a
    dumps->loads cycle by value (key *order* is preserved by json; dict equality
    ignores order regardless).
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_tabularizable(value: Any) -> bool:
    """True iff `value` is a list of >=2 dicts that all share an identical key set.

    The shared-keyset requirement is what makes tabularization unambiguous: one
    column header reconstructs every row. Heterogeneous lists are left untouched
    (still lossless, just no saving) rather than padded with sentinels in v1.
    """
    if not isinstance(value, list) or len(value) < 2:
        return False
    if not all(isinstance(item, dict) for item in value):
        return False
    first_keys = set(value[0].keys())
    return all(set(item.keys()) == first_keys for item in value[1:])


def compress_structure(obj: Any) -> Any:
    """Recursively fold every qualifying list-of-uniform-dicts into a table.

    Bottom-up: children are transformed first, then the (transformed) list is
    wrapped if it qualifies. Mirrored exactly by `decompress_structure`.
    """
    if isinstance(obj, dict):
        return {k: compress_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        children = [compress_structure(item) for item in obj]
        if _is_tabularizable(children):
            cols = list(children[0].keys())
            rows = [[row[c] for c in cols] for row in children]
            return {TABLE_MARKER: 1, "cols": cols, "rows": rows}
        return children
    return obj


def decompress_structure(obj: Any) -> Any:
    """Exact inverse of `compress_structure`. Top-down: unwrap, then recurse."""
    if isinstance(obj, dict):
        if obj.get(TABLE_MARKER) == 1 and "cols" in obj and "rows" in obj:
            cols = obj["cols"]
            records = [dict(zip(cols, row)) for row in obj["rows"]]
            return [decompress_structure(rec) for rec in records]
        return {k: decompress_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decompress_structure(item) for item in obj]
    return obj


def compress(obj: Any) -> str:
    """Full Tier-0 pipeline: structural fold, then minified serialization."""
    return minify(compress_structure(obj))


def decompress(text: str) -> Any:
    """Inverse of `compress`: parse, then structural unfold."""
    return decompress_structure(json.loads(text))


def roundtrip_ok(obj: Any) -> bool:
    """The lossless GATE. True iff the Tier-0 pipeline is byte-faithful by value."""
    return decompress(compress(obj)) == obj
