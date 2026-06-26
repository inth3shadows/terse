"""Corpus capture + shape bucketing.

The spike's verdict is only as good as the captured tools, so coverage is tracked
explicitly (see report.py) — a thin sample must not masquerade as "nothing to
compress". Shape buckets are the whole point: they expose where each tier is a
no-op (e.g. compact-JSON, single-object) versus where it pays (array-of-records).
"""

from __future__ import annotations

import json
from typing import Any

# Shape buckets. classify_shape returns one of these.
PRETTY_JSON = "pretty-json"
COMPACT_JSON = "compact-json"
ARRAY_OF_RECORDS = "array-of-records"
SINGLE_OBJECT = "single-object"
LONG_TEXT = "long-text"
OTHER = "other"

_LONG_TEXT_CHARS = 2000


def classify_shape(raw: str) -> str:
    """Bucket a raw tool-output string by structural shape.

    Heuristic and deliberately simple — the spike refines thresholds against the
    real corpus. Distinguishes pretty vs compact JSON by whitespace, and flags
    list-of-dicts (the bucket tabularize targets) separately from single objects.
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return LONG_TEXT if len(raw) >= _LONG_TEXT_CHARS else OTHER

    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return ARRAY_OF_RECORDS
    if isinstance(obj, dict):
        # A wrapper like {"result": [ ...records... ]} is still record-shaped.
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return ARRAY_OF_RECORDS
        looks_compact = "\n" not in raw and ": " not in raw
        return COMPACT_JSON if looks_compact else (
            COMPACT_JSON if len(raw) < 200 else PRETTY_JSON
        )
    looks_compact = "\n" not in raw
    return COMPACT_JSON if looks_compact else PRETTY_JSON


# TODO(spike): capture_from_tool(tool_name, raw) -> persist to corpus/ with shape
# tag + source tool, so coverage can be reported per tool. Wire from the CLI
# `terse capture` subcommand. See plan Section 7 "Corpus".
def capture(tool_name: str, raw: str) -> dict[str, Any]:
    """Record one captured payload with its shape + source. (Persistence: TODO.)"""
    return {"tool": tool_name, "shape": classify_shape(raw), "bytes": len(raw)}
