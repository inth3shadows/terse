"""Corpus capture + shape bucketing.

The spike's verdict is only as good as the captured tools, so coverage is tracked
explicitly (see report.py) — a thin sample must not masquerade as "nothing to
compress". Shape buckets are the whole point: they expose where each tier is a
no-op (e.g. compact-JSON, single-object) versus where it pays (array-of-records).

Persistence model: one JSON envelope per payload under corpus/, named
`{tool}__{sha8}.json`. The sha of the raw bytes makes capture idempotent (the
same payload re-captured overwrites the same file) and avoids stamping a
nondeterministic timestamp into the corpus (principle #31).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .transforms import _uniform_dict_list  # the one canonical "what tabularize folds" rule

# Shape buckets. classify_shape returns one of these.
PRETTY_JSON = "pretty-json"
COMPACT_JSON = "compact-json"
ARRAY_OF_RECORDS = "array-of-records"
SINGLE_OBJECT = "single-object"
LONG_TEXT = "long-text"
OTHER = "other"

_LONG_TEXT_CHARS = 2000
_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


# Cap recursion so an adversarially/pathologically nested payload (which json.loads
# will happily parse) can't blow the stack inside the classifier; real tool output is
# shallow, and at absurd depth the tabularizer itself would also bail, so returning
# "no record list" is the safe, mirror-preserving direction (#4).
_MAX_SHAPE_DEPTH = 200


def _find_record_list(obj: Any, _depth: int = 0) -> list[dict] | None:
    """The first list-of-uniform-dicts at ANY depth in obj (depth-first), else None.

    This is exactly what `transforms.compress_structure` folds into a table — a list
    of >=2 dicts that share one key set, nested arbitrarily deep — and it reuses the
    canonical `_uniform_dict_list` rule so the shape classifier, the probe/fluency
    record extractor, and the tabularizer can never drift on what counts as
    record-shaped (the bug behind #4: three hand-rolled "mirror" checks disagreed)."""
    if _depth > _MAX_SHAPE_DEPTH:
        return None
    if isinstance(obj, list):
        if _uniform_dict_list(obj):
            return obj
        for x in obj:
            found = _find_record_list(x, _depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for v in obj.values():
            found = _find_record_list(v, _depth + 1)
            if found is not None:
                return found
    return None


def _has_record_list(obj: Any) -> bool:
    """True if obj contains, at ANY depth, a list-of-uniform-dicts (the tabularize shape)."""
    return _find_record_list(obj) is not None


def extract_records(obj: Any) -> list[dict] | None:
    """Return the list-of-uniform-dicts inside obj (at any depth), else None.

    Mirrors what the tabularizer folds, so the probes reason about the same cells.
    Uniform keys are guaranteed, so callers may index every record by the first
    record's columns without a KeyError.
    """
    return _find_record_list(obj)


def classify_shape(raw: str) -> str:
    """Bucket a raw tool-output string by structural shape.

    Heuristic and deliberately simple — the spike refines thresholds against the
    real corpus. Distinguishes pretty vs compact JSON by whitespace, and flags
    record-shaped payloads (what tabularize targets) separately from single objects.
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return LONG_TEXT if len(raw) >= _LONG_TEXT_CHARS else OTHER
    except RecursionError:
        # On the 3.11 floor, json.loads itself recurses and overflows on a deeply nested
        # payload (3.12+ parse it iteratively). Too deep to parse == can't be classified
        # or compressed, so bucket it as unparseable rather than crash the measurement.
        return LONG_TEXT if len(raw) >= _LONG_TEXT_CHARS else OTHER

    is_pretty = "\n" in raw.strip()  # indented JSON has interior newlines; a lone
    #                                   trailing newline (e.g. from `jq -c`) is not pretty

    if _has_record_list(obj):
        return ARRAY_OF_RECORDS
    if isinstance(obj, dict):
        return PRETTY_JSON if is_pretty else COMPACT_JSON
    if isinstance(obj, list):
        return PRETTY_JSON if is_pretty else COMPACT_JSON
    # bare scalar JSON (number/string/bool/null)
    return COMPACT_JSON


def _sha8(raw: str) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def capture_payload(tool: str, raw: str, corpus_dir: str | Path) -> Path:
    """Persist one captured payload as a shape-tagged envelope. Idempotent by sha."""
    corpus = Path(corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    envelope = {
        "tool": tool,
        "shape": classify_shape(raw),
        "bytes": len(raw),
        "sha": _sha8(raw),
        "raw": raw,
    }
    safe_tool = _SANITIZE.sub("_", tool).strip("_") or "unknown"
    path = corpus / f"{safe_tool}__{envelope['sha']}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Load every captured envelope from corpus/, skipping the .gitkeep placeholder."""
    corpus = Path(corpus_dir)
    out: list[dict[str, Any]] = []
    for path in sorted(corpus.glob("*.json")):
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(env, dict) and "raw" in env and "tool" in env:
            out.append(env)
    return out


def coverage(envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tool and per-shape counts — surfaced in the report so thin samples show."""
    by_tool: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for env in envelopes:
        by_tool[env["tool"]] = by_tool.get(env["tool"], 0) + 1
        by_shape[env.get("shape", "?")] = by_shape.get(env.get("shape", "?"), 0) + 1
    return {"total": len(envelopes), "by_tool": by_tool, "by_shape": by_shape}
