"""Tier 0.7 (text): content-defined-chunking diff for non-JSON tool output (#25).

Cross-call diffing (`transforms.diff_encode`) only reasons about a list of uniform
dict records -- the shape `capture.classify_shape`/`measure.measure_payload` calls
"array-of-records". File reads, source excerpts, and log tails are none of that: they
are opaque text, so today they get `applicable: False` in `measure_payload` and zero
compression on every re-read, even in a tight debug loop that reads the same
(mostly-unchanged) file or log tail repeatedly.

Line-based diffing doesn't fit here: terse controls no line-length contract for
arbitrary tool output, and a single inserted character mid-line would show a whole
line as "changed" under a naive line diff. Content-defined chunking instead cuts
chunk boundaries wherever a rolling hash over a sliding window satisfies a cheap
condition, so a boundary depends only on nearby CONTENT, never on position. An edit
anywhere in the text only ever perturbs the chunk(s) it overlaps; every chunk before
and after it re-syncs and is identical to a chunk already seen in the prior result, so
it is representable as a reference instead of literal text -- the same rsync/restic
insight, applied to a per-tool cross-call diff instead of a file-transfer delta.

Operates on Python `str` (code points), not raw UTF-8 bytes: a chunk boundary can then
never land inside a multi-byte character, so every literal chunk is independently a
valid JSON string with no re-encoding step.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DIFF_MARKER = "__terse_textdiff__"

# Rolling-hash CDC parameters. WINDOW characters are considered per hash; a boundary is
# cut once a chunk is at least MIN_CHUNK long and either hits MAX_CHUNK (a hard cap, so
# one pathological span can't swallow the whole payload) or its rolling hash's low bits
# are all zero (~1-in-2**MASK_BITS chance per position, giving an average chunk size in
# that neighborhood). Tuned small -- not the ~8KB average typical of file-dedup CDC --
# because tool payloads here are KB-scale text, not GB-scale files, and a debug-loop
# re-read often changes one line among hundreds.
_WINDOW = 32
_MIN_CHUNK = 24
_MAX_CHUNK = 192
_MASK_BITS = 6
_BASE = 257
_MOD = (1 << 61) - 1  # a Mersenne prime; keeps the rolling hash cheap and well-spread
_POW = pow(_BASE, _WINDOW - 1, _MOD)
_MASK = (1 << _MASK_BITS) - 1


def _chunk(text: str) -> list[str]:
    """Split text into content-defined chunks. Deterministic and position-independent:
    the same substring chunks the same way wherever it occurs, so an edit only ever
    invalidates the chunk(s) it overlaps."""
    n = len(text)
    if n == 0:
        return []
    if n <= _MIN_CHUNK:
        return [text]
    bounds: list[int] = []
    start = 0
    h = 0
    for i in range(n):
        length = i - start + 1
        if length > _WINDOW:
            outgoing = ord(text[i - _WINDOW])
            h = (h - outgoing * _POW) % _MOD
        h = (h * _BASE + ord(text[i])) % _MOD
        if length >= _MIN_CHUNK and (length >= _MAX_CHUNK or (length >= _WINDOW and h & _MASK == 0)):
            bounds.append(i + 1)
            start = i + 1
            h = 0
    if start < n:
        bounds.append(n)
    out = []
    prev = 0
    for b in bounds:
        out.append(text[prev:b])
        prev = b
    return out


def _fingerprint(chunk: str) -> str:
    return hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:16]


def text_diff_encode(prev: str, curr: str) -> dict[str, Any] | None:
    """A self-describing lossless diff of curr against prev's chunks, or None if
    nothing representable applies. Every returned diff is proven exact against `curr`
    before it is returned, so a caller never has to trust the greedy match itself --
    a hash collision between two different chunks just fails the check and falls
    through to None (fail-closed, same contract as `transforms.diff_encode`)."""
    prev_chunks = _chunk(prev)
    if not prev_chunks:
        return None
    curr_chunks = _chunk(curr)

    by_fp: dict[str, int] = {}
    for idx, ch in enumerate(prev_chunks):
        by_fp.setdefault(_fingerprint(ch), idx)  # first occurrence is enough to reference

    ops: list[list] = []
    literal: list[str] = []

    def _flush() -> None:
        if literal:
            ops.append(["+", "".join(literal)])
            literal.clear()

    for ch in curr_chunks:
        idx = by_fp.get(_fingerprint(ch))
        if idx is not None and prev_chunks[idx] == ch:
            _flush()
            if ops and ops[-1][0] == "=" and ops[-1][2] == idx - 1:
                ops[-1][2] = idx
            else:
                ops.append(["=", idx, idx])
            continue
        literal.append(ch)
    _flush()

    diff = {DIFF_MARKER: 1, "ops": ops}
    try:
        if text_diff_decode(prev, diff) == curr:
            return diff
    except (IndexError, KeyError, ValueError):
        pass
    return None


def text_diff_decode(prev: str, diff: dict[str, Any]) -> str:
    """Exact inverse of text_diff_encode: rebuild curr from prev + the diff ops."""
    prev_chunks = _chunk(prev)
    out = []
    for op in diff["ops"]:
        if op[0] == "=":
            out.append("".join(prev_chunks[op[1]:op[2] + 1]))
        elif op[0] == "+":
            out.append(op[1])
        else:
            raise ValueError(f"unknown text-diff op: {op[0]!r}")
    return "".join(out)


def text_diff_roundtrip_ok(prev: str, curr: str) -> bool:
    """The lossless GATE for text diffing: True iff a diff exists and rebuilds curr
    exactly."""
    diff = text_diff_encode(prev, curr)
    return diff is not None and text_diff_decode(prev, diff) == curr


def text_diff_wire(prev: str, curr: str, tool: str = "") -> str | None:
    """The model-facing text-diff envelope, or None if no lossless diff applies. Mirrors
    `transforms.diff_wire`'s self-describing shape: a base anchor hash of the prior text
    (already visible to the model) plus ops it applies inline against that prior text."""
    diff = text_diff_encode(prev, curr)
    if diff is None:
        return None
    base = hashlib.sha1(prev.encode("utf-8")).hexdigest()[:8]
    label = f" {tool}" if tool else ""
    note = (f"Text diff of the previous{label} result above: rebuild by processing "
            '"ops" in order -- ["=",a,b] copies chunks a..b of the PRIOR text (same '
            'content-defined chunking, so chunk indices are stable), ["+",s] inserts '
            "literal text s -- and concatenating every piece.")
    return json.dumps({**diff, "of": tool, "base": base, "note": note},
                       separators=(",", ":"), ensure_ascii=False)
