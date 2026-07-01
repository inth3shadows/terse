"""Tier-1 lossy field reduction — the FIRST transform that deliberately drops data.

Scope: `truncate` (self-contained) and `drop-to-retrieve` (issue #10) live here as pure
primitives. The latter takes an injected `sink`/`resolve` so the stateful store and the
retrieve tool stay in the proxy, not this layer — this module never holds session state.
`summarize` (needs a model) is still deferred, only parsed and warned by the policy layer.

Because data is dropped, the lossless round-trip gate no longer applies. Each lossy mode
has a deterministic replacement gate asserting the ONLY differences between the original
and the output are at fields the policy explicitly marked as lossy — never a `critical`
field, never an unmarked one — and that each marked change is *valid* for that mode:
  - `truncate` -> `acceptable_loss`: each change is a prefix of the original plus an
    explicit loss annotation.
  - `drop-to-retrieve` -> `droppable_loss`: each change is a handle marker that `resolve`s
    back to the EXACT original value, i.e. recoverable == acceptable; or a value left in
    place because it was under the size floor.
Everything is fail-closed: a path that doesn't resolve, an unrecoverable handle, or a gate
failure skips the lossy step and keeps the lossless output — the caller decides.

Field paths are an explicit, small subset: `a`, `a.b`, `a[].b`, `[].b`. `[]` iterates a
list; anything else is a dict key. No globbing in v1 — unknown shape raises PathError.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterator

DEFAULT_MAX = 120  # default truncate length when a field spec omits "max"

# drop-to-retrieve (#10): a field marked {"lossy":"drop-to-retrieve"} is replaced inline
# by this marker, and the original is persisted to an injected sink to be served back when
# the model calls the retrieve tool. The handle is content-addressed => deterministic,
# dedups equal values, no RNG. A value whose serialized form is under DROP_MIN is left in
# place: a retrieve round-trip isn't worth saving a handful of tokens.
DROP_KEY = "__terse_dropped__"
RETRIEVE_TOOL = "terse.retrieve"
DEFAULT_DROP_MIN = 200  # min serialized length (chars) of a value worth dropping
HANDLE_LEN = 12         # sha1 hex prefix; 48 bits, ample for a per-session store

# A loss annotation is appended so a reader (human or model) sees the field was cut and
# by how much. Distinctive enough not to collide with real content.
_STR_MARK = "…⟨+{n} chars⟩"
_LIST_MARK = "…⟨+{n} items⟩"


class PathError(ValueError):
    """A field path didn't resolve against the payload's shape — fail closed."""


def _parse_path(path: str) -> list[str]:
    """'result[].body' -> ['result', '[]', 'body']. '[]' is a list-iteration step."""
    tokens = [t for t in path.replace("[]", ".[].").split(".") if t != ""]
    if not tokens:
        raise PathError(f"empty field path: {path!r}")
    return tokens


def _truncate(value: Any, max_len: int) -> Any:
    """Keep the first max_len chars (string) or items (list), annotating the loss.
    Non-truncatable scalars (number/bool/dict/None) pass through unchanged."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + _STR_MARK.format(n=len(value) - max_len)
    if isinstance(value, list) and len(value) > max_len:
        return value[:max_len] + [_LIST_MARK.format(n=len(value) - max_len)]
    return value


def _is_truncation(orig: Any, out: Any, max_len: int) -> bool:
    """True iff `out` is exactly what truncating `orig` at max_len would produce."""
    return out == _truncate(orig, max_len)


def _apply_at(node: Any, tokens: list[str], fn: Callable[[Any], Any]) -> Any:
    """Return a copy of `node` with `fn` applied at the leaf/leaves named by tokens.
    A missing dict key is a no-op (the field isn't in this payload). A `[]` step on a
    non-list, or a key step on a non-dict, is a shape mismatch -> PathError (fail closed)."""
    if not tokens:
        return fn(node)
    head, rest = tokens[0], tokens[1:]
    if head == "[]":
        if not isinstance(node, list):
            raise PathError(f"expected a list at '[]', got {type(node).__name__}")
        return [_apply_at(x, rest, fn) for x in node]
    if not isinstance(node, dict):
        raise PathError(f"expected an object at {head!r}, got {type(node).__name__}")
    if head not in node:
        return node
    new = dict(node)
    new[head] = _apply_at(node[head], rest, fn)
    return new


def _copy_at(dst: Any, src: Any, tokens: list[str]) -> Any:
    """Return a copy of `dst` with the value(s) at `tokens` replaced by `src`'s value(s)
    at the same location. Used by the gate to prove ONLY marked paths differ."""
    if not tokens:
        return src
    head, rest = tokens[0], tokens[1:]
    if head == "[]":
        if not (isinstance(dst, list) and isinstance(src, list) and len(dst) == len(src)):
            raise PathError("list length/shape mismatch at '[]'")
        return [_copy_at(d, s, rest) for d, s in zip(dst, src)]
    if not (isinstance(dst, dict) and isinstance(src, dict)):
        raise PathError(f"object mismatch at {head!r}")
    if head not in dst:
        return dst
    new = dict(dst)
    new[head] = _copy_at(dst[head], src.get(head) if isinstance(src, dict) else None, rest)
    return new


def _leaf_pairs(orig: Any, out: Any, tokens: list[str]) -> Iterator[tuple[Any, Any]]:
    """Yield (orig_leaf, out_leaf) pairs at the path, walking both in lockstep."""
    if not tokens:
        yield orig, out
        return
    head, rest = tokens[0], tokens[1:]
    if head == "[]":
        if not (isinstance(orig, list) and isinstance(out, list) and len(orig) == len(out)):
            raise PathError("list length/shape mismatch at '[]'")
        for o, t in zip(orig, out):
            yield from _leaf_pairs(o, t, rest)
        return
    if not (isinstance(orig, dict) and isinstance(out, dict)):
        raise PathError(f"object mismatch at {head!r}")
    if head in orig and head in out:
        yield from _leaf_pairs(orig[head], out[head], rest)


def critical_paths(rule: Any) -> set[str]:
    """Field paths marked `{"critical": true}` — the denylist that lossy never touches."""
    return {p for p, s in rule.fields.items() if isinstance(s, dict) and s.get("critical")}


def _truncate_specs(rule: Any) -> list[tuple[str, dict]]:
    """The (path, spec) entries that are truncate AND not marked critical."""
    crit = critical_paths(rule)
    return [(p, s) for p, s in rule.fields.items()
            if isinstance(s, dict) and s.get("lossy") == "truncate" and p not in crit]


def apply_lossy(obj: Any, rule: Any) -> Any:
    """Apply every truncate spec to `obj`, returning a new structure. Critical fields are
    never touched. Raises PathError if a path doesn't resolve (caller falls back)."""
    out = obj
    for path, spec in _truncate_specs(rule):
        max_len = int(spec.get("max", DEFAULT_MAX))
        out = _apply_at(out, _parse_path(path), lambda v, m=max_len: _truncate(v, m))
    return out


def acceptable_loss(orig: Any, out: Any, rule: Any) -> bool:
    """The lossy GATE (replaces the round-trip gate once data is dropped). True iff:
      1. the ONLY differences between orig and out are at truncate-marked, non-critical
         paths (rebuilding orig with out's marked values yields out exactly), and
      2. every marked leaf is a valid truncation of the original.
    Any PathError (shape mismatch) is treated as unacceptable -> fail closed."""
    try:
        rebuilt = orig
        for path, _ in _truncate_specs(rule):
            rebuilt = _copy_at(rebuilt, out, _parse_path(path))
        if rebuilt != out:
            return False  # something other than marked paths changed
        for path, spec in _truncate_specs(rule):
            max_len = int(spec.get("max", DEFAULT_MAX))
            for o, t in _leaf_pairs(orig, out, _parse_path(path)):
                if not _is_truncation(o, t, max_len):
                    return False
    except PathError:
        return False
    return True


# --- drop-to-retrieve (#10): pure primitives; store + retrieve tool live in the proxy ---
def _serialize(value: Any) -> str:
    """Canonical serialization used for both the handle and the size floor. A bare string
    is measured as-is; anything else as compact, key-sorted JSON (deterministic)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _handle(tool: str, path: str, serialized: str) -> str:
    """Content-addressed handle for a dropped value. Includes tool+path so identical bytes
    under different fields get distinct handles (clearer provenance), and is stable across
    runs (no RNG) so the same value dropped twice reuses one store slot."""
    digest = hashlib.sha1(f"{tool}\x00{path}\x00{serialized}".encode("utf-8")).hexdigest()
    return digest[:HANDLE_LEN]


def _drop(value: Any, tool: str, path: str, min_len: int,
          sink: Callable[[str, Any], None]) -> Any:
    """Persist `value` via `sink` and return the inline handle marker — unless its
    serialized form is under `min_len`, in which case it stays put (not worth a round-trip)."""
    serialized = _serialize(value)
    if len(serialized) < min_len:
        return value
    handle = _handle(tool, path, serialized)
    sink(handle, value)
    return {DROP_KEY: handle, "bytes": len(serialized), "retrieve": RETRIEVE_TOOL}


def _is_drop_marker(v: Any) -> bool:
    return isinstance(v, dict) and DROP_KEY in v


def _drop_specs(rule: Any) -> list[tuple[str, dict]]:
    """The (path, spec) entries that are drop-to-retrieve AND not marked critical."""
    crit = critical_paths(rule)
    return [(p, s) for p, s in rule.fields.items()
            if isinstance(s, dict) and s.get("lossy") == "drop-to-retrieve" and p not in crit]


def apply_drops(obj: Any, rule: Any, tool: str, sink: Callable[[str, Any], None]) -> Any:
    """Replace every drop-marked, non-critical field of `obj` with a handle marker,
    persisting each original via `sink(handle, value)`. Returns a new structure; critical
    fields are never touched. Raises PathError if a path doesn't resolve (caller falls back)."""
    out = obj
    for path, spec in _drop_specs(rule):
        min_len = int(spec.get("min", DEFAULT_DROP_MIN))
        out = _apply_at(out, _parse_path(path),
                        lambda v, _p=path, _m=min_len: _drop(v, tool, _p, _m, sink))
    return out


def _is_drop(orig_leaf: Any, out_leaf: Any, resolve: Callable[[str], Any]) -> bool:
    """A single dropped leaf is acceptable iff it is either left in place (under the size
    floor) or replaced by a marker whose handle `resolve`s back to the EXACT original."""
    if out_leaf == orig_leaf:
        return True  # under the floor: untouched, trivially lossless
    if not _is_drop_marker(out_leaf):
        return False
    try:
        return resolve(out_leaf[DROP_KEY]) == orig_leaf
    except KeyError:
        return False  # handle doesn't resolve -> not recoverable -> fail closed


def droppable_loss(orig: Any, out: Any, rule: Any, resolve: Callable[[str], Any]) -> bool:
    """The drop GATE (the analogue of `acceptable_loss` for drop-to-retrieve). True iff:
      1. the ONLY differences between orig and out are at drop-marked, non-critical paths, and
      2. every marked leaf is recoverable — its handle `resolve`s to the exact original (or
         the leaf was left in place because it was under the size floor).
    `resolve(handle)` returns the stored original or raises KeyError. Any KeyError or
    PathError is treated as unrecoverable -> fail closed."""
    try:
        rebuilt = orig
        for path, _ in _drop_specs(rule):
            rebuilt = _copy_at(rebuilt, out, _parse_path(path))
        if rebuilt != out:
            return False  # something other than marked paths changed
        for path, _ in _drop_specs(rule):
            for o, t in _leaf_pairs(orig, out, _parse_path(path)):
                if not _is_drop(o, t, resolve):
                    return False
    except PathError:
        return False
    return True
