"""Tier-1 lossy field reduction — the FIRST transform that deliberately drops data.

Scope (issue #2, truncate-first slice): `truncate` only. `summarize` (needs a model)
and `drop-to-retrieve` (needs a stateful store + a retrieve tool) are deferred to their
own issues; the policy layer still parses and warns about them.

Because data is dropped, the lossless round-trip gate no longer applies. Its replacement
is `acceptable_loss`: a deterministic invariant that the ONLY differences between the
original and the lossy output are at fields the policy explicitly marked lossy (never a
`critical` field, never an unmarked one), and that each marked change is a *valid*
truncation — a prefix of the original plus an explicit loss annotation. Everything is
fail-closed: a path that doesn't resolve, or a gate failure, skips the lossy step and
keeps the lossless output (the caller decides).

Field paths are an explicit, small subset: `a`, `a.b`, `a[].b`, `[].b`. `[]` iterates a
list; anything else is a dict key. No globbing in v1 — unknown shape raises PathError.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

DEFAULT_MAX = 120  # default truncate length when a field spec omits "max"

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
