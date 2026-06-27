"""Lossless transforms with a round-trip gate.

Tier 0   — minify (whitespace) + tabularize (fold repeated KEYS of record arrays)
Tier 0.5 — dictionary coding (fold repeated VALUES via an inline legend)

Each transform is paired with an exact inverse; `roundtrip_ok` asserts
decompress(compress(x)) == x over any JSON-native value. A failing round-trip is a
bug, not a tuning knob — token availability changes WHICH values get aliased, never
losslessness.

Dictionary coding folds repeated value-strings AND repeated whole subtrees (dicts /
lists) into the legend, keyed by canonical form. It stays model-legible: the legend
ships inline with the data, so a `~0` reference is resolved by reading the same payload
— never an out-of-band retrieve (the headroom failure mode). Aliases come from a sigil
namespace proven disjoint from every literal string in the payload, so decode is an
exact lookup. The dict tier is also size-guarded: it is committed only when it actually
reduces tokens, so it can never regress a payload (losslessness is separate, and
absolute — the round-trip gate).

Not yet built (deferred, per the plan): cross-call diffing; Tier 1 lossy modes
(truncate / drop-to-retrieve).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .tokenize import count_cl100k

# Structural markers. Chosen to be vanishingly unlikely in real tool output.
TABLE_MARKER = "__terse_table__"
DICT_MARKER = "__terse_dict__"
ALIAS_SIGIL = "~"


# --------------------------------------------------------------------------- #
# minify
# --------------------------------------------------------------------------- #
def minify(obj: Any) -> str:
    """Serialize with no insignificant whitespace. Lossless for JSON-native data."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Tier 0 — tabularize (fold repeated keys, including nested dict-columns)
# --------------------------------------------------------------------------- #
def _uniform_dict_list(value: Any) -> bool:
    """True iff `value` is a list of >=2 dicts that all share an identical key set."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    if not all(isinstance(item, dict) for item in value):
        return False
    first_keys = set(value[0].keys())
    return all(set(item.keys()) == first_keys for item in value[1:])


def _fold_records(records: list[dict]) -> tuple[dict, list]:
    """Fold a uniform-dict list into (spec, positional rows), recursing on dict-columns.

    A column whose values are themselves all uniform dicts is hoisted: its key-set
    moves to spec['subcols'][col] once, and each cell becomes a positional tuple.
    Non-dict columns are recursed through compress_structure (so a list-of-dicts cell
    becomes its own sub-table). spec = {'cols': [...], 'subcols': {col: spec, ...}}.
    """
    keys = list(records[0].keys())
    posrows = [[rec[k] for k in keys] for rec in records]
    subcols: dict = {}
    n = len(records)
    for ci, k in enumerate(keys):
        col = [posrows[ri][ci] for ri in range(n)]
        if _uniform_dict_list(col):
            sub_spec, sub_pos = _fold_records(col)
            subcols[k] = sub_spec
            for ri in range(n):
                posrows[ri][ci] = sub_pos[ri]
        else:
            for ri in range(n):
                posrows[ri][ci] = compress_structure(posrows[ri][ci])
    spec: dict = {"cols": keys}
    if subcols:
        spec["subcols"] = subcols
    return spec, posrows


def compress_structure(obj: Any) -> Any:
    """Recursively fold every qualifying list-of-uniform-dicts into a table,
    hoisting nested uniform-dict columns into a shared subcols header."""
    if isinstance(obj, dict):
        return {k: compress_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if _uniform_dict_list(obj):
            spec, posrows = _fold_records(obj)
            # `n` is a redundant row-count hint: it lets a reader self-check that it
            # enumerated every row (fidelity probe found terse's only recall gap was
            # under-enumeration of wide positional tables). decompress ignores it, so
            # the round-trip stays exact.
            table = {TABLE_MARKER: 1, "n": len(posrows), "cols": spec["cols"], "rows": posrows}
            if "subcols" in spec:
                table["subcols"] = spec["subcols"]
            return table
        return [compress_structure(item) for item in obj]
    return obj


def _unfold_row(row: list, cols: list, subcols: dict) -> dict:
    """Rebuild one record from a positional row + its (possibly nested) header."""
    rec = {}
    for ci, k in enumerate(cols):
        cell = row[ci]
        sub = subcols.get(k)
        if sub is not None:
            rec[k] = _unfold_row(cell, sub["cols"], sub.get("subcols", {}))
        else:
            rec[k] = decompress_structure(cell)
    return rec


def decompress_structure(obj: Any) -> Any:
    """Exact inverse of `compress_structure`. Top-down: unwrap, then recurse."""
    if isinstance(obj, dict):
        if obj.get(TABLE_MARKER) == 1 and "cols" in obj and "rows" in obj:
            cols = obj["cols"]
            subcols = obj.get("subcols", {})
            return [_unfold_row(row, cols, subcols) for row in obj["rows"]]
        return {k: decompress_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decompress_structure(item) for item in obj]
    return obj


# --------------------------------------------------------------------------- #
# Tier 0.5 — dictionary coding (fold repeated values)
# --------------------------------------------------------------------------- #
def _tok_text(text: str) -> int:
    """Token cost of a literal text; len-based fallback if no tiktoken."""
    c = count_cl100k(text)
    return c if c is not None else max(1, len(text) // 4)


def _tok(s: str) -> int:
    """Token cost of a string VALUE (i.e. JSON-quoted), incl. the alias sigils."""
    return _tok_text(json.dumps(s, ensure_ascii=False))


# A candidate is keyed by ("s", literal_string) or ("j", canonical_minified_json) so
# strings and whole subtrees share one dedup/aliasing path. The canonical form is the
# subtree's minified JSON — equal-by-value subtrees with the same key order collapse;
# a different key order is just a missed fold, never a correctness risk (the legend
# stores the real node, so decode is exact).
def _node_tok(key: tuple) -> int:
    """Token cost of a candidate in its VALUE position: a quoted string, or the raw
    (already-minified) JSON of a subtree."""
    kind, payload = key
    return _tok(payload) if kind == "s" else _tok_text(payload)


def _count_value_nodes(node: Any, counter: Counter) -> None:
    """Count VALUE-position nodes (not dict keys) by canonical form, recursively.
    Strings count as ("s", str); dicts/lists count as ("j", minified) AND recurse,
    so a repeated whole subtree and a repeated string inside it are both seen."""
    if isinstance(node, str):
        counter[("s", node)] += 1
    elif isinstance(node, list):
        counter[("j", minify(node))] += 1
        for x in node:
            _count_value_nodes(x, counter)
    elif isinstance(node, dict):
        counter[("j", minify(node))] += 1
        for v in node.values():
            _count_value_nodes(v, counter)
    # scalars (int/float/bool/None) are too cheap to alias


def _collect_all_strings(node: Any, out: set) -> None:
    """All strings anywhere (keys + values) — the avoid-set aliases must stay clear of."""
    if isinstance(node, str):
        out.add(node)
    elif isinstance(node, list):
        for x in node:
            _collect_all_strings(x, out)
    elif isinstance(node, dict):
        for k, v in node.items():
            out.add(k)
            _collect_all_strings(v, out)


def _b36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out


def _alias_gen(avoid: set):
    """Yield ~-sigil aliases guaranteed not to collide with any literal string."""
    i = 0
    while True:
        a = ALIAS_SIGIL + _b36(i)
        i += 1
        if a not in avoid:
            yield a


def _replace_nodes(node: Any, alias_for_str: dict, alias_for_json: dict) -> Any:
    """Replace value-position strings AND whole subtrees with their alias; keys left
    untouched. Top-down: a matched container is replaced before descending, so an
    aliased subtree is folded as a unit (nested candidates inside it are captured by
    the parent's legend entry, never double-aliased)."""
    if isinstance(node, str):
        return alias_for_str.get(node, node)
    if isinstance(node, (list, dict)):
        alias = alias_for_json.get(minify(node))
        if alias is not None:
            return alias
        if isinstance(node, list):
            return [_replace_nodes(x, alias_for_str, alias_for_json) for x in node]
        return {k: _replace_nodes(v, alias_for_str, alias_for_json) for k, v in node.items()}
    return node


def _collect_used_aliases(node: Any, legend: dict, out: set) -> None:
    """Aliases actually referenced in the (replaced) data. Legend values are stored
    literal — they hold no aliases — so references live only here."""
    if isinstance(node, str):
        if node in legend:
            out.add(node)
    elif isinstance(node, list):
        for x in node:
            _collect_used_aliases(x, legend, out)
    elif isinstance(node, dict):
        for v in node.values():
            _collect_used_aliases(v, legend, out)


def dict_encode(structure: Any) -> tuple[Any, dict]:
    """Fold repeated value-strings AND repeated whole subtrees into an inline legend.
    Returns (data, legend).

    Tokenizer-aware: a node is aliased only when (n-1)*tok(node) exceeds the legend +
    reference cost. legend maps alias -> original string-or-subtree; an empty legend
    means dictionary coding didn't pay. Unused aliases (a string whose every occurrence
    was swallowed by an aliased parent subtree) are pruned so they never cost tokens.
    """
    counts: Counter = Counter()
    _count_value_nodes(structure, counts)
    candidates = [(key, n) for key, n in counts.items()
                  if n >= 2 and (n - 1) * _node_tok(key) > 0]
    if not candidates:
        return structure, {}

    # Biggest potential first, so the cheapest aliases land on the biggest wins.
    candidates.sort(key=lambda kn: (kn[1] - 1) * _node_tok(kn[0]), reverse=True)

    avoid: set = set()
    _collect_all_strings(structure, avoid)
    gen = _alias_gen(avoid)

    alias_for_str: dict = {}
    alias_for_json: dict = {}
    legend: dict = {}
    for key, n in candidates:
        alias = next(gen)
        t = _node_tok(key)
        # Exact saving with this alias's real token cost: occurrences collapse to the
        # alias (n * ac), plus one legend entry (alias + value ~= ac + t).
        saving = (n * t) - (n * _tok(alias) + _tok(alias) + t)
        if saving <= 0:
            continue
        kind, payload = key
        if kind == "s":
            alias_for_str[payload] = alias
            legend[alias] = payload
        else:
            alias_for_json[payload] = alias
            legend[alias] = json.loads(payload)  # the real subtree, restored exactly

    if not (alias_for_str or alias_for_json):
        return structure, {}
    data = _replace_nodes(structure, alias_for_str, alias_for_json)
    used: set = set()
    _collect_used_aliases(data, legend, used)
    legend = {a: v for a, v in legend.items() if a in used}
    if not legend:
        return structure, {}
    return data, legend


def dict_decode(node: Any, legend: dict) -> Any:
    """Exact inverse of dict_encode's replacement: expand value-position aliases,
    including aliases that expand to whole subtrees. Legend values are alias-free, so
    the recursion into an expanded value terminates immediately."""
    if isinstance(node, str):
        return dict_decode(legend[node], legend) if node in legend else node
    if isinstance(node, list):
        return [dict_decode(x, legend) for x in node]
    if isinstance(node, dict):
        return {k: dict_decode(v, legend) for k, v in node.items()}
    return node


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def compress_tabular(obj: Any) -> str:
    """Tier-0 only (minify + tabularize), no dictionary coding. For measurement."""
    return minify(compress_structure(obj))


def compress_with(obj: Any, tabularize: bool = True, dictionary: bool = True) -> str:
    """Apply a selectable subset of lossless tiers, then minify.

    `decompress` auto-detects the markers, so any combination round-trips. minify is
    always applied (it is the serialization). Pass both False for minify-only.
    """
    structure = compress_structure(obj) if tabularize else obj
    base = minify(structure)
    if dictionary:
        data, legend = dict_encode(structure)
        if legend:
            coded = minify({DICT_MARKER: 1, "legend": legend, "data": data})
            # Net-token guard: with whole-subtree aliasing the per-candidate estimate
            # can mis-rank under nesting overlap, so commit the dict block only when it
            # is actually smaller. Losslessness is independent (the round-trip gate);
            # this guards SIZE — the dict tier can never regress the payload.
            if _tok_text(coded) < _tok_text(base):
                return coded
    return base


def compress(obj: Any) -> str:
    """Full pipeline: tabularize, then dictionary-code, then minify."""
    return compress_with(obj, tabularize=True, dictionary=True)


def decompress(text: str) -> Any:
    """Inverse of `compress`: parse, expand legend (if any), structural unfold."""
    parsed = json.loads(text)
    if isinstance(parsed, dict) and parsed.get(DICT_MARKER) == 1:
        data = dict_decode(parsed["data"], parsed["legend"])
        return decompress_structure(data)
    return decompress_structure(parsed)


def roundtrip_ok(obj: Any) -> bool:
    """The lossless GATE. True iff the full pipeline is byte-faithful by value."""
    return decompress(compress(obj)) == obj
