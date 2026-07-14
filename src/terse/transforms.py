"""Lossless transforms with a round-trip gate.

Tier 0   — minify (whitespace) + tabularize (fold repeated KEYS of record arrays)
Tier 0.5 — dictionary coding (fold repeated VALUES via an inline legend)
Tier 0.7 — cross-call diffing (encode a result as a lossless delta vs the prior
           same-tool result; stateful, applied by the proxy, opt-in)

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

Tier 1 lossy lives in `lossy.py` (truncate built; summarize / drop-to-retrieve deferred)
— it operates on the parsed object BEFORE these lossless tiers serialize it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any

from .tokenize import count_cl100k

# Structural markers. Chosen to be vanishingly unlikely in real tool output.
TABLE_MARKER = "__terse_table__"
DICT_MARKER = "__terse_dict__"
DIFF_MARKER = "__terse_diff__"
# The drop-to-retrieve inline handle marker (#10). Not a transforms envelope — it is
# produced by the lossy layer and consumed by the proxy's retrieve handler — but it lives
# in this registry so all `__terse_*` wire keys have one home and are reserved together.
DROPPED_MARKER = "__terse_dropped__"
ALIAS_SIGIL = "~"

# Keys reserved for terse's own envelopes. A real payload that already contains one
# can't be safely compressed: the consumer reads these markers per the format primer,
# so it would mis-reconstruct the user's literal dict as a terse envelope. The codec
# has no escape convention, so the only lossless move is to leave such a payload alone.
_RESERVED_MARKERS = frozenset({TABLE_MARKER, DICT_MARKER, DIFF_MARKER, DROPPED_MARKER})


def has_terse_marker(obj: Any) -> bool:
    """True if obj contains, at ANY depth, a dict key reserved for a terse envelope.

    decompress / the model's primer interpret these markers wherever they appear, so a
    collision anywhere — not just top-level — makes a payload unsafe to compress."""
    if isinstance(obj, dict):
        if not _RESERVED_MARKERS.isdisjoint(obj.keys()):
            return True
        return any(has_terse_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_terse_marker(x) for x in obj)
    return False


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
# Cross-call diffing (lossless) — encode curr as a delta against the prior same-tool
# result. The 91% same-tool token overlap the ceiling probe measured is the headroom.
#
# Self-describing, like every other tier: the diff names the prior result it bases on
# and carries the changes inline, so the model reads it against the previous turn's
# result already in its context — never an out-of-band retrieve. A diff is accepted
# ONLY if it reconstructs curr EXACTLY (verified at encode time), so it is lossless by
# construction. When no representable diff applies, `diff_encode` returns None and the
# caller falls back to the full compressed form (the dangling-reference fallback).
# --------------------------------------------------------------------------- #
def _locate_records(obj: Any) -> tuple[Any, list[dict]] | None:
    """(at, records) for the list-of-uniform-dicts in obj — `at` is None for a top-level
    list or the dict key that holds it. None if obj has no record list (mirrors what
    tabularize folds, so the diff reasons about the same rows)."""
    if isinstance(obj, list) and len(obj) >= 2 and all(isinstance(x, dict) for x in obj):
        return (None, obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, dict) for x in v):
                return (k, v)
    return None


def _diff_id_col(prev_recs: list[dict], curr_recs: list[dict]) -> str | None:
    """A column present in every record of both lists whose values are scalar (str/int)
    and unique within each list — usable to align rows across the two calls."""
    for c in prev_recs[0]:
        if not (all(c in r for r in prev_recs) and all(c in r for r in curr_recs)):
            continue
        pv = [r[c] for r in prev_recs]
        cv = [r[c] for r in curr_recs]
        vals = pv + cv
        if (all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in vals)
                and len(set(pv)) == len(pv) and len(set(cv)) == len(cv)):
            return c
    return None


def _encode_rows(prev: Any, curr: Any) -> dict | None:
    """Keyed row diff: changed/new records keyed by a stable id column, plus removals.

    Only represents the agent-loop pattern — surviving rows keep their relative order
    and new rows are appended. A reorder/interleave can't be reconstructed from
    (prev + this diff), so it returns None and a coarser strategy (or full) is used.
    """
    p, c = _locate_records(prev), _locate_records(curr)
    if not p or not c:
        return None
    (at_p, prev_recs), (at_c, curr_recs) = p, c
    if at_p != at_c:
        return None
    by = _diff_id_col(prev_recs, curr_recs)
    if by is None:
        return None
    prev_by = {r[by]: r for r in prev_recs}
    prev_order = [r[by] for r in prev_recs]
    curr_by = {r[by]: r for r in curr_recs}
    curr_order = [r[by] for r in curr_recs]
    del_ids = [i for i in prev_order if i not in curr_by]
    new_ids = [i for i in curr_order if i not in prev_by]
    survivors = [i for i in prev_order if i in curr_by]
    if survivors + new_ids != curr_order:
        return None  # reordered/interleaved — not representable as prev+delta
    changed = [curr_by[i] for i in survivors if curr_by[i] != prev_by[i]]
    new_recs = [curr_by[i] for i in new_ids]
    set_recs = changed + new_recs
    return {DIFF_MARKER: 1, "shape": "rows", "at": at_c, "by": by,
            "n": len(curr_recs), "set": set_recs, "new": new_ids,
            "del": del_ids, "same": len(curr_recs) - len(set_recs)}


def _decode_rows(prev: Any, diff: dict) -> Any:
    at, by = diff["at"], diff["by"]
    prev_recs = prev if at is None else prev[at]
    set_by = {r[by]: r for r in diff["set"]}
    del_set = set(diff["del"])
    result = [set_by.get(r[by], r) for r in prev_recs if r[by] not in del_set]
    result += [set_by[i] for i in diff["new"]]
    if at is None:
        return result
    out = copy.deepcopy(prev)
    out[at] = result
    return out


def _encode_keys(prev: Any, curr: Any) -> dict | None:
    """Shallow object key diff — the coarse fallback for two dicts (or a dict whose
    record list moved/reordered, where the row diff bows out)."""
    if not (isinstance(prev, dict) and isinstance(curr, dict)):
        return None
    set_k = {k: v for k, v in curr.items() if k not in prev or prev[k] != v}
    del_k = [k for k in prev if k not in curr]
    return {DIFF_MARKER: 1, "shape": "keys", "set": set_k, "del": del_k}


def _decode_keys(prev: Any, diff: dict) -> Any:
    del_set = set(diff["del"])
    out = {k: v for k, v in prev.items() if k not in del_set}
    out.update(diff["set"])
    return out


def diff_decode(prev: Any, diff: dict) -> Any:
    """Reconstruct curr from the prior value + a diff. Exact inverse of the matching
    encoder. Raises ValueError on an unknown shape."""
    shape = diff.get("shape")
    if shape == "rows":
        return _decode_rows(prev, diff)
    if shape == "keys":
        return _decode_keys(prev, diff)
    raise ValueError(f"unknown diff shape: {shape!r}")


def diff_encode(prev: Any, curr: Any) -> dict | None:
    """A self-describing lossless diff of curr against prev, or None if none applies.

    Strategies are tried finest-first (row diff, then coarse key diff); each is accepted
    ONLY if it reconstructs curr exactly, so a returned diff is lossless by construction.
    The caller still decides whether the diff is worth emitting (it must also be smaller).
    """
    for strat in (_encode_rows, _encode_keys):
        diff = strat(prev, curr)
        if diff is None:
            continue
        try:
            if diff_decode(prev, diff) == curr:
                return diff
        except (KeyError, TypeError, ValueError):
            pass
    return None


def diff_roundtrip_ok(prev: Any, curr: Any) -> bool:
    """The lossless GATE for diffing: True iff a diff exists and rebuilds curr exactly."""
    diff = diff_encode(prev, curr)
    return diff is not None and diff_decode(prev, diff) == curr


def diff_wire(prev: Any, curr: Any, tool: str = "") -> str | None:
    """The model-facing diff envelope text, or None if no lossless diff applies.

    The diff plus a self-describing note and a base anchor (a short hash of the prior
    value). Shared by the proxy (what ships) and the fluency-for-diff eval (what's
    measured), so the eval tests exactly the bytes the model would read.
    """
    diff = diff_encode(prev, curr)
    if diff is None:
        return None
    base = hashlib.sha1(minify(prev).encode("utf-8")).hexdigest()[:8]
    label = f" {tool}" if tool else ""
    # Note kept tight on purpose (#9): it is fixed per-diff overhead and the only format
    # guidance the proxy can give (it can't set a system prompt). Verified self-sufficient
    # by `terse fluency --diff` with NO system primer — the production condition.
    # Kept lean on purpose (#9): the inline note can't be made comprehension-sufficient
    # for weaker models by *length* — measurement showed wording doesn't recover them
    # (the system primer did, which the stdio proxy can't deliver). So minimize overhead
    # and address comprehension via a one-time format primer instead.
    if diff.get("shape") == "rows":
        note = (f"Diff of the previous{label} result above: from its records drop `del` "
                "ids, upsert `set` by the `by` field, append `new` ids; n=final count.")
    else:
        note = (f"Diff of the previous{label} result above: on that object remove `del` "
                "keys, then apply `set` key/values.")
    return minify({**diff, "of": tool, "base": base, "note": note})


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
