"""Lossless transforms with a round-trip gate.

Tier 0   — minify (whitespace) + tabularize (fold repeated KEYS of record arrays)
Tier 0.5 — dictionary coding (fold repeated VALUES via an inline legend)
Tier 0.6 — embedded JSON (fold a string leaf that is itself a JSON document, so the
           tiers above can reach records delivered double-encoded; opt-in)
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
import math
from collections import Counter
from typing import Any

from .tokenize import count_cl100k

# Structural markers. Chosen to be vanishingly unlikely in real tool output.
TABLE_MARKER = "__terse_table__"
DICT_MARKER = "__terse_dict__"
DIFF_MARKER = "__terse_diff__"
# Tier 0.6: a string leaf that is itself a JSON document (`embedded`).
JSON_STR_MARKER = "__terse_json__"
# The drop-to-retrieve inline handle marker (#10). Not a transforms envelope — it is
# produced by the lossy layer and consumed by the proxy's retrieve handler — but it lives
# in this registry so all `__terse_*` wire keys have one home and are reserved together.
DROPPED_MARKER = "__terse_dropped__"
# Sentinel for non-uniform record tabularization: when a column carries explicit JSON
# `null`, an absent key is encoded as this sentinel rather than `null`, so the decoder
# can distinguish "key omitted" from "key present, value null" unambiguously.
ABSENT_MARKER = "__terse_absent__"
ALIAS_SIGIL = "~"

# Keys reserved for terse's own envelopes. A real payload that already contains one
# can't be safely compressed: the consumer reads these markers per the format primer,
# so it would mis-reconstruct the user's literal dict as a terse envelope. The codec
# has no escape convention, so the only lossless move is to leave such a payload alone.
_RESERVED_MARKERS = frozenset({TABLE_MARKER, DICT_MARKER, DIFF_MARKER, DROPPED_MARKER,
                               JSON_STR_MARKER})

# Reserved as a VALUE, not a key — the one sentinel the codec writes into a cell rather
# than into a header. It needs its own set because `has_terse_marker` screens keys, and
# listing ABSENT_MARKER alongside the envelope keys would be a silent no-op: a payload
# whose own string value is "__terse_absent__" never trips a key check. Left unscreened,
# that value in a sentinel column decodes as "key absent" and the record loses the field —
# caught by `_lossless_stage`'s verify-before-emit, but at the cost of the whole payload's
# compression AND a `gate_fail` that `policy_gen._tool_decision` reads as "this tool's
# shape defeats the codec", marking it passthrough permanently.
_RESERVED_VALUES = frozenset({ABSENT_MARKER})

# The serializations the `embedded` tier can reproduce EXACTLY. A string leaf is folded only
# when one of these regenerates it byte-for-byte, so the id stored in "f" is a complete
# recipe for rebuilding the original bytes — decode never has to guess at formatting.
#
# WIRE CONTRACT: these ids are permanent. Adding a form is backward-safe (payloads already
# on the wire keep decoding); renaming one, or re-pointing it at different kwargs, silently
# corrupts every payload that used it. Append, never edit.
_EMBED_FORMS: dict[str, dict[str, Any]] = {
    # `json.dumps(body)` with no kwargs — what a server that double-encodes usually emits.
    "p": {},
    "P": {"ensure_ascii": False},
    # Minified. "C" is terse's own `minify` form, so terse-on-terse nests cleanly.
    "c": {"separators": (",", ":")},
    "C": {"separators": (",", ":"), "ensure_ascii": False},
    # Pretty-printed.
    "i2": {"indent": 2},
    "I2": {"indent": 2, "ensure_ascii": False},
    "i4": {"indent": 4},
    "I4": {"indent": 4, "ensure_ascii": False},
}

# Nesting cap shared by every codec boundary (capture's shape classifier, policy.apply,
# the proxy's diff path, measure). The transforms themselves recurse without a depth
# argument, so a boundary must check `exceeds_depth` BEFORE handing a payload in; a
# payload past the cap is treated like a marker collision — passed through untouched.
# 200 sits far under CPython's ~1000 recursion limit, leaving headroom for the frames
# the codec adds per level.
MAX_DEPTH = 200


def exceeds_depth(obj: Any, cap: int = MAX_DEPTH) -> bool:
    """True if obj nests containers deeper than `cap`. Iterative — safe to call on the
    pathological payloads it exists to screen out."""
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            if depth > cap:
                return True
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            if depth > cap:
                return True
            stack.extend((x, depth + 1) for x in node)
    return False


def has_terse_marker(obj: Any) -> bool:
    """True if obj contains, at ANY depth, a dict key reserved for a terse envelope or a
    string equal to a reserved sentinel VALUE.

    decompress / the model's primer interpret these markers wherever they appear, so a
    collision anywhere — not just top-level — makes a payload unsafe to compress."""
    if isinstance(obj, dict):
        if not _RESERVED_MARKERS.isdisjoint(obj.keys()):
            return True
        return any(has_terse_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_terse_marker(x) for x in obj)
    # isinstance, not `type(obj) is str`: a str subclass compares equal to the sentinel and
    # would decode as "absent" just the same, so it has to be screened too.
    return isinstance(obj, str) and obj in _RESERVED_VALUES


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


def is_tabularizable(value: Any) -> bool:
    """Would `compress_structure` fold this into a table? The canonical record-shape rule.

    The boolean front door onto `_tabularizable_dict_list`, for callers that only need the
    verdict — `capture`'s shape classifier and record extractor, which must agree with the
    codec on what "record-shaped" means or the measurement stack under-reports exactly the
    traffic the codec is best at (#204).
    """
    return _tabularizable_dict_list(value)[0]


def _tabularizable_dict_list(value: Any) -> tuple[bool, list | None, set | None, set | None]:
    """Return (ok, union_keys, absent_columns, sentinel_columns) for a list-of-dicts.

    union_keys is the union of all keys in deterministic first-seen order.
    absent_columns is every column index where at least one row omits the key (stored in
    the table so the decoder strips those cells). sentinel_columns is the subset of
    absent_columns where at least one row also carries an explicit JSON null — those
    columns encode absent cells as ABSENT_MARKER rather than plain null so the decoder
    can distinguish "omitted" from "present, value null".

    Returns (False, None, None, None) when the list is too small, not all-dicts, or too
    sparse (fewer than half the cells would be filled — the table header costs more than
    folding saves at that density; #154's emit-only-if-smaller is the last-resort backstop).
    """
    if not isinstance(value, list) or len(value) < 2:
        return False, None, None, None
    if not all(isinstance(item, dict) for item in value):
        return False, None, None, None

    # Union of all keys in deterministic first-seen order.
    seen: set[str] = set()
    union_keys: list[str] = []
    for d in value:
        for k in d:
            if k not in seen:
                seen.add(k)
                union_keys.append(k)

    # All-uniform is the common case — still tabularizable, just no absent cells.
    if len(union_keys) == len(value[0]) and all(len(d) == len(union_keys) for d in value):
        return True, union_keys, set(), set()

    # Shared-key density gate: refuse when fewer than half the cells are filled.
    total_cells = len(union_keys) * len(value)
    filled = sum(len(d) for d in value)
    if filled * 2 <= total_cells:
        return False, None, None, None

    # Per-column analysis.
    key_to_idx = {k: i for i, k in enumerate(union_keys)}
    has_explicit_null: set[int] = set()
    has_absent: set[int] = set()
    for d in value:
        for k, v in d.items():
            if v is None:
                has_explicit_null.add(key_to_idx[k])
        for k in seen - d.keys():
            has_absent.add(key_to_idx[k])

    sentinel_columns = has_explicit_null & has_absent
    return True, union_keys, has_absent, sentinel_columns


def _embed_json_string(s: str, tabularize: bool = True) -> dict | None:
    """Fold a string leaf that is ITSELF a JSON document — or None to leave it alone.

    `tabularize`/`dictionary` walk PARSED structure, so a payload delivered as a JSON
    *string* is a leaf they cannot reach. Measured on identical data: 41.9% saved as a real
    record array, 0.0% inside a string. Double-encoding is a common server convention —
    #143 measured ~21.6% of one fleet's tokens sitting at 0.0% from a single
    `{"response_text": json.dumps(body)}` return shape, and noted "it is not one tool".

    Folded ONLY when a form in `_EMBED_FORMS` reproduces `s` byte-for-byte. That bar is
    deliberately stricter than "parses to equal data", because `json.dumps(json.loads(s))`
    is not `s` in general: duplicate keys collapse to the last one, and `1.50` renormalizes
    to `1.5`. Both SKIP here rather than decode to something the server never sent —
    terse's guarantee is byte-faithfulness, and a tier that quietly downgraded it to
    "equivalent data" would redefine the property the project exists to provide.
    """
    # Every form emits a container with no leading whitespace, so anything else could not
    # be reproduced byte-exactly anyway — reject before paying for a parse.
    if s[:1] not in ("{", "["):
        return None
    try:
        parsed = json.loads(s)
    except (ValueError, RecursionError):
        return None
    if not isinstance(parsed, (dict, list)):
        return None
    # The same two invariants the outer codec honours: an embedded doc carrying a terse
    # marker would be mis-read as an envelope, and one past the depth cap would recurse.
    if has_terse_marker(parsed) or exceeds_depth(parsed):
        return None
    form = next((f for f, kw in _EMBED_FORMS.items() if json.dumps(parsed, **kw) == s), None)
    if form is None:
        return None
    # `tabularize` is FORWARDED, not re-defaulted. Folding the string opens a new structural
    # walk, and defaulting it to True there emitted `__terse_table__` inside `v` for a policy
    # whose tiers were `["minify", "embedded"]` — a marker the primer never documented,
    # because `emits_table()` was correctly False. Lossless either way, but it hands the model
    # an envelope with no explanation, which is exactly what `reachable_tiers` exists to
    # prevent. Found in adversarial review of #183; untested until then because every test
    # paired the two tiers.
    wrapper = {JSON_STR_MARKER: 1, "f": form,
               "v": compress_structure(parsed, embedded=True, tabularize=tabularize)}
    # Per-occurrence size guard. `dict_encode` guards per alias and `compress_with` guards
    # the payload as a whole; neither can see a single embedded doc that grew inside a
    # payload that shrank overall. Compared against the string as a JSON VALUE, since that
    # is what shipping it unfolded actually costs.
    if _tok_text(minify(wrapper)) >= _tok(s):
        return None
    return wrapper


def _fold_records(records: list[dict], embedded: bool = False,
                  union_keys: list | None = None,
                  absent_columns: set | None = None,
                  sentinel_columns: set | None = None) -> tuple[dict, list]:
    """Fold a (possibly non-uniform) dict list into (spec, positional rows), recursing
    on dict-columns.

    When `union_keys` is given, rows may omit keys — absent cells are filled with None
    or ABSENT_MARKER depending on `sentinel_columns`. Without it, assumes uniform dicts
    (the common case, and the only path before union-schema tabularize).

    A column whose values are themselves all uniform dicts is hoisted: its key-set
    moves to spec['subcols'][col] once, and each cell becomes a positional tuple.
    Non-dict columns are recursed through compress_structure (so a list-of-dicts cell
    becomes its own sub-table). spec = {'cols': [...], 'subcols': {col: spec, ...},
    'absent_cols': [...]}.
    """
    keys = union_keys if union_keys is not None else list(records[0].keys())
    absent_cols = absent_columns if absent_columns is not None else set()
    sentinel_cols = sentinel_columns if sentinel_columns is not None else set()
    key_to_idx = {k: i for i, k in enumerate(keys)}

    posrows = []
    for rec in records:
        row = []
        for k in keys:
            if k in rec:
                row.append(rec[k])
            else:
                row.append(ABSENT_MARKER if key_to_idx[k] in sentinel_cols else None)
        posrows.append(row)

    subcols: dict = {}
    n = len(records)
    for ci, k in enumerate(keys):
        col = [posrows[ri][ci] for ri in range(n)]
        # Skip sentinel cells when checking for nested uniform-dict columns.
        live_rows = [c for c in col if c is not ABSENT_MARKER]
        if len(live_rows) == n:
            if _uniform_dict_list(col):
                sub_spec, sub_pos = _fold_records(col, embedded)
                subcols[k] = sub_spec
                for ri in range(n):
                    posrows[ri][ci] = sub_pos[ri]
            else:
                ok, sub_union_keys, sub_absent, sub_sentinel = _tabularizable_dict_list(col)
                if ok:
                    sub_spec, sub_pos = _fold_records(col, embedded, sub_union_keys,
                                                       sub_absent, sub_sentinel)
                    subcols[k] = sub_spec
                    for ri in range(n):
                        posrows[ri][ci] = sub_pos[ri]
                else:
                    for ri in range(n):
                        posrows[ri][ci] = compress_structure(posrows[ri][ci], embedded)
        else:
            # Non-uniform column with holes — can't be a sub-table.
            for ri in range(n):
                posrows[ri][ci] = compress_structure(posrows[ri][ci], embedded)

    spec: dict = {"cols": keys}
    if subcols:
        spec["subcols"] = subcols
    if absent_cols:
        spec["absent_cols"] = sorted(absent_cols)
        if sentinel_cols:
            spec["sentinel_cols"] = sorted(sentinel_cols)
    return spec, posrows


def compress_structure(obj: Any, embedded: bool = False, tabularize: bool = True) -> Any:
    """Recursively fold every qualifying list-of-uniform-dicts into a table,
    hoisting nested uniform-dict columns into a shared subcols header.

    `embedded` additionally folds string leaves that are themselves JSON documents
    (Tier 0.6, opt-in). Both flags default to today's behaviour, so every existing caller
    is unchanged: `tabularize=False, embedded=True` runs the embedded fold alone.
    """
    if isinstance(obj, dict):
        return {k: compress_structure(v, embedded, tabularize) for k, v in obj.items()}
    if isinstance(obj, list):
        if tabularize and _uniform_dict_list(obj):
            spec, posrows = _fold_records(obj, embedded)
            table = {TABLE_MARKER: 1, "n": len(posrows), "cols": spec["cols"], "rows": posrows}
            if "subcols" in spec:
                table["subcols"] = spec["subcols"]
            if "absent_cols" in spec:
                table["absent_cols"] = spec["absent_cols"]
            if "sentinel_cols" in spec:
                table["sentinel_cols"] = spec["sentinel_cols"]
            return table
        if tabularize:
            ok, union_keys, absent_columns, sentinel_columns = _tabularizable_dict_list(obj)
            if ok:
                spec, posrows = _fold_records(obj, embedded, union_keys,
                                               absent_columns, sentinel_columns)
                table = {TABLE_MARKER: 1, "n": len(posrows),
                         "cols": spec["cols"], "rows": posrows}
                if "subcols" in spec:
                    table["subcols"] = spec["subcols"]
                if "absent_cols" in spec:
                    table["absent_cols"] = spec["absent_cols"]
                if "sentinel_cols" in spec:
                    table["sentinel_cols"] = spec["sentinel_cols"]
                return table
        return [compress_structure(item, embedded, tabularize) for item in obj]
    if embedded and isinstance(obj, str):
        return _embed_json_string(obj, tabularize) or obj
    return obj


def _unfold_row(row: list, cols: list, subcols: dict,
                absent_cols: set | frozenset = frozenset(),
                sentinel_cols: set | frozenset = frozenset()) -> dict:
    """Rebuild one record from a positional row + its (possibly nested) header.

    `absent_cols` lists columns where at least one row omits the key. `sentinel_cols`
    (a subset of absent_cols) lists columns where explicit null also appears — there,
    ABSENT_MARKER fills absent cells and `None` is preserved as a real value. In
    non-sentinel absent columns, `None` unambiguously means "key absent" and is skipped.
    """
    rec = {}
    for ci, k in enumerate(cols):
        cell = row[ci]
        if ci in absent_cols and ci not in sentinel_cols and cell is None:
            continue
        if ci in sentinel_cols and cell == ABSENT_MARKER:
            continue
        sub = subcols.get(k)
        if sub is not None:
            rec[k] = _unfold_row(cell, sub["cols"], sub.get("subcols", {}),
                                 _absent_cols_set(sub), _sentinel_cols_set(sub))
        else:
            rec[k] = decompress_structure(cell)
    return rec


def _absent_cols_set(spec: dict) -> set:
    ac = spec.get("absent_cols")
    return set(ac) if ac else set()


def _sentinel_cols_set(spec: dict) -> set:
    sc = spec.get("sentinel_cols")
    return set(sc) if sc else set()


def decompress_structure(obj: Any) -> Any:
    """Exact inverse of `compress_structure`. Top-down: unwrap, then recurse."""
    if isinstance(obj, dict):
        if obj.get(TABLE_MARKER) == 1 and "cols" in obj and "rows" in obj:
            cols = obj["cols"]
            subcols = obj.get("subcols", {})
            absent_cols = _absent_cols_set(obj)
            sentinel_cols = _sentinel_cols_set(obj)
            return [_unfold_row(row, cols, subcols, absent_cols, sentinel_cols)
                    for row in obj["rows"]]
        if obj.get(JSON_STR_MARKER) == 1 and "f" in obj and "v" in obj:
            kwargs = _EMBED_FORMS.get(obj["f"])
            if kwargs is None:
                # A form id this build doesn't know — a forward/corrupt payload. Raise
                # rather than hand back the envelope dict as if it were the data: the
                # proxy's verify-before-emit self-check turns this into a fall-back to
                # the plain minified form, which is the safe outcome.
                raise ValueError(f"unknown embedded-JSON form {obj['f']!r}")
            return json.dumps(decompress_structure(obj["v"]), **kwargs)
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
#
# Canonical forms are computed ONCE per node in a bottom-up pass (`_build_canon_memo`)
# and shared by the counting and replacement walks below. Calling `minify(node)` at
# every container level instead re-serializes the whole subtree per level — O(n·depth)
# json.dumps traversals, done twice over — which is the #79 hot-path cost on
# codegraph-scale payloads.
def _canon_memo_walk(node: Any, memo: dict[int, str]) -> str:
    if isinstance(node, list):
        got = memo.get(id(node))
        if got is None:
            got = "[" + ",".join(_canon_memo_walk(x, memo) for x in node) + "]"
            memo[id(node)] = got
        return got
    if isinstance(node, dict):
        got = memo.get(id(node))
        if got is None:
            if all(isinstance(k, str) for k in node):
                got = "{" + ",".join(
                    json.dumps(k, ensure_ascii=False) + ":" + _canon_memo_walk(v, memo)
                    for k, v in node.items()) + "}"
            else:
                # Non-str keys (impossible off the wire — json.loads only makes str
                # keys): delegate the key coercion (int/float/bool/None -> string,
                # incl. Infinity/NaN spellings) to json.dumps rather than replicate it.
                for v in node.values():
                    _canon_memo_walk(v, memo)
                got = minify(node)
            memo[id(node)] = got
        return got
    return minify(node)  # scalar; separators are moot, so this equals json.dumps


def _build_canon_memo(structure: Any) -> tuple[str, dict[int, str]]:
    """(canonical form of `structure`, id(node) -> canonical form for every container).

    Each container's minified JSON is assembled once from its children's already-built
    forms, byte-identical to `minify(node)` (pinned by test). id-keying is safe because
    the caller holds `structure` alive for the memo's whole lifetime.
    """
    memo: dict[int, str] = {}
    root = _canon_memo_walk(structure, memo)
    return root, memo


def _node_tok(key: tuple) -> int:
    """Token cost of a candidate in its VALUE position: a quoted string, or the raw
    (already-minified) JSON of a subtree."""
    kind, payload = key
    return _tok(payload) if kind == "s" else _tok_text(payload)


def _count_value_nodes(node: Any, counter: Counter, memo: dict[int, str],
                       skip: frozenset[str] = frozenset()) -> None:
    """Count VALUE-position nodes (not dict keys) by canonical form, recursively.
    Strings count as ("s", str); dicts/lists count as ("j", canonical) AND recurse,
    so a repeated whole subtree and a repeated string inside it are both seen.

    `skip`: canonical forms that will be replaced WHOLESALE by `_replace_nodes` (a
    locked-in subtree alias) and so are never descended into at replace time. Passing
    the final `alias_for_json` keys here (#326) makes the count match what will
    actually be reachable after subtree folding, instead of the pre-fold global count
    that also credits occurrences nested inside a to-be-swallowed subtree."""
    if isinstance(node, str):
        counter[("s", node)] += 1
    elif isinstance(node, list):
        counter[("j", memo[id(node)])] += 1
        if memo[id(node)] in skip:
            return
        for x in node:
            _count_value_nodes(x, counter, memo, skip)
    elif isinstance(node, dict):
        counter[("j", memo[id(node)])] += 1
        if memo[id(node)] in skip:
            return
        for v in node.values():
            _count_value_nodes(v, counter, memo, skip)
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


def _replace_nodes(node: Any, alias_for_str: dict, alias_for_json: dict,
                   memo: dict[int, str]) -> Any:
    """Replace value-position strings AND whole subtrees with their alias; keys left
    untouched. Top-down: a matched container is replaced before descending, so an
    aliased subtree is folded as a unit (nested candidates inside it are captured by
    the parent's legend entry, never double-aliased)."""
    if isinstance(node, str):
        return alias_for_str.get(node, node)
    if isinstance(node, (list, dict)):
        alias = alias_for_json.get(memo[id(node)])
        if alias is not None:
            return alias
        if isinstance(node, list):
            return [_replace_nodes(x, alias_for_str, alias_for_json, memo) for x in node]
        return {k: _replace_nodes(v, alias_for_str, alias_for_json, memo)
                for k, v in node.items()}
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


def dict_encode(structure: Any, memo: dict[int, str] | None = None) -> tuple[Any, dict]:
    """Fold repeated value-strings AND repeated whole subtrees into an inline legend.
    Returns (data, legend).

    Tokenizer-aware: a node is aliased only when (n-1)*tok(node) exceeds the legend +
    reference cost. legend maps alias -> original string-or-subtree; an empty legend
    means dictionary coding didn't pay. Unused aliases (a string whose every occurrence
    was swallowed by an aliased parent subtree) are pruned so they never cost tokens.

    `memo` is the canon memo from `_build_canon_memo(structure)`; pass it when the
    caller already built one (compress_with does), else it is built here.
    """
    if memo is None:
        _, memo = _build_canon_memo(structure)
    counts: Counter = Counter()
    _count_value_nodes(structure, counts, memo)
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

    # Re-validate BOTH string and subtree aliases against the LOCKED-IN subtree
    # selection (#326). The loop above computed every candidate's `saving` from the
    # global pre-fold count, which over-counts: `_replace_nodes` folds a matched
    # subtree wholesale and never descends into it, so an occurrence nested inside a
    # swallowed subtree is never actually replaced there. That applies just as much to
    # a subtree candidate nested inside ANOTHER accepted subtree candidate as it does
    # to a string (code-review finding on the first version of this fix, which only
    # re-validated `alias_for_str` — a subtree alias used only once after a sibling
    # subtree swallows its other copies is the same net-token-loss bug, just for
    # kind="j" instead of kind="s"). Recount with every accepted subtree excluded from
    # the walk, and drop any alias — string or subtree — whose real, reachable saving
    # no longer clears zero.
    #
    # One pass, not a fixed point: `skip` is the FULL initial `alias_for_json` set, so
    # every candidate's recount here is against the walk's minimum possible reachable
    # count (excluding every other locked-in subtree, including itself as a no-op —
    # `_count_value_nodes` counts a node's own occurrence before checking `skip`, so a
    # candidate's own top-level occurrences are never miscounted by its own presence in
    # `skip`). Removing an over-counted alias can only ever RAISE another candidate's
    # true count (its occurrences stop being swallowed too) — never lower it below what
    # was already verified here — so a surviving candidate's saving, checked once
    # against this minimum, remains valid however the others resolve.
    if alias_for_json:
        skip = frozenset(alias_for_json)
        real_counts: Counter = Counter()
        _count_value_nodes(structure, real_counts, memo, skip)
        for payload, alias in list(alias_for_json.items()):
            n = real_counts[("j", payload)]
            t = _node_tok(("j", payload))
            saving = (n * t) - (n * _tok(alias) + _tok(alias) + t)
            if n < 2 or saving <= 0:
                del alias_for_json[payload]
                del legend[alias]
        for payload, alias in list(alias_for_str.items()):
            n = real_counts[("s", payload)]
            t = _tok(payload)
            saving = (n * t) - (n * _tok(alias) + _tok(alias) + t)
            if n < 2 or saving <= 0:
                del alias_for_str[payload]
                del legend[alias]
        if not (alias_for_str or alias_for_json):
            return structure, {}

    data = _replace_nodes(structure, alias_for_str, alias_for_json, memo)
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


def compress_with(obj: Any, tabularize: bool = True, dictionary: bool = True,
                  embedded: bool = False) -> str:
    """Apply a selectable subset of lossless tiers, then minify.

    `decompress` auto-detects the markers, so any combination round-trips. minify is
    always applied (it is the serialization). Pass both False for minify-only.

    Emit-only-if-smaller (#154): the tiered form is NEVER returned when it tokenizes no
    smaller than plain minify. `tabularize`'s `__terse_table__` header and envelope cost a
    fixed number of tokens that a small record set (a 2-row `list_*`, a filtered query, a
    shrunk result) cannot amortize, so the codec could otherwise emit MORE tokens than the
    server sent — silently, since the reported saving is an average that hides a per-payload
    regression. `dict_encode` already guards its own legend per-alias; this is the same
    contract one level up, covering the table header the per-alias guard cannot see. The
    diff tier holds the identical "emit the delta only when it's smaller" contract.
    """
    plain = minify(obj)  # the lossless floor: never larger than a well-formed raw payload
    if not tabularize and not dictionary and not embedded:
        return plain
    structure = (compress_structure(obj, embedded=embedded, tabularize=tabularize)
                 if (tabularize or embedded) else obj)
    candidate = minify(structure)
    if dictionary:
        base, memo = _build_canon_memo(structure)  # root canon doubles as the minified base
        data, legend = dict_encode(structure, memo)
        if legend:
            coded = minify({DICT_MARKER: 1, "legend": legend, "data": data})
            # Net-token guard: with whole-subtree aliasing the per-candidate estimate
            # can mis-rank under nesting overlap, so commit the dict block only when it
            # is actually smaller. Losslessness is independent (the round-trip gate);
            # this guards SIZE — the dict tier can never regress the payload.
            if _tok_text(coded) < _tok_text(base):
                candidate = coded
            else:
                candidate = base
        else:
            candidate = base
    # Compared on the tokenizer, not bytes: a shorter byte string can tokenize longer.
    # Ties go to `plain` — no reason to ship the more complex form for zero saving.
    return candidate if _tok_text(candidate) < _tok_text(plain) else plain


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


def values_equal(a: Any, b: Any) -> bool:
    """Value equality for the lossless gate — identical to `==` except that two NaNs in
    the same position compare EQUAL.

    Plain `==` answers the gate's question wrongly for NaN. IEEE-754 says `nan != nan`, so
    a payload the codec handled perfectly is reported as a losslessness failure: Python's
    `json` emits the non-standard `NaN` token and reads it straight back, and the bytes
    are exact. The failure DIRECTION is safe (a failed self-check falls back to the plain
    minified form, so nothing is corrupted) — what it breaks is the measurement, and that
    has a track record of driving wrong decisions here. `policy_gen._tool_decision`
    disqualifies a whole tool on `gate_fail`, marking it `passthrough` permanently for a
    shape the codec handles fine, and `measure` zeroes its banked savings so real
    compression reads as 0%. Same family as #144: the codec is fine, the number
    describing it is not (#187).

    Only NaN needs this. `Infinity` compares equal to itself; `-0.0 == 0.0` is True and
    `-0.0` serialises back as `-0.0`, so both already pass unaided.

    Deliberately NOT a canonical-bytes comparison: that would also make key REORDERING
    visible, which `==` has always ignored and which the codec is free to do.
    """
    if isinstance(a, float) and isinstance(b, float):
        # `or` not `and`: two NaNs are equal here, and a NaN against any other float is
        # not — which `==` already gets right.
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        # key SET first: `==` compares dicts order-insensitively and so must this.
        return a.keys() == b.keys() and all(values_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        # `strict=True` is redundant behind the length check but keeps a future edit that
        # drops that check from silently comparing only the shorter prefix.
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    # Everything else defers to `==` unchanged — including bool/int cross-equality, which
    # this must not tighten: the gate's contract is "same as ==, plus NaN".
    return bool(a == b)


def roundtrip_ok(obj: Any) -> bool:
    """The lossless GATE. True iff the full pipeline is byte-faithful by value."""
    return values_equal(decompress(compress(obj)), obj)
