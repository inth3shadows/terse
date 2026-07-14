"""Question sources — deterministic, targeting terse's risky transforms (#78 split).

Three generators (uniform records, nested dict-map records, single flat record)
behind one entry point (`gen_questions`), plus the text-payload analogue
(`gen_text_diff_questions`). Every question is programmatically checkable and
tagged with the transform it stresses, so the report says *which* transform
costs comprehension, not just an aggregate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .. import text_diff
from ..capture import extract_records
from ..transforms import (
    _uniform_dict_list,
    compress_structure,
    dict_encode,
    has_terse_marker,
    minify,
)
from .scoring import _is_number


@dataclass
class Question:
    qid: str          # unique within a payload
    qtype: str        # count | lookup | enumerate | aggregate
    transform: str    # transform stressed: table | table+dict
    prompt: str
    instruction: str  # how the model must format its answer (keeps scoring exact)
    expected: Any


def _aliased_strings(obj: Any) -> set:
    """The value-STRINGS terse would fold into the `~`-legend for this payload. Used to
    steer the lookup question onto a dict-coded field, so it stresses `~N` resolution
    rather than a plain literal. (Legend values can now be whole subtrees, which are
    unhashable — filter to strings here; use `_aliased_canon` for subtree membership.)"""
    _, legend = dict_encode(compress_structure(obj))
    return {v for v in legend.values() if isinstance(v, str)}


def _aliased_canon(obj: Any) -> set:
    """Canonical (minified) forms of EVERY aliased value — string or whole subtree — so
    a question can be tagged as stressing alias resolution even when the alias expands
    to an object/list (the whole-subtree-aliasing case)."""
    _, legend = dict_encode(compress_structure(obj))
    return {minify(v) for v in legend.values()}


def _pick_id_col(records: list[dict], cols: list[str]) -> str | None:
    """A column whose values are scalar and unique — usable to address one record."""
    n = len(records)
    for c in cols:
        vals = [r[c] for r in records]
        if all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in vals) and len(set(vals)) == n:
            return c
    return None


def _pick_target_col(records: list[dict], cols: list[str], idcol: str, aliased: set) -> str | None:
    """Prefer a column whose values are dict-coded (stresses alias resolution);
    fall back to any other scalar column."""
    fallback = None
    for c in cols:
        if c == idcol:
            continue
        vals = [r[c] for r in records]
        if any(isinstance(v, str) and v in aliased for v in vals):
            return c
        if fallback is None and all(
            isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in vals
        ):
            fallback = c
    return fallback


def _pick_numeric_col(records: list[dict], cols: list[str], exclude: str | None = None) -> str | None:
    """An all-numeric column, preferring one other than the identifier (a max over
    unique ids is a trivial check); falls back to the id column only if it is the
    only numeric one."""
    fallback = None
    for c in cols:
        if not all(_is_number(r[c]) for r in records):
            continue
        if c == exclude:
            fallback = fallback or c
        else:
            return c
    return fallback


def _intersection_cols(records: list[dict]) -> list[str]:
    """Keys present in EVERY record, sorted for determinism. For a non-uniform record
    list (e.g. structure symbols, where only some carry line/hash) these are the only
    columns safe to index across all records."""
    common = set(records[0].keys())
    for r in records[1:]:
        common &= set(r.keys())
    return sorted(common)


def _nested_record_group(obj: Any) -> tuple[str, list[dict], list[str]] | None:
    """Reach a record list that terse's STRICT uniform extractor skips: a dict-map of
    parent records each holding a child list of dicts (runecho.structure's
    `files{path: {symbols: [...]}}`), where the child list is non-uniform (symbol kinds
    carry different keys). Returns (label, records, common_cols) deterministically — first
    match in source order, first parent in map order — else None.

    Fluency-local by design: it does NOT touch `extract_records`/`_uniform_dict_list`,
    which the tabularizer, probe, and drop-path logic (#47) share — widening their notion
    of a record would change what the codec folds. This only widens what the fluency
    harness can ASK about, so `proxy --diff` gets exercised on structure-shaped output
    (issue #71).

    Preferred OVER the uniform extractor for the dict-map case: an unscoped "how many
    records" is ambiguous when the payload holds many groups, and `extract_records` would
    otherwise return whichever group's child list happens to be uniform — a valid list but
    the wrong (unlabelled) question. So group-scoped questions win when a dict-map is
    present, regardless of any single group's uniformity."""
    if not isinstance(obj, dict):
        return None

    def _records_of(lst: Any) -> list[str] | None:
        if not isinstance(lst, list) or len(lst) < 2:
            return None
        if not all(isinstance(x, dict) for x in lst):
            return None
        return _intersection_cols(lst) or None

    for k, v in obj.items():
        # dict-map of parent records -> the first parent (map order) with a child record list
        if isinstance(v, dict) and v and all(isinstance(pv, dict) for pv in v.values()):
            for pkey, parent in v.items():
                for _ck, cv in parent.items():
                    if _records_of(cv):
                        return f"{k}[{json.dumps(pkey, ensure_ascii=False)}]", cv, _intersection_cols(cv)
        # a NON-uniform top-level list of dicts (uniform ones are extract_records' domain)
        if isinstance(v, list) and _records_of(v) and not _uniform_dict_list(v):
            return k, v, _intersection_cols(v)
    return None


def _nested_questions(obj: Any) -> list[Question]:
    """Questions for structure-shaped payloads (dict-map of records with non-uniform child
    lists) that `gen_questions`' uniform path can't reach — count/enumerate/lookup over the
    columns shared by every record, plus aggregate if a shared numeric column exists. Same
    deterministic, programmatically-checkable contract as `gen_questions` (issue #71)."""
    grp = _nested_record_group(obj)
    if grp is None:
        return []
    label, records, cols = grp
    n = len(records)
    qs: list[Question] = [Question(
        "count", "count", "nested",
        f"How many records are listed under {label}?",
        "Reply with only the integer count.", n)]

    # enumerate lists a column in order — duplicates are fine (order/count is the check).
    # Use the most-distinct string column (most informative); deterministic — ties resolve
    # to sorted-column order via max()'s stable first-max.
    str_cols = [c for c in cols if all(isinstance(r[c], str) for r in records)]
    if str_cols:
        enum_col = max(str_cols, key=lambda c: len({r[c] for r in records}))
        qs.append(Question(
            "enumerate", "enumerate", "nested",
            f"List the {enum_col!r} of every record under {label}, in order.",
            "Reply with a JSON array of the values and nothing else.",
            [r[enum_col] for r in records]))

    # lookup needs a column that UNIQUELY addresses one record — otherwise the prompt is
    # ambiguous and a truthful answer about a different matching record scores wrong. Reuse
    # the uniform path's uniqueness rule (`_pick_id_col`); skip lookup when none is unique
    # (common in structure: `kind` and even overloaded `name` repeat within a file).
    idcol = _pick_id_col(records, cols)
    if idcol is not None:
        tgt = next((c for c in cols if c != idcol), None)
        if tgt is not None:
            ri = n // 2  # idcol is unique, so any index gives an unambiguous prompt
            qs.append(Question(
                "lookup", "lookup", "nested",
                f"Under {label}, for the record whose {idcol!r} is "
                f"{json.dumps(records[ri][idcol], ensure_ascii=False)}, "
                f"what is the value of {tgt!r}?",
                "Reply with only the value, with no quotes and no extra words.",
                records[ri][tgt]))

    numcol = next((c for c in cols if all(_is_number(r[c]) for r in records)), None)
    if numcol is not None:
        qs.append(Question(
            "aggregate", "aggregate", "nested",
            f"What is the maximum value of {numcol!r} across the records under {label}?",
            "Reply with only the number.",
            max(r[numcol] for r in records)))
    return qs


def _flat_record_questions(obj: Any) -> list[Question]:
    """Questions for a SINGLE flat record — a dict of mostly scalar fields (a search
    hit, a status receipt, one KB row). These payloads tabularize to nothing, but they
    are a real diff surface: consecutive single-record results diff via the keys shape,
    and before this generator the soak/diff evals were blind to every such chain.
    Same contract as the other sources: deterministic, programmatically checkable,
    fluency-local (the codec/tabularizer rules are untouched)."""
    if not isinstance(obj, dict) or has_terse_marker(obj):
        return []
    # int/float lookups are unambiguous; strings must be short, non-empty scalars —
    # an empty expected would be indistinguishable from _safe_ask's empty-string
    # error return (same exclusion the text-diff questions apply).
    lookable = {
        k: v for k, v in obj.items()
        if (isinstance(v, (int, float)) and not isinstance(v, bool))
        or (isinstance(v, str) and 0 < len(v) <= 60)
    }
    if len(lookable) < 3:
        return []
    qs: list[Question] = [Question(
        "keys-count", "count", "flat-record",
        "How many top-level keys does the object have?",
        "Reply with only the integer count.",
        len(obj),
    )]
    keys = sorted(lookable)
    num_key = next((k for k in keys
                    if isinstance(lookable[k], (int, float))), None)
    str_key = next((k for k in keys if isinstance(lookable[k], str)), None)
    for i, k in enumerate(x for x in (num_key, str_key) if x is not None):
        qs.append(Question(
            f"field-{i}", "lookup", "flat-record",
            f"What is the value of the top-level field {k!r}?",
            "Reply with only the value, with no quotes and no extra words.",
            lookable[k],
        ))
    return qs


def gen_questions(obj: Any) -> list[Question]:
    """Generate deterministic, programmatically-checkable questions for a payload.

    Uniform record-shaped payloads (what terse tabularizes) yield questions directly;
    payloads whose records the strict extractor skips (structure's dict-map of non-uniform
    symbol lists) fall through to `_nested_questions` (#71); a SINGLE flat record (a
    search hit, a receipt) falls through to `_flat_record_questions` — the keys-diff
    surface the chain soak needs; everything else returns []. Selection is fully
    deterministic so a re-run is reproducible.
    """
    # Structure-shaped payloads (a dict-map of records) need GROUP-SCOPED questions; prefer
    # them over the uniform extractor, which would otherwise emit an unscoped, ambiguous
    # question from whichever group's child list happens to be uniform (#71).
    nested = _nested_questions(obj)
    if nested:
        return nested
    records = extract_records(obj)
    if not records:
        return _flat_record_questions(obj)
    cols = list(records[0].keys())
    n = len(records)
    aliased = _aliased_strings(obj)
    canon = _aliased_canon(obj)
    qs: list[Question] = []

    # count — enumeration fidelity; the motivation for the row-count hint.
    qs.append(Question(
        "count", "count", "table",
        "How many records does the dataset contain?",
        "Reply with only the integer count.",
        n,
    ))

    idcol = _pick_id_col(records, cols)
    if idcol is not None:
        tgt = _pick_target_col(records, cols, idcol, aliased)
        if tgt is not None:
            # Prefer a record whose target value is dict-coded (so the lookup truly
            # stresses `~N` resolution); else the middle record. Deterministic.
            ri = next((i for i in range(n)
                       if isinstance(records[i][tgt], str) and records[i][tgt] in aliased), n // 2)
            expected = records[ri][tgt]
            transform = "table+dict" if isinstance(expected, str) and expected in aliased else "table"
            qs.append(Question(
                "lookup", "lookup", transform,
                f"For the record whose {idcol!r} is {json.dumps(records[ri][idcol], ensure_ascii=False)}, "
                f"what is the value of {tgt!r}?",
                "Reply with only the value, with no quotes and no extra words.",
                expected,
            ))
        # enumerate — under-enumeration of wide tables was terse's measured recall gap.
        qs.append(Question(
            "enumerate", "enumerate", "table",
            f"List the {idcol!r} of every record, in order.",
            "Reply with a JSON array of the values and nothing else.",
            [r[idcol] for r in records],
        ))

    numcol = _pick_numeric_col(records, cols, exclude=idcol)
    if numcol is not None:
        qs.append(Question(
            "aggregate", "aggregate", "table",
            f"What is the maximum value of {numcol!r} across all records?",
            "Reply with only the number.",
            max(r[numcol] for r in records),
        ))

    # deref — a column whose cells are whole objects/lists. With whole-subtree aliasing
    # these become `~N` that expand to a STRUCTURE, so this question stresses the new,
    # harder comprehension case: resolve an alias to an entire object, not a string.
    if idcol is not None:
        blobcol = next((c for c in cols if c != idcol
                        and all(isinstance(r[c], (dict, list)) for r in records)), None)
        if blobcol is not None:
            ri = next((i for i in range(n) if minify(records[i][blobcol]) in canon), n // 2)
            expected = records[ri][blobcol]
            transform = "table+dict" if minify(expected) in canon else "table"
            qs.append(Question(
                "deref", "deref", transform,
                f"For the record whose {idcol!r} is {json.dumps(records[ri][idcol], ensure_ascii=False)}, "
                f"what is the full value of {blobcol!r}?",
                "Reply with only that value as compact JSON, and nothing else.",
                expected,
            ))
    return qs


def _text_diff_questions_from_lines(curr: str) -> list[Question]:
    """The anchor questions given curr's lines — assumes the caller already confirmed a
    lossless text diff applies. An empty target line is excluded from the "last-line"/
    "mid-line" lookup questions: an empty `expected` would be indistinguishable from
    `_safe_ask`'s empty-string return on a total answerer failure (a transport error
    scoring as a correct answer), so those questions are simply not asked when the
    anchor line would be empty."""
    lines = curr.splitlines()
    questions = [
        Question(qid="line-count", qtype="count", transform="text-diff",
                 prompt="How many lines does the text contain?",
                 instruction="Reply with just the number.",
                 expected=len(lines)),
    ]
    if lines and lines[-1]:
        questions.append(
            Question(qid="last-line", qtype="lookup", transform="text-diff",
                     prompt="What is the exact content of the LAST line of the text?",
                     instruction="Reply with just that line, nothing else.",
                     expected=lines[-1]),
        )
    # A mid-document line stresses a REFERENCED (unchanged) chunk instead of the edited
    # tail line-count/last-line stress — the part of the reconstruction those two
    # questions can't catch a regression in (e.g. corrupted chunk references that leave
    # the line count and final line intact).
    mid = len(lines) // 2
    if len(lines) > 2 and lines[mid]:
        questions.append(
            Question(qid="mid-line", qtype="lookup", transform="text-diff",
                     prompt=f"What is the exact content of line {mid + 1} (1-indexed) "
                            "of the text?",
                     instruction="Reply with just that line, nothing else.",
                     expected=lines[mid]),
        )
    return questions


def gen_text_diff_questions(prev: str, curr: str, tool: str = "") -> list[Question]:
    """Deterministic questions over unstructured text — the text-payload analogue of
    gen_questions (record/column-shaped JSON). [] if no lossless text diff applies
    (mirrors run_diff_payload's `wire is None` gate) — nothing to compare."""
    if text_diff.text_diff_wire(prev, curr, tool) is None:
        return []
    return _text_diff_questions_from_lines(curr)
