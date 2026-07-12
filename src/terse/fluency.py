"""Format-fluency eval: does a model read terse's compressed form as accurately
as raw JSON?

This answers the proxy's one open question (TECHNICAL.md "Known Limitations"):
*correctness* is test-covered — `decompress(compress(x)) == x` — but *usefulness*
rests on an untested assumption, that a model reading the table/legend form in place
of raw JSON loses no comprehension. We measure it the way this project measures
everything: deterministically, cross-model, and never as a single black-box number.

Method (the honesty bar, principle #24):
  - Ground truth is computed from the parsed records (count / lookup / enumerate /
    aggregate) and checked programmatically — no LLM-as-judge, which would re-import
    the nondeterminism we are trying to measure away.
  - Questions deliberately target the two transforms most likely to cost
    comprehension: `~N` dictionary-alias resolution and column->value mapping over
    wide/long positional tables (the under-enumeration gap the row-count hint exists
    for). Each question is tagged with the transform it stresses, so the report says
    *which* transform costs comprehension, not just an aggregate.
  - Scoring is PAIRED: the same questions, same order, over raw vs terse (+/- a
    one-time format primer). A regression is a question the model got right on raw
    and wrong on terse — a stronger signal than two independent accuracy rates.
  - The verdict gates on the WORST model, not the mean: a format that helps one model
    but breaks the real consumer is a regression (mirrors how `validate` reports
    cross-tokenizer divergence rather than averaging it away).

The answerer is a pluggable `(system, user) -> reply` callable, so the pure core
(question generation + scoring) runs offline with no network or key. The live backend
(`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible endpoint — the
broker pool or a loopback gateway — and adds zero new dependencies.
"""

from __future__ import annotations

import json
import re
import urllib.request
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Callable

from . import text_diff
from .capture import LONG_TEXT, OTHER, classify_shape, extract_records
from .tokenize import count_cl100k
from .transforms import compress, compress_structure, dict_encode, diff_wire, minify, _uniform_dict_list

# Loopback hosts where cleartext http is safe (never leaves the machine), so a Bearer
# key over http to one of these is fine — a local LiteLLM/CCR gateway is a common setup.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# An answerer takes (system_prompt, user_prompt) and returns the model's reply text.
# Empty system_prompt means "no system message".
Answerer = Callable[[str, str], str]

# One-time format primer. Deliberately short — in deployment it would be a single
# system note, not a per-call preamble (which would cost tokens on every call).
PRIMER = (
    "Some data below is in 'terse' compressed JSON — a lossless, denser encoding. "
    "Read it as the equivalent expanded JSON:\n"
    '- A table object {"__terse_table__":1,"n":N,"cols":[...],"rows":[[...],...]} '
    "is N records. Each row is POSITIONAL: the i-th value in a row belongs to the "
    'i-th column name in "cols". "n" is the exact row count — use it to check you '
    "read every row.\n"
    '- A dict object {"__terse_dict__":1,"legend":{"~0":"value",...},"data":...} '
    'means every "~K" token appearing inside "data" is shorthand for legend["~K"]; '
    "substitute the legend value back wherever you see its alias."
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Question generation — deterministic, targeting terse's risky transforms
# --------------------------------------------------------------------------- #
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
                for ck, cv in parent.items():
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


def gen_questions(obj: Any) -> list[Question]:
    """Generate deterministic, programmatically-checkable questions for a payload.

    Uniform record-shaped payloads (what terse tabularizes) yield questions directly;
    payloads whose records the strict extractor skips (structure's dict-map of non-uniform
    symbol lists) fall through to `_nested_questions` (#71); everything else returns [].
    Selection is fully deterministic so a re-run is reproducible.
    """
    # Structure-shaped payloads (a dict-map of records) need GROUP-SCOPED questions; prefer
    # them over the uniform extractor, which would otherwise emit an unscoped, ambiguous
    # question from whichever group's child list happens to be uniform (#71).
    nested = _nested_questions(obj)
    if nested:
        return nested
    records = extract_records(obj)
    if not records:
        return []
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


# --------------------------------------------------------------------------- #
# Scoring — deterministic, lenient on formatting, strict on the value
# --------------------------------------------------------------------------- #
def _norm_scalar(s: str) -> str:
    return s.strip().strip("\"'").strip().lower()


def _parse_list(reply: str) -> list | None:
    """Best-effort: a JSON array if present, else a comma/newline split. We instructed
    a JSON array, so the split is only a courtesy against formatting quirks."""
    start, end = reply.find("["), reply.rfind("]")
    if start != -1 and end > start:
        try:
            v = json.loads(reply[start:end + 1])
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    parts = [p.strip().strip("\"'") for p in re.split(r"[,\n]", reply) if p.strip()]
    return parts or None


def _matches_number(reply: str, expected: Any) -> bool:
    """True iff the expected number appears anywhere in the reply. Matching ANY number
    (not just the first) tolerates prose like "there are 6 records" without being fooled
    by a leading incidental number."""
    return any(abs(float(tok) - float(expected)) < 1e-9 for tok in _NUM.findall(reply))


def _extract_json(reply: str) -> Any:
    """Pull the first JSON object/array out of a reply (tolerating surrounding prose).
    Returns a sentinel-free value or raises ValueError if none parses."""
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = reply.find(open_c), reply.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(reply[i:j + 1])
            except json.JSONDecodeError:
                pass
    return json.loads(reply)  # last resort; raises if not JSON


def score(qtype: str, expected: Any, reply: str) -> bool:
    """True iff the reply conveys the expected answer. Tolerates surrounding prose/
    quotes; compares the value exactly (numbers within float epsilon). No blanket
    empty-reply reject — an empty reply only matches an empty expected scalar, which
    each branch already decides correctly."""
    reply = reply.strip()
    if qtype == "count" or (qtype == "aggregate" and _is_number(expected)):
        return _matches_number(reply, expected)
    if qtype == "enumerate":
        got = _parse_list(reply)
        if got is None:
            return False
        return [_norm_scalar(str(x)) for x in got] == [_norm_scalar(str(x)) for x in expected]
    if qtype == "deref":
        try:
            return _extract_json(reply) == expected  # JSON value-equality (dict order-insensitive)
        except (json.JSONDecodeError, ValueError):
            return False
    # lookup / generic scalar
    if _is_number(expected):
        return _matches_number(reply, expected)
    return _norm_scalar(reply) == _norm_scalar(str(expected))


# --------------------------------------------------------------------------- #
# Running an eval — live (pluggable answerer) and offline (pack + responses)
# --------------------------------------------------------------------------- #
def _user_prompt(prompt: str, instruction: str, data: str) -> str:
    return f"{prompt}\n{instruction}\n\nDATA:\n{data}"


def _safe_ask(answerer: Answerer, system: str, user: str) -> str:
    """Call the model, but never let one failed call abort a long multi-model run —
    a transport error / rate limit / refusal scores as a wrong answer, not a crash."""
    try:
        return answerer(system, user)
    except Exception:
        return ""


def _ask_n(answerer: Answerer, system: str, user: str,
           qtype: str, expected: Any, trials: int) -> int:
    """Ask the same question `trials` times; return how many replies scored correct
    (0..trials). Repeating at temperature 0 is not redundant — it surfaces the
    provider-side nondeterminism (batching / MoE routing) behind the ~5pt run-to-run
    accuracy wobble the report's binomial bound quantifies."""
    return sum(score(qtype, expected, _safe_ask(answerer, system, user)) for _ in range(trials))


def run_payload(obj: Any, raw_text: str, answerer: Answerer,
                primer: str = PRIMER, trials: int = 1) -> list[dict]:
    """Ask one payload's questions over raw / terse / terse+primer, `trials` times each.

    Each returned row carries per-form success COUNTS (0..trials) plus `trials`, not
    booleans. At trials=1 a count is 0 or 1 — truthy/falsy exactly like the old bool —
    so every existing aggregation keeps working unchanged.
    """
    terse_text = compress(obj)
    out: list[dict] = []
    for q in gen_questions(obj):
        raw_u = _user_prompt(q.prompt, q.instruction, raw_text)
        terse_u = _user_prompt(q.prompt, q.instruction, terse_text)
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "raw_ok": _ask_n(answerer, "", raw_u, q.qtype, q.expected, trials),
            "terse_ok": _ask_n(answerer, "", terse_u, q.qtype, q.expected, trials),
            "primer_ok": _ask_n(answerer, primer, terse_u, q.qtype, q.expected, trials),
        })
    return out


def run_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                primer: str = PRIMER, trials: int = 1) -> dict:
    """Run the eval for each named answerer over every record-shaped payload.

    Returns {model_name: [scored_row, ...]} where each row carries tool/sha plus the
    per-form success counts and trial count. Non-record payloads are skipped (they
    generate no questions — terse does not transform them).
    """
    results: dict[str, list[dict]] = {}
    for name, fn in answerers.items():
        rows: list[dict] = []
        for env in envelopes:
            try:
                obj = json.loads(env["raw"])
            except (json.JSONDecodeError, TypeError):
                continue
            for row in run_payload(obj, env["raw"], fn, primer, trials):
                rows.append({"tool": env["tool"], "sha": env.get("sha", "?"), **row})
        results[name] = rows
    return results


# --------------------------------------------------------------------------- #
# Cross-call diff fluency — does the model read a diff as well as the full result?
# (Issue #1 risk item: the round-trip gate proves the diff reconstructs, NOT that the
# model reads it. This measures the second thing, with NO system primer — only the diff's
# inline note, as the proxy delivers it — so flipping `proxy --diff` on is a measured
# decision under production conditions.)
# --------------------------------------------------------------------------- #
def run_diff_payload(prev_obj: Any, curr_obj: Any, answerer: Answerer,
                     tool: str = "", trials: int = 1) -> list[dict]:
    """Does the model answer questions about the CURRENT result as well from
    (previous full result + diff) as from the full current result?

    Both forms are asked with NO system primer — the proxy can't set one, so the diff's
    inline note is the only format guidance, exactly as in production. This is the
    honest test of #9's shortened note (an earlier version fed a full DIFF_PRIMER the
    proxy can't deliver, overstating comprehension).

    Returns rows carrying full-terse (`terse_ok`) and diff-form (`diff_ok`) success
    counts over the SAME questions. [] if curr is not record-shaped or no lossless diff
    applies (nothing to compare)."""
    questions = gen_questions(curr_obj)
    if not questions:
        return []
    wire = diff_wire(prev_obj, curr_obj, tool)
    if wire is None:
        return []
    curr_terse = compress(curr_obj)
    diff_data = f"PREVIOUS RESULT:\n{compress(prev_obj)}\n\nUPDATE (diff against it):\n{wire}"
    out: list[dict] = []
    for q in questions:
        full_u = _user_prompt(q.prompt, q.instruction, curr_terse)
        diff_u = _user_prompt(q.prompt, q.instruction, diff_data)
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "terse_ok": _ask_n(answerer, "", full_u, q.qtype, q.expected, trials),
            "diff_ok": _ask_n(answerer, "", diff_u, q.qtype, q.expected, trials),
        })
    return out


def _iter_consecutive_pairs(envelopes: list[dict]):
    """Group envelopes by tool, sort each group by sha (for determinism — the order the
    proxy would see them), and yield consecutive (tool, sha, prev_raw, curr_raw) tuples.
    Shared pairing scaffold for run_diff_fluency and run_text_diff_fluency; each applies
    its own JSON/text domain filter on top of this."""
    by_tool: dict[str, list[dict]] = {}
    for env in envelopes:
        by_tool.setdefault(env["tool"], []).append(env)
    for tool, envs in by_tool.items():
        envs = sorted(envs, key=lambda e: e.get("sha", ""))
        for prev_env, curr_env in zip(envs, envs[1:]):
            yield tool, curr_env.get("sha", "?"), prev_env["raw"], curr_env["raw"]


def _aggregate_by_model(pairs: list[tuple], answerers: dict[str, Answerer], trials: int,
                        payload_fn) -> dict:
    """Shared per-model aggregation loop for run_diff_fluency/run_text_diff_fluency:
    run `payload_fn` over every pair for every answerer and collect the rows."""
    results: dict[str, list[dict]] = {}
    for name, fn in answerers.items():
        rows: list[dict] = []
        for tool, csha, a, b in pairs:
            for row in payload_fn(a, b, fn, tool, trials=trials):
                rows.append({"tool": tool, "sha": csha, **row})
        results[name] = rows
    return results


def run_diff_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                     trials: int = 1) -> dict:
    """Run the diff-fluency eval over consecutive same-tool payload PAIRS (sorted by sha
    for determinism — the order the proxy would see them). Returns {model: [rows]}."""
    pairs: list[tuple] = []
    for tool, csha, prev_raw, curr_raw in _iter_consecutive_pairs(envelopes):
        try:
            prev_obj = json.loads(prev_raw)
            curr_obj = json.loads(curr_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        pairs.append((tool, csha, prev_obj, curr_obj))
    return _aggregate_by_model(pairs, answerers, trials, run_diff_payload)


# --------------------------------------------------------------------------- #
# Text-diff fluency — the text-payload analogue of the diff eval above.
#
# run_diff_payload/run_diff_fluency answer "does a model read a diff against a
# record-shaped JSON payload as well as the full-terse form?" These functions ask
# the same question for unstructured text (text_diff.py, Tier 0.7): does a model
# reconstruct the CURRENT text as accurately from (previous text + text-diff) as
# from the full current text? No new protocol is needed — it's the same paired
# single-shot Answerer comparison, just with a different question source and a
# different "form" — so these live here next to run_diff_payload rather than in
# their own module. (dropeval.py earns its own module because drop-to-retrieve is
# a genuinely different, stateful 2-turn tool-calling protocol; this isn't.)
# --------------------------------------------------------------------------- #
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


def run_text_diff_payload(prev: str, curr: str, answerer: Answerer,
                          tool: str = "", trials: int = 1) -> list[dict]:
    """Does the model answer questions about the CURRENT text as well from
    (previous text + text-diff) as from the full current text?

    Unlike run_diff_payload's "terse" form (Tier-0 compressed JSON), the control form
    here is just the raw current text — Tier 0 doesn't touch non-JSON payloads at all.
    Field names stay terse_ok/diff_ok anyway (not raw_ok/diff_ok) so the row shape is
    identical to run_diff_payload's — that identity is what lets diff_gap_rows/
    build_terminal_diff_report be reused UNCHANGED for text-diff-eval (a deliberate
    reuse-over-duplication tradeoff).

    Computes the diff wire exactly ONCE (unlike calling gen_text_diff_questions, whose
    own gate would otherwise recompute the same content-defined-chunking diff a second
    time) — text_diff_encode's chunk+hash+round-trip-proof cost is not free."""
    wire = text_diff.text_diff_wire(prev, curr, tool)
    if wire is None:
        return []
    questions = _text_diff_questions_from_lines(curr)
    diff_data = f"PREVIOUS RESULT:\n{prev}\n\nUPDATE (diff against it):\n{wire}"
    out: list[dict] = []
    for q in questions:
        full_u = _user_prompt(q.prompt, q.instruction, curr)
        diff_u = _user_prompt(q.prompt, q.instruction, diff_data)
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "terse_ok": _ask_n(answerer, "", full_u, q.qtype, q.expected, trials),
            "diff_ok": _ask_n(answerer, "", diff_u, q.qtype, q.expected, trials),
        })
    return out


def run_text_diff_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                          trials: int = 1) -> dict:
    """Same tool-pairing loop as run_diff_fluency, inverted: only pairs envelopes whose
    raw text is NOT valid JSON on EITHER side (text-diff's domain; JSON payloads are
    run_diff_fluency's domain instead) — classify_shape (already used by measure.py)
    catches the Python-3.11 RecursionError on deeply-nested JSON that a bare
    json.loads/except wouldn't."""
    pairs: list[tuple] = []
    for tool, csha, prev_raw, curr_raw in _iter_consecutive_pairs(envelopes):
        if classify_shape(curr_raw) not in (LONG_TEXT, OTHER):
            continue  # curr is JSON-shaped -> run_diff_fluency's domain, not this one's
        if classify_shape(prev_raw) not in (LONG_TEXT, OTHER):
            continue  # prev is JSON-shaped -> not a text-to-text transition
        pairs.append((tool, csha, prev_raw, curr_raw))
    return _aggregate_by_model(pairs, answerers, trials, run_text_diff_payload)


def build_pack(envelopes: list[dict], primer: str = PRIMER, trials: int = 1) -> dict:
    """Emit a self-contained eval pack (prompts + raw/terse forms + ground truth) so a
    run can be driven by any model client (e.g. via the secret-broker) and scored later
    with `score_pack`. Keyless: no model is called here. `trials` is a hint to the
    external driver: collect that many replies per form (as a list) for a bounded verdict."""
    payloads = []
    for env in envelopes:
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            continue
        qs = gen_questions(obj)
        if not qs:
            continue
        payloads.append({
            "tool": env["tool"], "sha": env.get("sha", "?"),
            "raw": env["raw"], "terse": compress(obj),
            "questions": [{
                "qid": q.qid, "qtype": q.qtype, "transform": q.transform,
                "prompt": q.prompt, "instruction": q.instruction, "expected": q.expected,
            } for q in qs],
        })
    return {"primer": primer, "trials": trials, "payloads": payloads}


def _score_form(qtype: str, expected: Any, form_val: Any) -> tuple[int, int]:
    """(successes, trials) for one form's collected reply(s). A single string is one
    trial; a list of strings is N trials (the multi-trial pack form). Returns (0, 1)
    for a missing/empty single reply, matching the prior single-trial behaviour."""
    replies = form_val if isinstance(form_val, list) else [form_val]
    if not replies:
        return 0, 0
    successes = sum(score(qtype, expected, r) for r in replies if isinstance(r, str))
    return successes, len(replies)


def score_pack(pack: dict, responses: dict) -> dict:
    """Score externally-collected responses against a pack's ground truth.

    responses: {model: {sha: {qid: {"raw": str|[str], "terse": ..., "primer": ...}}}}.
    A list value is multi-trial (N replies for that form). Returns the same
    {model: [scored_row,...]} shape as `run_fluency`, with per-form success counts.
    """
    index: dict[tuple, dict] = {}
    for p in pack["payloads"]:
        for q in p["questions"]:
            index[(p["sha"], q["qid"])] = {"tool": p["tool"], **q}
    results: dict[str, list[dict]] = {}
    for model, by_sha in responses.items():
        rows: list[dict] = []
        for sha, by_qid in by_sha.items():
            for qid, forms in by_qid.items():
                meta = index.get((sha, qid))
                if meta is None:
                    continue
                qtype, expected = meta["qtype"], meta["expected"]
                raw_k, raw_t = _score_form(qtype, expected, forms.get("raw", ""))
                terse_k, terse_t = _score_form(qtype, expected, forms.get("terse", ""))
                primer_k, primer_t = _score_form(qtype, expected, forms.get("primer", ""))
                rows.append({
                    "tool": meta["tool"], "sha": sha, "qid": qid,
                    "qtype": qtype, "transform": meta["transform"],
                    # forms are collected with the same trial count (build_pack's hint);
                    # store the max so an uneven hand-built file still reports sanely.
                    "trials": max(raw_t, terse_t, primer_t, 1),
                    "raw_ok": raw_k, "terse_ok": terse_k, "primer_ok": primer_k,
                })
        results[model] = rows
    return results


def token_summary(envelopes: list[dict]) -> list[dict]:
    """Per-payload raw vs terse cl100k token counts, so the report shows comprehension
    AND the saving it buys in the same place (record-shaped payloads only)."""
    rows: list[dict] = []
    for env in envelopes:
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not extract_records(obj):
            continue
        rows.append({
            "tool": env["tool"], "sha": env.get("sha", "?"),
            "raw_tok": count_cl100k(env["raw"]), "terse_tok": count_cl100k(compress(obj)),
        })
    return rows


# --------------------------------------------------------------------------- #
# Live backends — zero new dependencies
# --------------------------------------------------------------------------- #
def openai_answerer(base_url: str, api_key: str, model: str,
                    temperature: float = 0.0, timeout: int = 60) -> Answerer:
    """OpenAI-compatible /chat/completions answerer over stdlib urllib. Covers the
    broker pool (OpenRouter et al.) without an SDK dependency. temperature 0 for
    reproducibility."""
    parts = urlsplit(base_url)
    if api_key and parts.scheme == "http" and (parts.hostname or "").lower() not in _LOOPBACK_HOSTS:
        # An http:// base URL sends `Authorization: Bearer <key>` in cleartext — refuse
        # it for a non-loopback host rather than silently leak the key on the wire. A
        # loopback host (localhost LiteLLM/CCR) never leaves the machine, so it's allowed.
        raise ValueError(
            f"terse fluency: refusing to send an API key over cleartext http to "
            f"{parts.hostname!r} — use https, or a loopback host for a local gateway")
    url = base_url.rstrip("/") + "/chat/completions"

    def ask(system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Some OpenAI-compatible gateways return 200 with an error body (no choices);
        # surface a clear message instead of a bare KeyError.
        if "choices" not in data:
            raise RuntimeError(f"{model}: no choices in response: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"] or ""

    return ask
