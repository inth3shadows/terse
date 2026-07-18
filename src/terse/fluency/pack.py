"""Offline eval packs — build, score, and the one-time format primer (#78 split).

A pack is a self-contained eval file (prompts + raw/terse forms + ground truth) so
a run can be driven by any model client (e.g. via the secret-broker) and scored
later, keylessly, with `score_pack`.
"""

from __future__ import annotations

import json

from ..capture import extract_records
from ..tokenize import count_cl100k
from ..transforms import compress
from .questions import gen_questions
from .scoring import _score_form

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
                    # `trials` (the max) is the header/display count. The PER-FORM counts
                    # below are what the accuracy denominator uses: with an uneven
                    # hand-built pack (one form collected fewer replies), dividing a
                    # sparser form's successes by the shared max understates it. Each form
                    # is scored over its OWN trial count instead (see _form_stats).
                    "trials": max(raw_t, terse_t, primer_t, 1),
                    "raw_ok": raw_k, "terse_ok": terse_k, "primer_ok": primer_k,
                    "raw_trials": raw_t, "terse_trials": terse_t, "primer_trials": primer_t,
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
