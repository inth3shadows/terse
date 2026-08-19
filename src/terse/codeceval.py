"""Behavioral eval for the CODEC tier (#295): does a real tool-calling model's downstream
tool argument stay structurally identical whether it read the raw payload or terse's
compressed form?

`fluency.py` asks a different, weaker question — can a model answer a comprehension
question about terse's form, scored as a fixed 5% accuracy-gap tolerance. #295 argues that
tolerance is the wrong kind of answer: it is a budget for semantic damage, and terse's codec
tier claims to be *unconditionally lossless* (round-trip proven). A tolerance one layer up,
at the reader instead of the encoder, quietly reintroduces exactly the lossiness the codec
promises not to have.

It also hides WHERE accuracy is lost. Comprehension failures concentrate in the `deref`
question — reconstructing terse's compressed form (aliased `~N` legend entries, positional
table rows) back into the original JSON structure. That is exactly what an agent does when
it feeds a value from one tool's result into the next tool call's arguments. A `deref` miss
is not "a wrong answer" — it is a **malformed downstream tool argument**: a positional array
handed to a call expecting a keyed object, or an explicit `null` written where a record
simply had no such key. Averaging that against a `count` question's difficulty prices
structural corruption and a hard arithmetic question identically.

## What this module tests, and what it does not

This is deliberately narrower than `fluency.py`'s comprehension sweep: only `deref`
questions (`Question.qtype == "deref"`, `fluency/questions.py`) enter this eval. Every other
`qtype` (count/lookup/enumerate/aggregate) stays a comprehension question — this module
does not re-litigate whether a model can count. `deref` is the one question whose ANSWER
*is* a structural reconstruction, which makes it the one question a downstream-outcome
comparison can meaningfully replace a comprehension comparison for.

The comparison itself: the same `deref` question is put to a tool-calling model twice, once
fed the raw payload and once fed terse's compressed form, both times via a single stub tool
(`RECORD_VALUE_TOOL_DEF`) the model must call with the reconstructed value as its argument.
Scoring compares the tool-call ARGUMENT the model emits against `Question.expected` by plain
structural equality (`_value_matches`) — not a text-extraction heuristic, and not a
free-text comprehension score. This is what #295 calls "material equivalence": would the
model's downstream tool call carry the same value regardless of which form it read.

**No system primer** on either arm — the system message is OMITTED entirely (matching
`fluency.answerers`' `if system:` guard, not `harnesses.run_payload`'s bare `terse_ok` arm,
which was corrected here after review found the first draft sent an empty-string system
message where `fluency`'s own no-primer arm sends none at all — some OpenAI-compatible
backends reject an empty message). This is a deliberate, separate choice from #249's
primer-necessity question: this eval asks whether the codec is materially safe, not whether
a primer helps a model read it, and testing without one is the more conservative bound —
a pass here implies a pass with a primer, not the reverse.

**Tool-schema note, spiked live before this module was written** (against the gateway
`dropeval`'s own tests target, deepseek-v4-flash): a `dict`/`list` argument round-trips
through an untyped OpenAI-style `{"value": {"description": "..."}}` property unmangled. A
bare scalar (e.g. `42`) came back coerced to the string `"42"` — but `deref`'s `expected` is
ALWAYS a `dict` or `list` (`questions.py`'s `blobcol` selection requires
`isinstance(r[c], (dict, list))` for every record), so the scalar-coercion case never reaches
this eval's data and needs no workaround here. If this module is ever extended to a `qtype`
whose `expected` can be a bare scalar, re-spike before trusting the argument type.

Ground truth, tool-loop plumbing (`Turn`, `ToolCall`, `ToolAnswerer`, `openai_tool_answerer`,
`_safe_call`) are reused from `dropeval.py` rather than reinvented — same protocol, same
fail-open contract, same "an unanswered call is not a wrong answer" discipline (#263/#268).
Deliberately a SIBLING module rather than a mode inside `dropeval.py`: that file's docstring,
`TERSE_PRIMER` import, and 2-turn retrieve-hop protocol are drop-tier-specific, and #295
makes tier separation a first-class requirement ("split by tier — do not conflate"). This
module imports dropeval's tool-loop primitives; it does not extend dropeval's drop-specific
vocabulary.

Row shape emitted here carries the same KEYS as `harnesses.run_payload`'s convention
(`<form>_ok`/`<form>_trials`/`trials`/`fails`/`attempts`) so it slots into `report.arm_gap`/
`report._form_stats`/`report.paired_rows` unchanged — no new pairing logic needed, and
`tests/test_gap_gate_boundary.py`'s AST allowlist stays untouched as long as this module only
ever reaches those through `arm_gap`. One deliberate divergence: `<form>_trials` here is
always the FIXED `trials` count, never reduced by an errored call — see
`run_codec_payload`'s docstring for why (review finding 3 on PR #302: this eval renders a
safety verdict, and a silent non-answer must not be able to shrink itself out of the sample).
"""

from __future__ import annotations

import json
from typing import Any

from . import fluency
from .dropeval import ToolAnswerer, Turn, _safe_call

# OpenAI function-calling schemas can't cleanly express "any JSON type" — see the module
# docstring's spike note. `{"description": ...}` with no `"type"` is the untyped form that
# was measured to pass dict/list through unmangled; `openai_tool_answerer`'s `_to_openai_tool`
# reads this exactly like `proxy.RETRIEVE_TOOL_DEF`'s `inputSchema`.
RECORD_VALUE_TOOL = "terse.record_answer"
RECORD_VALUE_TOOL_DEF = {
    "name": RECORD_VALUE_TOOL,
    "description": ("Record the value you were asked to report. Call this with the exact "
                    "value as its argument — do not answer in prose."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "value": {"description": "The requested value, as its native JSON type "
                                     "(object or array for a `deref` question)."},
        },
        "required": ["value"],
    },
}


def gen_codec_questions(obj: Any) -> list[fluency.Question]:
    """The `deref` subset of `fluency.gen_questions(obj)` — the only question type whose
    answer is a structural reconstruction, and so the only one this eval's downstream-
    argument comparison can stand in for. `[]` if the payload has no `deref` question
    (no column of whole dict/list values) — fails closed, same as `dropeval.gen_drop_questions`."""
    return [q for q in fluency.gen_questions(obj) if q.qtype == "deref"]


def _value_matches(got: Any, expected: Any) -> bool:
    """Structural equality between a parsed tool-call argument and `Question.expected`.

    Plain `==`: Python dict/list equality is order-insensitive on keys and exact on values,
    which is exactly what `fluency/scoring.py`'s own `deref` case does over a parsed reply.
    Deliberately NOT routed through `fluency.scoring.score` — that function's job is
    extracting a JSON value out of free TEXT, and a tool-call argument is already parsed;
    reparsing it through a text extractor would reintroduce the free-text indirection #295
    is trying to remove. `==` also gives the exact distinction a `deref` failure destroys:
    `{"a": 1}` and `{"a": 1, "b": None}` are NOT equal — an absent key and an explicit null
    are different values, not two spellings of the same thing."""
    return got == expected


def _codec_instruction() -> str:
    """The instruction half of the user prompt for a codec-eval trial.

    Deliberately NOT `question.instruction` — that string is written for `fluency.py`'s
    single-shot text-reply protocol ("Reply with only that value as compact JSON, and
    nothing else"), which tells a compliant model to answer in PROSE. `tool_choice` is
    `"auto"` (`dropeval.openai_tool_answerer`), so a model that obeys that instruction
    literally never calls `RECORD_VALUE_TOOL` at all — a harness-caused miss with nothing
    to do with the codec, found by live trace during review of PR #302 (finding 2). This
    eval's whole premise is a downstream TOOL CALL, so the instruction has to ask for one."""
    return ("Call the tool with that value as its argument. Do not reply in prose, and do "
           "not call any other tool.")


def _ask_codec_question(question: fluency.Question, payload_text: str,
                        answerer: ToolAnswerer) -> tuple[bool, bool]:
    """One trial: ask `question` over `payload_text`, expect a `RECORD_VALUE_TOOL` call.
    Returns (matched, errored).

    `errored` means the call never produced a scorable turn — either a transport failure
    (`_safe_call`'s except branch) or a live backend returning 200 with neither text nor a
    tool call (`dropeval.openai_tool_answerer`'s `no_content` branch; the two are
    indistinguishable through the shared `Turn` contract, so both are treated the same way
    here). Deliberately scored as a MISS, not excluded from the trial count — mirroring
    `dropeval._run_question`'s documented stance ("excluding would be the dangerous
    direction"), not `fluency.harnesses`' per-form-trial-reduction convention: this eval
    renders a SAFE/UNSAFE verdict, and a silent non-answer — the worst possible outcome for
    a downstream tool call — must not be able to shrink itself out of the denominator and
    help a run reach SAFE (review finding 3 on PR #302). A model that reaches the tool
    definition and declines to call anything is scored the same way, for the same reason
    (not a transport error, but still the failure mode this eval exists to catch)."""
    messages: list[dict] = [
        {"role": "user", "content": fluency._user_prompt(question.prompt, _codec_instruction(),
                                                          payload_text)},
    ]
    turn: Turn = _safe_call(answerer, messages)
    if turn.error:
        return False, True  # counted as a miss by the caller, kept in the fixed denominator
    record_calls = [c for c in turn.tool_calls if c.name == RECORD_VALUE_TOOL]
    if not record_calls:
        return False, False
    got = record_calls[-1].arguments.get("value")
    return _value_matches(got, question.expected), False


def run_codec_payload(obj: Any, raw_text: str, answerer: ToolAnswerer,
                      trials: int = 1) -> list[dict]:
    """Ask each `deref` question in `obj` over raw vs terse, `trials` times each, via the
    tool-calling protocol. One row per question.

    `raw_trials`/`terse_trials` are the FIXED `trials` count, not reduced by errors (see
    `_ask_codec_question`'s docstring) — this differs from `fluency.harnesses.run_payload`'s
    convention on purpose, even though the row otherwise matches its key shape closely
    enough to flow through `report.arm_gap`/`_form_stats`/`paired_rows` unchanged. `fails`/
    `attempts` still track raw call losses, so `report._unmeasured`'s >20%-loss gate still
    catches a substantially-down backend and reports UNRESOLVED rather than a confident
    verdict computed over a small, self-selected surviving sample."""
    terse_text = fluency.compress(obj)
    out: list[dict] = []
    for q in gen_codec_questions(obj):
        raw_ok = terse_ok = raw_fail = terse_fail = 0
        for _ in range(trials):
            ok, err = _ask_codec_question(q, raw_text, answerer)
            raw_fail += int(err)
            raw_ok += int(ok)  # an errored call scores as a miss, not an exclusion
        for _ in range(trials):
            ok, err = _ask_codec_question(q, terse_text, answerer)
            terse_fail += int(err)
            terse_ok += int(ok)
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "raw_ok": raw_ok, "terse_ok": terse_ok,
            "raw_trials": trials,
            "terse_trials": trials,
            "fails": raw_fail + terse_fail,
            "attempts": trials * 2,
        })
    return out


def run_codec_fluency(envelopes: list[dict], answerers: dict[str, ToolAnswerer],
                      trials: int = 1) -> dict[str, list[dict]]:
    """Run the codec-tier eval for each named tool-capable answerer over every payload in
    the corpus that has at least one `deref` question. Mirrors `dropeval.run_drop_fluency`'s
    envelope-outer/model-inner nesting (question generation is model-independent, so it is
    derived once per envelope, not once per (model, envelope)) and `fluency.run_fluency`'s
    row-tagging convention.

    Each row additionally carries `"tool"` AND `"shape"` (`env["shape"]`, already written at
    capture time by `capture.record` — `capture.py`'s `classify_shape` call). No existing
    harness stamps `shape` onto its rows; it is the one field #295's per-`(tool, shape)`
    verdict needs that the comprehension harness never needed, because the comprehension
    report has always pooled globally rather than per shape bucket."""
    results: dict[str, list[dict]] = {name: [] for name in answerers}
    for env in envelopes:
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            continue  # deref needs parsed JSON structure; a non-JSON/text payload has none
        if not gen_codec_questions(obj):
            continue
        for name, answerer in answerers.items():
            for row in run_codec_payload(obj, env["raw"], answerer, trials=trials):
                results[name].append({
                    "tool": env.get("tool", "?"),
                    "shape": env.get("shape", "unknown"),
                    "sha": env.get("sha", "?"),
                    **row,
                })
    return results
