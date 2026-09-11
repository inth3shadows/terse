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

This is deliberately narrower than `fluency.py`'s comprehension sweep: only `deref` and
`enumerate` questions (`CODEC_QTYPES`, `fluency/questions.py`) enter this eval. `count` and
`aggregate` stay comprehension questions — their answers are computed, not carried, so this
module does not re-litigate whether a model can count. Each admitted type is a value an
agent would carry VERBATIM into its next tool call, and each stresses a different half of
what the codec does:

- `deref` — reconstruct one record's whole object/array column: alias (`~N`) and structure
  resolution.
- `enumerate` — read one column back out of EVERY row, in order: the positional table read
  the tabularizer introduces. Added for #403 Blocker 2: deref-only covered tools carrying
  11.9% of the codec's 30-day savings, because `kb.read.search`, `kb.read.list_principles`
  and `secret.list_credentials` have no container column; with `enumerate` it is 21.9%.

`lookup` is NOT admitted: its answer is a bare scalar, which the spike note below measured
being coerced to a string.

The comparison itself: the same question is put to a tool-calling model twice, once
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
bare scalar (e.g. `42`) came back coerced to the string `"42"` — but every admitted `qtype`
answers with a container: `deref`'s `expected` is ALWAYS a `dict` or `list` (`questions.py`'s
`blobcol` selection requires `isinstance(r[c], (dict, list))` for every record) and
`enumerate`'s is ALWAYS a list, so the bare-scalar case never reaches this eval's data. The
scalars INSIDE an `enumerate` list are checked by the pre-flight, whose payloads carry both
string and integer ids. If this module is ever extended to a `qtype` whose `expected` can
be a bare scalar, re-spike before trusting the argument type.

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
import time
from collections.abc import Callable
from typing import Any

from . import capture, fluency
from .dropeval import ToolAnswerer, Turn, _safe_call
from .tokenize import count_cl100k
from .transforms import has_terse_marker

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


CODEC_QTYPES = ("deref", "enumerate")


def codec_changes(obj: Any) -> bool:
    """Does the codec actually encode this payload — a table, a legend — rather than pass it
    through minified?

    A payload it leaves alone gives both arms the same JSON, so every trial is a free match:
    it cannot fail, and it would still count toward `_CODEC_MIN_TRIALS` and push a cell
    toward SAFE. No `deref` question ever landed on such a payload (all 28 on the live
    corpus were encoded), but `enumerate` does: 10 of its first 147 corpus questions were on
    payloads the codec declined (#403 Blocker 2). Skipped, never scored.

    Output that does not parse as JSON counts as a change: it is not a passthrough, so it
    is exactly what the eval should look at."""
    try:
        return has_terse_marker(json.loads(fluency.compress(obj)))
    except ValueError:
        return True


def gen_codec_questions(obj: Any) -> list[fluency.Question]:
    """The `CODEC_QTYPES` subset of `fluency.gen_questions(obj)` — the question types whose
    answer is a value carried verbatim into a downstream tool argument, and container-typed
    (see the module docstring). `[]` if the payload has neither (no id column to enumerate,
    no column of whole dict/list values) — fails closed, same as
    `dropeval.gen_drop_questions`."""
    return [q for q in fluency.gen_questions(obj) if q.qtype in CODEC_QTYPES]


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


def encodes_as_json_string(got: Any, expected: Any) -> bool:
    """Did the model send the RIGHT VALUE as a JSON *string* instead of as its native type?

    `{"value": "[\\"Navidrome\\"]"}` where `["Navidrome"]` was asked for: content exact,
    type wrong. `_value_matches` rejects it, correctly — a downstream tool declaring
    `"type": "array"` would reject it too, so it is a genuinely malformed tool call, not a
    spelling of a right answer. But it is malformed for a reason that has NOTHING to do
    with the codec: the model encodes the same way on the raw arm and the terse arm, so it
    zeroes BOTH and the eval learns nothing about compression.

    Measured 2026-09-09, live, against `~/.config/terse/session-corpus`: `deepseek-v4-pro`
    stringifies **5 of 5** deref answers, `qwen3-coder-30b-fl` **0 of 5**. That produced a
    0%-accuracy control on 5 of 6 (tool, shape) cells and `report.arm_gap`'s `broken
    control` -> UNRESOLVED, with nothing anywhere naming the cause; it took a live probe to
    find. This predicate exists so the harness can say it instead.

    Not a scoring path. NOTHING calls this to turn a miss into a hit — that would accept an
    argument a strict tool rejects, which is exactly the loosening #295 forbids. It is
    diagnosis only: `preflight_encoding` uses it to name the reason it refuses a model.

    Only a container `expected` qualifies. A string one would make every correctly-answered
    string question look like a mismatch (`json.loads('"a"') == "a"`), and `None` would
    label a real `null` answer (`json.loads("null") is None`) the same way. Every
    `CODEC_QTYPES` answer is a dict or list, so both are defence in depth rather than live
    cases. `RecursionError` is caught because the string is a model's reply: a pathological
    `"[[[[…"` must read as "not this failure mode", not abort the pre-flight."""
    if not isinstance(got, str) or not isinstance(expected, (dict, list)):
        return False
    try:
        return bool(json.loads(got) == expected)
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
        return False


# Two fixed payloads, one per container type a real `deref` answer can take (`questions.py`'s
# `blobcol` requires every cell to be a dict or a list), because a model can stringify one
# type and not the other. The PAYLOADS are fixed rather than drawn from the corpus so the
# pre-flight cannot fail for corpus reasons (a thin corpus is a SEPARATE defect, #403
# Blocker 2; conflating them is what made the 2026-09-09 run unreadable). The QUESTIONS are
# not hand-written: `preflight_questions` derives them with the same `gen_codec_questions`
# the sweep uses, so prompt wording and `expected` cannot drift from what is scored.
# The asked-for values (record index 1, `questions.py`'s `n // 2` pick) are NESTED — a list
# of objects, an object holding a list and an empty object — because real `deref` values
# are, and a model that stringifies only inner containers would otherwise pass here and
# still zero both arms of the sweep. Each payload also yields an `enumerate` question, and
# the two carry string and integer ids respectively, so a model that stringifies the
# scalars inside a list is caught too.
PREFLIGHT_PAYLOADS: tuple[dict, ...] = (
    {"result": [{"id": "a", "rows": [{"k": "x"}]},
                {"id": "b", "rows": [{"k": "y", "v": [1, 2]}, {"k": "z"}]}]},
    {"result": [{"id": 1, "meta": {"owner": "ann"}},
                {"id": 2, "meta": {"owner": "bo", "tags": ["q", "r"], "extra": {}}}]},
)
PREFLIGHT_ATTEMPTS = 3


class PreflightError(RuntimeError):
    """At least one model failed `preflight_encoding`; the run is refused before the sweep."""


def preflight_questions() -> list[tuple[fluency.Question, str]]:
    """`(question, payload text)` for each of `PREFLIGHT_PAYLOADS` — the raw-arm form, the
    same `json.dumps` text a captured envelope's `raw` is.

    Raw arm only, deliberately. The pre-flight asks whether a model can express a container
    argument AT ALL, which is a property of the model and the tool protocol. A miss on the
    terse form is the thing the sweep exists to measure; refusing a model for it here would
    decide the verdict before the run instead of reporting it.

    One question per `CODEC_QTYPES` entry per payload, in generation order. Raises rather
    than returning fewer: if `questions.py` ever stops emitting one of them for these
    payloads, a shorter list would let every model pass a pre-flight that skipped the very
    answer type the sweep then scores."""
    out: list[tuple[fluency.Question, str]] = []
    for payload in PREFLIGHT_PAYLOADS:
        qs = gen_codec_questions(payload)
        got = sorted(q.qtype for q in qs)
        if got != sorted(CODEC_QTYPES):
            raise RuntimeError(f"pre-flight payload {payload!r} yields {got}, expected one "
                               f"each of {sorted(CODEC_QTYPES)} — questions.py changed")
        out.extend((q, json.dumps(payload)) for q in qs)
    return out


def _preflight_miss(turn: Turn, expected: Any) -> str | None:
    """`None` if a scorable pre-flight turn matched, else why not — named by failure mode,
    since the three need different remedies (a prompt, a different model, a harder look)."""
    called, got = _recorded_value(turn)
    if not called:
        return (f"answered without calling {RECORD_VALUE_TOOL} "
                f"(text: {(turn.text or '')[:60]!r}) — this eval scores a tool-call argument")
    if _value_matches(got, expected):
        return None
    if encodes_as_json_string(got, expected):
        return ("sends container arguments as a JSON STRING "
                f"({got!r} for {expected!r}) — right content, wrong type. It would encode "
                "both arms the same way and score 0% on each, so the run would learn nothing "
                "about the codec")
    return f"failed a 2-record pre-flight question: got {got!r}, expected {expected!r}"


def preflight_encoding(answerer: ToolAnswerer, attempts: int = PREFLIGHT_ATTEMPTS) -> str | None:
    """`None` if this model can participate in the codec eval, else a one-line reason.

    The 2026-09-09 run burned 57 minutes and 324 calls to discover that one of its two
    models could not express a container argument. That is a property of the model and the
    tool protocol, knowable in a handful of calls, and independent of the corpus and the
    codec — so it is checked once per model before the sweep.

    Every request goes through `_codec_turn`, the function a real trial sends through, and
    every reply through `_recorded_value`. The first version copied that request inline and
    documented it as shared; review of #403 showed a mutation to the copy alone passed
    every test. `test_preflight_sends_the_request_a_real_trial_sends` pins the equality.

    Each question needs `attempts` SCORABLE answers, and **every** one must match. Measured
    while building this: `deepseek-v4-pro` at `temperature=0.0` answered one call with
    `{"value": "[\\"x\\", \\"y\\"]"}` and another with no `value` at all, so a single call
    diagnoses the same model differently run to run. A model that expresses a container
    argument only sometimes still poisons the arm it is on. The reason reported is the FIRST
    failure seen, with a count, so a flaky model reads as flaky.

    An errored turn (transport failure, or 200 with no content) is NOT an answer: it says
    nothing about encoding, and one timeout must not cost a good model its place in the
    run. It is retried, up to `attempts` extra calls per question; a backend that cannot
    produce `attempts` scorable turns in `2 * attempts` calls fails the pre-flight. When a
    model's ANSWERS had already failed before the backend gave out, that failure leads the
    reason: a JSON-string model behind a flaky gateway must not read as "retry and see"."""
    attempts = max(1, attempts)
    first_reason: str | None = None
    bad = scored = 0
    questions = preflight_questions()
    for qi, (question, payload_text) in enumerate(questions, 1):
        answered = calls = 0
        while answered < attempts and calls < 2 * attempts:
            calls += 1
            turn = _codec_turn(question, payload_text, answerer)
            if turn.error:
                continue
            answered += 1
            reason = _preflight_miss(turn, question.expected)
            if reason is not None:
                bad += 1
                if first_reason is None:
                    first_reason = reason
        scored += answered
        if answered < attempts:
            # `calls` is per question and `bad/scored` runs across questions, so the loss
            # names its question: "6 of 6" after question 1 answered cleanly must not read
            # as a backend that never answered at all (round-3 review).
            lost = (f"backend returned no usable turn on {calls - answered} of {calls} calls "
                    f"for pre-flight question {qi} of {len(questions)}")
            if first_reason is None:
                return f"{lost} — cannot tell whether this model can express a container argument"
            return f"{first_reason} [{bad}/{scored} pre-flight answers failed before the {lost}]"
    if first_reason is None:
        return None
    return f"{first_reason} [{bad}/{scored} pre-flight answers failed]"


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
    turn = _codec_turn(question, payload_text, answerer)
    if turn.error:
        return False, True  # counted as a miss by the caller, kept in the fixed denominator
    called, got = _recorded_value(turn)
    if not called:
        return False, False
    return _value_matches(got, question.expected), False


def _codec_turn(question: fluency.Question, payload_text: str, answerer: ToolAnswerer) -> Turn:
    """The one request a codec-eval question is sent as — a single user message, no system
    message — shared by the sweep (`_ask_codec_question`) and `preflight_encoding`, so a
    model that passes the pre-flight passed on the request it is scored on."""
    messages: list[dict] = [
        {"role": "user", "content": fluency._user_prompt(question.prompt, _codec_instruction(),
                                                          payload_text)},
    ]
    return _safe_call(answerer, messages)


def _recorded_value(turn: Turn) -> tuple[bool, Any]:
    """`(called, value)`: whether the turn called `RECORD_VALUE_TOOL`, and the `value`
    argument of its LAST such call (`None` when the key is absent — which `_value_matches`
    scores as a miss against any `deref` `expected`)."""
    record_calls = [c for c in turn.tool_calls if c.name == RECORD_VALUE_TOOL]
    if not record_calls:
        return False, None
    return True, record_calls[-1].arguments.get("value")


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


def _payload_tokens(raw_text: str, obj: Any) -> dict[str, int]:
    """cl100k token counts for one payload's two arms, stamped onto every row that payload
    produces (#303). Empty when tiktoken is unavailable — `count_cl100k` returns `None`
    there, and a savings table that silently reads a missing count as zero would print a
    100% saving for an unmeasured run.

    Measured on the FORM THE MODEL ACTUALLY READ: `raw_text` is the same string
    `run_codec_payload` puts on the raw arm, and `fluency.compress(obj)` is byte-identical
    to the terse arm's payload. Recomputing the corpus-wide savings from `measure` instead
    would report a different number against the same verdict, for two reasons — both
    MEASURED ONCE, on 2026-09-01, against the 1,524-envelope corpus at
    `~/.config/terse/session-corpus`, and neither one an invariant:

    - `run_codec_fluency` only emits rows for payloads that yield a `deref` question. 8 of
      1,524 did, so the corpus-wide table would cover 190x the payloads the verdict does.
    - 36 of those 1,524 envelopes carried a stored `shape` that `classify_shape(raw)` no
      longer agreed with, so the two tables would not bucket the same payload the same way.
      That was `#355`, FIXED — `capture.envelope_shape` re-classifies at the read, so this
      second reason is gone. The first stands on its own and is why the split remains.

    Both figures will drift as the corpus grows and as `classify_shape` changes. They are
    cited as the evidence for a design choice already made, not as facts anything reads.

    Per PAYLOAD, not per row: one payload yields one row per question, and every one of
    them carries these same two counts. `build_codec_verdict_report` de-duplicates by
    `sha` before summing — summing the rows directly would multiply a payload's tokens by
    its question count, and again by the number of models that answered it."""
    raw_tok = count_cl100k(raw_text)
    terse_tok = count_cl100k(fluency.compress(obj))
    if raw_tok is None or terse_tok is None:
        return {}
    return {"raw_tokens": raw_tok, "terse_tokens": terse_tok}


def run_codec_fluency(envelopes: list[dict], answerers: dict[str, ToolAnswerer],
                      trials: int = 1,
                      progress: Callable[[str], None] | None = None,
                      preflight: bool = True) -> dict[str, list[dict]]:
    """Run the codec-tier eval for each named tool-capable answerer over every payload in
    the corpus that has at least one `deref` question. Mirrors `dropeval.run_drop_fluency`'s
    envelope-outer/model-inner nesting (question generation is model-independent, so it is
    derived once per envelope, not once per (model, envelope)) and `fluency.run_fluency`'s
    row-tagging convention.

    Each row additionally carries `"tool"` AND `"shape"` (`capture.envelope_shape`, which
    classifies the envelope's `raw` LIVE rather than trusting the bucket stored at capture
    time — a per-`(tool, shape)` verdict filed under a shape the codec no longer assigns
    answers the wrong question, #355). No existing
    harness stamps `shape` onto its rows; it is the one field #295's per-`(tool, shape)`
    verdict needs that the comprehension harness never needed, because the comprehension
    report has always pooled globally rather than per shape bucket.

    Rows also carry `sha` and, when a tokenizer is available, the payload's `raw_tokens` /
    `terse_tokens` (`_payload_tokens`) — PER PAYLOAD, repeated on each of its question rows,
    for `report._codec_savings_section` to de-duplicate by `sha` and report beside the
    verdict."""
    # Pre-flight EVERY model before touching the corpus (#403). A model that cannot express
    # a container tool argument scores 0% on both arms and takes the run's verdict down as
    # `broken control` with no named cause; a few calls per model buy a named refusal in
    # seconds instead of an hour of UNRESOLVED.
    # REFUSE the run, never drop the model and continue. The verdict is the WORST model's,
    # so running without a failed model can only move a cell toward SAFE, and neither the
    # rows nor the report would carry a trace of the model that was asked for. Review of
    # #403 reproduced it: the same corpus rendered UNRESOLVED with the model and a clean
    # SAFE without it. Re-running with a corrected `--models` costs seconds.
    # `preflight=False` exists for tests whose subject is envelope handling and whose stub
    # answerer is deliberately degenerate. The CLI never passes it:
    # `test_the_cli_refuses_a_codec_verdict_run_a_model_cannot_answer` drives `main`.
    if preflight:
        failed = {name: why for name, a in answerers.items()
                  if (why := preflight_encoding(a)) is not None}
        if failed:
            raise PreflightError(
                f"terse fluency --codec-verdict: refused before the sweep — {len(failed)} of "
                f"{len(answerers)} model(s) failed the container-argument pre-flight:\n"
                + "".join(f"  {name}: {why}\n" for name, why in failed.items())
                + "A backend that returned no usable turn may pass on a retry. A model whose "
                "ANSWERS failed cannot take part: remove it from --models, knowing the "
                "verdict then covers only the models you name. That choice is yours to make "
                "and to disclose — the run will not make it silently, because the verdict is "
                "set by the worst model and dropping one can only move it toward SAFE.")
    results: dict[str, list[dict]] = {name: [] for name in answerers}
    progress = fluency.guarded(progress)
    started = time.monotonic()
    for i, env in enumerate(envelopes, 1):
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            obj = None  # deref needs parsed JSON structure; a non-JSON/text payload has none
        if obj is None or not gen_codec_questions(obj) or not codec_changes(obj):
            # One line per skip, like `run_drop_fluency` (#267): `done` reaches `total`
            # without M near-identical lines per skipped envelope.
            if progress is not None:
                progress(f"[fluency --codec-verdict] (skipped) {env.get('tool', '?'):<24} "
                         f"{i}/{len(envelopes)} payload(s)")
            continue
        toks = _payload_tokens(env["raw"], obj)
        # `sha` is OMITTED, never defaulted, when the envelope has no usable one. `tool`
        # defaults because it is a LABEL — a group headed `?` is legible. `shape` carries a
        # label default too, but nothing can reach it: `json.loads(env["raw"])` above already
        # `continue`d on a non-str `raw`, and `envelope_shape` only falls back to the stored
        # value when there is no `raw` string to classify. It is kept as the signature's own
        # documented behaviour rather than as a live branch. `sha` is
        # an IDENTITY: `report._codec_savings_section` de-duplicates payloads by it, so a
        # shared `"?"` placeholder silently collapses every sha-less payload in a group into
        # whichever was seen first, and reports the survivor's tokens as the group's total.
        # That is exactly the defect the renderer's own guard was added to catch, and a
        # placeholder here walks straight past it: `"?"` is a non-empty `str`. Second review
        # of #303 found the renderer hardened and this line still emitting the placeholder.
        # `capture.record` always writes `sha`, but `capture.load_corpus` does not require
        # it, so a foreign or hand-built corpus reaches here without one.
        sha = env.get("sha")
        tags: dict[str, Any] = {"tool": env.get("tool", "?"),
                                "shape": capture.envelope_shape(env, "unknown")}
        if isinstance(sha, str) and sha:
            tags["sha"] = sha
        for name, answerer in answerers.items():
            for row in run_codec_payload(obj, env["raw"], answerer, trials=trials):
                results[name].append({**tags, **toks, **row})
            if progress is not None:
                progress(fluency.progress_line("fluency --codec-verdict", name, i,
                                               len(envelopes), results[name], started))
    return results
