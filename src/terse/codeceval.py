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
import time
from collections.abc import Callable
from typing import Any

from . import capture, fluency
from .dropeval import ToolAnswerer, Turn, _safe_call
from .tokenize import count_cl100k

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
    diagnosis only: `preflight_encoding` refuses such a model up front, and the per-row
    counts let a report say which arm was lost to encoding rather than to comprehension.

    `expected` that is itself a string is excluded: `json.loads('"a"') == "a"` would make
    every correctly-answered string question look like a mismatch. `deref`'s `expected` is
    always a dict or list, so this is defence in depth rather than a live case."""
    if not isinstance(got, str) or isinstance(expected, (str, int, float, bool)):
        return False
    try:
        return bool(json.loads(got) == expected)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


# A deref question whose answer is unambiguous from a 2-record payload, used only to ask a
# model "how do you encode a container argument?" before spending an hour finding out. Kept
# here rather than generated from the corpus so the pre-flight cannot itself fail for
# corpus reasons (a thin corpus is a SEPARATE defect; conflating them is what made the
# 2026-09-09 run unreadable).
PREFLIGHT_PAYLOAD = {"result": [{"id": "a", "tags": ["x", "y"]}, {"id": "b", "tags": ["z"]}]}
PREFLIGHT_PROMPT = "For the record whose 'id' is 'a', what is the full value of 'tags'?"
PREFLIGHT_EXPECTED = ["x", "y"]


PREFLIGHT_ATTEMPTS = 3


def preflight_encoding(answerer: ToolAnswerer, attempts: int = PREFLIGHT_ATTEMPTS) -> str | None:
    """`None` if this model can participate in the codec eval, else a one-line reason.

    The 2026-09-09 run burned 57 minutes and 324 calls to discover that one of its two
    models could not express a container argument. That is a property of the model and the
    tool protocol, knowable in ONE call, and independent of the corpus, the codec and the
    question set — so it is checked once per model before the sweep, and the model is
    dropped with its reason named rather than silently scoring 0%.

    Uses the same `_ask` path and the same `RECORD_VALUE_TOOL_DEF` as a real trial, so a
    model that passes here is exercising the identical protocol it will be scored on. A
    pre-flight that constructed its own request could pass while the real path fails.

    Sampled `attempts` times, and **every** attempt must match. Measured while building
    this: `deepseek-v4-pro` at `temperature=0.0` answered one call with
    `{"value": "[\\"x\\", \\"y\\"]"}` and another with no `value` at all, so a
    single call diagnoses the same model differently run to run. Requiring all attempts is
    the right bar rather than a strict one — a model that expresses a container argument
    only sometimes still poisons the arm it is on, and the reported reason is the FIRST
    failure seen, with a count, so a flaky model reads as flaky instead of as whichever
    mode happened to land first."""
    q = fluency.Question(qid="preflight", qtype="deref", transform="none",
                         prompt=PREFLIGHT_PROMPT, instruction="", expected=PREFLIGHT_EXPECTED)
    messages = [{"role": "user", "content": fluency._user_prompt(
        q.prompt, _codec_instruction(), json.dumps(PREFLIGHT_PAYLOAD))}]
    first_reason: str | None = None
    bad = 0
    for _ in range(max(1, attempts)):
        turn: Turn = _safe_call(answerer, messages)
        reason: str | None
        if turn.error:
            reason = f"pre-flight call failed ({turn.error})"
        else:
            calls = [c for c in turn.tool_calls if c.name == RECORD_VALUE_TOOL]
            if not calls:
                reason = (f"answered without calling {RECORD_VALUE_TOOL} "
                          f"(text: {(turn.text or '')[:60]!r}) — this eval scores a "
                          f"tool-call argument")
            else:
                got = calls[-1].arguments.get("value")
                if _value_matches(got, PREFLIGHT_EXPECTED):
                    reason = None
                elif encodes_as_json_string(got, PREFLIGHT_EXPECTED):
                    reason = ("sends container arguments as a JSON STRING "
                              f"({got!r} for {PREFLIGHT_EXPECTED!r}) — right content, wrong "
                              "type. It would encode both arms the same way and score 0% on "
                              "each, so the run would learn nothing about the codec")
                else:
                    reason = (f"failed a 2-record pre-flight question: got {got!r}, "
                              f"expected {PREFLIGHT_EXPECTED!r}")
        if reason is not None:
            bad += 1
            if first_reason is None:
                first_reason = reason
    if first_reason is None:
        return None
    return f"{first_reason} [{bad}/{max(1, attempts)} pre-flight attempts failed]"


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
    emit = fluency.guarded(progress)
    # Pre-flight EVERY model before touching the corpus (#403). A model that cannot express
    # a container tool argument scores 0% on both arms and takes the whole run's verdict
    # down with it as `broken control`; one call per model buys the difference between a
    # named refusal in seconds and an hour of UNRESOLVED. Dropped, never scored: a row from
    # such a model is not evidence about the codec in either direction.
    # `preflight=False` exists for tests whose subject is envelope handling (row stamping,
    # skip accounting) and whose stub answerer is deliberately degenerate — a stub that
    # "never calls the tool" is precisely what the pre-flight is built to reject, so those
    # fixtures would otherwise have to fake a capability they are not testing. NEVER pass
    # it from the CLI: `test_cli_codec_verdict_never_disables_the_preflight` pins that.
    usable: dict[str, ToolAnswerer] = dict(answerers) if not preflight else {}
    for name, a in (answerers.items() if preflight else ()):
        why = preflight_encoding(a)
        if why is None:
            usable[name] = a
        else:
            if emit:
                emit(f"[codec-verdict] EXCLUDED {name}: {why}")
    if preflight and answerers and not usable:
        # Every model failed. Returning empty rows would render as UNRESOLVED "no data",
        # which is indistinguishable from a thin corpus and is exactly the ambiguity #403
        # was filed about — so say which failure this was.
        raise RuntimeError(
            "terse fluency --codec-verdict: no model can express a container tool argument, "
            "so no run is possible. See the EXCLUDED line(s) above for each model's reason.")
    answerers = usable
    results: dict[str, list[dict]] = {name: [] for name in answerers}
    progress = emit
    started = time.monotonic()
    for i, env in enumerate(envelopes, 1):
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            obj = None  # deref needs parsed JSON structure; a non-JSON/text payload has none
        if obj is None or not gen_codec_questions(obj):
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
