"""Behavioral eval for drop-to-retrieve (#10): does a REAL tool-calling model actually
call `terse.retrieve` when a dropped field matters, and leave it alone when it doesn't?

`fluency.py` answers a different question — does a model read terse's compressed FORM as
accurately as raw JSON — with a single-shot `(system, user) -> reply` answerer. That
protocol can't express a tool call, so it is structurally unable to test drop-to-retrieve:
the only way to find out whether a model reaches for the tool is to actually hand it the
tool and watch. This module adds a second, tool-capable answerer protocol and a 2-turn
loop that mirrors exactly what `proxy.py` does in production — same primer, same tool
definition, same miss-string on an unresolved handle — so a pass here is evidence about
the real deployed behavior, not a proxy for it.

Method (the same honesty bar as fluency.py, principle #24):
  - Ground truth is computed offline from `policy.apply`'s own drop-sink callback — the
    exact mechanism the proxy uses — never guessed or re-derived.
  - Two questions per drop-marked payload: a RECALL question that is answerable only by
    calling retrieve (over-fetch is not scored here — not calling is simply wrong), and a
    PRECISION question answerable entirely from visible data (calling retrieve here is an
    unnecessary round-trip — over-fetch, scored as a miss).
  - The verdict gates on the WORST model across recall, precision, and final-answer
    accuracy (report.py), not the mean — a policy that's unsafe for the worst model in the
    fleet is unsafe, full stop.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from . import capture, fluency
from . import lossy as lossy_mod
from . import policy as policy_mod
from .proxy import TERSE_PRIMER
from .transport import guard_cleartext_credential


# --------------------------------------------------------------------------- #
# Tool-loop answerer protocol — the existing fluency.Answerer (system, user) -> str
# can't express a tool call; this is a provider-neutral running conversation instead.
# `messages` is an OpenAI-style list: {"role": "system"|"user"|"assistant"|"tool", ...},
# with assistant turns carrying `tool_calls` and tool turns carrying `tool_call_id` +
# `content`. The harness (run_drop_payload) owns the loop; the answerer only ever sees
# and returns one turn at a time, so a live backend stays a thin, stateless adapter.
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    text: str                              # final assistant text ("" if it only called a tool)
    tool_calls: list[ToolCall] = field(default_factory=list)  # empty if answered directly
    # True when the turn produced no usable answer: the call never reached the model
    # (transport error, 4xx, rate limit), OR it returned 200 with no content and no tool
    # call (#268 — a token-budget or safety-filter stop, not a comprehension failure).
    # Scored rows carry the count so a run that failed to ASK cannot be read as a model
    # that failed to ANSWER — the two are indistinguishable in the accuracy columns alone.
    error: bool = False


ToolAnswerer = Callable[[list[dict]], Turn]  # messages -> one assistant turn


# --------------------------------------------------------------------------- #
# Question generation — deterministic, ground truth from policy.apply's own drop sink
# --------------------------------------------------------------------------- #
@dataclass
class DropQuestion:
    qid: str
    kind: str  # "recall" | "precision"
    prompt: str
    instruction: str
    expected: Any
    needs_retrieve: bool
    expected_handle: str | None = None
    # Overrides the kind's default fluency.score qtype. Set where the answer's canonical
    # form differs from the kind's usual one — e.g. a recall answer that is a line NUMBER
    # is graded as a count, not a dereferenced value.
    qtype: str | None = None


# Maps a DropQuestion.kind to the fluency.score qtype it should be graded with. Recall
# answers are the full original value (arbitrary JSON) -> "deref" (JSON value-equality,
# prose-tolerant). Precision reuses fluency's "count" question verbatim -> "count".
_QTYPE_FOR_KIND = {"recall": "deref", "precision": "count"}


def _staged_apply(obj: Any, rule: Any, tool: str) -> tuple[policy_mod.Applied, dict[str, Any]]:
    """Run `policy.apply` with a single-rule policy wrapping `rule`, collecting every
    successfully-dropped handle->original-value pair into a fresh staging dict via the
    drop_sink callback. `apply()` only calls the sink for handles that passed the
    droppable-loss gate and were actually committed (see policy.py), so an empty staging
    dict here means nothing was dropped — not a partial/failed drop."""
    raw = json.dumps(obj)
    pol = policy_mod.Policy(rules=[rule])
    staging: dict[str, Any] = {}
    applied = policy_mod.apply(raw, tool, pol, drop_sink=staging.__setitem__)
    return applied, staging


def _staged_apply_text(raw: str, rule: Any, tool: str) -> tuple[policy_mod.Applied, dict[str, Any]]:
    """`_staged_apply` for a NON-JSON payload: `raw` is handed to `policy.apply` verbatim
    rather than re-serialized, since a text payload has no object to dump."""
    pol = policy_mod.Policy(rules=[rule])
    staging: dict[str, Any] = {}
    applied = policy_mod.apply(raw, tool, pol, drop_sink=staging.__setitem__)
    return applied, staging


def _control_rule(rule: Any) -> Any:
    """`rule` with every drop-to-retrieve field spec removed — the CONTROL arm's policy.

    The control has to differ from the treatment in exactly one way: the drop. So it is not
    the raw payload and not `compress()` — it is the *same rule*, same tiers, same tabular
    and dictionary transforms, with only the `lossy: drop-to-retrieve` entries stripped.
    Anything else would confound the drop with a codec difference and reproduce, one level
    over, the defect this arm exists to fix (#269).

    Text selectors are stripped too: `_text_drop_specs` addresses spans of a non-JSON
    payload and is the drop under test on that path, so leaving it in would give the
    control arm the very cut it is supposed to lack."""
    dropped = {p for p, _ in lossy_mod._drop_specs(rule)}
    dropped |= {p for p, _ in lossy_mod._text_drop_specs(rule)}
    return replace(rule, fields={p: s for p, s in rule.fields.items() if p not in dropped})


def _control_text(payload: Any, rule: Any, tool: str, *, is_json: bool) -> str:
    """The text the model would have seen had this rule carried no drop at all.

    `drop_sink` is deliberately omitted: with the drop specs stripped there is nothing to
    stage, and passing a sink would imply otherwise to a reader."""
    pol = policy_mod.Policy(rules=[_control_rule(rule)])
    raw = json.dumps(payload) if is_json else payload
    return policy_mod.apply(raw, tool, pol).text


# A line-numbered source line, the form every `codegraph_explore` block uses: `184\t<code>`.
_GUTTER_RE = re.compile(r"^\s*(\d+)\t")


def _line_recall_question(handle: str, anchor: str, target: str) -> DropQuestion:
    """The recall question for a located line — phrased so its answer has ONE canonical
    written form, which is what makes exact scoring correct rather than merely strict.

    Asking for the line's TEXT does not have that property. Measured across 4 models,
    every one of them found the right line and then wrote it back normalized — gutter
    stripped, indentation collapsed:

        expected  '184\\t\\t\\t\\texports = append(exports, extractExports(noComments)...)'
        answered  '\\t\\t\\texports = append(exports, extractExports(noComments)...)'

    The obvious repair — a whitespace-tolerant comparator — is the wrong one: it weakens
    the scorer for every question in order to fix the phrasing of one, and a comparator
    that ignores whitespace also passes answers that are wrong in whitespace-significant
    content. Ask for a token that is already canonical instead. A line NUMBER is an
    integer, graded by the same `count` qtype the precision question uses, and a line that
    is blank apart from its gutter still has one — so the degenerate-target case
    disappears rather than needing a filter.

    Not every payload is line-numbered, so the fallback asks for the line's first
    whitespace-delimited token: also free of leading indentation, also a single canonical
    string. Weaker (a short token is more guessable), which is why it is the fallback."""
    m = _GUTTER_RE.match(target)
    if m:
        # Quote the anchor's CONTENT, never its own gutter. Quoting the whole line put the
        # anchor's number in the prompt, and the answer is the next number — so 93% of
        # questions were answerable by adding one, with no retrieval at all. Retrieve-recall
        # was unaffected (it counts tool calls, and every model still called), but answer
        # accuracy was measuring arithmetic on a leaked value.
        return DropQuestion(
            qid="drop-text-recall",
            kind="recall",
            prompt=(f"The omitted block with handle {handle!r} is line-numbered, and "
                    f"contains exactly one line whose text after its line-number prefix is "
                    f"{json.dumps(_GUTTER_RE.sub('', anchor))}. What is the line number of "
                    f"the line immediately after it, ignoring blank lines?"),
            instruction="Reply with a single integer and nothing else.",
            expected=int(m.group(1)),
            needs_retrieve=True,
            expected_handle=handle,
            # NOT `count`: the retrieved block is injected into the conversation and carries
            # every one of its line numbers, so `_matches_number`'s present-anywhere rule
            # would pass a reply that merely echoes the block. `sole_number` requires the
            # answer to stand alone, which a block echo cannot satisfy.
            qtype="sole_number",
        )
    return DropQuestion(
        qid="drop-text-recall",
        kind="recall",
        prompt=(f"The omitted block with handle {handle!r} contains exactly one line whose "
                f"text is {json.dumps(anchor)}. What is the first whitespace-delimited "
                f"token of the line immediately after it, ignoring blank lines?"),
        instruction="Reply with that token as a JSON string, and nothing else.",
        expected=target.split()[0],
        needs_retrieve=True,
        expected_handle=handle,
    )


def _text_questions_and_staging(
    raw: str, rule: Any, tool: str
) -> tuple[list[DropQuestion], policy_mod.Applied | None, dict[str, Any] | None]:
    """The `_questions_and_staging` analogue for a span-addressed text payload.

    The stakes differ from the JSON case and the questions are built to match. A dropped
    JSON field is one value among many; a dropped fenced code block is a chunk of source
    the surrounding prose may explicitly tell the model it has "already read" — so the
    recall question asks for an EXACT line of a dropped block, located by a unique anchor
    line rather than by ordinal (unanswerable without retrieving, and un-guessable — see
    the anchor comment below for why the ordinal form was wrong), while the precision
    question totals the visible markers'
    `bytes` fields (readable off the emitted text without ever needing the dropped content,
    so any retrieve call here is a pure over-fetch). Both come from `apply()`'s own sink.
    """
    if not lossy_mod._text_drop_specs(rule):
        return [], None, None  # no text selector on this rule -> nothing to test

    applied, staging = _staged_apply_text(raw, rule, tool)
    if applied.text == raw or not staging:
        return [], None, None  # every span was under the size floor, or the gate failed

    markers = lossy_mod._TEXT_MARKER_RE.findall(applied.text)
    if not markers:
        return [], None, None

    # Pick the LARGEST dropped span: the one whose absence a model is most likely to try
    # to paper over from context instead of retrieving — the hardest honest case.
    handle = max(markers, key=lambda h: len(staging.get(h, "")))
    span = staging.get(handle)
    if not isinstance(span, str):
        return [], None, None
    # The target line is located by an ANCHOR — a line unique within the span — and not by
    # its ordinal. Asking for "non-blank line number 81 of 160" measured the wrong skill:
    # across 4 models and 49 answered questions, EVERY model retrieved the right block with
    # the right handle (100% recall, 100% handle) and then returned some real line of that
    # block at the wrong ordinal — 0% "accuracy" that was a counting failure, not a
    # comprehension one. Since final-accuracy gates the verdict, the old form made the
    # text drop-eval structurally unpassable and so could never authorize a policy.
    # Anchoring keeps every property that matters — the answer is un-guessable, is not
    # present in the retained prose, and is only obtainable by actually reading the
    # retrieved span — while dropping the arithmetic.
    # "Blank" must mean blank TO THE READER. A line-numbered source line that is empty
    # after its gutter (`81\t`) is non-blank to `str.strip` — it still has "81" in it — but
    # every model reads it as the blank line it renders as, and skips it. Every remaining
    # recall failure was that exact off-by-one, agreed on unanimously by all models against
    # the ground truth: the truth was wrong, not the answers. Strip the gutter before
    # deciding, and gutter-only lines stop being eligible as anchor or target as well.
    lines = [ln for ln in span.splitlines() if _GUTTER_RE.sub("", ln).strip()]
    if len(lines) < 3:
        return [], None, None  # too small to pose a non-trivial recall question
    # Scan from the midpoint (fences and the first source line are the parts most likely to
    # be echoed by the retained prose) for the first line that occurs exactly once and is
    # followed by a different line — both properties are needed for the answer to be
    # unambiguous. Deterministic: same span in, same question out.
    # Uniqueness is judged on the gutter-STRIPPED line, because that is what the prompt
    # quotes (the gutter is withheld so the answer can't be derived by adding one). Judging
    # the numbered line instead made every line trivially unique — the number guarantees it
    # — while the locator the model actually sees could match many lines, so "contains
    # exactly one line whose text is X" was simply false and the question unanswerable.
    bare = [_GUTTER_RE.sub("", ln) for ln in lines]
    counts = Counter(bare)
    order = sorted(range(len(lines) - 1), key=lambda i: (abs(i - len(lines) // 2), i))
    pair = next(((lines[i], lines[i + 1]) for i in order
                 if counts[bare[i]] == 1 and lines[i + 1] != lines[i]), None)
    if pair is None:
        return [], None, None  # no unambiguous anchor (e.g. every line identical)
    anchor, target = pair
    recall_q = _line_recall_question(handle, anchor, target)
    # Anchored on the SUM of the visible markers' `bytes` fields, not the marker count:
    # a count is 1 on the common single-drop payload and so is answerable by guessing,
    # which inflated answer accuracy while proving nothing. A byte total has to be read
    # off the emitted text, yet still never requires the dropped content itself — so it
    # stays a clean no-over-fetch probe, the way the JSON path's record-count anchor is.
    total_bytes = sum(len(staging[h]) for h in dict.fromkeys(markers) if h in staging)
    precision_q = DropQuestion(
        qid="drop-text-precision",
        kind="precision",
        prompt=("Summing the \"bytes\" field of every omitted-block marker shown in this "
                "payload, what is the total?"),
        instruction="Reply with a single integer and nothing else.",
        expected=total_bytes,
        needs_retrieve=False,
        expected_handle=None,
    )
    return [recall_q, precision_q], applied, staging


def gen_text_drop_questions(raw: str, rule: Any, tool: str) -> list[DropQuestion]:
    """One recall + one precision question for a text payload whose rule actually drops
    spans, else [] (nothing to test — same fail-closed honesty bar as the JSON path)."""
    return _text_questions_and_staging(raw, rule, tool)[0]


def run_drop_text_payload(raw: str, rule: Any, tool: str, answerer: ToolAnswerer,
                          trials: int = 1) -> list[dict]:
    """`run_drop_payload` for a non-JSON payload. [] when nothing was dropped."""
    questions, applied, staging = _text_questions_and_staging(raw, rule, tool)
    if not questions:
        return []
    assert applied is not None and staging is not None
    return _run_questions_against(questions, applied, staging, answerer, trials=trials)


def _questions_and_staging(
    obj: Any, rule: Any, tool: str
) -> tuple[list[DropQuestion], policy_mod.Applied | None, dict[str, Any] | None]:
    """Shared core of `gen_drop_questions`: generates the (recall, precision) question
    pair AND returns the `(applied, staging)` that `_staged_apply` computed along the
    way, so `run_drop_payload` can reuse them instead of a second `policy.apply()` pass
    over the same payload. `applied`/`staging` are only meaningful when the question
    list is non-empty — every early-exit path returns `None` for both instead of
    fabricating a value the caller has no use for anyway."""
    if not lossy_mod._drop_specs(rule):
        return [], None, None  # nothing marked drop-to-retrieve on this rule -> nothing to test

    records, list_path = capture.find_record_list_with_path(obj)
    if records is None or list_path is None:
        return [], None, None  # not record-shaped (or no simple field path) -> terse wouldn't drop here

    applied, staging = _staged_apply(obj, rule, tool)
    if applied.skipped or not staging:
        return [], None, None  # every candidate field was under the size floor, or the gate failed

    # Intersection, not `records[0].keys()`: #204 widened `find_record_list_with_path` to
    # whatever the tabularizer folds, so a record may lack a key the first one has, and the
    # pickers index every record by these columns.
    cols = fluency._intersection_cols(records)
    idcol = fluency._pick_id_col(records, cols)
    if idcol is None:
        return [], None, None  # can't address a specific record without a unique scalar id column

    # Find the (record, field) whose handle actually landed in `staging` — content-
    # addressed handles are deterministic (sha1 of tool+path+serialized value, no RNG),
    # so recomputing here reproduces exactly what apply() committed.
    prefix = f"{list_path}."
    hit: tuple[int, str, Any, str] | None = None
    for path, spec in lossy_mod._drop_specs(rule):
        if not path.startswith(prefix):
            continue
        field_name = path[len(prefix):]
        if "[]" in field_name or "." in field_name:
            continue  # nested-below-record paths are out of scope for v1
        min_len = int(spec.get("min", lossy_mod.DEFAULT_DROP_MIN))
        for i, rec in enumerate(records):
            if field_name not in rec:
                continue
            value = rec[field_name]
            serialized = lossy_mod._serialize(value)
            if len(serialized) < min_len:
                continue  # left in place by the size floor -> never got a marker
            handle = lossy_mod._handle(tool, path, serialized)
            if handle in staging and staging[handle] == value:
                hit = (i, field_name, value, handle)
                break
        if hit is not None:
            break
    if hit is None:
        return [], None, None
    ri, field_name, value, handle = hit

    recall_q = DropQuestion(
        qid="drop-recall",
        kind="recall",
        prompt=(f"For the record whose {idcol!r} is "
                f"{json.dumps(records[ri][idcol], ensure_ascii=False)}, what is the full "
                f"value of {field_name!r}?"),
        instruction="Reply with the value as compact JSON, and nothing else.",
        expected=value,
        needs_retrieve=True,
        expected_handle=handle,
    )

    # Precision anchor: the "count" question always exists for a non-empty record list
    # and never depends on any single field's content, so it can never accidentally need
    # the dropped value — a robust, deterministic no-overfetch probe.
    count_q = next((q for q in fluency.gen_questions(obj) if q.qtype == "count"), None)
    if count_q is None:
        return [], None, None
    precision_q = DropQuestion(
        qid="drop-precision",
        kind="precision",
        prompt=count_q.prompt,
        instruction=count_q.instruction,
        expected=count_q.expected,
        needs_retrieve=False,
        expected_handle=None,
    )
    return [recall_q, precision_q], applied, staging


def gen_drop_questions(obj: Any, rule: Any, tool: str) -> list[DropQuestion]:
    """Generate one recall + one precision question for a record-shaped payload that
    actually has a drop-marked field, else [] (nothing to test — fail closed rather than
    fabricate an un-answerable question, mirroring the rest of this project's honesty
    bar). Only a direct scalar field on the record list (e.g. `result[].body`) is
    supported in v1 — matches the drop path shapes exercised in test_proxy.py/#10.
    """
    questions, _applied, _staging = _questions_and_staging(obj, rule, tool)
    return questions


# --------------------------------------------------------------------------- #
# The 2-turn tool-loop driver — mirrors the real proxy's retrieve protocol exactly
# --------------------------------------------------------------------------- #
def _miss_text(handle: Any) -> str:
    """The exact miss string `proxy.Interceptor.answer_retrieve` emits for an unresolved
    handle, copied verbatim so this eval's miss-handling matches production behavior — a
    model that has learned to recover from a real miss must see the same words here."""
    return (f"terse: dropped-field handle {handle!r} is no "
            "longer available (evicted, or the session "
            "reconnected). Re-run the original tool to get "
            "the value again.")


def _safe_call(answerer: ToolAnswerer, messages: list[dict]) -> Turn:
    """Call the model, but never let one failed call abort a long multi-model run — a
    transport error / rate limit / refusal scores as "didn't answer, didn't retrieve",
    not a crash. Mirrors fluency._safe_ask's fail-open contract — including, since #263,
    its insistence that a failure stay DISTINGUISHABLE from a real answer: `_safe_ask`
    returns None where this returns `error=True`.

    The failure is RECORDED, not just absorbed: an unreachable model produces exactly the
    same row as a model that declined to call retrieve (0% recall), so a harness fault
    otherwise renders as a confident behavioral FAIL. That is not hypothetical — every
    drop-eval run against an OpenAI-compatible endpoint 400'd on the tool NAME
    (`terse.retrieve` violates OpenAI's `^[a-zA-Z0-9_-]+$`; see `_oai_name`) and reported a
    clean 0%-recall verdict rather than an error."""
    try:
        return answerer(messages)
    except Exception as exc:
        # Say WHY on stderr. Swallowing the exception silently made a degraded run
        # uninterpretable: a real 3-model run lost 12/48 calls on one backend and the
        # output carried no way to tell a token-budget stop from a rate limit from an
        # billing failure — so the cause got guessed instead of read. `fluency`'s
        # answerer already prints its `finish_reason` for exactly this reason; this path
        # is the one that did not.
        print(f"terse dropeval: call failed ({type(exc).__name__}: {exc!s:.200}) — counted "
              f"as a non-answer, not scored", file=sys.stderr)
        return Turn(text="", tool_calls=[], error=True)


def _assistant_tool_call_message(turn: Turn, calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": turn.text or "",
        "tool_calls": [
            {"id": c.call_id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in calls
        ],
    }


def _run_question(question: DropQuestion, applied_text: str, staging: dict[str, Any],
                  answerer: ToolAnswerer) -> tuple[bool, bool, bool, bool]:
    """Run ONE trial of the 2-turn retrieve protocol for `question`. Returns
    (retrieve_ok, answer_ok, handle_ok, errored) for that single trial."""
    messages: list[dict] = [
        {"role": "system", "content": TERSE_PRIMER},
        {"role": "user", "content": fluency._user_prompt(question.prompt, question.instruction,
                                                          applied_text)},
    ]
    turn = _safe_call(answerer, messages)
    retrieve_calls = [c for c in turn.tool_calls if c.name == lossy_mod.RETRIEVE_TOOL]
    retrieved = bool(retrieve_calls)

    errored = turn.error
    if retrieved:
        messages.append(_assistant_tool_call_message(turn, retrieve_calls))
        hit_expected_handle = False
        for c in retrieve_calls:
            call_handle = c.arguments.get("handle")
            if call_handle in staging:
                content = lossy_mod._serialize(staging[call_handle])
                if call_handle == question.expected_handle:
                    hit_expected_handle = True
            else:
                content = _miss_text(call_handle)
            messages.append({"role": "tool", "tool_call_id": c.call_id, "content": content})
        final = _safe_call(answerer, messages)
        errored = errored or final.error
        final_text = final.text
    else:
        hit_expected_handle = False
        final_text = turn.text

    qtype = question.qtype or _QTYPE_FOR_KIND[question.kind]
    answer_ok = fluency.score(qtype, question.expected, final_text)
    retrieve_ok = retrieved == question.needs_retrieve
    # A model that never called retrieve trivially "used the right handle" (nothing to
    # check) — handle_ok only penalizes a WRONG handle, not a missing call (that miss is
    # already captured by retrieve_ok).
    handle_ok = (not retrieved) or hit_expected_handle
    return retrieve_ok, answer_ok, handle_ok, errored


def _run_questions_against(questions: list[DropQuestion], applied: policy_mod.Applied,
                           staging: dict[str, Any], answerer: ToolAnswerer,
                           trials: int = 1, control_text: str | None = None) -> list[dict]:
    """Run `trials` trials of the real 2-turn retrieve protocol for each of `questions`
    against one `answerer`, over an already-staged `(applied, staging)` pair. Split out
    of `run_drop_payload` so `run_drop_fluency` can compute `_questions_and_staging`
    ONCE per envelope and reuse it across every configured model, instead of
    re-deriving it (a JSON parse + a `policy.apply()` pass) once per model.

    `control_text` (#269) is the same payload with the drop rule stripped. When given,
    every question is ALSO asked against it and scored into `control_ok`, so
    final-accuracy becomes a gap between two measured arms instead of a gap against an
    assumed-perfect ideal that was never run. This matters because the recall answer is
    JSON value-equality against a 500+ character prose field: a model handed the
    UN-dropped payload does not reproduce that verbatim either — it paraphrases — so the
    fixed-100% control charged the drop for a verbatim-reproduction limit that has
    nothing to do with dropping. It doubles the call count, which is why it is opt-in at
    the call site rather than unconditional.

    The control reuses `_run_question` with an EMPTY staging rather than a bespoke
    single-turn path: identical prompt assembly, identical primer, identical scoring, and
    a control that spuriously calls retrieve still gets a coherent miss-response and a
    final answer. The only difference between the arms is the payload text, which is the
    whole point of a control."""
    rows: list[dict] = []
    for q in questions:
        retrieve_ok = answer_ok = handle_ok = errors = 0
        control_ok = control_errors = 0
        for _ in range(trials):
            r_ok, a_ok, h_ok, err = _run_question(q, applied.text, staging, answerer)
            errors += int(err)
            # An errored trial contributes to NO success counter of its own arm. It is not
            # merely "not correct": `_run_question` still computes `retrieve_ok` as
            # `retrieved == needs_retrieve`, so a precision question whose call never
            # reached the model scores a free +1 for not retrieving. Counting that while
            # also removing the trial from `retrieve_trials` makes successes EXCEED trials,
            # which drives `_form_stats`'s `p̂(1-p̂)` negative and crashes the whole report
            # on `math.sqrt`. Found by a live run, not by the unit tests — hence the
            # k<=t invariant test that now guards it.
            if not err:
                retrieve_ok += int(r_ok)
                answer_ok += int(a_ok)
                handle_ok += int(h_ok)
            # The control runs on EVERY trial, independently of whether the treatment
            # errored. Skipping it after a treatment failure would save a call but leave
            # `control_trials` counting attempts the control never made — the same
            # successes-exceed-trials class of bug, one arm over. The arms are independent
            # measurements; only `paired_rows` relates them, and it does so afterwards.
            if control_text is not None:
                _, c_ok, _, c_err = _run_question(q, control_text, {}, answerer)
                control_errors += int(c_err)
                if not c_err:
                    control_ok += int(c_ok)
        row = {
            "qid": q.qid, "kind": q.kind, "trials": trials,
            "retrieve_ok": retrieve_ok, "answer_ok": answer_ok, "handle_ok": handle_ok,
            # NO per-form denominator for the TREATMENT arm — deliberately, and note the
            # asymmetry with `control_trials` below. An earlier revision of this change
            # added `retrieve_trials`/`answer_trials`/`handle_trials` = `trials - errors`,
            # which `_form_stats` prefers over the shared `trials`, and so removed errored
            # trials from the recall/precision/handle denominators. That reverses the
            # decision `openai_tool_answerer` documents ("DELIBERATELY NOT excluded from
            # the accuracy denominators ... excluding would be the dangerous direction"),
            # and it is not theoretical: measured on identical model behaviour, a recall
            # column of 33% (FAIL) became 100% (PASS, "safe to enable drop-to-retrieve") at
            # an 11% error rate — far under the INCONCLUSIVE gate. `Turn.error` covers a
            # `no_content` / token-budget stop, which is prompt-length correlated, and the
            # treatment arm carries the longest prompt here, so the losses are exactly the
            # arm-correlated kind. final-accuracy is protected by `paired_rows`;
            # recall/precision/handle are not.
            #
            # The direction is what decides it. gap = treatment - control:
            #   treatment errors scored as MISSES  -> treatment lower -> gap more negative
            #                                         -> the drop looks worse. Conservative.
            #   control errors scored as MISSES    -> control lower  -> gap less negative
            #                                         -> the drop looks BETTER. Dangerous.
            # So the treatment keeps every trial in its denominator, and only the control
            # gets one that excludes its own failures.
            # Total failed calls across BOTH arms, with the matching attempt count, so the
            # INCONCLUSIVE gate reads a true ratio rather than dividing two-arm failures by
            # one arm's trials.
            "errors": errors + control_errors,
            # ...and the SPLIT, because the total alone cannot answer the question that
            # matters. #299's hazard is that attrition is ARM-CORRELATED — the treatment
            # runs two turns to the control's one, so it should fail first under a
            # token-budget stop. A collapsed count makes that untestable, and a real run
            # then had its 12/48 failures ATTRIBUTED to that mechanism with no evidence
            # either way. These two fields are what make the claim checkable.
            "treatment_errors": errors,
            "control_errors": control_errors,
            "attempts": trials * (2 if control_text is not None else 1),
        }
        if control_text is not None:
            row["control_ok"] = control_ok
            row["control_trials"] = trials - control_errors
        rows.append(row)
    return rows


def run_drop_payload(obj: Any, raw: str, rule: Any, tool: str, answerer: ToolAnswerer,
                     trials: int = 1, control: bool = False) -> list[dict]:
    """Ask each of a payload's drop questions `trials` times over the real 2-turn
    protocol. Returns one row per question carrying per-metric success COUNTS (0..trials)
    plus `trials` — the same convention fluency.py uses so report.py's `_form_stats`
    works unchanged. [] if the payload has no drop-marked field (nothing to test).
    """
    # `raw` is accepted for interface symmetry with fluency's run_payload/run_diff_payload
    # (and so a future caller could pass the originally-captured text); the compressed-
    # with-markers text and the drop store must come from the SAME apply() call the
    # questions were derived from, so this reuses _questions_and_staging's own
    # (applied, staging) rather than recomputing them with a second policy.apply() pass,
    # and rather than trusting a possibly-stale `raw`.
    questions, applied, staging = _questions_and_staging(obj, rule, tool)
    if not questions:
        return []
    assert applied is not None and staging is not None  # guaranteed when questions is non-empty
    ctl = _control_text(obj, rule, tool, is_json=True) if control else None
    return _run_questions_against(questions, applied, staging, answerer, trials=trials,
                                  control_text=ctl)


def run_drop_fluency(envelopes: list[dict], rule_for: Callable[..., Any],
                     answerers: dict[str, ToolAnswerer], trials: int = 1,
                     control: bool = False) -> dict:
    """Run the drop-eval for each named tool-capable answerer over every record-shaped,
    drop-marked payload in the corpus. Mirrors `fluency.run_diff_fluency`'s shape.
    Returns {model_name: [scored_row, ...]}; a payload/tool with no drop-marked field
    contributes no rows (gen_drop_questions returns [] for it).

    Loop nesting is envelope-outer, model-inner (not the reverse): the JSON parse and
    `_questions_and_staging` derivation for a payload are the SAME regardless of which
    model answers it, so doing that work per-envelope instead of per-(model, envelope)
    avoids M times the redundant parsing/policy.apply() work for M configured models."""
    results: dict[str, list[dict]] = {name: [] for name in answerers}
    for env in envelopes:
        tool = env["tool"]
        # Look the rule up the way the PROXY does — bare tool plus the recorded server. A
        # policy generated from a server-tagged corpus carries qualified rule names, so a
        # bare-name lookup falls through to the defaults, finds no `fields`, and the whole
        # eval silently scores nothing while still printing that it verified the drops
        # (the #149 failure mode, one lookup removed).
        rule = rule_for(tool, env.get("server"))
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            # Not JSON: the span-addressed text path is the only one that can drop here.
            # Same envelope-outer/model-inner nesting and the same scored-row shape, so a
            # text payload's results merge into the report exactly like a JSON one's.
            text_raw = env.get("raw") or ""
            questions, applied, staging = _text_questions_and_staging(text_raw, rule, tool)
            if not questions:
                continue
            assert applied is not None and staging is not None
            ctl = _control_text(text_raw, rule, tool, is_json=False) if control else None
            for name, fn in answerers.items():
                for row in _run_questions_against(questions, applied, staging, fn,
                                                  trials=trials, control_text=ctl):
                    results[name].append({"tool": tool, "sha": env.get("sha", "?"), **row})
            continue
        questions, applied, staging = _questions_and_staging(obj, rule, tool)
        if not questions:
            continue
        assert applied is not None and staging is not None
        # Computed once per envelope, like the questions themselves — the control text is
        # the same regardless of which model answers it.
        ctl = _control_text(obj, rule, tool, is_json=True) if control else None
        for name, fn in answerers.items():
            for row in _run_questions_against(questions, applied, staging, fn, trials=trials,
                                              control_text=ctl):
                results[name].append({"tool": tool, "sha": env.get("sha", "?"), **row})
    return results


# --------------------------------------------------------------------------- #
# Tool-capable live backend — zero new dependencies (mirrors fluency.openai_answerer's
# urllib pattern, just carrying a `tools` param + parsing tool_calls).
# --------------------------------------------------------------------------- #
# OpenAI's function-calling schema constrains a function name to ^[a-zA-Z0-9_-]+$. MCP tool
# names carry no such restriction, and terse's own retrieve tool is `terse.retrieve` — the
# dot is REJECTED with a 400 by every OpenAI-compatible endpoint. Since the whole request
# fails, the model never sees the question, `_safe_call` absorbs the error, and the eval
# reports 0% retrieve-recall: a harness fault that reads exactly like a model that refuses
# to call the tool. The name is therefore rewritten on the way out and mapped BACK on the
# way in, so scoring still matches against the real MCP name and the proxy's wire protocol
# (where `terse.retrieve` is valid and unchanged) is untouched.
_OAI_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _oai_name(mcp_name: str) -> str:
    """An MCP tool name rewritten to OpenAI's permitted function-name alphabet."""
    return _OAI_NAME_RE.sub("_", mcp_name)


def _to_openai_tool(tool_def: dict) -> dict:
    """RETRIEVE_TOOL_DEF's MCP `inputSchema` shape -> OpenAI function-calling `parameters`."""
    return {
        "type": "function",
        "function": {
            "name": _oai_name(tool_def["name"]),
            "description": tool_def.get("description", ""),
            "parameters": tool_def.get("inputSchema", {}),
        },
    }


def openai_tool_answerer(base_url: str, api_key: str, model: str, tools: list[dict],
                         temperature: float = 0.0, timeout: int = 60) -> ToolAnswerer:
    """OpenAI-compatible /chat/completions answerer, tool-calling variant, over stdlib
    urllib (no SDK dependency, matching fluency.openai_answerer). `tools` (RETRIEVE_TOOL_DEF
    shape) is bound at construction time — every call to the returned answerer offers the
    same tool set, as a real client would."""
    # Same cleartext-credential refusal fluency.openai_answerer makes. This constructor
    # sends the identical `Authorization: Bearer <key>` and had NO such guard, so a
    # `--base-url http://remote/v1` put the key on the wire in the clear.
    guard_cleartext_credential(base_url, bool(api_key), what="terse drop-eval")
    url = base_url.rstrip("/") + "/chat/completions"
    oai_tools = [_to_openai_tool(t) for t in tools]
    # Reverse of the `_oai_name` rewrite, so a returned tool_call is scored against the MCP
    # name the questions and `_run_question` are written in terms of.
    mcp_name = {_oai_name(t["name"]): t["name"] for t in tools}
    # A silent last-wins collapse here would reintroduce the exact bug this module fixes:
    # two MCP names that sanitize to one wire name would map a returned call back to the
    # wrong one, and `_run_question`'s `c.name == RETRIEVE_TOOL` filter would score a real
    # retrieval as "didn't retrieve". Single-tool today, but fail loud if that ever changes.
    if len(mcp_name) != len(tools):
        raise ValueError(f"tool names collide after OpenAI-name sanitization: "
                         f"{sorted(t['name'] for t in tools)}")

    def ask(messages: list[dict]) -> Turn:
        body = json.dumps({
            "model": model, "messages": messages, "temperature": temperature,
            "tools": oai_tools, "tool_choice": "auto",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "choices" not in data:
            raise RuntimeError(f"{model}: no choices in response: {data.get('error', data)}")
        msg = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}  # a malformed tool-call payload scores as "called with no args"
            name = fn.get("name", "")
            calls.append(ToolCall(call_id=tc.get("id", ""), name=mcp_name.get(name, name),
                                  arguments=arguments))
        # `content or ""` collapsed "the model produced NO content" into "it answered,
        # with nothing" — the same erasure #268 fixes in `fluency.answerers`. Here it left
        # `error` False, so a 200 carrying `content: null` was indistinguishable from a
        # model that read the payload and answered wrongly; #269 already has to see past a
        # missing control arm, and an unrecorded confound underneath makes that table
        # unreadable.
        #
        # DELIBERATELY NOT excluded from the accuracy denominators, unlike the fluency
        # side. `_run_question` still scores this turn as a miss, and that is the
        # conservative direction HERE: a non-answer counted as a miss makes the drop rule
        # look WORSE, so it can only under-sell enabling a lossy tier, never over-sell it.
        # Excluding would be the dangerous direction — see `report._unmeasured`, where
        # per-arm exclusion of prompt-length-correlated failures turned a real FAIL into a
        # PASS. `errors` carries the count, and the >=50% inconclusive gate catches the
        # gross case.
        #
        # Empty text WITH tool calls is normal and must not be flagged — that is a turn
        # that called `terse.retrieve` instead of answering, which `Turn.text`'s own
        # comment documents. Only "no text AND no calls" is a non-answer.
        text = msg.get("content")
        no_content = (text is None or not text.strip()) and not calls
        if no_content:
            # finish_reason is the actionable half — `length` means raise max_tokens,
            # `content_filter` means the payload tripped a filter. Neither is "unreachable",
            # which is the other thing `error=True` means.
            print(f"terse dropeval: {model} returned no content and called no tool "
                  f"(finish_reason={data['choices'][0].get('finish_reason')!r}) — "
                  f"counted in `errors`; the accuracy columns still score it as a miss",
                  file=sys.stderr)
        return Turn(text=text or "", tool_calls=calls, error=no_content)

    return ask
