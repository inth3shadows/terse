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
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import capture, fluency
from . import lossy as lossy_mod
from . import policy as policy_mod
from .proxy import TERSE_PRIMER


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


def _text_questions_and_staging(
    raw: str, rule: Any, tool: str
) -> tuple[list[DropQuestion], policy_mod.Applied | None, dict[str, Any] | None]:
    """The `_questions_and_staging` analogue for a span-addressed text payload.

    The stakes differ from the JSON case and the questions are built to match. A dropped
    JSON field is one value among many; a dropped fenced code block is a chunk of source
    the surrounding prose may explicitly tell the model it has "already read" — so the
    recall question asks for an EXACT line of a dropped block (unanswerable without
    retrieving, and un-guessable), while the precision question asks how many blocks were
    omitted (answerable by counting visible markers, so any retrieve call here is a pure
    over-fetch). Both are computed from `apply()`'s own sink, never guessed.
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
    # A line from the middle of the block: the fence lines and the first line of source
    # are the parts most likely to be echoed elsewhere in the retained prose (a `####
    # path` heading, an info string), so the midpoint is the least guessable choice.
    lines = [ln for ln in span.splitlines() if ln.strip()]
    if len(lines) < 3:
        return [], None, None  # too small to pose a non-trivial recall question
    target_i = len(lines) // 2
    target = lines[target_i]

    recall_q = DropQuestion(
        qid="drop-text-recall",
        kind="recall",
        prompt=(f"The omitted block with handle {handle!r} contains "
                f"{len(lines)} non-blank lines. What is the exact text of its "
                f"non-blank line number {target_i + 1} (1-indexed)?"),
        instruction="Reply with that line's exact text as a JSON string, and nothing else.",
        expected=target,
        needs_retrieve=True,
        expected_handle=handle,
    )
    precision_q = DropQuestion(
        qid="drop-text-precision",
        kind="precision",
        prompt="How many blocks were omitted from this payload?",
        instruction="Reply with a single integer and nothing else.",
        expected=len(markers),
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

    cols = list(records[0].keys())
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
    not a crash. Mirrors fluency._safe_ask's fail-open contract."""
    try:
        return answerer(messages)
    except Exception:
        return Turn(text="", tool_calls=[])


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
                  answerer: ToolAnswerer) -> tuple[bool, bool, bool]:
    """Run ONE trial of the 2-turn retrieve protocol for `question`. Returns
    (retrieve_ok, answer_ok, handle_ok) for that single trial."""
    messages: list[dict] = [
        {"role": "system", "content": TERSE_PRIMER},
        {"role": "user", "content": fluency._user_prompt(question.prompt, question.instruction,
                                                          applied_text)},
    ]
    turn = _safe_call(answerer, messages)
    retrieve_calls = [c for c in turn.tool_calls if c.name == lossy_mod.RETRIEVE_TOOL]
    retrieved = bool(retrieve_calls)

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
        final_text = final.text
    else:
        hit_expected_handle = False
        final_text = turn.text

    qtype = _QTYPE_FOR_KIND[question.kind]
    answer_ok = fluency.score(qtype, question.expected, final_text)
    retrieve_ok = retrieved == question.needs_retrieve
    # A model that never called retrieve trivially "used the right handle" (nothing to
    # check) — handle_ok only penalizes a WRONG handle, not a missing call (that miss is
    # already captured by retrieve_ok).
    handle_ok = (not retrieved) or hit_expected_handle
    return retrieve_ok, answer_ok, handle_ok


def _run_questions_against(questions: list[DropQuestion], applied: policy_mod.Applied,
                           staging: dict[str, Any], answerer: ToolAnswerer,
                           trials: int = 1) -> list[dict]:
    """Run `trials` trials of the real 2-turn retrieve protocol for each of `questions`
    against one `answerer`, over an already-staged `(applied, staging)` pair. Split out
    of `run_drop_payload` so `run_drop_fluency` can compute `_questions_and_staging`
    ONCE per envelope and reuse it across every configured model, instead of
    re-deriving it (a JSON parse + a `policy.apply()` pass) once per model."""
    rows: list[dict] = []
    for q in questions:
        retrieve_ok = answer_ok = handle_ok = 0
        for _ in range(trials):
            r_ok, a_ok, h_ok = _run_question(q, applied.text, staging, answerer)
            retrieve_ok += int(r_ok)
            answer_ok += int(a_ok)
            handle_ok += int(h_ok)
        rows.append({
            "qid": q.qid, "kind": q.kind, "trials": trials,
            "retrieve_ok": retrieve_ok, "answer_ok": answer_ok, "handle_ok": handle_ok,
        })
    return rows


def run_drop_payload(obj: Any, raw: str, rule: Any, tool: str, answerer: ToolAnswerer,
                     trials: int = 1) -> list[dict]:
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
    return _run_questions_against(questions, applied, staging, answerer, trials=trials)


def run_drop_fluency(envelopes: list[dict], rule_for: Callable[[str], Any],
                     answerers: dict[str, ToolAnswerer], trials: int = 1) -> dict:
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
        rule = rule_for(tool)
        try:
            obj = json.loads(env["raw"])
        except (json.JSONDecodeError, TypeError):
            # Not JSON: the span-addressed text path is the only one that can drop here.
            # Same envelope-outer/model-inner nesting and the same scored-row shape, so a
            # text payload's results merge into the report exactly like a JSON one's.
            questions, applied, staging = _text_questions_and_staging(
                env.get("raw") or "", rule, tool)
            if not questions:
                continue
            assert applied is not None and staging is not None
            for name, fn in answerers.items():
                for row in _run_questions_against(questions, applied, staging, fn,
                                                  trials=trials):
                    results[name].append({"tool": tool, "sha": env.get("sha", "?"), **row})
            continue
        questions, applied, staging = _questions_and_staging(obj, rule, tool)
        if not questions:
            continue
        assert applied is not None and staging is not None
        for name, fn in answerers.items():
            for row in _run_questions_against(questions, applied, staging, fn, trials=trials):
                results[name].append({"tool": tool, "sha": env.get("sha", "?"), **row})
    return results


# --------------------------------------------------------------------------- #
# Tool-capable live backend — zero new dependencies (mirrors fluency.openai_answerer's
# urllib pattern, just carrying a `tools` param + parsing tool_calls).
# --------------------------------------------------------------------------- #
def _to_openai_tool(tool_def: dict) -> dict:
    """RETRIEVE_TOOL_DEF's MCP `inputSchema` shape -> OpenAI function-calling `parameters`."""
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
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
    url = base_url.rstrip("/") + "/chat/completions"
    oai_tools = [_to_openai_tool(t) for t in tools]

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
            calls.append(ToolCall(call_id=tc.get("id", ""), name=fn.get("name", ""),
                                  arguments=arguments))
        return Turn(text=msg.get("content") or "", tool_calls=calls)

    return ask
