"""Tests for the drop-to-retrieve behavioral eval — does a real tool-calling model call
`terse.retrieve` when a dropped field is needed, and leave it alone when it isn't?

The pure core (question generation, the 2-turn loop driver, scoring) is exercised
offline with scripted `ToolAnswerer` fakes standing in for a live model — the same
fake-answerer idiom test_fluency.py uses for the single-shot Answerer protocol. Live
backend (openai_tool_answerer) is a thin urllib adapter and is not unit-tested here,
mirroring fluency.py's own precedent.
"""

from __future__ import annotations

import json
import re
import urllib.request

import pytest

from terse import dropeval, lossy
from terse.policy import Policy, Rule
from terse.proxy import RETRIEVE_TOOL_DEF, Interceptor
from terse.report import build_dropeval_report

TOOL = "demo.get"

# One record ("id": 1) has a body big enough to clear the 200-char drop floor; the other
# two are short and stay in place — exactly one field should end up marked.
PAYLOAD = {"result": [
    {"id": 1, "body": "B" * 300},
    {"id": 2, "body": "short"},
    {"id": 3, "body": "short"},
]}

DROP_RULE = Rule(TOOL, ("minify", "tabularize", "dictionary"),
                 fields={"result[].body": {"lossy": "drop-to-retrieve"}})

# No field marked lossy at all -> policy.apply never drops anything for this rule.
NO_DROP_RULE = Rule(TOOL, ("minify", "tabularize", "dictionary"))

ALL_SHORT_PAYLOAD = {"result": [
    {"id": 1, "body": "short-a"},
    {"id": 2, "body": "short-b"},
]}


def _miss_text_from_proxy(handle: str) -> str:
    """Ground truth for a retrieve miss: drive the REAL proxy.Interceptor.answer_retrieve
    path (empty store) and read back its error text, so this test can't drift from
    production wording even if it changes."""
    inter = Interceptor(Policy(rules=[DROP_RULE]))
    reply = json.loads(inter.answer_retrieve(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "terse.retrieve", "arguments": {"handle": handle}},
    })))
    return reply["result"]["content"][0]["text"]


def _questions():
    qs = dropeval.gen_drop_questions(PAYLOAD, DROP_RULE, TOOL)
    return {q.kind: q for q in qs}


# --------------------------------------------------------------------------- #
# gen_drop_questions
# --------------------------------------------------------------------------- #
def test_gen_drop_questions_builds_one_recall_and_one_precision_question():
    qs = _questions()
    assert set(qs) == {"recall", "precision"}
    recall, precision = qs["recall"], qs["precision"]
    assert recall.needs_retrieve is True
    assert recall.expected == "B" * 300
    assert recall.expected_handle  # a real handle string, non-empty
    assert precision.needs_retrieve is False
    assert precision.expected_handle is None
    assert precision.expected == 3  # record count, unaffected by the drop


def test_no_questions_when_nothing_clears_the_drop_floor():
    # every candidate value is under the 200-char min -> nothing gets dropped -> no
    # recall question is even representable.
    assert dropeval.gen_drop_questions(ALL_SHORT_PAYLOAD, DROP_RULE, TOOL) == []


def test_no_questions_for_a_no_drop_policy():
    assert dropeval.gen_drop_questions(PAYLOAD, NO_DROP_RULE, TOOL) == []


# --------------------------------------------------------------------------- #
# run_drop_payload — recall question forces a real retrieve to be correct
# --------------------------------------------------------------------------- #
def test_recall_question_scores_zero_without_a_retrieve_call():
    def never_retrieves(messages):
        return dropeval.Turn(text="I don't know", tool_calls=[])

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     never_retrieves, trials=3)
    recall_row = next(r for r in rows if r["kind"] == "recall")
    assert recall_row["retrieve_ok"] == 0
    assert recall_row["answer_ok"] == 0
    assert recall_row["trials"] == 3


def test_recall_question_scores_full_marks_when_retrieve_resolves_correctly():
    qs = _questions()
    handle = qs["recall"].expected_handle

    def retrieves_then_answers(messages):
        if len(messages) == 2:  # first turn: system + user, no tool result yet
            return dropeval.Turn(text="", tool_calls=[
                dropeval.ToolCall(call_id="c1", name="terse.retrieve", arguments={"handle": handle}),
            ])
        # second turn: the tool result is the last message. lossy._serialize leaves a
        # bare string unquoted (readability), so a real model must still wrap it as
        # proper JSON to satisfy "reply with the value as compact JSON" — exactly what
        # this fake does, same as a competent model would.
        tool_result = messages[-1]["content"]
        return dropeval.Turn(text=json.dumps(tool_result), tool_calls=[])

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     retrieves_then_answers, trials=3)
    recall_row = next(r for r in rows if r["kind"] == "recall")
    assert recall_row["retrieve_ok"] == 3
    assert recall_row["answer_ok"] == 3
    assert recall_row["handle_ok"] == 3


def test_run_drop_payload_applies_policy_only_once(monkeypatch):
    # Regression: run_drop_payload used to call gen_drop_questions (which internally
    # calls _staged_apply) AND THEN call _staged_apply again itself with identical
    # args — a flat 2x cost on policy.apply()'s parse/tabularize/dictionary-encode
    # pass over every payload, before any per-question trial loop even started.
    from terse import policy as policy_mod

    calls = {"n": 0}
    real_apply = policy_mod.apply

    def counting_apply(*args, **kwargs):
        calls["n"] += 1
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(policy_mod, "apply", counting_apply)

    def never_retrieves(messages):
        return dropeval.Turn(text="I don't know", tool_calls=[])

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     never_retrieves, trials=1)
    assert rows  # sanity: the payload does have drop-marked questions
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# run_drop_payload — precision question penalizes over-fetch
# --------------------------------------------------------------------------- #
def test_precision_question_scores_zero_on_gratuitous_retrieve():
    def always_retrieves(messages):
        if len(messages) == 2:
            return dropeval.Turn(text="", tool_calls=[
                dropeval.ToolCall(call_id="c1", name="terse.retrieve", arguments={"handle": "whatever"}),
            ])
        return dropeval.Turn(text="3", tool_calls=[])

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     always_retrieves, trials=3)
    precision_row = next(r for r in rows if r["kind"] == "precision")
    assert precision_row["retrieve_ok"] == 0  # over-fetched every trial


def test_precision_question_scores_full_marks_answering_from_visible_data():
    def answers_directly(messages):
        return dropeval.Turn(text="3", tool_calls=[])

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     answers_directly, trials=3)
    precision_row = next(r for r in rows if r["kind"] == "precision")
    assert precision_row["retrieve_ok"] == 3
    assert precision_row["answer_ok"] == 3


# --------------------------------------------------------------------------- #
# Handle resolution matches proxy.answer_retrieve's real semantics
# --------------------------------------------------------------------------- #
def test_handle_resolves_to_exact_original_or_the_real_miss_string():
    applied, staging = dropeval._staged_apply(PAYLOAD, DROP_RULE, TOOL)
    recall_q = _questions()["recall"]

    def make_answerer(handle):
        state = {"n": 0}

        def ask(messages):
            state["n"] += 1
            if state["n"] == 1:
                return dropeval.Turn(text="", tool_calls=[
                    dropeval.ToolCall(call_id="c1", name="terse.retrieve",
                                      arguments={"handle": handle}),
                ])
            captured.append(messages[-1]["content"])
            return dropeval.Turn(text="done")

        return ask

    # a valid handle resolves to the exact original, serialized the same way lossy does
    captured: list = []
    dropeval._run_question(recall_q, applied.text, staging, make_answerer(recall_q.expected_handle))
    assert captured[-1] == lossy._serialize(recall_q.expected)

    # an unknown handle resolves to the identical miss string answer_retrieve emits
    captured = []
    dropeval._run_question(recall_q, applied.text, staging, make_answerer("does-not-exist"))
    assert captured[-1] == _miss_text_from_proxy("does-not-exist")


# --------------------------------------------------------------------------- #
# build_dropeval_report
# --------------------------------------------------------------------------- #
def _row(kind, retrieve_ok, answer_ok, handle_ok, trials=1):
    qid = "drop-recall" if kind == "recall" else "drop-precision"
    return {"tool": "t", "sha": "s", "qid": qid, "kind": kind, "trials": trials,
            "retrieve_ok": retrieve_ok, "answer_ok": answer_ok, "handle_ok": handle_ok}


def test_build_dropeval_report_verdict_gates_on_the_worst_model():
    good_rows = [_row("recall", 1, 1, 1), _row("precision", 1, 1, 1)] * 10
    poor_rows = [_row("recall", 0, 0, 0), _row("precision", 1, 1, 1)] * 10  # poor recall
    report = build_dropeval_report({"good": good_rows, "poor": poor_rows})
    verdict = report.split("## Verdict", 1)[1]
    assert "FAIL" in verdict
    assert "keep" in verdict.lower() and "off" in verdict.lower()


def test_build_dropeval_report_passes_when_every_model_is_reliable():
    rows = [_row("recall", 1, 1, 1), _row("precision", 1, 1, 1)] * 10
    report = build_dropeval_report({"m1": rows, "m2": rows})
    verdict = report.split("## Verdict", 1)[1]
    assert "PASS" in verdict
    assert "FAIL" not in verdict
    assert "safe to enable drop-to-retrieve" in verdict


def test_build_dropeval_report_empty_results_shows_guidance_and_does_not_crash():
    assert "No tool-capable model" in build_dropeval_report({})
    assert "No tool-capable model" in build_dropeval_report({"m": []})


def test_run_drop_fluency_computes_questions_once_per_envelope_not_per_model(monkeypatch):
    # Regression: run_drop_fluency's outer loop was over MODELS, redoing the JSON
    # parse + _questions_and_staging derivation (a policy.apply() pass) for every
    # model even though that work is entirely model-independent — for M models and E
    # envelopes this ran M times more often than necessary.
    calls = {"n": 0}
    real = dropeval._questions_and_staging

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(dropeval, "_questions_and_staging", counting)

    def never_retrieves(messages):
        return dropeval.Turn(text="I don't know", tool_calls=[])

    envelopes = [{"tool": TOOL, "sha": "abc", "raw": json.dumps(PAYLOAD)}]
    answerers = {"model-a": never_retrieves, "model-b": never_retrieves,
                "model-c": never_retrieves}
    results = dropeval.run_drop_fluency(envelopes, lambda t: DROP_RULE, answerers, trials=1)
    assert set(results) == {"model-a", "model-b", "model-c"}
    assert all(results[m] for m in results)  # each model still got real rows
    assert calls["n"] == 1  # one envelope -> one derivation, reused across all 3 models


# --------------------------------------------------------------------------- #
# The live OpenAI-compatible bridge. Previously untested "because it's a thin urllib
# adapter" — and that is precisely where the feature-killing bug lived: `terse.retrieve`
# is a legal MCP tool name and an ILLEGAL OpenAI function name, so every drop-eval run
# against a real endpoint 400'd and scored a confident 0% retrieve-recall.
# --------------------------------------------------------------------------- #
_OPENAI_FN_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def dropeval_text_rule():
    from terse.lossy import TEXT_SELECTOR_CODE_BLOCKS
    return Rule("codegraph_explore", ("minify",),
                fields={TEXT_SELECTOR_CODE_BLOCKS: {"lossy": "drop-to-retrieve"}})


def test_openai_tool_name_matches_openai_alphabet():
    # The MCP name is dotted; the wire name must not be, or the request is rejected
    # outright (400) before the model ever sees the question.
    assert not _OPENAI_FN_NAME.match(RETRIEVE_TOOL_DEF["name"])
    wire = dropeval._to_openai_tool(RETRIEVE_TOOL_DEF)["function"]["name"]
    assert _OPENAI_FN_NAME.match(wire), wire
    assert wire == "terse_retrieve"


def _fake_urlopen(captured: dict, tool_name_in_reply: str):
    """Stand in for the endpoint: record the request body, reply with one tool call."""
    class _Resp:
        def __init__(self, payload): self._payload = payload
        def read(self): return json.dumps(self._payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": tool_name_in_reply, "arguments": json.dumps({"handle": "h1"})}}],
        }}]})
    return fake


def test_openai_tool_answerer_sends_sanitized_name_and_maps_the_reply_back(monkeypatch):
    captured: dict = {}
    # A real endpoint echoes back the SANITIZED name it was given, so the adapter must
    # map it home — otherwise `_run_question`'s `c.name == RETRIEVE_TOOL` filter never
    # matches and every retrieve call is scored as "didn't retrieve".
    monkeypatch.setattr(urllib.request, "urlopen",
                        _fake_urlopen(captured, "terse_retrieve"))
    ask = dropeval.openai_tool_answerer("http://127.0.0.1:1/v1", "k", "m", tools=[RETRIEVE_TOOL_DEF])
    turn = ask([{"role": "user", "content": "hi"}])

    sent = captured["body"]["tools"][0]["function"]["name"]
    assert _OPENAI_FN_NAME.match(sent)
    assert [c.name for c in turn.tool_calls] == [lossy.RETRIEVE_TOOL]
    assert turn.tool_calls[0].arguments == {"handle": "h1"}
    assert not turn.error


def test_safe_call_records_the_failure_instead_of_scoring_it_as_a_refusal():
    def unreachable(messages):
        raise urllib.error.URLError("connection refused")

    turn = dropeval._safe_call(unreachable, [])
    assert turn.error and not turn.tool_calls

    rows = dropeval.run_drop_payload(PAYLOAD, json.dumps(PAYLOAD), DROP_RULE, TOOL,
                                     unreachable, trials=2)
    assert rows, "the payload has a drop-marked field, so questions were generated"
    # Every attempt failed: the row must say so, not just show 0 retrieve_ok.
    assert all(r["errors"] == r["trials"] for r in rows)


def test_report_refuses_a_verdict_when_the_calls_failed():
    rows = [{"qid": "q", "kind": "recall", "trials": 1, "retrieve_ok": 0,
             "answer_ok": 0, "handle_ok": 1, "errors": 1},
            {"qid": "q2", "kind": "precision", "trials": 1, "retrieve_ok": 1,
             "answer_ok": 0, "handle_ok": 1, "errors": 1}]
    report = build_dropeval_report({"model-a": rows})
    assert "INCONCLUSIVE" in report
    assert "failed 2/2 model calls" in report
    # The old output asserted a behavioral conclusion from a dead backend.
    assert "keep drop-to-retrieve off until this improves" not in report


def test_the_terminal_chart_refuses_the_same_run_the_markdown_does():
    """`dropeval_gap_rows`' docstring promises the two verdicts 'can never disagree'.
    Reporting failed calls only in the markdown would have broken exactly that: the forest
    plot would draw bars from transport errors, indistinguishable from a model that
    answered and got it wrong."""
    from terse.terminal_report import build_terminal_dropeval_report

    rows = [{"qid": "q", "kind": "recall", "trials": 1, "retrieve_ok": 0,
             "answer_ok": 0, "handle_ok": 1, "errors": 1},
            {"qid": "q2", "kind": "precision", "trials": 1, "retrieve_ok": 1,
             "answer_ok": 0, "handle_ok": 1, "errors": 1}]
    results = {"model-a": rows}
    assert "INCONCLUSIVE" in build_dropeval_report(results)
    assert "INCONCLUSIVE" in build_terminal_dropeval_report(results, color=False)


def test_sole_number_scoring_rejects_a_block_echo():
    """The numbered recall answer is scored `sole_number`, not `count`. The retrieved block
    is injected into the conversation carrying every one of its line numbers, so a reply
    that echoes the block contains the target number — `count`'s present-anywhere rule would
    pass it without the model having located the right line."""
    from terse.fluency.scoring import score
    assert score("sole_number", 61, "61")
    assert score("sole_number", 61, '"61"')
    assert score("sole_number", 61, "The line number is 61.")
    # A block echo: many line numbers, including the right one -> must NOT pass.
    assert not score("sole_number", 61, "40\tfoo 41\tbar 61\tbaz 79\tqux")
    # `count` DOES pass that echo — the reason we switched.
    assert score("count", 61, "40\tfoo 41\tbar 61\tbaz 79\tqux")


def test_the_recall_question_uses_sole_number_not_count():
    numbered = "\n".join(f"{i}\tcode call{i}(ctx, payload) // widening the covered region"
                         for i in range(40, 90))
    doc = f"## E\n\n### Source Code\n\n```go\n{numbered}\n```\n\nProse.\n"
    recall, _ = dropeval.gen_text_drop_questions(doc, dropeval_text_rule(), "codegraph_explore")
    assert recall.qtype == "sole_number"


def test_oai_name_collision_fails_loud_not_silent():
    """A future multi-tool eval whose names sanitize to one wire name must not silently
    map returned calls to the wrong tool — the very failure class this module fixes."""
    import pytest
    colliding = [dict(RETRIEVE_TOOL_DEF),
                 {**RETRIEVE_TOOL_DEF, "name": "terse_retrieve"}]  # sanitizes identically
    with pytest.raises(ValueError, match="collide"):
        dropeval.openai_tool_answerer("http://127.0.0.1:1/v1", "k", "m", tools=colliding)


# --- cleartext-credential guard parity with fluency.openai_answerer (audit 2026-07-23) ---
# This constructor sends the same `Authorization: Bearer <key>` fluency's does, but had
# NO guard at all — so `--base-url http://remote/v1` put the key on the wire in the clear.

def test_openai_tool_answerer_refuses_cleartext_key_to_remote_host():
    with pytest.raises(ValueError, match="cleartext http"):
        dropeval.openai_tool_answerer("http://api.example.com/v1", "sk-secret", "m",
                                      tools=[RETRIEVE_TOOL_DEF])


def test_openai_tool_answerer_allows_loopback_https_and_keyless_http():
    # The three shapes that carry no wire exposure — same allowances fluency makes.
    assert callable(dropeval.openai_tool_answerer("http://127.0.0.1:3456/v1", "sk-secret",
                                                  "m", tools=[RETRIEVE_TOOL_DEF]))
    assert callable(dropeval.openai_tool_answerer("https://api.example.com/v1", "sk-secret",
                                                  "m", tools=[RETRIEVE_TOOL_DEF]))
    assert callable(dropeval.openai_tool_answerer("http://api.example.com/v1", "",
                                                  "m", tools=[RETRIEVE_TOOL_DEF]))
