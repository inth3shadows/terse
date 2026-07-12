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

from terse import dropeval, lossy
from terse.policy import Policy, Rule
from terse.proxy import Interceptor
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
