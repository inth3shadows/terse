"""Tests for the codec-tier material-preservation eval (#295) — does a real tool-calling
model's downstream tool-call argument stay structurally identical whether it read raw JSON
or terse's compressed form?

Same fake-`ToolAnswerer` idiom as test_dropeval.py: a scripted callable stands in for a
live model, driven by message count/content rather than a real backend.
"""

from __future__ import annotations

import json

from terse import codeceval, fluency
from terse.dropeval import ToolCall, Turn

# A uniform record list with an id column ("id") and a whole-object column ("meta") whose
# cells are all dicts — exactly what questions.py's deref generator requires (a blobcol
# where every record's value is a dict/list).
PAYLOAD = {"result": [
    {"id": 1, "meta": {"owner": "alice", "tags": ["x", "y"]}},
    {"id": 2, "meta": {"owner": "bob", "tags": ["z"]}},
    {"id": 3, "meta": {"owner": "carol", "tags": []}},
]}
RAW_TEXT = json.dumps(PAYLOAD)


def _deref_question():
    qs = codeceval.gen_codec_questions(PAYLOAD)
    assert len(qs) == 1  # PAYLOAD has exactly one whole-object column
    return qs[0]


# --------------------------------------------------------------------------- #
# gen_codec_questions — qtype filter, not transform
# --------------------------------------------------------------------------- #
def test_gen_codec_questions_keeps_only_deref():
    all_qs = fluency.gen_questions(PAYLOAD)
    assert {q.qtype for q in all_qs} >= {"count", "enumerate", "deref"}
    codec_qs = codeceval.gen_codec_questions(PAYLOAD)
    assert codec_qs and all(q.qtype == "deref" for q in codec_qs)
    assert len(codec_qs) < len(all_qs)  # the comprehension-only qtypes were dropped


def test_gen_codec_questions_empty_when_no_deref_question():
    # No whole-object/array column -> questions.py never emits a deref question.
    flat = {"result": [{"id": 1, "n": 10}, {"id": 2, "n": 20}]}
    assert fluency.gen_questions(flat)  # sanity: still a valid comprehension payload
    assert codeceval.gen_codec_questions(flat) == []


# --------------------------------------------------------------------------- #
# _value_matches — the absent-vs-null distinction a `deref` failure destroys
# --------------------------------------------------------------------------- #
def test_value_matches_is_order_insensitive_on_dict_keys():
    assert codeceval._value_matches({"a": 1, "b": None}, {"b": None, "a": 1})


def test_value_matches_distinguishes_absent_key_from_explicit_null():
    # {"a": 1} (no "b" key at all) must NOT match {"a": 1, "b": None} (explicit null) —
    # this is exactly the structural corruption #295 says a deref failure produces.
    assert not codeceval._value_matches({"a": 1}, {"a": 1, "b": None})
    assert not codeceval._value_matches({"a": 1, "b": None}, {"a": 1})


def test_value_matches_distinguishes_positional_list_from_keyed_object():
    assert not codeceval._value_matches([1, 2, 3], {"0": 1, "1": 2, "2": 3})


# --------------------------------------------------------------------------- #
# run_codec_payload — the tool-call comparison itself
# --------------------------------------------------------------------------- #
def test_run_codec_payload_scores_full_marks_when_the_call_matches():
    q = _deref_question()

    def answers_correctly(messages):
        # The DATA block carries the payload text; scripted answerer replies with
        # whichever expected value the question is currently asking about, regardless of
        # raw vs terse framing — this fake tests the SCORING path, not comprehension.
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": q.expected}),
        ])

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, answers_correctly, trials=3)
    row = next(r for r in rows if r["qid"] == q.qid)
    assert row["raw_ok"] == 3
    assert row["terse_ok"] == 3
    assert row["raw_trials"] == 3
    assert row["terse_trials"] == 3
    assert row["fails"] == 0
    assert row["attempts"] == 6


def test_run_codec_payload_scores_zero_when_the_call_argument_is_wrong():
    def answers_wrongly(messages):
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": {"totally": "different"}}),
        ])

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, answers_wrongly, trials=2)
    row = rows[0]
    assert row["raw_ok"] == 0
    assert row["terse_ok"] == 0
    # Wrong is not the same as unmeasured: the calls still landed.
    assert row["raw_trials"] == 2
    assert row["terse_trials"] == 2
    assert row["fails"] == 0


def test_run_codec_payload_scores_zero_but_not_errored_when_the_model_answers_in_prose():
    def answers_in_prose(messages):
        return Turn(text="the value is whatever", tool_calls=[])

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, answers_in_prose, trials=1)
    row = rows[0]
    assert row["raw_ok"] == 0
    assert row["terse_ok"] == 0
    assert row["raw_trials"] == 1  # the call landed — declining to call the tool is a MISS,
    assert row["terse_trials"] == 1  # not a transport failure, so it stays in the denominator
    assert row["fails"] == 0


def test_run_codec_payload_excludes_a_transport_failure_from_the_denominator():
    def always_errors(messages):
        raise RuntimeError("connection refused")

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, always_errors, trials=2)
    row = rows[0]
    assert row["raw_trials"] == 0
    assert row["terse_trials"] == 0
    assert row["fails"] == 4  # 2 raw + 2 terse
    assert row["attempts"] == 4


def test_run_codec_payload_asks_the_same_question_against_both_forms():
    # The raw and terse arms must be asked the SAME question text/instruction — only the
    # DATA block should differ. Captures the user-prompt content seen by the fake and
    # confirms both arms carried an identical prompt/instruction pair.
    seen: list[str] = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return Turn(text="", tool_calls=[])

    codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, capture, trials=1)
    q = _deref_question()
    assert all(q.prompt in s and q.instruction in s for s in seen)
    # Different DATA blocks: the raw arm's prompt must literally contain raw JSON text,
    # the terse arm's must contain terse's compressed form, not the same string twice.
    assert seen[0] != seen[1]


# --------------------------------------------------------------------------- #
# run_codec_fluency — envelope tagging (tool + shape)
# --------------------------------------------------------------------------- #
def test_run_codec_fluency_stamps_tool_and_shape_on_every_row():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    envelopes = [{"tool": "demo.get", "shape": "array-of-records", "sha": "abc123",
                 "raw": RAW_TEXT}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1)
    assert results["m1"]
    for row in results["m1"]:
        assert row["tool"] == "demo.get"
        assert row["shape"] == "array-of-records"
        assert row["sha"] == "abc123"


def test_run_codec_fluency_skips_a_payload_with_no_deref_question():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    flat = {"result": [{"id": 1, "n": 10}, {"id": 2, "n": 20}]}
    envelopes = [{"tool": "demo.get", "shape": "array-of-records", "sha": "x",
                 "raw": json.dumps(flat)}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1)
    assert results["m1"] == []


def test_run_codec_fluency_skips_non_json_payloads():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    envelopes = [{"tool": "demo.get", "shape": "long-text", "sha": "x", "raw": "not json"}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1)
    assert results["m1"] == []
