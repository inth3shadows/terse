"""Tests for the codec-tier material-preservation eval (#295) — does a real tool-calling
model's downstream tool-call argument stay structurally identical whether it read raw JSON
or terse's compressed form?

Same fake-`ToolAnswerer` idiom as test_dropeval.py: a scripted callable stands in for a
live model, driven by message count/content rather than a real backend.
"""

from __future__ import annotations

import json

import pytest

from terse import codeceval, fluency
from terse.dropeval import ToolCall, Turn

# A uniform record list with an id column ("id") and a whole-object column ("meta") whose
# cells are all dicts — exactly what questions.py's deref generator requires (a blobcol
# where every record's value is a dict/list). SIX records, not three: `run_codec_fluency`
# skips a payload the codec passes through unencoded (#403 Blocker 2), and it declines to
# fold a list this narrow until there are enough rows for a table to pay.
PAYLOAD = {"result": [
    {"id": i, "meta": {"owner": owner, "tags": ["x", "y"]}}
    for i, owner in enumerate(["alice", "bob", "carol", "dave", "erin", "frank"], 1)
]}
RAW_TEXT = json.dumps(PAYLOAD)


def _deref_question():
    qs = [q for q in codeceval.gen_codec_questions(PAYLOAD) if q.qtype == "deref"]
    assert len(qs) == 1  # PAYLOAD has exactly one whole-object column
    return qs[0]


# --------------------------------------------------------------------------- #
# gen_codec_questions — qtype filter, not transform
# --------------------------------------------------------------------------- #
def test_gen_codec_questions_keeps_only_the_codec_qtypes():
    # count/aggregate answers are computed, not carried; lookup's is a bare scalar.
    all_qs = fluency.gen_questions(PAYLOAD)
    assert {q.qtype for q in all_qs} >= {"count", "enumerate", "deref", "aggregate"}
    codec_qs = codeceval.gen_codec_questions(PAYLOAD)
    assert sorted(q.qtype for q in codec_qs) == ["deref", "enumerate"]


def test_gen_codec_questions_enumerates_a_payload_with_no_container_column():
    # #403 Blocker 2: kb.read.search had 85 transformed payloads and zero deref questions.
    flat = {"result": [{"id": 1, "n": 10}, {"id": 2, "n": 20}]}
    qs = codeceval.gen_codec_questions(flat)
    assert [(q.qtype, q.expected) for q in qs] == [("enumerate", [1, 2])]


def test_gen_codec_questions_empty_without_an_id_or_container_column():
    # No unique scalar column to enumerate, no whole-object column to deref.
    flat = {"result": [{"k": "x", "n": 10}, {"k": "x", "n": 10}]}
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


def test_run_codec_payload_counts_a_transport_failure_as_a_miss_in_a_fixed_denominator():
    # A safety verdict must not let a silent non-answer shrink itself out of the sample
    # (review finding 3 on PR #302) — unlike `harnesses.run_payload`'s comprehension arms,
    # `raw_trials`/`terse_trials` stay fixed at `trials` regardless of errors, and the
    # errored call scores as a miss (not a match) rather than being excluded.
    def always_errors(messages):
        raise RuntimeError("connection refused")

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, always_errors, trials=2)
    row = rows[0]
    assert row["raw_ok"] == 0
    assert row["terse_ok"] == 0
    assert row["raw_trials"] == 2  # fixed, not reduced
    assert row["terse_trials"] == 2
    assert row["fails"] == 4  # 2 raw + 2 terse — still tracked for the _unmeasured gate
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
    qs = codeceval.gen_codec_questions(PAYLOAD)
    assert len(seen) == 2 * len(qs) == 4
    for q in qs:
        arms = [s for s in seen if q.prompt in s]
        assert len(arms) == 2, (q.prompt, seen)
        assert all(codeceval._codec_instruction() in s for s in arms)
        # Different DATA blocks: the raw arm's prompt must literally contain raw JSON text,
        # the terse arm's must contain terse's compressed form, not the same string twice.
        assert arms[0] != arms[1]


def test_run_codec_payload_sends_no_system_message_at_all():
    # Not an EMPTY system message — some OpenAI-compatible backends reject one outright
    # (review finding 4 on PR #302). Only a user message should ever be sent.
    seen: list[list[dict]] = []

    def capture(messages):
        seen.append(messages)
        return Turn(text="", tool_calls=[])

    codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, capture, trials=1)
    assert seen
    for messages in seen:
        assert all(m["role"] != "system" for m in messages)
        assert [m["role"] for m in messages] == ["user"]


def test_run_codec_payload_uses_the_last_tool_call_when_the_model_calls_twice():
    q = _deref_question()

    def calls_twice_first_correct(messages):
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": q.expected}),
            ToolCall(call_id="c2", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": {"wrong": "value"}}),
        ])

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, calls_twice_first_correct, trials=1)
    row = rows[0]
    # Last-wins: the LATER (wrong) call decides the score, not the first (correct) one.
    assert row["raw_ok"] == 0
    assert row["terse_ok"] == 0


def test_run_codec_payload_scores_zero_when_the_value_key_is_missing():
    def calls_without_value_key(messages):
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL, arguments={}),
        ])

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, calls_without_value_key, trials=1)
    row = rows[0]
    assert row["raw_ok"] == 0
    assert row["terse_ok"] == 0
    assert row["raw_trials"] == 1  # the call landed; a missing key is a miss, not a transport error
    assert row["fails"] == 0


# --------------------------------------------------------------------------- #
# run_codec_fluency — envelope tagging (tool + shape)
# --------------------------------------------------------------------------- #
def test_run_codec_fluency_stamps_tool_and_shape_on_every_row():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    envelopes = [{"tool": "demo.get", "shape": "array-of-records", "sha": "abc123",
                 "raw": RAW_TEXT}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1, preflight=False)
    assert results["m1"]
    for row in results["m1"]:
        assert row["tool"] == "demo.get"
        assert row["shape"] == "array-of-records"
        assert row["sha"] == "abc123"


def test_run_codec_fluency_skips_a_payload_with_no_codec_question():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    flat = {"result": [{"k": "x", "n": 10}, {"k": "x", "n": 10}]}
    envelopes = [{"tool": "demo.get", "shape": "array-of-records", "sha": "x",
                 "raw": json.dumps(flat)}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1, preflight=False)
    assert results["m1"] == []


def test_run_codec_fluency_skips_non_json_payloads():
    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    envelopes = [{"tool": "demo.get", "shape": "long-text", "sha": "x", "raw": "not json"}]
    results = codeceval.run_codec_fluency(envelopes, {"m1": never_calls}, trials=1, preflight=False)
    assert results["m1"] == []


# --- #403: container-argument encoding pre-flight -----------------------------------

NO_CALL = object()
ERROR = object()


def preflight_stub(answer):
    """Answerer that recognises which pre-flight question a request carries (by its prompt
    AND payload text — each payload carries one question per codec qtype) and returns
    `answer(expected, n)` as the `value` argument, `n` being the 0-based call index across
    the whole pre-flight. `NO_CALL` returns a prose turn, `ERROR` an errored one. A request
    that is not a pre-flight question is answered `{"wrong": True}`, so a stub that clears
    the pre-flight still scores nothing on the corpus."""
    n = {"i": 0}

    def answerer(messages):
        content = messages[-1]["content"]
        expected = next((q.expected for q, text in codeceval.preflight_questions()
                         if q.prompt in content and text in content), None)
        if expected is None:
            v = {"wrong": True}
        else:
            v = answer(expected, n["i"])
            n["i"] += 1
        if v is NO_CALL:
            return Turn(text='["x", "y"]', tool_calls=[])
        if v is ERROR:
            return Turn(text="", error=True)
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL, arguments={"value": v})])
    return answerer


@pytest.mark.parametrize("got,expected,is_mismatch", [
    ('["x"]', ["x"], True),                 # the deepseek-v4-pro case
    ('{"a": 1}', {"a": 1}, True),
    ('[]', [], True),
    (["x"], ["x"], False),                  # correctly typed -> not a mismatch
    ("nope", ["x"], False),                 # not even parseable
    ('["z"]', ["x"], False),                # parses, but wrong content: a real miss
    ('{"a": 1}', [{"a": 1}], False),        # parses to the wrong container type
    ('"a"', "a", False),                    # string expected: excluded by design
    ("null", None, False),                  # null expected: excluded by design
    ("1", 1, False),
    (None, ["x"], False),
    pytest.param("[" * 100_000, ["x"], False, id="deeply-nested"),  # pathological: no crash
])
def test_encodes_as_json_string_identifies_only_right_content_wrong_type(got, expected,
                                                                        is_mismatch):
    assert codeceval.encodes_as_json_string(got, expected) is is_mismatch


def test_encoding_detection_never_turns_a_miss_into_a_hit():
    # The whole point: this is diagnosis, not scoring. A stringified answer stays WRONG.
    assert codeceval._value_matches('["x"]', ["x"]) is False
    assert codeceval.encodes_as_json_string('["x"]', ["x"]) is True


def test_preflight_asks_one_deref_question_per_container_type():
    # A model can stringify objects and not arrays; the first version only asked an array.
    # And each asked-for value NESTS a container: a model that stringifies only inner values
    # would otherwise pass here and zero both arms of the sweep (round-2 review of #403).
    # Each payload also asks one `enumerate`, over string ids and integer ids respectively.
    qs = codeceval.preflight_questions()
    derefs = [q for q, _ in qs if q.qtype == "deref"]
    enums = [q for q, _ in qs if q.qtype == "enumerate"]
    assert len(qs) == 4 and len(derefs) == 2 and len(enums) == 2
    assert sorted(type(q.expected).__name__ for q in derefs) == ["dict", "list"]
    for q in derefs:
        inner = q.expected if isinstance(q.expected, list) else list(q.expected.values())
        assert any(isinstance(v, (dict, list)) for v in inner), q.expected
    assert sorted(type(q.expected[0]).__name__ for q in enums) == ["int", "str"]


def test_preflight_refuses_to_check_nothing_if_question_generation_changes(monkeypatch):
    # An empty question list would pass every model through a pre-flight that asked nothing.
    real = codeceval.gen_codec_questions
    monkeypatch.setattr(codeceval, "gen_codec_questions", lambda obj: [])
    with pytest.raises(RuntimeError, match=r"yields \[\], expected one each"):
        codeceval.preflight_encoding(preflight_stub(lambda e, n: e))
    # Losing ONE type is the same hole: the sweep would score `enumerate` answers the
    # pre-flight never checked a model could encode.
    monkeypatch.setattr(codeceval, "gen_codec_questions",
                        lambda obj: [q for q in real(obj) if q.qtype == "deref"])
    with pytest.raises(RuntimeError, match=r"yields \['deref'\], expected one each"):
        codeceval.preflight_encoding(preflight_stub(lambda e, n: e))
    # Nor is the COUNT the test: two of one type still never checks the other.
    monkeypatch.setattr(codeceval, "gen_codec_questions",
                        lambda obj: [q for q in real(obj) if q.qtype == "deref"] * 2)
    with pytest.raises(RuntimeError, match=r"yields \['deref', 'deref'\], expected one each"):
        codeceval.preflight_encoding(preflight_stub(lambda e, n: e))


def test_preflight_sends_the_request_a_real_trial_sends():
    # Review of #403: the pre-flight's request was a copy, and a mutation that changed only
    # the copy passed every test. Same question and payload text -> identical messages.
    def recorder(sink):
        def answerer(messages):
            sink.append(messages)
            return Turn(text="", error=True)
        return answerer

    q, text = codeceval.preflight_questions()[0]
    via_preflight: list = []
    via_trial: list = []
    codeceval.preflight_encoding(recorder(via_preflight), attempts=1)
    codeceval._ask_codec_question(q, text, recorder(via_trial))
    assert via_trial and via_preflight[0] == via_trial[0]


def test_preflight_passes_a_model_that_sends_native_containers():
    assert codeceval.preflight_encoding(preflight_stub(lambda e, n: e)) is None


def test_preflight_names_the_json_string_encoding_and_counts_answers():
    why = codeceval.preflight_encoding(preflight_stub(lambda e, n: json.dumps(e)), attempts=3)
    assert why and "JSON STRING" in why and "[12/12 pre-flight answers failed]" in why, why


def test_preflight_catches_a_model_that_stringifies_only_objects():
    stub = preflight_stub(lambda e, n: json.dumps(e) if isinstance(e, dict) else e)
    why = codeceval.preflight_encoding(stub, attempts=3)
    assert why and "JSON STRING" in why and "[3/12 pre-flight answers failed]" in why, why


def test_preflight_rejects_a_model_that_is_only_sometimes_right():
    # Measured on deepseek-v4-pro at temperature=0: one call stringified, another sent no
    # `value` at all. A model that expresses a container argument only sometimes still
    # poisons the arm it is on, so ALL answers must match.
    stub = preflight_stub(lambda e, n: json.dumps(e) if n == 1 else e)
    why = codeceval.preflight_encoding(stub, attempts=3)
    assert why and "[1/12 pre-flight answers failed]" in why, why


def test_preflight_names_a_model_that_answers_in_prose():
    why = codeceval.preflight_encoding(preflight_stub(lambda e, n: NO_CALL), attempts=3)
    assert why and why.startswith(f"answered without calling {codeceval.RECORD_VALUE_TOOL}")


def test_preflight_retries_an_errored_turn_instead_of_failing_the_model():
    # One timeout says nothing about encoding; before this, it excluded a perfect model.
    stub = preflight_stub(lambda e, n: ERROR if n == 0 else e)
    assert codeceval.preflight_encoding(stub, attempts=3) is None


def test_preflight_reports_a_backend_that_never_answers_as_that():
    why = codeceval.preflight_encoding(preflight_stub(lambda e, n: ERROR), attempts=3)
    assert why and why.startswith(
        "backend returned no usable turn on 6 of 6 calls for pre-flight question 1 of 4 — ")


def test_preflight_needs_every_scorable_answer_not_just_one():
    # One good answer then five errors: 1 of the 3 answers the question needs.
    why = codeceval.preflight_encoding(preflight_stub(lambda e, n: ERROR if n else e), attempts=3)
    assert why and why.startswith(
        "backend returned no usable turn on 5 of 6 calls for pre-flight question 1 of 4")


def test_preflight_names_the_question_a_backend_loss_happened_on():
    # Question 1 answered cleanly 3 times; the loss is question 2's alone (round-3 review).
    why = codeceval.preflight_encoding(preflight_stub(lambda e, n: ERROR if n >= 3 else e),
                                       attempts=3)
    assert why and why.startswith(
        "backend returned no usable turn on 6 of 6 calls for pre-flight question 2 of 4")


def test_preflight_reports_the_first_failure_not_the_latest():
    stub = preflight_stub(lambda e, n: json.dumps(e) if n == 0 else NO_CALL if n == 1 else e)
    why = codeceval.preflight_encoding(stub, attempts=3)
    assert why and why.startswith("sends container arguments as a JSON STRING"), why
    assert "[2/12 pre-flight answers failed]" in why


def test_preflight_keeps_an_answer_failure_ahead_of_a_later_backend_failure():
    # A JSON-string model behind a flaky gateway must not read as "retry and see".
    stub = preflight_stub(lambda e, n: json.dumps(e) if n < 2 else ERROR)
    why = codeceval.preflight_encoding(stub, attempts=3)
    assert why and why.startswith("sends container arguments as a JSON STRING"), why
    assert ("[2/2 pre-flight answers failed before the backend returned no usable turn on "
            "4 of 6 calls for pre-flight question 1 of 4]") in why, why


def test_run_codec_fluency_refuses_the_run_when_any_model_fails_the_preflight():
    # Dropping the failed model and continuing let a cell read SAFE over fewer models than
    # were asked for (review of #403). The refusal lands BEFORE any corpus question: `good`
    # passes the pre-flight, so only running the sweep first could reach `corpus_calls`.
    corpus_calls: list = []
    passes = preflight_stub(lambda e, n: e)

    def good(messages):
        if not any(text in messages[-1]["content"] for _, text in codeceval.preflight_questions()):
            corpus_calls.append(messages)
        return passes(messages)

    bad = preflight_stub(lambda e, n: json.dumps(e))
    envelopes = [{"tool": "demo.get", "sha": "abc", "raw": RAW_TEXT}]
    with pytest.raises(codeceval.PreflightError) as exc:
        codeceval.run_codec_fluency(envelopes, {"good": good, "bad": bad}, trials=1)
    msg = str(exc.value)
    assert "1 of 2 model(s)" in msg and "  bad: sends container arguments as a JSON STRING" in msg
    assert "  good:" not in msg
    assert corpus_calls == []


def test_run_codec_fluency_preflights_with_the_default_attempt_count():
    # Pins PREFLIGHT_ATTEMPTS through the production call: a model wrong only on its third
    # answer (n == 2) passes a 1-attempt pre-flight, and a 2-attempt one reports "1/8".
    late_miss = preflight_stub(lambda e, n: json.dumps(e) if n == 2 else e)
    with pytest.raises(codeceval.PreflightError, match="1/12 pre-flight answers failed"):
        codeceval.run_codec_fluency([], {"m": late_miss}, trials=1)

