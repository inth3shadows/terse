"""The codec verdict against the model that actually reads terse's output (Claude, via
`claude -p`) and with the primer production delivers.

Both deployed passthroughs were decided on a Qwen model with no primer, while the fleet's
reader is Claude and the live router ships the primer in the system prompt. These pin the
two run conditions that let the verdict ask the production question instead.
"""

from __future__ import annotations

import json

from terse import codeceval, fluency
from terse.cli import main
from terse.dropeval import ToolCall, Turn
from terse.report import build_codec_verdict_report, codec_call_rate

PAYLOAD = {"result": [
    {"id": i, "meta": {"owner": owner, "tags": ["x", "y"]}}
    for i, owner in enumerate(["alice", "bob", "carol", "dave", "erin", "frank"], 1)
]}
RAW_TEXT = json.dumps(PAYLOAD)


def _text_answerer(reply_for):
    """A text-channel answerer that records every request and answers `reply_for(msgs)`."""
    seen: list[list[dict]] = []

    def answer(messages):
        seen.append(messages)
        return Turn(text=reply_for(messages))

    answer.channel = "text"
    return answer, seen


def _expected(messages):
    """Answer each question correctly, whatever form the payload is in."""
    user = messages[-1]["content"]
    for q in codeceval.gen_codec_questions(PAYLOAD):
        if q.prompt in user:
            return json.dumps(q.expected)
    raise AssertionError("question not found in prompt")


def test_cli_text_answerer_splits_system_and_user_and_reads_None_as_an_error(monkeypatch):
    calls = []

    def fake_cli(alias):
        def ask(system, user):
            calls.append((alias, system, user))
            return None if "fail" in user else '["a"]'
        return ask

    monkeypatch.setattr(fluency, "cli_answerer", fake_cli)
    ans = codeceval.cli_text_answerer("haiku")
    assert codeceval.answer_channel(ans) == "text"
    ok = ans([{"role": "system", "content": "PRIMER"}, {"role": "user", "content": "q"}])
    assert ok == Turn(text='["a"]') and calls[-1] == ("haiku", "PRIMER", "q")
    # A None is a lost call, never an empty answer — scored through `fails`, not accuracy.
    assert ans([{"role": "user", "content": "fail"}]).error is True


def test_a_text_backend_is_asked_for_a_json_reply_not_a_tool_call():
    ans, seen = _text_answerer(_expected)
    q = codeceval.gen_codec_questions(PAYLOAD)[0]
    codeceval._codec_turn(q, RAW_TEXT, ans)
    user = seen[-1][-1]["content"]
    assert codeceval._TEXT_INSTRUCTION in user
    assert codeceval._codec_instruction() not in user
    assert all(m["role"] != "system" for m in seen[-1])   # no primer -> no system message


def test_the_primer_reaches_the_terse_arm_only():
    ans, seen = _text_answerer(_expected)
    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, ans, trials=1, primer="PRIMER")
    terse_text = fluency.compress(PAYLOAD)
    for msgs in seen:
        user = msgs[-1]["content"]
        has_system = any(m["role"] == "system" and m["content"] == "PRIMER" for m in msgs)
        # terse arm <=> primer: the counterfactual is terse installed vs not installed
        assert has_system == (terse_text in user)
    assert rows and all(r["primer"] is True for r in rows)


def test_text_rows_omit_the_tool_call_counters_so_SAFE_is_reachable():
    ans, _ = _text_answerer(_expected)
    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, ans, trials=20)
    assert rows and all(r["channel"] == "text" for r in rows)
    assert all("raw_calls" not in r and "terse_calls" not in r for r in rows)
    assert all(r["raw_ok"] == r["terse_ok"] == 20 for r in rows)
    # A 0 here would read as "declined the tool every time" and block SAFE forever.
    assert codec_call_rate(rows, "terse_ok") is None
    tagged = [{**r, "tool": "t", "shape": "array-of-records"} for r in rows]
    report = build_codec_verdict_report({"cli:haiku": tagged})
    assert "**SAFE**" in report
    assert "**Text channel:** `cli:haiku`" in report
    assert "**Primer:**" not in report


def test_tool_rows_keep_their_counters():
    def tool_ans(messages):
        return Turn(text="", tool_calls=[ToolCall(
            call_id="c1", name=codeceval.RECORD_VALUE_TOOL, arguments={"value": json.loads(
                _expected(messages))})])
    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, tool_ans, trials=1)
    assert all(r["channel"] == "tool" and r["raw_calls"] == 1 for r in rows)


def test_codec_verdict_accepts_a_cli_model_and_drop_eval_still_refuses_it(tmp_path, monkeypatch):
    from terse.capture import capture_payload
    corpus = tmp_path / "c"
    corpus.mkdir()
    capture_payload("kb.read.x", RAW_TEXT, corpus, server="kb", manual=True)
    built = []

    def fake_text(alias):
        built.append(alias)
        ans, _ = _text_answerer(lambda m: "[]")
        return ans

    monkeypatch.setattr(codeceval, "cli_text_answerer", fake_text)
    main(["fluency", "--codec-verdict", "--corpus", str(corpus), "--models", "cli:haiku",
          "--out", str(tmp_path / "r.md")])
    assert built == ["haiku"]
    # Every other tool-calling mode still refuses: it has no text-channel adapter.
    import argparse

    import pytest

    from terse.cli import _build_answerers
    ns = argparse.Namespace(base_url=None, api_key_env=None, models="cli:haiku")
    with pytest.raises(SystemExit, match="does not support --drop-eval"):
        _build_answerers(ns, lambda *a: None, mode_name="--drop-eval")
