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
# Three payloads, two questions each: SAFE needs 5 complete questions (`_CODEC_MIN_QUESTIONS`).
PAYLOADS = [{"result": [
    {"id": i + k * 10, "meta": {"owner": f"{owner}{k}", "tags": ["x", "y"]}}
    for i, owner in enumerate(["alice", "bob", "carol", "dave", "erin", "frank"], 1)
]} for k in range(3)]


def _text_answerer(reply_for):
    """A text-channel answerer that records every request and answers `reply_for(msgs)`."""
    seen: list[list[dict]] = []

    def answer(messages):
        seen.append(messages)
        return Turn(text=reply_for(messages))

    answer.channel = "text"
    return answer, seen


def _expected(messages):
    """Answer each question correctly, whatever form the payload is in — the pre-flight's
    questions included, so a CLI run gets past it."""
    user = messages[-1]["content"]
    # Keyed by the payload actually in the prompt (raw or compressed): the payloads share
    # question wording, so matching the prompt alone would answer from the wrong one.
    known = [(q, [t]) for q, t in codeceval.preflight_questions()]
    known += [(q, [json.dumps(pl), fluency.compress(pl)])
              for pl in [PAYLOAD, *PAYLOADS] for q in codeceval.gen_codec_questions(pl)]
    for q, texts in known:
        if q.prompt in user and any(t in user for t in texts):
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
    rows = [r for pl in PAYLOADS
            for r in codeceval.run_codec_payload(pl, json.dumps(pl), ans, trials=20)]
    assert len(rows) >= 5 and all(r["channel"] == "text" for r in rows)
    assert all(r["raw_parsed"] == r["terse_parsed"] == 20 for r in rows)
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


def test_a_cli_run_completes_and_primer_reaches_only_the_terse_arm(tmp_path, monkeypatch):
    """Through `main`, not `_build_answerers`: review found the old version of this test
    passed on a run the pre-flight refused (rc 2, no report), and that the `--primer` CLI
    wiring was pinned by nothing — `primer=""` hardcoded in cli.py survived the suite."""
    from terse.capture import capture_payload
    corpus = tmp_path / "c"
    corpus.mkdir()
    for pl in PAYLOADS:
        capture_payload("kb.read.x", json.dumps(pl), corpus, server="kb", manual=True)
    seen: list[list[dict]] = []

    def fake_text(alias):
        ans, log = _text_answerer(_expected)
        seen.append(log)  # type: ignore[arg-type]
        return ans

    monkeypatch.setattr(codeceval, "cli_text_answerer", fake_text)
    out = tmp_path / "r.md"
    assert main(["fluency", "--codec-verdict", "--primer", "--corpus", str(corpus),
                 "--trials", "20", "--models", "cli:haiku", "--out", str(out)]) == 0
    report = out.read_text()
    assert "**SAFE**" in report and "**Primer:**" in report and "**Text channel:**" in report
    raws = [json.dumps(pl) for pl in PAYLOADS]
    terses = [fluency.compress(pl) for pl in PAYLOADS]
    sweep = [m for m in seen[0] if any(t in m[-1]["content"] for t in raws + terses)]
    assert sweep
    for msgs in sweep:
        has_primer = any(m["role"] == "system" for m in msgs)
        assert has_primer == any(t in msgs[-1]["content"] for t in terses)
        # the SAME text instruction on both arms — pairing depends on it
        assert codeceval._TEXT_INSTRUCTION in msgs[-1]["content"]


def test_drop_eval_still_refuses_a_cli_model():
    import argparse

    import pytest

    from terse.cli import _build_answerers
    ns = argparse.Namespace(base_url=None, api_key_env=None, models="cli:haiku")
    with pytest.raises(SystemExit, match="does not support --drop-eval"):
        _build_answerers(ns, lambda *a: None, mode_name="--drop-eval")


def test_a_backend_that_dies_mid_question_costs_both_arms_alike():
    """Review: raw trials then terse trials turned a quota wall into raw k / terse 0 —
    UNSAFE from transport loss. Interleaved, the arms lose at most one call apart."""
    state = {"n": 0}

    def dies_after_five(messages):
        state["n"] += 1
        if state["n"] > 5:
            return Turn(text="", error=True)
        return Turn(text=_expected(messages))
    dies_after_five.channel = "text"  # type: ignore[attr-defined]

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, dies_after_five, trials=10)
    first = rows[0]
    assert abs(first["raw_answered"] - first["terse_answered"]) <= 1
    assert abs(first["raw_ok"] - first["terse_ok"]) <= 1


def test_a_right_value_in_the_wrong_format_is_a_format_miss_not_a_parse():
    def explains_first(messages):
        return Turn(text="Here it is: " + _expected(messages))
    explains_first.channel = "text"  # type: ignore[attr-defined]

    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, explains_first, trials=2)
    assert all(r["raw_ok"] == r["terse_ok"] == 0 for r in rows)
    assert all(r["raw_parsed"] == r["terse_parsed"] == 0 for r in rows)


def test_the_primer_counts_toward_the_terse_arms_input_limit():
    # Review: request_tokens ignored the ~555-token primer, so a terse request really over
    # the limit passed the check and errored on the terse arm only.
    import pytest

    from terse.tokenize import count_cl100k
    if count_cl100k("x") is None:
        pytest.skip("no tokenizer")
    q = codeceval.gen_codec_questions(PAYLOAD)[0]
    terse_text = fluency.compress(PAYLOAD)
    bare = codeceval.request_tokens(q, terse_text)
    primer = "primer " * 400
    assert codeceval.request_tokens(q, terse_text, system=primer) > bare
    limit = max(codeceval.request_tokens(q2, t, None) or 0
                for q2 in codeceval.gen_codec_questions(PAYLOAD)
                for t in (RAW_TEXT, terse_text)) + 1
    assert codeceval.oversized_arms(PAYLOAD, RAW_TEXT, limit) == []
    over = codeceval.oversized_arms(PAYLOAD, RAW_TEXT, limit, primer=primer)
    assert [arm for arm, _ in over] == ["terse"]
