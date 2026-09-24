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


def test_the_primer_reaches_the_terse_arm_only_and_inline():
    # Since #451 the router attaches the primer as block 0 of the first terse-marked result,
    # so the terse arm reads it INLINE ahead of the payload; no system message on either arm.
    ans, seen = _text_answerer(_expected)
    rows = codeceval.run_codec_payload(PAYLOAD, RAW_TEXT, ans, trials=1, primer="PRIMER")
    terse_text = fluency.compress(PAYLOAD)
    for msgs in seen:
        assert all(m["role"] != "system" for m in msgs)
        user = msgs[-1]["content"]
        assert ("PRIMER\n\n" + terse_text in user) == (terse_text in user)
        assert ("PRIMER" in user) == (terse_text in user)
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
    """Through `main`: a `cli:` run completes (rc 0) and `--primer` sends the primer the
    policy builds, inline, on the terse arm only. The first version of this test passed on a
    run the pre-flight refused, and nothing pinned the CLI wiring of `--primer`."""
    from terse.capture import capture_payload
    from terse.policy import load_policy
    from terse.proxy import union_primer
    corpus = tmp_path / "c"
    corpus.mkdir()
    for pl in PAYLOADS:
        capture_payload("kb.read.x", json.dumps(pl), corpus, server="kb", manual=True)
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "kb.*"}, "tiers": ["minify", "tabularize", "dictionary"]}]}))
    expected_primer = union_primer([(load_policy(str(policy)), "kb")])
    assert expected_primer
    seen: list[list[dict]] = []

    def fake_text(alias):
        ans, log = _text_answerer(_expected)
        seen.append(log)  # type: ignore[arg-type]
        return ans

    monkeypatch.setattr(codeceval, "cli_text_answerer", fake_text)
    out = tmp_path / "r.md"
    assert main(["fluency", "--codec-verdict", "--primer", "--policy", str(policy),
                 "--corpus", str(corpus), "--trials", "20", "--models", "cli:haiku",
                 "--out", str(out)]) == 0
    report = out.read_text()
    assert "**SAFE**" in report and "**Primer:**" in report and "**Text channel:**" in report
    terses = [fluency.compress(pl) for pl in PAYLOADS]
    raws = [json.dumps(pl) for pl in PAYLOADS]
    sweep = [m for m in seen[0] if any(t in m[-1]["content"] for t in raws + terses)]
    assert sweep
    for msgs in sweep:
        user = msgs[-1]["content"]
        is_terse = any(t in user for t in terses)
        assert (expected_primer in user) == is_terse
        assert codeceval._TEXT_INSTRUCTION in user


def test_primer_needs_a_policy_and_the_codec_verdict(tmp_path):
    from terse.capture import capture_payload
    corpus = tmp_path / "c"
    corpus.mkdir()
    capture_payload("kb.read.x", RAW_TEXT, corpus, server="kb", manual=True)
    assert main(["fluency", "--codec-verdict", "--primer", "--corpus", str(corpus),
                 "--models", "cli:haiku", "--out", str(tmp_path / "r.md")]) == 2
    assert main(["fluency", "--primer", "--corpus", str(corpus),
                 "--out", str(tmp_path / "f.md")]) == 2

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
    # Through `run_codec_fluency`, the call site: review found a direct `oversized_arms` test
    # survived a mutation that stopped the sweep passing the primer.
    import pytest

    from terse.tokenize import count_cl100k
    if count_cl100k("x") is None:
        pytest.skip("no tokenizer")
    primer = "primer " * 400
    terse_text = fluency.compress(PAYLOAD)
    limit = max(codeceval.request_tokens(q, t) or 0
                for q in codeceval.gen_codec_questions(PAYLOAD)
                for t in (RAW_TEXT, terse_text)) + 1
    env = {"tool": "kb.read.x", "server": "kb", "raw": RAW_TEXT, "sha": "a" * 40,
           "shape": "array-of-records", "manual": True}
    ans, _ = _text_answerer(_expected)
    unprimed = codeceval.run_codec_fluency([env], {"m": ans}, trials=1, preflight=False,
                                           limits={"m": limit})
    assert not unprimed.excluded
    primed = codeceval.run_codec_fluency([env], {"m": ans}, trials=1, preflight=False,
                                         limits={"m": limit}, primer=primer)
    assert [e.arm for e in primed.excluded] == ["terse"]



def test_the_preflight_asks_the_primed_request_when_there_is_a_primer():
    ans, seen = _text_answerer(_expected)
    assert codeceval.preflight_encoding(ans, attempts=1, primer="PRIMER") is None
    assert seen and all(m[-1]["content"].count("PRIMER\n\n") == 1 for m in seen)
    seen.clear()
    codeceval.preflight_encoding(ans, attempts=1)
    assert seen and all("PRIMER" not in m[-1]["content"] for m in seen)


def test_R3_the_primer_and_the_payload_come_from_the_same_policy(tmp_path, monkeypatch):
    from terse.capture import capture_payload
    from terse.policy import apply as policy_apply
    from terse.policy import load_policy
    # Long repeated values, so the default codec's dictionary tier fires and the
    # tabularize-only policy's form differs from it.
    statuses = ["awaiting-triage-from-maintainer", "blocked-on-upstream-dependency"]
    teams = ["platform-infrastructure-team", "developer-experience-team"]
    payloads = [{"result": [{"id": i + k * 100, "status": statuses[i % 2],
                             "team": teams[(i // 2) % 2], "meta": {"n": i}}
                            for i in range(12)]} for k in range(2)]
    corpus = tmp_path / "c"
    corpus.mkdir()
    for pl in payloads:
        capture_payload("kb.read.x", json.dumps(pl), corpus, server="kb", manual=True)
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "kb.*"}, "tiers": ["minify", "tabularize"]}]}))
    pol = load_policy(str(policy))
    expected_terse = [policy_apply(json.dumps(pl), "kb.read.x", pol, server="kb",
                                   force_lossless=True).text for pl in payloads]
    assert all("__terse_table__" in t and "__terse_dict__" not in t for t in expected_terse)
    assert all(t != fluency.compress(pl) for t, pl in zip(expected_terse, payloads, strict=True)), (
        "fixture cannot tell the policy's form from the default codec's")
    seen: list[list[dict]] = []

    def fake_text(alias):
        ans, log = _text_answerer(lambda m: "[]")
        seen.append(log)  # type: ignore[arg-type]
        return ans

    # The pre-flight is skipped here: its subject is the sweep's payload, not admission.
    monkeypatch.setattr(codeceval, "preflight_encoding", lambda *a, **k: None)
    monkeypatch.setattr(codeceval, "cli_text_answerer", fake_text)
    main(["fluency", "--codec-verdict", "--primer", "--policy", str(policy), "--corpus",
          str(corpus), "--trials", "1", "--models", "cli:haiku", "--out", str(tmp_path / "r")])
    users = [m[-1]["content"] for m in seen[0]]
    assert any(t in u for u in users for t in expected_terse), (
        "the terse arm was not compressed under --policy")
    assert not any(fluency.compress(pl) in u for u in users for pl in payloads)


def _dict_payloads():
    statuses = ["awaiting-triage-from-maintainer", "blocked-on-upstream-dependency"]
    teams = ["platform-infrastructure-team", "developer-experience-team"]
    return [{"result": [{"id": i + k * 100, "status": statuses[i % 2],
                         "team": teams[(i // 2) % 2], "meta": {"n": i}}
                        for i in range(12)]} for k in range(2)]


def _envs(payloads):
    return [{"tool": "kb.read.x", "server": "kb", "raw": json.dumps(pl, indent=2),
             "sha": f"{k:040d}", "shape": "array-of-records", "manual": True}
            for k, pl in enumerate(payloads)]


def _policy(tiers):
    from terse.policy import Policy, Rule
    return Policy(rules=[Rule(tool_glob="kb.*", tiers=tuple(tiers))])


def test_R3_a_payload_the_policy_leaves_alone_is_not_asked():
    ans, _ = _text_answerer(_expected)
    run = codeceval.run_codec_fluency(_envs(PAYLOADS), {"m": ans}, trials=4,
                                      preflight=False, primer="PRIMER", policy=_policy([]))
    assert run.rows.get("m", []) == [] and run.skipped_unaskable == len(PAYLOADS)


def test_R3_savings_and_the_input_limit_use_the_policy_form():
    from terse.policy import apply as policy_apply
    from terse.tokenize import count_cl100k
    if count_cl100k("x") is None:
        import pytest
        pytest.skip("no tokenizer")
    payloads, pol = _dict_payloads(), _policy(["minify", "tabularize"])
    envs = _envs(payloads)
    policy_form = [policy_apply(e["raw"], "kb.read.x", pol, server="kb",
                                force_lossless=True).text for e in envs]
    default_form = [fluency.compress(pl) for pl in payloads]
    assert policy_form != default_form
    ans, _ = _text_answerer(lambda m: "[]")
    run = codeceval.run_codec_fluency(envs, {"m": ans}, trials=1, preflight=False,
                                      policy=pol)
    tokens = {r["sha"]: r["terse_tokens"] for r in run.rows["m"]}
    assert sorted(tokens.values()) == sorted(count_cl100k(t) for t in policy_form)
    # The input-limit check sizes the terse arm from the POLICY form: at a limit nothing
    # fits under, the terse exclusion reports the policy form's largest request, which
    # differs from the default codec's (the raw arm is always larger, so no limit can sit
    # between the two forms — the reported size is what pins the call site).
    qs = codeceval.gen_codec_questions(payloads[0])
    want = max(codeceval.request_tokens(q, policy_form[0]) or 0 for q in qs)
    other = max(codeceval.request_tokens(q, default_form[0]) or 0 for q in qs)
    assert want != other
    run2 = codeceval.run_codec_fluency(envs[:1], {"m": ans}, trials=1, preflight=False,
                                       policy=pol, limits={"m": 1})
    terse_ex = [e for e in run2.excluded if e.arm == "terse"]
    assert [e.tokens for e in terse_ex] == [want]
