"""#582 item 8: one malformed payload or file never aborts a whole command. A >4300-digit
integer raises a plain ValueError, deep nesting a RecursionError, a torn multibyte write a
UnicodeDecodeError; readers used to catch only JSONDecodeError."""
from __future__ import annotations

import json

import pytest

from terse import transforms

HUGE_INT = '{"n": ' + "9" * 5000 + "}"
DEEP = "[" * 800 + "]" * 800
VERY_DEEP = "[" * 100_000 + "]" * 100_000


@pytest.mark.parametrize("raw", [HUGE_INT, VERY_DEEP, "nope", None])
def test_parsing_a_hostile_payload_raises_only_parse_errors(raw):
    with pytest.raises(transforms.PARSE_ERRORS):
        json.loads(raw)


def _env(raw, sha="s1", tool="t.get"):
    return {"raw": raw, "sha": sha, "tool": tool, "shape": "records"}


def test_load_corpus_skips_bad_utf8_and_deep_files(tmp_path):
    from terse.capture import load_corpus
    (tmp_path / "bad.json").write_bytes(b'{"raw": "\xff"}')
    (tmp_path / "deep.json").write_text(VERY_DEEP, encoding="utf-8")
    (tmp_path / "ok.json").write_text(json.dumps(_env('{"a":1}')), encoding="utf-8")
    assert [e["sha"] for e in load_corpus(tmp_path)] == ["s1"]


def test_load_history_survives_bad_bytes_and_non_dict_lines(tmp_path):
    from terse.history import load_history
    p = tmp_path / "h.jsonl"
    p.write_bytes(b'{"ts": 1}\n\xff\xfe\n[1]\n5\n{"ts": 2}\n')
    assert [r["ts"] for r in load_history(p)] == [1, 2]


def test_load_stats_survives_bad_bytes_and_huge_ints(tmp_path):
    from terse.stats import load_stats
    p = tmp_path / "stats.jsonl"
    p.write_bytes(b'{"ts": 1, "tool": "a"}\n\xff\n' + HUGE_INT.encode() + b'\n{"ts": 2}\n')
    assert [r["ts"] for r in load_stats(p)] == [1, 2]


def test_cross_tokenizer_savings_passes_a_deep_payload_through():
    from terse.measure import cross_tokenizer_savings
    rows = cross_tokenizer_savings([_env(DEEP)])
    assert len(rows) == 1


@pytest.mark.parametrize("raw", [HUGE_INT, DEEP])
def test_fluency_pack_and_harness_skip_unparseable_payloads(raw):
    from terse.fluency.harnesses import build_chain_windows, diff_pairs
    from terse.fluency.pack import build_pack
    envs = [_env(raw, "a"), _env(raw, "b")]
    build_pack(envs)
    diff_pairs(envs)
    build_chain_windows(envs)


def test_run_fluency_skips_an_unparseable_payload():
    from terse.fluency.harnesses import run_fluency
    run_fluency([_env(HUGE_INT)], {"m": lambda system, user: "1"})


# --------------------------------------------------------------------------- #
# Review round 1 of #591: B1-B9.
# --------------------------------------------------------------------------- #
def _records(salt: int = 0, depth: int = 0, n: int = 8) -> list[dict]:
    """A record list the question generators ask about. `depth` > 0 hangs one leaf
    `depth` lists deep off the first record."""
    recs = [{"id": i, "name": f"n{i}", "v": i * 3 + salt,
             "evidence": "lorem ipsum dolor " * 4 + str(i)} for i in range(n)]
    if depth:
        leaf: list = []
        for _ in range(depth):
            leaf = [leaf]
        recs[0]["blob"] = leaf
    return recs


# 250 levels is past MAX_DEPTH (200) but well inside what the codec's and the question
# generator's recursive walks handle; 600 is past what they handle (RecursionError).
LEAF_250 = [json.dumps(_records(s, 250)) for s in range(3)]
LEAF_600 = [json.dumps(_records(s, 600)) for s in range(3)]
HOSTILE_REPLIES = ["[" + "9" * 5000 + "]", VERY_DEEP, "[" * 3000 + "]" * 3000,
                   '{"a":' * 100_000]
REPLY_IDS = ["huge_int", "deep_100k", "deep_3000", "deep_object"]


def _envs(raws, tool="srv.t"):
    return [{"raw": r, "sha": f"s{i}", "tool": tool, "server": "srv", "shape": "x"}
            for i, r in enumerate(raws)]


# B1: a model's reply is untrusted input to the scorer.
@pytest.mark.parametrize("reply", HOSTILE_REPLIES, ids=REPLY_IDS)
@pytest.mark.parametrize("qtype,expected", [("enumerate", ["a"]), ("deref", {"a": 1})])
def test_score_reads_an_unparseable_reply_as_a_miss(qtype, expected, reply):
    from terse.fluency.scoring import score
    assert score(qtype, expected, reply) is False


@pytest.mark.parametrize("reply", HOSTILE_REPLIES, ids=REPLY_IDS)
def test_run_fluency_scores_a_hostile_reply_and_keeps_the_rows(reply):
    from terse.fluency.harnesses import run_fluency
    from terse.fluency.questions import gen_questions
    recs = _records()
    out = run_fluency([_env(json.dumps(recs))], {"m": lambda system, user: reply})
    assert len(out["m"]) == len(gen_questions(recs)) > 0


@pytest.mark.parametrize("reply", HOSTILE_REPLIES, ids=REPLY_IDS)
def test_drop_eval_question_survives_a_hostile_final_reply(reply):
    from terse.dropeval import DropQuestion, ToolCall, Turn, _run_question
    q = DropQuestion(qid="q1", prompt="p", instruction="i", expected={"a": 1},
                     kind="recall", needs_retrieve=True, expected_handle="h1")
    turns = iter([Turn(text="", tool_calls=[ToolCall("c1", "terse.retrieve", {"handle": "h1"})]),
                  Turn(text=reply)])
    assert len(_run_question(q, "applied", {"h1": {"a": 1}}, lambda m: next(turns))) == 4


# B2: `policy generate`, `tune` and `probe` read the same corpus payloads.
def test_policy_drop_candidates_skip_a_payload_too_deep_to_parse():
    from terse.policy_gen import _drop_candidates
    _drop_candidates([VERY_DEEP, HUGE_INT, json.dumps(_records())])


def _write_corpus(d, raws):
    d.mkdir()
    for i, raw in enumerate(raws):
        srv = f"srv{i % 2}"   # two servers, so `probe --cross-server` has peers to compare
        env = {"tool": f"{srv}.list", "server": srv, "raw": raw, "sha": f"s{i}",
               "shape": "x", "bytes": len(raw), "captured_at": i + 1}
        (d / f"e{i}.json").write_text(json.dumps(env), encoding="utf-8")
    return d


@pytest.mark.parametrize("argv", [["probe"], ["probe", "--cross-server"],
                                  ["policy", "generate"], ["tune"]])
def test_corpus_commands_complete_on_a_payload_too_deep_to_parse(tmp_path, argv):
    from terse.cli import main
    corpus = _write_corpus(tmp_path / "c", [json.dumps(_records(1)), json.dumps(_records(2)),
                                            VERY_DEEP, HUGE_INT])
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    extra = ["--ledger", str(ledger)] if argv == ["tune"] else []
    assert main([*argv, "--corpus", str(corpus), "--out", str(tmp_path / "out"), *extra]) == 0


# B3: an unparseable PRIOR envelope is treated like a missing one and overwritten.
@pytest.mark.parametrize("prior", [b'{"captured_at": 1, "raw": "\xff"}', HUGE_INT.encode(),
                                   VERY_DEEP.encode()], ids=["bad_utf8", "huge_int", "deep"])
def test_capture_overwrites_an_unparseable_prior_envelope(tmp_path, prior):
    from terse.capture import capture_payload
    path = capture_payload("t.get", '{"a": 1}', tmp_path)
    path.write_bytes(prior)
    assert capture_payload("t.get", '{"a": 1}', tmp_path) == path
    assert json.loads(path.read_text(encoding="utf-8"))["raw"] == '{"a": 1}'


# B4: no fixed depth screen. A payload past MAX_DEPTH that the walks handle is evaluated
# exactly as before #591; one they cannot handle is skipped and SAID to be too deep.
def test_a_valid_payload_past_max_depth_is_evaluated_as_before():
    from terse import codeceval, dropeval, measure, transforms
    from terse import policy as policy_mod
    from terse.dropeval import Turn
    from terse.fluency import harnesses, pack
    from terse.fluency.questions import gen_questions
    envs = _envs(LEAF_250)
    obj = json.loads(LEAF_250[0])
    assert transforms.exceeds_depth(obj)
    n_q = len(gen_questions(obj))
    assert n_q > 0
    assert len(harnesses.run_fluency(envs[:1], {"m": lambda s, u: "1"})["m"]) == n_q
    built = pack.build_pack(envs[:1])
    assert len(built["payloads"]) == 1 and "skipped_too_deep" not in built
    assert len(pack.token_summary(envs[:1])) == 1
    assert len(harnesses.diff_pairs(envs)) == 2
    assert harnesses.run_diff_fluency(envs, {"m": lambda s, u: "1"})["m"]
    assert harnesses.build_chain_windows(envs)
    run = codeceval.run_codec_fluency(envs[:1], {"m": lambda m: Turn(text="1")},
                                      preflight=False)
    assert run.rows["m"] and run.skipped_unaskable == 0
    rule = policy_mod.Rule(tool_glob="*", tiers=("minify", "table"),
                           fields={"[].evidence": {"lossy": "drop-to-retrieve", "min": 10}})
    probe = dropeval._probe_envelope(envs[0], lambda t, s: rule)[0]
    assert probe.reason == "not_compressible" and "deeper than" in probe.detail
    row = measure.cross_tokenizer_savings(envs[:1])[0]
    comp = transforms.compress(transforms.loads_faithful(LEAF_250[0]))
    assert row["cl100k_base"]["compressed"] < row["cl100k_base"]["raw"]
    assert row["cl100k_base"]["compressed"] == measure.count(comp, "cl100k_base")


def test_a_payload_too_deep_for_the_codec_is_skipped_and_said_so():
    from terse import codeceval, measure
    from terse.dropeval import Turn
    from terse.fluency import harnesses, pack
    envs = _envs(LEAF_600)
    lines: list[str] = []
    assert harnesses.run_fluency(envs[:1], {"m": lambda s, u: "1"},
                                 progress=lines.append)["m"] == []
    assert any("too deep" in line for line in lines), lines
    built = pack.build_pack(envs[:1])
    assert built["payloads"] == [] and built["skipped_too_deep"] == 1
    assert pack.token_summary(envs[:1]) == []
    lines.clear()
    assert harnesses.run_diff_fluency(envs, {"m": lambda s, u: "1"},
                                      progress=lines.append)["m"] == []
    assert any("too deep" in line for line in lines), lines
    harnesses.run_diff_soak(envs, {"m": lambda s, u: "1"})
    lines.clear()
    run = codeceval.run_codec_fluency(envs[:1], {"m": lambda m: Turn(text="1")},
                                      preflight=False, progress=lines.append)
    assert run.skipped_unaskable == 1 and not run.rows["m"]
    assert any("too deep" in line for line in lines), lines
    row = measure.cross_tokenizer_savings(envs[:1])[0]
    assert row["cl100k_base"]["pct"] is None and "too deep" in row["skipped"]


def test_a_payload_too_deep_to_parse_is_reported_as_too_deep():
    from terse import dropeval, measure
    from terse import policy as policy_mod
    from terse.fluency import harnesses, pack
    from terse.report import build_tokenizer_report
    envs = _envs([VERY_DEEP])
    row = measure.cross_tokenizer_savings(envs)[0]
    assert row["cl100k_base"]["pct"] is None and "too deep" in row["skipped"]
    assert "too deep" in build_tokenizer_report([row])
    rule = policy_mod.Rule(tool_glob="*", tiers=("minify", "table"),
                           fields={"[].evidence": {"lossy": "drop-to-retrieve", "min": 10}})
    probe = dropeval._probe_envelope(envs[0], lambda t, s: rule)[0]
    assert probe.reason == "not_compressible" and "too deep" in probe.detail
    assert pack.build_pack(envs)["skipped_too_deep"] == 1
    lines: list[str] = []
    harnesses.run_fluency(envs, {"m": lambda s, u: "1"}, progress=lines.append)
    assert any("too deep" in line for line in lines), lines


# B5: a backend that sends `function.arguments` already parsed.
class _Resp:
    def __init__(self, body: dict):
        self._b = json.dumps(body).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _tool_reply(arguments):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "terse_retrieve", "arguments": arguments}}]}}]}


def _answerer(monkeypatch, arguments):
    from terse import dropeval
    from terse.proxy import RETRIEVE_TOOL_DEF

    class _Opener:
        def open(self, req, timeout=None):
            return _Resp(_tool_reply(arguments))

    monkeypatch.setattr(dropeval, "guarded_opener", lambda url: _Opener())
    return dropeval.openai_tool_answerer("http://127.0.0.1:1/v1", "", "m",
                                         tools=[RETRIEVE_TOOL_DEF])


def test_tool_call_arguments_sent_as_an_object_are_used_as_is(monkeypatch):
    ask = _answerer(monkeypatch, {"handle": "h1"})
    assert ask([{"role": "user", "content": "q"}]).tool_calls[0].arguments == {"handle": "h1"}


def test_tool_call_arguments_of_another_type_stay_a_harness_error(monkeypatch):
    from terse import dropeval
    ask = _answerer(monkeypatch, ["h1"])
    assert dropeval._safe_call(ask, [{"role": "user", "content": "q"}]).error


# B6: an unreadable corpus file is skipped with an accurate reason.
def test_load_corpus_skips_unreadable_files_and_names_each_reason(tmp_path, capsys):
    import os

    from terse.capture import load_corpus
    (tmp_path / "ok.json").write_text(json.dumps(_env('{"a":1}')), encoding="utf-8")
    (tmp_path / "corrupt.json").write_text("{nope", encoding="utf-8")
    (tmp_path / "bad.json").write_bytes(b'{"raw": "\xff"}')
    (tmp_path / "deep.json").write_text(VERY_DEEP, encoding="utf-8")
    (tmp_path / "huge.json").write_text(HUGE_INT, encoding="utf-8")
    (tmp_path / "dir.json").mkdir()
    locked = tmp_path / "locked.json"
    locked.write_text(json.dumps(_env('{"a":2}', sha="s2")), encoding="utf-8")
    os.chmod(locked, 0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running as a user the file mode does not lock out (root)")
        assert [e["sha"] for e in load_corpus(tmp_path)] == ["s1"]
    finally:
        os.chmod(locked, 0o600)
    err = capsys.readouterr().err
    assert "skipped 6" in err
    for reason in ("1 corrupt JSON", "1 not UTF-8", "1 nested too deep",
                   "1 integer past the digit limit", "2 unreadable file"):
        assert reason in err, err


# B7: a line with one invalid byte is dropped, not decoded into a different record.
def test_load_stats_and_history_drop_a_line_with_an_invalid_byte(tmp_path):
    from terse.history import load_history
    from terse.stats import load_stats
    p = tmp_path / "l.jsonl"
    p.write_bytes(b'{"ts": 1, "tool": "srv.a"}\n{"ts": 3, "tool": "srv.caf\xff"}\n'
                  b'{"ts": 2, "tool": "srv.b"}\n')
    assert [r["ts"] for r in load_stats(p)] == [1, 2]
    assert [r["ts"] for r in load_history(p)] == [1, 2]


# B8: only the PARSE is contained; a fault in the codec itself is not scored as 0%.
def test_cross_tokenizer_savings_does_not_hide_a_codec_fault(monkeypatch):
    from terse import measure, transforms

    def broken(obj):
        raise TypeError("codec bug")

    monkeypatch.setattr(transforms, "compress", broken)
    with pytest.raises(TypeError, match="codec bug"):
        measure.cross_tokenizer_savings([_env('{"a": [1, 2]}')])


# B9: a non-string handle misses with production's wording (`proxy`: handle '').
def test_a_non_string_retrieve_handle_misses_like_production():
    from terse.dropeval import DropQuestion, ToolCall, Turn, _run_question
    q = DropQuestion(qid="q1", prompt="p", instruction="i", expected={"a": 1},
                     kind="recall", needs_retrieve=True, expected_handle="h1")
    seen: list = []

    def ans(messages):
        seen.append(list(messages))
        if len(seen) == 1:
            return Turn(text="", tool_calls=[ToolCall("c1", "terse.retrieve", {"handle": ["x"]})])
        return Turn(text='{"a": 1}')

    _run_question(q, "applied", {"h1": {"a": 1}}, ans)
    tool_msg = [m for m in seen[-1] if m.get("role") == "tool"][0]
    assert "handle ''" in tool_msg["content"]
