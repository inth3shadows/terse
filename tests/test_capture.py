"""Tests for corpus capture persistence, including the permission hardening on
`capture_payload`/`append_audit` — both can persist real MCP tool traffic to disk."""
from __future__ import annotations

import json
import stat

from terse.capture import (
    append_audit,
    capture_payload,
    find_record_list_with_path,
    load_corpus,
)


def test_capture_payload_writes_owner_only_file(tmp_path):
    raw = json.dumps({"id": 1})
    path = capture_payload("demo.tool", raw, tmp_path / "corpus")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["raw"] == raw


def test_capture_payload_is_idempotent_by_sha_and_stays_restricted(tmp_path):
    raw = json.dumps({"id": 1})
    corpus = tmp_path / "corpus"
    p1 = capture_payload("demo.tool", raw, corpus)
    p2 = capture_payload("demo.tool", raw, corpus)
    assert p1 == p2
    assert stat.S_IMODE(p1.stat().st_mode) == 0o600


def test_load_corpus_replays_in_capture_order_not_filename(tmp_path):
    # capture_payload stamps a monotonic captured_at; load_corpus must return that order,
    # even when the sha-based filenames sort the other way (the #64 session replay depends
    # on it). "zzz" is captured FIRST but its filename sorts LAST.
    corpus = tmp_path / "corpus"
    capture_payload("t", json.dumps({"v": "zzz"}), corpus)
    capture_payload("t", json.dumps({"v": "aaa"}), corpus)
    order = [json.loads(e["raw"])["v"] for e in load_corpus(corpus)]
    assert order == ["zzz", "aaa"]                       # capture order, not sorted filename


def test_capture_payload_preserves_captured_at_on_rewrite(tmp_path):
    # Re-capturing identical content must keep the FIRST-sighting timestamp (idempotency).
    corpus = tmp_path / "corpus"
    p = capture_payload("t", json.dumps({"v": 1}), corpus)
    first = json.loads(p.read_text(encoding="utf-8"))["captured_at"]
    capture_payload("t", json.dumps({"v": 1}), corpus)
    assert json.loads(p.read_text(encoding="utf-8"))["captured_at"] == first


def test_append_audit_writes_owner_only_and_appends(tmp_path):
    log = tmp_path / "debug.jsonl"
    append_audit({"id": 1}, log)
    append_audit({"id": 2}, log)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["id"] for ln in lines] == [1, 2]


def test_find_record_list_with_path_returns_expressible_drop_path():
    recs = [{"a": 1}, {"a": 2}]
    assert find_record_list_with_path({"result": recs}) == (recs, "result[]")
    assert find_record_list_with_path(recs) == (recs, "[]")                 # top-level list
    assert find_record_list_with_path({"data": {"items": recs}}) == (recs, "data.items[]")


def test_find_record_list_with_path_none_when_no_simple_path():
    assert find_record_list_with_path({"x": 1}) == (None, None)             # no record list
    assert find_record_list_with_path([1, 2, 3]) == (None, None)           # list of scalars
    # a record list nested inside another list has no simple field path -> not returned
    assert find_record_list_with_path([[{"a": 1}, {"a": 2}]]) == (None, None)


def test_envelope_records_server_and_result_id(tmp_path):
    p = capture_payload("structure", '{"a":1}', tmp_path / "c",
                        server="runecho", result_id="ab12cd34:7")
    env = json.loads(p.read_text())
    assert env["server"] == "runecho"
    assert env["result_id"] == "ab12cd34:7"


def test_unknown_server_and_result_are_omitted_not_nulled(tmp_path):
    # One spelling of "nothing": consumers check for absence, never for a null that means
    # the same thing.
    env = json.loads(capture_payload("t", '{"a":1}', tmp_path / "c").read_text())
    assert "server" not in env and "result_id" not in env


def test_recapture_preserves_the_first_result_id_with_its_timestamp(tmp_path):
    # The corpus is idempotent by sha and an envelope describes a payload's FIRST sighting.
    # Keeping a later result id while keeping the earlier timestamp would leave the grouping
    # key and the clock disagreeing about which call this envelope stands for.
    corpus = tmp_path / "c"
    first = json.loads(capture_payload("t", '{"a":1}', corpus, result_id="s:1").read_text())
    again = json.loads(capture_payload("t", '{"a":1}', corpus, result_id="s:9").read_text())
    assert again["result_id"] == "s:1" == first["result_id"]
    assert again["captured_at"] == first["captured_at"]


def test_qualified_tool_mirrors_the_runtime_lookup():
    from terse.capture import qualified_tool
    from terse.policy import Policy

    # Each case must equal the FIRST candidate `select` tries, or a rule authored under the
    # returned name is unreachable.
    for env, tool, server in [
        ({"tool": "structure", "server": "runecho"}, "structure", "runecho"),
        ({"tool": "kb.read.search", "server": "kb"}, "kb.read.search", "kb"),
        ({"tool": "peer__structure", "server": "peer"}, "peer__structure", "peer"),
    ]:
        assert qualified_tool(env) == Policy._match_candidates(tool, server)[0]

    assert qualified_tool({"tool": "structure"}) == "structure"          # no server recorded
    assert qualified_tool({"tool": "structure", "server": ""}) == "structure"
