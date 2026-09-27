"""Offline tests for mcp_share.py against synthetic transcript fixtures --
no real claude -p transcript needed."""
from __future__ import annotations

import sys
from pathlib import Path

# The flat cost_per_task scripts are importable by bare module name (`arms`, `runner`,
# ...). Done here, not in a conftest.py: a second `conftest` module shadowed
# tests/conftest.py in a whole-repo run (`from conftest import ...` there then failed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import mcp_share as ms


def _write_transcript(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_mcp_share_basic_ratio(tmp_path):
    mcp_result = "A" * 40
    other_text = "B" * 60
    records = [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "t1", "name": "mcp__terse__kb_read_get"},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": mcp_result},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": other_text},
        ]}},
    ]
    path = tmp_path / "t.jsonl"
    _write_transcript(path, records)

    share = ms.mcp_share(path)

    expected_mcp_chars = ms._content_len(mcp_result)
    expected_total_chars = sum(ms._content_len(r["message"]["content"]) for r in records)
    assert share == expected_mcp_chars / expected_total_chars
    assert 0 < share < 1


def test_mcp_share_ignores_non_mcp_tool(tmp_path):
    records = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Read"},  # built-in, not mcp__
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "C" * 100},
        ]}},
    ]
    path = tmp_path / "t.jsonl"
    _write_transcript(path, records)
    assert ms.mcp_share(path) == 0.0


def test_mcp_share_empty_transcript_is_zero(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("")
    assert ms.mcp_share(path) == 0.0


def test_mcp_share_all_mcp_is_close_to_one(tmp_path):
    records = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__terse__kb_read_get"},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "D" * 1000},
        ]}},
    ]
    path = tmp_path / "t.jsonl"
    _write_transcript(path, records)
    assert ms.mcp_share(path) > 0.9


def test_mcp_share_skips_malformed_json_lines_without_crashing(tmp_path):
    path = tmp_path / "t.jsonl"
    good = json.dumps({"type": "assistant",
                        "message": {"content": [{"type": "text", "text": "hi"}]}})
    path.write_text("not json\n" + good + "\n")
    assert ms.mcp_share(path) == 0.0


def test_mcp_share_unmatched_tool_result_id_not_counted(tmp_path):
    """A tool_result whose tool_use_id doesn't match any recorded mcp__
    tool_use (e.g. the matching tool_use line was truncated/missing) must
    not be counted as MCP share -- absence of proof isn't proof of MCP."""
    records = [
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "unknown-id", "content": "Z" * 500},
        ]}},
    ]
    path = tmp_path / "t.jsonl"
    _write_transcript(path, records)
    assert ms.mcp_share(path) == 0.0


# ----------------------------------------------------------- #325: primer_attached

def _primer_text():
    from terse.proxy import PRIMER_HEAD
    return PRIMER_HEAD + "- Table {...}\n"


def _result_record(content):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": content}]}}


def test_primer_attached_true_for_a_text_block_primer(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [_result_record([
        {"type": "text", "text": _primer_text()}, {"type": "text", "text": "{\"a\":1}"}])])
    assert ms.primer_attached(path) is True


def test_primer_attached_true_for_a_typed_wrapper_primer(tmp_path):
    wrapped = json.dumps({"__terse_primer__": _primer_text(), "__terse_payload__": {"a": 1}})
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [_result_record(wrapped)])
    assert ms.primer_attached(path) is True


def test_primer_attached_false_without_primer_or_for_empty_wrapper(tmp_path):
    empty = json.dumps({"__terse_primer__": "", "__terse_payload__": {"a": 1}})
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [_result_record(empty), _result_record("plain result")])
    assert ms.primer_attached(path) is False


def test_primer_attached_ignores_the_head_outside_tool_results(tmp_path):
    # e.g. the model quoting the primer back, or a doc in the prompt -- not an attach.
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [{"type": "assistant", "message": {"content": [
        {"type": "text", "text": _primer_text()}]}}])
    assert ms.primer_attached(path) is False


def test_primer_attached_true_when_only_in_an_offloaded_result(tmp_path):
    path = tmp_path / "sess-1.jsonl"
    _write_transcript(path, [_result_record("Output too large, saved to file")])
    off = tmp_path / "sess-1" / "tool-results"
    off.mkdir(parents=True)
    (off / "r1.txt").write_text(_primer_text())
    assert ms.primer_attached(path) is True


def test_primer_attached_none_without_a_transcript(tmp_path):
    assert ms.primer_attached(None) is None
    assert ms.primer_attached(tmp_path / "missing.jsonl") is None
