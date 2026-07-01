"""Tier 0.7 text: content-defined-chunking diff for non-JSON tool output (#25). Every
accepted diff must rebuild curr exactly; the codec returns None when nothing
representable applies (the caller then sends the raw text)."""

from __future__ import annotations

from terse import text_diff as TD


def _log(n, changed_line=None, extra=""):
    lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}" for i in range(n)]
    if changed_line is not None:
        lines[changed_line] = "[ERROR] worker crashed: connection reset"
    return "\n".join(lines) + extra


def test_identical_text_yields_diff_that_roundtrips_to_a_single_copy_run():
    text = _log(60)
    diff = TD.text_diff_encode(text, text)
    assert diff is not None and diff[TD.DIFF_MARKER] == 1
    assert TD.text_diff_decode(text, diff) == text
    # unchanged text should collapse to one contiguous copy run, not per-chunk ops
    copy_ops = [op for op in diff["ops"] if op[0] == "="]
    assert len(copy_ops) == 1


def test_single_changed_line_roundtrips():
    prev = _log(80)
    curr = _log(80, changed_line=40)
    diff = TD.text_diff_encode(prev, curr)
    assert diff is not None
    assert TD.text_diff_decode(prev, diff) == curr


def test_appended_lines_roundtrip():
    prev = _log(50)
    curr = prev + "\n[0050] worker heartbeat ok, queue_depth=0\n[0051] worker heartbeat ok, queue_depth=1"
    diff = TD.text_diff_encode(prev, curr)
    assert diff is not None
    assert TD.text_diff_decode(prev, diff) == curr


def test_prepended_text_still_roundtrips_even_though_all_chunks_shift():
    prev = _log(50)
    curr = "=== run started ===\n" + prev
    diff = TD.text_diff_encode(prev, curr)
    assert diff is not None
    assert TD.text_diff_decode(prev, diff) == curr


def test_completely_different_text_either_falls_back_or_still_roundtrips():
    prev = _log(30)
    curr = "totally unrelated content " * 20
    diff = TD.text_diff_encode(prev, curr)
    assert diff is None or TD.text_diff_decode(prev, diff) == curr


def test_empty_curr_roundtrips():
    prev = _log(20)
    diff = TD.text_diff_encode(prev, "")
    assert diff is not None
    assert TD.text_diff_decode(prev, diff) == ""


def test_empty_prev_has_nothing_to_reference():
    assert TD.text_diff_encode("", "some new text") is None


def test_short_text_below_min_chunk_still_roundtrips():
    assert TD.text_diff_roundtrip_ok("hi", "hi there")
    assert TD.text_diff_roundtrip_ok("same", "same")


def test_multibyte_characters_never_split_a_chunk_mid_character():
    # a chunk cut inside a multi-byte character would produce a str that can't even
    # exist in Python, so this is really testing _chunk operates on code points.
    prev = ("日本語のログ出力です。" * 20) + "normal ascii tail " * 10
    curr = prev.replace("ログ出力です。", "エラー発生しました。", 1)
    diff = TD.text_diff_encode(prev, curr)
    assert diff is not None
    assert TD.text_diff_decode(prev, diff) == curr


def test_roundtrip_gate_helper():
    prev, curr = _log(40), _log(40, changed_line=10)
    assert TD.text_diff_roundtrip_ok(prev, curr)
    assert not TD.text_diff_roundtrip_ok("", "brand new content, nothing to reference")


def test_diff_is_smaller_than_curr_for_a_mostly_unchanged_repeated_file():
    prev = _log(200)
    curr = _log(200, changed_line=100)
    wire = TD.text_diff_wire(prev, curr, tool="fs.read")
    assert wire is not None
    assert len(wire) < len(curr)


def test_wire_envelope_is_valid_json_and_carries_tool_and_base():
    import json
    prev, curr = _log(30), _log(30, changed_line=5)
    wire = TD.text_diff_wire(prev, curr, tool="fs.read")
    env = json.loads(wire)
    assert env[TD.DIFF_MARKER] == 1
    assert env["of"] == "fs.read"
    assert "base" in env and "note" in env
    assert TD.text_diff_decode(prev, env) == curr
