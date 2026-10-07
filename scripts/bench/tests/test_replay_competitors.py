"""Tests for the competitor replay (scripts/bench/replay_competitors.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pytest
import replay_competitors as rc
import replay_sessions as rs

from terse import transforms
from terse.tokenize import count_cl100k

RECORDS = {"result": [{"id": i, "kind": "node", "state": "active"} for i in range(30)]}


def test_rebuild_returns_an_untouched_result_as_it_was_shown():
    text = json.dumps(RECORDS, indent=2)
    assert rc.rebuild(text) == (text, RECORDS)
    assert rc.rebuild("plain words, not JSON") == ("plain words, not JSON", None)
    # Prose that happens to end in JSON is the raw result, not a primer and a payload.
    tail = 'the answer is {"a": 1}'
    assert rc.rebuild(tail) == (tail, None)


def test_rebuild_decodes_what_terse_changed_back_to_the_minified_raw_result():
    shown = transforms.compress(RECORDS)
    assert shown != transforms.minify(RECORDS)
    assert rc.rebuild(shown) == (transforms.minify(RECORDS), RECORDS)


def test_rebuild_unwraps_the_typed_primer_whether_or_not_the_payload_was_changed():
    for payload in (json.loads(transforms.compress(RECORDS)), RECORDS):
        shown = transforms.minify({rc.PRIMER_KEY: "how to read this", rc.PAYLOAD_KEY: payload})
        assert rc.rebuild(shown) == (transforms.minify(RECORDS), RECORDS)


def test_rebuild_takes_a_text_primer_off_the_front():
    shown = "Some tool results are terse-compressed {like this}.\n" + transforms.compress(RECORDS)
    assert rc.rebuild(shown) == (transforms.minify(RECORDS), RECORDS)


def test_rebuild_gives_up_on_a_diff_and_on_a_value_that_still_carries_a_marker():
    assert rc.rebuild('{"__terse_diff__":1,"set":{"a":1},"del":[]}') == (None, None)
    assert rc.rebuild('{"__terse_textdiff__":1,"ops":[]}') == (None, None)
    # A changed text result whose blocks were dropped is not JSON: it comes back as shown,
    # and the size gate is what rejects it.
    dropped = "primer\nfile a.py\n[code omitted]"
    assert rc.rebuild(dropped) == (dropped, None)


def test_verified_allows_half_a_percent_or_three_tokens():
    assert rc.verified(1000, 1005) and not rc.verified(1000, 1006)
    assert rc.verified(10, 13) and not rc.verified(10, 14)


def test_weigh_sums_scale_times_carry_over_every_transcript_that_carried_the_result(tmp_path):
    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps({
        "ts": 100, "server": "kb", "tool": "kb.read.search", "decision": "compressed",
        "raw_chars": 4000, "out_chars": 2400, "raw_tokens": 1000, "out_tokens": 600}) + "\n")
    led = rs.load_ledger(path)[0]
    res = rs.Result("a", "mcp__terse__kb_read_search", 100.5, 2400, 650, False, "{}")
    opus = "claude-opus-5-5"

    def call(read=0, w1=0, billed=True):
        return rs.Call(opus, 0, read, 0, w1, 0, billed)

    original = [("call", call(w1=100)), ("result", res), ("call", call(read=100, w1=700))]
    resumed = [("call", call(w1=100, billed=False)), ("result", res),
               ("call", call(read=100, w1=700, billed=False)), ("call", call(read=800, w1=50))]
    got = rc.weigh([(path, original), (path, resumed)], {"a": led}, {opus: 1.5})
    # 2.0 for the write in the first transcript, one 0.05 read in the resumed one.
    assert got["a"]["weight"] == pytest.approx(1.5 * (2.0 + 0.05))
    assert got["a"]["ledger_raw"] == 1000 and got["a"]["shown_tokens"] == 650
    assert got["a"]["changed"] is True


def test_encoder_tokens_passes_through_what_an_encoder_could_not_handle():
    raw = transforms.minify(RECORDS)
    n = count_cl100k(raw)
    sizes, flags = rc.encoder_tokens(n, RECORDS, n, 80, {"toon": "x", "lossless": False},
                                     {"error": "boom"})
    assert sizes["toon"] == n and flags["toon_passthrough"]
    assert sizes["headroom_default"] == sizes["headroom_agent"] == n
    assert sizes["headroom_verified"] == n
    assert flags["headroom_error"] and not flags["headroom_dropped"]
    assert sizes["terse_shipped"] == 80
    assert sizes["terse_codec"] == count_cl100k(transforms.compress(RECORDS)) < n


def test_encoder_tokens_counts_each_output_and_flags_a_drop():
    raw = transforms.minify(RECORDS)
    n = count_cl100k(raw)
    sizes, flags = rc.encoder_tokens(
        n, RECORDS, n, 80, {"toon": "a b c", "lossless": True},
        {"default": "one two", "agent": "one two three", "verified": raw, "dropped": True})
    assert sizes["toon"] == count_cl100k("a b c") and not flags["toon_passthrough"]
    assert sizes["headroom_default"] == count_cl100k("one two")
    assert sizes["headroom_agent"] == count_cl100k("one two three")
    assert sizes["headroom_verified"] == n and flags["headroom_dropped"]
    # Not JSON: the codec has nothing to work on.
    sizes, _ = rc.encoder_tokens(1, None, 1, 1, None, None)
    assert sizes["terse_codec"] == 1


def test_terse_shipped_keeps_the_replays_delta_when_the_rebuild_is_a_token_off():
    # Ledger raw 1,000, shown 997: terse saved 3. The rebuild came out at 998.
    sizes, _ = rc.encoder_tokens(998, None, 1000, 997, None, None)
    assert 998 - sizes["terse_shipped"] == 3


def test_run_toon_survives_line_separators_inside_a_payload():
    pytest.importorskip("subprocess")
    if not (rc.HERE / "node_modules").exists():
        pytest.skip("npm install not run in scripts/bench")
    payload = json.dumps({"rows": [{"t": "a\u2028b"}, {"t": "c\u0085d"}]}, ensure_ascii=False)
    got = rc.run_toon({"k": payload})
    assert got["k"]["lossless"] is True


def test_summarize_weights_each_saving_and_reports_coverage_and_what_was_left_out():
    def row(tool, raw, weight, dropped=False, **sizes):
        base = dict.fromkeys(rc.ENCODERS, raw)
        base.update(sizes)
        return {"tool": tool, "weight": weight, "raw_tokens": raw, "ledger_raw": raw,
                "sizes": base, "flags": {"toon_passthrough": False, "headroom_error": False,
                                         "headroom_dropped": dropped}}

    rows = [row("a", 1000, 2.0, terse_shipped=400, toon=1100),
            row("b", 500, 10.0, dropped=True, terse_shipped=450, headroom_default=100)]
    left = {"size_differs": {"results": 2, "raw": 3000.0, "saved": 1500.0,
                             "tools": {"codegraph_explore": 2500.0, "x": 500.0}}}
    got = rc.summarize(rows, left, 10000.0)
    assert got["all"]["raw_weighted"] == 7000.0 and got["coverage_pct"] == 70.0
    assert got["rebuild_residual_pct"] == 0.0
    # (600 x 2 + 50 x 10) of 7,000
    assert got["all"]["terse_shipped"] == {"saved_weighted": 1700.0, "saved_pct": 24.29}
    assert got["all"]["toon"]["saved_pct"] == -2.86          # an encoder may cost tokens
    assert got["by_tool"]["b"]["headroom_default"]["saved_pct"] == 80.0
    assert got["headroom_default_with_marker"] == {
        "results": 1, "saved_weighted": 4000.0, "saved_pct_of_all": 57.14}
    assert got["headroom_default_without_marker"]["saved_weighted"] == 0.0
    assert got["left_out"]["size_differs"]["terse_saved_pct"] == 50.0
    assert got["left_out"]["size_differs"]["top_tools"] == {"codegraph_explore": 2500.0,
                                                            "x": 500.0}
    text = rc.render(got)
    assert "70.0% of the raw-path cost" in text and "upper bound" in text
    assert "left out, size_differs: 2 results" in text and "terse saved 50.0% there" in text
