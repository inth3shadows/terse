"""Offline tests for replay_sessions.py: ledger grouping, the ledger-to-transcript join,
the offload rules, and the per-call carry pricing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Importable by bare module name, as cost_per_task's tests do (no conftest.py: a second
# one shadows tests/conftest.py in a whole-repo run).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import replay_sessions as rs

OPUS = "claude-opus-5-5"      # cache read 0.05x
HAIKU = "claude-haiku-4-5"    # cache read 0.1x


def call(read=0, w1=0, w5=0, inp=0, out=0, model=OPUS, billed=True):
    return rs.Call(model, inp, read, w5, w1, out, billed)


def row(ts, tool="kb.read.search", raw=1000, out=600, **kw):
    rec = {"ts": ts, "server": "kb", "tool": tool, "decision": "compressed",
           "raw_chars": raw * 4, "out_chars": out * 4, "raw_tokens": raw, "out_tokens": out}
    rec.update(kw)
    return rec


def ledger_file(tmp_path, rows):
    path = tmp_path / "stats.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def result(tid="t1", name="mcp__terse__kb_read_search", ts=100.5, chars=2400, tokens=600,
           offloaded=False, args="{}"):
    return rs.Result(tid, name, ts, chars, tokens, offloaded, args)


def test_norm_tool_folds_ledger_and_transcript_spellings():
    assert (rs.norm_tool("kb__kb.read.search") == rs.norm_tool("mcp__terse__kb_read_search")
            == "kb_read_search")


def test_parse_ts_keeps_milliseconds():
    assert rs.parse_ts("1970-01-01T00:01:40.250Z") == 100.25
    assert rs.parse_ts("1970-01-01T00:01:40Z") == 100.0


def test_load_ledger_groups_blocks_of_one_result_and_skips_events(tmp_path):
    path = ledger_file(tmp_path, [
        row(100), row(100), row(101),                 # one three-block result
        row(105),                                     # a later result
        row(100, tool="kb.read.get"),                 # another tool at the same second
        {"ts": 100, "server": "kb", "event": "primer", "tokens": 500},
        {"ts": 100, "server": "kb", "event": "retrieve", "tool": "x", "tokens": 9},
    ])
    got = rs.load_ledger(path)
    assert sorted((g.tool, len(g.rows)) for g in got) == [
        ("kb_read_get", 1), ("kb_read_search", 1), ("kb_read_search", 3)]


def test_join_claims_each_ledger_result_once(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(104)]))
    joined, counts = rs.join(ledger, [result("a", ts=100.4), result("b", ts=104.6),
                                      result("c", ts=104.9)])
    assert joined["a"] is ledger[0] and joined["b"] is ledger[1]
    assert "c" not in joined
    assert counts == {"ledger_results": 2, "joined": 2, "size_mismatch": 0,
                      "no_transcript": 0, "groups_split": 0, "changed_results": 2, "changed_joined": 2,
                      "saved_tokens_joined_pct": 100.0, "mismatch_offsets": {}}


def test_join_prefers_the_ledger_row_at_or_before_the_result(tmp_path):
    # The ledger stamp is floored, so 102 cannot be the row behind a result at 101.2
    # even though it is nearer than 100.
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(102)]))
    joined, _ = rs.join(ledger, [result("a", ts=101.2), result("b", ts=102.3)])
    assert joined["a"] is ledger[0] and joined["b"] is ledger[1]


def test_join_ignores_results_outside_the_window_or_of_another_tool(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)]))
    assert rs.join(ledger, [result("late", ts=100 + rs.JOIN_BEFORE + 1)])[0] == {}
    joined, counts = rs.join(ledger, [result("other", name="mcp__terse__kb_read_get")])
    assert joined == {} and counts["no_transcript"] == 1


def test_join_matches_a_copied_tool_use_id_once(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(102)]))
    joined, _ = rs.join(ledger, [result("a", ts=100.4), result("a", ts=100.4)])
    assert list(joined) == ["a"]


def test_join_splits_parallel_calls_that_were_grouped_as_one_result(tmp_path):
    # Three parallel calls land in one second and group as one "result" of 7,200 chars.
    # Each transcript result shows 2,400, so the group is split and joined row by row;
    # pricing the group against one result would have claimed all three savings at once.
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(100), row(101)]))
    assert len(ledger) == 1
    results = [result(t, ts=101.1 + i / 10) for i, t in enumerate("abc")]
    joined, counts = rs.join(ledger, results)
    assert sorted(joined) == ["a", "b", "c"]
    assert all(len(led.rows) == 1 for led in joined.values())
    assert len({id(led) for led in joined.values()}) == 3
    assert counts == {"ledger_results": 3, "joined": 3, "size_mismatch": 0,
                      "no_transcript": 0, "groups_split": 1, "changed_results": 3, "changed_joined": 3,
                      "saved_tokens_joined_pct": 100.0, "mismatch_offsets": {}}


def test_join_keeps_a_true_multi_block_result_whole(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(100), row(101)]))
    joined, counts = rs.join(ledger, [result("a", ts=101.2, chars=7230)])
    assert len(joined["a"].rows) == 3 and counts["groups_split"] == 0


def test_join_drops_a_pairing_whose_sizes_disagree(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)]))
    joined, counts = rs.join(ledger, [result("a", chars=900)])
    assert joined == {} and counts["size_mismatch"] == 1 and counts["joined"] == 0
    assert counts["mismatch_offsets"] == {"-2000..-200": 1}


def test_join_does_not_let_a_foreign_result_take_the_owners_row(tmp_path):
    # Another server's `search` lands first with a size that cannot be this row's.
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)]))
    joined, counts = rs.join(ledger, [
        result("foreign", name="mcp__other__kb_read_search", ts=100.2, chars=9000),
        result("owner", ts=100.6)])
    assert list(joined) == ["owner"] and counts["size_mismatch"] == 0


def test_join_accepts_the_smallest_and_the_largest_primer(tmp_path):
    for extra in (379 + 2, 966 + 2, 2237 + 16):
        ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)]))
        assert list(rs.join(ledger, [result("a", chars=2400 + extra)])[0]) == ["a"]


def test_join_gives_parallel_calls_of_different_sizes_their_own_rows(tmp_path):
    # The 2,400-char result comes first in time and would fit the 1,000-char row as
    # "1,000 plus a primer". The same-size row must win.
    ledger = rs.load_ledger(ledger_file(tmp_path, [
        row(100, raw=5000, out=250), row(100, raw=700, out=600)]))
    assert len(ledger) == 1
    joined, counts = rs.join(ledger, [result("big", ts=100.3, chars=2400),
                                      result("small", ts=100.4, chars=1000)])
    assert joined["big"].text_out_chars == 2400 and joined["small"].text_out_chars == 1000
    assert counts["groups_split"] == 1 and counts["joined"] == 2


def test_a_split_row_never_pairs_with_a_whole_multi_block_result(tmp_path):
    # A true two-block result (1,000 + 1,500 chars) and the same tool called again
    # alongside it: two results arrived with the group, so it is split up front. The whole
    # result (2,516 chars) must not then pass as "block one plus a primer". Losing it is
    # acceptable; pricing it against one block is not.
    ledger = rs.load_ledger(ledger_file(tmp_path, [
        row(100, raw=500, out=250), row(100, raw=700, out=375), row(103, raw=900, out=600)]))
    joined, counts = rs.join(ledger, [result("whole", ts=100.4, chars=2516),
                                      result("later", ts=101.9, chars=2400)])
    assert list(joined) == ["later"]
    assert counts["joined"] == 1 and counts["size_mismatch"] == 2
    assert counts["joined"] + counts["size_mismatch"] + counts["no_transcript"] == \
        counts["ledger_results"] == 3


def test_a_multi_block_result_survives_the_same_tool_called_seconds_later(tmp_path):
    # Not parallel: the second call lands 5 s after the group's rows. The group stays
    # whole and joins its own result.
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(100), row(101), row(106)]))
    joined, counts = rs.join(ledger, [result("whole", ts=101.3, chars=7230),
                                      result("later", ts=106.4)])
    assert len(joined["whole"].rows) == 3 and len(joined["later"].rows) == 1
    assert counts["groups_split"] == 0 and counts["ledger_results"] == 2


def test_a_split_parallel_call_that_carried_the_primer_is_dropped_visibly(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(100)]))
    joined, counts = rs.join(ledger, [result("a", ts=100.3, chars=2400 + 2100),
                                      result("b", ts=100.4, chars=2400)])
    assert list(joined) == ["b"]
    assert counts["size_mismatch"] == 1 and counts["mismatch_offsets"] == {"350..3200": 1}


def test_join_accepts_a_result_longer_by_the_primer_or_shown_as_a_preview(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100), row(200, raw=30000, out=20000)]))
    joined, _ = rs.join(ledger, [result("a", chars=2400 + 2100),
                                 result("b", ts=200.3, chars=310, offloaded=True)])
    assert sorted(joined) == ["a", "b"]


def test_an_offloaded_result_cannot_take_a_small_row_from_its_owner(tmp_path):
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)]))
    joined, _ = rs.join(ledger, [result("preview", ts=100.2, chars=310, offloaded=True),
                                 result("owner", ts=100.6)])
    assert list(joined) == ["owner"]


def test_classify_prices_raw_minus_what_the_model_was_shown(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=1000, out=600)]))[0]
    # 650 shown, not the ledger's 600: the primer rode this result and is charged.
    assert rs.classify(result(tokens=650), led) == ("priced", 350, 0)


def test_classify_reads_the_typed_field_when_that_is_what_was_shown(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(
        100, raw=3000, out=1300, raw_chars=12000, out_chars=5000,
        structured_chars=8000, structured_out_chars=3000,
        structured_tokens=2000, structured_out_tokens=800)]))[0]
    assert rs.classify(result(chars=3010, tokens=800), led) == ("priced", 1200, 0)
    # A text-reading client saw the 2,000-char text block: raw text was 1,000 tokens.
    assert rs.classify(result(chars=2000, tokens=500), led) == ("priced", 500, 0)


def test_classify_claims_nothing_for_offloaded_results(tmp_path):
    big = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=30000, out=26000)]))[0]
    assert rs.classify(result(offloaded=True, tokens=500), big) == ("offloaded", 0, 0)
    small = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=12000, out=11900)]))[0]
    assert rs.classify(result(offloaded=True, tokens=500), small) == ("offload_only_out", 0, 0)


def test_classify_kept_inline_reports_the_shown_size_for_the_bound(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=30000, out=9000)]))[0]
    assert rs.classify(result(tokens=9000), led) == ("kept_inline", 0, 9000)


def test_classify_skips_rows_without_token_counts(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw_tokens=None, out_tokens=None)]))[0]
    assert rs.classify(result(), led)[0] == "no_tokens"


def test_tail_weight_is_the_calls_own_write_mix():
    assert call(read=5000, w1=1000).tail_weight() == 2.0
    assert call(read=5000, w5=1000).tail_weight() == 1.25
    assert call(inp=1000).tail_weight() == 1.0
    assert call(read=5000).tail_weight() == pytest.approx(0.05)  # nothing new was written


def test_carry_pays_write_once_then_read_on_every_later_call():
    calls = [call(read=10000, w1=1000), call(read=11000, w1=200), call(read=11200, w1=100)]
    weighted, usd, n = rs.carry(calls, 0, 3)
    assert n == 3 and weighted == pytest.approx(2.0 + 0.05 + 0.05)
    assert usd == pytest.approx(weighted * 4.0 / 1e6)


def test_carry_charges_the_write_rate_again_on_a_cache_rebuild():
    calls = [call(read=10000, w1=1000), call(read=0, w1=11500)]
    assert rs.carry(calls, 0, 2)[0] == pytest.approx(2.0 + 2.0)


def test_carry_stops_when_the_history_was_dropped():
    calls = [call(read=100000, w1=1000), call(read=3000, w1=2000), call(read=5000, w1=100)]
    assert rs.carry(calls, 0, 3)[2] == 1


def test_carry_skips_a_call_another_transcript_already_billed():
    calls = [call(read=10000, w1=1000, billed=False), call(read=11000, w1=200, billed=False),
             call(read=11200, w1=100)]
    weighted, _, n = rs.carry(calls, 0, 3)
    assert n == 1 and weighted == pytest.approx(0.05)


def test_carry_has_no_usd_for_an_unpriced_model_but_keeps_the_weight():
    weighted, usd, _ = rs.carry([call(w1=100, model="claude-unknown-9")], 0, 1)
    assert weighted == 2.0 and usd is None


def test_carry_uses_each_models_own_read_rate():
    calls = [call(read=1000, w5=100, model=HAIKU), call(read=1100, w5=50, model=HAIKU)]
    assert rs.carry(calls, 0, 2)[0] == pytest.approx(1.25 + 0.1)


def test_replay_prices_a_result_until_the_compaction_boundary(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=1000, out=600)]))
    res = result("a", tokens=600)
    events = [("call", call(read=10000, w1=500)), ("result", res),
              ("call", call(read=10500, w1=700)), ("call", call(read=11200, w1=100)),
              ("compact",), ("call", call(read=11000, w1=100))]
    out = rs.replay(events, {"a": led[0]})
    item = out["items"][0]
    assert item["carried_calls"] == 2
    assert item["weighted"] == pytest.approx(400 * (2.0 + 0.05))
    assert out["net_weighted"] == pytest.approx(item["weighted"])
    assert out["calls"] == 4 and out["model"] == OPUS


def test_replay_prices_nothing_when_compaction_follows_the_result_at_once(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=1000, out=600)]))
    events = [("call", call(read=10000, w1=500)), ("result", result("a")), ("compact",),
              ("call", call(read=9000, w1=100)), ("call", call(read=9100, w1=100))]
    item = rs.replay(events, {"a": led[0]})["items"][0]
    assert item["carried_calls"] == 0 and item["weighted"] == 0


def test_replay_scales_the_delta_by_the_models_tokenizer_ratio(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=1000, out=600)]))
    events = [("call", call(w1=100)), ("result", result("a")), ("call", call(read=100, w1=700))]
    plain = rs.replay(events, {"a": led[0]})["items"][0]
    scaled = rs.replay(events, {"a": led[0]}, {OPUS: 1.5, None: 9.0})["items"][0]
    other = rs.replay(events, {"a": led[0]}, {None: 2.0})["items"][0]
    assert plain["weighted"] == pytest.approx(400 * 2.0)
    assert scaled["weighted"] == pytest.approx(600 * 2.0)
    assert other["weighted"] == pytest.approx(800 * 2.0)
    assert plain["delta_tokens"] == scaled["delta_tokens"] == 400   # cl100k kept as is


def test_replay_leaves_copied_history_out_of_the_session_total():
    events = [("call", call(read=1000, w1=100, billed=False)), ("call", call(read=1100, w1=50))]
    out = rs.replay(events, {})
    assert out["calls"] == 2 and out["billed_calls"] == 1
    assert out["session_weighted"] == pytest.approx(1100 * 0.05 + 50 * 2.0)


def test_a_copied_result_keeps_its_weight_but_is_counted_once(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100, raw=1000, out=600)]))[0]
    original = [("call", call(w1=100)), ("result", result("a")), ("call", call(read=100, w1=700))]
    resumed = [("call", call(w1=100, billed=False)), ("result", result("a")),
               ("call", call(read=100, w1=700, billed=False)), ("call", call(read=800, w1=50))]
    rows = [rs.replay(ev, {"a": led}) for ev in (original, resumed)]
    assert [r["items"][0]["copy"] for r in rows] == [False, True]
    assert rows[1]["items"][0]["weighted"] == pytest.approx(400 * 0.05)
    counts = {"ledger_results": 1, "joined": 1, "size_mismatch": 0, "no_transcript": 0,
              "groups_split": 0, "changed_results": 1, "changed_joined": 1,
              "saved_tokens_joined_pct": 100.0, "mismatch_offsets": {}}
    got = rs.summarize(rows, counts, {})
    assert got["by_status"]["priced"]["results"] == 1
    assert got["by_status"]["priced"]["delta_tokens"] == 400
    assert got["by_status"]["priced"]["weighted"] == pytest.approx(400 * 2.0 + 400 * 0.05)
    assert got["copied_results"] == 1 and got["recall"]["changed"]["n"] == 1


def test_tokenizer_ratios_need_enough_samples_per_model_else_fall_back_to_pooled():
    got = rs.tokenizer_ratios({OPUS: [1.2] * 20, HAIKU: [2.0] * 3})
    assert got[OPUS] == 1.2 and HAIKU not in got and got[None] == 1.2
    assert rs.tokenizer_ratios({}) == {None: 1.0}


def test_summarize_reports_unpriced_weight_and_the_untouched_check():
    item = {"tool": "t", "status": "priced", "delta_tokens": 400, "kept_inline_tokens": 0,
            "carried_calls": 2, "weighted": 800.0, "usd": None, "model": "claude-x",
            "changed": True, "recall_3": False, "shown_tokens": 600, "kept_inline_weighted": 0,
            "copy": False}
    same = dict(item, delta_tokens=3, weighted=6.0, usd=0.00002, changed=False, model=OPUS)
    session = {"calls": 3, "model": OPUS, "session_weighted": 10000.0, "net_weighted": 806.0,
               "items": [item, same]}
    counts = {"ledger_results": 2, "joined": 2, "size_mismatch": 0, "no_transcript": 0,
              "groups_split": 0, "changed_results": 1, "changed_joined": 1,
              "saved_tokens_joined_pct": 100.0, "mismatch_offsets": {}}
    got = rs.summarize([session], counts, {})
    assert got["weighted_without_usd_price"] == 800.0 and got["items_without_usd_price"] == 1
    assert got["untouched_check"] == {"results": 1, "delta_tokens": 3, "shown_tokens": 600}
    assert got["join"]["rate_pct"] == 100.0
    text = rs.render(got)
    assert "NOT in the usd figure" in text and "expect ~0" in text


def test_scan_transcript_marks_a_response_seen_in_an_earlier_file_unbilled(tmp_path):
    usage = {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 50}
    rec = {"type": "assistant", "requestId": "r1",
           "message": {"id": "m1", "model": OPUS, "usage": usage, "content": []}}
    new = dict(rec, requestId="r2")
    first = _records(tmp_path / "a.jsonl", [rec])
    second = _records(tmp_path / "b.jsonl", [rec, new])
    billed: set = set()
    assert [e[1].billed for e in rs.scan_transcript(first, billed)] == [True]
    assert [e[1].billed for e in rs.scan_transcript(second, billed)] == [False, True]


def test_replay_charges_a_retrieve_its_tokens_and_its_extra_call():
    res = result("r", name="mcp__terse__terse_retrieve", tokens=1000)
    events = [("call", call(read=20000, w1=500)), ("result", res),
              ("call", call(read=20500, w1=1100))]
    item = rs.replay(events, {})["items"][0]
    assert item["status"] == "retrieve"
    assert item["weighted"] == pytest.approx(-(1000 * 2.0) - 20500 * 0.05)


def test_replay_ignores_results_terse_never_handled():
    events = [("call", call(w1=100)), ("result", result("x")), ("call", call(read=100, w1=10))]
    assert rs.replay(events, {})["items"] == []


def test_replay_flags_an_identical_call_repeated_right_away(tmp_path):
    led = rs.load_ledger(ledger_file(tmp_path, [row(100), row(200)]))
    events = [("call", call(w1=100)), ("result", result("a", ts=100.2)),
              ("call", call(read=100, w1=10)), ("result", result("b", ts=200.2)),
              ("call", call(read=110, w1=10))]
    items = rs.replay(events, {"a": led[0], "b": led[1]})["items"]
    assert [i["recall_3"] for i in items] == [True, False]


def _records(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_scan_transcript_dedups_split_responses_and_reads_mcp_results(tmp_path):
    usage = {"input_tokens": 2, "cache_read_input_tokens": 900,
             "cache_creation_input_tokens": 100, "output_tokens": 5,
             "cache_creation": {"ephemeral_1h_input_tokens": 100, "ephemeral_5m_input_tokens": 0}}
    zero = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}
    use = {"type": "tool_use", "id": "tu1", "name": "mcp__terse__kb_read_get", "input": {"a": 1}}
    bash = {"type": "tool_use", "id": "tu2", "name": "Bash", "input": {}}

    def assistant(content, usage=usage, rid="r1", mid="m1"):
        return {"type": "assistant", "requestId": rid,
                "message": {"id": mid, "model": OPUS, "usage": usage, "content": content}}

    def tool_result(tid, content):
        return {"type": "user", "timestamp": "2026-10-01T00:00:01.500Z",
                "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                         "content": content}]}}

    path = _records(tmp_path / "s.jsonl", [
        assistant([{"type": "text", "text": "hi"}]), assistant([use, bash]),
        tool_result("tu1", [{"type": "text", "text": "abc"}, {"type": "text", "text": "de"}]),
        tool_result("tu2", "not an mcp tool"),
        {"type": "system", "subtype": "compact_boundary"},
        assistant([], usage=zero, rid="r2", mid="m2"),
        assistant([], rid="r3", mid="m3"),
    ])
    events = rs.scan_transcript(path)
    assert [e[0] for e in events] == ["call", "result", "compact", "call"]
    call0, res = events[0][1], events[1][1]
    assert (call0.read, call0.w1, call0.w5, call0.inp) == (900, 100, 0, 2)
    assert (res.tool_use_id, res.chars, res.offloaded) == ("tu1", 5, False)
    assert res.args == '{"a": 1}' and res.ts == rs.parse_ts("2026-10-01T00:00:01.500Z")


def test_scan_transcript_marks_a_persisted_preview_as_offloaded(tmp_path):
    use = {"type": "tool_use", "id": "tu1", "name": "mcp__kb__list", "input": {}}
    path = _records(tmp_path / "s.jsonl", [
        {"type": "assistant", "requestId": "r", "message": {
            "id": "m", "model": OPUS, "content": [use],
            "usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "user", "timestamp": "2026-10-01T00:00:00Z", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu1",
             "content": "<persisted-output>\nOutput too large (57.3KB)..."}]}},
    ])
    assert rs.scan_transcript(path)[1][1].offloaded is True


def test_both_wordings_of_the_saved_to_file_notice_count_as_offloaded():
    assert rs.is_offload_notice("<persisted-output>\nOutput too large (57.3KB). Full output")
    assert rs.is_offload_notice(
        "Error: result (73,483 characters) exceeds maximum allowed tokens. Output has been saved to /x")
    assert not rs.is_offload_notice("Error: result (bad) was not found")
    assert not rs.is_offload_notice('{"result": "Error: result (1 characters) exceeds"}')


def test_calibration_uses_only_a_large_result_alone_between_two_calls():
    big = result(tokens=4000)
    events = [("call", call(read=10000, w1=500, out=100)), ("result", big),
              ("call", call(read=10500, w1=5300))]
    assert rs.calibration_samples(events) == [(OPUS, pytest.approx(5200 / 4000))]
    assert rs.calibration_samples([events[0], ("result", result(tokens=50)), events[2]]) == []


def test_main_stops_before_pricing_when_the_join_rate_is_low(tmp_path, capsys):
    ledger = ledger_file(tmp_path, [row(100), row(200)])
    (tmp_path / "projects" / "p").mkdir(parents=True)
    out = tmp_path / "out"
    code = rs.main(["--ledger", str(ledger), "--transcripts", str(tmp_path / "projects"),
                    "--out", str(out)])
    assert code == 3 and not out.exists()
    assert "stopping before pricing" in capsys.readouterr().err


def test_the_gate_ignores_passthrough_rows_that_have_no_transcript(tmp_path):
    # A capture script calls the proxy with no model session: its rows are passthrough
    # and join nothing. They must not stop a run whose changed results all joined.
    quiet = [row(300 + 10 * i, decision="passthrough", raw=50, out=50) for i in range(9)]
    ledger = rs.load_ledger(ledger_file(tmp_path, [row(100)] + quiet))
    _, counts = rs.join(ledger, [result("a")])
    assert (counts["joined"], counts["ledger_results"]) == (1, 10)
    assert (counts["changed_joined"], counts["changed_results"]) == (1, 1)


def test_main_writes_a_row_per_transcript_and_a_summary(tmp_path, capsys):
    ledger = ledger_file(tmp_path, [row(1790812801, raw=1000, out=600)])
    proj = tmp_path / "projects" / "p"
    proj.mkdir(parents=True)
    use = {"type": "tool_use", "id": "tu1", "name": "mcp__terse__kb_read_search", "input": {}}
    usage = {"input_tokens": 0, "cache_read_input_tokens": 1000, "output_tokens": 10,
             "cache_creation_input_tokens": 100,
             "cache_creation": {"ephemeral_1h_input_tokens": 100, "ephemeral_5m_input_tokens": 0}}
    _records(proj / "s.jsonl", [
        {"type": "assistant", "requestId": "r1",
         "message": {"id": "m1", "model": OPUS, "usage": usage, "content": [use]}},
        {"type": "user", "timestamp": "2026-10-01T00:00:01.400Z", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "word " * 480}]}},
        {"type": "assistant", "requestId": "r2",
         "message": {"id": "m2", "model": OPUS, "usage": usage, "content": []}},
    ])
    out = tmp_path / "out"
    assert rs.main(["--ledger", str(ledger), "--transcripts", str(tmp_path / "projects"),
                    "--out", str(out)]) == 0
    rows = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["items"][0]["status"] == "priced"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["join"]["joined"] == 1 and summary["join"]["rate_pct"] == 100.0
    assert summary["net_weighted"] == rows[0]["net_weighted"] > 0
    assert "join: 1 of 1" in capsys.readouterr().out
