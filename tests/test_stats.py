"""Savings ledger: payload-free records, rotation, aggregation, the stats report."""

from __future__ import annotations

import json

import pytest

from terse import stats as stats_mod
from terse.stats import (
    aggregate,
    append_stats,
    build_record,
    build_stats_report,
    build_stats_writer,
    classify_decision,
    default_stats_log,
    load_stats,
    parse_window,
    server_label,
)

RAW = json.dumps({"result": [{"id": i, "status": "active"} for i in range(20)]}, indent=2)


# --- decision classification (text-sniff, no proxy state) ---

def test_classify_decision_covers_all_four_labels():
    assert classify_decision(RAW, RAW, passthrough=True) == "passthrough"
    assert classify_decision(RAW, RAW, passthrough=False) == "unchanged"
    assert classify_decision(RAW, '{"__terse_diff__":1,"shape":"rows"}',
                             passthrough=False) == "diff"
    assert classify_decision(RAW, '{"__terse_textdiff__":1,"ops":[]}',
                             passthrough=False) == "diff"
    assert classify_decision(RAW, '{"__terse_table__":1,"n":20}',
                             passthrough=False) == "compressed"


def test_classify_decision_diff_marker_only_counts_in_envelope_head():
    # A compressed payload that merely CONTAINS the marker string deep inside (e.g. a
    # tool result about terse itself) must not be misread as a diff.
    emitted = '{"__terse_table__":1,"rows":[["__terse_diff__"]]}'
    assert classify_decision(RAW, emitted, passthrough=False) == "compressed"


# --- record shape: the payload-free property ---

def test_build_record_stores_sizes_and_labels_never_content():
    rec = build_record("runecho-mcp", "structure", RAW, '{"__terse_table__":1}',
                       passthrough=False)
    assert rec["server"] == "runecho-mcp" and rec["tool"] == "structure"
    assert rec["decision"] == "compressed"
    assert rec["raw_chars"] == len(RAW) and rec["out_chars"] == len('{"__terse_table__":1}')
    assert isinstance(rec["ts"], int)
    # tokens are ints with tiktoken installed (the dev environment), or None without
    assert rec["raw_tokens"] is None or rec["raw_tokens"] > rec["out_tokens"] > 0
    # the property that makes always-on safe: no payload content in any value
    assert "active" not in json.dumps(rec)


def test_server_label_stdio_url_and_empty():
    assert server_label(["/home/x/.local/bin/runecho-mcp"]) == "runecho-mcp"
    assert server_label(["https://kb.example.com/mcp"]) == "kb.example.com"
    assert server_label([]) == "unknown"


def test_default_stats_log_honors_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_stats_log() == tmp_path / "terse" / "stats.jsonl"


# --- append/load: rotation, torn lines, the since filter ---

def test_append_and_load_roundtrip(tmp_path):
    log = tmp_path / "nested" / "stats.jsonl"       # parent created on demand
    for i in range(3):
        append_stats({"ts": 100 + i, "raw_chars": 10, "out_chars": 5}, log)
    recs = load_stats(log)
    assert [r["ts"] for r in recs] == [100, 101, 102]


def test_append_rotates_past_max_bytes_and_load_reads_both_generations(tmp_path):
    log = tmp_path / "stats.jsonl"
    rec = {"ts": 1, "raw_chars": 10, "out_chars": 5}
    append_stats(rec, log, max_bytes=1)             # write 1: creates the live file
    append_stats(rec, log, max_bytes=1)             # write 2: rotates, then appends
    assert (tmp_path / "stats.jsonl.1").exists()
    assert len(load_stats(log)) == 2                # .1 + live both loaded


def test_load_skips_torn_lines_and_filters_since(tmp_path):
    log = tmp_path / "stats.jsonl"
    append_stats({"ts": 100, "raw_chars": 1, "out_chars": 1}, log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": 200, "raw_ch')            # a torn line from a crashed writer
    append_stats({"ts": 300, "raw_chars": 1, "out_chars": 1}, log)
    assert [r["ts"] for r in load_stats(log)] == [100, 300]
    assert [r["ts"] for r in load_stats(log, since_ts=200)] == [300]


def test_parse_window_units_and_rejects_garbage():
    assert parse_window("30m") == 1800
    assert parse_window("24h") == 86400
    assert parse_window("7d") == 7 * 86400
    for bad in ("", "7", "d7", "3 days", "-1d"):
        try:
            parse_window(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_window accepted {bad!r}")


# --- aggregation ---

def _rec(server="s", tool="t", decision="compressed", raw_t=100, out_t=40,
         raw_c=400, out_c=160, ts=1, diff_reason=None):
    rec = {"ts": ts, "server": server, "tool": tool, "decision": decision,
           "raw_chars": raw_c, "out_chars": out_c,
           "raw_tokens": raw_t, "out_tokens": out_t}
    if diff_reason is not None:
        rec["diff_reason"] = diff_reason
    return rec


def test_aggregate_totals_decisions_and_per_tool_rows():
    agg = aggregate([
        _rec(tool="a", decision="diff", raw_t=100, out_t=10),
        _rec(tool="a", decision="compressed", raw_t=100, out_t=50),
        _rec(tool="b", decision="unchanged", raw_t=10, out_t=10),
    ])
    assert agg["total"]["results"] == 3
    assert agg["total"]["raw_tokens"] == 210 and agg["total"]["out_tokens"] == 70
    assert agg["decisions"] == {"diff": 1, "compressed": 1, "unchanged": 1}
    rows = agg["tools"]
    assert rows[0]["tool"] == "a"                   # sorted by tokens saved, desc
    assert rows[0]["results"] == 2 and rows[0]["diffs"] == 1


def test_aggregate_keeps_untokenized_records_out_of_token_totals():
    # A record written without tiktoken must not blend char-sized zeros into the token
    # sums — it is counted separately so the report can show the gap explicitly.
    agg = aggregate([_rec(raw_t=100, out_t=40),
                     _rec(raw_t=None, out_t=None, raw_c=1000, out_c=100)])
    assert agg["total"]["raw_tokens"] == 100 and agg["total"]["out_tokens"] == 40
    assert agg["total"]["untokenized"] == 1
    assert agg["total"]["raw_chars"] == 1400        # chars always cover everything


def test_aggregate_ignores_non_ledger_records():
    assert aggregate([{"ts": 1, "something": "else"}])["total"]["results"] == 0


def test_aggregate_tallies_diff_reasons_and_tolerates_their_absence():
    # Phase 1: the diff_reason breakdown counts only records that carry the field, so a
    # ledger mixing old (no reason) and new records aggregates cleanly.
    agg = aggregate([
        _rec(diff_reason="emitted"),
        _rec(diff_reason="no_prior"),
        _rec(diff_reason="not_smaller_diff_args"),
        _rec(),  # older record, no diff_reason — must not become a phantom bucket
    ])
    assert agg["diff_reasons"] == {"emitted": 1, "no_prior": 1, "not_smaller_diff_args": 1}


def test_aggregate_diff_reasons_empty_when_no_record_has_one():
    assert aggregate([_rec(), _rec()])["diff_reasons"] == {}


# --- report rendering ---

def test_report_shows_savings_and_per_tool_rows():
    agg = aggregate([_rec(server="runecho", tool="structure", decision="diff",
                          raw_t=1000, out_t=100)])
    out = build_stats_report(agg, log_path="/x/stats.jsonl", window="7d")
    assert "last 7d" in out
    assert "1,000 -> 100" in out and "saved 900" in out and "90.0%" in out
    assert "runecho" in out and "structure" in out and "diff=1" in out


def test_report_shows_diff_reason_breakdown_when_present():
    agg = aggregate([_rec(decision="diff", diff_reason="emitted"),
                     _rec(decision="compressed", diff_reason="not_smaller_diff_args")])
    out = build_stats_report(agg, log_path="/x/stats.jsonl")
    assert "diff reasons:" in out
    assert "emitted=1" in out and "not_smaller_diff_args=1" in out


def test_report_omits_diff_reason_line_for_legacy_ledger():
    # A ledger of only pre-Phase-1 records has no reasons — the line must not appear.
    out = build_stats_report(aggregate([_rec()]), log_path="/x/stats.jsonl")
    assert "diff reasons:" not in out


def test_report_empty_ledger_says_so():
    out = build_stats_report(aggregate([]), log_path="/x/stats.jsonl")
    assert "no results recorded" in out


def test_report_empty_window_points_at_the_window_not_at_nothing_ever():
    # A --since window that filters everything out isn't the same as an empty ledger:
    # the old message ("no results recorded") pointed at the wrong cause.
    out = build_stats_report(aggregate([]), log_path="/x", window="30m")
    assert "no results in the last 30m" in out
    assert "widen --since" in out
    assert "no results recorded" not in out


def test_report_all_untokenized_falls_back_to_chars_explicitly():
    agg = aggregate([_rec(raw_t=None, out_t=None, raw_c=1000, out_c=100)])
    out = build_stats_report(agg, log_path="/x")
    assert "tokens: unavailable" in out and "1,000 -> 100" in out


def test_report_per_tool_table_uses_chars_when_untokenized_not_all_zeros():
    # Without tiktoken at record time, the per-tool table used to render every token
    # column as 0 while the header honestly showed char savings — the most useful part
    # of the report went blank. It must mirror the header and fall back to chars.
    agg = aggregate([_rec(server="rune", tool="structure",
                          raw_t=None, out_t=None, raw_c=1000, out_c=100)])
    out = build_stats_report(agg, log_path="/x")
    assert "chr raw" in out and "chr out" in out  # table switched units, labeled
    assert "tok raw" not in out                    # no token column shown at all
    # the per-tool row carries real char numbers + a computed saving, not zeros
    table = out.splitlines()[-1]
    assert "structure" in table and "1,000" in table and "90.0%" in table


def test_report_shows_diff_hit_rate_per_tool():
    # The diff hit rate (diffs / results) is the metric the ledger exists for; a bare
    # count is meaningless without its denominator.
    agg = aggregate([_rec(tool="a", decision="diff"),
                     _rec(tool="a", decision="diff"),
                     _rec(tool="a", decision="compressed"),
                     _rec(tool="a", decision="compressed")])
    out = build_stats_report(agg, log_path="/x")
    assert "diff%" in out
    table = out.splitlines()[-1]
    assert table.split()[2] == "4" and table.split()[3] == "2"  # results, diffs
    assert "50%" in table                                       # 2 / 4 hit rate


# --- the proxy-side writer callback ---

def test_build_stats_writer_appends_a_record(tmp_path):
    log = tmp_path / "stats.jsonl"
    writer = build_stats_writer(log, "runecho-mcp")
    writer("structure", RAW, '{"__terse_table__":1}', False)
    recs = load_stats(log)
    assert len(recs) == 1
    assert recs[0]["server"] == "runecho-mcp" and recs[0]["decision"] == "compressed"


def test_build_stats_writer_propagates_failures(tmp_path, monkeypatch):
    # The writer owns I/O and nothing else: it must NOT swallow. Stats stays never
    # load-bearing, but the swallow-and-warn-once belongs to the single caller that has
    # the per-sink bookkeeping, proxy.Interceptor._warn_sink — catching here too made
    # that unconditional first-failure warning dead code (#131). Pinned end-to-end by
    # test_proxy.py::test_run_proxy_broken_stats_log_warns_without_debug.
    def boom(*_a, **_kw):
        raise OSError("disk full")
    monkeypatch.setattr(stats_mod, "append_stats", boom)
    writer = build_stats_writer(tmp_path / "s.jsonl", "s")
    with pytest.raises(OSError, match="disk full"):
        writer("tool", RAW, RAW, False)
