"""Drop-rule tuning from the ledger (#252).

A retrieve row says a dropped value was fetched back; until `dropped_paths` a tool row
did not say a drop happened at all, so a rule's retrieve RATE had no denominator. Each
result row now carries `dropped_paths` (rule path -> values dropped), `aggregate` joins it
with the retrieve rows into `drop_rules`, `terse stats` shows them, and
`terse tune --suggest` turns them into print-only suggestions.
"""

from __future__ import annotations

import json

import pytest

from terse import lossy
from terse.cli import main
from terse.policy import Policy, Rule
from terse.proxy import Interceptor, PrimerLatch, union_primer
from terse.stats import (
    aggregate,
    build_retrieve_record,
    build_retrieve_writer,
    build_stats_report,
    build_stats_writer,
    suggest_drop_tuning,
)

PATH = "$text.code_blocks"
POLICY = Policy(rules=[Rule("codegraph_*", ("minify",),
                            fields={PATH: {"lossy": "drop-to-retrieve"}})])


def _block(body: str, n: int = 80) -> str:
    return "```python\n" + (body + "\n") * n + "```\n"


def _handles(text: str) -> list[str]:
    return [json.loads(line)[lossy.DROP_KEY] for line in text.splitlines()
            if line.startswith("{") and lossy.DROP_KEY in line]


def _call(inter: Interceptor, mid: int, text: str) -> str:
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": "codegraph_explore"}}))
    out = inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": mid,
         "result": {"content": [{"type": "text", "text": text}]}}))
    # every text block: the lazy primer may ride in front of the result
    return "\n".join(b["text"] for b in json.loads(out)["result"]["content"])


def _ledger_interceptor(log):
    return Interceptor(POLICY, server_name="kb", ledger_label="kb",
                       stats=build_stats_writer(log, "kb"),
                       stats_retrieve=build_retrieve_writer(log, "kb"))


def _rows(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


# --- the ledger records what each result dropped ---

def test_a_result_row_records_its_drops_and_the_rate_joins_its_retrieves(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = _ledger_interceptor(log)
    text = _call(inter, 1, "intro\n" + _block("a = 0") + _block("b = 1") + _block("c = 2"))
    _call(inter, 2, "nothing to drop here")
    inter.answer_retrieve(json.dumps(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": lossy.RETRIEVE_TOOL, "arguments": {"handle": _handles(text)[1]}}}))

    results = [r for r in _rows(log) if "raw_chars" in r]
    assert [r.get("dropped_paths") for r in results] == [{PATH: 3}, None]
    assert "a = 0" not in log.read_text()          # still payload-free

    (rule,) = aggregate(_rows(log))["drop_rules"]
    assert rule == {"server": "kb", "tool": "codegraph_explore", "path": PATH,
                    "dropped_calls": 1, "dropped_values": 3, "retrieved": 1,
                    "rate": 1.0, "indexes": {1: 1}}


def test_a_result_with_no_drops_records_none_after_one_that_did(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = _ledger_interceptor(log)
    _call(inter, 1, _block("a = 0"))
    _call(inter, 2, "plain text, no code")
    assert [r.get("dropped_paths") for r in _rows(log)] == [{PATH: 1}, None]


def test_a_multi_block_result_counts_its_drops_once_on_its_first_row(tmp_path):
    """Per result, like the typed field: one result is one dropped call."""
    log = tmp_path / "stats.jsonl"
    inter = _ledger_interceptor(log)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "codegraph_explore"}}))
    inter.transform_response(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
        {"type": "text", "text": "one\n" + _block("a = 0")},
        {"type": "text", "text": "two\n" + _block("b = 1")}]}}))
    rows = _rows(log)
    assert len(rows) == 2
    assert [r.get("dropped_paths") for r in rows] == [{PATH: 2}, None]
    assert aggregate(rows)["drop_rules"][0]["dropped_calls"] == 1


def test_a_held_result_that_went_out_raw_records_no_drop(tmp_path):
    """A router peer compresses a typed result tentatively while its primer is owed, and
    reverts to the raw line when the primer cannot ride it (here: the client would offload
    the typed field). The drops of that pass never reached the client, so none is recorded."""
    log = tmp_path / "stats.jsonl"
    pol = Policy(rules=[Rule("codegraph_*", ("minify",), structured="auto",
                             fields={PATH: {"lossy": "drop-to-retrieve"}})])
    latch = PrimerLatch(mode="always")
    latch.set_text(union_primer([(pol, "kb")], structured_wrap=True))
    inter = Interceptor(pol, server_name="kb", ledger_label="kb", lazy_primer=False,
                        shared_primer=latch, stats=build_stats_writer(log, "kb"))
    inter.client_name = "claude-code"
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "codegraph_explore"}}))
    raw = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "content": [{"type": "text", "text": _block("x = 0")}],
        "structuredContent": {"rows": [{"id": i, "note": f"row {i} " + "word " * 40}
                                       for i in range(1500)]}}})
    assert inter.transform_response(raw) == raw          # held: the original line
    assert latch.pending()
    assert "dropped_paths" not in _rows(log)[0]


def _drop_row(ts, n=1, tool="codegraph_explore"):
    return {"ts": ts, "server": "kb", "tool": tool, "decision": "compressed",
            "raw_chars": 100, "out_chars": 10, "dropped_paths": {PATH: n}}


def _retrieve_row(ts, index=None):
    rec = build_retrieve_record("kb", "codegraph_explore", PATH, hit=True, payload="x",
                                index=index)
    rec["ts"] = ts
    return rec


def test_retrieves_before_the_first_recorded_drop_are_not_in_the_rate():
    """A retrieve written before `dropped_paths` existed has no denominator."""
    (rule,) = aggregate([_retrieve_row(5), _drop_row(10), _drop_row(11),
                         _retrieve_row(12, index=0)])["drop_rules"]
    assert (rule["dropped_calls"], rule["retrieved"], rule["rate"]) == (2, 1, 0.5)


def test_the_first_drop_is_the_earliest_not_the_first_read():
    """`aggregate` does not assume rows arrive in time order (several proxies append to
    one ledger)."""
    (rule,) = aggregate([_drop_row(10), _drop_row(5), _retrieve_row(7)])["drop_rules"]
    assert rule["retrieved"] == 1


def test_malformed_dropped_paths_are_ignored():
    rows = [_drop_row(1), {**_drop_row(2), "dropped_paths": {PATH: True}},
            {**_drop_row(3), "dropped_paths": [PATH]}, {**_drop_row(4), "dropped_paths": {PATH: 0}}]
    (rule,) = aggregate(rows)["drop_rules"]
    assert rule["dropped_calls"] == 1


def test_no_drop_rows_means_no_drop_rules_and_no_stats_section(tmp_path):
    agg = aggregate([{**_drop_row(1), "dropped_paths": None}, _retrieve_row(2)])
    assert agg["drop_rules"] == []
    assert "drop-to-retrieve use" not in build_stats_report(agg, log_path=tmp_path / "x")


def test_stats_shows_each_rule_and_its_rate(tmp_path):
    agg = aggregate([_drop_row(1), _drop_row(2), _drop_row(3), _drop_row(4),
                     _retrieve_row(5, index=0)])
    report = build_stats_report(agg, log_path=tmp_path / "x")
    assert "drop-to-retrieve use" in report
    line = next(ln for ln in report.splitlines() if ln.startswith("kb ") and PATH in ln
                and "%" in ln)
    assert line.split()[-3:] == ["4", "1", "25.0%"]


# --- suggestions ---

def _rule(dropped, retrieved, indexes=None):
    return {"server": "kb", "tool": "codegraph_explore", "path": PATH,
            "dropped_calls": dropped, "dropped_values": dropped, "retrieved": retrieved,
            "rate": retrieved / dropped, "indexes": indexes or {}}


def test_a_rule_retrieved_on_a_quarter_of_its_drops_is_retracted():
    (line,) = suggest_drop_tuning([_rule(100, 25)])
    assert "retract:" in line and "25.0%" in line
    (below,) = suggest_drop_tuning([_rule(100, 24)])
    assert "retract" not in below


def test_retrieves_clustered_at_one_block_suggest_keep_first():
    lines = suggest_drop_tuning([_rule(40, 6, {0: 3, 2: 2, 5: 1})])
    assert any(ln.endswith("keep_first: 1 -- 3 of 6 indexed retrieves fetched block 0.")
               for ln in lines)
    # under half at one index, or too few indexed retrieves: no keep_first
    assert not any("keep_first" in ln
                   for ln in suggest_drop_tuning([_rule(40, 6, {0: 2, 2: 2, 5: 2})]))
    assert not any("keep_first" in ln
                   for ln in suggest_drop_tuning([_rule(40, 4, {0: 4})]))


def test_widening_needs_enough_drops_and_a_low_rate_and_carries_the_caveat():
    (line,) = suggest_drop_tuning([_rule(100, 2)])
    assert "widen candidate" in line and "not that its answers were unharmed" in line
    assert "no change" in suggest_drop_tuning([_rule(99, 0)])[0]
    assert "no change" in suggest_drop_tuning([_rule(100, 3)])[0]


def test_thresholds_are_parameters():
    assert "retract" in suggest_drop_tuning([_rule(100, 10)], retract_rate=0.1)[0]
    assert "widen" in suggest_drop_tuning([_rule(10, 0)], widen_min=10)[0]
    assert "widen" in suggest_drop_tuning([_rule(100, 5)], widen_rate=0.05)[0]


# --- the command ---

def _write(log, rows):
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_tune_suggest_prints_suggestions_from_the_ledger(tmp_path, capsys):
    log = tmp_path / "stats.jsonl"
    _write(log, [_drop_row(i) for i in range(1, 5)] + [_retrieve_row(9, index=0)])
    assert main(["tune", "--suggest", "--ledger", str(log),
                 "--corpus", str(tmp_path / "no-corpus")]) == 0
    out = capsys.readouterr().out
    assert "1 drop rule(s)" in out
    assert f"kb codegraph_explore {PATH}: retract:" in out
    assert out.rstrip().endswith("Suggestions only: nothing was changed.")


def test_tune_suggest_flags_reach_the_thresholds(tmp_path, capsys):
    log = tmp_path / "stats.jsonl"
    _write(log, [_drop_row(i) for i in range(1, 5)] + [_retrieve_row(9)])
    assert main(["tune", "--suggest", "--ledger", str(log), "--retract-rate", "0.5"]) == 0
    assert "retract" not in capsys.readouterr().out


def test_tune_suggest_without_a_ledger_fails(tmp_path, capsys):
    assert main(["tune", "--suggest", "--ledger", str(tmp_path / "missing.jsonl")]) == 2
    assert "no ledger" in capsys.readouterr().err


def test_tune_suggest_on_a_ledger_with_no_drops_says_so(tmp_path, capsys):
    log = tmp_path / "stats.jsonl"
    _write(log, [{**_drop_row(1), "dropped_paths": None}])
    assert main(["tune", "--suggest", "--ledger", str(log)]) == 0
    assert "No ledger row records a drop yet" in capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [("--retract-rate", "1.5"), ("--widen-rate", "-0.1"),
                                        ("--widen-min-drops", "0")])
def test_tune_suggest_rejects_out_of_range_thresholds(flag, value):
    with pytest.raises(SystemExit):
        main(["tune", "--suggest", flag, value])
