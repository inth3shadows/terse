"""A drop rule's FIRINGS in the ledger: the denominator of its retrieve rate (#252).

The ledger recorded every `terse.retrieve` billed to a rule (#251) but never how often that
rule dropped anything, so "2 retrieves" could mean 2 of 2 or 2 of 2,000. These tests pin
that each result row carries `drops: {rule path: spans}`, that `aggregate` joins it to the
retrieves as a rate, and that older rows are backfilled from `text_dropped` — labelled as
an estimate, never passed off as a count.
"""

from __future__ import annotations

import json

from terse.policy import Policy, Rule
from terse.proxy import Interceptor
from terse.stats import (
    aggregate,
    build_retrieve_record,
    build_retrieve_section,
    build_stats_writer,
    load_stats,
)

TEXT_DROP = Policy(rules=[Rule("codegraph_*", ("minify", "tabularize", "dictionary"),
                               fields={"$text.code_blocks": {"lossy": "drop-to-retrieve"}})])
PATH = "$text.code_blocks"


def _code(n_blocks: int) -> str:
    # Distinct bodies: handles are content-addressed, so identical spans share ONE handle
    # and are one dropped value (a single retrieve returns them all).
    blocks = ("```python\n" + (f"x_{i} = {i}\n" * 80) + "```\n" for i in range(n_blocks))
    return "intro\n" + "prose between\n".join(blocks)


def _drive(inter: Interceptor, mid: int, text: str) -> None:
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": "codegraph_explore"}}))
    inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": mid,
         "result": {"content": [{"type": "text", "text": text}]}}))


def _rows(log) -> list[dict]:
    return [r for r in load_stats(log) if not r.get("event")]


def test_identical_spans_are_one_dropped_value(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = Interceptor(TEXT_DROP, stats=build_stats_writer(log, "cg"))
    same = "```python\n" + ("x = 1\n" * 80) + "```\n"
    _drive(inter, 1, "a\n" + same + "b\n" + same)
    (row,) = _rows(log)
    assert row["drops"] == {PATH: 1}


def test_a_dropping_result_records_its_spans_per_rule_path(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = Interceptor(TEXT_DROP, stats=build_stats_writer(log, "cg"))
    _drive(inter, 1, _code(3))
    (row,) = _rows(log)
    assert row["drops"] == {PATH: 3}


def test_a_result_that_dropped_nothing_carries_no_drops_field(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = Interceptor(TEXT_DROP, stats=build_stats_writer(log, "cg"))
    _drive(inter, 1, "just prose, no code at all")
    (row,) = _rows(log)
    assert "drops" not in row


def test_one_results_drops_never_leak_into_the_next_row(tmp_path):
    log = tmp_path / "stats.jsonl"
    inter = Interceptor(TEXT_DROP, stats=build_stats_writer(log, "cg"))
    _drive(inter, 1, _code(2))
    _drive(inter, 2, "plain prose this time")
    first, second = _rows(log)
    assert first["drops"] == {PATH: 2} and "drops" not in second


def test_an_old_seven_argument_stats_callback_is_not_handed_drops():
    seen: list = []
    inter = Interceptor(Policy(rules=[Rule("*", ("minify",))]),
                        stats=lambda *a: seen.append(a))
    _drive(inter, 1, json.dumps({"a": [1, 2, 3]}))
    assert len(seen) == 1 and len(seen[0]) == 7


def _block(**kw):
    return {"ts": 1, "server": "cg", "tool": "codegraph_explore", "decision": "compressed",
            "raw_chars": 900, "out_chars": 300, "raw_tokens": 200, "out_tokens": 60, **kw}


def test_the_rate_is_retrieves_per_dropped_span():
    agg = aggregate([_block(drops={PATH: 3}), _block(drops={PATH: 1}),
                     build_retrieve_record("cg", "codegraph_explore", PATH, hit=True,
                                           payload="x" * 40)])
    (rule,) = agg["drop_rules"]
    assert (rule["blocks"], rule["spans"], rule["retrieves"]) == (2, 4, 1)
    assert rule["rate"] == 0.25 and rule["estimated"] == 0


def test_a_rule_that_dropped_but_was_never_retrieved_is_listed_at_zero():
    (rule,) = aggregate([_block(drops={PATH: 5})])["drop_rules"]
    assert rule["rate"] == 0.0 and rule["retrieves"] == 0


def test_pre_252_text_dropped_rows_are_backfilled_as_estimates():
    (rule,) = aggregate([_block(diff_reason="text_dropped"),
                         _block(diff_reason="text_dropped", drops={PATH: 2})])["drop_rules"]
    # The row WITH `drops` is counted from it, not double-counted by its diff_reason.
    assert (rule["blocks"], rule["spans"], rule["estimated"]) == (2, 3, 1)


def test_a_retrieve_with_no_drop_in_the_window_has_no_rate():
    (rule,) = aggregate([build_retrieve_record("cg", "codegraph_explore", PATH, hit=True,
                                               payload="x")])["drop_rules"]
    assert rule["spans"] == 0 and rule["rate"] is None


def test_malformed_drops_values_are_ignored_not_summed():
    agg = aggregate([_block(drops={PATH: True, "x": "3", "y": -1, "z": 2})])
    assert [(r["path"], r["spans"]) for r in agg["drop_rules"]] == [("z", 2)]


def test_the_report_shows_dropped_and_rate_and_says_the_rate_is_a_lower_bound():
    agg = aggregate([_block(drops={PATH: 4}), _block(diff_reason="text_dropped"),
                     build_retrieve_record("cg", "codegraph_explore", PATH, hit=True,
                                           payload="x" * 40)])
    text = "\n".join(build_retrieve_section(agg))
    assert "dropped" in text and "rate" in text
    line = next(ln for ln in text.splitlines() if PATH in ln)
    assert "~" in line and "20%" in line          # 1 retrieve / 5 spans, one estimated
    assert "LOWER bound" in text


def test_the_section_appears_for_a_rule_that_only_dropped():
    text = "\n".join(build_retrieve_section(aggregate([_block(drops={PATH: 7})])))
    assert PATH in text and "0%" in text
