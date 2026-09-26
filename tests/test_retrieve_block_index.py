"""The retrieve ledger row carries the dropped block's POSITION in its result (#252).

`keep_first` and `retract_after` are tuned by hand from a one-off transcript scan. For
`terse tune` to suggest them from live evidence, a retrieve row has to say WHICH block of
the result the model fetched back. The index counts qualifying blocks (at or above the
rule's `min`) in result order, INCLUDING blocks kept inline by `keep_first` -- the same
numbering `keep_first` counts in, so "index < N" reads directly as "keep_first N would
have kept it". JSON-field drops have no block position and record none.
"""

from __future__ import annotations

import json

from terse import lossy
from terse.policy import Policy, Rule
from terse.proxy import Interceptor
from terse.stats import aggregate, build_retrieve_record, build_retrieve_writer

JSON_DROP = Policy(rules=[Rule("gh.*", ("minify",),
                               fields={"result[].body": {"lossy": "drop-to-retrieve"}})])


def _text_policy(**spec):
    return Policy(rules=[Rule("codegraph_*", ("minify",),
                              fields={"$text.code_blocks":
                                      {"lossy": "drop-to-retrieve", **spec}})])


def _block(body: str, n: int = 80) -> str:
    return "```python\n" + (body + "\n") * n + "```\n"


def _retrieve_call(mid, handle):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": "terse.retrieve", "arguments": {"handle": handle}}})


def _handles(out: str) -> list[str]:
    return [json.loads(line)[lossy.DROP_KEY] for line in out.splitlines()
            if line.startswith("{") and lossy.DROP_KEY in line]


def _recorder():
    rows: list[tuple] = []

    def rec(server, tool, path, hit, payload, index=None):
        rows.append((tool, path, hit, index))

    return rows, rec


# --- the drop step records the position ---

def test_each_dropped_block_records_its_position_among_qualifying_blocks():
    rule = _text_policy().rules[0]
    text = "intro\n" + _block("a = 0") + "mid\n" + _block("b = 1") + _block("c = 2")
    origin: dict = {}
    out = lossy.apply_text_drops(text, rule, "codegraph_explore", lambda h, v: None, origin)
    hs = _handles(out)
    assert [origin[h] for h in hs] == [
        ("codegraph_explore", "$text.code_blocks", 0),
        ("codegraph_explore", "$text.code_blocks", 1),
        ("codegraph_explore", "$text.code_blocks", 2)]


def test_a_block_under_the_floor_does_not_take_a_position():
    """Same numbering `keep_first` counts in: a sub-`min` block never drops and never
    counts, so it must not shift the index a tune suggestion is read against."""
    rule = _text_policy().rules[0]
    text = "```\ntiny\n```\n" + _block("a = 0") + _block("b = 1")
    origin: dict = {}
    out = lossy.apply_text_drops(text, rule, "codegraph_explore", lambda h, v: None, origin)
    assert [origin[h][2] for h in _handles(out)] == [0, 1]


def test_kept_blocks_still_count_so_the_index_reads_against_keep_first():
    rule = _text_policy(keep_first=1).rules[0]
    text = _block("a = 0") + _block("b = 1") + _block("c = 2")
    origin: dict = {}
    out = lossy.apply_text_drops(text, rule, "codegraph_explore", lambda h, v: None, origin)
    assert [origin[h][2] for h in _handles(out)] == [1, 2]


def test_identical_blocks_collapse_to_one_handle_indexed_at_the_first_position():
    """Handles are content-addressed over (tool, path, bytes), so the same span at two
    positions is ONE handle. The first position wins, deterministically."""
    rule = _text_policy().rules[0]
    dup = _block("same = 1")
    text = dup + _block("other = 2") + dup
    origin: dict = {}
    out = lossy.apply_text_drops(text, rule, "codegraph_explore", lambda h, v: None, origin)
    hs = _handles(out)
    assert len(hs) == 3 and hs[0] == hs[2] and len(set(hs)) == 2
    assert origin[hs[0]][2] == 0
    assert origin[hs[1]][2] == 1


def test_a_json_field_drop_records_no_position():
    from terse import policy as policy_mod
    payload = json.dumps({"result": [{"id": 1, "body": "B" * 400}]})
    applied = policy_mod.apply(payload, "gh.api.list", JSON_DROP,
                               drop_sink={}.__setitem__, server="gh")
    assert list(applied.drop_origins.values()) == [("gh.api.list", "result[].body", None)]


# --- the ledger record ---

def test_the_record_carries_index_only_when_known():
    with_index = build_retrieve_record("kb", "codegraph_explore", "$text.code_blocks",
                                       hit=True, payload="x", index=2)
    assert with_index["index"] == 2
    without = build_retrieve_record("gh", "gh.api.list", "result[].body", hit=True,
                                    payload="x")
    assert "index" not in without


def test_old_rows_without_index_aggregate_exactly_like_new_ones():
    old = build_retrieve_record("kb", "codegraph_explore", "$text.code_blocks",
                                hit=True, payload="x" * 50)
    new = dict(old, index=0)
    assert aggregate([old])["retrieves"] == aggregate([new])["retrieves"]


def test_the_writer_puts_the_index_on_disk(tmp_path):
    log = tmp_path / "stats.jsonl"
    write = build_retrieve_writer(log, "kb")
    write("kb", "codegraph_explore", "$text.code_blocks", True, "SECRET" * 10, index=3)
    write("gh", "gh.api.list", "result[].body", True, "SECRET" * 10)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["index"] == 3 and "index" not in rows[1]
    assert "SECRET" not in log.read_text()


# --- through the real proxy path ---

def test_a_text_retrieve_reports_the_block_position():
    rows, rec = _recorder()
    inter = Interceptor(_text_policy(), stats_retrieve=rec)
    out = inter._compress("prose\n" + _block("a = 0") + _block("b = 1"), "codegraph_explore")
    h0, h1 = _handles(out)
    inter.answer_retrieve(_retrieve_call(1, h1))
    inter.answer_retrieve(_retrieve_call(2, h0))
    assert rows == [("codegraph_explore", "$text.code_blocks", True, 1),
                    ("codegraph_explore", "$text.code_blocks", True, 0)]


def test_a_json_retrieve_reports_no_position_and_keeps_the_five_arg_callback():
    """A JSON drop has no index, so the callback is called exactly as before (#251) -- a
    5-argument writer still works on that path."""
    rows: list[tuple] = []
    inter = Interceptor(JSON_DROP,
                        stats_retrieve=lambda s, t, p, h, v: rows.append((t, p, h)))
    out = inter._compress(json.dumps({"result": [{"id": 1, "body": "B" * 400}]}),
                          "gh.api.list")
    handle = json.loads(out)["result"][0]["body"][lossy.DROP_KEY]
    inter.answer_retrieve(_retrieve_call(1, handle))
    assert rows == [("gh.api.list", "result[].body", True)]


def test_retract_after_still_counts_hits_with_the_widened_origin():
    inter = Interceptor(_text_policy(retract_after=1))
    out = inter._compress(_block("a = 0"), "codegraph_explore")
    (h,) = _handles(out)
    inter.answer_retrieve(_retrieve_call(1, h))
    assert inter._retrieve_hits == {("codegraph_explore", "$text.code_blocks"): 1}
    again = inter._compress(_block("b = 1"), "codegraph_explore")
    assert lossy.DROP_KEY not in again
