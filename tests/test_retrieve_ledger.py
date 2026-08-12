"""A drop rule's COST: `terse.retrieve` round-trips recorded in the payload-free ledger (#251).

Before this, the ledger measured only the saving side of `drop-to-retrieve` — the tokens a
dropped field never spent. It could not see the model spending a whole extra tool call to
fetch that field back, so a rule dropping a field the model ALWAYS needs was
indistinguishable in the data from one dropping a field it never needs. These tests pin the
attribution (which rule caused the retrieve), the payload-free property, and — most
importantly — that a retrieve row can never be miscounted into the published savings figure.
"""

from __future__ import annotations

import json

from terse.policy import Policy, Rule
from terse.proxy import Interceptor
from terse.stats import RETRIEVE_EVENT, aggregate, build_retrieve_record

DROP = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                          fields={"result[].body": {"lossy": "drop-to-retrieve"}})])
TEXT_DROP = Policy(rules=[Rule("codegraph_*", ("minify", "tabularize", "dictionary"),
                               fields={"$text.code_blocks": {"lossy": "drop-to-retrieve"}})])


def _retrieve_call(mid, handle):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": "terse.retrieve", "arguments": {"handle": handle}}})


def _recorder():
    """Capture what the retrieve writer would have appended, without touching disk."""
    rows: list[tuple] = []

    def rec(server, tool, path, hit, payload):
        rows.append((server, tool, path, hit, payload))

    return rows, rec


def _drop_a_field(inter, tool="gh.api.list"):
    """Drive a real compression that commits a drop, and return the emitted handle."""
    from terse import lossy
    payload = json.dumps({"result": [{"id": 1, "body": "B" * 400}]})
    out = inter._compress(payload, tool)
    marker = json.loads(out)["result"][0]["body"]
    return marker[lossy.DROP_KEY]


# --- the record itself ---

def test_a_retrieve_record_never_enters_the_savings_total():
    """THE load-bearing guard. `aggregate` skips any row without int raw_chars/out_chars.
    A retrieve row must therefore never carry those keys — otherwise every retrieve would
    be counted as a compressed block and its bytes folded into the published savings
    percentage, silently corrupting the one number terse publishes about itself."""
    rec = build_retrieve_record("gh", "gh.api.list", "result[].body", hit=True,
                                payload="x" * 500)
    agg = aggregate([rec])
    assert agg["total"]["blocks"] == 0
    assert agg["total"]["raw_chars"] == 0 and agg["total"]["out_chars"] == 0
    # ...and the cost is not merely discarded — it lands on its own axis.
    assert agg["retrieves"][0]["calls"] == 1 and agg["retrieves"][0]["bytes"] == 500

    # TWO independent guards keep it out, and both are pinned here because either alone
    # would be enough to pass this test while leaving the other free to rot:
    #   1. the `event` marker, checked first
    assert rec["event"] == RETRIEVE_EVENT
    assert aggregate([dict(rec, raw_chars=500, out_chars=500)])["total"]["blocks"] == 0
    #   2. the absence of the result-record size keys, which is what excludes a row whose
    #      event marker was lost (an older writer, a hand-edited ledger)
    assert "raw_chars" not in rec and "out_chars" not in rec
    stripped = {k: v for k, v in rec.items() if k != "event"}
    assert aggregate([stripped])["total"]["blocks"] == 0
    # Control: a row that is BOTH unmarked and result-shaped IS counted — so the two
    # assertions above are about the guards, not about `aggregate` ignoring everything.
    assert aggregate([dict(stripped, raw_chars=500, out_chars=500)])["total"]["blocks"] == 1


def test_the_report_prints_the_cost_beside_the_rule_that_caused_it():
    """The readout, not just the data. Instrumentation nothing renders is instrumentation
    nobody acts on — and this section exists precisely because the per-tool table above it
    shows a lossy rule's saving with no counterweight."""
    from terse.stats import build_retrieve_section
    agg = aggregate([
        build_retrieve_record("kb", "codegraph_explore", "$text.code_blocks",
                              hit=True, payload="z" * 900),
        build_retrieve_record("kb", "codegraph_explore", "$text.code_blocks", hit=False),
    ])
    # The counters themselves, not just that a row exists: a hit and a miss are different
    # outcomes and folding one into the other is invisible in a rendered row otherwise.
    row = agg["retrieves"][0]
    assert (row["calls"], row["hits"], row["misses"]) == (2, 1, 1)
    assert row["bytes"] == 900               # the miss contributed 0, the hit 900

    text = "\n".join(build_retrieve_section(agg))
    assert "drop-to-retrieve cost" in text
    assert "codegraph_explore" in text and "$text.code_blocks" in text
    # The miss FOOTNOTE, which only renders when misses > 0 — not the bare word "miss",
    # which also appears in the column header and so matched even when misses were
    # miscounted as hits (this assertion previously survived that mutation).
    assert "1 miss(es)" in text
    assert "returned nothing" in text
    assert "z" * 20 not in text              # ...and still no payload


def test_the_cost_table_falls_back_to_chars_exactly_as_the_savings_table_does():
    """On a tiktoken-less ledger the token column is all zeros. The per-tool table above
    already switches to chars for precisely that reason; rendering only tokens here turned
    the whole cost table into a wall of nothing while the byte counts sat right there,
    known. A cost the operator cannot read is a cost they will not act on."""
    from terse.stats import build_retrieve_section
    agg = aggregate([build_retrieve_record("kb", "codegraph_explore", "$text.code_blocks",
                                           hit=True, payload="z" * 900)])
    # Simulate the tiktoken-less ledger: sizes known, tokens not.
    agg["retrieves"][0]["tokens"] = 0
    agg["retrieves"][0]["untokenized"] = 1

    as_chars = "\n".join(build_retrieve_section(agg, use_tokens=False))
    assert "chr" in as_chars and "900" in as_chars
    # The tiktoken footnote belongs only to the token rendering — under the char fallback
    # nothing is uncounted, so printing it would contradict the column beside it.
    assert "uncounted" not in as_chars

    as_tokens = "\n".join(build_retrieve_section(agg, use_tokens=True))
    assert "tok" in as_tokens and "uncounted" in as_tokens


def test_the_report_section_is_absent_rather_than_empty_when_nothing_was_recorded():
    """An empty section would read as "your drop rules cost nothing" — a claim a ledger
    written before this feature (or an install with no drop rule) cannot make."""
    from terse.stats import build_retrieve_section
    assert build_retrieve_section(aggregate([])) == []


def test_a_retrieve_record_is_payload_free():
    secret = "SUPER-SECRET-BODY-VALUE"
    rec = build_retrieve_record("gh", "gh.api.list", "result[].body", hit=True,
                                payload=secret)
    assert secret not in json.dumps(rec)
    # ...but the SIZE of it is recorded — that is the cost being measured.
    assert rec["bytes"] == len(secret)
    assert rec["event"] == RETRIEVE_EVENT


def test_a_miss_records_zero_bytes_and_hit_false():
    """A miss is the worst cell in the table — the model spent a call and got nothing
    back — so it has to be distinguishable from a hit, not merely absent."""
    rec = build_retrieve_record("gh", "gh.api.list", "result[].body", hit=False)
    assert rec["hit"] is False and rec["bytes"] == 0
    hit = build_retrieve_record("gh", "gh.api.list", "result[].body", hit=True, payload="ab")
    assert hit["hit"] is True and hit["bytes"] == 2


# --- attribution through the real proxy path ---

def test_a_json_retrieve_is_billed_to_the_rule_that_dropped_the_field():
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec)
    handle = _drop_a_field(inter)
    assert inter.answer_retrieve(_retrieve_call(1, handle)) is not None
    assert len(rows) == 1
    _srv, tool, path, hit, payload = rows[0]
    assert (tool, path, hit) == ("gh.api.list", "result[].body", True)
    assert payload == "B" * 400          # the cost measured is the real returned value


def test_a_text_retrieve_is_billed_to_its_text_selector():
    """The fleet's only lossy-by-default rule is a text selector, so this is the path
    that actually fires in production — attribution has to survive it, not just the
    JSON-field path."""
    rows, rec = _recorder()
    inter = Interceptor(TEXT_DROP, stats_retrieve=rec)
    text = "prose before\n```python\n" + ("x = 1\n" * 80) + "```\nprose after\n"
    out = inter._compress(text, "codegraph_explore")
    assert "__terse_dropped__" in out
    handle = json.loads(out.split("\n")[1])["__terse_dropped__"]
    assert inter.answer_retrieve(_retrieve_call(2, handle)) is not None
    assert [(t, p, h) for _s, t, p, h, _ in rows] == [
        ("codegraph_explore", "$text.code_blocks", True)]


def test_an_unattributed_handle_still_records_rather_than_vanishing():
    """A handle with no provenance — put straight into the store, or surviving from before
    this feature — must still be billed SOMEWHERE. Dropping the row would under-count the
    cost side and bias every retune toward "drops are free"."""
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec)
    inter._drop_put("orphan", "value")
    inter.answer_retrieve(_retrieve_call(3, "orphan"))
    assert len(rows) == 1 and rows[0][3] is True
    assert rows[0][2] == ""             # empty path == "attribution unknown"


def test_a_miss_is_recorded_through_the_proxy_path_too():
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec)
    reply = json.loads(inter.answer_retrieve(_retrieve_call(4, "never-existed")))
    assert reply["result"]["isError"] is True
    assert len(rows) == 1 and rows[0][3] is False and rows[0][4] == ""


# --- the staged/commit contract, and store lifecycle ---

def test_a_failed_recoverability_gate_leaves_no_attribution_behind(monkeypatch):
    """Origins are staged with the values and published on the SAME commit as them.

    This pins the ORDER, not merely the outcome: `apply_drops` runs and stages provenance,
    and only then does the gate refuse. Publishing at staging time instead of at commit
    time would leave an orphan origin describing a drop that never reached the store — and
    a later retrieve (of some other handle) could be billed to a rule that never fired.

    An earlier version of this test used a never-lossy server, which returns BEFORE
    `apply_drops` is ever called; it passed against the inverted implementation and pinned
    nothing. Forcing the gate itself to fail is what makes the ordering observable."""
    from terse import lossy as lossy_mod
    from terse import policy as policy_mod
    pol = Policy(rules=[Rule("gh.*", ("minify",),
                             fields={"result[].body": {"lossy": "drop-to-retrieve"}})])
    payload = json.dumps({"result": [{"id": 1, "body": "B" * 400}]})
    committed: dict = {}

    monkeypatch.setattr(lossy_mod, "droppable_loss", lambda *a, **k: False)
    refused = policy_mod.apply(payload, "gh.api.list", pol,
                               drop_sink=committed.__setitem__, server="gh")
    assert refused.drop_origins == {}, "provenance published despite a refused drop"
    assert committed == {}, "value committed despite a refused drop"
    assert refused.text == json.dumps(json.loads(payload), separators=(",", ":"))

    # Control: with the real gate the SAME payload does produce provenance — so the
    # assertions above are about the gate, not about dead plumbing.
    monkeypatch.undo()
    ok = policy_mod.apply(payload, "gh.api.list", pol,
                          drop_sink=committed.__setitem__, server="gh")
    assert list(ok.drop_origins.values()) == [("gh.api.list", "result[].body")]
    assert len(committed) == 1


def test_a_never_lossy_server_produces_no_attribution():
    """The other way a drop can fail to commit: the server-level lossless floor, which
    returns before the drop path runs at all."""
    from terse import policy as policy_mod
    rule = Rule("gh.*", ("minify",), fields={"result[].body": {"lossy": "drop-to-retrieve"}})
    pol_never = Policy(rules=[rule], never_lossy_servers=frozenset({"gh"}))
    committed: dict = {}
    applied = policy_mod.apply(json.dumps({"result": [{"id": 1, "body": "B" * 400}]}),
                               "gh.api.list", pol_never,
                               drop_sink=committed.__setitem__, server="gh")
    assert applied.drop_origins == {} and committed == {}


def test_attribution_is_evicted_in_lockstep_with_the_value():
    """The store is explicitly capped; the origins map mirrors it and must be capped by
    the same eviction, or it grows without bound beside a bounded dict."""
    inter = Interceptor(DROP)
    inter._drop_origin["h0"] = ("gh.api.list", "result[].body")
    inter._drop_put("h0", "v0")
    assert "h0" in inter._drop_origin
    original_max = Interceptor.DROPPED_MAX
    try:
        Interceptor.DROPPED_MAX = 1
        inter._drop_put("h1", "v1")      # evicts h0
    finally:
        Interceptor.DROPPED_MAX = original_max
    assert "h0" not in inter.dropped
    assert "h0" not in inter._drop_origin


def test_a_reconnect_clears_attribution_with_the_store():
    inter = Interceptor(DROP, server_name="gh")
    handle = _drop_a_field(inter)
    assert inter._drop_origin.get(handle) is not None
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {}}))
    assert inter.dropped == {} and inter._drop_origin == {}


def test_a_retrieve_is_billed_to_the_peer_that_dropped_it_not_the_one_that_answers():
    """Under multiproxy the router answers EVERY `terse.retrieve` through `peers[0]`
    (`_route_call`), so the answering Interceptor is almost never the one that dropped the
    value — and only `peers[0]`'s writer is ever invoked.

    That made the `server` column name the wrong peer: a `kb` rule's cost was filed under
    `gh`, where it does not join with that tool's own result rows and points an operator at
    the wrong peer. Caught in review; the earlier version of this test invoked peer B's
    writer DIRECTLY, a call production never makes, so it passed throughout.

    The label is therefore captured at DROP time and travels in the shared origins map."""
    from collections import OrderedDict
    from threading import Lock

    store: OrderedDict = OrderedDict()
    lock, boxed, origins = Lock(), [0], {}
    rows_first, rec_first = _recorder()
    # peers[0] — the one the router always routes retrieves through. Its own writer is
    # bound to ITS label, which is exactly the mislabeling this pins.
    peer_first = Interceptor(DROP, store=store, store_lock=lock, dropped_bytes=boxed,
                             origins=origins, stats_retrieve=rec_first,
                             ledger_label="gh")
    peer_other = Interceptor(DROP, store=store, store_lock=lock, dropped_bytes=boxed,
                             origins=origins, ledger_label="kb")
    handle = _drop_a_field(peer_other)
    # Exactly how `_route_call` does it: peers[0] answers a handle another peer dropped.
    assert peer_first.answer_retrieve(_retrieve_call(7, handle)) is not None
    assert len(rows_first) == 1
    server, tool, path, hit, _payload = rows_first[0]
    assert server == "kb", "billed to the answering peer instead of the dropping one"
    assert (tool, path, hit) == ("gh.api.list", "result[].body", True)


def test_an_unattributed_retrieve_falls_back_to_the_answering_proxys_own_label():
    """A handle with no provenance still has to be billed somewhere. There is no better
    answer available than the proxy that served it, and dropping the row entirely would
    under-count the cost side."""
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec, ledger_label="gh")
    inter._drop_put("orphan", "value")
    inter.answer_retrieve(_retrieve_call(11, "orphan"))
    assert rows[0][0] == "gh" and rows[0][2] == ""


def test_a_broken_ledger_never_breaks_a_served_retrieve():
    """Stats is never load-bearing: the value is already resolved when the row is written,
    so a ledger failure must not turn a good retrieve into an error reply."""
    def exploding(server, tool, path, hit, payload):
        raise OSError("disk full")

    inter = Interceptor(DROP, stats_retrieve=exploding)
    handle = _drop_a_field(inter)
    reply = json.loads(inter.answer_retrieve(_retrieve_call(8, handle)))
    assert not reply["result"].get("isError")
    assert reply["result"]["content"][0]["text"] == "B" * 400
