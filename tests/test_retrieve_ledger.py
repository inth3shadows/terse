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

    def rec(tool, path, hit, payload):
        rows.append((tool, path, hit, payload))

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
    tool, path, hit, payload = rows[0]
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
    assert [(t, p, h) for t, p, h, _ in rows] == [
        ("codegraph_explore", "$text.code_blocks", True)]


def test_an_unattributed_handle_still_records_rather_than_vanishing():
    """A handle with no provenance — put straight into the store, or surviving from before
    this feature — must still be billed SOMEWHERE. Dropping the row would under-count the
    cost side and bias every retune toward "drops are free"."""
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec)
    inter._drop_put("orphan", "value")
    inter.answer_retrieve(_retrieve_call(3, "orphan"))
    assert len(rows) == 1 and rows[0][2] is True
    assert rows[0][1] == ""             # empty path == "attribution unknown"


def test_a_miss_is_recorded_through_the_proxy_path_too():
    rows, rec = _recorder()
    inter = Interceptor(DROP, stats_retrieve=rec)
    reply = json.loads(inter.answer_retrieve(_retrieve_call(4, "never-existed")))
    assert reply["result"]["isError"] is True
    assert len(rows) == 1 and rows[0][2] is False and rows[0][3] == ""


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


def test_attribution_is_shared_across_multiproxy_peers():
    """Under multiproxy any peer's Interceptor may answer a retrieve for a handle a
    DIFFERENT peer dropped (the store is shared). The origins map is shared for the same
    reason — a private one would lose attribution on exactly the fleet shape that has a
    lossy-by-default rule."""
    from collections import OrderedDict
    from threading import Lock

    store: OrderedDict = OrderedDict()
    lock, boxed, origins = Lock(), [0], {}
    rows, rec = _recorder()
    peer_a = Interceptor(DROP, store=store, store_lock=lock, dropped_bytes=boxed,
                         origins=origins)
    peer_b = Interceptor(DROP, store=store, store_lock=lock, dropped_bytes=boxed,
                         origins=origins, stats_retrieve=rec)
    handle = _drop_a_field(peer_a)
    # peer B answers for a handle peer A dropped, and still knows whose rule it was.
    assert peer_b.answer_retrieve(_retrieve_call(7, handle)) is not None
    assert [(t, p) for t, p, _, _ in rows] == [("gh.api.list", "result[].body")]


def test_a_broken_ledger_never_breaks_a_served_retrieve():
    """Stats is never load-bearing: the value is already resolved when the row is written,
    so a ledger failure must not turn a good retrieve into an error reply."""
    def exploding(tool, path, hit, payload):
        raise OSError("disk full")

    inter = Interceptor(DROP, stats_retrieve=exploding)
    handle = _drop_a_field(inter)
    reply = json.loads(inter.answer_retrieve(_retrieve_call(8, handle)))
    assert not reply["result"].get("isError")
    assert reply["result"]["content"][0]["text"] == "B" * 400
