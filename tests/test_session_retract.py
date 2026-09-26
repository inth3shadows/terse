"""Per-session retract (#252): once a session has fetched a rule's dropped values back
`retract_after` times, that tool's later results keep the field inline until reconnect.

Measured 2026-09-26 over 151 live codegraph_explore results: after a session's first
retrieve, 72% of its later results needed one too (38% before), and stopping the drop there
cut the results needing any retrieve from 80 to 32.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from threading import Lock

from terse.policy import Policy, Rule, _lossy_warnings
from terse.proxy import Interceptor

TIERS = ("minify", "tabularize", "dictionary")
TEXT = "prose before\n```python\n" + ("x = 1\n" * 80) + "```\nprose after\n"


def _policy(**spec):
    return Policy(rules=[Rule("codegraph_*", TIERS, fields={
        "$text.code_blocks": {"lossy": "drop-to-retrieve", **spec}})])


def _retrieve(inter, handle, mid=9):
    msg = json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                      "params": {"name": "terse.retrieve", "arguments": {"handle": handle}}})
    return inter.answer_retrieve(msg)


def _handle(out):
    return json.loads(out.split("\n")[1])["__terse_dropped__"]


def test_without_retract_after_drops_keep_happening():
    inter = Interceptor(_policy())
    _retrieve(inter, _handle(inter._compress(TEXT, "codegraph_explore")))
    assert "__terse_dropped__" in inter._compress(TEXT, "codegraph_explore")


def test_retract_after_one_hit_keeps_later_results_inline():
    inter = Interceptor(_policy(retract_after=1))
    first = inter._compress(TEXT, "codegraph_explore")
    assert "__terse_dropped__" in first
    _retrieve(inter, _handle(first))
    assert inter._compress(TEXT, "codegraph_explore") == TEXT


def test_a_miss_does_not_count_toward_retract():
    inter = Interceptor(_policy(retract_after=1))
    inter._compress(TEXT, "codegraph_explore")
    _retrieve(inter, "deadbeefdeadbeef")
    assert "__terse_dropped__" in inter._compress(TEXT, "codegraph_explore")


def test_retract_after_two_needs_two_hits():
    inter = Interceptor(_policy(retract_after=2))
    _retrieve(inter, _handle(inter._compress(TEXT, "codegraph_explore")))
    second = inter._compress(TEXT, "codegraph_explore")
    assert "__terse_dropped__" in second
    _retrieve(inter, _handle(second))
    assert inter._compress(TEXT, "codegraph_explore") == TEXT


def test_retract_is_scoped_to_the_tool_that_was_retrieved():
    inter = Interceptor(_policy(retract_after=1))
    _retrieve(inter, _handle(inter._compress(TEXT, "codegraph_explore")))
    assert "__terse_dropped__" in inter._compress(TEXT, "codegraph_node")


def test_reconnect_resets_the_retract():
    inter = Interceptor(_policy(retract_after=1))
    _retrieve(inter, _handle(inter._compress(TEXT, "codegraph_explore")))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {"clientInfo": {"name": "claude-code"}}}))
    assert "__terse_dropped__" in inter._compress(TEXT, "codegraph_explore")


def test_router_peers_share_the_retract():
    # The router answers every retrieve through peers[0], so a hit on a handle another
    # peer dropped must retract that other peer's rule.
    store: OrderedDict = OrderedDict()
    lock, boxed, origins, hits = Lock(), [0], {}, {}
    pol = _policy(retract_after=1)
    first = Interceptor(pol, store=store, store_lock=lock, dropped_bytes=boxed,
                        origins=origins, retrieve_hits=hits, ledger_label="a")
    other = Interceptor(pol, store=store, store_lock=lock, dropped_bytes=boxed,
                        origins=origins, retrieve_hits=hits, ledger_label="codegraph")
    _retrieve(first, _handle(other._compress(TEXT, "codegraph_explore")))
    assert other._compress(TEXT, "codegraph_explore") == TEXT


def test_invalid_retract_after_warns():
    for bad in (0, -1, True, "1"):
        rule = _policy(retract_after=bad).rules[0]
        assert any("retract_after" in w for w in _lossy_warnings(rule)), bad
