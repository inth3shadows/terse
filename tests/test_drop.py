"""Tier-1 lossy drop-to-retrieve (#10) pure primitives: the content-addressed handle, the
size floor, the critical denylist, and the `droppable_loss` gate that replaces the round-
trip gate with a recoverability invariant. The stateful store + retrieve tool live in the
proxy; here the store is a fake dict injected via sink/resolve."""

from __future__ import annotations

import json

from terse import lossy, transforms
from terse.policy import Policy, Rule, _lossy_warnings, apply


def _rule(fields):
    return Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"), fields=fields)


def _sink_store():
    """A fake per-session store: sink persists, resolve reads (KeyError on miss)."""
    store: dict = {}

    def sink(handle, value):
        store[handle] = value

    def resolve(handle):
        return store[handle]

    return store, sink, resolve


# --- handle ---
def test_handle_is_deterministic_and_content_addressed():
    h1 = lossy._handle("t", "p", "abc")
    h2 = lossy._handle("t", "p", "abc")
    assert h1 == h2 and len(h1) == lossy.HANDLE_LEN
    assert lossy._handle("t", "p", "abcd") != h1            # content-sensitive
    assert lossy._handle("t", "other", "abc") != h1         # path-sensitive


# --- apply_drops ---
def test_apply_drops_replaces_large_field_and_persists():
    obj = {"result": [{"id": i, "body": "B" * 300} for i in range(3)]}
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    out = lossy.apply_drops(obj, rule, "gh.api.x", sink)
    for r in out["result"]:
        assert lossy._is_drop_marker(r["body"])
        assert resolve(r["body"][lossy.DROP_KEY]) == "B" * 300   # exact recovery
    assert [r["id"] for r in out["result"]] == [0, 1, 2]         # unmarked fields intact
    assert obj["result"][0]["body"] == "B" * 300                 # original untouched
    assert len(store) == 1                                       # identical bodies dedup


def test_under_floor_value_is_left_in_place():
    obj = {"result": [{"id": 1, "body": "short"}]}
    store, sink, _ = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve", "min": 200}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    assert out["result"][0]["body"] == "short"                  # untouched
    assert store == {}                                          # nothing persisted


def test_critical_field_is_never_dropped():
    obj = {"result": [{"id": 1, "body": "B" * 300}]}
    store, sink, _ = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve", "critical": True}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    assert out == obj
    assert store == {}
    assert lossy.critical_paths(rule) == {"result[].body"}


def test_drop_a_list_valued_field_wholesale():
    obj = {"result": [{"id": 1, "tags": ["x"] * 100}]}          # serialized well over floor
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].tags": {"lossy": "drop-to-retrieve"}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    marker = out["result"][0]["tags"]
    assert lossy._is_drop_marker(marker)
    assert resolve(marker[lossy.DROP_KEY]) == ["x"] * 100


# --- droppable_loss gate ---
def test_gate_accepts_a_recoverable_drop():
    obj = {"result": [{"id": i, "body": "B" * 300} for i in range(3)]}
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    assert lossy.droppable_loss(obj, out, rule, resolve)


def test_gate_accepts_under_floor_untouched():
    obj = {"result": [{"id": 1, "body": "short"}]}
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve", "min": 200}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    assert lossy.droppable_loss(obj, out, rule, resolve)        # nothing changed


def test_gate_rejects_change_to_an_unmarked_field():
    obj = {"result": [{"id": 1, "body": "B" * 300, "url": "keep"}]}
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    out["result"][0]["url"] = "TAMPERED"                        # unmarked field changed
    assert not lossy.droppable_loss(obj, out, rule, resolve)


def test_gate_fails_closed_when_handle_unresolvable():
    obj = {"result": [{"id": 1, "body": "B" * 300}]}
    store, sink, resolve = _sink_store()
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    out = lossy.apply_drops(obj, rule, "t", sink)
    store.clear()                                               # eviction / post-reconnect
    assert not lossy.droppable_loss(obj, out, rule, resolve)


def test_gate_rejects_a_marker_that_resolves_to_the_wrong_value():
    obj = {"result": [{"id": 1, "body": "B" * 300}]}
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    out = {"result": [{"id": 1, "body": {lossy.DROP_KEY: "deadbeef", "bytes": 300,
                                         "retrieve": lossy.RETRIEVE_TOOL}}]}
    resolve = {"deadbeef": "WRONG"}.__getitem__
    assert not lossy.droppable_loss(obj, out, rule, resolve)


def test_gate_fails_closed_on_shape_mismatch():
    # path says result[].body but result is an object, not a list -> PathError -> fail closed
    obj = {"result": {"body": "B" * 300}}
    rule = _rule({"result[].body": {"lossy": "drop-to-retrieve"}})
    assert not lossy.droppable_loss(obj, obj, rule, {}.__getitem__)


# --- end to end through policy.apply ---
def test_apply_executes_drop_with_a_sink_and_warns_not_lossless():
    obj = {"result": [{"id": i, "body": "LONGBODY" * 40} for i in range(4)]}
    raw = json.dumps(obj)
    store: dict = {}
    p = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                           fields={"result[].body": {"lossy": "drop-to-retrieve"}})])
    res = apply(raw, "gh.api.x", p, drop_sink=store.__setitem__)
    assert any("NOT lossless" in w for w in res.warnings)
    decoded = transforms.decompress(res.text)
    for r in decoded["result"]:
        assert lossy._is_drop_marker(r["body"])
        assert store[r["body"][lossy.DROP_KEY]] == "LONGBODY" * 40   # exactly recoverable
    assert [r["id"] for r in decoded["result"]] == [0, 1, 2, 3]      # ids preserved
    assert len(res.text) < len(raw)


def test_apply_without_a_sink_keeps_drop_lossless_and_warns():
    obj = {"result": [{"id": 1, "body": "B" * 300}]}
    raw = json.dumps(obj)
    p = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                           fields={"result[].body": {"lossy": "drop-to-retrieve"}})])
    res = apply(raw, "gh.api.x", p)                                  # no drop_sink
    assert any("needs the proxy store" in w for w in res.warnings)
    assert transforms.decompress(res.text) == obj                   # lossless fallback


def test_drop_is_no_longer_warned_as_unimplemented_but_summarize_still_is():
    assert not any("not implemented" in w
                   for w in _lossy_warnings(_rule({"x": {"lossy": "drop-to-retrieve"}})))
    assert any("not implemented" in w
               for w in _lossy_warnings(_rule({"x": {"lossy": "summarize"}})))
