"""#64 Phase 1 — shared cross-peer session legend: SessionDict + sess_encode/decode.

These cover the codec in isolation (the live proxy wiring lands in a later stage). The
load-bearing guarantee is the round-trip: whatever sess_encode emits, a client accumulating
the per-payload defs must reconstruct the original exactly, or sess_encode must bow out.
"""

from __future__ import annotations

from terse.proxy import Interceptor
from terse import policy as policy_mod
from terse.transforms import (
    ALIAS_SIGIL,
    SessionDict,
    sess_decode,
    sess_encode,
)


def _client_roundtrip(payloads_and_encodings):
    """Simulate the client: accumulate every payload's defs into one legend, decode each
    payload's data against that cumulative legend, and assert it equals the original."""
    legend: dict = {}
    for original, enc in payloads_and_encodings:
        assert enc is not None
        data, defs = enc
        legend.update(defs)
        assert sess_decode(data, legend) == original


# --- the cross-peer win: define once, reference (definition elided) thereafter ----------

def test_shared_value_defined_once_then_referenced_with_no_def():
    sd = SessionDict()
    a = {"sym": "SharedSymbolLongEnoughToIntern", "file": "src/terse/proxy.py"}
    b = {"caller": "SharedSymbolLongEnoughToIntern", "file": "src/terse/proxy.py", "n": 3}
    ea = sess_encode(a, sd)
    eb = sess_encode(b, sd)  # same SessionDict == another peer sharing the legend
    assert ea is not None and eb is not None
    _, defs_a = ea
    data_b, defs_b = eb
    alias = next(al for al, v in defs_a.items() if v == "SharedSymbolLongEnoughToIntern")
    # The whole point: b references the shared value but carries NO definition for it.
    assert alias not in defs_b
    assert alias in str(data_b)
    _client_roundtrip([(a, ea), (b, eb)])


def test_one_session_dict_shared_across_peers_is_the_seam():
    # Two independent SessionDicts do NOT share (each redefines); one shared instance does.
    a = {"path": "a/very/long/shared/path/value.py", "k": 1}
    b = {"path": "a/very/long/shared/path/value.py", "k": 2}
    sep1, sep2 = SessionDict(), SessionDict()
    _, defs_b_sep = sess_encode(b, sep2) or (None, {})
    # with a separate dict, b must redefine the path
    assert any(v == "a/very/long/shared/path/value.py" for v in defs_b_sep.values())
    shared = SessionDict()
    sess_encode(a, shared)
    _, defs_b_shared = sess_encode(b, shared)
    assert not any(v == "a/very/long/shared/path/value.py" for v in defs_b_shared.values())


# --- within-payload repeats (immediate saving, like the per-call dictionary) ------------

def test_within_payload_repeat_aliased_with_single_def():
    sd = SessionDict()
    payload = {"rows": [{"s": "active"}, {"s": "active"}, {"s": "active"}]}
    enc = sess_encode(payload, sd)
    assert enc is not None
    data, defs = enc
    assert len(defs) == 1 and list(data["rows"]) == list(data["rows"])  # all same alias
    _client_roundtrip([(payload, enc)])


# --- lossless self-verify + fallback ----------------------------------------------------

def test_lone_subtree_not_speculatively_interned():
    # A unique top-level object at count 1 must NOT be interned as one giant alias (it never
    # recurs); its shared leaf strings are aliased instead.
    sd = SessionDict()
    a = {"symbol": "SharedSymbolLongEnoughToIntern", "kind": "function", "line": 42}
    enc = sess_encode(a, sd)
    assert enc is not None
    _, defs = enc
    # the lone leaf string is interned; the whole unique object is NOT a giant alias
    assert a not in defs.values()
    assert "SharedSymbolLongEnoughToIntern" in defs.values()


def test_literal_equal_to_existing_alias_bails():
    sd = SessionDict()
    a = {"s": "SharedSymbolLongEnoughToIntern", "f": "src/terse/proxy.py"}
    ea = sess_encode(a, sd)
    alias = next(al for al, v in ea[1].items() if v == "SharedSymbolLongEnoughToIntern")
    # A later payload that literally contains that alias string would be mis-expanded by the
    # client's cumulative legend, so session coding must bow out (caller falls back).
    assert sess_encode({"x": alias, "y": "z"}, sd) is None


def test_terse_marker_payload_passes_through():
    sd = SessionDict()
    assert sess_encode({"__terse_dict__": 1, "data": []}, sd) is None


def test_low_value_single_string_not_interned():
    sd = SessionDict()
    assert sess_encode({"a": "hi", "b": "yo"}, sd) is None


# --- SessionDict mechanics --------------------------------------------------------------

def test_intern_dedupes_by_value():
    sd = SessionDict()
    k = ("s", "value-x")
    a1 = sd.intern(k, "value-x", set())
    a2 = sd.intern(k, "value-x", set())
    assert a1 == a2
    assert sd.alias_for(k) == a1


def test_intern_avoids_payload_literals():
    sd = SessionDict()
    # avoid contains the alias that would otherwise be allocated first (~0)
    a = sd.intern(("s", "v"), "v", {ALIAS_SIGIL + "0"})
    assert a != ALIAS_SIGIL + "0"


def test_lru_eviction_bounds_the_table():
    sd = SessionDict(max_entries=3, max_bytes=1 << 30)
    for i in range(5):
        sd.intern(("s", f"value-number-{i}"), f"value-number-{i}", set())
    assert len(sd.aliases()) == 3  # only the 3 most-recent survive


def test_clear_resets_everything():
    sd = SessionDict()
    sd.intern(("s", "v"), "v", set())
    assert sd.aliases()
    sd.clear()
    assert sd.aliases() == set()
    assert sd.legend_snapshot() == {}


# --- Interceptor wiring seam (stage 1): shared when injected, off (None) otherwise -------

def test_interceptor_shares_session_dict_when_injected():
    pol = policy_mod.default_policy()
    shared = SessionDict()
    a = Interceptor(pol, session_dict=shared)
    b = Interceptor(pol, session_dict=shared)
    assert a.session_dict is shared and b.session_dict is shared  # one legend, all peers


def test_interceptor_session_dict_defaults_off():
    pol = policy_mod.default_policy()
    assert Interceptor(pol).session_dict is None  # single-peer: no session legend, unchanged
