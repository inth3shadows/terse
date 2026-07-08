"""#64 Phase 1 stages 3-4 — the session dictionary wired into the LIVE proxy path, plus
the guards (keyframe re-emission, transactional rollback, reconnect-clear, retrieve-backing),
the opt-in flag gating, and the measure replay.

Stage 1-2 (`test_session_dict.py`) covered the codec in isolation. These cover it in motion:
what the Interceptor actually emits, and the safety rails that let a cross-payload reference
scheme ship without silently stranding a definition the client never received.
"""

from __future__ import annotations

import json

from terse import policy as policy_mod
from terse import transforms
from terse.lossy import RETRIEVE_TOOL
from terse.measure import measure_session_dict
from terse.proxy import Interceptor
from terse.transforms import (
    SessionDict,
    sess_compress,
    sess_encode,
)

SYM = "SharedSymbolLongEnoughToIntern"  # a lone string >= SESS_MIN_TOK tokens


# --- helpers ----------------------------------------------------------------------------

def _session_pol():
    pol = policy_mod.default_policy()
    pol.session_dict = True
    return pol


def _req(mid, name="srv.tool"):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": name}})


def _result_msg(mid, text):
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": text}]}})


def _emit(inter, mid, payload, name="srv.tool"):
    """Drive one tool result through the Interceptor; return the emitted result text."""
    inter.note_request(_req(mid, name))
    out = inter.transform_response(_result_msg(mid, json.dumps(payload)))
    return json.loads(out)["result"]["content"][0]["text"]


def _client_reconstruct(envelope_texts):
    """Simulate the client accumulating each payload's `def` into one cumulative legend and
    resolving every payload against it — the exact contract a live model must satisfy."""
    legend: dict = {}
    out = []
    for text in envelope_texts:
        p = json.loads(text)
        legend.update(p.get("def", {}))
        data = transforms.sess_decode(p["data"], legend)
        out.append(transforms.decompress_structure(data))
    return out


# --- live proxy path: define once, reference (definition elided) across payloads ---------

def test_live_path_defines_then_references_across_payloads():
    inter = Interceptor(_session_pol(), session_dict=SessionDict())
    a = {"sym": SYM, "file": "src/terse/proxy.py"}
    b = {"caller": SYM, "file": "src/terse/proxy.py", "n": 3}
    ta = _emit(inter, 1, a)
    tb = _emit(inter, 2, b, name="other.tool")  # a DIFFERENT peer/tool reusing the value
    env_a, env_b = json.loads(ta), json.loads(tb)
    assert env_a[transforms.SESS_MARKER] == 1
    alias = next(al for al, v in env_a["def"].items() if v == SYM)
    # the second payload references the shared value with NO definition of its own
    assert alias not in env_b.get("def", {})
    assert alias in json.dumps(env_b["data"])
    # and the accumulating client reconstructs both originals exactly
    assert _client_reconstruct([ta, tb]) == [a, b]


def test_live_path_inert_when_flag_off():
    # Same payloads, flag off -> ordinary compressed form, never a session envelope.
    inter = Interceptor(policy_mod.default_policy())  # session_dict None -> off
    assert not inter.session
    ta = _emit(inter, 1, {"sym": SYM, "file": "src/terse/proxy.py"})
    assert transforms.SESS_MARKER not in ta
    assert transforms.decompress(ta) == {"sym": SYM, "file": "src/terse/proxy.py"}


# --- keyframe re-emission (the #8-analogue guard) ---------------------------------------

def test_keyframe_reemits_definition_after_bound():
    sd = SessionDict()
    kf = 2
    sess_encode({"s": SYM, "x": 0}, sd, keyframe=kf)              # define (refs=0)
    e1 = sess_encode({"s": SYM, "x": 1}, sd, keyframe=kf)         # ref 1 -> elided
    e2 = sess_encode({"s": SYM, "x": 2}, sd, keyframe=kf)         # ref 2 -> elided
    e3 = sess_encode({"s": SYM, "x": 3}, sd, keyframe=kf)         # ref 3 -> exceeds bound, re-emit
    assert SYM not in e1[1].values()
    assert SYM not in e2[1].values()
    assert SYM in e3[1].values()                                  # definition re-anchored


def test_keyframe_zero_never_reemits():
    sd = SessionDict()
    sess_encode({"s": SYM, "x": 0}, sd, keyframe=0)
    for i in range(1, 6):
        enc = sess_encode({"s": SYM, "x": i}, sd, keyframe=0)
        assert SYM not in enc[1].values()                        # elided forever


# --- transactional rollback: a bail must leave the shared table untouched ----------------

def test_swallowed_child_def_is_rolled_back():
    # A high-token string interned but then swallowed by an interned parent subtree had its
    # definition pruned; it must NOT remain in the table, or a later bare reference to it
    # would dangle against a definition the client never saw.
    sd = SessionDict()
    sub = {"sym": SYM, "k": "x"}
    payload = {"a": sub, "b": sub}                               # sub repeats -> interned whole
    enc = sess_encode(payload, sd)
    assert enc is not None
    # the subtree is interned; the swallowed lone string is NOT left stranded in the table
    assert SYM not in sd.legend_snapshot().values()
    assert any(v == sub for v in sd.legend_snapshot().values())


def test_nothing_aliased_leaves_table_clean():
    sd = SessionDict()
    assert sess_encode({"a": "hi", "b": "yo"}, sd) is None
    assert sd.aliases() == set()


def test_drop_is_exact_inverse_of_intern():
    sd = SessionDict()
    k = ("s", "value-xyz")
    sd.intern(k, "value-xyz", set())
    assert sd.aliases()
    sd.drop(k)
    assert sd.aliases() == set()
    assert sd.legend_snapshot() == {}
    # re-interning after a drop still yields a usable (fresh) alias
    assert sd.intern(k, "value-xyz", set()).startswith(transforms.ALIAS_SIGIL)


# --- reconnect-clear (the #20-analogue guard) -------------------------------------------

def test_reconnect_clears_session_dict():
    shared = SessionDict()
    inter = Interceptor(_session_pol(), session_dict=shared)
    _emit(inter, 1, {"sym": SYM, "file": "src/terse/proxy.py"})
    assert shared.aliases()                                       # something was interned
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize"}))
    assert shared.aliases() == set()                             # a re-handshake resets it


# --- retrieve-backing (#64 "Beyond"): every session def is fetchable via terse.retrieve ---

def test_session_defs_are_retrieve_backed():
    inter = Interceptor(_session_pol(), session_dict=SessionDict())
    text = _emit(inter, 1, {"sym": SYM, "file": "src/terse/proxy.py"})
    alias = next(al for al, v in json.loads(text)["def"].items() if v == SYM)
    reply = inter.answer_retrieve(json.dumps(
        {"jsonrpc": "2.0", "id": 42, "method": "tools/call",
         "params": {"name": RETRIEVE_TOOL, "arguments": {"handle": alias}}}))
    assert reply is not None
    result = json.loads(reply)["result"]
    assert not result.get("isError")
    assert SYM in result["content"][0]["text"]                   # the real value came back


def test_retrieve_tool_advertised_under_session_dict():
    # tools/list must gain terse.retrieve when session-dict is on, even with no drop rule.
    inter = Interceptor(_session_pol(), session_dict=SessionDict())
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}))
    out = inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 3, "result": {"tools": [{"name": "srv.tool"}]}}))
    names = [t["name"] for t in json.loads(out)["result"]["tools"]]
    assert RETRIEVE_TOOL in names


# --- flag gating ------------------------------------------------------------------------

def test_session_flag_requires_both_policy_and_injected_dict():
    off = policy_mod.default_policy()
    on = _session_pol()
    assert Interceptor(on, session_dict=SessionDict()).session is True
    assert Interceptor(on).session is False                      # policy on, no shared dict
    assert Interceptor(off, session_dict=SessionDict()).session is False  # dict, policy off


def test_cli_rejects_diff_and_session_dict_together():
    from terse.cli import main
    assert main(["proxy", "--diff", "--session-dict", "--", "echo"]) == 2


# --- sess_compress wrapper + self-contained decompress ----------------------------------

def test_sess_compress_first_payload_is_self_contained():
    sd = SessionDict()
    payload = {"rows": [{"s": "active"}, {"s": "active"}, {"s": "active"}], "sym": SYM}
    res = sess_compress(payload, sd)
    assert res is not None
    envelope, defs = res
    assert json.loads(envelope)[transforms.SESS_MARKER] == 1
    # the first payload carries every definition it uses -> decompress round-trips standalone
    assert transforms.decompress(envelope) == payload


def test_sess_compress_bails_to_none_without_mutating():
    sd = SessionDict()
    assert sess_compress({"a": "hi"}, sd) is None                # nothing worth aliasing
    assert sd.aliases() == set()


# --- measure replay ---------------------------------------------------------------------

def test_measure_session_dict_reports_cross_payload_saving():
    # A big value repeated across payloads should make the session run beat the per-call
    # baseline: its definition is sent once and elided (referenced) on every later payload,
    # a saving a fresh per-call legend structurally cannot capture. It must clear the small
    # per-payload envelope overhead, so the shared string is deliberately long.
    shared = "src/terse/proxy/interceptor/session_dictionary_shared_symbol_" * 3
    envelopes = [{"raw": json.dumps({"hit": shared, "i": i})} for i in range(8)]
    s = measure_session_dict(envelopes, keyframe=0)
    assert s["payloads"] == 8
    assert s["applicable"] == 8
    assert s["saved_cl100k"] > 0                                 # cross-payload elision wins
    assert s["saved_pct"] > 0
    assert s["elided_wins"] >= 7                                 # all but the first define


def test_measure_session_dict_neutral_on_unique_payloads():
    # Nothing shared across payloads -> no cross-payload win (may be ~0, never a crash).
    envelopes = [{"raw": json.dumps({"uniq": f"value-{i}-{i*7}", "i": i})} for i in range(4)]
    s = measure_session_dict(envelopes, keyframe=0)
    assert s["payloads"] == 4
    assert s["baseline_cl100k"] > 0
