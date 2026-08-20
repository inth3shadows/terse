"""Proxy: pure Interceptor logic + an end-to-end run against a fake MCP server."""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import sys

from terse import text_diff, transforms
from terse.policy import Policy, Rule
from terse.proxy import PRIMER_HEAD, Interceptor, run_proxy

FULL = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])
FAKE = pathlib.Path(__file__).parent / "fake_mcp_server.py"


def _records_text():
    return json.dumps({"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                                  for i in range(20)]}, indent=2)


def _result_msg(mid, text):
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": text}]}})


# --- pure Interceptor logic ---

def test_tracks_request_and_compresses_matching_result():
    inter = Interceptor(FULL, lazy_primer=False)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    out = inter.transform_response(_result_msg(7, _records_text()))
    msg = json.loads(out)
    text = msg["result"]["content"][0]["text"]
    assert text != _records_text()                       # actually transformed
    assert transforms.decompress(text) == json.loads(_records_text())  # losslessly
    assert inter.pending == {}                            # id consumed


def test_error_reply_pops_pending_entry_too():
    # Regression: transform_response's early-return guard checked "result" not in msg
    # BEFORE popping pending, so an error-shaped reply (no "result" key — e.g. a
    # genuine downstream JSON-RPC error, or HttpTransport's own synthesized fail-open
    # error) left its pending entry lingering until PENDING_MAX eviction instead of
    # being cleaned up immediately.
    inter = Interceptor(FULL)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    error_reply = json.dumps({"jsonrpc": "2.0", "id": 7,
                              "error": {"code": -32000, "message": "boom"}})
    out = inter.transform_response(error_reply)
    assert out == error_reply         # forwarded unchanged — not a tracked result
    assert inter.pending == {}        # but the pending entry was still popped


def test_pending_map_is_bounded_under_unanswered_calls():
    # tools/call ids that never get a result (timed-out / abandoned) must not leak the
    # pending map without bound (#22). Evicts oldest-first; recent ids survive.
    inter = Interceptor(FULL)
    for i in range(Interceptor.PENDING_MAX + 50):
        inter.note_request(_req(i, "gh.api.items"))
    assert len(inter.pending) <= Interceptor.PENDING_MAX
    assert (Interceptor.PENDING_MAX + 49) in inter.pending   # newest kept
    assert 0 not in inter.pending                            # oldest evicted
    # an evicted id's late result just forwards uncompressed (fail-open), not a crash
    assert inter.transform_response(_result_msg(0, _records_text())) == \
        _result_msg(0, _records_text())


def test_concurrent_note_and_transform_do_not_crash_under_eviction():
    # The two pump threads call note_request and transform_response concurrently on the
    # same Interceptor. The #22 eviction iterates `pending` while the other thread pops
    # it; without the lock, `next(iter(...))` raises "dictionary changed size during
    # iteration" and kills the request pump. The lock must make this safe.
    import threading

    inter = Interceptor(FULL)
    inter.PENDING_MAX = 16                                # force constant eviction churn
    errors: list[Exception] = []
    N = 4000

    def noter():
        try:
            for i in range(N):
                inter.note_request(_req(i, "gh.api.items"))
        except Exception as e:  # noqa: BLE001 — capture, don't swallow into the thread
            errors.append(e)

    def transformer():
        try:
            for i in range(N):
                inter.transform_response(_result_msg(i, _records_text()))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=noter), threading.Thread(target=transformer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []                                  # no RuntimeError from the race
    assert len(inter.pending) <= inter.PENDING_MAX       # still bounded


def test_untracked_result_passes_through_unchanged():
    inter = Interceptor(FULL)
    line = _result_msg(99, _records_text())              # no matching request noted
    assert inter.transform_response(line) == line


def test_initialize_and_errors_pass_through():
    inter = Interceptor(FULL)
    init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "x"}}})
    assert inter.transform_response(init) == init
    err = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -1, "message": "no"}})
    assert inter.transform_response(err) == err


def test_notification_and_non_json_pass_through():
    inter = Interceptor(FULL)
    notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"})
    assert inter.transform_response(notif) == notif
    assert inter.transform_response("not json") == "not json"


def test_note_request_survives_non_dict_params_instead_of_killing_the_pump():
    # `msg.get("params") or {}` only neutralised FALSY junk, so a truthy non-object
    # `params` raised AttributeError out of note_request. That exception surfaces in the
    # client->server pump THREAD (proxy.pump -> fwd -> note_request), which kills
    # forwarding for the rest of the session — a malformed request taking the whole proxy
    # down, from a method that only does bookkeeping. Found via the demo server's tests.
    inter = Interceptor(FULL)
    for junk in ("not-an-object", [1, 2], 7):
        inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                       "params": junk}))
        inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                       "params": junk}))
    assert inter.pending == {}      # nothing recordable was recorded
    assert inter.client_name is None
    # and a well-formed call after the junk is still tracked — the state machine is intact
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "gh.x"}}))
    assert inter.pending[3][0] == "gh.x"


def test_malformed_requests_cannot_kill_the_pump_on_the_drop_policy_path():
    # `answer_retrieve` runs BEFORE note_request whenever the policy has a drop rule, so
    # it — not note_request — is the branch that fires on a deployed install. The default
    # policy has no drop, which is exactly why the earlier test passed while this path was
    # still live. `bool` is intentionally NOT excluded: it is a hashable int subclass.
    from terse.lossy import RETRIEVE_TOOL
    pol = Policy(rules=[Rule("gh.*", ("minify",),
                             fields={"$.big": {"lossy": "drop-to-retrieve"}})])
    assert pol.has_drop()
    inter = Interceptor(pol)
    for junk in ("oops", [1, 2], 7):
        assert inter.answer_retrieve(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": junk}) ) is None
    # a real retrieve call with non-object `arguments` must answer, not raise
    reply = inter.answer_retrieve(json.dumps(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": RETRIEVE_TOOL, "arguments": "oops"}}))
    assert reply is not None and json.loads(reply)["id"] == 2


def test_note_request_declines_a_non_hashable_id_instead_of_raising():
    # `mid` becomes a dict key, so `"id": {"a": 1}` raised TypeError out of the same pump
    # thread — the identical session-wide forwarding death by a different door.
    inter = Interceptor(FULL)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": {"a": 1}, "method": "tools/call",
                                   "params": {"name": "gh.x"}}))
    assert inter.pending == {}


def test_non_json_text_content_is_left_alone():
    inter = Interceptor(FULL)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                   "params": {"name": "gh.x"}}))
    line = _result_msg(5, "just a sentence, not json")
    assert inter.transform_response(line) == line        # nothing to compress, unchanged


def test_skip_policy_leaves_result_unchanged():
    inter = Interceptor(Policy(rules=[Rule("gh.*", ())]))  # passthrough tier
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "gh.x"}}))
    line = _result_msg(3, _records_text())
    assert inter.transform_response(line) == line


# --- cross-call diffing (opt-in) ---

DIFF = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))], diff=True)


def _req(mid, name):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": name}})


def _records(n, change=None):
    rows = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(n)]
    if change is not None:
        rows[change]["status"] = "closed"
    return {"result": rows}


def _emit(inter, mid, tool, payload):
    inter.note_request(_req(mid, tool))
    out = inter.transform_response(_result_msg(mid, json.dumps(payload)))
    return json.loads(out)["result"]["content"][0]["text"]


def test_first_call_has_no_prior_so_sends_full_compressed():
    inter = Interceptor(DIFF, lazy_primer=False)
    text = _emit(inter, 1, "gh.api.items", _records(40))
    assert transforms.DIFF_MARKER not in text
    assert transforms.decompress(text) == _records(40)


def test_second_same_tool_result_emits_smaller_lossless_diff():
    inter = Interceptor(DIFF)
    prev, curr = _records(40), _records(40, change=5)
    full = _emit(inter, 1, "gh.api.items", prev)
    diff_text = _emit(inter, 2, "gh.api.items", curr)
    env = json.loads(diff_text)
    assert env.get(transforms.DIFF_MARKER) == 1          # a diff was emitted
    assert transforms.diff_decode(prev, env) == curr     # and reconstructs curr exactly
    assert _cost_lt(diff_text, full)                     # and it is smaller


# --- Phase 0: the in-context invariant (a diff base is per-session, never persisted) ---

def _disjoint(n, base):
    """n records whose ids start at `base` — two calls with different bases share no
    record, so a diff between them is never smaller than the full (the base 'lost')."""
    return {"result": [{"id": base + i, "status": "active",
                        "url": "https://x.example/api/items"} for i in range(n)]}


def test_diff_base_is_not_shared_across_interceptors():
    # A base lives only in the Interceptor that produced it — never persisted to disk, never
    # shared across sessions — so a fresh session's FIRST sight of a tool re-anchors as a
    # full. This pins the invariant that makes the diff safe: it names "the prior result
    # already in the model's context", which a cross-session base would not be.
    prev, curr = _records(40), _records(40, change=5)
    a = Interceptor(DIFF, lazy_primer=False)
    _emit(a, 1, "gh.api.items", prev)
    assert transforms.DIFF_MARKER in _emit(a, 2, "gh.api.items", curr)   # A diffs
    b = Interceptor(DIFF, lazy_primer=False)                             # new session
    assert transforms.DIFF_MARKER not in _emit(b, 1, "gh.api.items", curr)  # no shared base


def test_reconnect_clears_diff_base_and_args_so_next_result_re_anchors():
    # An `initialize` means the client rebuilt its context window, so no prior result a diff
    # could reference survives — every base (and its args attribution) must drop.
    inter = Interceptor(DIFF, lazy_primer=False)
    prev, curr = _records(40), _records(40, change=5)
    _emit(inter, 1, "gh.api.items", prev)
    assert transforms.DIFF_MARKER in _emit(inter, 2, "gh.api.items", curr)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "initialize"}))
    assert inter.last == {} and inter.last_args == {}
    assert transforms.DIFF_MARKER not in _emit(inter, 3, "gh.api.items", _records(40, change=7))


# --- Phase 1: the diff_reason ledger datum (why a diff did/didn't fire) ---

def _req_args(mid, name, args=None):
    params = {"name": name}
    if args is not None:
        params["arguments"] = args
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": params})


def _emit_args(inter, mid, tool, payload, args=None):
    inter.note_request(_req_args(mid, tool, args))
    inter.transform_response(_result_msg(mid, json.dumps(payload)))


def _capture_stats():
    reasons: list = []

    def stats(tool, raw, emitted, passthrough, diff_reason=None, structured=None,
              structured_out=None):
        reasons.append(diff_reason)

    return reasons, stats


def test_diff_reason_no_prior_then_emitted():
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    _emit_args(inter, 1, "gh.api.items", _records(40), {"q": "a"})
    assert reasons[-1] == "no_prior"                       # tool unseen this session
    _emit_args(inter, 2, "gh.api.items", _records(40, change=5), {"q": "a"})
    assert reasons[-1] == "emitted"                        # small change diffs smaller


def test_diff_reason_splits_same_vs_different_args_when_delta_loses():
    # Disjoint record sets never diff smaller than the full, so the base "loses". The datum
    # that decides whether arg-keying is worth building: was that losing base a DIFFERENT-
    # args call (arg-keying could offer a same-args base instead) or the SAME args (an
    # encoding miss keying would not fix)?
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    _emit_args(inter, 1, "gh.api.items", _disjoint(40, 0), {"page": 1})
    _emit_args(inter, 2, "gh.api.items", _disjoint(40, 1000), {"page": 2})
    assert reasons[-1] == "not_smaller_diff_args"          # base was the page=1 call
    _emit_args(inter, 3, "gh.api.items", _disjoint(40, 2000), {"page": 2})
    assert reasons[-1] == "not_smaller_same_args"          # base now the page=2 call


def test_diff_off_by_default_and_policy_true_enables():
    # Policy.diff defaults OFF since #170: the tier is correct (its #72/#75 validation
    # still holds) but its 190-token primer paragraph outweighed what the tier saved at a
    # 0.38% production hit rate: 5,052 tokens banked over the measurement window, erased by
    # ~27 primer attaches. A plain policy therefore does NOT diff …
    inter = Interceptor(DIFF)
    prev, curr = _records(40), _records(40, change=5)
    _emit(inter, 1, "gh.api.items", prev)
    t2 = _emit(inter, 2, "gh.api.items", curr)
    env = json.loads(t2)
    assert env.get(transforms.DIFF_MARKER) == 1
    assert transforms.diff_decode(prev, env) == curr
    # … and an explicit "diff": false opt-out still sends fulls both times.
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))], diff=False)
    inter = Interceptor(off)
    t1 = _emit(inter, 1, "gh.api.items", prev)
    t2 = _emit(inter, 2, "gh.api.items", curr)
    assert transforms.DIFF_MARKER not in t1 and transforms.DIFF_MARKER not in t2
    assert transforms.decompress(t2) == curr


def test_diff_not_emitted_when_it_would_not_be_smaller():
    # an unrelated second payload makes any diff at least as large as the full form,
    # so the proxy keeps the full compressed result (fallback), still lossless.
    inter = Interceptor(DIFF)
    _emit(inter, 1, "gh.api.items", _records(40))
    other = {"result": [{"k": i, "v": "x" * 50} for i in range(40)]}
    text = _emit(inter, 2, "gh.api.items", other)
    assert transforms.DIFF_MARKER not in text
    assert transforms.decompress(text) == other


def test_keyframe_forces_full_after_k_consecutive_diffs():
    # With interval K, the (K+1)th same-tool result is a full keyframe, not a diff, so a
    # chained diff never drifts more than K turns from a self-contained anchor (#8).
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))],
                 diff=True, diff_keyframe_interval=3)
    inter = Interceptor(pol, lazy_primer=False)
    texts = [_emit(inter, 1, "gh.api.items", _records(40))]            # full (no prior)
    for i in range(2, 8):                                              # small change each call
        texts.append(_emit(inter, i, "gh.api.items", _records(40, change=i % 40)))
    is_diff = [transforms.DIFF_MARKER in t for t in texts]
    assert is_diff == [False, True, True, True, False, True, True]    # F D D D | F(keyframe) D D
    # the keyframe (index 4, i.e. call i=5) reconstructs WITHOUT any prior — self-contained
    assert transforms.decompress(texts[4]) == _records(40, change=5)


def test_keyframe_interval_zero_never_forces_full():
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))],
                 diff=True, diff_keyframe_interval=0)
    inter = Interceptor(pol)
    texts = [_emit(inter, 1, "gh.api.items", _records(40))]
    for i in range(2, 8):
        texts.append(_emit(inter, i, "gh.api.items", _records(40, change=i % 40)))
    assert all(transforms.DIFF_MARKER in t for t in texts[1:])        # every follow-up is a diff


def test_non_json_result_evicts_diff_base_so_next_re_anchors():
    # JSON A -> non-JSON error -> JSON C for the same tool. The non-JSON result is the
    # model's visible "previous result", so C must NOT diff against the now-invisible A;
    # the base is evicted and C re-anchors as a full, else reconstruction applies the
    # delta to the wrong base (#8).
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))], diff=True)
    inter = Interceptor(pol)
    _emit(inter, 1, "gh.api.items", _records(40))                    # full (no prior); sets base
    inter.note_request(_req(2, "gh.api.items"))                      # same tool, non-JSON result
    err = inter.transform_response(_result_msg(2, "upstream error: rate limited"))
    assert json.loads(err)["result"]["content"][0]["text"] == "upstream error: rate limited"
    # base evicted -> the next JSON result is a full keyframe, not a diff against A
    c = _emit(inter, 3, "gh.api.items", _records(40, change=5))
    assert transforms.DIFF_MARKER not in c
    assert transforms.decompress(c) == _records(40, change=5)


def test_reinitialize_resets_diff_bases_to_prevent_desync():
    # A client re-handshake (new `initialize`) means the model's context — and the prior
    # result a diff would reference — is gone. Every diff base must drop so the next
    # result re-anchors as a full, never a delta against a lost base (#20).
    inter = Interceptor(DIFF, lazy_primer=False)
    _emit(inter, 1, "gh.api.items", _records(40))            # sets the diff base
    assert "gh.api.items" in inter.last
    inter.note_request(_req(9, "gh.api.slow"))              # an in-flight, unanswered call
    inter.note_request(_init_req(2))                         # client reconnects
    assert inter.last == {} and inter.since_keyframe == {}   # bases dropped
    assert inter.pending == {}                               # stale ids dropped too (#20/#22)
    text = _emit(inter, 3, "gh.api.items", _records(40, change=5))
    assert transforms.DIFF_MARKER not in text               # full keyframe, not a diff
    assert transforms.decompress(text) == _records(40, change=5)


def _cost_lt(a, b):
    from terse.proxy import _cost
    return _cost(a) < _cost(b)


# --- cross-call text diffing for non-JSON results (Tier 0.7 text, #25) ---

def _log_text(n, changed_line=None):
    lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}" for i in range(n)]
    if changed_line is not None:
        lines[changed_line] = "[ERROR] worker crashed: connection reset"
    return "\n".join(lines)


def _emit_text(inter, mid, tool, text):
    inter.note_request(_req(mid, tool))
    out = inter.transform_response(_result_msg(mid, text))
    return json.loads(out)["result"]["content"][0]["text"]


def test_first_non_json_result_has_no_prior_so_passes_through_raw():
    inter = Interceptor(DIFF)
    text = _log_text(80)
    assert _emit_text(inter, 1, "fs.read", text) == text


def test_second_non_json_result_emits_smaller_lossless_text_diff():
    inter = Interceptor(DIFF, lazy_primer=False)
    prev, curr = _log_text(200), _log_text(200, changed_line=100)
    raw_first = _emit_text(inter, 1, "fs.read", prev)
    diff_text = _emit_text(inter, 2, "fs.read", curr)
    env = json.loads(diff_text)
    assert env.get(text_diff.DIFF_MARKER) == 1
    assert text_diff.text_diff_decode(prev, env) == curr
    assert _cost_lt(diff_text, curr)
    assert raw_first == prev  # sanity: first call was untouched


def test_text_diff_off_by_default_and_policy_true_enables():
    # Same default for the CDC text path: off by default (#170), on via "diff": true.
    inter = Interceptor(DIFF, lazy_primer=False)
    prev, curr = _log_text(80), _log_text(80, changed_line=40)
    _emit_text(inter, 1, "fs.read", prev)
    t2 = _emit_text(inter, 2, "fs.read", curr)
    env = json.loads(t2)
    assert env.get(text_diff.DIFF_MARKER) == 1
    assert text_diff.text_diff_decode(prev, env) == curr
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))], diff=False)
    inter = Interceptor(off, lazy_primer=False)
    t1 = _emit_text(inter, 1, "fs.read", prev)
    t2 = _emit_text(inter, 2, "fs.read", curr)
    assert t1 == prev and t2 == curr


def test_text_diff_not_emitted_when_it_would_not_be_smaller():
    inter = Interceptor(DIFF)
    _emit_text(inter, 1, "fs.read", _log_text(20))
    other = "totally unrelated content " * 5
    text = _emit_text(inter, 2, "fs.read", other)
    assert text_diff.DIFF_MARKER not in text
    assert text == other


def test_passthrough_policy_never_text_diffs_even_with_diff_on():
    # empty tiers = a policy that says "hands off this tool entirely" (mirrors the JSON
    # diff path, which also never engages for a passthrough-tiered tool).
    pol = Policy(rules=[Rule("fs.*", ())], diff=True)
    inter = Interceptor(pol)
    prev, curr = _log_text(50), _log_text(50, changed_line=10)
    _emit_text(inter, 1, "fs.read", prev)
    text = _emit_text(inter, 2, "fs.read", curr)
    assert text == curr
    assert inter.last_text == {}


def test_text_diff_keyframe_forces_raw_after_k_consecutive_diffs():
    pol = Policy(rules=[Rule("fs.*", ("minify", "tabularize", "dictionary"))],
                 diff=True, diff_keyframe_interval=2)
    inter = Interceptor(pol)
    texts = [_emit_text(inter, 1, "fs.read", _log_text(100))]           # raw (no prior)
    for i in range(2, 7):
        texts.append(_emit_text(inter, i, "fs.read", _log_text(100, changed_line=i)))
    is_diff = [text_diff.DIFF_MARKER in t for t in texts]
    assert is_diff == [False, True, True, False, True, True]           # F D D | F(keyframe) D D


def test_json_and_text_diff_bases_are_independent_for_the_same_tool():
    # A tool that sometimes returns JSON and sometimes plain text must not let one
    # shape's diff base leak into the other's codec.
    inter = Interceptor(DIFF)
    _emit(inter, 1, "mixed.tool", _records(20))               # JSON base set
    _emit_text(inter, 2, "mixed.tool", _log_text(50))         # non-JSON: evicts JSON base
    assert inter.last.get("mixed.tool") is None
    diff_text = _emit_text(inter, 3, "mixed.tool", _log_text(50, changed_line=5))
    assert text_diff.DIFF_MARKER in diff_text
    assert text_diff.text_diff_decode(_log_text(50), json.loads(diff_text)) == _log_text(50, changed_line=5)


def test_text_diff_reinitialize_resets_bases_to_prevent_desync():
    inter = Interceptor(DIFF)
    _emit_text(inter, 1, "fs.read", _log_text(50))
    assert "fs.read" in inter.last_text
    inter.note_request(_init_req(2))
    assert inter.last_text == {} and inter.since_text_keyframe == {}


# --- one-time format primer via initialize.instructions (#13) ---

def _init_req(mid=1):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "initialize", "params": {}})


def _init_resp(mid=1, instructions=None):
    result = {"protocolVersion": "1", "capabilities": {}, "serverInfo": {"name": "s"}}
    if instructions is not None:
        result["instructions"] = instructions
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": result})


def test_initialize_reply_gets_format_primer():
    # lazy_primer=False: exercises the still-preserved eager `_augment_initialize` path
    # (what every multiproxy peer runs) directly. The new default (lazy_primer=True) is
    # covered by the lazy-primer test block near the bottom of this file (#168 phase 2).
    inter = Interceptor(DIFF, lazy_primer=False)   # diff opt-in, so the primer covers every form
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1)))
    instr = out["result"]["instructions"]
    assert "__terse_table__" in instr and "__terse_diff__" in instr   # covers all forms
    assert "__terse_textdiff__" in instr
    assert out["result"]["serverInfo"]["name"] == "s"                 # rest untouched


def test_initialize_preserves_existing_instructions():
    # lazy_primer=False: same reasoning as test_initialize_reply_gets_format_primer above.
    inter = Interceptor(FULL, lazy_primer=False)
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1, "USE TOOL X FIRST.")))
    instr = out["result"]["instructions"]
    assert "USE TOOL X FIRST." in instr and "__terse_table__" in instr


def test_untracked_initialize_passes_through_unchanged():
    # never saw the request -> don't touch the reply
    inter = Interceptor(FULL)
    resp = _init_resp(1)
    assert inter.transform_response(resp) == resp


def test_clear_init_id_prevents_stale_reply_misidentification():
    # Regression (multiproxy broadcast case): note_request sets init_id, but if the
    # reply carrying that id never reaches transform_response (multiproxy swallows a
    # broadcast peer's reply and merges it separately), the one-time reset never fires
    # and init_id stays stale — a LATER unrelated reply reusing that same id would then
    # be misidentified as the initialize reply and corrupted via _augment_initialize.
    # clear_init_id() lets a caller reset it proactively when it knows the reply won't
    # flow through transform_response.
    inter = Interceptor(FULL)
    inter.note_request(_init_req("terse-b0-1"))
    assert inter.init_id == "terse-b0-1"
    inter.clear_init_id()
    assert inter.init_id is None

    # a later, unrelated tools/call reply that happens to reuse that exact id string
    # must be treated as a normal (untracked) message, not an initialize reply.
    later = json.dumps({"jsonrpc": "2.0", "id": "terse-b0-1",
                        "result": {"content": [{"type": "text", "text": "normal"}]}})
    out = json.loads(inter.transform_response(later))
    assert "instructions" not in out["result"]  # NOT run through _augment_initialize


def test_primer_injected_once_not_per_message():
    inter = Interceptor(FULL)
    inter.note_request(_init_req(1))
    inter.transform_response(_init_resp(1))
    # a second initialize-shaped reply with the same id is no longer tracked -> untouched
    resp2 = _init_resp(1)
    assert inter.transform_response(resp2) == resp2


# --- #168 phase 2: lazy primer (default lazy_primer=True) ---

def test_lazy_primer_attaches_to_first_compressible_result_not_initialize():
    inter = Interceptor(FULL)      # lazy_primer=True is the default
    inter.note_request(_init_req(1))
    init_out = json.loads(inter.transform_response(_init_resp(1)))
    assert "instructions" not in init_out["result"]      # no eager priming

    _note_call(inter, 2, "gh.api.items")
    out = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out["result"]["content"]
    assert len(blocks) == 2
    assert PRIMER_HEAD in blocks[0]["text"]                # leading primer block
    assert transforms.decompress(blocks[1]["text"]) == json.loads(_records_text())
    assert inter._primer_sent is True


def test_lazy_primer_never_sent_if_no_wrapped_tool_called():
    inter = Interceptor(FULL)
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1)))
    assert "instructions" not in out["result"]
    assert inter._primer_sent is False     # still owed — nothing was ever called to pay it


def test_lazy_primer_skips_a_changed_result_with_no_wire_form_marker():
    # minify-only tier: bytes change (whitespace stripped) but no terse marker is ever
    # emitted, so `changed=True` alone must not be misread as "a wire form appeared."
    # Isolates the marker-substring check from the structuredContent guard (see the next
    # test for that one).
    pol = Policy(rules=[Rule("plain.*", ("minify",)),
                        Rule("gh.*", ("minify", "tabularize", "dictionary"))])
    inter = Interceptor(pol)
    _note_call(inter, 1, "plain.get")
    spaced = json.dumps({"a": 1}, indent=2)
    out1 = json.loads(inter.transform_response(_result_msg(1, spaced)))
    assert out1["result"]["content"][0]["text"] != spaced   # actually changed (minified)
    assert len(out1["result"]["content"]) == 1               # no primer block inserted
    assert inter._primer_sent is False

    # a later call that DOES emit a marker is the one that gets it
    _note_call(inter, 2, "gh.api.items")
    out2 = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out2["result"]["content"]
    assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]
    assert inter._primer_sent is True


def test_lazy_primer_skips_a_pure_mirror_drop():
    # structured: replace drops the text mirror entirely — changed=True (the block is
    # gone) but nothing terse-encoded survives in `content` to explain.
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                             structured="replace")])
    inter = Interceptor(pol)
    _note_call(inter, 1, "gh.items")
    payload = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    result = {"content": [{"type": "text", "text": json.dumps(payload)}],
             "structuredContent": payload}
    out = json.loads(inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": result})))
    assert out["result"]["content"] == []      # mirror dropped, nothing left to attach to
    assert inter._primer_sent is False

    # a later, text-only compressible call (no structuredContent) still gets it
    _note_call(inter, 2, "gh.api.items")
    out2 = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out2["result"]["content"]
    assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]
    assert inter._primer_sent is True


def test_lazy_primer_skips_when_structuredcontent_carries_the_marker_too():
    # structured: compress keeps the text block (unlike "replace" above) but rewrites
    # structuredContent with the same terse markers. Claude Code discards the text block
    # entirely whenever structuredContent is present (measured,
    # scripts/probe/structured_content/) — a leading primer block here would be thrown
    # away right alongside it, regardless of which field the marker landed in.
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                             structured="compress")])
    inter = Interceptor(pol)
    _note_call(inter, 1, "gh.items")
    payload = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    result = {"content": [{"type": "text", "text": json.dumps(payload)}],
             "structuredContent": payload}
    out = json.loads(inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": result})))
    assert '"__terse_' in json.dumps(out["result"]["structuredContent"])  # it DID rewrite
    assert len(out["result"]["content"]) == 1       # text block untouched, not dropped
    assert inter._primer_sent is False              # but no primer attached to it

    # a later, text-only compressible call (no structuredContent) still gets it
    _note_call(inter, 2, "gh.api.items")
    out2 = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out2["result"]["content"]
    assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]
    assert inter._primer_sent is True


def test_lazy_primer_skips_when_structuredcontent_present_but_untouched():
    # structured: leave (the default for an unknown/no client) never rewrites
    # structuredContent — it stays the tool's own native shape, no terse marker in it
    # at all. The guard is on PRESENCE, not on whether terse rewrote it: this is the
    # most common real-world trigger for the accepted residual gap (most tools don't
    # opt a client into structured rewriting), so it needs its own coverage rather than
    # riding on the "compress" test above.
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                             structured="leave")])
    inter = Interceptor(pol)
    _note_call(inter, 1, "gh.items")
    payload = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    result = {"content": [{"type": "text", "text": json.dumps(payload)}],
             "structuredContent": payload}
    out = json.loads(inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": result})))
    assert out["result"]["structuredContent"] == payload   # left alone, no marker
    assert '"__terse_' in out["result"]["content"][0]["text"]  # text block DID compress
    assert len(out["result"]["content"]) == 1               # but no primer block attached
    assert inter._primer_sent is False

    # a later, text-only compressible call (no structuredContent) still gets it
    _note_call(inter, 2, "gh.api.items")
    out2 = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out2["result"]["content"]
    assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]
    assert inter._primer_sent is True


def test_lazy_primer_sent_exactly_once_across_many_calls():
    inter = Interceptor(FULL)
    for i in range(1, 6):
        _note_call(inter, i, "gh.api.items")
        out = json.loads(inter.transform_response(_result_msg(i, _records_text())))
        blocks = out["result"]["content"]
        if i == 1:
            assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]
        else:
            assert len(blocks) == 1
            assert PRIMER_HEAD not in blocks[0]["text"]
    assert inter._primer_sent is True


def test_lazy_primer_resets_on_reconnect():
    inter = Interceptor(FULL)
    _note_call(inter, 1, "gh.api.items")
    out1 = json.loads(inter.transform_response(_result_msg(1, _records_text())))
    assert PRIMER_HEAD in out1["result"]["content"][0]["text"]
    assert inter._primer_sent is True

    inter.note_request(_init_req(2))       # reconnect: new context, new session
    assert inter._primer_sent is False     # owed again

    _note_call(inter, 3, "gh.api.items")
    out2 = json.loads(inter.transform_response(_result_msg(3, _records_text())))
    blocks = out2["result"]["content"]
    assert len(blocks) == 2 and PRIMER_HEAD in blocks[0]["text"]


# --- raw-payload capture tee (#32) ---

def test_capture_tees_raw_text_before_compression():
    captured: list[tuple[str, str]] = []
    inter = Interceptor(FULL, capture=lambda tool, raw, **kw: captured.append((tool, raw)))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    raw = _records_text()
    out = inter.transform_response(_result_msg(7, raw))
    # captured payload is the RAW pre-compression text, tagged by tool...
    assert captured == [("gh.api.items", raw)]
    # ...while the client still received the compressed (transformed) form
    assert json.loads(out)["result"]["content"][0]["text"] != raw


def test_note_request_tool_name_qualifies_capture_but_not_policy_selection():
    # Regression (multiproxy): capture/audit must see a peer-qualified tool name (so
    # two peers' same-named tools don't collide into one capture-corpus bucket), but
    # compression/policy-tier lookup must still use the BARE name the policy's own
    # rules match against — conflating the two broke policy selection for a peer with
    # a custom policy_path.
    captured: list[tuple[str, str]] = []
    audited = []
    inter = Interceptor(FULL, capture=lambda tool, raw, **kw: captured.append((tool, raw)),
                        audit=audited.append)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}),
                       tool_name="gh__gh.api.items")
    raw = _records_text()
    out = inter.transform_response(_result_msg(7, raw))

    # capture sees the peer-qualified name...
    assert captured == [("gh__gh.api.items", raw)]
    # ...and so does the audit record's display field...
    assert audited[0]["tool"] == "gh__gh.api.items"
    # ...but the policy still matched (and compressed) against the BARE name, exactly
    # as it would have without the peer prefix.
    assert audited[0]["tiers"] == ["minify", "tabularize", "dictionary"]
    assert json.loads(out)["result"]["content"][0]["text"] != raw


def test_capture_failure_never_affects_forwarding():
    def boom(tool: str, raw: str) -> None:
        raise OSError("read-only corpus")
    inter = Interceptor(FULL, capture=boom, lazy_primer=False)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    out = inter.transform_response(_result_msg(1, _records_text()))
    # despite the capture raising, the result is still compressed losslessly and delivered
    text = json.loads(out)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == json.loads(_records_text())


def test_run_proxy_capture_dir_writes_loadable_corpus(tmp_path):
    from terse.capture import load_corpus

    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "gh.api.items"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    corpus = tmp_path / "corpus"
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   capture_dir=str(corpus))
    assert rc == 0
    envs = load_corpus(corpus)
    # exactly the one tools/call result was teed (the initialize reply is not a tool call)
    assert len(envs) == 1 and envs[0]["tool"] == "gh.api.items"
    # and it captured the RAW payload, consumable by verify/measure
    assert json.loads(envs[0]["raw"])["result"][0]["status"] == "active"


def test_run_proxy_capture_dir_failure_does_not_break_traffic(tmp_path):
    # point --capture-dir at an existing FILE: capture_payload's mkdir fails on every
    # call, but the proxy must still forward and compress (capture is never load-bearing).
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    requests = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "gh.api.items"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    # lazy_primer=False: this test is about capture-sink resilience, not primer delivery.
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   capture_dir=str(blocker), lazy_primer=False)
    assert rc == 0
    line = [ln for ln in cout.getvalue().splitlines() if ln.strip()][0]
    text = json.loads(line)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == {"result": [
        {"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]}


def _two_tool_calls() -> str:
    return "\n".join(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                                 "params": {"name": "gh.api.items"}})
                     for i in (1, 2)) + "\n"


def test_run_proxy_broken_capture_warns_once_without_debug(tmp_path, capsys):
    # #131: the sink callbacks used to swallow their own failures behind --debug, so
    # Interceptor._warn_sink's unconditional first-failure line could never fire and a
    # --capture-dir that captures NOTHING looked like a perfectly normal run.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    cin, cout = io.StringIO(_two_tool_calls()), io.StringIO()
    # lazy_primer=False: this test is about capture-sink resilience, not primer delivery.
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   capture_dir=str(blocker), lazy_primer=False)
    assert rc == 0
    warnings = [ln for ln in capsys.readouterr().err.splitlines()
                if "capture skipped" in ln]
    # exactly ONE line despite two failing calls, and it names the sink + the tool
    assert len(warnings) == 1
    assert warnings[0].startswith("[terse-proxy] gh.api.items: capture skipped: ")
    assert "silenced unless --debug" in warnings[0]
    # and the client still got both results (a dead sink stays fail-open); the first is
    # the full compressed payload, the second a diff against it, exactly as with a
    # healthy capture dir
    lines = [ln for ln in cout.getvalue().splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == [1, 2]
    assert transforms.decompress(
        json.loads(lines[0])["result"]["content"][0]["text"]) == {
            "result": [{"id": i, "status": "active",
                        "url": "https://x.example/api/items"} for i in range(20)]}


def test_run_proxy_broken_audit_log_warns_without_debug(tmp_path, capsys):
    # --debug-log at a DIRECTORY: append_audit's open() fails on every call (#131).
    cin, cout = io.StringIO(_two_tool_calls()), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   debug_log=str(tmp_path))
    assert rc == 0
    warnings = [ln for ln in capsys.readouterr().err.splitlines() if "audit skipped" in ln]
    assert len(warnings) == 1 and warnings[0].startswith("[terse-proxy] gh.api.items: ")
    assert len([ln for ln in cout.getvalue().splitlines() if ln.strip()]) == 2


def test_run_proxy_broken_stats_log_warns_without_debug(tmp_path, capsys):
    # --stats-log at a DIRECTORY. Stats is the on-by-default sink, so a silently dead
    # ledger is the one most likely to go unnoticed — and it is what makes a later
    # `terse measure --corpus` report a percentage over whatever subset survived (#131).
    cin, cout = io.StringIO(_two_tool_calls()), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   stats_log=str(tmp_path))
    assert rc == 0
    warnings = [ln for ln in capsys.readouterr().err.splitlines() if "stats skipped" in ln]
    assert len(warnings) == 1 and warnings[0].startswith("[terse-proxy] gh.api.items: ")
    assert len([ln for ln in cout.getvalue().splitlines() if ln.strip()]) == 2


def test_run_proxy_debug_log_writes_replay_trace(tmp_path):
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "gh.api.items"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    log = tmp_path / "audit.jsonl"
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   debug_log=str(log))
    assert rc == 0
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # exactly the one tools/call result was logged (initialize is not a tool call)
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tool"] == "gh.api.items" and rec["id"] == 2 and rec["changed"] is True
    blk = rec["blocks"][0]
    assert json.loads(blk["raw"])["result"][0]["status"] == "active"   # raw payload
    assert transforms.decompress(blk["emitted"]) == json.loads(blk["raw"])  # lossless


# --- downstream lifecycle: no orphaned child (#21) ---

def test_terminate_child_reaps_running_downstream():
    import subprocess as sp

    from terse.proxy import _terminate_child

    proc = sp.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    assert proc.poll() is None                      # running
    _terminate_child(proc)
    assert proc.poll() is not None                  # reaped, not orphaned


def test_terminate_child_is_noop_on_already_exited():
    import subprocess as sp

    from terse.proxy import _terminate_child

    proc = sp.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    _terminate_child(proc)                           # must not raise on a dead child
    assert proc.poll() is not None


# --- #19: fail-fast on a downstream with nothing to proxy at all ---

def test_stdio_transport_error_only_flags_a_missing_command():
    # #5: a URL is now a valid, dispatchable downstream (HttpTransport) — no longer
    # rejected here. Only "nothing after --" remains an error.
    from terse.proxy import stdio_transport_error

    assert stdio_transport_error([]) is not None                       # nothing given
    assert stdio_transport_error(["https://example.com/mcp"]) is None  # URL: now OK
    assert stdio_transport_error(["sse://host/path"]) is None          # any scheme: OK
    assert stdio_transport_error(["uvx", "some-mcp-server"]) is None   # a real command
    assert stdio_transport_error([sys.executable, str(FAKE)]) is None


def test_run_proxy_rejects_empty_downstream_without_launching():
    cin, cout = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'), io.StringIO()
    rc = run_proxy([], FULL, stdin=cin, stdout=cout)
    assert rc == 2
    assert cout.getvalue() == ""        # nothing launched, nothing forwarded


def test_run_proxy_refuses_to_start_without_a_required_server_name(capsys):
    # The gap this closes: a server-scoped `require_server_name` rule needs
    # `_match_candidates` to synthesize the qualified candidate to ever match, which
    # only happens when `server` is truthy. Without this guard, omitting --server-name
    # would silently make the rule unreachable and fall through to the permissive
    # unmatched-tool default instead of refusing outright.
    pol = Policy(rules=[Rule("secret-broker.*", (), capture=False,
                             require_server_name=True)])
    cin, cout = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout)
    assert rc == 2
    assert cout.getvalue() == ""        # nothing launched, nothing forwarded
    err = capsys.readouterr().err
    assert "require_server_name" in err and "secret-broker.*" in err


def test_run_proxy_refuses_to_start_with_an_empty_server_name(capsys):
    # `Policy._match_candidates` gates the qualified candidate on `if server` (falsy),
    # so `--server-name ""` is exactly as unreachable there as omitting the flag. An
    # `is None` check here would let that empty string slip past this refusal into the
    # same silent-fallback gap the guard exists to close (review-caught).
    pol = Policy(rules=[Rule("secret-broker.*", (), capture=False,
                             require_server_name=True)])
    cin, cout = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout, server_name="")
    assert rc == 2
    assert cout.getvalue() == ""


def test_run_proxy_starts_normally_when_the_required_server_name_is_given():
    pol = Policy(rules=[Rule("secret-broker.*", (), capture=False,
                             require_server_name=True)])
    cin, cout = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout,
                   server_name="secret-broker", lazy_primer=False)
    assert rc == 0
    assert cout.getvalue() != ""        # the fake server actually ran and replied


def test_run_proxy_is_unaffected_by_missing_server_name_when_no_rule_requires_it():
    # Regression guard: FULL has no `require_server_name` rule, so the existing,
    # long-standing "server_name is optional" behavior must be completely unchanged.
    cin, cout = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   lazy_primer=False)
    assert rc == 0
    assert cout.getvalue() != ""


def test_run_proxy_reports_unlaunchable_command_cleanly():
    # a command that cannot be exec'd must surface as a clean exit code, not a traceback
    cin, cout = io.StringIO(""), io.StringIO()
    rc = run_proxy(["/no/such/terse-downstream-binary"], FULL, stdin=cin, stdout=cout)
    assert rc == 127
    assert cout.getvalue() == ""


# --- end-to-end through a real subprocess ---

def test_run_proxy_end_to_end_compresses_losslessly():
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "gh.api.items"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    by_id = {json.loads(ln)["id"]: json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()}

    # initialize: serverInfo intact, no eager primer (#168 phase 2 default)
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake"
    assert "instructions" not in by_id[1]["result"]
    # tools/call result: first compressible result, so the primer arrives as a LEADING
    # block ahead of the compressed data block, end-to-end over the real subprocess
    blocks = by_id[2]["result"]["content"]
    assert len(blocks) == 2
    assert PRIMER_HEAD in blocks[0]["text"]
    text = blocks[1]["text"]
    expected = {"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                           for i in range(20)]}
    assert transforms.decompress(text) == expected
    assert len(text) < len(_records_text())


def test_run_proxy_end_to_end_text_diffs_repeated_non_json_reads():
    # A real subprocess run (not the pure Interceptor) reading the "same file" twice via
    # fs.read, whose 2nd result has one line changed -- proves Tier 0.7 text (#25) fires
    # over the actual stdio pump, not just in isolated unit tests.
    pol = Policy(rules=[Rule("fs.*", ("minify", "tabularize", "dictionary"))], diff=True)
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "fs.read"}}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "fs.read"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    # lazy_primer=False: this test is about text-diff mechanics, not primer delivery.
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout, lazy_primer=False)
    assert rc == 0
    by_id = {json.loads(ln)["id"]: json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()}

    first = by_id[2]["result"]["content"][0]["text"]
    second = by_id[3]["result"]["content"][0]["text"]
    assert first == _log_text(200)                            # 1st read: untouched, no prior
    assert text_diff.DIFF_MARKER in second                     # 2nd read: a text diff was sent
    assert text_diff.text_diff_decode(first, json.loads(second)) == _log_text(200, changed_line=100)
    assert len(second) < len(_log_text(200, changed_line=100))  # actually smaller over the wire


# --- #23: audit/replay log ---

def _note_call(inter, mid, name):
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": name}}))


def test_audit_emits_one_record_per_result_in_order():
    records = []
    inter = Interceptor(FULL, audit=records.append)
    for mid in (1, 2):
        _note_call(inter, mid, "gh.api.items")
        inter.transform_response(_result_msg(mid, _records_text()))
    assert [r["id"] for r in records] == [1, 2]            # one record/result, in order
    rec = records[0]
    assert rec["tool"] == "gh.api.items"
    assert rec["changed"] is True
    assert rec["tiers"] == ["minify", "tabularize", "dictionary"]
    blk = rec["blocks"][0]
    assert blk["raw"] == _records_text()                   # raw snapshot, pre-transform
    assert blk["emitted"] != _records_text()               # emitted, post-transform
    assert transforms.decompress(blk["emitted"]) == json.loads(_records_text())  # lossless


def test_audit_logs_unchanged_passthrough_result():
    # A passthrough tool (no tiers) is left alone — still audited, since "terse touched
    # nothing" is exactly what you want recorded when a result looks wrong.
    records = []
    inter = Interceptor(Policy(rules=[Rule("gh.*", ())]), audit=records.append)
    _note_call(inter, 5, "gh.api.items")
    out = inter.transform_response(_result_msg(5, _records_text()))
    assert out == _result_msg(5, _records_text())          # byte-identical forward
    assert len(records) == 1
    rec = records[0]
    assert rec["changed"] is False
    assert rec["blocks"][0]["raw"] == rec["blocks"][0]["emitted"]  # raw == emitted


def test_audit_failure_never_breaks_forwarding():
    def boom(_record):
        raise RuntimeError("disk full")
    inter = Interceptor(FULL, audit=boom, lazy_primer=False)
    _note_call(inter, 9, "gh.api.items")
    out = inter.transform_response(_result_msg(9, _records_text()))
    # Forwarding is unaffected by the audit explosion: still the compressed, lossless result.
    text = json.loads(out)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == json.loads(_records_text())


def test_blocking_sink_does_not_stall_a_concurrent_note_request():
    """A sink that BLOCKS (not raises) must not hold `_local_lock`.

    The fail-open `try/except` around each sink only ever caught a sink that raised.
    A sink that hangs — full disk mid-retry, stalled network mount, slow fsync — used
    to hold `_local_lock` for the whole of `transform_response`, and `note_request`
    takes that same lock, so every subsequent tools/call on the connection wedged
    behind it. Sinks now run after the lock is released.
    """
    import threading

    entered = threading.Event()
    release = threading.Event()

    def hangs(_tool, _payload, **_kw):
        entered.set()
        release.wait(timeout=10)   # blocks; never raises

    inter = Interceptor(FULL, capture=hangs)
    _note_call(inter, 11, "gh.api.items")

    done = threading.Event()
    t = threading.Thread(
        target=lambda: (inter.transform_response(_result_msg(11, _records_text())),
                        done.set()),
        daemon=True)
    t.start()
    assert entered.wait(timeout=5), "capture sink was never reached"

    # The sink is mid-hang. A concurrent note_request must still complete — this is the
    # assertion that fails (times out) if the sink is invoked under `_local_lock`.
    noted = threading.Event()
    threading.Thread(
        target=lambda: (inter.note_request(json.dumps(
            {"jsonrpc": "2.0", "id": 12, "method": "tools/call",
             "params": {"name": "gh.api.items"}})), noted.set()),
        daemon=True).start()
    assert noted.wait(timeout=5), "note_request blocked behind a hanging sink"

    release.set()
    assert done.wait(timeout=5)
    t.join(timeout=5)


def test_no_audit_callback_is_byte_identical():
    plain = Interceptor(FULL)
    audited = Interceptor(FULL, audit=lambda _r: None)
    _note_call(plain, 3, "gh.api.items")
    _note_call(audited, 3, "gh.api.items")
    assert plain.transform_response(_result_msg(3, _records_text())) == \
        audited.transform_response(_result_msg(3, _records_text()))


def test_append_audit_writes_one_json_line_per_call(tmp_path):
    from terse.capture import append_audit
    log = tmp_path / "nested" / "audit.jsonl"           # parent created on demand
    append_audit({"tool": "a", "id": 1}, log)
    append_audit({"tool": "b", "id": 2}, log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [1, 2]


# --- #85: policy `"capture": false` — never persist this tool's payloads ---

SECRET = json.dumps({"credential": "sk-live-super-secret-value"})
# One proxy, two tools: only the credential-returning one is capture-gated. This shape
# is the point — the gate is per RULE, not per proxy (which `--capture-dir`'s presence
# or absence already gives you).
CAPTURE_GATED = Policy(rules=[Rule("secret.*", (), capture=False),
                             Rule("gh.*", ("minify", "tabularize", "dictionary"))])


def test_capture_false_blocks_the_corpus_tee_but_a_sibling_tool_still_captures():
    captured = []
    inter = Interceptor(CAPTURE_GATED, capture=lambda tool, raw, **kw: captured.append((tool, raw)))
    _note_call(inter, 1, "secret.reveal")
    inter.transform_response(_result_msg(1, SECRET))
    assert captured == []                                  # nothing persisted at all
    _note_call(inter, 2, "gh.api.items")
    inter.transform_response(_result_msg(2, _records_text()))
    assert [t for t, _ in captured] == ["gh.api.items"]    # sibling unaffected


def test_capture_false_blocks_the_audit_replay_log_too():
    # The audit record embeds the raw payload in blocks:[{raw, emitted}] — the identical
    # exposure. Gating only the corpus tee would be half a guard.
    records = []
    inter = Interceptor(CAPTURE_GATED, audit=records.append)
    _note_call(inter, 1, "secret.reveal")
    inter.transform_response(_result_msg(1, SECRET))
    assert records == []
    _note_call(inter, 2, "gh.api.items")
    inter.transform_response(_result_msg(2, _records_text()))
    assert [r["tool"] for r in records] == ["gh.api.items"]


def test_capture_false_still_counts_in_the_payload_free_stats_ledger():
    # The ledger records sizes + decision, never content — so a capture-gated tool is
    # still measured, just never quoted. Losing the row would be a needless blind spot.
    seen = []
    inter = Interceptor(CAPTURE_GATED, stats=lambda *a: seen.append(a))
    _note_call(inter, 1, "secret.reveal")
    inter.transform_response(_result_msg(1, SECRET))
    assert len(seen) == 1
    tool, raw, emitted, passthrough, reason, structured, structured_out = seen[0]
    assert tool == "secret.reveal" and passthrough is True and reason == "diff_off"
    assert raw == SECRET and emitted == SECRET             # passthrough: untouched


def test_capture_false_does_not_change_what_the_client_receives():
    plain = Interceptor(Policy(rules=[Rule("secret.*", ())]))
    gated = Interceptor(CAPTURE_GATED)
    _note_call(plain, 1, "secret.reveal")
    _note_call(gated, 1, "secret.reveal")
    assert plain.transform_response(_result_msg(1, SECRET)) == \
        gated.transform_response(_result_msg(1, SECRET))


def test_run_proxy_capture_false_writes_no_corpus_file_for_that_tool(tmp_path):
    # End-to-end through the real proxy + a real corpus dir: the gated tool's payload
    # must not exist on disk in any form.
    from terse.capture import load_corpus
    pol = Policy(rules=[Rule("gh.*", (), capture=False)])   # fake server's tool is gh.api.items
    requests = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "gh.api.items"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    corpus = tmp_path / "corpus"
    log = tmp_path / "audit.jsonl"
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout,
                   capture_dir=str(corpus), debug_log=str(log))
    assert rc == 0
    assert load_corpus(corpus) == []                        # no envelope written
    assert not log.exists() or log.read_text(encoding="utf-8") == ""
    # and the payload reached the client untouched
    assert "active" in cout.getvalue()


# --- savings-ledger stats callback (payload-free, always-on-able) ---

def test_stats_callback_sees_raw_and_emitted_per_result():
    from terse.stats import classify_decision
    seen = []
    # lazy_primer=False: this test pins content[0] as the emitted block, orthogonal to
    # primer delivery (#168 phase 2).
    inter = Interceptor(FULL, stats=lambda *a: seen.append(a), lazy_primer=False)
    _note_call(inter, 1, "gh.api.items")
    out = inter.transform_response(_result_msg(1, _records_text()))
    assert len(seen) == 1
    tool, raw, emitted, passthrough, reason, structured, structured_out = seen[0]
    assert tool == "gh.api.items" and passthrough is False and reason == "diff_off"
    assert raw == _records_text()                       # true pre-transform snapshot
    assert emitted == json.loads(out)["result"]["content"][0]["text"]
    assert classify_decision(raw, emitted, passthrough) == "compressed"


def test_stats_callback_works_without_audit_and_labels_a_diff():
    # stats alone must trigger the raw-text snapshot (it used to be audit-gated), and a
    # second same-tool call that ships a cross-call delta classifies as "diff".
    from terse.stats import classify_decision
    seen = []
    inter = Interceptor(DIFF, stats=lambda *a: seen.append(a))
    first = {"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                        for i in range(20)]}
    second = json.loads(json.dumps(first))
    second["result"][0]["status"] = "closed"
    _note_call(inter, 1, "gh.api.items")
    inter.transform_response(_result_msg(1, json.dumps(first)))
    _note_call(inter, 2, "gh.api.items")
    inter.transform_response(_result_msg(2, json.dumps(second)))
    assert [classify_decision(r, e, p)
            for (_t, r, e, p, _rsn, _sc, _so) in seen] == ["compressed", "diff"]
    assert [s[4] for s in seen] == ["no_prior", "emitted"]   # the diff_reason datum agrees


def test_stats_passthrough_tool_is_labeled_passthrough():
    from terse.stats import classify_decision
    seen = []
    inter = Interceptor(Policy(rules=[Rule("gh.*", ())]), stats=lambda *a: seen.append(a))
    _note_call(inter, 5, "gh.api.items")
    inter.transform_response(_result_msg(5, _records_text()))
    (tool, raw, emitted, passthrough, reason, structured, structured_out), = seen
    assert passthrough is True and raw == emitted and reason == "diff_off"
    assert classify_decision(raw, emitted, passthrough) == "passthrough"


def test_stats_failure_never_breaks_forwarding():
    def boom(*_a):
        raise RuntimeError("disk full")
    inter = Interceptor(FULL, stats=boom, lazy_primer=False)
    _note_call(inter, 9, "gh.api.items")
    out = inter.transform_response(_result_msg(9, _records_text()))
    text = json.loads(out)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == json.loads(_records_text())


def test_no_stats_callback_is_byte_identical():
    plain = Interceptor(FULL)
    counted = Interceptor(FULL, stats=lambda *_a: None)
    _note_call(plain, 3, "gh.api.items")
    _note_call(counted, 3, "gh.api.items")
    assert plain.transform_response(_result_msg(3, _records_text())) == \
        counted.transform_response(_result_msg(3, _records_text()))


def test_run_proxy_stats_log_writes_payload_free_ledger(tmp_path):
    from terse.stats import load_stats
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "gh.api.items"}}),
    ]) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    log = tmp_path / "stats.jsonl"
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   stats_log=str(log))
    assert rc == 0
    recs = load_stats(log)
    # Two rows, and they are different KINDS: the one tools/call result (initialize is not
    # a tool call), plus the lazy primer this session actually attached to it (#311). The
    # primer row is asserted on rather than filtered away silently, because "the ledger
    # grew a row" is exactly the kind of change a bare count hides.
    assert len(recs) == 2
    primer_rows = [r for r in recs if r.get("event") == "primer"]
    assert len(primer_rows) == 1
    assert primer_rows[0]["cadence"] == "once/session"
    assert primer_rows[0]["tokens"]
    assert "raw_chars" not in primer_rows[0]      # never enters the savings total
    rec = next(r for r in recs if r.get("event") != "primer")
    assert rec["tool"] == "gh.api.items"
    assert rec["server"] == pathlib.Path(sys.executable).name  # downstream identity
    assert rec["decision"] == "compressed"
    assert rec["raw_chars"] > rec["out_chars"] > 0
    # payload-free: nothing from the fake server's records leaks into the ledger
    assert "active" not in log.read_text(encoding="utf-8")


def test_interceptor_server_name_makes_a_server_scoped_rule_match(tmp_path):
    # End-to-end of #83 through the real message path: the policy names a server-scoped
    # rule, the tool arrives bare. Without server_name the rule misses (defaults compress
    # it); with it, the rule's passthrough tiers take effect.
    pol = Policy(rules=[Rule("runecho.*", ())])          # () = hands off entirely
    blind = Interceptor(pol)
    named = Interceptor(pol, server_name="runecho")
    _note_call(blind, 1, "structure")
    _note_call(named, 1, "structure")
    assert blind.transform_response(_result_msg(1, _records_text())) != \
        _result_msg(1, _records_text())                  # rule missed -> defaults ran
    assert named.transform_response(_result_msg(1, _records_text())) == \
        _result_msg(1, _records_text())                  # rule matched -> passthrough


def test_run_proxy_stats_server_name_labels_the_ledger_over_the_command_basename(tmp_path):
    # The command basename misreads a launcher-wrapped server (kb behind sb-run labels
    # itself "sb-run"); the config's own name is the truthful identity (#83).
    from terse.stats import load_stats
    requests = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "gh.api.items"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    log = tmp_path / "stats.jsonl"
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   stats_log=str(log), server_name="runecho")
    assert rc == 0
    recs = load_stats(log)
    assert recs[0]["server"] == "runecho"          # not "python" (the basename fallback)


def test_run_proxy_stats_default_none_writes_nothing(tmp_path, monkeypatch):
    # The API default is disabled (None) — only cli.py resolves the default-ON path —
    # so a direct run_proxy caller must leave $XDG_STATE_HOME untouched.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    requests = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "gh.api.items"}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout)
    assert rc == 0
    assert not (tmp_path / "terse").exists()


# --- drop-to-retrieve: store + tools/list injection (#10, Phase 2) ---

DROP = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                          fields={"result[].body": {"lossy": "drop-to-retrieve"}})])


def _tools_list(mid, names):
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"tools": [{"name": n} for n in names]}})


def test_injects_retrieve_tool_into_tools_list_when_drop_enabled():
    inter = Interceptor(DROP)
    out = json.loads(inter.transform_response(_tools_list(1, ["gh.api.items"])))
    assert "terse.retrieve" in [t["name"] for t in out["result"]["tools"]]
    # idempotent: re-listing an already-injected list doesn't duplicate it
    again = json.loads(inter.transform_response(json.dumps(out)))
    assert [t["name"] for t in again["result"]["tools"]].count("terse.retrieve") == 1


def test_no_retrieve_tool_when_drop_disabled():
    inter = Interceptor(FULL)                                  # no drop-marked fields
    tl = _tools_list(1, ["gh.api.items"])
    out = inter.transform_response(tl)
    assert out == tl and "terse.retrieve" not in out           # forwarded unchanged


def test_drop_result_populates_store_and_carries_the_marker():
    inter = Interceptor(DROP)
    out = _emit(inter, 9, "gh.api.items", {"result": [{"id": 1, "body": "B" * 400}]})
    assert transforms.DROPPED_MARKER in out                    # emitted with a handle
    assert len(inter.dropped) == 1
    handle = next(iter(inter.dropped))
    assert inter.dropped[handle] == "B" * 400                  # original stored, recoverable


def test_reconnect_clears_the_drop_store():
    inter = Interceptor(DROP)
    _emit(inter, 9, "gh.api.items", {"result": [{"id": 1, "body": "B" * 400}]})
    assert inter.dropped and inter._dropped_bytes_box[0] > 0
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize"}))
    assert len(inter.dropped) == 0 and inter._dropped_bytes_box[0] == 0


def test_drop_store_evicts_lru_over_count_cap():
    inter = Interceptor(DROP)
    inter.DROPPED_MAX = 3                                       # shadow the class cap
    for i in range(5):
        inter._drop_put(f"h{i}", "x" * 10)
    assert list(inter.dropped) == ["h2", "h3", "h4"]           # two oldest evicted


def test_drop_store_evicts_over_byte_cap():
    inter = Interceptor(DROP)
    inter.DROPPED_MAX_BYTES = 25
    inter._drop_put("a", "x" * 10)
    inter._drop_put("b", "y" * 10)
    inter._drop_put("c", "z" * 10)                             # 30 > 25 -> evict oldest (a)
    assert "a" not in inter.dropped and inter._dropped_bytes_box[0] == 20


def test_drop_store_refreshes_recency_on_reinsert():
    inter = Interceptor(DROP)
    inter.DROPPED_MAX = 2
    inter._drop_put("a", "x" * 10)
    inter._drop_put("b", "y" * 10)
    inter._drop_put("a", "x" * 10)                             # touch a -> most-recent
    inter._drop_put("c", "z" * 10)                             # evict LRU = b
    assert list(inter.dropped) == ["a", "c"] and inter._dropped_bytes_box[0] == 20


def test_shared_store_lock_does_not_serialize_unrelated_peers_transform_response():
    # Regression: a single Lock() shared across every peer's Interceptor used to be
    # held for transform_response's ENTIRE body (compression + capture/audit I/O), so
    # one slow peer's response processing blocked every other peer sharing that lock —
    # even though only the drop store (self.dropped/_dropped_bytes_box) actually needs
    # cross-peer exclusion. _local_lock (always private) now covers the bulk of the
    # method; _store_lock (the one multiproxy shares) covers only _drop_put/
    # answer_retrieve. Prove it directly: a slow capture callback on peer A must not
    # delay peer B's transform_response, even though they share a store_lock.
    import threading
    import time

    shared_store: dict = {}
    shared_store_lock = threading.Lock()

    started_a = threading.Event()
    release_a = threading.Event()

    def slow_capture(tool, raw, **kw):
        started_a.set()
        release_a.wait(timeout=5)  # blocks peer A's transform_response indefinitely

    inter_a = Interceptor(FULL, capture=slow_capture, store=shared_store,
                          store_lock=shared_store_lock)
    inter_b = Interceptor(FULL, store=shared_store, store_lock=shared_store_lock)
    inter_a.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                     "params": {"name": "gh.api.items"}}))
    inter_b.note_request(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                     "params": {"name": "gh.api.items"}}))

    t_a = threading.Thread(target=lambda: inter_a.transform_response(
        _result_msg(1, _records_text())))
    t_a.start()
    assert started_a.wait(timeout=5)  # peer A is now blocked mid-transform_response

    # peer B's transform_response must complete promptly — NOT wait for peer A.
    start = time.monotonic()
    out_b = inter_b.transform_response(_result_msg(2, _records_text()))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"peer B waited {elapsed:.2f}s — still serialized behind peer A"
    assert json.loads(out_b)["result"]["content"][0]["text"] != _records_text()  # B compressed fine

    release_a.set()
    t_a.join(timeout=5)


# --- drop-to-retrieve: serving terse.retrieve (#10, Phase 3) ---

def _retrieve_call(mid, handle):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": "terse.retrieve", "arguments": {"handle": handle}}})


def test_answer_retrieve_returns_the_stored_original():
    inter = Interceptor(DROP)
    inter._drop_put("abc123", "the original body value")
    reply = json.loads(inter.answer_retrieve(_retrieve_call(5, "abc123")))
    assert reply["id"] == 5
    assert reply["result"]["content"][0]["text"] == "the original body value"
    assert not reply["result"].get("isError")


def test_answer_retrieve_serializes_a_structured_original():
    inter = Interceptor(DROP)
    inter._drop_put("h", {"a": [1, 2, 3]})
    reply = json.loads(inter.answer_retrieve(_retrieve_call(9, "h")))
    assert json.loads(reply["result"]["content"][0]["text"]) == {"a": [1, 2, 3]}


def test_answer_retrieve_miss_is_a_legible_error_not_a_protocol_error():
    inter = Interceptor(DROP)
    reply = json.loads(inter.answer_retrieve(_retrieve_call(6, "gone")))
    assert reply["id"] == 6 and reply["result"]["isError"] is True
    assert "no longer available" in reply["result"]["content"][0]["text"]


def test_answer_retrieve_ignores_non_retrieve_lines():
    inter = Interceptor(DROP)
    assert inter.answer_retrieve(_req(7, "gh.api.items")) is None          # a real tool call
    assert inter.answer_retrieve("not json") is None
    assert inter.answer_retrieve(
        json.dumps({"jsonrpc": "2.0", "id": 8, "method": "initialize"})) is None


def test_pump_swallow_writes_nothing_else_forwards():
    from terse.proxy import SWALLOW, pump
    src = io.StringIO("keep\ndrop\nkeep2\n")
    dst = io.StringIO()
    pump(src, dst, lambda line: SWALLOW if line == "drop" else None)
    assert dst.getvalue().splitlines() == ["keep", "keep2"]


def test_run_proxy_injects_retrieve_tool_into_a_live_tools_list():
    requests = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                           "params": {}}) + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], DROP, stdin=cin, stdout=cout)
    assert rc == 0
    resp = json.loads([ln for ln in cout.getvalue().splitlines() if ln.strip()][0])
    assert "terse.retrieve" in [t["name"] for t in resp["result"]["tools"]]


def test_primer_documents_the_drop_marker_and_retrieve_tool():
    # Load-bearing: without this the model sees an opaque marker and never fetches the value.
    from terse.proxy import TERSE_PRIMER
    assert transforms.DROPPED_MARKER in TERSE_PRIMER
    assert "terse.retrieve" in TERSE_PRIMER


def test_run_proxy_answers_retrieve_without_forwarding_downstream():
    # A miss handle is enough to prove the swallow: the reply is OUR synthesized error, and
    # the downstream fake never saw the call (it would have returned records if forwarded).
    requests = _retrieve_call(1, "nope") + "\n"
    cin, cout = io.StringIO(requests), io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], DROP, stdin=cin, stdout=cout)
    assert rc == 0
    resp = json.loads([ln for ln in cout.getvalue().splitlines() if ln.strip()][0])
    assert resp["id"] == 1 and resp["result"]["isError"] is True
    assert '"status"' not in resp["result"]["content"][0]["text"]           # not the fake's records


# --- drop-to-retrieve over a TEXT payload (`$text.code_blocks`) --------------------- #

TEXT_DROP = Policy(rules=[Rule("codegraph_*", ("minify", "tabularize", "dictionary"),
                               fields={"$text.code_blocks":
                                       {"lossy": "drop-to-retrieve"}})])
_SRC = "\n".join(f"    line {i} of a source file long enough to matter" for i in range(20))
_MD = f"## Exploration\n\nFound 3 symbols.\n\n#### src/a.py\n\n```python\n{_SRC}\n```\n"


def _emit_text(inter, mid, tool, text):
    """`_emit` for a non-JSON payload: the raw text goes on the wire as-is."""
    inter.note_request(_req(mid, tool))
    out = inter.transform_response(_result_msg(mid, text))
    return json.loads(out)["result"]["content"][0]["text"]


def test_text_drop_emits_marker_stores_original_and_retrieve_serves_it_back():
    inter = Interceptor(TEXT_DROP, lazy_primer=False)
    out = _emit_text(inter, 1, "codegraph_explore", _MD)
    assert transforms.DROPPED_MARKER in out
    assert "line 10 of a source file" not in out       # the block really left the wire
    assert "Found 3 symbols." in out                   # the prose really stayed
    handle = next(iter(inter.dropped))
    # The retrieve tool must serve back the exact bytes, through the real proxy handler.
    served = json.loads(inter.answer_retrieve(
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "terse.retrieve",
                               "arguments": {"handle": handle}}})))
    assert served["result"]["content"][0]["text"] == f"```python\n{_SRC}\n```\n"


def test_text_drop_clears_the_text_diff_base():
    inter = Interceptor(dataclasses.replace(TEXT_DROP, diff=True))
    _emit_text(inter, 1, "codegraph_explore", "plain text result with no fences at all")
    assert inter.last_text.get("codegraph_explore") is not None   # normal CDC base stored
    _emit_text(inter, 2, "codegraph_explore", _MD)
    # A dropped payload must not become a diff base: the next raw text re-anchors full.
    assert "codegraph_explore" not in inter.last_text


def test_text_payload_untouched_without_a_text_selector():
    inter = Interceptor(FULL)
    assert _emit_text(inter, 1, "codegraph_explore", _MD) == _MD


# --- #116: multi-block join (cross-block tabularize + diff unlock) ---

def _rec_blocks(n, change=None):
    rows = [{"id": i, "status": "active", "url": "https://x.example/api/items"}
            for i in range(n)]
    if change is not None:
        rows[change]["status"] = "closed"
    return [json.dumps(r) for r in rows]


def _msg_content(mid, content):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": {"content": content}})


def _emit_multi(inter, mid, tool, texts, extra_blocks=None):
    """Emit a multi-text-block result; return the emitted content list. `extra_blocks`,
    if given, is a list of (index, block) to splice in among the text blocks."""
    content = [{"type": "text", "text": t} for t in texts]
    if extra_blocks:
        for idx, block in extra_blocks:
            content.insert(idx, block)
    inter.note_request(_req(mid, tool))
    out = inter.transform_response(_msg_content(mid, content))
    return json.loads(out)["result"]["content"]


def test_join_collapses_n_text_blocks_to_one_record_array():
    inter = Interceptor(DIFF, lazy_primer=False)
    raws = _rec_blocks(5)
    content = _emit_multi(inter, 1, "gh.api.items", raws)
    assert len(content) == 1 and content[0]["type"] == "text"
    assert transforms.TABLE_MARKER in content[0]["text"]          # folded across blocks
    assert transforms.decompress(content[0]["text"]) == [json.loads(r) for r in raws]


def test_join_preserves_non_text_blocks_in_position():
    inter = Interceptor(DIFF)
    raws = _rec_blocks(2)
    image = {"type": "image", "data": "abc", "mimeType": "image/png"}
    link = {"type": "resource_link", "uri": "file:///x"}
    # order: image, text0, text1, link — the joined block takes the FIRST text slot
    content = _emit_multi(inter, 1, "gh.api.items", raws,
                          extra_blocks=[(0, image), (3, link)])
    assert [b["type"] for b in content] == ["image", "text", "resource_link"]
    assert content[0] == image and content[2] == link
    assert transforms.decompress(content[1]["text"]) == [json.loads(r) for r in raws]


def test_join_emits_a_diff_on_the_second_same_tool_result():
    inter = Interceptor(DIFF)
    prev, curr = _rec_blocks(40), _rec_blocks(40, change=3)
    _emit_multi(inter, 1, "gh.api.items", prev)
    content = _emit_multi(inter, 2, "gh.api.items", curr)
    assert len(content) == 1
    env = json.loads(content[0]["text"])
    assert env.get(transforms.DIFF_MARKER) == 1                  # the 71% unlock: a diff!
    assert transforms.diff_decode([json.loads(r) for r in prev], env) == \
        [json.loads(r) for r in curr]


def test_join_reports_one_stats_record_not_n():
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    _emit_multi(inter, 1, "gh.api.items", _rec_blocks(5))
    assert reasons == ["no_prior"]                              # ONE record, first = full


def test_join_audit_pairs_the_joined_block_with_newline_joined_raw():
    records = []
    inter = Interceptor(DIFF, audit=records.append)
    raws = _rec_blocks(3)
    _emit_multi(inter, 1, "gh.api.items", raws)
    assert len(records) == 1
    blocks = records[0]["blocks"]
    assert len(blocks) == 1                                     # single (raw, emitted) pair
    assert blocks[0]["raw"] == "\n".join(raws)                  # true wire cost the model saw


def test_join_captures_the_array_once_not_per_block(tmp_path):
    captured = []
    inter = Interceptor(DIFF, capture=lambda tool, text, **kw: captured.append(text))
    raws = _rec_blocks(4)
    _emit_multi(inter, 1, "gh.api.items", raws)
    assert len(captured) == 1                                   # one corpus payload, not 4
    assert json.loads(captured[0]) == [json.loads(r) for r in raws]  # the joined array shape


def _reason_of(inter, mid, tool, texts):
    reasons, stats = _capture_stats()
    inter.stats = stats
    _emit_multi(inter, mid, tool, texts)
    return reasons[-1]


def test_join_refusals_fall_back_to_per_block_and_record_why():
    good = _rec_blocks(2)

    # non-JSON block: 2 blocks stay, reason names the refusal
    inter = Interceptor(DIFF)
    reasons, stats = _capture_stats()
    inter.stats = stats
    content = _emit_multi(inter, 1, "gh.api.items", [good[0], "not json {"])
    assert len(content) == 2                                    # NOT collapsed
    assert reasons[-1] == "multiblock_non_json"

    # heterogeneous (a non-dict block)
    assert _reason_of(Interceptor(DIFF), 1, "gh.api.items",
                      [good[0], json.dumps([1, 2, 3])]) == "multiblock_heterogeneous"

    # marker collision
    assert _reason_of(Interceptor(DIFF), 1, "gh.api.items",
                      [json.dumps({transforms.TABLE_MARKER: 1}), good[0]]) == \
        "multiblock_marker"

    # join disabled by policy
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))],
                 diff=True, join_blocks=False)
    assert _reason_of(Interceptor(off), 1, "gh.api.items", good) == "multiblock_off"

    # explicit passthrough tier
    passthru = Policy(rules=[Rule("gh.*", ())], diff=True)
    assert _reason_of(Interceptor(passthru), 1, "gh.api.items", good) == \
        "multiblock_passthrough"


# --- #140: partial join — fold the record run, leave the non-records per-block ---

def test_partial_join_folds_record_run_beside_a_trailing_error_string():
    # kb.read.list_principles in the wild: many object blocks + one bare error string.
    # The FULL join refuses (non_json), but the records must still fold.
    inter = Interceptor(DIFF, lazy_primer=False)
    raws = _rec_blocks(5)
    err = "Error executing tool gh.api.items: upstream 503"
    content = _emit_multi(inter, 1, "gh.api.items", raws + [err])
    assert len(content) == 2                                      # folded block + the error
    assert transforms.TABLE_MARKER in content[0]["text"]         # records folded across blocks
    assert transforms.decompress(content[0]["text"]) == [json.loads(r) for r in raws]
    assert content[1]["text"] == err                             # error left byte-for-byte


def test_partial_join_folds_around_an_interspersed_non_record_block():
    # A JSON array between two record runs breaks the run: each contiguous run of >=2
    # objects folds on its own, the array passes through in place.
    inter = Interceptor(DIFF)
    a, b = _rec_blocks(3), _rec_blocks(2)
    arr = json.dumps([1, 2, 3])
    content = _emit_multi(inter, 1, "gh.api.items", a + [arr] + b)
    assert len(content) == 3
    assert transforms.decompress(content[0]["text"]) == [json.loads(r) for r in a]
    assert json.loads(content[1]["text"]) == [1, 2, 3]           # the array, per-block (minified)
    assert transforms.decompress(content[2]["text"]) == [json.loads(r) for r in b]


def test_partial_join_records_multiblock_partial_reason():
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    _emit_multi(inter, 1, "gh.api.items", _rec_blocks(4) + ["bare error"])
    assert reasons and all(r == "multiblock_partial" for r in reasons)


def test_partial_join_does_not_fold_a_lone_record_beside_a_non_record():
    # One object + one error string: no run of >=2 objects, so nothing folds and the
    # result stays on the plain per-block path.
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    content = _emit_multi(inter, 1, "gh.api.items", [_rec_blocks(1)[0], "bare error"])
    assert len(content) == 2
    assert reasons[-1] == "multiblock_non_json"                  # fell back, did not partial-fold


def test_partial_join_captures_folded_array_and_leftover_separately(tmp_path):
    captured = []
    inter = Interceptor(DIFF, capture=lambda tool, text, **kw: captured.append(text))
    raws = _rec_blocks(3)
    err = "boom"
    _emit_multi(inter, 1, "gh.api.items", raws + [err])
    assert len(captured) == 2                                    # the array + the leftover
    assert json.loads(captured[0]) == [json.loads(r) for r in raws]
    assert captured[1] == err


def test_partial_join_audit_pairs_the_run_and_the_leftover():
    records = []
    inter = Interceptor(DIFF, audit=records.append)
    raws = _rec_blocks(3)
    err = "boom"
    _emit_multi(inter, 1, "gh.api.items", raws + [err])
    blocks = records[0]["blocks"]
    assert len(blocks) == 2
    assert blocks[0]["raw"] == "\n".join(raws)                   # folded run's true wire cost
    assert blocks[1]["raw"] == err


def test_partial_join_establishes_no_diff_base():
    # A partial fold must NOT leave a diff base: the next same-tool result re-anchors as a
    # full rather than diffing against a folded subset whose boundaries may have moved.
    inter = Interceptor(DIFF)
    _emit_multi(inter, 1, "gh.api.items", _rec_blocks(5) + ["err"])
    assert "gh.api.items" not in inter.last                      # no base parked
    assert "gh.api.items" not in inter.last_joined
    # a subsequent all-record result for the same tool sends a full, not a diff
    content = _emit_multi(inter, 2, "gh.api.items", _rec_blocks(5))
    assert transforms.DIFF_MARKER not in content[0]["text"]


def test_partial_join_stays_lossless_reconstructs_every_block():
    inter = Interceptor(DIFF)
    a = _rec_blocks(4)
    arr, err = json.dumps({"nested": [1, 2]}), "plain error text"
    # object-run, a lone object (arr is a dict here -> counts as a record but lone), error
    content = _emit_multi(inter, 1, "gh.api.items", a + [arr, err])
    # a[0..3] fold; arr is a dict adjacent to the run? No — it IS a record, so it extends
    # the run to 5. Reconstruct the whole folded array + the error.
    assert transforms.decompress(content[0]["text"]) == \
        [json.loads(r) for r in a] + [json.loads(arr)]
    assert content[1]["text"] == err


def test_join_to_single_shape_flip_re_anchors_instead_of_cross_shape_diff():
    inter = Interceptor(DIFF)
    _emit_multi(inter, 1, "gh.api.items", _rec_blocks(5))     # joins -> base is an array
    assert inter.last_joined.get("gh.api.items") is True
    # a single-block result for the same tool: the shapes are incompatible (array vs the
    # {"result": [...]} object), so it must re-anchor as a full, not diff across the flip
    single = _emit(inter, 2, "gh.api.items", _records(40))
    assert transforms.DIFF_MARKER not in single
    assert inter.last_joined.get("gh.api.items") is False
    # once re-anchored on the single shape, the NEXT single result diffs normally
    assert transforms.DIFF_MARKER in _emit(inter, 3, "gh.api.items", _records(40, change=2))


def test_join_shape_flip_reason_is_reanchor():
    reasons, stats = _capture_stats()
    inter = Interceptor(DIFF, stats=stats)
    _emit_multi(inter, 1, "gh.api.items", _rec_blocks(5))
    _emit(inter, 2, "gh.api.items", _records(5))
    assert reasons[-1] == "reanchor"


def test_join_off_still_compresses_each_block_and_labels_multiblock_off():
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))],
                 diff=True, join_blocks=False)
    reasons, stats = _capture_stats()
    inter = Interceptor(off, stats=stats)
    raws = _rec_blocks(3)
    content = _emit_multi(inter, 1, "gh.api.items", raws)
    assert len(content) == 3                                    # each block kept
    for b, r in zip(content, raws, strict=True):
        assert transforms.decompress(b["text"]) == json.loads(r)  # still compressed, lossless
    assert reasons[-1] == "multiblock_off"


def test_join_on_iserror_result_stays_fully_lossless():
    inter = Interceptor(DIFF)
    raws = _rec_blocks(3)
    content = [{"type": "text", "text": t} for t in raws]
    inter.note_request(_req(1, "gh.api.items"))
    msg = json.dumps({"jsonrpc": "2.0", "id": 1,
                      "result": {"content": content, "isError": True}})
    out = json.loads(inter.transform_response(msg))["result"]["content"]
    assert len(out) == 1                                        # still joined (all JSON dicts)
    assert transforms.decompress(out[0]["text"]) == [json.loads(r) for r in raws]


def test_join_diff_off_folds_records_but_never_diffs():
    nodiff = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))],
                    diff=False, join_blocks=True)
    reasons, stats = _capture_stats()
    inter = Interceptor(nodiff, stats=stats, lazy_primer=False)
    raws = _rec_blocks(5)
    c1 = _emit_multi(inter, 1, "gh.api.items", raws)
    c2 = _emit_multi(inter, 2, "gh.api.items", raws)
    assert len(c1) == 1 and transforms.TABLE_MARKER in c1[0]["text"]   # folded
    assert transforms.DIFF_MARKER not in c2[0]["text"]                # but never a diff
    assert reasons[-1] == "joined"


# --- server-initiated requests must not consume a tracked call's pending entry ---

def test_server_initiated_request_with_colliding_id_does_not_break_tracking():
    # JSON-RPC gives each direction its own id space and both sides conventionally number
    # from 1, so a server's roots/list (or sampling/createMessage) id routinely collides
    # with an in-flight tools/call id. Popping `pending` for it left the REAL result
    # untracked: silently forwarded UNCOMPRESSED and missing from the ledger.
    reasons, stats = _capture_stats()
    inter = Interceptor(FULL, stats=stats, lazy_primer=False)
    inter.note_request(_req(1, "gh.api.items"))
    assert 1 in inter.pending

    server_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "roots/list"})
    assert inter.transform_response(server_req) == server_req   # forwarded byte-for-byte
    assert 1 in inter.pending                                   # tracking SURVIVES

    payload = _records(30)
    out = inter.transform_response(_result_msg(1, json.dumps(payload)))
    text = json.loads(out)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == payload               # still compressed, lossless
    assert transforms.TABLE_MARKER in text
    assert reasons                                              # and still recorded


def test_method_bearing_response_still_takes_the_response_path():
    # The guard must not be "has a method key" alone: a message carrying BOTH `method` and
    # a `result` is a response (however spec-sloppy), not a server-initiated request. If it
    # were forwarded as a request, every such result would silently go uncompressed and its
    # `pending` entry would leak to PENDING_MAX eviction. Same predicate multiproxy uses.
    inter = Interceptor(FULL, lazy_primer=False)
    inter.note_request(_req(2, "gh.api.items"))
    payload = _records(30)
    odd = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})
    out = inter.transform_response(odd)
    assert transforms.decompress(json.loads(out)["result"]["content"][0]["text"]) == payload
    assert 2 not in inter.pending          # consumed as the response it is


def test_true_notification_returns_before_the_server_request_guard():
    # A real notification has NO id and returns at the id-is-None check, ahead of the new
    # guard -- the invariant that keeps the guard from being reached by accident.
    inter = Interceptor(FULL)
    note = json.dumps({"jsonrpc": "2.0", "method": "notifications/message",
                       "params": {"level": "info", "data": "hello"}})
    assert inter.transform_response(note) == note


def test_server_initiated_request_colliding_with_initialize_id_keeps_the_primer():
    # The init_id branch has the same exposure: a server request colliding with the
    # initialize id would consume it, and the REAL initialize reply would then never get
    # the terse primer injected. lazy_primer=False: exercises the still-preserved eager
    # path (as run by every multiproxy peer).
    inter = Interceptor(FULL, lazy_primer=False)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "initialize"}))
    server_req = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "roots/list"})
    assert inter.transform_response(server_req) == server_req
    reply = json.dumps({"jsonrpc": "2.0", "id": 5,
                        "result": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    out = json.loads(inter.transform_response(reply))
    assert "terse" in (out["result"].get("instructions") or "").lower()   # primer survived


# --- #128: compressing the typed `structuredContent` field (opt-in) ---

def _structured_result_msg(mid, payload):
    """A spec-shaped pair: the serialized JSON in a text block AND the typed field."""
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": json.dumps(payload)}],
                                  "structuredContent": payload}})


_SC_PAYLOAD = {"rows": [{"id": i, "status": "active", "city": "Berlin"} for i in range(12)]}


def _structured_policy(mode):
    return Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                              structured=mode)])


def test_structured_content_is_left_alone_by_default(tmp_path):
    # The default must stay the pre-#128 behavior: the typed field carries a declared
    # outputSchema, and a client may hand it straight to a validator.
    inter = Interceptor(_structured_policy("leave"))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    assert out["result"]["structuredContent"] == _SC_PAYLOAD          # byte-identical
    # ...while the text block IS compressed, exactly as before
    assert "__terse_" in out["result"]["content"][0]["text"]


def test_structured_content_is_compressed_when_the_rule_opts_in(tmp_path):
    inter = Interceptor(_structured_policy("compress"))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    sc = out["result"]["structuredContent"]
    # the typed field now carries a terse envelope...
    assert "__terse_table__" in json.dumps(sc) or "__terse_dict__" in json.dumps(sc)
    # ...that still losslessly decodes to the original, which is the whole contract
    assert transforms.decompress(json.dumps(sc)) == _SC_PAYLOAD
    # and it is genuinely smaller than what it replaced
    assert len(json.dumps(sc)) < len(json.dumps(_SC_PAYLOAD))


def test_structured_content_absent_is_not_invented(tmp_path):
    inter = Interceptor(_structured_policy("compress"))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_result_msg(1, _records_text())))
    assert "structuredContent" not in out["result"]


def test_structured_content_ledger_counts_the_compressed_size(tmp_path):
    # The ledger must follow the field, and the two SIDES must not agree once it is
    # compressed (#141): the raw side carries the ORIGINAL typed field, the emitted side
    # the compressed one. Charging the compressed size to both (the pre-#141 bug)
    # understated the real wire saving.
    seen = []
    inter = Interceptor(_structured_policy("compress"), stats=lambda *a: seen.append(a))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD))
    structured_raw, structured_out = seen[0][5], seen[0][6]
    original = json.dumps(_SC_PAYLOAD, separators=(",", ":"))
    # Raw side: the original, uncompressed — NOT the compressed form (that was the bug).
    assert structured_raw == original
    assert "__terse_" not in structured_raw
    # Emitted side: the compressed form, genuinely smaller.
    assert structured_out is not None and "__terse_" in structured_out
    assert len(structured_out) < len(original)
    # And the record built from them charges each side its own size, so the saving is honest.
    from terse.stats import build_record
    rec = build_record("s", "gh.items", "", "", False, None, structured_raw, structured_out)
    assert rec["structured_chars"] == len(original)          # raw side = full price
    assert rec["structured_out_chars"] == len(structured_out)  # emitted side = compressed
    assert rec["raw_chars"] > rec["out_chars"]               # a real, non-understated saving


def _init_req_for(client: str | None):
    params: dict = {"protocolVersion": "2025-06-18", "capabilities": {}}
    if client is not None:
        params["clientInfo"] = {"name": client, "version": "9.9.9"}
    return json.dumps({"jsonrpc": "2.0", "id": 99, "method": "initialize",
                       "params": params})


def _auto_run(client: str | None):
    """Handshake as `client`, then call a tool returning a structuredContent pair."""
    inter = Interceptor(Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))]))
    inter.note_request(_init_req_for(client))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    return out["result"]["structuredContent"]


def test_structured_auto_compresses_for_a_measured_safe_client():
    # The default is "auto": with no `structured` key in the policy at all, a client
    # measured not to validate the typed field gets it compressed (#128).
    sc = _auto_run("claude-code")
    assert transforms.decompress(json.dumps(sc)) == _SC_PAYLOAD      # still lossless
    assert len(json.dumps(sc)) < len(json.dumps(_SC_PAYLOAD))


def test_structured_auto_fails_closed_for_unknown_and_absent_clients():
    # An unlisted client, and a handshake with no clientInfo at all, must both keep the
    # server's own object byte-identical — the conservative branch is the default one.
    assert _auto_run("some-other-client") == _SC_PAYLOAD
    assert _auto_run(None) == _SC_PAYLOAD


def test_structured_auto_fails_closed_with_no_handshake_at_all():
    # A library caller driving Interceptor directly never feeds it an initialize.
    inter = Interceptor(Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))]))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    assert out["result"]["structuredContent"] == _SC_PAYLOAD


def test_explicit_leave_overrides_a_safe_client():
    # The operator's explicit setting outranks the client-based resolution, both ways.
    inter = Interceptor(_structured_policy("leave"))
    inter.note_request(_init_req_for("claude-code"))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    assert out["result"]["structuredContent"] == _SC_PAYLOAD


# --- #128 option 2: `"structured": "replace"` drops the redundant text mirror ---

def _replace_run(result: dict, *, mode="replace", tiers=("minify", "tabularize", "dictionary"),
                 stats=None):
    """Drive one tools/call whose result is `result`, under `structured=mode`."""
    # lazy_primer=False: these tests are about the mirror-drop mechanism, not primer
    # delivery (that has its own dedicated coverage, #168 phase 2).
    inter = Interceptor(Policy(rules=[Rule("gh.*", tiers, structured=mode)]), stats=stats,
                        lazy_primer=False)
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result})
    return json.loads(inter.transform_response(line))["result"]


def _pair(payload, **extra):
    return {"content": [{"type": "text", "text": json.dumps(payload)}],
            "structuredContent": payload, **extra}


def test_structured_replace_drops_the_mirror():
    out = _replace_run(_pair(_SC_PAYLOAD))
    # The mirror is gone entirely — an empty `content` is the shape measured to reach the
    # model intact (`scripts/probe/structured_content/`, the `nomirror` probe).
    assert out["content"] == []
    # ...and everything the model needs is in the typed field, still losslessly.
    assert transforms.decompress(json.dumps(out["structuredContent"])) == _SC_PAYLOAD


def test_structured_replace_keeps_a_block_that_is_not_a_faithful_mirror():
    # The guard that makes this safe rather than merely measured: a block carrying
    # something the typed field does not is not a mirror, whatever the spec calls it.
    result = _pair(_SC_PAYLOAD)
    result["content"][0]["text"] = json.dumps({"note": "half the story", "rows": []})
    out = _replace_run(result)
    assert len(out["content"]) == 1
    assert out["content"][0]["text"]                       # still there, still non-empty


def test_structured_replace_keeps_the_mirror_on_an_error_result():
    # A model recovering from a failure has to be able to READ the failure.
    out = _replace_run(_pair(_SC_PAYLOAD, isError=True))
    assert len(out["content"]) == 1


def test_structured_replace_does_nothing_without_a_typed_field():
    # Nothing to fall back on: dropping the block here would blank the result.
    out = _replace_run({"content": [{"type": "text", "text": json.dumps(_SC_PAYLOAD)}]})
    assert len(out["content"]) == 1
    assert "structuredContent" not in out


def test_structured_replace_is_inert_when_the_rule_has_no_tiers():
    # `tiers: []` is the "hands off this tool" switch; removing a block is the most
    # hands-on thing terse does.
    out = _replace_run(_pair(_SC_PAYLOAD), tiers=())
    assert len(out["content"]) == 1
    assert out["structuredContent"] == _SC_PAYLOAD


def test_structured_replace_keeps_a_non_json_block():
    result = _pair(_SC_PAYLOAD)
    result["content"][0]["text"] = "Fetched 12 rows."      # prose, not a serialization
    out = _replace_run(result)
    assert out["content"][0]["text"] == "Fetched 12 rows."


def test_structured_replace_keeps_multiple_text_blocks():
    # Two blocks means at most one of them mirrors the field; terse cannot tell which
    # without guessing, so it drops neither.
    result = _pair(_SC_PAYLOAD)
    result["content"].append({"type": "text", "text": json.dumps({"page": 2})})
    out = _replace_run(result)
    # BOTH survive — either as two blocks or collapsed into the #116 joined one. Asserting
    # on the decoded payload rather than the block count is what makes this catch a drop
    # of the first block while the second one covers for it.
    decoded = json.dumps([transforms.decompress(b["text"])
                          for b in out["content"] if b.get("type") == "text"])
    assert "page" in decoded and "Berlin" in decoded


def test_structured_replace_preserves_non_text_blocks():
    # Only the mirror goes. An image block is not a duplicate of anything.
    image = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
    result = _pair(_SC_PAYLOAD)
    result["content"].append(image)
    out = _replace_run(result)
    assert out["content"] == [image]


def test_structured_replace_reports_the_dropped_block_as_zero_to_the_ledger():
    # The wire truth, per #133: the block cost nothing because it was not sent. Reporting
    # it as "unchanged" would credit terse with a saving on the typed field alone while
    # still charging for a block nobody received.
    seen = []
    _replace_run(_pair(_SC_PAYLOAD), stats=lambda *a: seen.append(a))
    tool, raw, emitted, passthrough, reason, structured, structured_out = seen[0]
    assert emitted == ""
    assert raw == json.dumps(_SC_PAYLOAD)                  # the sink still sees the original
    assert reason == "mirror_dropped"
    # replace = compress the typed field AND drop the mirror block, so the two ledger sides
    # differ (#141): raw side is the original, emitted side the compressed field.
    assert structured is not None and "__terse_" not in structured        # raw side
    assert structured_out is not None and "__terse_" in structured_out    # emitted side
    assert len(structured_out) < len(structured)


def test_structured_replace_tells_the_audit_trace_it_changed_the_result():
    # The replay trace exists to record the decision. A record saying `changed: false`
    # beside an emitted "" would be it lying about the only thing it is for.
    seen = []
    inter = Interceptor(Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                                           structured="replace")]),
                        audit=lambda rec: seen.append(rec))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    inter.transform_response(json.dumps({"jsonrpc": "2.0", "id": 1,
                                         "result": _pair(_SC_PAYLOAD)}))
    assert seen[0]["changed"] is True
    assert seen[0]["blocks"][0]["emitted"] == ""
    assert seen[0]["blocks"][0]["raw"] == json.dumps(_SC_PAYLOAD)


def test_structured_auto_never_resolves_to_replace():
    # "auto" tops out at "compress". Dropping a block is the first mode that removes
    # information from the wire, and no client is defaulted into it.
    inter = Interceptor(Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))]))
    inter.note_request(_init_req_for("claude-code"))
    inter.note_request('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                       '"params":{"name":"gh.items"}}')
    out = json.loads(inter.transform_response(_structured_result_msg(1, _SC_PAYLOAD)))
    assert len(out["result"]["content"]) == 1              # mirror survives
    assert "__terse_" in json.dumps(out["result"]["structuredContent"])   # but compressed


def test_structured_replace_survives_a_payload_too_deep_to_parse():
    # `_mirror_to_drop` runs OUTSIDE `_compress`'s fail-open wrapper, so an escaping
    # exception takes down the whole tool call. Nesting deep enough raises RecursionError
    # from the C parser (the depth cap in #79 exists for this), and a deep `==` recurses
    # too. Neither may reach the caller.
    # Depth chosen by measurement, not by `recursionlimit * k`: CPython's C scanner
    # swallows 20k nested arrays fine and only blows up near 100k, so a smaller number
    # would make this test pass whether the guard is there or not.
    depth = 100_000
    deep_text = "[" * depth + "]" * depth
    result = {"content": [{"type": "text", "text": deep_text}],
              "structuredContent": {"rows": []}}
    out = _replace_run(result)
    assert out["content"][0]["text"] == deep_text          # passed through, not dropped


def test_structured_replace_resets_the_TEXT_diff_base_too():
    # The client's actual previous result for this tool is an empty content array, so a
    # later CDC text diff whose `=` ops reference the dropped block is unrecoverable — the
    # model never received the text being referenced. All six state maps must reset.
    inter = Interceptor(Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                                           structured="replace")], diff=True))
    def call(mid, result):
        inter.note_request(f'{{"jsonrpc":"2.0","id":{mid},"method":"tools/call",'
                           '"params":{"name":"gh.items"}}')
        return json.loads(inter.transform_response(
            json.dumps({"jsonrpc": "2.0", "id": mid, "result": result})))["result"]

    prose = "a long prose payload that is not JSON at all. " * 200
    call(1, {"content": [{"type": "text", "text": prose}]})       # establishes a text base
    assert inter.last_text.get("gh.items")
    call(2, _pair(_SC_PAYLOAD))                                    # mirror dropped
    assert "gh.items" not in inter.last_text
    assert "gh.items" not in inter.since_text_keyframe
    out = call(3, {"content": [{"type": "text", "text": prose + "and one more line."}]})
    assert "__terse_textdiff__" not in out["content"][0]["text"]   # re-anchored, not diffed


def test_structured_replace_will_not_drop_a_block_that_only_LOOKS_equal():
    # Python `==` treats True == 1 and 1 == 1.0, so value-level equality would delete a
    # block saying `true` in favour of a typed field saying `1`.
    result = {"content": [{"type": "text", "text": '{"ok":true,"n":1.0}'}],
              "structuredContent": {"ok": 1, "n": 1}}
    out = _replace_run(result)
    assert len(out["content"]) == 1
    assert out["content"][0]["text"] == '{"ok":true,"n":1.0}'


# --- capture identity: which server, which result (#148, #152) ---

def test_capture_is_told_the_server_and_the_result_the_block_belonged_to():
    calls: list[dict] = []
    inter = Interceptor(FULL, server_name="runecho",
                        capture=lambda tool, raw, **kw: calls.append(kw))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    inter.transform_response(_result_msg(7, _records_text()))
    assert calls == [{"server": "runecho", "result_id": "0.7"}]


def test_every_block_of_one_result_carries_the_same_result_id():
    # The property the corpus needs: "these blocks arrived together" is READ, not inferred
    # from how close their file writes landed (#148).
    calls: list[dict] = []
    inter = Interceptor(FULL, capture=lambda tool, raw, **kw: calls.append(kw))
    raws = [json.dumps({"id": i, "note": "x"}) for i in range(3)]
    _emit_multi(inter, 4, "gh.api.blocks", raws)
    assert len(calls) >= 1
    assert {c["result_id"] for c in calls} == {"0.4"}


def test_result_ids_are_scoped_to_the_proxy_run_not_bare_jsonrpc_ids():
    # A JSON-RPC id restarts at 1 every session while one corpus dir accumulates many
    # sessions, so two runs' `id: 1` would fuse into one "result" if stored bare.
    from terse.proxy import _build_capture_and_audit

    seen: list[str | None] = []

    def fake(tool, raw, corpus_dir, *, server=None, result_id=None):
        seen.append(result_id)

    import terse.capture as capture_mod
    real, capture_mod.capture_payload = capture_mod.capture_payload, fake
    try:
        cap_a, _ = _build_capture_and_audit("/tmp/nope", None, "aaaa1111")
        cap_b, _ = _build_capture_and_audit("/tmp/nope", None, "bbbb2222")
        cap_a("t", "{}", server=None, result_id=1)
        cap_b("t", "{}", server=None, result_id=1)
        # ...and with no session to scope it, the id is dropped rather than stored ambiguously
        cap_c, _ = _build_capture_and_audit("/tmp/nope", None, None)
        cap_c("t", "{}", server=None, result_id=1)
    finally:
        capture_mod.capture_payload = real

    assert seen == ["aaaa1111:1", "bbbb2222:1", None]
    assert len(set(seen[:2])) == 2


def test_a_reconnect_makes_a_reused_jsonrpc_id_a_different_result():
    # Review finding: the session id is minted once per PROCESS, but a client that
    # re-initializes restarts its ids at 1. Without a per-handshake generation, `sess:1`
    # from before and after the reconnect name the same result and the corpus fuses two
    # unrelated calls — the #148 defect arriving by the one door left open. `note_request`
    # already resets `pending` on this exact event for the mirror-image reason.
    calls: list[dict] = []
    inter = Interceptor(FULL, capture=lambda tool, raw, **kw: calls.append(kw))
    call = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "gh.api.items"}})

    inter.note_request(call)
    inter.transform_response(_result_msg(1, _records_text()))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                   "params": {}}))
    inter.note_request(call)
    inter.transform_response(_result_msg(1, _records_text()))

    assert len({c["result_id"] for c in calls}) == 2


# ---------------------------------------------------------------------------
# #168: the primer is assembled per-policy, so a server documents only the wire
# forms it can actually emit.
# ---------------------------------------------------------------------------

def _pol(tiers=None, *, diff=True, drop=False, rules=None):
    """Default tiers = EVERY valid tier, derived rather than spelled out: the
    "reaches every form" primer test below is only meaningful if adding a tier
    automatically widens this helper, instead of leaving it silently stale."""
    from terse import policy as P
    tiers = P.VALID_TIERS if tiers is None else tiers
    rs = list(rules or [])
    if drop:
        rs.append(P.Rule(tool_glob="*", tiers=tuple(tiers),
                         fields={"$text": {"lossy": "drop-to-retrieve"}}))
    return P.Policy(rules=rs, default_tiers=tuple(tiers), diff=diff)


def test_every_section_selected_reproduces_the_whole_primer_verbatim():
    """A policy reaching every form must be byte-identical to the shipped constant, so the
    decomposition cannot silently reword text that readers (and dropeval) depend on.

    Note `default_policy()` alone does NOT reproduce it: it has no field drop, so the
    dropped-field paragraph is correctly absent."""
    from terse.proxy import TERSE_PRIMER, build_primer
    assert build_primer(_pol(drop=True)) == TERSE_PRIMER


def test_minify_only_policy_emits_no_primer_at_all():
    """`tiers: ["minify"]` puts no terse marker on the wire — minified JSON is just JSON.
    With diffing dead alongside it, there is nothing to explain."""
    from terse.proxy import build_primer
    assert build_primer(_pol(("minify",), diff=False)) == ""


def test_deny_everything_policy_emits_no_primer():
    """A default-deny shape — no tiers, no diff, no drop — emits no primer at all. Before
    #168 it paid a full primer to describe forms it is structurally forbidden from
    producing. This is a property of the POLICY: `secret-broker` under the shipped example
    and live policies is NOT this shape and pays 248."""
    from terse.proxy import build_primer
    assert build_primer(_pol((), diff=False)) == ""


def test_diff_flag_alone_cannot_resurrect_the_primer_when_no_tier_is_reachable():
    """`diff: true` with `tiers: ()` keeps no diff base (the interceptor skips those
    tools), so the diff section must not be emitted on its strength alone."""
    from terse.proxy import build_primer
    assert build_primer(_pol((), diff=True)) == ""


def test_primer_names_every_wire_token_the_table_codec_can_emit():
    """The codec and the paragraph that explains it must stay in step: a wire token the
    tabularizer emits but the primer never names is invisible to the reader, which is a
    fidelity bug the round-trip gate cannot see (it only proves terse can decode its own
    output, not that the model can). Union-schema tabularize shipped `absent_cols`,
    `sentinel_cols` and the `__terse_absent__` cell sentinel; measured on 24
    absent-vs-null questions, three models scored 54.2%/54.2%/79.2% against the
    unextended paragraph and 100% against this one.

    Derived from a real emission rather than a hardcoded list, so the next header key
    added to the codec fails here instead of shipping unexplained."""
    from terse import transforms as T
    from terse.proxy import PRIMER_TABLE

    records = [{"name": f"sym_{i}", "kind": "function",
                "owner": {"login": "eric", "type": "User"}} for i in range(8)]
    for i in (1, 5):
        records[i].pop("kind")                       # holes -> absent_cols
    for i in (0, 2, 4):
        records[i]["note"] = None                    # explicit null -> sentinel_cols
    records[3]["note"] = "x"

    table = T.compress_structure(records)
    assert table.get(T.TABLE_MARKER) == 1
    # Every optional header key the encoder can attach, in one emission — otherwise the
    # loop below silently vouches for a key it never saw.
    assert {"absent_cols", "sentinel_cols", "subcols"} <= set(table)

    for key in table:
        assert key in PRIMER_TABLE, f"table header key {key!r} is unexplained in the primer"
    assert T.ABSENT_MARKER in PRIMER_TABLE


def test_tabularize_without_dictionary_documents_the_table_but_not_the_legend():
    from terse.proxy import PRIMER_DICT, PRIMER_TABLE, build_primer
    out = build_primer(_pol(("minify", "tabularize"), diff=False))
    assert PRIMER_TABLE in out
    assert PRIMER_DICT not in out


def test_diff_section_is_omitted_when_diffing_is_off():
    """The diff paragraph is 190 of the primer's 555 cl100k tokens — the single largest
    section — for a tier measured at a ~0.4% hit rate in production."""
    from terse.proxy import PRIMER_DIFF, build_primer
    assert PRIMER_DIFF not in build_primer(_pol(diff=False))
    assert PRIMER_DIFF in build_primer(_pol(diff=True))


def test_dropped_section_tracks_has_drop():
    from terse.proxy import PRIMER_DROPPED, build_primer
    assert PRIMER_DROPPED not in build_primer(_pol())
    assert PRIMER_DROPPED in build_primer(_pol(drop=True))


def test_a_rule_reaching_a_tier_the_default_does_not_still_documents_it():
    """The primer is injected before any tool is named, so the gate is the UNION over
    every reachable rule — not just the default."""
    from terse import policy as P
    from terse.proxy import PRIMER_TABLE, build_primer
    pol = P.Policy(rules=[P.Rule(tool_glob="wide.*", tiers=("minify", "tabularize"))],
                   default_tiers=("minify",), diff=False)
    assert PRIMER_TABLE in build_primer(pol)


def test_union_primer_documents_a_form_any_single_peer_can_emit():
    from terse.proxy import PRIMER_DROPPED, PRIMER_TABLE, union_primer
    quiet = _pol(("minify",), diff=False)
    dropper = _pol(("minify",), diff=False, drop=True)
    out = union_primer([(quiet, "a"), (dropper, "b")])
    assert PRIMER_DROPPED in out
    assert PRIMER_TABLE not in out          # no peer reaches tabularize
    assert union_primer([(quiet, "a"), (quiet, "b")]) == ""


def test_initialize_injects_nothing_when_the_policy_emits_no_form():
    """End-to-end: a deny-everything server's initialize reply is forwarded untouched."""
    from terse.proxy import Interceptor
    inter = Interceptor(_pol((), diff=False))
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1)))
    assert "instructions" not in out["result"]
    assert out["result"]["serverInfo"]["name"] == "s"      # rest untouched


def test_initialize_primer_injection_is_idempotent_across_varying_assemblies():
    """Idempotency keys on PRIMER_HEAD, not the whole string, because the assembly now
    varies per policy — a full-primer check would re-inject onto a shortened one."""
    from terse.proxy import PRIMER_HEAD, Interceptor
    inter = Interceptor(_pol(("minify", "tabularize"), diff=False))
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1, PRIMER_HEAD + "already here")))
    assert out["result"]["instructions"].count(PRIMER_HEAD) == 1


def test_a_total_cover_rule_makes_default_tiers_unreachable_for_that_server():
    """The live policy sets `default_tiers` to all three tiers, so without terminating the
    walk at a total-cover rule every server would appear to reach every tier and the
    primer could never shrink. `kb.*` covers every kb tool, so kb's ceiling is that rule's
    tiers — not the default's."""
    from terse import policy as P
    from terse.proxy import PRIMER_DICT, PRIMER_TABLE, build_primer
    pol = P.Policy(rules=[P.Rule(tool_glob="kb.*", tiers=("minify", "tabularize"))],
                   default_tiers=("minify", "tabularize", "dictionary"), diff=False)
    assert pol.reachable_tiers("kb") == {"minify", "tabularize"}
    out = build_primer(pol, "kb")
    assert PRIMER_TABLE in out and PRIMER_DICT not in out
    # Unknown server identity cannot exclude anything, so the default still counts.
    assert "dictionary" in pol.reachable_tiers(None)


def test_reachable_tiers_never_undercounts_what_select_returns():
    """THE invariant: `select(tool, server).tiers <= reachable_tiers(server)`, for every
    tool. Under-inclusion means the client gets an envelope with nothing explaining it.

    Pins the bug this gate shipped with in review: an earlier revision skipped rules whose
    literal prefix named a different server, but `select` is candidate-major over
    `_match_candidates`, whose second candidate is the tool's own UNQUALIFIED name — so
    `kb.*` matches `kb.read.search` no matter which server serves it."""
    from terse import policy as P
    pol = P.Policy(
        rules=[P.Rule(tool_glob="gh.*", tiers=("minify", "tabularize", "dictionary")),
               P.Rule(tool_glob="kb.*", tiers=("minify", "tabularize", "dictionary")),
               P.Rule(tool_glob="read.*", tiers=("minify", "tabularize"))],
        default_tiers=("minify",), diff=False)
    servers = ("kb", "knowledge", "runecho", "gh", None)
    tools = ("kb.read.search", "gh.issues", "read.search", "structure", "kb.propose.extend")
    for server in servers:
        reachable = pol.reachable_tiers(server)
        for tool in tools:
            assert set(pol.select(tool, server).tiers) <= reachable, (server, tool)


def test_a_server_serving_another_servers_tool_names_still_gets_its_primer():
    """The live repro from review: config key `knowledge`, tools named `kb.read.*`, under
    a hand-authored `kb.*` rule. The prefix-based exclusion emitted NO primer here while
    `select` returned all three tiers — a `__terse_table__` envelope with no legend."""
    from terse import policy as P
    from terse.proxy import PRIMER_DICT, PRIMER_TABLE, build_primer
    pol = P.Policy(rules=[P.Rule(tool_glob="kb.*",
                                 tiers=("minify", "tabularize", "dictionary"))],
                   default_tiers=("minify",), diff=False)
    assert set(pol.select("kb.read.search", "knowledge").tiers) == {
        "minify", "tabularize", "dictionary"}
    out = build_primer(pol, "knowledge")
    assert PRIMER_TABLE in out and PRIMER_DICT in out


def test_a_covering_rule_still_terminates_the_walk():
    """The one sound narrowing: `kb.*` matches every tool `kb` serves via candidate 0, so
    `select` can never reach a later rule or the default for that server."""
    from terse import policy as P
    pol = P.Policy(rules=[P.Rule(tool_glob="kb.*", tiers=("minify", "tabularize")),
                          P.Rule(tool_glob="*", tiers=("minify", "dictionary"))],
                   default_tiers=("minify", "tabularize", "dictionary"), diff=False)
    assert pol.reachable_tiers("kb") == {"minify", "tabularize"}
    assert "dictionary" in pol.reachable_tiers(None)   # unknown identity narrows nothing


# --- has_drop is a primer gate too, and was the one still ignoring the server (#168) ----


def _drop_rule(glob: str):
    from terse import policy as P
    return P.Rule(tool_glob=glob, tiers=("minify", "tabularize"),
                  fields={"$text.code_blocks": {"lossy": "drop-to-retrieve"}})


def test_a_total_cover_rule_hides_a_later_drop_rule_from_that_server():
    """`has_drop` was the last primer gate scanning every rule unconditionally while the
    other four took a server. Peers commonly share ONE policy file, so it answered "does
    this FILE contain a drop rule" — and a server whose own rule totally covers it, and
    therefore can never reach the drop rule at all, still paid the 64-token dropped-field
    paragraph AND advertised a `terse.retrieve` tool it could never mint a handle for.

    `select` returns the FIRST match, so the walk terminates at a total-cover rule for
    exactly the reason `reachable_tiers` terminates there."""
    from terse import policy as P
    from terse.proxy import PRIMER_DROPPED, build_primer

    pol = P.Policy(
        rules=[P.Rule(tool_glob="runecho.*", tiers=("minify", "tabularize")),
               _drop_rule("*codegraph_explore")],
        default_tiers=("minify", "tabularize"), diff=False)

    assert pol.has_drop("runecho") is False       # terminated before the drop rule
    assert pol.has_drop("codegraph") is True      # reaches it
    assert pol.has_drop() is True                 # no server: scan everything, as before
    assert PRIMER_DROPPED not in build_primer(pol, "runecho")
    assert PRIMER_DROPPED in build_primer(pol, "codegraph")


def test_rule_ORDER_decides_it_not_the_glob_text():
    """The narrowing is termination, not prefix matching. Move the drop rule ahead of the
    cover rule and the same server reaches it — which is what `select` would do."""
    from terse import policy as P

    after = P.Policy(rules=[P.Rule(tool_glob="kb.*", tiers=("minify",)), _drop_rule("gh.*")],
                     default_tiers=("minify",), diff=False)
    before = P.Policy(rules=[_drop_rule("gh.*"), P.Rule(tool_glob="kb.*", tiers=("minify",))],
                      default_tiers=("minify",), diff=False)
    assert after.has_drop("kb") is False
    assert before.has_drop("kb") is True


def test_has_drop_never_undercounts_what_select_would_actually_drop():
    """THE safety invariant, and the one that matters more than the tokens: if `select`
    hands any tool a drop-to-retrieve field for this server, `has_drop(server)` MUST be
    True. Under-inclusion means the proxy drops a field and then does not advertise
    `terse.retrieve` — a handle nobody can redeem, which is worse than a wasted paragraph.

    Mirrors `test_reachable_tiers_never_undercounts_what_select_returns`."""
    from terse import policy as P

    pol = P.Policy(
        rules=[P.Rule(tool_glob="gh.*", tiers=("minify", "tabularize")),
               _drop_rule("*codegraph_explore"),
               P.Rule(tool_glob="kb.*", tiers=("minify", "tabularize")),
               _drop_rule("read.*")],
        default_tiers=("minify",), diff=False)
    servers = ("gh", "kb", "codegraph", "knowledge", "runecho", None)
    tools = ("gh.issues", "kb.read.search", "codegraph_explore", "read.search",
             "structure", "codegraph.codegraph_explore")
    for server in servers:
        drops_somewhere = any(
            isinstance(f, dict) and f.get("lossy") == "drop-to-retrieve"
            for tool in tools for f in pol.select(tool, server).fields.values())
        if drops_somewhere:
            assert pol.has_drop(server) is True, (server, "would drop with no retrieve tool")


def test_a_router_still_documents_a_form_only_one_peer_can_emit():
    """The union errs toward inclusion: one peer that can drop is enough for the whole
    router's primer, because the client sees one server and cannot be told per-peer."""
    from terse import policy as P
    from terse.proxy import PRIMER_DROPPED, union_primer

    pol = P.Policy(
        rules=[P.Rule(tool_glob="runecho.*", tiers=("minify", "tabularize")),
               _drop_rule("*codegraph_explore")],
        default_tiers=("minify", "tabularize"), diff=False)
    assert PRIMER_DROPPED not in union_primer([(pol, "runecho")])
    assert PRIMER_DROPPED in union_primer([(pol, "runecho"), (pol, "codegraph")])


def test_the_covering_rule_can_ITSELF_be_the_drop_rule():
    """Ordering inside the walk: the rule's own fields are checked BEFORE its glob is
    tested for cover, so the single most ordinary per-server policy — one rule, scoped to
    the server, carrying the drop — is not terminated out of its own drop (review of #198).

    Nothing pinned this: the pre-existing union fixture happens to use glob `*`, so
    swapping the two checks failed only incidentally."""
    from terse import policy as P
    from terse.proxy import PRIMER_DROPPED, build_primer

    pol = P.Policy(rules=[_drop_rule("kb.*")], default_tiers=("minify",), diff=False)
    assert pol.has_drop("kb") is True
    assert PRIMER_DROPPED in build_primer(pol, "kb")


# --- _glob_covers_server fnmatch soundness (#199) ---


def test_glob_covers_server_handles_metacharacters_in_server_name():
    """`_glob_covers_server` must agree with `select`, which matches by fnmatch. A server
    name containing an fnmatch metacharacter (`kb[1]`) compares EQUAL to `kb[1].*` but does
    not fnmatch it — `[1]` is a character class, so that rule matches none of the server's
    tools. Cover must not claim a match `select` won't return (#199)."""
    from terse import policy as P
    assert not P.Policy._glob_covers_server("kb[1].*", "kb[1]")
    assert not P.Policy._glob_covers_server("kb[1]*", "kb[1]")


def test_glob_covers_server_accepts_the_three_normal_forms():
    from terse import policy as P
    assert P.Policy._glob_covers_server("*", "kb")
    assert P.Policy._glob_covers_server("kb.*", "kb")
    assert P.Policy._glob_covers_server("kb*", "kb")


def test_glob_covers_server_refuses_a_glob_that_only_matches_some_tools():
    """The other failure direction, and the one a single representative probe walks into:
    `fnmatch(f"{server}.x", glob)` calls `kb.?` and `kb.*x` covering, because both match
    the literal probe `kb.x` — while matching no real tool. Cover means "matches
    `{server}.` + ANY bare name", so a glob that constrains what follows the prefix, or
    anchors its end, covers nothing (#199)."""
    from terse import policy as P
    assert not P.Policy._glob_covers_server("kb.?", "kb")       # one char only
    assert not P.Policy._glob_covers_server("kb.*x", "kb")      # must end in x
    assert not P.Policy._glob_covers_server("kb.read.*", "kb")  # a real sub-namespace rule
    assert not P.Policy._glob_covers_server("kb.x", "kb")       # exact-name rule
    assert not P.Policy._glob_covers_server("gh.*", "kb")       # another server entirely


def test_a_partial_glob_does_not_hide_a_later_drop_rule():
    """The consequence of the case above, end to end. A false cover terminates `has_drop`'s
    walk early, so the proxy reports no-drop for a server that still reaches the drop rule:
    the dropped-field paragraph and the `terse.retrieve` tool both go missing while
    `apply` keeps minting handles, leaving the model an unretrievable
    `__terse_dropped__`. That is #168's failure re-entered through the cover check."""
    from terse import policy as P
    from terse.proxy import PRIMER_DROPPED, build_primer
    pol = P.Policy(
        rules=[P.Rule(tool_glob="kb.?", tiers=("minify",)), _drop_rule("*")],
        default_tiers=("minify",),
        diff=False)
    # `kb.?` matches no real kb tool, so `select` falls through to the drop rule.
    assert "$text.code_blocks" in pol.select("kb.read.search", "kb").fields
    assert pol.has_drop("kb") is True
    assert PRIMER_DROPPED in build_primer(pol, "kb")


# --- _match_candidates PREFIX_SEP insertion (#199) ---


def test_a_server_scoped_rule_already_matches_a_self_prefixed_peer_qualified_tool():
    """#199 proposed synthesizing `gh.gh.api.items` for peer-qualified tools, on the theory
    that `gh.*` had nothing to match. It does: `select` also tries the BARE candidate, and
    `gh.api.items` fnmatches `gh.*`. Pinned because the proposed fix was reverted — if this
    ever fails, the premise becomes real and the insertion is worth revisiting."""
    from terse import policy as P
    pol = P.Policy(rules=[P.Rule(tool_glob="gh.*", tiers=("minify", "tabularize"))],
                   default_tiers=(), diff=False)
    for tool in ("gh__gh.api.items", "gh__gh.rate_limit", "mcp__gh.search"):
        assert pol.select(tool, "gh").tool_glob == "gh.*", tool


def test_match_candidates_never_double_qualifies_a_self_prefixed_tool():
    """No insertion when the bare name already carries the server, with or without a
    PREFIX_SEP: `kb.kb.read.search` / `gh.gh.api.items` are not names any rule is written
    against, and offering them FIRST hands the match to a broader rule (see below)."""
    from terse import policy as P
    assert P.Policy._match_candidates("kb.read.search", "kb")[0] == "kb.read.search"
    assert P.Policy._match_candidates("gh__gh.api.items", "gh")[0] == "gh__gh.api.items"


def test_match_candidates_still_qualifies_when_the_bare_name_is_cross_peer():
    """The insertion that IS load-bearing: tool `mcp__kb.search` on server `gh`, where
    bare=`kb.search` does not start with `gh.` and a `gh.*` rule would otherwise miss."""
    from terse import policy as P
    assert P.Policy._match_candidates("mcp__kb.search", "gh")[0] == "gh.kb.search"


def test_a_broad_rule_must_not_outrank_a_specific_passthrough_rule():
    """Why the #199 insertion was reverted. Candidate order is major over rule order, so a
    synthesized `gh.gh.rate_limit` lets `gh.*` win one candidate BEFORE the specific
    `*.rate_limit` rule can match the bare name — turning an operator's explicit `tiers: []`
    passthrough into a lossy rule, and a `capture: false` exclusion into a capturable one."""
    from terse import policy as P
    passthrough = P.Rule(tool_glob="*.rate_limit", tiers=(), capture=False)
    broad = P.Rule(tool_glob="gh.*", tiers=("minify", "tabularize"),
                   fields={"result[].body": {"lossy": "truncate"}})
    pol = P.Policy(rules=[passthrough, broad], default_tiers=(), diff=False)
    chosen = pol.select("gh__gh.rate_limit", "gh")
    assert chosen.tool_glob == "*.rate_limit"
    assert chosen.tiers == ()          # still lossless
    assert chosen.capture is False     # still off-disk
    assert not chosen.fields           # no truncation inherited from the broad rule


# --- has_drop + server_never_lossy (#199) ---


def test_has_drop_returns_false_for_never_lossy_server():
    """A server that structurally forbids every drop returns False from `has_drop`, saving
    the 64-token dropped-field primer paragraph and the `terse.retrieve` advertisement."""
    from terse import policy as P
    pol = P.Policy(
        rules=[_drop_rule("*codegraph_explore")],
        default_tiers=("minify", "tabularize"),
        never_lossy_servers=frozenset(["vault-mcp"]),
        diff=False)
    assert pol.has_drop("vault-mcp") is False
    assert pol.has_drop("codegraph") is True   # not in the never_lossy set


def test_has_drop_never_lossy_overrides_even_a_covering_drop_rule():
    """`server_never_lossy` is checked BEFORE the rule walk, so even a drop rule whose glob
    totally covers the server yields False when the server is never-lossy."""
    from terse import policy as P
    from terse.proxy import PRIMER_DROPPED, build_primer
    pol = P.Policy(
        rules=[_drop_rule("vault-mcp.*")],
        default_tiers=("minify",),
        never_lossy_servers=frozenset(["vault-mcp"]),
        diff=False)
    assert pol.has_drop("vault-mcp") is False
    assert PRIMER_DROPPED not in build_primer(pol, "vault-mcp")


# --- the RUNTIME half of #168: the gate has to reach tools/list, not just the primer ---


_SHARED_POLICY_RULES = [
    ("runecho.*", None),
    ("*codegraph_explore", {"$text.code_blocks": {"lossy": "drop-to-retrieve"}}),
    ("kb.*", None),
]


def _shared_file_policy():
    """The shape every one of these peers actually ships with: ONE policy file, a drop rule
    for exactly one of them, and a covering rule per peer."""
    from terse import policy as P
    return P.Policy(
        rules=[P.Rule(tool_glob=g, tiers=("minify", "tabularize"), fields=f or {})
               for g, f in _SHARED_POLICY_RULES],
        default_tiers=("minify", "tabularize"), diff=False)


def test_a_server_that_cannot_drop_stops_advertising_terse_retrieve():
    """Round-2 review of #198: every new test sat at the `has_drop`/`build_primer` level, so
    reverting all three runtime call sites left the suite fully green — the behaviour the
    change leads with ("advertised a terse.retrieve it could never mint a handle for") had
    no test at all.

    This is the tools/list gate (`transform_response`), keyed on the Interceptor's own
    server name."""
    pol = _shared_file_policy()
    tl = _tools_list(1, ["runecho.structure"])
    assert "terse.retrieve" not in Interceptor(pol, server_name="runecho").transform_response(tl)
    # ...and the peer that genuinely reaches the drop rule still gets it.
    assert "terse.retrieve" in Interceptor(pol, server_name="codegraph").transform_response(
        _tools_list(1, ["codegraph_explore"]))
    # No server name = no basis to narrow, so the old whole-file answer stands.
    assert "terse.retrieve" in Interceptor(pol).transform_response(tl)


def test_answering_retrieve_is_ungated_even_where_advertising_is_not():
    """Answer >= advertise, matching multiproxy (round-2 review of #198).

    A retrieve call arriving at a server this build believes cannot drop is precisely the
    symptom of `_glob_covers_server`'s unsound cases (#199). Gating the ANSWER there would
    forward it to a downstream that never had the tool — turning one wasted paragraph into
    an unredeemable handle plus a -32601. Answering costs nothing when nothing was dropped:
    a legible miss, and the request never reaches downstream."""
    pol = _shared_file_policy()
    assert pol.has_drop("runecho") is False                 # would not advertise it
    cin = io.StringIO(_retrieve_call(1, "nope") + "\n")
    cout = io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout,
                   server_name="runecho")
    assert rc == 0
    resp = json.loads([ln for ln in cout.getvalue().splitlines() if ln.strip()][0])
    # OUR synthesized miss, and the downstream fake never saw the call — it would have
    # answered with its own records (or a -32601) had the request been forwarded.
    assert resp["id"] == 1 and resp["result"]["isError"] is True
    assert '"status"' not in resp["result"]["content"][0]["text"]
