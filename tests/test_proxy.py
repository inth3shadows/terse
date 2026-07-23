"""Proxy: pure Interceptor logic + an end-to-end run against a fake MCP server."""

from __future__ import annotations

import io
import json
import pathlib
import sys

from terse import text_diff, transforms
from terse.policy import Policy, Rule
from terse.proxy import Interceptor, run_proxy

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
    inter = Interceptor(FULL)
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
    inter = Interceptor(DIFF)
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
    a = Interceptor(DIFF)
    _emit(a, 1, "gh.api.items", prev)
    assert transforms.DIFF_MARKER in _emit(a, 2, "gh.api.items", curr)   # A diffs
    b = Interceptor(DIFF)                                                # new session
    assert transforms.DIFF_MARKER not in _emit(b, 1, "gh.api.items", curr)  # no shared base


def test_reconnect_clears_diff_base_and_args_so_next_result_re_anchors():
    # An `initialize` means the client rebuilt its context window, so no prior result a diff
    # could reference survives — every base (and its args attribution) must drop.
    inter = Interceptor(DIFF)
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

    def stats(tool, raw, emitted, passthrough, diff_reason=None):
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


def test_diff_on_by_default_and_policy_false_disables():
    # Since #75 completed the validation program, Policy.diff defaults ON: a plain
    # policy diffs the second same-tool result with no flag at all …
    inter = Interceptor(FULL)
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
    inter = Interceptor(pol)
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
    inter = Interceptor(DIFF)
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
    inter = Interceptor(DIFF)
    prev, curr = _log_text(200), _log_text(200, changed_line=100)
    raw_first = _emit_text(inter, 1, "fs.read", prev)
    diff_text = _emit_text(inter, 2, "fs.read", curr)
    env = json.loads(diff_text)
    assert env.get(text_diff.DIFF_MARKER) == 1
    assert text_diff.text_diff_decode(prev, env) == curr
    assert _cost_lt(diff_text, curr)
    assert raw_first == prev  # sanity: first call was untouched


def test_text_diff_on_by_default_and_policy_false_disables():
    # Same default flip for the CDC text path: on by default, off via "diff": false.
    inter = Interceptor(FULL)
    prev, curr = _log_text(80), _log_text(80, changed_line=40)
    _emit_text(inter, 1, "fs.read", prev)
    t2 = _emit_text(inter, 2, "fs.read", curr)
    env = json.loads(t2)
    assert env.get(text_diff.DIFF_MARKER) == 1
    assert text_diff.text_diff_decode(prev, env) == curr
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))], diff=False)
    inter = Interceptor(off)
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
    inter = Interceptor(FULL)
    inter.note_request(_init_req(1))
    out = json.loads(inter.transform_response(_init_resp(1)))
    instr = out["result"]["instructions"]
    assert "__terse_table__" in instr and "__terse_diff__" in instr   # covers all forms
    assert "__terse_textdiff__" in instr
    assert out["result"]["serverInfo"]["name"] == "s"                 # rest untouched


def test_initialize_preserves_existing_instructions():
    inter = Interceptor(FULL)
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


# --- raw-payload capture tee (#32) ---

def test_capture_tees_raw_text_before_compression():
    captured: list[tuple[str, str]] = []
    inter = Interceptor(FULL, capture=lambda tool, raw: captured.append((tool, raw)))
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
    inter = Interceptor(FULL, capture=lambda tool, raw: captured.append((tool, raw)),
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
    inter = Interceptor(FULL, capture=boom)
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
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   capture_dir=str(blocker))
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
    rc = run_proxy([sys.executable, str(FAKE)], FULL, stdin=cin, stdout=cout,
                   capture_dir=str(blocker))
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

    # initialize: serverInfo intact, and the format primer was injected end-to-end
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake"
    assert "__terse_table__" in by_id[1]["result"]["instructions"]
    # tools/call result compressed, smaller, and round-trips to the exact original
    text = by_id[2]["result"]["content"][0]["text"]
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
    rc = run_proxy([sys.executable, str(FAKE)], pol, stdin=cin, stdout=cout)
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
    inter = Interceptor(FULL, audit=boom)
    _note_call(inter, 9, "gh.api.items")
    out = inter.transform_response(_result_msg(9, _records_text()))
    # Forwarding is unaffected by the audit explosion: still the compressed, lossless result.
    text = json.loads(out)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == json.loads(_records_text())


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
    inter = Interceptor(CAPTURE_GATED, capture=lambda tool, raw: captured.append((tool, raw)))
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
    tool, raw, emitted, passthrough, reason = seen[0]
    assert tool == "secret.reveal" and passthrough is True and reason == "passthrough"
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
    inter = Interceptor(FULL, stats=lambda *a: seen.append(a))
    _note_call(inter, 1, "gh.api.items")
    out = inter.transform_response(_result_msg(1, _records_text()))
    assert len(seen) == 1
    tool, raw, emitted, passthrough, reason = seen[0]
    assert tool == "gh.api.items" and passthrough is False and reason == "no_prior"
    assert raw == _records_text()                       # true pre-transform snapshot
    assert emitted == json.loads(out)["result"]["content"][0]["text"]
    assert classify_decision(raw, emitted, passthrough) == "compressed"


def test_stats_callback_works_without_audit_and_labels_a_diff():
    # stats alone must trigger the raw-text snapshot (it used to be audit-gated), and a
    # second same-tool call that ships a cross-call delta classifies as "diff".
    from terse.stats import classify_decision
    seen = []
    inter = Interceptor(FULL, stats=lambda *a: seen.append(a))
    first = {"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                        for i in range(20)]}
    second = json.loads(json.dumps(first))
    second["result"][0]["status"] = "closed"
    _note_call(inter, 1, "gh.api.items")
    inter.transform_response(_result_msg(1, json.dumps(first)))
    _note_call(inter, 2, "gh.api.items")
    inter.transform_response(_result_msg(2, json.dumps(second)))
    assert [classify_decision(r, e, p) for (_t, r, e, p, _rsn) in seen] == ["compressed", "diff"]
    assert [s[4] for s in seen] == ["no_prior", "emitted"]   # the diff_reason datum agrees


def test_stats_passthrough_tool_is_labeled_passthrough():
    from terse.stats import classify_decision
    seen = []
    inter = Interceptor(Policy(rules=[Rule("gh.*", ())]), stats=lambda *a: seen.append(a))
    _note_call(inter, 5, "gh.api.items")
    inter.transform_response(_result_msg(5, _records_text()))
    (tool, raw, emitted, passthrough, reason), = seen
    assert passthrough is True and raw == emitted and reason == "passthrough"
    assert classify_decision(raw, emitted, passthrough) == "passthrough"


def test_stats_failure_never_breaks_forwarding():
    def boom(*_a):
        raise RuntimeError("disk full")
    inter = Interceptor(FULL, stats=boom)
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
    # exactly the one tools/call result was recorded (initialize is not a tool call)
    assert len(recs) == 1
    rec = recs[0]
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

    def slow_capture(tool, raw):
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
    inter = Interceptor(TEXT_DROP)
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
    inter = Interceptor(TEXT_DROP)
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
    inter = Interceptor(DIFF)
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
    inter = Interceptor(DIFF, capture=lambda tool, text: captured.append(text))
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
    inter = Interceptor(nodiff, stats=stats)
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
    inter = Interceptor(FULL, stats=stats)
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
    inter = Interceptor(FULL)
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
    # the terse primer injected.
    inter = Interceptor(FULL)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "initialize"}))
    server_req = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "roots/list"})
    assert inter.transform_response(server_req) == server_req
    reply = json.dumps({"jsonrpc": "2.0", "id": 5,
                        "result": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    out = json.loads(inter.transform_response(reply))
    assert "terse" in (out["result"].get("instructions") or "").lower()   # primer survived
