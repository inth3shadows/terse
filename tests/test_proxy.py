"""Proxy: pure Interceptor logic + an end-to-end run against a fake MCP server."""

from __future__ import annotations

import io
import json
import pathlib
import sys

from terse import transforms
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


def test_diff_off_by_default_sends_full_both_times():
    inter = Interceptor(FULL)  # diff flag defaults off
    prev, curr = _records(40), _records(40, change=5)
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
    line = [l for l in cout.getvalue().splitlines() if l.strip()][0]
    text = json.loads(line)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == {"result": [
        {"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]}


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
    by_id = {json.loads(l)["id"]: json.loads(l) for l in cout.getvalue().splitlines() if l.strip()}

    # initialize: serverInfo intact, and the format primer was injected end-to-end
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake"
    assert "__terse_table__" in by_id[1]["result"]["instructions"]
    # tools/call result compressed, smaller, and round-trips to the exact original
    text = by_id[2]["result"]["content"][0]["text"]
    expected = {"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                           for i in range(20)]}
    assert transforms.decompress(text) == expected
    assert len(text) < len(_records_text())
