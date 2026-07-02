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


def test_text_diff_off_by_default_sends_raw_both_times():
    inter = Interceptor(FULL)  # diff flag defaults off
    prev, curr = _log_text(80), _log_text(80, changed_line=40)
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
    line = [ln for ln in cout.getvalue().splitlines() if ln.strip()][0]
    text = json.loads(line)["result"]["content"][0]["text"]
    assert transforms.decompress(text) == {"result": [
        {"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]}


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
