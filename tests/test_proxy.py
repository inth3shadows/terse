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


def _cost_lt(a, b):
    from terse.proxy import _cost
    return _cost(a) < _cost(b)


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

    # initialize forwarded untouched
    assert by_id[1]["result"]["serverInfo"]["name"] == "fake"
    # tools/call result compressed, smaller, and round-trips to the exact original
    text = by_id[2]["result"]["content"][0]["text"]
    expected = {"result": [{"id": i, "status": "active", "url": "https://x.example/api/items"}
                           for i in range(20)]}
    assert transforms.decompress(text) == expected
    assert len(text) < len(_records_text())
