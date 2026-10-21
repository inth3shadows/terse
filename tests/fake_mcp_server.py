#!/usr/bin/env python3
"""Minimal fake MCP stdio server for proxy integration tests.

Newline-delimited JSON-RPC. Responds to `initialize` and `tools/call`; ignores
notifications; exits on stdin EOF. The tools/call result is a pretty-printed record
array so minify + tabularize + dictionary all have something to fold, EXCEPT for the
`fs.read` tool, which returns plain (non-JSON) log text -- so a real proxy run can
exercise the Tier 0.7 text diff (#25) against a live subprocess, not just the pure
Interceptor unit tests.

Three optional env knobs, for the router's snapshot tests (#270): FAKE_INIT_DELAY (seconds
to sleep before answering `initialize` -- a slow-starting peer), FAKE_TOOLS (comma-separated
names `tools/list` reports instead of the default two), and FAKE_DIE_IF (a path: if it exists
at launch, exit at once -- a peer that fails to start on THIS run only, while its config
entry, and so the snapshot fingerprint, stays the same across runs).

Wire-log tools (tests/test_wire_log.py), selected by tools/call name: `fake.big` (a text
result of `arguments.n` chars), `fake.error` (a JSON-RPC error reply), `fake.echo` (echoes
`arguments` back in the result), `fake.noise` (writes a blank line, a whitespace line and a
non-JSON line before its reply), `fake.crash` (exits 3 without replying), `fake.ask` (sends
the client a server-initiated request -- `arguments.method`, default elicitation/create --
waits for the client's reply and returns it as the result). A tools/call carrying
`params._meta.progressToken` first gets a notifications/progress echoing its arguments. A
JSON-RPC batch array is answered with an array. FAKE_SECRET (a string) makes this server
leak it everywhere it can: every tools/call result and a notifications/message before it,
resources/read|list, resources/templates/list, prompts/list, and any other request's result.
"""
import json
import os
import sys
import time

RECORDS = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]

_fs_read_calls = 0


def _log_text(n, changed_line=None):
    lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}" for i in range(n)]
    if changed_line is not None:
        lines[changed_line] = "[ERROR] worker crashed: connection reset"
    return "\n".join(lines)


def _write(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


_asks = 0


def _ask(method):
    """Send a server-initiated request and block until the client's reply to it."""
    global _asks
    _asks += 1
    sid = f"srv-{_asks}"
    _write({"jsonrpc": "2.0", "id": sid, "method": method,
            "params": {"message": "need input"}})
    for raw in sys.stdin:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("id") == sid and "method" not in msg:
            return msg
    return {}


def _secret_result(method, params, secret):
    if method == "resources/read":
        return {"contents": [{"uri": params.get("uri"), "text": secret}]}
    if method == "resources/list":
        return {"resources": [{"uri": f"secret://{secret}", "name": secret}]}
    if method == "resources/templates/list":
        return {"resourceTemplates": [{"uriTemplate": f"t://{secret}/{{x}}", "name": secret}]}
    if method == "prompts/list":
        return {"prompts": [{"name": f"p-{secret}"}]}
    return {"echo": secret}


def main() -> None:
    global _fs_read_calls
    die_if = os.environ.get("FAKE_DIE_IF")
    if die_if and os.path.exists(die_if):
        sys.exit(1)
    init_delay = float(os.environ.get("FAKE_INIT_DELAY") or 0)
    tools = [{"name": n} for n in
             (os.environ.get("FAKE_TOOLS") or "gh.api.items,fs.read").split(",")]
    secret = os.environ.get("FAKE_SECRET")
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, list):
            out = [r for m in msg if isinstance(m, dict) and m.get("id") is not None
                   for r in [{"jsonrpc": "2.0", "id": m["id"], "result": {
                       "content": [{"type": "text", "text": json.dumps(
                           (m.get("params") or {}).get("arguments") or {})}]}}]]
            if out:
                _write(out)
            continue
        if not isinstance(msg, dict):
            continue
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            if init_delay:
                time.sleep(init_delay)
            # "stdio_peer" is a harmless marker key (no test asserts capabilities is
            # empty) that lets a multiproxy test prove THIS fake was actually reached
            # by a broadcast, by checking for it in the merged capabilities union.
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"protocolVersion": "2024-11-05",
                               "capabilities": {"stdio_peer": True},
                               "serverInfo": {"name": "fake", "version": "0"}}}
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "fake.crash":
                sys.exit(3)
            token = (params.get("_meta") or {}).get("progressToken")
            if token is not None:
                _write({"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": token, "progress": 1,
                                   "message": json.dumps(args)}})
            if secret:
                _write({"jsonrpc": "2.0", "method": "notifications/message",
                        "params": {"level": "info", "data": secret}})
            if name == "fake.ask":
                reply = _ask(args.get("method", "elicitation/create"))
                _write({"jsonrpc": "2.0", "id": mid, "result": {"content": [
                    {"type": "text", "text": json.dumps(reply.get("result"))}]}})
                continue
            if name == "fake.noise":
                sys.stdout.write("\n   \nnot json {\n")
                sys.stdout.flush()
            if name == "fake.error":
                resp = {"jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32050, "message": "fake failure"}}
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                continue
            if secret:
                text = secret + json.dumps(args)
            elif name == "fake.big":
                text = "x" * int(args.get("n", 0))
            elif name == "fake.echo":
                text = json.dumps(args)
            elif name == "fs.read":
                _fs_read_calls += 1
                # 2nd+ read of the "same file" has one line changed -- the realistic
                # debug-loop shape the CDC text diff targets.
                text = _log_text(200, changed_line=100 if _fs_read_calls >= 2 else None)
            else:
                text = json.dumps({"result": RECORDS}, indent=2)
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"tools": tools}}
        elif method and method.startswith("notifications/"):
            continue  # notifications get no response
        elif secret and mid is not None:
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": _secret_result(method, msg.get("params") or {}, secret)}
        elif mid is not None:
            resp = {"jsonrpc": "2.0", "id": mid, "result": {}}
        else:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
