#!/usr/bin/env python3
"""Minimal fake MCP stdio server for proxy integration tests.

Newline-delimited JSON-RPC. Responds to `initialize` and `tools/call`; ignores
notifications; exits on stdin EOF. The tools/call result is a pretty-printed record
array so minify + tabularize + dictionary all have something to fold, EXCEPT for the
`fs.read` tool, which returns plain (non-JSON) log text -- so a real proxy run can
exercise the Tier 0.7 text diff (#25) against a live subprocess, not just the pure
Interceptor unit tests.
"""
import json
import sys

RECORDS = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]

_fs_read_calls = 0


def _log_text(n, changed_line=None):
    lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}" for i in range(n)]
    if changed_line is not None:
        lines[changed_line] = "[ERROR] worker crashed: connection reset"
    return "\n".join(lines)


def main() -> None:
    global _fs_read_calls
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "serverInfo": {"name": "fake", "version": "0"}}}
        elif method == "tools/call":
            name = (msg.get("params") or {}).get("name")
            if name == "fs.read":
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
                    "result": {"tools": [{"name": "gh.api.items"}, {"name": "fs.read"}]}}
        elif method and method.startswith("notifications/"):
            continue  # notifications get no response
        elif mid is not None:
            resp = {"jsonrpc": "2.0", "id": mid, "result": {}}
        else:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
