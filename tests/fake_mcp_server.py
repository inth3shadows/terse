#!/usr/bin/env python3
"""Minimal fake MCP stdio server for proxy integration tests.

Newline-delimited JSON-RPC. Responds to `initialize` and `tools/call`; ignores
notifications; exits on stdin EOF. The tools/call result is a pretty-printed record
array so minify + tabularize + dictionary all have something to fold.
"""
import json
import sys

RECORDS = [{"id": i, "status": "active", "url": "https://x.example/api/items"} for i in range(20)]


def main() -> None:
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
            text = json.dumps({"result": RECORDS}, indent=2)
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False}}
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
