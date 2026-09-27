#!/usr/bin/env python3
"""Minimal fake MCP stdio server whose sole tool carries `annotations` (#467).

Neither `fake_mcp_server.py`'s tools nor the demo server's tools had `annotations`
before #467, so nothing in the suite could prove the proxy's `tools/list` pass-through
preserves a hint key it has never seen. This fixture exists only to carry one.
"""
import json
import sys

TOOL = {
    "name": "annotated.tool",
    "description": "A tool with annotations, for proving tools/list pass-through.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
}


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
        if mid is None:
            continue  # notification: no reply owed
        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": mid,
                    "result": {"protocolVersion": "2024-11-05",
                               "capabilities": {"tools": {}},
                               "serverInfo": {"name": "fake-annotated", "version": "0"}}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}}
        else:
            resp = {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method!r}"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
