#!/usr/bin/env python3
"""A tiny stdio MCP server, stdlib-only, that exists so terse has something to proxy.

terse is a PROXY: it has no tools of its own, it rewrites another server's results in
place. A container running `terse proxy` over nothing answers `tools/list` with nothing,
which is indistinguishable from a broken server to a registry inspector. This module is
the default downstream in `Dockerfile` so the image is a working, self-contained
demonstration: call `demo_orders` and the reply you receive is the terse-compressed form
of the JSON below, not the JSON itself.

Deliberately NOT `tests/fake_mcp_server.py`: that fixture omits `description` and
`inputSchema`, which a real client (and Glama's inspector) needs to render a tool at all.

The payload shape is chosen to exercise the codec end to end:
  - `demo_orders` -> a record array, so tabularize folds it to one header + N rows, and
    the repeated `status`/`region` values dictionary-code to `~N` aliases.
  - `demo_logs`   -> plain non-JSON text, so the text tier (and cross-call text diff)
    has something to work on rather than every tool being record-shaped.

Newline-delimited JSON-RPC on stdin/stdout; anything diagnostic goes to stderr, because
stdout is the wire. Exits on stdin EOF.
"""

from __future__ import annotations

import json
import sys

_REGIONS = ("us-east-1", "us-west-2", "eu-central-1")
_STATUSES = ("fulfilled", "pending", "backordered")

# 40 records: enough repetition that the dictionary tier pays (a `~0` alias costs about
# four tokens quoted, so aliasing only wins on genuinely repeated values).
ORDERS = [
    {
        "order_id": 100_000 + i,
        "customer": f"customer-{i % 12:02d}",
        "status": _STATUSES[i % 3],
        "region": _REGIONS[i % 3],
        "total_cents": 1_995 + (i * 137) % 48_000,
        "currency": "USD",
        "placed_at": f"2026-07-{(i % 28) + 1:02d}T09:{i % 60:02d}:00Z",
    }
    for i in range(40)
]

TOOLS = [
    {
        "name": "demo_orders",
        "description": (
            "Return a synthetic order book as a JSON record array. Nothing is fetched "
            "and nothing is stored — the point is the shape, so you can see what terse "
            "does to a record-shaped tool result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": f"How many orders to return (1-{len(ORDERS)}).",
                    "minimum": 1,
                    "maximum": len(ORDERS),
                    "default": len(ORDERS),
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "demo_logs",
        "description": (
            "Return a synthetic plain-text log tail. Non-JSON on purpose: it is what "
            "terse's text tier sees, as opposed to the record array demo_orders returns."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "integer",
                    "description": "How many log lines to return (1-500).",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 120,
                }
            },
            "additionalProperties": False,
        },
    },
]


def _clamp(value: object, low: int, high: int, default: int) -> int:
    """Coerce an untrusted `limit`/`lines` argument into range.

    A demo server still gets called with junk — a string, a null, a negative — and a
    traceback on stdout would corrupt the JSON-RPC stream, not just fail the call.
    """
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _log_text(n: int) -> str:
    return "\n".join(
        f"[2026-07-30T12:{i % 60:02d}:00Z] worker={i % 4} queue_depth={i % 7} ok"
        for i in range(n)
    )


def _result(mid: object, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": mid,
        "result": {"content": [{"type": "text", "text": text}], "isError": False},
    }


def _dispatch(msg: dict) -> dict | None:
    """One request in, one response out; None for anything that takes no reply."""
    mid, method = msg.get("id"), msg.get("method")
    params = msg.get("params") or {}
    args = params.get("arguments") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                # Echo the client's protocol version when it sends one: a hardcoded
                # version fails the handshake against any client on a different one.
                "protocolVersion": params.get("protocolVersion") or "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "terse-demo", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        if name == "demo_orders":
            limit = _clamp(args.get("limit"), 1, len(ORDERS), len(ORDERS))
            # Pretty-printed on purpose: whitespace is the first thing terse folds, and
            # a real server's JSON dump usually is indented.
            return _result(mid, json.dumps({"orders": ORDERS[:limit]}, indent=2))
        if name == "demo_logs":
            return _result(mid, _log_text(_clamp(args.get("lines"), 1, 500, 120)))
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": f"unknown tool: {name!r}"}],
                "isError": True,
            },
        }
    if method and method.startswith("notifications/"):
        return None  # notifications are one-way by definition
    if mid is None:
        return None  # an unknown NOTIFICATION, still no reply
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method!r}"}}


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn line is not worth killing the stream over
        resp = _dispatch(msg)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
