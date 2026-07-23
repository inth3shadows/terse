#!/usr/bin/env python3
"""A minimal stdio MCP server that emits `structuredContent` alongside the text mirror.

Exists so #128 can be measured against a fixture that cannot drift. The obvious
alternative — `@modelcontextprotocol/server-everything`'s `get-structured-content` — is
a real third-party server and worth cross-checking against, but it returns ONE flat
weather object, which is exactly the case issue #128 already verified benign (only
`minify` fires, so the text block stays literal JSON and matches `structuredContent`
byte-for-byte). The interesting case needs a record array, where `tabularize` fires and
the two fields stop agreeing in shape.

Two tools, both returning a spec-compliant pair (MCP 2025-06-18, server/tools: a tool
returning structured content SHOULD also return the serialized JSON in a TextContent
block for backwards compatibility):

  weather  -> one flat object      (benign: minify only, shapes stay identical)
  records  -> N records, N=30      (tabularize fires: shapes diverge, duplicate is big)

No dependencies, no network, no clock, no randomness — same request always produces the
same bytes, so a token count measured today is comparable to one measured next month.
"""
from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2025-06-18"

WEATHER = {"temperature": 22.5, "conditions": "Partly cloudy", "humidity": 65}

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "description": "Temperature in celsius"},
        "conditions": {"type": "string", "description": "Weather conditions description"},
        "humidity": {"type": "number", "description": "Relative humidity percentage"},
    },
    "required": ["temperature", "conditions", "humidity"],
}

RECORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "status": {"type": "string"},
                    "city": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["id", "status", "city", "url"],
            },
        }
    },
    "required": ["rows"],
}

# Deliberately the shape terse compresses best — uniform records with repeated values —
# so the measurement reflects the case where the untouched duplicate costs the most.
_CITIES = ["New York", "London", "Tokyo", "Berlin", "Sydney"]
RECORDS = {"rows": [{"id": i,
                     "status": "active" if i % 3 else "pending",
                     "city": _CITIES[i % len(_CITIES)],
                     "url": "https://records.example/api/items"}
                    for i in range(30)]}

TOOLS = [
    {"name": "weather",
     "description": "One flat structured object plus its text mirror (the benign case).",
     "inputSchema": {"type": "object", "properties": {}},
     "outputSchema": WEATHER_SCHEMA},
    {"name": "records",
     "description": "Thirty uniform records plus their text mirror (the tabularize case).",
     "inputSchema": {"type": "object", "properties": {}},
     "outputSchema": RECORDS_SCHEMA},
]

PAYLOADS = {"weather": WEATHER, "records": RECORDS}


def _result(payload: dict) -> dict:
    """The spec's backwards-compatibility pair: the serialized JSON in a text block AND
    the typed field. Kept in ONE place so the two provably start out identical — any
    divergence the capture observes is then terse's doing, not the fixture's."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}],
            "structuredContent": payload}


def _reply(mid, result) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result},
                                separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _error(mid, code: int, message: str) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                 "error": {"code": code, "message": message}},
                                separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue                                   # a notification; nothing to answer
        if method == "initialize":
            _reply(mid, {"protocolVersion": PROTOCOL_VERSION,
                         "capabilities": {"tools": {}},
                         "serverInfo": {"name": "structured-fixture", "version": "1.0.0"}})
        elif method == "tools/list":
            _reply(mid, {"tools": TOOLS})
        elif method == "tools/call":
            name = msg.get("params", {}).get("name")
            payload = PAYLOADS.get(name)
            if payload is None:
                _error(mid, -32602, f"unknown tool: {name!r}")
            else:
                _reply(mid, _result(payload))
        else:
            _error(mid, -32601, f"method not found: {method!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
