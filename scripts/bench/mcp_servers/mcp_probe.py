#!/usr/bin/env python3
"""Drive a real MCP server through the terse proxy and capture what terse did to it.

Sends `initialize` -> `tools/list` -> each requested `tools/call` **twice** (so the
cross-call diff tier is exercised), writing every raw payload into a capture corpus and a
payload-free stats ledger. Feed the corpus to `terse measure` for per-tool codec numbers
and the ledger to `terse stats` for the diff-reason breakdown.

Usage:
    mcp_probe.py <server_name> <corpus_dir> <stats_log> <calls_json> -- <server argv...>

    calls_json  JSON list of {"name": <tool>, "arguments": {...}}

Env:
    TERSE_BIN         terse executable (default: "terse" from PATH)
    PROBE_DEADLINE    seconds to wait for all responses (default: 300)

Example:
    mcp_probe.py filesystem ./corpus ./ledger.jsonl \
        '[{"name":"directory_tree","arguments":{"path":"/path/to/repo/lib"}}]' \
        -- npx -y @modelcontextprotocol/server-filesystem /path/to/repo

IMPORTANT: stdin is held OPEN until every expected response arrives. Closing it as soon as
the requests are written makes the proxy tear the child down mid-call — fast servers still
answer, but slow ones (a browser launch, an HTTP fetch) silently return nothing, which
reads as "that server is broken" when it is only the harness.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

TERSE_BIN = os.environ.get("TERSE_BIN", "terse")
DEADLINE = float(os.environ.get("PROBE_DEADLINE", "300"))


def main(argv: list[str]) -> int:
    try:
        server_name, corpus, stats_log, calls_json = argv[1:5]
        assert argv[5] == "--"
    except (ValueError, IndexError, AssertionError):
        print(__doc__)
        return 2
    server_argv = argv[6:]
    calls = json.loads(calls_json)

    reqs: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "terse-probe", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    idmap: dict[int, tuple[str, int]] = {}
    mid = 10
    for c in calls:
        for rep in (0, 1):                      # twice: the 2nd call can diff
            reqs.append({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                         "params": {"name": c["name"], "arguments": c.get("arguments", {})}})
            idmap[mid] = (c["name"], rep)
            mid += 1
    expected = set(idmap) | {1, 2}

    proc = subprocess.Popen(
        [TERSE_BIN, "proxy", "--server-name", server_name, "--capture-dir", corpus,
         "--stats-log", stats_log, "--"] + server_argv,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    out: dict[int, dict] = {}
    done = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") in expected:
                out[msg["id"]] = msg
                if expected <= set(out):
                    break
        done.set()

    threading.Thread(target=reader, daemon=True).start()

    assert proc.stdin is not None
    for r in reqs:
        proc.stdin.write(json.dumps(r) + "\n")
    proc.stdin.flush()

    done.wait(DEADLINE)                          # keep stdin open until answers land
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()

    n_tools = len(out.get(2, {}).get("result", {}).get("tools", []))
    print(f"[{server_name}] init={'result' in out.get(1, {})} tools={n_tools}")
    for mid_, (name, rep) in idmap.items():
        msg = out.get(mid_)
        if msg is None:
            print(f"  {name} rep{rep}: NO RESPONSE (raise PROBE_DEADLINE?)")
            continue
        if "error" in msg:
            print(f"  {name} rep{rep}: ERROR {str(msg['error'])[:110]}")
            continue
        if rep != 1:
            continue
        content = msg.get("result", {}).get("content", [])
        text = content[0]["text"] if content and content[0].get("type") == "text" else ""
        is_diff = False
        try:
            env = json.loads(text)
            is_diff = isinstance(env, dict) and env.get("__terse_diff__") == 1
        except ValueError:
            pass
        print(f"  {name:24} rep1 blocks={len(content)} chars={len(text)}"
              f"{' DIFF' if is_diff else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
