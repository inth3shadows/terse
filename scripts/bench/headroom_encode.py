#!/usr/bin/env python3
"""Run headroom over tool results, one per input line. For `replay_competitors.py`.

Runs under headroom's own interpreter (it is not a terse dependency), with no network and
no state on disk. headroom's default retrieve cache is a SQLite file holding the original
of everything it drops, so the script refuses to start unless HEADROOM_STATELESS is set
and loopback is the only network interface:

    HEADROOM_STATELESS=1 HEADROOM_WORKSPACE_DIR=<tmp> \\
        unshare -rn <venv>/bin/python headroom_encode.py < in.jsonl > out.jsonl

In:  {"id": ..., "text": <the raw tool result>}
Out: {"id": ..., "default": <text>, "agent": <text>, "verified": <text>,
      "dropped": bool, "transforms": [...]}  or  {"id": ..., "error": "..."}

  default   `headroom.compress` as shipped. May drop rows or strings behind a marker.
  agent     `mode="agent"`, headroom's own lossless regime for agents.
  verified  `verify_lossless=True`: anything headroom cannot prove round-trips is put
            back, which includes most dict-wrapped lists whether or not anything was lost.
  dropped   the default output carries one of headroom's removal markers.

`protect_recent=0` because the result under test is the last message, and headroom
otherwise leaves recent messages alone.
"""
from __future__ import annotations

import json
import os
import socket
import sys

MODEL = "claude-sonnet-4-5-20250929"
# What headroom leaves where it removed content (0.40.0). Its own
# `contains_removal_marker` misses the log-truncation form, so the strings are listed.
DROP_MARKERS = ("<<ccr:", "_ccr_dropped", "Retrieve more: hash=", "lines omitted")


def has_drop_marker(out: str, raw: str) -> bool:
    return any(mark in out and mark not in raw for mark in DROP_MARKERS)


def result_text(messages: list, sent: list) -> str:
    """The tool result's text out of headroom's messages. Raises if headroom did anything
    but rewrite that one block in place: content moved elsewhere would read as a saving."""
    if len(messages) != 3 or messages[:2] != sent[:2]:
        raise ValueError("headroom changed the message list")
    content = messages[-1].get("content")
    if not (isinstance(content, list) and len(content) == 1
            and content[0].get("type") == "tool_result"
            and content[0].get("tool_use_id") == "t1"):
        raise ValueError("headroom changed the tool_result block")
    block = content[0].get("content", "")
    if isinstance(block, list):
        block = "".join(b.get("text", "") for b in block if isinstance(b, dict))
    if not isinstance(block, str):
        block = json.dumps(block)
    if not block.strip() and sent[-1]["content"][0]["content"].strip():
        raise ValueError("headroom emptied the tool result")
    return block


def encode(compress, text: str, **kw) -> tuple[str, list]:
    messages = [
        {"role": "user", "content": "run the tool"},
        {"role": "assistant",
         "content": [{"type": "tool_use", "id": "t1", "name": "tool", "input": {}}]},
        {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}]},
    ]
    res = compress(messages, model=MODEL, protect_recent=0, **kw)
    # `compress` catches its own exceptions and hands the input back with no token count.
    if not res.tokens_before:
        raise RuntimeError("headroom returned the input after an internal error")
    return result_text(res.messages, messages), [str(t) for t in res.transforms_applied or []]


def main() -> int:
    if os.environ.get("HEADROOM_STATELESS", "").lower() not in ("1", "true", "yes", "on"):
        print("refusing to run: set HEADROOM_STATELESS=1 (headroom's retrieve cache writes "
              "originals to disk)", file=sys.stderr)
        return 2
    if [name for _, name in socket.if_nameindex()] != ["lo"]:
        print("refusing to run: a network interface other than lo exists (use unshare -rn)",
              file=sys.stderr)
        return 2
    import headroom

    for line in sys.stdin:
        rec = json.loads(line)
        try:
            default, transforms = encode(headroom.compress, rec["text"])
            agent, _ = encode(headroom.compress, rec["text"], mode="agent")
            verified, _ = encode(headroom.compress, rec["text"], verify_lossless=True)
            out = {"id": rec["id"], "default": default, "agent": agent, "verified": verified,
                   "dropped": has_drop_marker(default, rec["text"]), "transforms": transforms}
        except Exception as exc:  # one bad payload must not end the run
            out = {"id": rec["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
