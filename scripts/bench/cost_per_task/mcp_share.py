"""Approximate MCP token share for a Claude Code transcript.

terse only touches MCP tool results, so a task that never calls an MCP tool
measures terse's effect at ~zero by construction -- the paper's "limited
addressable fraction" in miniature (see the harness plan). Anthropic's
`usage` block is per-TURN, not per-tool-call, so there is no exact per-call
token count to read off it anywhere in the transcript. This is therefore a
documented APPROXIMATION, not a billed number: the fraction of transcript
message content (by character count) that came from an MCP tool's
`tool_result`, against all message content in the transcript. Character
count rather than a real tokenizer, because the goal is a same-transcript
RATIO -- whatever bias a char-count proxy has over a real tokenizer applies
about equally to the numerator and the denominator.
"""
from __future__ import annotations

import json
from pathlib import Path


def _content_len(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_len(b) for b in content)
    if isinstance(content, dict):
        return sum(_content_len(v) for v in content.values()
                    if isinstance(v, (str, list, dict)))
    return 0


def mcp_share(transcript_path: Path) -> float:
    """Fraction (0.0-1.0) of `transcript_path`'s message content that is MCP
    tool_result content. Returns 0.0 for a transcript with no content at all
    (an empty/failed run is not addressable by definition, not an error in
    this function)."""
    mcp_use_ids: set[str] = set()
    mcp_chars = 0
    total_chars = 0
    with Path(transcript_path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if content is None:
                continue
            total_chars += _content_len(content)
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use" and isinstance(block.get("name"), str):
                    if block["name"].startswith("mcp__"):
                        bid = block.get("id")
                        if bid:
                            mcp_use_ids.add(bid)
                elif btype == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid in mcp_use_ids:
                        mcp_chars += _content_len(block.get("content"))
    return (mcp_chars / total_chars) if total_chars else 0.0
