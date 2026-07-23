"""READ-ONLY mitmproxy addon: what does the MCP client actually put in the model's context?

Answers the one question issue #128 turns on. Every `tool_result` block in an outbound
`/v1/messages` request is a verbatim record of what the client decided the model should
see; comparing it against what the MCP server sent tells us whether the untouched
`structuredContent` duplicate is a real token cost or a phantom one.

SECURITY, by construction rather than by convention:
  * Never mutates a flow. There is no code path here that writes to `flow.request` or
    `flow.response` — this cannot alter what the client or the API receives.
  * Never reads request headers. The session's OAuth bearer lives there; this addon has
    no reason to see it and so does not touch `flow.request.headers` at all.
  * Extracts ONLY `tool_result` blocks, not whole conversations, so an unrelated prompt
    in the same session is not swept into the artifact.
  * Writes 0600 to a path given by CAP_OUT (the scratchpad — never the repo).

Output: JSONL, one object per captured tool_result block.
"""
from __future__ import annotations

import json
import os
import re

OUT = os.environ.get("CAP_OUT", "/tmp/structured-context-capture.jsonl")

# Only count tokens if tiktoken is importable; the capture itself must not depend on it,
# so a missing encoder degrades to byte counts rather than losing the whole run.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                                       # noqa: BLE001 — optional
    _ENC = None


def _tokens(text: str) -> int | None:
    return len(_ENC.encode(text)) if _ENC is not None else None


def _write(record: dict) -> None:
    # 0600 on create: the artifact quotes tool output, which on a real server could be
    # anything. Same posture as terse's own capture sinks (decision #204).
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def _text_of(block: dict) -> str:
    """A tool_result's `content` is either a string or a list of typed parts. Normalize
    to the concatenated text the model reads, so the token count is the real one."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _classify(text: str) -> dict:
    """What survived into the model's context, judged from the text itself.

    `structured_marker` keys off a field name unique to the fixture's typed payload;
    `terse_marker` off terse's own envelope. Both present => the client forwarded the
    duplicate AND terse compressed only one of them, which is the expensive case."""
    return {
        "terse_envelope": bool(re.search(r"__terse_(table|dict|diff|dropped)__", text)),
        "looks_like_records": '"rows"' in text or '"result"' in text,
        "chars": len(text),
        "tokens": _tokens(text),
    }


def request(flow) -> None:
    if "/v1/messages" not in flow.request.path:
        return
    if "anthropic.com" not in flow.request.pretty_host:
        return
    body = flow.request.get_text(strict=False) or ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _text_of(block)
            record = {
                "tool_use_id": block.get("tool_use_id"),
                "is_error": block.get("is_error", False),
                # The whole point: which top-level keys did the CLIENT put on the block?
                # If it ever forwards structuredContent it has to appear here.
                "block_keys": sorted(block.keys()),
                "verbatim": text,
            }
            record.update(_classify(text))
            _write(record)
