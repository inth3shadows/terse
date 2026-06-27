"""MCP stdio proxy: compress downstream tool-call results per policy, transparently.

Sits between an MCP client (e.g. Claude Code) and one downstream MCP server. It
launches the server as a subprocess and forwards JSON-RPC both ways. The ONLY
thing it changes is the text of a `tools/call` *result*, which it runs through
`policy.apply()` using the tool name recorded from the matching request.

Design guarantees:
  - Transparent: every non-(tools/call-result) message is forwarded byte-for-byte.
  - Fail-open: any parse/compress error forwards the ORIGINAL message. A compression
    layer must never lose or corrupt a tool result.
  - Frame-safe: MCP stdio is newline-delimited JSON; terse minified output has no
    embedded newlines, so a compressed result stays one line.

The pure message logic lives in `Interceptor` (unit-tested without any I/O). The
`run_proxy` shell wires it to a subprocess with two pump threads.
"""

from __future__ import annotations

import json
import subprocess
import sys
from threading import Thread
from typing import Any, Callable, Optional, TextIO

from . import policy as policy_mod
from . import transforms
from .tokenize import count_cl100k


def _cost(text: str) -> int:
    """Token cost, falling back to byte length where tiktoken is unavailable."""
    c = count_cl100k(text)
    return c if c is not None else len(text)


# A one-time, system-level explanation of terse's wire forms, injected into the MCP
# `initialize` result's `instructions` field (#13). Measurement showed a *system-level*
# primer recovers comprehension that an inline per-result note cannot (the stdio proxy
# can't set a system prompt); `instructions` is the channel clients add to that context.
# Covers the always-on table/dict forms AND the opt-in diff form, so it helps base
# comprehension too — paid once per session, not per result.
TERSE_PRIMER = (
    "Some tool results are 'terse'-compressed (a lossless, denser JSON encoding); some "
    "are sent as diffs against the previous result of the same tool. Read each as the "
    "equivalent full JSON:\n"
    '- Table {"__terse_table__":1,"n":N,"cols":[...],"rows":[[...]]}: N records, each row '
    'POSITIONAL — its i-th value belongs to the i-th name in "cols". "n" is the exact count.\n'
    '- Dict {"__terse_dict__":1,"legend":{"~0":value,...},"data":...}: every "~K" token '
    'inside "data" stands for legend["~K"] — substitute it back.\n'
    '- Diff {"__terse_diff__":1,"shape":"rows","by":COL,"set":[...],"new":[...],"del":[...],'
    '"n":N}: update the PREVIOUS same-tool result — from its records drop ids in "del", '
    'overwrite/insert each record in "set" matched by its "by" field, append ids in "new"; '
    '"n" is the final record count. A {"shape":"keys","set":{...},"del":[...]} diff instead '
    'removes "del" keys and applies "set" key/values to the previous object. '
    "Always reason about the fully reconstructed result."
)


class Interceptor:
    """Pure JSON-RPC message logic. Tracks request id -> tool name and compresses
    matching results. No I/O; both methods take and return a single line of text
    (without the trailing newline).

    When `policy.diff` is on, it also keeps the previous per-tool result and emits a
    lossless delta when that is smaller than the full compressed form — the stateful
    cross-call lever. It is fail-open and self-verifying: a diff is sent only when it
    provably reconstructs the result, and the full form is always the fallback."""

    def __init__(self, pol: policy_mod.Policy, debug: bool = False):
        self.policy = pol
        self.pending: dict[Any, str] = {}
        self.debug = debug
        self.diff = pol.diff
        self.last: dict[str, Any] = {}  # tool -> previous result object (the diff base)
        self.init_id: Any = None        # id of the initialize request, to prime its reply

    def note_request(self, line: str) -> None:
        """Record id -> tool name for tools/call requests, and the initialize request id
        (so its reply can carry the format primer). Side-effect only."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize" and mid is not None:
            self.init_id = mid
            return
        if method != "tools/call":
            return
        name = (msg.get("params") or {}).get("name")
        if mid is not None and isinstance(name, str):
            self.pending[mid] = name

    def transform_response(self, line: str) -> str:
        """Compress the text of a tracked tools/call result; prime the initialize reply;
        else return unchanged."""
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line
        if not isinstance(msg, dict) or "result" not in msg or msg.get("id") is None:
            return line
        if msg["id"] == self.init_id:
            self.init_id = None  # one-time
            primed = self._augment_initialize(msg)
            return primed if primed is not None else line
        tool = self.pending.pop(msg["id"], None)
        if tool is None:
            return line  # not a tracked tools/call response (tools/list, ...)

        result = msg.get("result")
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, list):
            return line

        text_blocks = [b for b in content
                       if isinstance(b, dict) and b.get("type") == "text"
                       and isinstance(b.get("text"), str)]

        changed = False
        # Diffing reasons about ONE logical payload, so it only engages for a single
        # text block (the overwhelmingly common tool-result shape); multi-block results
        # take the plain per-block compression path.
        if self.diff and len(text_blocks) == 1:
            changed = self._compress_or_diff(text_blocks[0], tool)
        else:
            for block in text_blocks:
                new_text = self._compress(block["text"], tool)
                if new_text != block["text"]:
                    block["text"] = new_text
                    changed = True
        if not changed:
            return line
        # Re-serialize compactly. JSON-RPC is semantics, not formatting; no newlines.
        return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def _compress_or_diff(self, block: dict, tool: str) -> bool:
        """Compress one block, preferring a lossless delta vs the prior same-tool result
        when it is smaller. Updates the per-tool diff base. Returns whether the block
        text changed. Fail-open: any error leaves the block untouched and state intact."""
        text = block["text"]
        try:
            applied = policy_mod.apply(text, tool, self.policy)
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: passthrough on error: {exc}\n")
            return False
        if applied.skipped:
            return False  # passthrough tool: leave fully alone, keep no diff state

        chosen = applied.text
        try:
            curr = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            curr = None
        if curr is not None:
            prev = self.last.get(tool)
            if prev is not None:
                wire = self._diff_wire(prev, curr, tool)
                if wire is not None and _cost(wire) < _cost(applied.text):
                    chosen = wire
                    if self.debug:
                        sys.stderr.write(
                            f"[terse-proxy] {tool}: diff {_cost(applied.text)}->{_cost(wire)} "
                            f"tok vs full compressed\n")
            # Base the NEXT diff on the true current value, whichever form we emit:
            # the model's reconstructable state after this turn is `curr` either way.
            self.last[tool] = curr

        if chosen != text:
            block["text"] = chosen
            return True
        return False

    def _augment_initialize(self, msg: dict) -> Optional[str]:
        """Prepend the terse format primer to the initialize result's `instructions` (#13),
        preserving any the downstream server set. Idempotent. Returns the reserialized
        line, or None to forward unchanged."""
        result = msg.get("result")
        if not isinstance(result, dict):
            return None
        existing = result.get("instructions")
        existing = existing if isinstance(existing, str) else ""
        if TERSE_PRIMER in existing:
            return None
        result["instructions"] = TERSE_PRIMER + (f"\n\n{existing}" if existing else "")
        if self.debug:
            sys.stderr.write("[terse-proxy] injected terse format primer into "
                             "initialize.instructions\n")
        return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)

    def _diff_wire(self, prev: Any, curr: Any, tool: str) -> Optional[str]:
        """The on-the-wire diff envelope, or None if no lossless diff applies. Self-
        describing: it names the prior result (already in the model's context) and
        carries the changes inline, so the model reconstructs without an out-of-band
        retrieve. Shared with the fluency-for-diff eval via `transforms.diff_wire`."""
        try:
            return transforms.diff_wire(prev, curr, tool)
        except Exception:  # noqa: BLE001 — fail-open
            return None

    def _compress(self, text: str, tool: str) -> str:
        """policy.apply with a hard fail-open: any error returns the original text."""
        try:
            applied = policy_mod.apply(text, tool, self.policy)
            if self.debug and not applied.skipped and applied.text != text:
                sys.stderr.write(
                    f"[terse-proxy] {tool}: {len(text)}->{len(applied.text)} bytes "
                    f"(tiers={list(applied.tiers)})\n"
                )
            return applied.text
        except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
            if self.debug:
                sys.stderr.write(f"[terse-proxy] {tool}: passthrough on error: {exc}\n")
            return text


def pump(src: TextIO, dst: TextIO, transform: Callable[[str], Optional[str]]) -> None:
    """Read lines from src, apply transform (None = drop nothing here, forward), write
    to dst with a single trailing newline. Stops at EOF. Used for both directions."""
    for raw in src:
        line = raw.rstrip("\n")
        if not line:
            continue
        out = transform(line)
        if out is None:
            out = line
        dst.write(out + "\n")
        dst.flush()


def run_proxy(
    cmd: list[str],
    pol: policy_mod.Policy,
    debug: bool = False,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> int:
    """Launch the downstream MCP server `cmd` and proxy stdio through `Interceptor`."""
    cin = stdin or sys.stdin
    cout = stdout or sys.stdout
    inter = Interceptor(pol, debug=debug)

    proc = subprocess.Popen(  # noqa: S603 — cmd is operator-supplied, by design
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        encoding="utf-8",
    )
    assert proc.stdin is not None and proc.stdout is not None

    def client_to_server() -> None:
        def fwd(line: str) -> str:
            inter.note_request(line)
            return line  # forward request unchanged; only observe
        try:
            pump(cin, proc.stdin, fwd)
        finally:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass

    def server_to_client() -> None:
        pump(proc.stdout, cout, inter.transform_response)

    t_up = Thread(target=client_to_server, daemon=True)
    t_down = Thread(target=server_to_client, daemon=True)
    t_up.start()
    t_down.start()
    rc = proc.wait()
    t_down.join(timeout=2.0)
    return rc
