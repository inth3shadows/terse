#!/usr/bin/env python3
"""Drive a real MCP server through the terse proxy and capture what terse did to it.

Sends `initialize` -> `tools/list` -> each requested `tools/call` **twice**, writing every
raw payload into a capture corpus and a payload-free stats ledger. Feed the corpus to
`terse measure` for per-tool codec numbers and the ledger to `terse stats` for the
diff-reason breakdown.

Usage:
    mcp_probe.py <server_name> <corpus_dir> <stats_log> <calls_json> -- <server argv...>

    calls_json  JSON list of {"name": <tool>, "arguments": {...}}

Env:
    TERSE_BIN         terse executable (default: "terse" from PATH)
    PROBE_DEADLINE    seconds to wait for any single response (default: 300)
    PROBE_STDERR      set to 1 to inherit the proxy's stderr instead of teeing it to
                      <stats_log>.stderr, where terse's launch failures are written. (Its
                      capture/stats sink errors are NOT: those are swallowed unless the
                      proxy runs with --debug, which is why the artifact check below
                      exists.)

Exit status is non-zero if ANY request failed, so a sweep loop can detect a bad run.

Design notes (each of these exists because getting it wrong produced a *silently wrong
measurement*, not a crash):

  * **stdin stays open** until every response arrives. Closing it as soon as the requests
    are written makes the proxy tear the child down mid-call: fast servers still answer,
    slow ones (browser launch, HTTP fetch) return nothing, which reads as "that server is
    broken".
  * **Only `result`/`error` messages count as responses.** Servers legitimately send their
    own *requests* (`roots/list`, `sampling/createMessage`) and choose their own small
    integer ids, which collide with this probe's id space. Treating one as a response ends
    the run early and reports an empty corpus as a clean measurement. Inbound requests are
    answered `-32601` so a server that blocks on one does not stall to the deadline.
  * **`result.isError` is a failure.** MCP's normal tool-failure convention is a text block
    with `isError: true` — a mistyped path or a wrong argument name returns one. terse tees
    it to the corpus before the probe ever sees it, so this cannot *prevent* the poisoning;
    it makes the run exit non-zero so you know to discard the corpus and ledger and re-run.
  * **The two repeats are serialized**, not pipelined. Servers dispatch concurrently, so
    batched requests can be answered out of order; the proxy sets its diff base in arrival
    order, which would make the diff measurement nondeterministic.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading

TERSE_BIN = os.environ.get("TERSE_BIN", "terse")
DEADLINE = float(os.environ.get("PROBE_DEADLINE", "300"))
INHERIT_STDERR = os.environ.get("PROBE_STDERR") == "1"


class Probe:
    """Owns the proxy subprocess and the request/response correlation."""

    def __init__(self, argv: list[str], stderr_target):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_target,
            text=True, start_new_session=True)   # own process group -> killable as a tree
        self._cv = threading.Condition()
        self._responses: dict[int, dict] = {}
        self._stdin_lock = threading.Lock()
        self._eof = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _write(self, msg: dict) -> None:
        with self._stdin_lock:
            if self.proc.stdin is None or self.proc.stdin.closed:
                return
            try:
                self.proc.stdin.write(json.dumps(msg) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def _read_loop(self) -> None:
        try:
            if self.proc.stdout is None:
                return
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                # A server-initiated REQUEST (has "method" and an id) — not a response.
                # Its id may collide with ours, so it must never satisfy a wait.
                if "method" in msg:
                    if msg.get("id") is not None:
                        self._write({"jsonrpc": "2.0", "id": msg["id"],
                                     "error": {"code": -32601,
                                               "message": "probe does not implement "
                                                          f"{msg.get('method')!r}"}})
                    continue
                if "result" not in msg and "error" not in msg:
                    continue
                with self._cv:
                    self._responses[msg.get("id")] = msg
                    self._cv.notify_all()
        finally:
            with self._cv:                     # never strand a waiter on reader death
                self._eof = True
                self._cv.notify_all()

    def request(self, mid: int, method: str, params: dict | None = None) -> dict | None:
        """Send one request and wait for ITS response. Returns None on timeout/EOF."""
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)
        with self._cv:
            ok = self._cv.wait_for(lambda: mid in self._responses or self._eof, DEADLINE)
            if not ok:
                return None
            return self._responses.get(mid)

    def notify(self, method: str) -> None:
        self._write({"jsonrpc": "2.0", "method": method})

    def close(self) -> None:
        """Shut the tree down gracefully. SIGKILL alone would bypass terse's own
        `finally: transport.close()`, orphaning grandchildren (a browser, a language
        server) that outlive the probe."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self.proc.kill()


def _describe(msg: dict | None) -> tuple[bool, str]:
    """(ok, description) for one response."""
    if msg is None:
        return False, "NO RESPONSE (raise PROBE_DEADLINE?)"
    if "error" in msg:
        return False, f"PROTOCOL ERROR {str(msg['error'])[:110]}"
    result = msg.get("result") or {}
    content = result.get("content") or []
    text = content[0].get("text", "") if content and content[0].get("type") == "text" else ""
    if result.get("isError"):
        return False, f"TOOL ERROR {text[:110]}"
    is_diff = False
    try:
        env = json.loads(text)
        is_diff = isinstance(env, dict) and env.get("__terse_diff__") == 1
    except ValueError:
        pass
    return True, f"blocks={len(content)} chars={len(text)}{' DIFF' if is_diff else ''}"


def main(argv: list[str]) -> int:
    if len(argv) < 7 or argv[5] != "--":
        print(__doc__)
        return 2
    server_name, corpus, stats_log, calls_json = argv[1:5]
    server_argv = argv[6:]
    try:
        calls = json.loads(calls_json)
    except ValueError as exc:
        print(f"calls_json is not valid JSON: {exc}")
        return 2

    proxy_argv = [TERSE_BIN, "proxy", "--server-name", server_name,
                  "--capture-dir", corpus, "--stats-log", stats_log, "--"] + server_argv

    err_path = stats_log + ".stderr"
    with contextlib.ExitStack() as stack:
        err_fh = (None if INHERIT_STDERR
                  else stack.enter_context(open(err_path, "w", encoding="utf-8")))
        return _run(proxy_argv, server_name, calls, err_fh, err_path,
                    corpus, stats_log)


def _run(proxy_argv: list[str], server_name: str, calls: list[dict], err_fh,
         err_path: str, corpus: str, stats_log: str) -> int:
    probe = Probe(proxy_argv, err_fh)
    failed = False
    try:
        init = probe.request(1, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "terse-probe", "version": "0"}})
        ok, desc = _describe(init)
        if not ok:
            # failed BEFORE returning, or the finally's tail-dump is skipped for the single
            # most common real failure (downstream cannot launch) and the only thing printed
            # is misleading "raise PROBE_DEADLINE" advice.
            failed = True
            print(f"[{server_name}] initialize FAILED: {desc}")
            return 1
        probe.notify("notifications/initialized")

        listed = probe.request(2, "tools/list")
        tools = (listed or {}).get("result", {}).get("tools", []) if listed else []
        print(f"[{server_name}] init=True tools={len(tools)}")
        if not tools:
            print("  WARNING: server advertised no tools")
            failed = True

        mid = 10
        for call in calls:
            name = call["name"]
            # Serialized: rep1 is sent only after rep0's response has landed, so the
            # proxy's diff base is set in a deterministic order.
            for rep in (0, 1):
                resp = probe.request(mid, "tools/call",
                                     {"name": name, "arguments": call.get("arguments", {})})
                ok, desc = _describe(resp)
                if not ok:
                    print(f"  {name} rep{rep}: {desc}")
                    failed = True
                    if resp is None:
                        # Abort rather than pay DEADLINE for every remaining request (that
                        # turned a bounded wait into DEADLINE x 2 x len(calls)). Also stops
                        # rep1 going out after a rep0 timeout, where a late rep0 could still
                        # set the proxy's diff base concurrently with rep1.
                        print("  aborting: no response (a later reply would race the "
                              "diff base)")
                        return 1
                elif rep == 1:
                    print(f"  {name:24} rep1 {desc}")
                mid += 1
        # The numbers in BENCHMARKS §6 come from the corpus and the ledger, not from the
        # JSON-RPC responses -- so a run that answered every request but wrote neither is
        # still a failed measurement. terse swallows capture/stats sink errors unless
        # --debug is passed, so nothing else reports this.
        expected_records = 2 * len(calls)
        n_payloads = (len([f for f in os.listdir(corpus) if f.endswith(".json")])
                      if os.path.isdir(corpus) else 0)
        if n_payloads <= 0:
            print(f"  ARTIFACT CHECK: corpus {corpus!r} has no captured payloads")
            failed = True
        n_records = 0
        if os.path.isfile(stats_log):
            with open(stats_log, encoding="utf-8") as fh:
                n_records = sum(1 for line in fh if line.strip())
        if n_records < expected_records:
            print(f"  ARTIFACT CHECK: ledger has {n_records} record(s), expected "
                  f"{expected_records}")
            failed = True
    finally:
        probe.close()
        if err_fh is not None:
            err_fh.flush()            # flush, not close: the ExitStack owns closing it
            if failed and os.path.getsize(err_path) > 0:
                print(f"  --- proxy stderr ({err_path}) ---")
                with open(err_path, encoding="utf-8") as fh:
                    for line in fh.read().splitlines()[-10:]:
                        print(f"  {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
