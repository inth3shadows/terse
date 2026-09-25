"""Fluency ledger: one JSONL row per live model call the fluency harnesses make.

The harness scores an entire run in memory and (until now) threw every per-probe
call away — `terse fluency`'s report is the only trace a run left, and the report
aggregates away exactly the raw material (which prompt, which reply, which trial)
a later investigation would want. This module is the raw record: one row per
`harnesses._ask_n` call — the choke point every harness funnels through (see that
function's docstring) — so `run_payload`'s four arms, `run_diff_payload`'s two,
`run_chain_payload`'s chain form, and `run_text_diff_payload`'s text form all log
through the same path with no per-caller duplication.

Fail-open, same contract as stats.py/capture.py's ledgers: a full disk, an
unwritable path, or a bad $TERSE_FLUENCY_LEDGER override must never change a
run's (correct, unanswered) result. Warn ONCE to stderr and keep going — a
per-call warning on a dead path would flood a 45-minute multi-model run's
progress lines into unreadability, exactly the failure mode `guarded()` (in
harnesses.py) already exists to prevent for the `progress` callback.

The repo has a standing rule (#263/#268, see `_ask_n`'s docstring) against
conflating "the model was never reached" / "the model answered nothing" with "the
model answered wrong". This ledger preserves that distinction on disk: `reply` is
`None` and `correct` is `None` for an unanswered call, never scored as a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .. import __version__
from .._secure_io import append_restricted, mkdir_restricted

_warned = False  # module-level, mirrors tokenize.py's _warned_degraded: warn once total


def default_ledger_path() -> Path:
    """$XDG_STATE_HOME/terse/fluency-ledger.jsonl (fallback ~/.local/state) — the exact
    convention `stats.default_stats_log()` uses for the savings ledger, so this file
    picks up the test suite's existing `_isolate_xdg_state` autouse fixture (conftest.py)
    for free instead of writing into a developer's real state dir on every test run.
    `$TERSE_FLUENCY_LEDGER` overrides the path itself; that env var is read at call time
    in `_ledger_path`, not here, so a test can flip it between calls."""
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(state) / "terse" / "fluency-ledger.jsonl"


def _ledger_path() -> Path | None:
    """Resolve the effective ledger path for THIS call. Unset env -> the default
    path. Env set to the empty string -> `None`, meaning disabled — an explicit
    opt-out, distinct from "not configured", per the plan's contract."""
    env = os.environ.get("TERSE_FLUENCY_LEDGER")
    if env is None:
        return default_ledger_path()
    if env == "":
        return None
    return Path(env)


def log_call(*, harness: str, model: str, arm: str, qtype: str, expected: Any,
            reply: str | None, correct: bool | None, unanswered: bool, trial: int,
            system: str, user: str) -> None:
    """Append one row for a single model call. Called from `_ask_n` once per trial,
    after the call resolves and before its result is folded into the (ok, fails)
    counters `_ask_n` returns.

    `expected` is passed through whatever a question's `expected` is (str, int,
    list, ...) — `json.dumps` below is what makes it JSON-safe, not this function.
    """
    path = _ledger_path()
    if path is None:
        return
    # Hashed together (not two separate hashes) so the row identifies one exact
    # (system, user) pair with one field, matching how the harnesses always send
    # the two joined as a single call — a "\x00" separator can't collide with real
    # prompt text (system/user prompts here are never NUL-bearing JSON/text).
    prompt = f"{system}\x00{user}"
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "terse_version": __version__,
        "harness": harness,
        "model": model,
        "arm": arm,
        "qtype": qtype,
        "expected": expected,
        "reply": reply,
        "correct": correct,
        "unanswered": unanswered,
        "trial": trial,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
    }
    try:
        mkdir_restricted(path.parent)
        append_restricted(path, json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 — a ledger write must never break a run
        global _warned
        if not _warned:
            _warned = True
            sys.stderr.write(
                f"[terse fluency] ledger write to {path} failed ({exc}); continuing "
                f"without it (further failures this run are suppressed)\n")
