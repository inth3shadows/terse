"""Automatic success checkers for the cost_per_task task suite.

Every checker takes the model's final answer text plus the task's own
`check` dict (from tasks.json) and returns `(success, detail)` -- `detail`
is a short human-readable reason and is always populated, on a PASS as well
as a FAIL, because a green run with no reason attached is as unauditable as
a red one.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from pathlib import Path


_EMPHASIS = re.compile(r"^(\*\*|__|\*|_)(.+?)\1$", re.DOTALL)


def _norm(s: str) -> str:
    """Strip surrounding whitespace, quotes and backticks, then one layer of WHOLE-answer
    markdown emphasis (`**x**`, `__x__`, `*x*`, `_x_`), then repeat. Emphasis matters for
    fairness: the operator's `minimal` output style (arms B/C only) bolds the first line of
    every answer, so a correct `**2026-06-27**` failed in B/C where arm A's unstyled answer
    passed (pilot 2026-09-26). Only a wrapper around the WHOLE answer is removed; "not
    **kb-mcp**" still fails."""
    prev = None
    s = s.strip()
    while s != prev:
        prev = s
        s = s.strip().strip("`\"' \t\n")
        m = _EMPHASIS.match(s)
        if m:
            s = m.group(2)
    return s.lower()


def check_exact_match(answer: str, check: dict) -> tuple[bool, str]:
    """The WHOLE normalized answer must equal the normalized expected value
    -- not merely contain it. A whole-word substring search (the previous
    behavior) wrongly PASSES "not kb-mcp" or "kb-mcp, caddy-edge, or
    router-1": the expected token appears as a bare word in both, even
    though neither is the answer the task asked for ("ONLY the hostname, no
    punctuation, no explanation"). Exact whole-answer equality is the only
    check consistent with that prompt contract."""
    expected = check["expected"]
    got = _norm(answer)
    want = _norm(expected)
    ok = bool(want) and got == want
    return ok, f"expected {expected!r} (normalized {want!r}), got {answer!r} (normalized {got!r})"


def check_regex_match(answer: str, check: dict) -> tuple[bool, str]:
    """Extract the first match of `check['pattern']` (group 1 if the pattern
    has one, else the whole match) and compare numerically to
    `check['expected']` when both parse as floats, else as normalized text."""
    pattern = check["pattern"]
    expected = check["expected"]
    m = re.search(pattern, answer)
    if not m:
        return False, f"pattern {pattern!r} found no match in {answer!r}"
    got = m.group(1) if m.groups() else m.group(0)
    try:
        ok = math.isclose(float(got), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        ok = _norm(got) == _norm(expected)
    return ok, f"expected {expected!r}, extracted {got!r} from {answer!r}"


def check_command(answer: str, check: dict, workdir: Path) -> tuple[bool, str]:
    """Run `check['command']` in `workdir` via the shell; success is exit
    code == `check.get('expect_exit', 0)`. `answer` is unused but kept in
    the signature so every checker has the same shape for `run_check`'s
    dispatch table.

    If `check['protect_file']` is set (the control task's test_calc.py), its
    sha256 is verified against `check['protect_file_sha256']` BEFORE the
    command runs at all -- a model that "fixes" the test instead of the bug
    (e.g. loosening the assertion) must fail the task even if the doctored
    test then happens to pass, and checking the hash first means that
    doctored-and-passing case is caught rather than masked by a green exit
    code."""
    del answer
    protect_file = check.get("protect_file")
    if protect_file:
        expected_hash = check["protect_file_sha256"]
        protected_path = workdir / protect_file
        if not protected_path.exists():
            return False, f"protected file {protect_file!r} is missing from {workdir}"
        actual_hash = hashlib.sha256(protected_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            return False, (f"protected file {protect_file!r} was MODIFIED "
                            f"(sha256 {actual_hash} != expected {expected_hash}) -- "
                            f"the model edited the protected test instead of fixing "
                            f"the actual bug")
    command = check["command"]
    timeout = check.get("timeout", 120)
    # PYTHONDONTWRITEBYTECODE: a fixed-size same-second edit to a source file
    # (e.g. "a - b" -> "a + b", same byte length) can leave a stale .pyc
    # whose embedded mtime+size still matches, so a re-run silently imports
    # the OLD module. Each real run gets its own fresh workdir so this can't
    # happen in production, but it bit the harness's own tests (which reuse
    # one tmp dir across two edits) and is one env var to close for good.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(command, shell=True, cwd=workdir, capture_output=True,
                               text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, f"command {command!r} timed out after {timeout}s"
    expect_exit = check.get("expect_exit", 0)
    ok = proc.returncode == expect_exit
    tail = (proc.stdout + proc.stderr)[-500:]
    return ok, f"command {command!r} exit={proc.returncode} (expected {expect_exit}): ...{tail}"


_DISPATCH = {
    "exact_match": check_exact_match,
    "regex_match": check_regex_match,
    "command": check_command,
}


def run_check(task: dict, answer: str, workdir: Path) -> tuple[bool, str]:
    """Dispatch to the checker named by `task['check']['type']`."""
    check = task["check"]
    kind = check["type"]
    fn = _DISPATCH.get(kind)
    if fn is None:
        raise ValueError(f"unknown check type {kind!r} for task {task.get('id')!r}")
    if kind == "command":
        return check_command(answer, check, workdir)
    return fn(answer, check)
