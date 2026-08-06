"""The launch path baked into every wrapped MCP config must actually launch.

`install_mcp` writes `[sys.executable, "-m", "terse"]` as the command for every wrapped
entry (`terse_invocation`), so `python -m terse` is THE production entrypoint — if it fails,
every wrapped server on every user's machine fails to start.

`src/terse/__main__.py` had **0% coverage**. The existing tests assert the config *string*
(`entry["args"] == ["-m", "terse", "proxy", ...]`) and never execute it, which is false
confidence of the worst kind: breaking the import inside `__main__.py` leaves `python -m
terse` raising `ImportError` while all 1,287 other tests pass. Verified by doing exactly
that before writing this file.

These tests run the launcher's OWN output as a subprocess rather than a hand-written argv,
so the thing under test is the thing that ships — the same principle as
`test_published_primer_sizes.py`: two things that must stay in step get a test that reads
both, not a comment asking people to remember.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from terse.install_mcp import terse_invocation


def _run(argv: list[str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*argv, *extra], capture_output=True, text=True, timeout=120,
                          check=False)


@pytest.fixture(autouse=True)
def _default_launcher(monkeypatch):
    """Clear `$TERSE_MCP_CMD` for EVERY test in this file.

    Found in review, and it defeated the file's whole purpose: `terse_invocation()` honours
    that documented override and returns the operator's console script instead of
    `[sys.executable, "-m", "terse"]`. With the variable set, tests 1 and 2 passed even with
    `__main__.py`'s import broken — the subprocess never touched `__main__.py` at all, so
    the exact regression this file exists to catch was invisible. Only the last test cleared
    it; autouse makes that the default rather than something each test must remember."""
    monkeypatch.delenv("TERSE_MCP_CMD", raising=False)


def test_the_launcher_install_mcp_writes_actually_starts_terse():
    """Not `[sys.executable, "-m", "terse"]` spelled out here — `terse_invocation()`'s real
    return value, executed. A test with its own copy of the argv would keep passing after
    the launcher changed to something that does not run."""
    argv = terse_invocation()
    proc = _run(argv, "--version")
    assert proc.returncode == 0, (
        f"the launcher baked into every wrapped MCP config does not start: "
        f"{argv} --version exited {proc.returncode}\nstderr: {proc.stderr}")
    assert proc.stdout.strip(), "no version on stdout"


def test_the_module_entrypoint_reports_the_cli_exit_code():
    """`__main__` is `raise SystemExit(main())`, so a nonzero CLI result has to survive to
    the shell — a wrapped entry that always exits 0 hides a failed proxy from the client
    supervising it. `stats --log <missing>` is the repo's own documented exit-2 path."""
    argv = terse_invocation()
    ok = _run(argv, "--version")
    bad = _run(argv, "stats", "--log", "/nonexistent/terse-does-not-exist.jsonl")
    assert ok.returncode == 0
    assert bad.returncode == 2, (
        f"expected the CLI's exit 2 to propagate through `-m terse`, got "
        f"{bad.returncode}\nstderr: {bad.stderr}")
    # The MESSAGE too, not just the integer (found in review): argparse also exits 2 on an
    # unknown flag, so renaming `--log` would leave this green while proving nothing about
    # the documented no-ledger branch it is meant to exercise.
    assert "no ledger" in bad.stderr, bad.stderr


def test_the_console_script_entrypoint_agrees_with_the_module_one():
    """`pyproject` ships a `terse` console script as well, and README/USAGE use it
    throughout while wrapped configs use `-m terse`. Two documented entrypoints that must
    behave identically — pinned against each other rather than separately.

    Skipped rather than failed when the console script is absent: a bare `pytest` against a
    source checkout that was never installed has no `terse` on PATH, and that is a property
    of the environment, not a defect in the code."""
    # `shutil.which` was wrong (found in review): it finds *a* terse, not the one belonging
    # to this interpreter. On a machine with a global uv-tool terse installed, a plain
    # `./.venv/bin/pytest` compared the venv module against that unrelated install and
    # FAILED on two legitimately different versions — the same cry-wolf failure this branch
    # fixes in the secret gate. Resolve the script that ships beside `sys.executable`.
    script_path = Path(sys.executable).parent / "terse"
    if not script_path.exists():
        pytest.skip(f"no console script at {script_path} (source checkout, not installed)")
    module = _run(terse_invocation(), "--version")
    script = _run([str(script_path)], "--version")
    assert script.returncode == module.returncode == 0
    assert script.stdout.strip() == module.stdout.strip(), (
        "the console script and `-m terse` report different versions — one of them is "
        "resolving to a different install")


def test_the_entrypoint_uses_this_interpreter_not_whatever_is_on_path(monkeypatch):
    """`terse_invocation` returns an ABSOLUTE interpreter path on purpose: a wrapped entry is
    launched by an MCP client whose PATH is not the operator's shell PATH, so `terse` or
    `python` by bare name resolves unpredictably (or not at all). Pinned because the
    absolute path is the whole reason `-m terse` is used instead of the console script."""
    # $TERSE_MCP_CMD deliberately overrides this (documented, for versioned-venv installs),
    # so clear it: the assertion is about the DEFAULT branch, and inheriting the operator's
    # environment would make the test pass or fail on their shell rather than on the code.
    monkeypatch.delenv("TERSE_MCP_CMD", raising=False)
    argv = terse_invocation()
    assert argv[0] == sys.executable
    assert argv[0].startswith("/") or ":" in argv[0], f"not an absolute path: {argv[0]!r}"
    assert argv[1:] == ["-m", "terse"]
