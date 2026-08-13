"""The launch path baked into every wrapped MCP config must actually launch.

`terse_invocation` picks the command for every wrapped entry, so whatever it returns is THE
production entrypoint — if it fails, every wrapped server on every user's machine fails to
start. Since #275 that is normally an installed `terse` console script, with
`[sys.executable, "-m", "terse"]` as the fallback when terse is installed nowhere but the
venv running install-mcp. Both tiers are executed here; neither is allowed to be a string
this file merely asserts.

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

import os
import subprocess
import sys
from pathlib import Path

import pytest

from terse.install_mcp import terse_invocation


def _run(argv: list[str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*argv, *extra], capture_output=True, text=True, timeout=120,
                          check=False)


# `conftest._isolate_launcher_selection` pins `$PATH`/`$TERSE_MCP_CMD` for every test in
# the suite, which is what keeps the tests below on THIS checkout's interpreter instead of
# a globally installed terse at some other version. Both inputs matter and both have
# already made this file vacuous once: with `$TERSE_MCP_CMD` set, the first two tests
# passed with `__main__.py`'s import broken, because the subprocess never reached
# `__main__.py` at all.


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX shim script")
@pytest.mark.parametrize("tier", ["console-script", "interpreter"])
def test_the_entrypoint_is_absolute_and_runs_whichever_tier_supplies_it(
        tmp_path, monkeypatch, tier):
    """A wrapped entry is launched by an MCP client whose PATH is not the operator's shell
    PATH, so `terse` or `python` by bare name resolves unpredictably (or not at all).
    Absoluteness is the invariant; which tier of `terse_invocation` supplies it is not.

    This asserted `argv[0] == sys.executable` before #275 — stricter than the reason its
    own docstring gave. An absolute console-script path satisfies "never resolved off PATH
    at launch time" just as well, and is now preferred, because `sys.executable` under
    `uv run` is a worktree venv that gets deleted. Matching the assertion to the stated
    rationale lets it cover BOTH tiers, which pinning one exact string could not — and
    tier 2 is the one that now ships, so leaving it unexecuted would mean every test in
    this file verifies only the fallback.

    The console script is a real shim onto THIS checkout rather than the installed terse,
    for the same reason the shared PATH guard exists: a test that runs an install nobody
    is editing reports on the wrong code."""
    if tier == "console-script":
        script = tmp_path / "bin" / "terse"
        script.parent.mkdir(parents=True)
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m terse "$@"\n',
                          encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", str(script.parent))  # outside sys.prefix -> tier 2
    argv = terse_invocation()
    assert argv == ([str(script)] if tier == "console-script"
                    else [sys.executable, "-m", "terse"])
    assert Path(argv[0]).is_absolute(), f"not an absolute path: {argv[0]!r}"

    proc = _run(argv, "--version")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("terse ")
