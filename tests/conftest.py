"""Shared test guards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_launcher_selection(monkeypatch):
    """Make `install_mcp.terse_invocation()` return the same thing on a developer's box as
    it does in CI, by pinning the two environment inputs it reads.

    Since #275 it prefers any `terse` console script on `$PATH` from outside this venv.
    In CI (`uv sync`, project installed only into `.venv`) nothing qualifies and it yields
    `[sys.executable, "-m", "terse"]`; on a box with a global `~/.local/bin/terse` it
    yields `['/home/…/.local/bin/terse']` instead. Both are correct behaviour, but only
    ~15 of the ~45 `do_install` call sites monkeypatch `terse_invocation`, so the rest
    were asserting against whichever argv SHAPE the developer happened to have — a
    one-element command on a dev box, three elements in CI. That is how the Windows
    `terse.exe` regression stayed invisible: the shape that breaks detection is the shape
    CI never produces.

    Dropping only the `$PATH` entries that actually provide a `terse` forces the CI shape
    everywhere, and — unlike replacing `$PATH` wholesale with this venv's bin — leaves
    `git` on it, which `default_repo_path` and the changelog tests shell out to. The venv's
    own bin goes first because it is inside `sys.prefix`, i.e. exactly what tier 2 skips.
    Tests that care about tier 2 build their own `$PATH` on top; `$TERSE_MCP_CMD` is
    cleared rather than set, because setting it skips the tiers entirely and would make
    those tests vacuous."""
    monkeypatch.delenv("TERSE_MCP_CMD", raising=False)
    kept = [d for d in os.environ.get("PATH", os.defpath).split(os.pathsep)
            if d and not any((Path(d) / n).exists() for n in ("terse", "terse.exe"))]
    monkeypatch.setenv("PATH",
                       os.pathsep.join([str(Path(sys.executable).parent), *kept]))
    # `console_script_version` is memoized (one subprocess per install, not two). Tests
    # reuse tmp paths across a session and some rewrite a stub in place, so a result
    # cached by an earlier test would decide a later one.
    from terse.install_mcp import console_script_version
    console_script_version.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_xdg_state(monkeypatch, tmp_path_factory):
    """Point $XDG_STATE_HOME at a per-session temp dir so no test — present or future —
    can write the proxy's default-on savings ledger (stats.py) into the real
    ~/.local/state. Tests that care about the path set their own value on top."""
    monkeypatch.setenv("XDG_STATE_HOME",
                       str(tmp_path_factory.getbasetemp() / "xdg-state"))


# --------------------------------------------------------------------------- #
# Shared drop-eval CLI fixtures (#386 review): ONE copy of the payload, the drop-field
# spec, the policy doc and the `_cmd_fluency` Namespace, so a new required `args.<field>`
# is added in one place and every driver test sees it.
# --------------------------------------------------------------------------- #

DROP_EVAL_PAYLOAD = {"result": [{"id": i, "name": f"n{i}",
                                 "body": f"{i} " + "lorem ipsum dolor " * 40}
                                for i in range(6)]}
DROP_EVAL_FIELDS = {"result[].body": {"lossy": "drop-to-retrieve", "min": 200}}


def drop_eval_policy_doc(tiers, *, suggested=True):
    """A policy doc with one rule for `kb.read.list_principles`. `suggested=True` carries
    the drop as `_suggested_fields` (the `tune --drop-eval` input); `False` as live
    `fields` (the `fluency --drop-eval` input)."""
    rule = {"match": {"tool": "kb.read.list_principles"}, "tiers": tiers}
    if suggested:
        rule["_suggested_fields"] = dict(DROP_EVAL_FIELDS)
        rule["_suggested_fields_note"] = "…"
    else:
        rule["fields"] = dict(DROP_EVAL_FIELDS)
    return {"version": 1,
            "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
            "policies": [rule]}


def drop_eval_envelope(sha="a", raw=None):
    return {"tool": "kb.read.list_principles", "server": None, "sha": sha,
            "raw": json.dumps(DROP_EVAL_PAYLOAD) if raw is None else raw}


def fluency_drop_eval_args(*, corpus, policy, out, no_control=True, trials=1):
    """The full `_cmd_fluency` Namespace for a `--drop-eval` run."""
    return argparse.Namespace(
        corpus=str(corpus), policy=str(policy), drop_eval=True, out=str(out),
        trials=trials, no_control=no_control, accept_degraded=False, bars=False, html=False,
        diff=False, diff_soak=False, text_diff_eval=False, codec_verdict=False,
        pack=None, models=None, base_url=None, api_key=None)
