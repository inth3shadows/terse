"""#424 — a row ABSENT from its scope cannot win the precedence slot over a present one.

A `folded` row is an entry removed from that scope's `mcpServers` when it was stashed behind
the router, so the client launches the LOWER-scope definition of the name. Keeping the folded
row deleted the live writer before `_contested_labels` could count it, and the router banked
rows another process wrote.
"""
from __future__ import annotations

import json

import pytest

from terse.stats import _precedence_winner, primer_liability


def _pol(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "*"}, "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    return str(p)


def _row(server, state, wraps, pol, scope, identity=None, explicit=None):
    r = {"scope": scope, "server": server, "state": state, "wraps": wraps, "policy": pol}
    if identity is not None:
        r["ledger_identity"], r["ledger_identity_explicit"] = identity, explicit
    return r


def _agg(*rows):
    tools = [{"server": s, "tool": "t", "blocks": b, "tokenized": b, "raw_tokens": r,
              "out_tokens": o, "context_raw_tokens": r, "context_out_tokens": o,
              "raw_chars": 0, "out_chars": 0, "diffs": 0} for s, b, r, o in rows]
    keys = ("blocks", "raw_tokens", "out_tokens", "context_raw_tokens", "context_out_tokens")
    return {"total": {k: sum(t[k] for t in tools) for k in keys}
            | {"raw_chars": 0, "out_chars": 0, "untokenized": 0},
            "decisions": {}, "diff_reasons": {}, "tools": tools}


@pytest.mark.parametrize("absent", ["folded", "folded-unstashed", "orphaned-stash"])
def test_the_issues_repro_the_live_lower_scope_duplicate_contests_the_router(
        tmp_path, absent):
    """#424's own reproduction. Before: `terse ledger_labels ['gh','kb'] blocks 15
    contested_labels [] KEEP`, and the user-scope proxy absent from the report."""
    pol = _pol(tmp_path)
    rows = [_row("terse", "router", "gh, kb", pol, "project"),
            _row("kb", absent, None, pol, "project"),
            _row("kb", "wrapped", "kb-server --stdio", pol, "user",
                 identity="kb", explicit=True)]
    liab = primer_liability(rows, _agg(("gh", 5, 5_000, 1_000), ("kb", 10, 10_000, 2_000)))
    served = {s["server"]: s for s in liab["servers"]}

    # #396's answer for a label two processes write: unattributable on BOTH entries, never
    # banked by either. Before the fix the router read `blocks 15` and KEEP.
    for name in ("terse", "kb"):
        assert served[name]["contested_labels"] == ["kb"]
        assert served[name]["break_even_verdict"] == "router/live duplicate label"
        assert served[name]["verdict"] == "INSUFFICIENT"
    assert served["terse"]["ledger_labels"] == ["gh"]
    # ...and the definition that actually runs is reported, not silently dropped.
    assert served["kb"]["scope"] == "user"


def test_a_name_defined_ONLY_by_an_absent_row_keeps_its_slot():
    """Still a total order: a plain folded peer behind its router has no present definition
    anywhere, and must not vanish from the precedence map."""
    # Real scan order (user -> project), so first-wins cannot pass for scope ranking.
    rows = [{"scope": "user", "server": "kb", "state": "orphaned-stash"},
            {"scope": "project", "server": "kb", "state": "folded"}]
    assert _precedence_winner(rows) == {"kb": 1}


@pytest.mark.parametrize("present", ["wrapped", "wrapped-unstashed", "folded-and-live",
                                     "unwrapped", "router"])
def test_every_present_state_still_wins_on_scope_alone(present):
    """The absent rank must not leak onto present states: #398's rule is unchanged for them,
    project beating user."""
    rows = [{"scope": "user", "server": "kb", "state": "wrapped"},
            {"scope": "project", "server": "kb", "state": present}]
    assert _precedence_winner(rows) == {"kb": 1}


def test_absent_rank_beats_scope_rank_even_from_local():
    rows = [{"scope": "user", "server": "kb", "state": "unwrapped"},
            {"scope": "local", "server": "kb", "state": "folded"}]
    assert _precedence_winner(rows) == {"kb": 0}


def test_the_absent_states_are_exactly_the_scan_rows_missing_from_mcpServers(tmp_path):
    """The rule is only as good as the state list it keys on, and that list lives in
    `install_mcp`. Driven through the REAL scanner over a real multiproxy install, a stash
    entry whose live entry was hand-deleted, and a folded peer whose stash record is gone:
    for every row, "absent" by `_ABSENT_FROM_SCOPE` must equal "not in mcpServers"."""
    from terse import install_mcp as im
    from terse.stats import _ABSENT_FROM_SCOPE

    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "kb": {"command": "kb-mcp"}, "gh": {"command": "gh-mcp"},
        "ghost": {"command": "ghost-mcp"}, "keep": {"command": "keep-mcp"},
        "other": {"command": "other-mcp"}}}), encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}),
                   encoding="utf-8")
    im.do_install(["kb", "gh"], str(pol), cfg=cfg, multiproxy=True)
    im.do_install(["ghost", "keep"], str(pol), cfg=cfg)
    live = json.loads(cfg.read_text())
    del live["mcpServers"]["ghost"]                       # -> orphaned-stash
    cfg.write_text(json.dumps(live), encoding="utf-8")
    stash = json.loads(im.stash_path(cfg).read_text())
    del stash["user"]["gh"]                               # -> folded-unstashed
    im.stash_path(cfg).write_text(json.dumps(stash), encoding="utf-8")

    present = set(json.loads(cfg.read_text())["mcpServers"])
    rows = [r for r in im.scan_scopes(cfg=cfg) if r["scope"] == "user"]
    states = {r["server"]: r["state"] for r in rows}
    # Every absent state is exercised, so a renamed state cannot pass by omission.
    assert set(_ABSENT_FROM_SCOPE) <= set(states.values()), states
    for r in rows:
        assert (r["state"] in _ABSENT_FROM_SCOPE) == (r["server"] not in present), r


def test_ambiguity_sees_the_running_entry_not_the_absent_row_above_it(tmp_path):
    """Review finding: `_ambiguous_labels` is the third consumer. Project `a` is folded, so
    the user `a` runs and guesses `python` like `b` does. Keeping the folded row let `b` bank
    10 `python` rows the running `a` also wrote — #285's double count."""
    from terse.stats import _ambiguous_labels
    pol = _pol(tmp_path)
    rows = [_row("a", "wrapped", "/usr/bin/python -m a", pol, "user",
                 identity="python", explicit=False),
            _row("b", "wrapped", "/usr/bin/python -m b", pol, "user",
                 identity="python", explicit=False),
            _row("terse", "router", "a", pol, "project"),
            _row("a", "folded", None, pol, "project")]
    assert _ambiguous_labels(rows) == {"python"}
    liab = primer_liability(rows, _agg(("python", 10, 1_000, 500)))
    served = {s["server"]: s for s in liab["servers"]}
    assert served["a"]["ledger_labels"] == [] and served["b"]["ledger_labels"] == []
