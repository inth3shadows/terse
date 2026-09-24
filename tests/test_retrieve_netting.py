"""#438 — a `terse.retrieve` hit is netted out of the context saving the verdict reads.

A hit returns the very value the drop rule was credited with, so the gross saving counts a
saving the model then paid back. On the live ledger that was 145,606 tok against 1,508,007
(9.7% fleet-wide, 24% of `codegraph_explore`'s own saving), and before this it moved no
verdict: the cost rendered only in `terse stats`' retrieve table, never in the break-even
or on `--recommend`.

Driven through `main(["stats", ...])` wherever a number reaches a screen, because the unit
(`primer_liability`) being right says nothing about the call sites that render it.
"""
from __future__ import annotations

import json
import time

import pytest

from terse.cli import main
from terse.stats import append_stats, primer_liability


@pytest.fixture
def install(monkeypatch, tmp_path):
    """One wrapped `kb` entry whose ledger label is `kb-server` — sized from a fixed
    policy, never the real user config."""
    import terse.install_mcp as install_mcp

    pol = tmp_path / "pol.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "*"},
                                             "tiers": ["minify", "tabularize"]}]}),
                   encoding="utf-8")
    monkeypatch.setattr(install_mcp, "scan_scopes",
                        lambda *a, **k: [{"scope": "user", "server": "kb",
                                          "state": "wrapped", "wraps": "kb-server",
                                          "policy": str(pol)}])


def _ledger(tmp_path, *, retrieves, tokens=1_000, untokenized=0):
    """10 blocks saving 900 tok each (9,000 total), plus `retrieves` hits of `tokens`."""
    log = tmp_path / "stats.jsonl"
    ts = int(time.time())
    for _ in range(10):
        append_stats({"ts": ts, "server": "kb-server", "tool": "t",
                      "decision": "compressed", "raw_chars": 4_000, "out_chars": 400,
                      "raw_tokens": 1_000, "out_tokens": 100}, log)
    for i in range(retrieves + untokenized):
        append_stats({"ts": ts, "server": "kb-server", "tool": "t", "event": "retrieve",
                      "path": "$.body", "hit": True, "bytes": 4 * tokens,
                      "tokens": tokens if i < retrieves else None}, log)
    return log


def _run(capsys, log, *flags):
    capsys.readouterr()
    assert main(["stats", "--log", str(log), *flags]) == 0
    return capsys.readouterr().out


def test_the_published_saving_is_net_of_retrieves_and_the_wire_figure_is_not(
        install, tmp_path, capsys):
    liab = json.loads(_run(capsys, _ledger(tmp_path, retrieves=3), "--json"))[
        "primer_liability"]
    assert liab["retrieve_tokens"] == 3_000
    assert liab["saved_tokens"] == 9_000 - 3_000
    # The pipe figure stays gross: a retrieve is a separate call, not a worse compression.
    assert liab["wire_saved_tokens"] == 9_000
    # ...and the per-entry rate — what the verdict actually divides — is netted too, not
    # only the headline. 6,000 over 10 tokenized blocks.
    assert liab["servers"][0]["saved_per_block"] == pytest.approx(600.0)
    assert liab["servers"][0]["contributors"][0]["saved_tokens"] == 6_000


def test_retrieves_that_cost_more_than_the_drop_saved_flip_the_verdict_to_unwrap(
        install, tmp_path, capsys):
    """The decisive case: the model fetched back more than terse ever withheld. Gross, this
    entry reads KEEP; net, it is paying terse to make it do extra work."""
    def verdict(sub, retrieves):
        (tmp_path / sub).mkdir()
        out = _run(capsys, _ledger(tmp_path / sub, retrieves=retrieves), "--recommend")
        return next(ln.split()[1] for ln in out.splitlines() if ln.startswith("  kb "))

    assert verdict("gross", 0) == "KEEP"
    # 10 x 1,000 fetched back against 9,000 withheld: a net rate of -100/block.
    assert verdict("net", 10) == "UNWRAP"


@pytest.mark.parametrize("flags", [(), ("--recommend",)])
def test_both_screens_name_the_deduction_beside_the_netted_number(
        install, tmp_path, capsys, flags):
    out = _run(capsys, _ledger(tmp_path, retrieves=3), *flags)
    assert "net of 3,000 tok the model spent on terse.retrieve" in out


@pytest.mark.parametrize("flags", [(), ("--recommend",)])
def test_neither_screen_mentions_retrieves_when_none_happened(
        install, tmp_path, capsys, flags):
    out = _run(capsys, _ledger(tmp_path, retrieves=0), *flags)
    assert "terse.retrieve" not in out


def test_an_untokenized_retrieve_makes_the_net_an_upper_bound_and_says_so(
        install, tmp_path, capsys):
    log = _ledger(tmp_path, retrieves=1, untokenized=2)
    liab = json.loads(_run(capsys, log, "--json"))["primer_liability"]
    assert (liab["retrieve_tokens"], liab["retrieve_untokenized"]) == (1_000, 2)
    out = _run(capsys, log, "--recommend")
    assert "net of 1,000 tok" in out
    assert "2 retrieve(s) recorded without tiktoken" in out and "upper bound" in out


def test_a_routers_retrieve_is_billed_to_the_peer_that_dropped_the_value(tmp_path):
    """Keyed by the ORIGIN label (`build_retrieve_writer`), the same key as the tool rows —
    so a router's contributor ranking moves on the peer that paid, and only that one."""
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "*"}, "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    tools = [{"server": s, "tool": "t", "blocks": 5, "tokenized": 5, "raw_tokens": 5_000,
              "out_tokens": 1_000, "raw_chars": 0, "out_chars": 0, "diffs": 0}
             for s in ("codegraph", "kb")]
    agg = {"total": {"blocks": 10, "raw_tokens": 10_000, "out_tokens": 2_000,
                     "raw_chars": 0, "out_chars": 0, "untokenized": 0},
           "decisions": {}, "diff_reasons": {}, "tools": tools,
           "retrieves": [{"server": "codegraph", "tool": "t", "path": "$.x", "calls": 2,
                          "hits": 2, "misses": 0, "bytes": 0, "tokens": 1_500,
                          "untokenized": 0}]}
    liab = primer_liability([{"scope": "user", "server": "terse", "state": "router",
                              "wraps": "codegraph, kb", "policy": str(pol)}], agg)
    by = {c["label"]: c["saved_tokens"] for c in liab["servers"][0]["contributors"]}
    assert by == {"codegraph": 4_000 - 1_500, "kb": 4_000}
    assert liab["servers"][0]["saved_per_block"] == pytest.approx(6_500 / 10)
    assert liab["saved_tokens"] == 8_000 - 1_500


def _pol(tmp_path):
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "*"}, "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    return str(pol)


def _agg(tools, retrieves, untokenized=()):
    rows = [{"server": s, "tool": "t", "blocks": b, "tokenized": b, "encoded": b,
             "raw_tokens": r, "out_tokens": o, "context_raw_tokens": r,
             "context_out_tokens": o, "raw_chars": 0, "out_chars": 0, "diffs": 0}
            for s, b, r, o in tools]
    return {"total": {k: sum(t[k] for t in rows) for k in
                      ("blocks", "raw_tokens", "out_tokens", "context_raw_tokens",
                       "context_out_tokens")} | {"raw_chars": 0, "out_chars": 0,
                                                 "untokenized": 0},
            "decisions": {}, "diff_reasons": {}, "tools": rows, "primers": [],
            "retrieves": [{"server": s, "tool": "t", "path": "$.x", "calls": 1, "hits": 1,
                           "misses": 0, "bytes": 0, "tokens": tk,
                           "untokenized": int(s in untokenized)}
                          for s, tk in retrieves]}


def test_a_retrieve_under_an_idle_contested_label_is_not_billed_to_the_router(tmp_path):
    """Review of #438. `kb` is claimed by the router AND a live duplicate (#396) and wrote
    no tool rows this window — only a retrieve, whose drop predates the window. Netting it
    flipped the router to UNWRAP on a cost the duplicate may have paid, and printed a
    negative contributor under both entries."""
    pol = _pol(tmp_path)
    rows = [{"scope": "user", "server": "terse", "state": "router", "wraps": "gh, kb",
             "policy": pol},
            {"scope": "user", "server": "kb", "state": "folded-and-live",
             "wraps": "kb-server --stdio", "policy": pol, "ledger_identity": "kb",
             "ledger_identity_explicit": True}]
    tools = [("gh", 10, 10_000, 2_000)]
    base = primer_liability(rows, _agg(tools, []))
    liab = primer_liability(rows, _agg(tools, [("kb", 9_000)]))
    assert [(s["verdict"], s["saved_per_block"]) for s in liab["servers"]] == \
        [(s["verdict"], s["saved_per_block"]) for s in base["servers"]]
    assert all(c["saved_tokens"] >= 0
               for s in liab["servers"] for c in s["contributors"])


def test_an_unpaired_retrieve_moves_neither_the_headline_nor_any_row(tmp_path):
    """The headline and the per-entry rows are netted by ONE set, so they cannot disagree:
    a retrieve-only label (drop before the window) is out of both. Before the review fix
    the fleet total dropped by it while the entry read `never called`."""
    pol = _pol(tmp_path)
    rows = [{"scope": "user", "server": n, "state": "wrapped", "wraps": f"{n}-srv",
             "policy": pol} for n in ("gh", "kb")]
    liab = primer_liability(rows, _agg([("gh-srv", 10, 10_000, 2_000)],
                                       [("kb-srv", 5_000), ("gh-srv", 1_000)],
                                       untokenized=("kb-srv",)))
    assert liab["retrieve_tokens"] == 1_000
    # An unknown cost on an unpaired label is out of the net too, so it makes no bound.
    assert liab["retrieve_untokenized"] == 0
    assert liab["saved_tokens"] == 8_000 - 1_000
    assert sum(c["saved_tokens"] for s in liab["servers"] for c in s["contributors"]) \
        == liab["saved_tokens"]
