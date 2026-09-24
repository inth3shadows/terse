"""#440 / #441 / #443 — the verdict screen says what is true about its own numbers.

#440: a saving from an aggregate without BOTH context sums was published as a context
saving with nothing saying otherwise. #441: `--recommend` replaces the ledger tables, so
the basis line never reached the screen that decides KEEP/UNWRAP. #443: a no-primer entry
driven negative by retrieve cost read `expanding` though nothing expanded.

End-to-end through `main(["stats", ...])` where a line reaches a screen; the unit being
right says nothing about the call sites that render it.
"""
from __future__ import annotations

import json
import time

import pytest

from terse.cli import main
from terse.stats import (
    append_stats,
    build_primer_section,
    build_recommend_section,
    build_record,
    primer_liability,
)

CONTEXT_LINE = "savings are on the CONTEXT basis"
WIRE_LINE = "on the WIRE basis"


def _pol(tmp_path, tiers=("minify", "tabularize")):
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "*"}, "tiers": list(tiers)}]}), encoding="utf-8")
    return str(pol)


def _liab(tmp_path, total, tools=None, retrieves=(), tiers=("minify", "tabularize")):
    """One wrapped `kb` entry over a hand-built agg whose `total` is taken VERBATIM, so a
    test controls exactly which context keys are present."""
    tools = tools if tools is not None else [{"server": "kb", "tool": "t", **total}]
    for t in tools:
        t.setdefault("blocks", 4)
        t.setdefault("tokenized", t["blocks"])
    agg = {"total": {"blocks": sum(t["blocks"] for t in tools), "raw_chars": 0,
                     "out_chars": 0, "untokenized": 0, **total},
           "decisions": {}, "diff_reasons": {}, "tools": tools,
           "retrieves": [{"server": "kb", "tool": "t", "path": "$.x", "calls": 1, "hits": 1,
                          "misses": 0, "bytes": 0, "tokens": tk, "untokenized": 0}
                         for tk in retrieves]}
    return primer_liability([{"scope": "user", "server": "kb", "state": "wrapped",
                              "wraps": "kb", "policy": _pol(tmp_path, tiers)}], agg)


def _screens(liab):
    return "\n".join(build_primer_section(liab)), "\n".join(build_recommend_section(liab))


def test_an_aggregate_without_context_sums_is_marked_wire_on_both_screens(tmp_path):
    liab = _liab(tmp_path, {"raw_tokens": 20_000, "out_tokens": 8_000})
    assert liab["saved_basis"] == "wire"
    assert liab["saved_tokens"] == 12_000     # the fallback is kept: zero would be worse
    for screen in _screens(liab):
        assert WIRE_LINE in screen and CONTEXT_LINE not in screen


def test_a_half_present_context_pair_falls_back_to_wire_not_to_a_negative_saving(tmp_path):
    """`context_raw - out_tokens` subtracts a wire figure that includes the text mirror from
    a context figure that excludes it — a large negative saving under a context label."""
    liab = _liab(tmp_path, {"raw_tokens": 20_000, "out_tokens": 8_000,
                            "context_raw_tokens": 5_000})
    assert liab["saved_basis"] == "wire"
    assert liab["saved_tokens"] == 12_000
    assert liab["servers"][0]["saved_per_block"] == pytest.approx(3_000)


def test_a_total_with_the_pair_over_rows_without_it_is_mixed(tmp_path):
    ctx = {"raw_tokens": 20_000, "out_tokens": 8_000, "context_raw_tokens": 20_000,
           "context_out_tokens": 8_000}
    liab = _liab(tmp_path, ctx,
                 tools=[{"server": "kb", "tool": "t", "raw_tokens": 20_000,
                         "out_tokens": 8_000}])
    assert liab["saved_basis"] == "mixed"
    for screen in _screens(liab):
        assert "PARTLY on the WIRE basis" in screen


def test_the_basis_line_is_gated_on_the_raw_sides_so_equal_savings_cannot_hide_it(
        tmp_path):
    """The cancel case: savings read identical on both bases (1,000 each) while the raw sides
    differ, because one peer's text block expanded and offset another's compression. A gate
    comparing the savings prints nothing here; the numbers on screen did move."""
    liab = _liab(tmp_path, {"raw_tokens": 2_000, "out_tokens": 1_000,
                            "context_raw_tokens": 1_500, "context_out_tokens": 500})
    assert liab["saved_basis"] == "context" and liab["context_differs_from_wire"] is True
    for screen in _screens(liab):
        assert CONTEXT_LINE in screen


def test_no_basis_line_when_no_row_moved_to_the_context_basis(tmp_path):
    liab = _liab(tmp_path, {"raw_tokens": 2_000, "out_tokens": 1_000,
                            "context_raw_tokens": 2_000, "context_out_tokens": 1_000})
    assert liab["saved_basis"] == "context" and liab["context_differs_from_wire"] is False
    for screen in _screens(liab):
        # "basis", not the exact new strings: the unconditional line this replaced
        # ("savings in this section are on the CONTEXT basis") must not come back either.
        assert "basis" not in screen


def test_an_older_blob_without_the_fields_claims_no_basis(tmp_path):
    liab = _liab(tmp_path, {"raw_tokens": 20_000, "out_tokens": 8_000})
    # Only `saved_basis` removed: an absent basis must not default to `context` even when
    # the differs flag (from a newer writer) says the bases moved.
    del liab["saved_basis"]
    liab["context_differs_from_wire"] = True
    for screen in _screens(liab):
        assert CONTEXT_LINE not in screen and WIRE_LINE not in screen


@pytest.fixture
def install(monkeypatch, tmp_path):
    import terse.install_mcp as install_mcp
    pol = _pol(tmp_path)
    monkeypatch.setattr(install_mcp, "scan_scopes",
                        lambda *a, **k: [{"scope": "user", "server": "kb",
                                          "state": "wrapped", "wraps": "kb-server",
                                          "policy": pol}])


@pytest.mark.parametrize("flags", [(), ("--recommend",)])
def test_a_rewritten_typed_field_puts_the_context_line_on_both_real_screens(
        install, tmp_path, capsys, flags):
    """#441's own reproduction, through the real command: `terse stats --recommend | grep
    -ci "context|wire"` returned 0 on a ledger whose rates had moved."""
    log = tmp_path / "stats.jsonl"
    typed = json.dumps({"rows": [{"id": i, "name": f"item {i}", "tags": ["a", "b"]}
                                 for i in range(40)]})
    text = json.dumps({"content": [{"type": "text", "text": typed}]})
    for _ in range(10):
        rec = build_record("kb-server", "t", text, text[: len(text) // 3], False,
                           structured=typed, structured_out=typed[: len(typed) // 3])
        rec["ts"] = int(time.time())
        append_stats(rec, log)
    capsys.readouterr()
    assert main(["stats", "--log", str(log), *flags]) == 0
    out = capsys.readouterr().out
    assert CONTEXT_LINE in out


def test_a_no_primer_entry_negative_only_through_retrieves_reads_retrieved_back(tmp_path):
    """#443. Every payload compressed (8,000 saved); the model fetched back 9,000. Nothing
    expanded, so `expanding` named a cause that did not happen."""
    liab = _liab(tmp_path, {"raw_tokens": 10_000, "out_tokens": 2_000,
                            "context_raw_tokens": 10_000, "context_out_tokens": 2_000},
                 retrieves=(9_000,), tiers=())
    srv = liab["servers"][0]
    assert srv["break_even_verdict"] == "no primer" and srv["retrieve_tokens"] == 9_000
    assert (srv["verdict"], srv["verdict_reason"]) == ("UNWRAP", "retrieved back")


def test_a_no_primer_entry_that_expands_is_still_expanding_with_retrieves_on_top(tmp_path):
    liab = _liab(tmp_path, {"raw_tokens": 2_000, "out_tokens": 2_400,
                            "context_raw_tokens": 2_000, "context_out_tokens": 2_400},
                 retrieves=(100,), tiers=())
    assert liab["servers"][0]["verdict_reason"] == "expanding"


def test_a_bool_is_not_a_context_sum(tmp_path):
    liab = _liab(tmp_path, {"raw_tokens": 20_000, "out_tokens": 8_000,
                            "context_raw_tokens": True, "context_out_tokens": False})
    assert liab["saved_basis"] == "wire" and liab["saved_tokens"] == 12_000


def test_retrieved_back_at_exactly_zero_gross_despite_float_rounding(tmp_path):
    """Review finding: net -1,019 over 7 blocks makes `rate * tok` -1019.0000000000001, so a
    passthrough entry (saved exactly 0) with one 1,019-tok retrieve read `expanding`."""
    tools = [{"server": "kb", "tool": "t", "blocks": 9, "tokenized": 7, "raw_tokens": 2_000,
              "out_tokens": 2_000, "context_raw_tokens": 2_000, "context_out_tokens": 2_000}]
    liab = _liab(tmp_path, {"raw_tokens": 2_000, "out_tokens": 2_000,
                            "context_raw_tokens": 2_000, "context_out_tokens": 2_000},
                 tools=tools, retrieves=(1_019,), tiers=())
    assert liab["servers"][0]["verdict_reason"] == "retrieved back"


def test_an_entrys_retrieve_tokens_are_its_own_not_the_fleets(tmp_path):
    """Pattern 1: per-server `retrieve_tokens` must cover the SAME labels as that entry's
    rate. Summed fleet-wide, an entry that really expanded borrows another's retrieves and
    reads `retrieved back`."""
    pol = _pol(tmp_path, tiers=())
    tools = [{"server": "kb", "tool": "t", "blocks": 4, "tokenized": 4,
              "raw_tokens": 2_000, "out_tokens": 2_500, "context_raw_tokens": 2_000,
              "context_out_tokens": 2_500},
             {"server": "gh", "tool": "t", "blocks": 4, "tokenized": 4,
              "raw_tokens": 10_000, "out_tokens": 2_000, "context_raw_tokens": 10_000,
              "context_out_tokens": 2_000}]
    total = {k: sum(t[k] for t in tools) for k in
             ("blocks", "raw_tokens", "out_tokens", "context_raw_tokens",
              "context_out_tokens")}
    agg = {"total": {**total, "raw_chars": 0, "out_chars": 0, "untokenized": 0},
           "decisions": {}, "diff_reasons": {}, "tools": tools,
           "retrieves": [{"server": "gh", "tool": "t", "path": "$.x", "calls": 1,
                          "hits": 1, "misses": 0, "bytes": 0, "tokens": 9_000,
                          "untokenized": 0}]}
    liab = primer_liability([{"scope": "user", "server": n, "state": "wrapped",
                              "wraps": n, "policy": pol} for n in ("kb", "gh")], agg)
    by = {s["server"]: s for s in liab["servers"]}
    assert (by["kb"]["retrieve_tokens"], by["kb"]["verdict_reason"]) == (0, "expanding")
    assert (by["gh"]["retrieve_tokens"], by["gh"]["verdict_reason"]) == (
        9_000, "retrieved back")
