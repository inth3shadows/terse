"""`terse stats --json` is a public surface, so its shape is pinned field by field.

USAGE calls it "the raw aggregate, for scripts" and `verify --json`'s own docs cite it as
the machine-readable precedent to mirror. That makes it a contract with people who are not
in this repo — and it carried **37 fields across four nested shapes** with exactly two
assertions on any of them (`total.blocks` and `total.raw_tokens`, in
`test_cli.py::test_stats_cmd_json_output`).

The cost of that showed up immediately. In one day the liability blob gained
`session_once_tokens`, `session_covered`, `free`, `uncertain` and a per-server `cadence`,
the tool rows gained `encoded`, and `per_turn_tokens` was **redefined** from "every wrapped
server" to "the recurring ones only" — a consumer reading it would have seen the number drop
with nothing failing anywhere. Prose in the CHANGELOG is the only thing that warned them.

The manifests below are deliberately EXACT, not "at least these". A removal or a rename is a
break, and an addition is a decision worth making on purpose — the failure message says so,
because a contract test that silently tolerates growth stops being a contract. Types are
asserted alongside names: a consumer reading `per_turn_tokens` as an int must not one day
get a string, which a presence-only check would wave through.

The whole thing is driven through `main(["stats", ...])` rather than by calling `aggregate`
and `primer_liability` directly. The composition is part of the contract — `cli` merges the
two with `{**agg, "primer_liability": liability}`, so the top-level key set is not something
either function alone can be asked about.
"""

from __future__ import annotations

import json

import pytest

from terse.cli import main
from terse.stats import append_stats

# --- the contract ---------------------------------------------------------------------

TOP_LEVEL = {"total", "decisions", "diff_reasons", "tools", "versions", "primer_liability"}

TOTAL = {"blocks", "raw_chars", "out_chars", "raw_tokens", "out_tokens",
         "untokenized", "unversioned"}

# `encoded` is the newest and the one most likely to be dropped by someone tidying
# `aggregate`: `primer_liability` reads it to tell a server that merely was CALLED from one
# whose lazy primer could actually have fired, so losing it silently reverts that fix.
TOOL_ROW = {"server", "tool", "blocks", "tokenized", "encoded", "diffs",
            "raw_chars", "out_chars", "raw_tokens", "out_tokens"}

VERSION_ROW = {"blocks", "tokenized", "raw_tokens", "out_tokens"}

LIABILITY = {"servers", "per_turn_tokens", "session_once_tokens", "unresolved",
             "idle", "free", "uncertain", "saved_tokens", "turns_covered",
             "session_covered"}

LIABILITY_SERVER = {"server", "scope", "state", "primer_tokens", "ledger_labels",
                    "blocks", "tokenized_blocks", "cadence",
                    "saved_per_block", "blocks_to_break_even", "break_even_verdict"}

# name -> the types a consumer may receive. `None` is listed explicitly wherever it is a
# real answer, because this module's whole discipline is that unknown is not zero — a
# consumer that treats `blocks: null` as 0 draws the conclusion the None exists to prevent.
TYPES: dict[str, tuple[type | None, ...]] = {
    "blocks": (int, type(None)), "tokenized": (int,), "encoded": (int,),
    "diffs": (int,), "raw_chars": (int,), "out_chars": (int,),
    "raw_tokens": (int,), "out_tokens": (int,),
    "untokenized": (int,), "unversioned": (int,),
    "server": (str,), "tool": (str,), "scope": (str, type(None)),
    "state": (str, type(None)), "primer_tokens": (int, type(None)),
    "ledger_labels": (list,), "tokenized_blocks": (int, type(None)),
    "cadence": (str,), "saved_per_block": (float, int, type(None)),
    "blocks_to_break_even": (float, int, type(None)),
    "break_even_verdict": (str, type(None)),
    "per_turn_tokens": (int,), "session_once_tokens": (int,), "unresolved": (int,),
    "idle": (list,), "free": (list,), "uncertain": (list,), "saved_tokens": (int,),
    "turns_covered": (float, int, type(None)),
    "session_covered": (float, int, type(None)),
    "servers": (list,),
}

_ADD_ON_PURPOSE = ("Fields here are a public contract (USAGE: \"the raw aggregate, for "
                   "scripts\"). If this addition/removal is intentional, update the "
                   "manifest AND note it in the CHANGELOG — a consumer's script has no "
                   "other warning.")


def _scan_rows():
    return [{"scope": "user", "server": "kb", "state": "wrapped",
             "wraps": "kb-server --stdio", "policy": None},
            {"scope": "user", "server": "terse", "state": "router",
             "wraps": "kb, runecho", "policy": None},
            # No recoverable ledger label — the row that exercises every `None` the
            # contract promises a consumer may receive.
            {"scope": "project", "server": "mystery", "state": "wrapped",
             "wraps": "", "policy": None}]


@pytest.fixture
def stats_json(tmp_path, capsys, monkeypatch):
    """`terse stats --json` over a fixed ledger and a fixed install.

    `scan_scopes` is monkeypatched because it reads the REAL user config: without this the
    liability half of the output — over half the contract — would vary by machine, and the
    one existing test of this command has that dependency today."""
    import terse.install_mcp as install_mcp

    def run(records, scan=None):
        log = tmp_path / "stats.jsonl"
        for rec in records:
            append_stats(rec, log)
        monkeypatch.setattr(install_mcp, "scan_scopes",
                            lambda *a, **k: _scan_rows() if scan is None else scan)
        capsys.readouterr()
        assert main(["stats", "--log", str(log), "--json"]) == 0
        return json.loads(capsys.readouterr().out)
    return run


def _rec(**kw):
    base = {"ts": 1, "version": "9.9.9", "server": "kb-server", "tool": "kb.read.search",
            "decision": "compressed", "raw_chars": 400, "out_chars": 40,
            "raw_tokens": 100, "out_tokens": 10}
    return {**base, **kw}


def _check_types(where: str, obj: dict) -> None:
    for key, value in obj.items():
        allowed = TYPES.get(key)
        if allowed is None:
            continue
        # `bool` is an `int` subclass; a flag arriving where a count belongs would pass a
        # plain isinstance check and then be summed as 0/1 by a consumer.
        assert not isinstance(value, bool), f"{where}.{key} is a bool"
        assert isinstance(value, allowed), (
            f"{where}.{key} is {type(value).__name__} ({value!r}), contract says "
            f"{[t.__name__ for t in allowed]}")


def test_the_top_level_shape_is_exactly_these_keys(stats_json):
    out = stats_json([_rec()])
    assert set(out) == TOP_LEVEL, _ADD_ON_PURPOSE


def test_the_total_and_tool_rows_are_exactly_these_keys(stats_json):
    out = stats_json([_rec(), _rec(decision="passthrough", out_chars=400, out_tokens=100)])
    assert set(out["total"]) == TOTAL, _ADD_ON_PURPOSE
    assert out["tools"], "a ledger with records must produce tool rows"
    for row in out["tools"]:
        assert set(row) == TOOL_ROW, _ADD_ON_PURPOSE
        _check_types("tools[]", row)
    _check_types("total", out["total"])


def test_the_version_rows_are_exactly_these_keys(stats_json):
    out = stats_json([_rec()])
    assert out["versions"], "a versioned record must produce a version row"
    for row in out["versions"].values():
        assert set(row) == VERSION_ROW, _ADD_ON_PURPOSE
        _check_types("versions[]", row)


def test_the_liability_blob_is_exactly_these_keys(stats_json):
    out = stats_json([_rec()])
    liab = out["primer_liability"]
    assert set(liab) == LIABILITY, _ADD_ON_PURPOSE
    _check_types("primer_liability", liab)
    assert liab["servers"], "the fixture installs three primer-paying entries"
    for row in liab["servers"]:
        assert set(row) == LIABILITY_SERVER, _ADD_ON_PURPOSE
        _check_types("primer_liability.servers[]", row)


def test_decisions_and_diff_reasons_are_string_keyed_counters(stats_json):
    """Free-form by design — the key set is whatever decisions the ledger saw — so the
    contract is the SHAPE, not the names. A consumer sums the values."""
    out = stats_json([_rec(), _rec(decision="passthrough"),
                      _rec(decision="diff", diff_reason="emitted")])
    for name in ("decisions", "diff_reasons"):
        assert isinstance(out[name], dict)
        assert all(isinstance(k, str) for k in out[name]), name
        assert all(isinstance(v, int) and not isinstance(v, bool)
                   for v in out[name].values()), name
    assert out["decisions"]["passthrough"] == 1


def test_an_unresolvable_row_returns_null_not_zero(stats_json):
    """The distinction this module exists to keep, asserted through the wire rather than in
    Python: a consumer that reads `blocks: 0` concludes "installed and never called", which
    is an accusation. `null` means "no ledger label was recoverable, so we never found the
    rows to ask" — and JSON has no way to say that except `null`."""
    out = stats_json([_rec()])
    row = next(s for s in out["primer_liability"]["servers"] if s["server"] == "mystery")
    assert row["blocks"] is None and row["tokenized_blocks"] is None
    assert row["break_even_verdict"] == "no ledger label"
    assert row["cadence"] == "once/session (?)"
    assert "mystery" in out["primer_liability"]["uncertain"]
    assert "mystery" not in out["primer_liability"]["free"]


def test_primer_liability_is_null_rather_than_absent_when_it_cannot_be_sized(
        tmp_path, capsys, monkeypatch):
    """`cli` merges `{**agg, "primer_liability": liability}` and `liability` is None when
    sizing raised, so the key is present with a null. Pinned because the two failure shapes
    are different code for a consumer: `.get("primer_liability")` returning None either way
    hides that the report ran and the liability alone failed."""
    import terse.install_mcp as install_mcp

    def boom(*a, **k):
        raise RuntimeError("malformed config")

    log = tmp_path / "stats.jsonl"
    append_stats(_rec(), log)
    monkeypatch.setattr(install_mcp, "scan_scopes", boom)
    capsys.readouterr()
    assert main(["stats", "--log", str(log), "--json"]) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert "primer_liability" in out and out["primer_liability"] is None
    assert set(out) == TOP_LEVEL, _ADD_ON_PURPOSE
    assert "could not size the primer liability" in captured.err


def test_an_empty_window_still_emits_the_whole_shape(stats_json):
    """A `--since` that filters everything out, or a fresh install, must not hand a script a
    half-built document — the fields it reads have to exist so it can report zero rather
    than crash on a KeyError."""
    out = stats_json([_rec(ts=1)], scan=[])
    assert set(out) == TOP_LEVEL, _ADD_ON_PURPOSE
    assert set(out["total"]) == TOTAL, _ADD_ON_PURPOSE
    assert out["primer_liability"]["servers"] == []
    assert out["primer_liability"]["per_turn_tokens"] == 0
    assert out["primer_liability"]["turns_covered"] is None
    assert out["primer_liability"]["session_covered"] is None


def test_the_encoded_fallback_holds_on_a_ledger_that_straddles_the_counter(stats_json):
    """The gap left open when `encoded` shipped: its fallback was pinned only at the
    all-or-nothing extremes. Here ONE server's rows straddle the change — some written
    before the counter existed, some after — and the answer must be the conservative one.

    `aggregate` derives `encoded` from each record's `decision`, and a pre-counter record
    still HAS a decision, so a straddling ledger is counted correctly rather than falling
    back at all. The fallback is for a hand-rolled `agg` dict with no `encoded` key, which
    is a different thing and is covered in `test_primer_liability.py`."""
    old = _rec(decision="passthrough", out_chars=400, out_tokens=100)
    old.pop("version")                       # pre-version, pre-counter era
    out = stats_json([old, old, _rec()])     # ...plus one real compressed block
    row = next(r for r in out["tools"] if r["server"] == "kb-server")
    assert row["blocks"] == 3 and row["encoded"] == 1
    liab = next(s for s in out["primer_liability"]["servers"] if s["server"] == "kb")
    assert liab["cadence"] == "once/session"     # one encoded block is enough to bill
    assert out["primer_liability"]["session_once_tokens"] > 0


def test_the_whole_document_survives_a_json_round_trip(stats_json):
    """`json.dumps` in `cli` is the only thing between these objects and a script. A set, a
    tuple or a Path leaking into the blob raises there and takes the whole command down —
    `cadence` and the `free`/`uncertain`/`idle` lists are all built from comprehensions over
    internal state, and `_silent_peers`-style tuple returns are exactly the shape that
    serializes fine as a list and then compares unequal on the way back."""
    out = stats_json([_rec()])
    assert json.loads(json.dumps(out)) == out
    for key in ("idle", "free", "uncertain"):
        assert isinstance(out["primer_liability"][key], list)


def test_a_record_whose_decision_cannot_be_read_over_bills_rather_than_under_bills(
        stats_json):
    """`aggregate` tolerates a record with no `decision`, calling it `"unknown"`. That block
    counts toward `blocks`, and the question is whether it counts toward `encoded`.

    Not counting it under-bills: a server whose every readable block was unknown would land
    in `free` and the report would tell the operator it "costs nothing at all". Counting it
    over-bills, which is the direction `_cadence` already argues is the safe one — so
    `encoded` is derived by EXCLUDING the two decisions that prove no marker shipped, rather
    than by including the two that may have. Identical for any record terse wrote
    (`classify_decision` returns exactly one of four; 0 of 2,115 live records lack the
    field), so this only ever decides a hand-written or third-party line."""
    mystery = _rec()
    mystery.pop("decision")
    out = stats_json([mystery, mystery])
    row = next(r for r in out["tools"] if r["server"] == "kb-server")
    assert row["blocks"] == 2 and row["encoded"] == 2
    assert out["decisions"] == {"unknown": 2}
    liab = next(s for s in out["primer_liability"]["servers"] if s["server"] == "kb")
    assert liab["cadence"] == "once/session"        # billed, not filed as free
    assert out["primer_liability"]["free"] == []


def test_a_passthrough_only_server_is_still_read_as_never_encoding(stats_json):
    """The other side of the same predicate — inverting it must not turn every block into an
    encoded one. `passthrough` and `unchanged` both PROVE no terse marker reached the wire,
    so a server that only ever produced those never paid its primer."""
    out = stats_json([_rec(decision="passthrough", out_chars=400, out_tokens=100),
                      _rec(decision="unchanged", out_chars=400, out_tokens=100)])
    row = next(r for r in out["tools"] if r["server"] == "kb-server")
    assert row["blocks"] == 2 and row["encoded"] == 0
    liab = next(s for s in out["primer_liability"]["servers"] if s["server"] == "kb")
    assert liab["cadence"] == "once/session (unpaid)"
    assert out["primer_liability"]["free"] == ["kb"]
