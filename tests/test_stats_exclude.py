"""`terse stats --exclude` leaves out rows that never belonged in the ledger (#573).

A scratch `terse proxy` run without `--no-stats` wrote 2,075 synthetic records into the
live ledger, ~90% of its raw tokens, and the only way out was hand-editing the JSONL.
"""

from __future__ import annotations

import json

from terse.cli import main
from terse.stats import exclude_records


def _rec(server, tool, raw=100, out=50):
    return {"ts": 1, "server": server, "tool": tool, "decision": "compressed",
            "raw_chars": raw, "out_chars": out, "raw_tokens": raw, "out_tokens": out}


REAL = [_rec("kb", "kb.read.search"), _rec("cg", "codegraph_explore")]
FAKE = [_rec("kb", "kb__pub", 10_000, 9_990), _rec("python3.14", "pub", 10_000, 9_990),
        _rec("cg", "cg__sec", 10_000, 9_990)]


def test_a_glob_matches_the_canonical_tool_under_any_server():
    assert exclude_records(REAL + FAKE, ["*/pub", "*/sec"]) == REAL


def test_a_server_glob_drops_that_servers_rows_including_toolless_ones():
    primer = {"ts": 1, "server": "kb", "kind": "primer"}
    kept = exclude_records([*REAL, primer], ["kb/*"])
    assert kept == [REAL[1]]
    assert exclude_records([*REAL, primer], ["kb/"]) == REAL   # the toolless row alone


def test_a_real_cross_peer_qualification_is_not_canonicalized_away():
    """`gh__pub` recorded under server `kb` is a genuine cross-peer name: it stays
    `kb/gh__pub`, so `*/pub` does not reach it."""
    rec = _rec("kb", "gh__pub")
    assert exclude_records([rec], ["*/pub"]) == [rec]
    assert exclude_records([rec], ["*/gh__pub"]) == []


def test_no_globs_returns_everything():
    assert exclude_records(REAL + FAKE, []) == REAL + FAKE


def test_the_report_totals_leave_the_excluded_rows_out(tmp_path, capsys):
    log = tmp_path / "stats.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in REAL + FAKE))
    assert main(["stats", "--log", str(log), "--json",
                 "--exclude", "*/pub", "--exclude", "*/sec"]) == 0
    total = json.loads(capsys.readouterr().out)["total"]
    assert total["raw_tokens"] == 200 and total["out_tokens"] == 100
    assert main(["stats", "--log", str(log), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["raw_tokens"] == 30_200
