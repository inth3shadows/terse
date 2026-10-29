"""`terse tune --from-ledger` (#252 phase B): one deterministic verdict per live drop rule."""

from __future__ import annotations

import json

import pytest

from terse import cli, droptune
from terse.policy import Policy, Rule
from terse.stats import build_retrieve_record

PATH = "$text.code_blocks"


def _block(n: int = 1, **kw):
    return {"ts": 1, "server": "cg", "tool": "codegraph_explore", "decision": "compressed",
            "raw_chars": 900, "out_chars": 300, "raw_tokens": 200, "out_tokens": 60,
            "drops": {PATH: n}, **kw}


def _retrieves(k: int, index: int | None = None):
    return [build_retrieve_record("cg", "codegraph_explore", PATH, hit=True, payload="x" * 40,
                                  index=index) for _ in range(k)]


def _one(records, policy=None):
    (row,) = droptune.verdicts(records, policy)
    return row


def test_below_the_evidence_floor_no_verdict_is_reached():
    row = _one([_block(droptune.MIN_SPANS - 1)] + _retrieves(20))
    assert row["verdict"] == "insufficient" and row["change"] is None


def test_a_rule_fetched_back_often_is_retracted():
    row = _one([_block(50)] + _retrieves(20))
    assert row["verdict"] == "retract" and row["change"] == {"remove": "drop-to-retrieve"}


def test_a_rarely_fetched_rule_is_kept_and_widening_is_only_a_gated_candidate():
    row = _one([_block(100)] + _retrieves(2))
    assert row["verdict"] == "keep" and row["change"] is None
    assert "--drop-eval" in row["candidate"]


def test_between_the_bands_the_rule_is_watched():
    row = _one([_block(100)] + _retrieves(20))
    assert row["verdict"] == "watch" and row["change"] is None


def test_retrieves_clustered_on_early_blocks_raise_keep_first():
    records = [_block(100)] + _retrieves(8, index=0) + _retrieves(2, index=4)
    row = _one(records)
    assert row["verdict"] == "raise_keep_first" and row["change"] == {"keep_first": 1}


def test_keep_first_is_not_suggested_when_the_policy_already_covers_it():
    pol = Policy(rules=[Rule("codegraph_*", ("minify",),
                             fields={PATH: {"lossy": "drop-to-retrieve", "keep_first": 1}})])
    records = [_block(100)] + _retrieves(8, index=0) + _retrieves(2, index=4)
    row = _one(records, pol)
    assert row["keep_first"] == 1 and row["verdict"] == "watch"


def test_too_few_indexed_retrieves_never_move_keep_first():
    records = [_block(100)] + _retrieves(droptune.MIN_INDEXED_RETRIEVES - 1, index=0)
    assert _one(records)["verdict"] != "raise_keep_first"


def test_estimated_history_is_named_in_the_reason():
    records = [_block(0, diff_reason="text_dropped", drops=None) for _ in range(40)]
    row = _one(records)
    assert row["estimated"] == 40 and "predate drop recording" in row["reason"]


def test_smallest_covering_k():
    assert droptune._smallest_covering_k([0, 0, 0, 0, 3], 0.8) == 1
    assert droptune._smallest_covering_k([0, 1, 2, 3, 4], 0.8) == 4


def test_the_report_states_thresholds_and_that_nothing_is_applied():
    text = droptune.build_report(droptune.verdicts([_block(100)] + _retrieves(2)))
    assert "KEEP" in text and "min_spans=30" in text and "nothing is applied" in text


def _ledger(tmp_path, records):
    log = tmp_path / "stats.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    return log


def test_cli_prints_json_and_never_needs_a_corpus(tmp_path, capsys):
    log = _ledger(tmp_path, [_block(100)] + _retrieves(2))
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": []}))
    rc = cli.main(["tune", "--from-ledger", "--ledger", str(log), "--policy", str(pol),
                   "--corpus", str(tmp_path / "no-corpus-here"), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["thresholds"]["min_spans"] == droptune.MIN_SPANS
    assert [r["verdict"] for r in out["rules"]] == ["keep"]


@pytest.mark.parametrize("extra,msg", [(["--out", "x.json"], "--out is not supported"),
                                       (["--since", "bogus"], "tune --from-ledger")])
def test_cli_refuses_out_and_a_bad_window(tmp_path, capsys, extra, msg):
    log = _ledger(tmp_path, [_block(100)])
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": []}))
    rc = cli.main(["tune", "--from-ledger", "--ledger", str(log), "--policy", str(pol), *extra])
    assert rc == 2 and msg in capsys.readouterr().err


def test_cli_refuses_a_missing_ledger(tmp_path, capsys):
    rc = cli.main(["tune", "--from-ledger", "--ledger", str(tmp_path / "none.jsonl"),
                   "--policy", str(tmp_path / "p.json")])
    assert rc == 2 and "no ledger" in capsys.readouterr().err
