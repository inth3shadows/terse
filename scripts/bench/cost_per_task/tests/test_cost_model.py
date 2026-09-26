"""Offline tests for cost_model.py: TTL-aware cache-write pricing, subagent
transcript discovery, first-turn cache stats, and the USD cost-gap check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cost_model as cm

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))  # scripts/bench, for ab_session
import importlib.util

_spec = importlib.util.spec_from_file_location("ab_session", HERE.parent / "ab_session.py")
ab_session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab_session)


def _write(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _assistant(req_id, msg_id, usage, model="claude-haiku-4-5", version="2.1.283"):
    return {"type": "assistant", "requestId": req_id, "version": version,
            "message": {"id": msg_id, "model": model, "usage": usage}}


def test_ttl_split_prices_5m_and_1h_differently(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [
        _assistant("r1", "m1", {
            "input_tokens": 100, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1000, "output_tokens": 10,
            "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0},
        }),
    ])
    result = cm.compute_ttl_weighted([path], ab_session)
    assert result["cache_write_5m_tokens"] == 1000
    assert result["cache_write_1h_tokens"] == 0
    expected = (100 * ab_session.W_INPUT + 1000 * cm.W_CACHE_WRITE_5M + 10 * ab_session.W_OUTPUT)
    assert result["weighted_cost"] == expected


def test_1h_write_costs_more_than_5m_for_the_same_token_count(tmp_path):
    p5 = tmp_path / "five.jsonl"
    p1 = tmp_path / "one.jsonl"
    _write(p5, [_assistant("r1", "m1", {
        "input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 1000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0}})])
    _write(p1, [_assistant("r1", "m1", {
        "input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 1000,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000}})])
    r5 = cm.compute_ttl_weighted([p5], ab_session)
    r1 = cm.compute_ttl_weighted([p1], ab_session)
    assert r1["weighted_cost"] > r5["weighted_cost"]
    assert r5["weighted_cost"] == 1000 * cm.W_CACHE_WRITE_5M
    assert r1["weighted_cost"] == 1000 * cm.W_CACHE_WRITE_1H


def test_fallback_to_5m_rate_when_ttl_split_absent(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_assistant("r1", "m1", {
        "input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 500})])  # no "cache_creation" key at all
    result = cm.compute_ttl_weighted([path], ab_session)
    assert result["cache_write_5m_tokens"] == 500
    assert result["cache_write_1h_tokens"] == 0
    assert result["weighted_cost"] == 500 * cm.W_CACHE_WRITE_FALLBACK


def test_dedups_repeated_requestid_message_id(tmp_path):
    path = tmp_path / "t.jsonl"
    usage = {"input_tokens": 10, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 1}
    _write(path, [_assistant("r1", "m1", usage), _assistant("r1", "m1", usage)])
    result = cm.compute_ttl_weighted([path], ab_session)
    assert result["turns"] == 1
    assert result["input_tokens"] == 10


def test_first_turn_cache_stats_are_the_first_turn_only(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [
        _assistant("r1", "m1", {"input_tokens": 1, "cache_read_input_tokens": 5,
                                 "cache_creation_input_tokens": 7, "output_tokens": 1}),
        _assistant("r2", "m2", {"input_tokens": 1, "cache_read_input_tokens": 999,
                                 "cache_creation_input_tokens": 999, "output_tokens": 1}),
    ])
    result = cm.compute_ttl_weighted([path], ab_session)
    assert result["first_turn_cache_read_tokens"] == 5
    assert result["first_turn_cache_write_tokens"] == 7


def test_subagent_transcripts_included_in_totals(tmp_path):
    main = tmp_path / "s1.jsonl"
    sub_dir = tmp_path / "s1" / "subagents"
    sub_dir.mkdir(parents=True)
    sub = sub_dir / "agent-abc.jsonl"
    usage = {"input_tokens": 100, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 0}
    _write(main, [_assistant("r1", "m1", usage)])
    _write(sub, [_assistant("r2", "m2", usage)])

    found = cm.find_subagent_transcripts(main)
    assert found == [sub]
    result = cm.compute_ttl_weighted([main] + found, ab_session)
    assert result["input_tokens"] == 200  # both transcripts counted


def test_find_subagent_transcripts_empty_when_no_dir(tmp_path):
    main = tmp_path / "s1.jsonl"
    main.write_text("")
    assert cm.find_subagent_transcripts(main) == []


def test_estimate_usd_cost_known_model(tmp_path):
    counts = {"input_tokens": 1_000_000, "cache_read_tokens": 0,
              "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0, "output_tokens": 0}
    assert cm.estimate_usd_cost("claude-haiku-4-5", counts) == 1.00


def test_estimate_usd_cost_unknown_model_raises_instead_of_returning_none():
    # Plan Partial #3: a silently-None cost for an unpriced model makes
    # cost_gap unjudgeable on every row for that model without ever saying
    # so -- fail loudly instead so the price table gets fixed.
    counts = {"input_tokens": 1, "cache_read_tokens": 0,
              "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0, "output_tokens": 0}
    try:
        cm.estimate_usd_cost("some-unknown-model", counts)
    except ValueError as e:
        assert "some-unknown-model" in str(e)
    else:
        raise AssertionError("expected ValueError for a model with no price entry")


def test_estimate_usd_cost_normalizes_dated_model_id_to_family():
    # "claude-sonnet-5-20260115" isn't a literal MODEL_USD_PER_MTOK key, but
    # it must price identically to "claude-sonnet-5" rather than raising.
    counts = {"input_tokens": 1_000_000, "cache_read_tokens": 0,
              "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0, "output_tokens": 0}
    dated = cm.estimate_usd_cost("claude-sonnet-5-20260115", counts)
    bare = cm.estimate_usd_cost("claude-sonnet-5", counts)
    assert dated == bare == 2.00


def test_normalize_model_family_strips_date_suffix():
    assert cm.normalize_model_family("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert cm.normalize_model_family("claude-opus-5-5-20260101") == "claude-opus-5-5"


def test_normalize_model_family_leaves_bare_id_unchanged():
    assert cm.normalize_model_family("claude-sonnet-5") == "claude-sonnet-5"


def test_compute_ttl_weighted_uses_opus_cache_read_weight_not_flat_point_one(tmp_path):
    # Plan Partial #3: Opus 5.5's real cache-read price is 0.05x of input
    # ($0.20 / $4.00), not the flat 0.1x ab_session.W_CACHE_READ uses for
    # every other current model. The OLD behavior (a single flat weight
    # applied regardless of model) would compute this as
    # 1000 * ab_session.W_CACHE_READ == 100, which this test must NOT see.
    path = tmp_path / "t.jsonl"
    _write(path, [_assistant("r1", "m1", {
        "input_tokens": 0, "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 0, "output_tokens": 0},
        model="claude-opus-5-5")])
    result = cm.compute_ttl_weighted([path], ab_session)
    assert result["weighted_cost"] == 1000 * 0.05
    assert result["weighted_cost"] != 1000 * ab_session.W_CACHE_READ


def test_compute_ttl_weighted_mixed_model_transcript_weights_each_record_by_its_own_model(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [
        _assistant("r1", "m1", {"input_tokens": 0, "cache_read_input_tokens": 1000,
                                 "cache_creation_input_tokens": 0, "output_tokens": 0},
                   model="claude-opus-5-5"),
        _assistant("r2", "m2", {"input_tokens": 0, "cache_read_input_tokens": 1000,
                                 "cache_creation_input_tokens": 0, "output_tokens": 0},
                   model="claude-haiku-4-5"),
    ])
    result = cm.compute_ttl_weighted([path], ab_session)
    # opus record at 0.05x + haiku record at 0.1x -- NOT both at either rate.
    assert result["weighted_cost"] == 1000 * 0.05 + 1000 * 0.10


def test_is_real_turn_usage_false_for_the_cli_zero_usage_sentinel():
    fake = {"input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert cm.is_real_turn_usage(fake) is False
    assert cm.is_real_turn_usage(None) is False
    assert cm.is_real_turn_usage({}) is False


def test_is_real_turn_usage_true_for_any_nonzero_field():
    assert cm.is_real_turn_usage({"input_tokens": 0, "output_tokens": 1,
                                   "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 0}) is True


def test_count_real_turns_excludes_synthetic_zero_usage_record(tmp_path):
    # A run whose ONLY assistant record is the CLI's injected error notice
    # (usage is the all-zero sentinel) must count as ZERO real turns, not
    # one -- the old behavior (ab_session.SessionStats.turns, which counts
    # any assistant record with a truthy usage dict) would see this as 1
    # turn and wrongly let a genuine infra failure through the
    # nonzero-exit-no-turn check.
    path = tmp_path / "t.jsonl"
    fake_usage = {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    _write(path, [_assistant("r1", "m1", fake_usage)])
    assert cm.count_real_turns([path]) == 0


def test_count_real_turns_counts_real_records_only(tmp_path):
    path = tmp_path / "t.jsonl"
    fake_usage = {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    real_usage = {"input_tokens": 10, "output_tokens": 1,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    _write(path, [_assistant("r1", "m1", fake_usage), _assistant("r2", "m2", real_usage)])
    assert cm.count_real_turns([path]) == 1


def test_cost_gap_flag_true_when_over_threshold():
    assert cm.cost_gap_flag(1.10, 1.00) is True   # 10% off


def test_cost_gap_flag_false_when_within_threshold():
    assert cm.cost_gap_flag(1.02, 1.00) is False  # 2% off


def test_cost_gap_flag_none_when_either_side_missing():
    assert cm.cost_gap_flag(None, 1.00) is None
    assert cm.cost_gap_flag(1.00, None) is None
