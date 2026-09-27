"""Offline tests for analysis.py's cost-per-success math and paired
bootstrap CI, using synthetic JSONL-row dicts (no real claude -p output)."""
from __future__ import annotations

import sys
from pathlib import Path

# The flat cost_per_task scripts are importable by bare module name (`arms`, `runner`,
# ...). Done here, not in a conftest.py: a second `conftest` module shadowed
# tests/conftest.py in a whole-repo run (`from conftest import ...` there then failed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis as an


def _row(task, arm, rep, success, cost, share=0.5, model="haiku"):
    return {"model": model, "task_id": task, "arm": arm, "rep": rep,
            "success": success, "weighted_cost": cost, "mcp_share_approx": share}


def test_task_arm_stats_charges_failed_reps_against_successes():
    runs = [
        _row("t1", "B", 1, True, 100),
        _row("t1", "B", 2, False, 50),   # failed rep still spent tokens
        _row("t1", "B", 3, True, 100),
    ]
    s = an.task_arm_stats(runs)
    assert s["n"] == 3
    assert s["successes"] == 2
    assert s["total_cost"] == 250
    assert s["cost_per_success"] == 125  # 250 / 2, not 200 / 2


def test_task_arm_stats_zero_successes_is_none_not_infinite():
    runs = [_row("t1", "A", 1, False, 100)]
    s = an.task_arm_stats(runs)
    assert s["successes"] == 0
    assert s["cost_per_success"] is None


def test_task_arm_stats_unaddressable_flag_thresholds_on_mean_share():
    low = [_row("t1", "B", 1, True, 100, share=0.01)]
    high = [_row("t1", "B", 1, True, 100, share=0.5)]
    boundary = [_row("t1", "B", 1, True, 100, share=an.UNADDRESSABLE_THRESHOLD)]
    assert an.task_arm_stats(low)["unaddressable"] is True
    assert an.task_arm_stats(high)["unaddressable"] is False
    assert an.task_arm_stats(boundary)["unaddressable"] is False  # strict <, not <=


def test_task_arm_stats_raises_on_none_cost_for_a_non_infra_row():
    # A row with no infra_error flag but a None weighted_cost (e.g. a bug
    # upstream that lost the transcript) must be REFUSED, never silently
    # treated as a free (0.0-cost) run -- that would understate cost/success.
    runs = [{"task_id": "t1", "arm": "B", "rep": 1, "success": True,
             "weighted_cost": None, "mcp_share_approx": 0.5}]
    try:
        an.task_arm_stats(runs)
    except ValueError as e:
        assert "weighted_cost" in str(e)
    else:
        raise AssertionError("expected ValueError for a None-cost non-infra row")


def test_partition_infra_rows_excludes_infra_and_keeps_the_rest():
    rows = [
        _row("t1", "B", 1, True, 100),
        {**_row("t1", "B", 2, False, None), "infra_error": "timeout"},
    ]
    analyzable, infra = an.partition_infra_rows(rows)
    assert len(analyzable) == 1 and analyzable[0]["rep"] == 1
    assert len(infra) == 1 and infra[0]["infra_error"] == "timeout"


def test_analyze_excludes_infra_rows_and_never_raises_on_their_none_cost():
    rows = [
        _row("t1", "B", 1, True, 100),
        {**_row("t1", "B", 2, False, None), "infra_error": "rate_limit"},
    ]
    report = an.analyze(rows)  # must not raise -- the infra row is excluded first
    assert report["haiku"]["arms"]["B"]["total_cost"] == 100


def test_arm_aggregate_pools_across_tasks():
    by_task = {
        "t1": {"B": {"total_cost": 100, "successes": 1}},
        "t2": {"B": {"total_cost": 300, "successes": 2}},
    }
    agg = an.arm_aggregate(by_task, "B")
    assert agg["total_cost"] == 400
    assert agg["successes"] == 3
    assert agg["cost_per_success"] == 400 / 3


def test_arm_aggregate_zero_successes_is_none():
    by_task = {"t1": {"B": {"total_cost": 100, "successes": 0}}}
    assert an.arm_aggregate(by_task, "B")["cost_per_success"] is None


def _by_task_from_rows(rows):
    grouped = an._group_by_model_task_arm(rows)
    model = next(iter(grouped))
    return {t: {arm: an.task_arm_stats(runs) for arm, runs in arm_runs.items()}
            for t, arm_runs in grouped[model].items()}


def test_paired_bootstrap_ci_excludes_zero_when_arms_clearly_differ():
    rows = []
    for i in range(8):
        task = f"t{i}"
        for rep in range(1, 4):
            rows.append(_row(task, "C", rep, True, 50))    # cheap, always succeeds
            rows.append(_row(task, "B", rep, True, 150))   # 3x more expensive
    by_task = _by_task_from_rows(rows)
    result = an.paired_bootstrap_ci(by_task, "B", "C", n_boot=1000, seed=1)
    assert result["ci"] is not None
    assert result["excludes_zero"] is True
    assert result["ci"][0] > 0          # B costs strictly more than C in every draw
    assert result["observed"] == 100.0  # 150 - 50


def test_paired_bootstrap_ci_includes_zero_when_arms_are_identical():
    rows = []
    for i in range(8):
        task = f"t{i}"
        for rep in range(1, 4):
            rows.append(_row(task, "C", rep, True, 100))
            rows.append(_row(task, "B", rep, True, 100))
    by_task = _by_task_from_rows(rows)
    result = an.paired_bootstrap_ci(by_task, "B", "C", n_boot=1000, seed=1)
    assert result["ci"] is not None
    assert result["excludes_zero"] is False
    assert result["ci"][0] <= 0 <= result["ci"][1]
    assert result["observed"] == 0.0


def test_paired_bootstrap_ci_reports_reason_on_no_shared_tasks():
    by_task = {"t1": {"B": {"total_cost": 100, "successes": 1}}}  # no "C" anywhere
    result = an.paired_bootstrap_ci(by_task, "B", "C", n_boot=100, seed=1)
    assert result["ci"] is None
    assert "no task has both" in result["reason"]


def test_paired_bootstrap_ci_is_truly_paired_identical_arms_give_exact_zero_ci():
    # Tasks have deliberately DIFFERENT per-task cost so that if the bootstrap
    # drew separate resample picks for each arm (breaking the pairing), the
    # two arms' resampled aggregates would land on different task multisets
    # and produce a nonzero diff on most draws. B and C are IDENTICAL
    # per-task, so a properly PAIRED draw (same picks for both arms) must
    # give a diff of EXACTLY zero on every single draw -- CI = [0, 0].
    rows = []
    costs = [50, 100, 150, 200, 300, 10, 700, 25, 900, 5]
    for i, c in enumerate(costs):
        task = f"t{i}"
        rows.append(_row(task, "B", 1, True, c))
        rows.append(_row(task, "C", 1, True, c))  # identical to B
    by_task = _by_task_from_rows(rows)
    result = an.paired_bootstrap_ci(by_task, "B", "C", n_boot=500, seed=3)
    assert result["ci"] == (0.0, 0.0)
    assert result["observed"] == 0.0


def test_paired_bootstrap_ci_reports_insufficient_data_with_rare_successes():
    # Only ONE task, and it never succeeds in arm B -- every bootstrap draw
    # is unusable, so the CI must say so rather than fabricate one from n=0.
    by_task = {"t1": {"B": {"total_cost": 100, "successes": 0},
                       "C": {"total_cost": 50, "successes": 1}}}
    result = an.paired_bootstrap_ci(by_task, "B", "C", n_boot=200, seed=1)
    assert result["ci"] is None
    assert "too few" in result["reason"]


def test_task_unaddressable_from_b_uses_bs_share_not_arm_as_own():
    # Arm A structurally reads ~0 MCP share (no MCP servers) -- it must NOT
    # be the source of the unaddressable judgement, or every task would
    # read unaddressable regardless of B/C's real MCP usage.
    stats = {"A": {"mean_mcp_share": 0.0}, "B": {"mean_mcp_share": 0.8},
             "C": {"mean_mcp_share": 0.02}}
    assert an.task_unaddressable_from_b(stats) is False


def test_task_unaddressable_from_b_true_when_b_share_is_low():
    stats = {"A": {"mean_mcp_share": 0.0}, "B": {"mean_mcp_share": 0.01}}
    assert an.task_unaddressable_from_b(stats) is True


def test_task_unaddressable_from_b_true_when_b_missing():
    stats = {"A": {"mean_mcp_share": 0.9}}
    assert an.task_unaddressable_from_b(stats) is True


def test_success_not_lower_true_when_c_matches_or_beats_b():
    by_task = {"t1": {"C": {"successes": 3, "n": 3}, "B": {"successes": 2, "n": 3}}}
    assert an.success_not_lower(by_task, "C", "B") is True


def test_success_not_lower_false_when_c_is_worse():
    by_task = {"t1": {"C": {"successes": 1, "n": 3}, "B": {"successes": 3, "n": 3}}}
    assert an.success_not_lower(by_task, "C", "B") is False


def test_success_not_lower_false_when_no_shared_data():
    assert an.success_not_lower({}, "C", "B") is False


def test_paired_bootstrap_ci_task_subset_restricts_draws():
    by_task = {
        "t1": {"B": {"total_cost": 100, "successes": 1}, "A": {"total_cost": 50, "successes": 1}},
    }
    result_all = an.paired_bootstrap_ci(by_task, "B", "A", n_boot=50, seed=1)
    assert result_all["observed"] == 50.0
    result_restricted = an.paired_bootstrap_ci(by_task, "B", "A", n_boot=50, seed=1,
                                                task_subset=set())
    assert result_restricted["ci"] is None
    assert "no task" in result_restricted["reason"]


def test_analyze_restricts_b_minus_a_to_tasks_all_three_arms_share():
    rows = []
    for arm, cost in (("A", 100), ("B", 50), ("C", 10)):
        rows.append(_row("t1", arm, 1, True, cost))
    # t2: only A and B ran it (C never did) -- must be EXCLUDED from B-A's
    # contrast even though the raw (B, A) pair both have data for it, per
    # "report B-A on the common subset of tasks ALL arms can do".
    rows.append(_row("t2", "A", 1, True, 1000))
    rows.append(_row("t2", "B", 1, True, 1))
    report = an.analyze(rows)
    ba = report["haiku"]["contrasts"]["B-A"]
    assert ba["observed"] == 50 - 100  # only t1 -- not diluted by t2's huge A cost


def test_format_report_labels_the_cheaper_arm_when_significant():
    rows = []
    for i in range(8):
        task = f"t{i}"
        for rep in range(1, 4):
            rows.append(_row(task, "C", rep, True, 50))
            rows.append(_row(task, "B", rep, True, 150))
    report = an.analyze(rows)
    text = an.format_report(report)
    assert "C is cheaper" in text


def test_analyze_groups_by_model_independently():
    rows = [
        _row("t1", "B", 1, True, 100, model="haiku"),
        _row("t1", "B", 1, True, 999, model="sonnet"),
    ]
    report = an.analyze(rows)
    assert set(report) == {"haiku", "sonnet"}
    assert report["haiku"]["arms"]["B"]["total_cost"] == 100
    assert report["sonnet"]["arms"]["B"]["total_cost"] == 999


def test_analyze_and_format_report_smoke():
    rows = [
        _row("t1", "A", 1, False, 80),
        _row("t1", "B", 1, True, 100),
        _row("t1", "C", 1, True, 60),
    ]
    report = an.analyze(rows)
    assert "haiku" in report
    text = an.format_report(report)
    assert "t1" in text
    assert "cost/succ" in text
    assert "paired bootstrap" in text


def test_row_flag_reasons_flags_cost_gap_and_subagent_cost():
    assert an.row_flag_reasons({"cost_gap": True}) == ["cost_gap"]
    assert an.row_flag_reasons({"has_subagent_cost": True}) == ["has_subagent_cost"]
    assert an.row_flag_reasons({"cost_gap": True, "has_subagent_cost": True}) == \
        ["cost_gap", "has_subagent_cost"]
    assert an.row_flag_reasons({"cost_gap": False, "has_subagent_cost": False}) == []
    assert an.row_flag_reasons({}) == []


def test_flagged_rows_collects_task_arm_rep_and_reasons():
    rows = [
        {**_row("t1", "B", 1, True, 100), "cost_gap": True},
        {**_row("t1", "C", 2, True, 50), "has_subagent_cost": True},
        {**_row("t2", "A", 1, True, 10)},  # clean -- not flagged
    ]
    flags = an.flagged_rows(rows)
    assert len(flags) == 2
    assert {"task_id": "t1", "arm": "B", "rep": 1, "reasons": ["cost_gap"]} in flags
    assert {"task_id": "t1", "arm": "C", "rep": 2, "reasons": ["has_subagent_cost"]} in flags


def test_analyze_surfaces_flagged_rows_per_model():
    rows = [
        {**_row("t1", "B", 1, True, 100), "cost_gap": True},
        _row("t1", "C", 1, True, 50),
    ]
    report = an.analyze(rows)
    assert len(report["haiku"]["flagged_rows"]) == 1
    assert report["haiku"]["flagged_rows"][0]["arm"] == "B"


def test_format_report_prints_flagged_rows_section():
    rows = [
        {**_row("t1", "B", 1, True, 100), "cost_gap": True},
        _row("t1", "C", 1, True, 50),
    ]
    report = an.analyze(rows)
    text = an.format_report(report)
    assert "FLAGGED for cost-accounting review" in text
    assert "cost_gap" in text


def test_format_report_labels_b_minus_a_contrast():
    rows = [_row("t1", arm, 1, True, cost) for arm, cost in
            (("A", 100), ("B", 80), ("C", 10))]
    report = an.analyze(rows)
    text = an.format_report(report)
    assert "your setup vs vanilla" in text


def test_first_turn_cache_stats_means_over_rows_with_data():
    rows = [
        {**_row("t1", "B", 1, True, 100), "first_turn_cache_read_tokens": 10,
         "first_turn_cache_write_tokens": 20},
        {**_row("t1", "B", 2, True, 100), "first_turn_cache_read_tokens": 30,
         "first_turn_cache_write_tokens": 40},
        {**_row("t1", "A", 1, True, 100)},  # no first-turn data -- must not pollute B's mean
    ]
    stats = an.first_turn_cache_stats(rows, "B")
    assert stats["n"] == 2
    assert stats["mean_first_turn_cache_read"] == 20
    assert stats["mean_first_turn_cache_write"] == 30


def test_first_turn_cache_stats_counts_cold_runs():
    # A measured run whose first turn read NOTHING from cache paid a full
    # cold write (warm-up missed or expired) -- surfaced, not averaged away.
    rows = [
        {**_row("t1", "C", 1, True, 100), "first_turn_cache_read_tokens": 0,
         "first_turn_cache_write_tokens": 14000},
        {**_row("t1", "C", 2, True, 100), "first_turn_cache_read_tokens": 9000,
         "first_turn_cache_write_tokens": 4000},
    ]
    assert an.first_turn_cache_stats(rows, "C")["cold"] == 1


def test_format_report_shows_cold_count():
    rows = [
        {**_row("t1", "B", 1, True, 100), "first_turn_cache_read_tokens": 0,
         "first_turn_cache_write_tokens": 20},
    ]
    text = an.format_report(an.analyze(rows))
    assert "cold=1" in text


def test_first_turn_cache_stats_none_when_no_data_for_arm():
    stats = an.first_turn_cache_stats([_row("t1", "A", 1, True, 100)], "C")
    assert stats["n"] == 0
    assert stats["mean_first_turn_cache_read"] is None
    assert stats["mean_first_turn_cache_write"] is None


def test_analyze_reports_first_turn_cache_per_arm():
    rows = [
        {**_row("t1", "B", 1, True, 100), "first_turn_cache_read_tokens": 10,
         "first_turn_cache_write_tokens": 20},
    ]
    report = an.analyze(rows)
    assert report["haiku"]["first_turn_cache"]["B"]["mean_first_turn_cache_read"] == 10


def test_load_rows_reads_jsonl(tmp_path):
    import json
    path = tmp_path / "r.jsonl"
    rows = [_row("t1", "B", 1, True, 100), _row("t1", "C", 1, True, 50)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")  # trailing blank line
    loaded = an.load_rows([path])
    assert len(loaded) == 2
