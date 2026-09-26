"""Cost-per-successful-task analysis for the cost_per_task harness.

Reads the runner's JSONL rows and reports, per model:
  - cost per SUCCESSFUL task per arm: total weighted cost across ALL reps of
    a task (a failed rep still spent tokens and is charged against fewer
    successes, never dropped), divided by how many of those reps succeeded.
  - success rate per arm/task.
  - a paired bootstrap 95% CI for C-B, B-A, C-A. Resampling is at the TASK
    level, not the run level: the statistic being differenced (cost per
    success) is itself a ratio aggregated across a task's reps, so
    resampling individual runs would break the pairing the plan calls for
    ("paired per task"). Each bootstrap draw resamples task ids with
    replacement and uses the SAME resampled multiset for both arms in the
    contrast, which is what makes it a *paired* bootstrap rather than two
    independent ones.
  - every task, flagging "unaddressable" ones (mean MCP token share under
    5%) rather than dropping them -- terse cannot help a task it never
    touches, and the paper's point is exactly that most tasks are like this.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ("A", "B", "C")
CONTRASTS = (("C", "B"), ("B", "A"), ("C", "A"))
UNADDRESSABLE_THRESHOLD = 0.05
# Plan Partial #7: B-A isolates the delta between the operator's REAL setup
# (CLAUDE.md, output style, settings-derived env, plugins -- everything
# `setting_sources=None` loads) and vanilla Claude Code with none of that
# (arm A's `setting_sources=""` + `--safe-mode`) -- it is NOT "terse vs no
# terse" (that's C-B) and mislabeling it that way in a report reader skims
# is exactly the kind of confound this harness exists to catch.
CONTRAST_LABELS = {
    "B-A": "your setup vs vanilla (CLAUDE.md, output style, settings env, plugins)",
}


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def partition_infra_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split `rows` into (analyzable, infra_error) -- an infra_error row
    (timeout, rate limit, a non-zero exit with no model turn, unparseable
    CLI output) is an INFRASTRUCTURE failure, not a task result: it says
    nothing about whether the task is achievable or what it costs, so it
    must never enter cost-per-success math. Excluded here, at the one choke
    point every caller of `analyze` goes through, rather than trusted to be
    filtered upstream."""
    infra = [r for r in rows if r.get("infra_error")]
    analyzable = [r for r in rows if not r.get("infra_error")]
    return analyzable, infra


def row_flag_reasons(row: dict) -> list[str]:
    """Why `row`'s cost accounting needs a human look before the numbers are
    trusted -- 'cost_gap' (this harness's own weighted-token/USD estimate
    and the CLI's own total_cost_usd disagree by >5%: an unpriced model
    normalization miss or an accounting bug on either side) and/or
    'has_subagent_cost' (a Task-spawned subagent's transcript contributed to
    this row's cost despite --disallowedTools Task on every arm -- should
    never happen, and if it does the cost/tool-call counts may not mean what
    they normally mean for this row). [] when the row needs no such
    review (plan Partial #3)."""
    reasons = []
    if row.get("cost_gap"):
        reasons.append("cost_gap")
    if row.get("has_subagent_cost"):
        reasons.append("has_subagent_cost")
    return reasons


def flagged_rows(rows: list[dict]) -> list[dict]:
    """Every analyzable row with a non-empty `row_flag_reasons`, as
    {task_id, arm, rep, reasons} -- collected, never silently averaged over
    or dropped, so a report reader can see exactly which runs' cost numbers
    are suspect (plan Partial #3: analysis must FLAG or refuse rows with
    cost_gap/has_subagent_cost, not ignore the fields)."""
    out = []
    for r in rows:
        reasons = row_flag_reasons(r)
        if reasons:
            out.append({"task_id": r.get("task_id"), "arm": r.get("arm"),
                        "rep": r.get("rep"), "reasons": reasons})
    return out


def first_turn_cache_stats(rows: list[dict], arm: str) -> dict:
    """Mean first-turn cache-read/cache-write tokens for `arm`, over rows
    that recorded them (a transcript must have existed). Plan Partial #9:
    the harness now always runs one discarded warm-up rep per arm before the
    measured reps specifically to absorb a cold-start cache-write spike --
    reporting the measured reps' own first-turn numbers lets a reader check
    whether that warm-up actually worked, instead of just trusting it did."""
    reads = [r["first_turn_cache_read_tokens"] for r in rows
             if r.get("arm") == arm and r.get("first_turn_cache_read_tokens") is not None]
    writes = [r["first_turn_cache_write_tokens"] for r in rows
              if r.get("arm") == arm and r.get("first_turn_cache_write_tokens") is not None]
    return {
        "n": len(reads),
        "mean_first_turn_cache_read": statistics.mean(reads) if reads else None,
        "mean_first_turn_cache_write": statistics.mean(writes) if writes else None,
    }


def _group_by_model_task_arm(rows: list[dict]) -> dict:
    """rows -> {model: {task_id: {arm: [row, ...]}}}"""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        out[r["model"]][r["task_id"]][r["arm"]].append(r)
    return out


def task_arm_stats(runs: list[dict]) -> dict:
    """One (task, arm)'s aggregate over its reps: n, successes, success
    rate, total cost (successes AND failures -- a failed rep still spent
    tokens), cost per success (None if zero successes -- an undefined ratio,
    not infinite cost), and the mean approximate MCP share over reps that
    produced a transcript."""
    n = len(runs)
    successes = sum(1 for r in runs if r.get("success"))
    for r in runs:
        if r.get("weighted_cost") is None:
            raise ValueError(
                f"row task_id={r.get('task_id')!r} arm={r.get('arm')!r} "
                f"rep={r.get('rep')!r} has weighted_cost=None and is not "
                f"flagged infra_error -- refusing to treat it as free cost "
                f"(0.0 would silently understate cost-per-success). Either "
                f"the row is a genuine infra failure that should carry "
                f"infra_error, or the cost pipeline has a bug.")
    total_cost = sum(r["weighted_cost"] for r in runs)
    shares = [r["mcp_share_approx"] for r in runs
              if r.get("mcp_share_approx") is not None]
    mean_share = statistics.mean(shares) if shares else 0.0
    return {
        "n": n, "successes": successes,
        "success_rate": (successes / n) if n else 0.0,
        "total_cost": total_cost,
        "cost_per_success": (total_cost / successes) if successes else None,
        "mean_mcp_share": mean_share,
        "unaddressable": mean_share < UNADDRESSABLE_THRESHOLD,
    }


def arm_aggregate(by_task: dict, arm: str) -> dict:
    """Cost per successful task for `arm`, aggregated across ALL tasks in
    `by_task`: sum(total_cost) / sum(successes). `cost_per_success` is None
    if the arm has zero successes anywhere. `success_rate` is pooled over
    every rep the arm ran (n), not just tasks it succeeded at."""
    total_cost = sum(s[arm]["total_cost"] for s in by_task.values() if arm in s)
    successes = sum(s[arm]["successes"] for s in by_task.values() if arm in s)
    n = sum(s[arm].get("n", 0) for s in by_task.values() if arm in s)
    return {"total_cost": total_cost, "successes": successes, "n": n,
            "success_rate": (successes / n) if n else 0.0,
            "cost_per_success": (total_cost / successes) if successes else None}


def success_not_lower(by_task: dict, arm: str, baseline: str) -> bool:
    """True if `arm`'s pooled success rate, over the tasks both `arm` and
    `baseline` have data for, is >= `baseline`'s. This is the Win
    criterion's third condition (plan: "C's cost per successful task is
    lower... AND C's success rate is not lower") -- a cost win achieved by
    `arm` simply failing more (and so never paying for the hard cases) must
    not read as a win. False (never a pass) when there is no shared data to
    judge it by."""
    task_ids = [t for t, arms in by_task.items() if arm in arms and baseline in arms]
    arm_n = sum(by_task[t][arm]["n"] for t in task_ids)
    base_n = sum(by_task[t][baseline]["n"] for t in task_ids)
    if arm_n == 0 or base_n == 0:
        return False
    arm_succ = sum(by_task[t][arm]["successes"] for t in task_ids)
    base_succ = sum(by_task[t][baseline]["successes"] for t in task_ids)
    return (arm_succ / arm_n) >= (base_succ / base_n)


def task_unaddressable_from_b(task_stats: dict) -> bool:
    """Whether a task is 'unaddressable' by terse, judged from arm B's mean
    MCP share alone (plan blocker #11) -- B is "the user's setup without
    terse", so its own tool-call pattern says whether the task touches MCP
    at all, independent of anything terse itself does to the transcript.
    Arm A structurally reads ~0 MCP share (no MCP servers configured) and
    would trivially mark every task unaddressable if used as the source;
    arm C's share can differ from B's purely because terse compresses the
    text, which is not what 'does this task touch MCP' should measure. True
    when B has no data at all for the task -- nothing to judge it
    addressable BY."""
    b = task_stats.get("B")
    if b is None:
        return True
    return b["mean_mcp_share"] < UNADDRESSABLE_THRESHOLD


def _aggregate_for_picks(picks: list[str], by_task: dict, arm: str) -> float | None:
    """Cost per success for `arm`, aggregated over a GIVEN task multiset
    `picks` (not drawn here) -- the caller draws `picks` once and passes the
    same list to both arms in a contrast, which is what makes the bootstrap
    PAIRED rather than two independent resamples."""
    total_cost = sum(by_task[t][arm]["total_cost"] for t in picks if arm in by_task[t])
    successes = sum(by_task[t][arm]["successes"] for t in picks if arm in by_task[t])
    return (total_cost / successes) if successes else None


def paired_bootstrap_ci(by_task: dict, arm_hi: str, arm_lo: str, *,
                         n_boot: int = 2000, seed: int = 0,
                         task_subset: set[str] | None = None) -> dict:
    """95% CI for (arm_hi's cost/success) - (arm_lo's cost/success),
    resampling TASKS with replacement -- paired because the SAME resampled
    task multiset is used to recompute both arms' aggregates in each draw. A
    draw where either arm lands with zero successes is skipped (not
    zero-filled): an undefined ratio is not the same claim as a zero
    effect. Returns 'ci': None with a `reason` when too few draws were
    usable to say anything (too few tasks, or one arm rarely succeeds).

    `task_subset`, if given, further restricts which tasks are eligible
    (plan blocker #11: "report B-A on the common subset of tasks all arms
    can do") -- `analyze` passes the tasks all of A/B/C have data for so
    that contrast is comparable to C-B/C-A on the same task universe, not
    diluted by tasks only some arms attempted."""
    task_ids = [t for t, arms in by_task.items() if arm_hi in arms and arm_lo in arms
                and (task_subset is None or t in task_subset)]
    if not task_ids:
        return {"ci": None, "observed": None, "n_boot": n_boot, "n_used": 0,
                "reason": f"no task has both arm {arm_hi} and {arm_lo}"
                          + (" in the given task_subset" if task_subset is not None else "")}

    # Observed delta over the SAME restricted task_ids the bootstrap draws
    # from -- not the full by_task -- so 'observed' and the CI describe the
    # identical task universe.
    obs_hi = _aggregate_for_picks(task_ids, by_task, arm_hi)
    obs_lo = _aggregate_for_picks(task_ids, by_task, arm_lo)
    observed = (obs_hi - obs_lo) if (obs_hi is not None and obs_lo is not None) else None

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        # Draw the task multiset ONCE per iteration and apply it to BOTH
        # arms -- a paired bootstrap resamples the pairing unit (the task),
        # not each arm independently. Drawing separate picks per arm (the
        # bug this replaced) can show a nonzero delta between two IDENTICAL
        # arms purely from picking different tasks on each side.
        picks = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        hi = _aggregate_for_picks(picks, by_task, arm_hi)
        lo = _aggregate_for_picks(picks, by_task, arm_lo)
        if hi is None or lo is None:
            continue
        diffs.append(hi - lo)

    min_usable = max(50, n_boot // 10)
    if len(diffs) < min_usable:
        return {"ci": None, "observed": observed, "n_boot": n_boot, "n_used": len(diffs),
                "reason": "too few bootstrap draws had successes in both arms to "
                          "size a CI -- needs more tasks/reps"}

    diffs.sort()
    lo_idx = int(0.025 * len(diffs))
    hi_idx = min(len(diffs) - 1, int(0.975 * len(diffs)))
    ci = (diffs[lo_idx], diffs[hi_idx])
    return {"ci": ci, "observed": observed, "n_boot": n_boot, "n_used": len(diffs),
            "excludes_zero": ci[0] > 0 or ci[1] < 0}


def analyze(rows: list[dict]) -> dict:
    analyzable, _infra = partition_infra_rows(rows)
    grouped = _group_by_model_task_arm(analyzable)
    report: dict = {}
    for model, by_task_raw in grouped.items():
        by_task = {t: {arm: task_arm_stats(runs) for arm, runs in arm_runs.items()}
                   for t, arm_runs in by_task_raw.items()}
        # The tasks every one of A/B/C has data for -- B-A is restricted to
        # this so it is comparable to C-B/C-A on the same task universe
        # (plan blocker #11), rather than diluted by a task only some arms
        # attempted.
        common_tasks = {t for t, arm_stats in by_task.items()
                         if all(a in arm_stats for a in ARMS)}
        arms_agg = {a: arm_aggregate(by_task, a) for a in ARMS}
        contrasts = {}
        for hi, lo in CONTRASTS:
            subset = common_tasks if (hi, lo) == ("B", "A") else None
            contrasts[f"{hi}-{lo}"] = paired_bootstrap_ci(by_task, hi, lo, task_subset=subset)
        c_success_not_lower = success_not_lower(by_task, "C", "B")
        unaddressable = {t: task_unaddressable_from_b(s) for t, s in by_task.items()}
        model_rows = [r for arm_runs in by_task_raw.values() for runs in arm_runs.values()
                      for r in runs]
        first_turn_cache = {a: first_turn_cache_stats(model_rows, a) for a in ARMS}
        report[model] = {"tasks": by_task, "arms": arms_agg, "contrasts": contrasts,
                          "c_success_not_lower": c_success_not_lower,
                          "unaddressable": unaddressable,
                          "flagged_rows": flagged_rows(model_rows),
                          "first_turn_cache": first_turn_cache}
    return report


def format_report(report: dict, infra: list[dict] | None = None) -> str:
    lines: list[str] = []
    if infra:
        lines.append(f"{len(infra)} row(s) EXCLUDED as infra failures (re-run these):")
        for r in infra:
            lines.append(f"  {r.get('task_id')} arm={r.get('arm')} rep={r.get('rep')} "
                          f"{r.get('infra_error')}: {r.get('error')}")
    for model, m in report.items():
        lines.append(f"\n=== model: {model} ===")
        if m.get("flagged_rows"):
            lines.append(f"{len(m['flagged_rows'])} row(s) FLAGGED for cost-accounting "
                          f"review (cost_gap and/or has_subagent_cost -- verify before "
                          f"trusting their cost):")
            for f in m["flagged_rows"]:
                lines.append(f"  {f['task_id']} arm={f['arm']} rep={f['rep']} "
                              f"reasons={','.join(f['reasons'])}")
        lines.append(f"{'task':<32} {'arm':<3} {'n':>3} {'succ':>4} {'rate':>6} "
                      f"{'cost/succ':>12} {'mcp_share':>10} {'flag':>14}")
        for task_id in sorted(m["tasks"]):
            # 'unaddressable' is a TASK-level judgement from arm B's MCP
            # share alone -- the same flag for every arm's row on this task
            # (plan blocker #11), not each arm's own (structurally-zero-for-A)
            # share.
            flag = "unaddressable" if m["unaddressable"].get(task_id) else ""
            for arm in ARMS:
                s = m["tasks"][task_id].get(arm)
                if s is None:
                    continue
                cps = f"{s['cost_per_success']:.0f}" if s["cost_per_success"] is not None else "n/a"
                lines.append(f"{task_id:<32} {arm:<3} {s['n']:>3} {s['successes']:>4} "
                              f"{s['success_rate']:>6.0%} {cps:>12} "
                              f"{s['mean_mcp_share']:>10.1%} {flag:>14}")
        lines.append("\narm aggregate cost per successful task (all tasks pooled):")
        for arm in ARMS:
            a = m["arms"][arm]
            cps = f"{a['cost_per_success']:.0f}" if a["cost_per_success"] is not None else "n/a"
            lines.append(f"  {arm}: {cps} weighted tokens/success "
                          f"({a['successes']}/{a['n']} runs succeeded, "
                          f"{a['success_rate']:.0%} success rate)")
        lines.append(f"\nWin criterion check: C's success rate not lower than B's -> "
                      f"{'YES' if m['c_success_not_lower'] else 'NO'}")
        lines.append("\npaired bootstrap 95% CI (arm_hi - arm_lo, weighted tokens/success):")
        for key, c in m["contrasts"].items():
            hi, lo = key.split("-")
            label = CONTRAST_LABELS.get(key)
            key_display = f"{key} [{label}]" if label else key
            if c["ci"] is None:
                lines.append(f"  {key_display}: {c['reason']}")
            elif c["excludes_zero"]:
                cheaper = lo if c["observed"] > 0 else hi
                lines.append(f"  {key_display}: observed {c['observed']:+.0f}, "
                              f"95% CI [{c['ci'][0]:+.0f}, {c['ci'][1]:+.0f}]  "
                              f"SIGNIFICANT -- {cheaper} is cheaper")
            else:
                lines.append(f"  {key_display}: observed {c['observed']:+.0f}, "
                              f"95% CI [{c['ci'][0]:+.0f}, {c['ci'][1]:+.0f}]  "
                              f"inconclusive (CI crosses 0)")
        if m.get("first_turn_cache"):
            lines.append("\nfirst-turn cache (mean over measured reps; the warm-up rep "
                          "itself is excluded, written to warmup.jsonl):")
            for arm in ARMS:
                ft = m["first_turn_cache"].get(arm)
                if not ft or ft["n"] == 0:
                    continue
                lines.append(f"  {arm}: read={ft['mean_first_turn_cache_read']:.0f} "
                              f"write={ft['mean_first_turn_cache_write']:.0f} (n={ft['n']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)
    rows = load_rows(args.jsonl)
    if not rows:
        print("no rows to analyze", file=sys.stderr)
        return 2
    _analyzable, infra = partition_infra_rows(rows)
    report = analyze(rows)
    if args.json:
        print(json.dumps({"report": report, "infra_excluded": infra}, indent=2))
    else:
        print(format_report(report, infra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
