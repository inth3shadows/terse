"""Tests for the codec-tier material-preservation verdict (#295) — `report.codec_verdict`
and `build_codec_verdict_report`.

This gate is deliberately NOT a tolerance: any PAIRED excess of terse-arm misses beyond
what the raw arm also missed is UNSAFE regardless of sample size, and a clean run needs
`_CODEC_MIN_TRIALS` zero-failure trials before it can print SAFE rather than UNRESOLVED.
Every boundary below is the kind of edge a tolerance-shaped gate would get wrong by
construction — this file exists so an accidental reintroduction of a ratio/threshold shows
up as a red test, not a silent regression.

The identical-performance fixtures below (`test_identical_failure_on_both_arms_...`) pin the
PR #302 review's finding 1: a first draft compared `terse_ok` to `terse_trials` alone,
ignoring `raw_ok` entirely, and printed UNSAFE for a model that failed EQUALLY on raw and
terse — i.e. for behavior with no demonstrated codec-caused difference at all."""
from __future__ import annotations

import json

from terse.report import (
    _CODEC_MIN_CALL_RATE,
    _CODEC_MIN_TRIALS,
    build_codec_verdict_report,
    codec_call_rate,
    codec_verdict,
)


def _row(qid: str, raw_ok: int, terse_ok: int, trials: int = 1,
         raw_calls: int | None = None, terse_calls: int | None = None) -> dict:
    row = {
        "qid": qid, "qtype": "deref", "transform": "table", "trials": trials,
        "raw_ok": raw_ok, "terse_ok": terse_ok,
        "raw_trials": trials, "terse_trials": trials,
        "fails": 0, "attempts": trials * 2,
    }
    # OMITTED by default, so every pre-#403 fixture in this file keeps exercising the
    # no-counter path — which is the shape of every result file written before the
    # compliance gate existed.
    if raw_calls is not None:
        row["raw_calls"] = raw_calls
    if terse_calls is not None:
        row["terse_calls"] = terse_calls
    return row


# --------------------------------------------------------------------------- #
# codec_verdict
# --------------------------------------------------------------------------- #
def test_a_single_structural_failure_is_UNSAFE_regardless_of_sample_size():
    # 1 failure out of a large, otherwise-clean sample — a ratio-shaped gate would round
    # this down to "safe enough"; the demonstrated-corruption gate must not.
    rows = [_row(f"q{i}", 1, 1) for i in range(50)]
    rows.append(_row("q-bad", 1, 0))  # one paired miss
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNSAFE"


def test_zero_failures_below_the_trial_floor_is_UNRESOLVED_not_SAFE():
    rows = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS - 1)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNRESOLVED"


def test_identical_failure_on_both_arms_is_not_UNSAFE():
    # raw_ok == terse_ok on every row: the model is imperfect, but EQUALLY imperfect on
    # raw and terse — no demonstrated codec-caused difference. Below the trial floor, so
    # this is UNRESOLVED (not enough clean evidence), never UNSAFE.
    rows = [_row(f"q{i}", 0, 0, trials=1) for i in range(5)]
    verdict, gap = codec_verdict(rows)
    assert verdict != "UNSAFE"


def test_identical_partial_failure_on_both_arms_at_the_trial_floor_is_SAFE():
    # A single row, 25 trials each, raw and terse both succeed on the SAME 20/25 — no
    # excess terse-specific miss, and n clears the floor.
    rows = [_row("q1", raw_ok=20, terse_ok=20, trials=25)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "SAFE"


def test_a_genuine_terse_specific_regression_is_UNSAFE_even_when_raw_is_imperfect():
    # raw succeeds MORE often than terse on the same row — a real excess, not just raw's
    # own baseline imperfection. Must still be UNSAFE.
    rows = [_row("q1", raw_ok=20, terse_ok=15, trials=25)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNSAFE"


def test_zero_failures_at_exactly_the_trial_floor_is_SAFE():
    rows = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "SAFE"


def test_an_unmeasured_backend_reports_UNRESOLVED_not_a_confident_verdict():
    # Every row carries zero completed trials -> `_unmeasured` fires inside `arm_gap`.
    rows = [{
        "qid": "q1", "qtype": "deref", "transform": "table", "trials": 1,
        "raw_ok": 0, "terse_ok": 0, "raw_trials": 0, "terse_trials": 0,
        "fails": 2, "attempts": 2,
    }]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNRESOLVED"
    assert gap.excluded == "unmeasured"


def test_empty_rows_is_UNRESOLVED():
    verdict, gap = codec_verdict([])
    assert verdict == "UNRESOLVED"


def test_a_single_failure_beats_an_otherwise_unresolved_thin_sample():
    # A thin sample (below the floor) that ALSO shows a failure must still be UNSAFE, not
    # UNRESOLVED — the failure is dispositive; sample size only gates a clean run.
    rows = [_row("q1", 1, 0)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNSAFE"


# --------------------------------------------------------------------------- #
# build_codec_verdict_report — grouped by (tool, shape), gates on the worst model
# --------------------------------------------------------------------------- #
def _tagged(rows: list[dict], tool: str, shape: str) -> list[dict]:
    return [{"tool": tool, "shape": shape, **r} for r in rows]


def test_report_groups_by_tool_and_shape_not_globally():
    clean = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    results = {
        "m1": (_tagged(clean, "tool-a", "array-of-records")
              + _tagged(clean, "tool-b", "array-of-records")),
    }
    report = build_codec_verdict_report(results)
    assert "`tool-a`" in report
    assert "`tool-b`" in report
    # Two distinct group rows, not one pooled global line.
    assert report.count("| **SAFE**") == 2


def test_report_gates_on_the_worst_model_within_a_group():
    clean = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    broken = [_row("q-bad", 1, 0)]
    results = {
        "good-model": _tagged(clean, "tool-a", "array-of-records"),
        "bad-model": _tagged(broken, "tool-a", "array-of-records"),
    }
    report = build_codec_verdict_report(results)
    assert "**UNSAFE**" in report
    assert "`bad-model`" in report


def test_report_with_no_results_says_so_rather_than_rendering_an_empty_table():
    report = build_codec_verdict_report({})
    assert "## Verdict by tool and shape" not in report
    assert "| **" not in report  # no rendered verdict table row
    assert "No tool-capable model answered" in report


# --------------------------------------------------------------------------- #
# Tool-call compliance (#403) — gates SAFE, never UNSAFE
# --------------------------------------------------------------------------- #
def test_a_clean_run_below_the_call_rate_floor_is_UNRESOLVED_not_SAFE():
    # 40 zero-failure trials, well past `_CODEC_MIN_TRIALS` — but half of the terse arm's
    # answers arrived in prose, so the run never showed a value surviving into a downstream
    # tool argument. SAFE is a claim about that, and this run cannot make it.
    rows = [_row(f"q{i}", 2, 2, trials=2, raw_calls=2, terse_calls=1) for i in range(20)]
    assert codec_verdict(rows)[0] == "UNRESOLVED"


def test_the_same_clean_run_at_full_compliance_is_SAFE():
    # The mutation control for the test above: identical accuracy, identical trial count,
    # only the channel counter changes. If the gate is removed, the test above goes SAFE
    # and this one is unchanged — so the pair localises the gate.
    rows = [_row(f"q{i}", 2, 2, trials=2, raw_calls=2, terse_calls=2) for i in range(20)]
    assert codec_verdict(rows)[0] == "SAFE"


def test_low_compliance_on_the_RAW_arm_also_withholds_SAFE():
    # The premise is symmetric: a paired comparison where the control arm mostly answered
    # in prose has not demonstrated the tool-argument path either.
    rows = [_row(f"q{i}", 2, 2, trials=2, raw_calls=0, terse_calls=2) for i in range(20)]
    assert codec_verdict(rows)[0] == "UNRESOLVED"


def test_exactly_at_the_call_rate_floor_is_SAFE():
    # Inclusive `>=`, matching `_CODEC_MIN_TRIALS`. At the default 0.80 floor, 8 of 10.
    per_arm = 10
    at_floor = round(_CODEC_MIN_CALL_RATE * per_arm)
    rows = [_row(f"q{i}", per_arm, per_arm, trials=per_arm,
                 raw_calls=per_arm, terse_calls=at_floor) for i in range(2)]
    assert codec_verdict(rows)[0] == "SAFE"
    one_under = [_row(f"q{i}", per_arm, per_arm, trials=per_arm,
                      raw_calls=per_arm, terse_calls=at_floor - 1) for i in range(2)]
    assert codec_verdict(one_under)[0] == "UNRESOLVED"


def test_low_compliance_does_NOT_suppress_an_UNSAFE_verdict():
    # The direction that matters. Values are scored on whichever channel they arrive by, so
    # an observed excess is real regardless of compliance — withholding it would suppress
    # the finding this tier exists to make.
    rows = [_row(f"q{i}", 2, 2, trials=2, raw_calls=0, terse_calls=0) for i in range(20)]
    rows.append(_row("q-bad", 2, 0, trials=2, raw_calls=0, terse_calls=0))
    assert codec_verdict(rows)[0] == "UNSAFE"


def test_rows_without_the_counter_are_not_read_as_compliant_or_as_failing():
    # Every result file written before #403 has no `raw_calls`/`terse_calls`. Reading a
    # missing counter as 0 would retract those runs on evidence never collected; reading it
    # as full compliance would silently re-certify them. `codec_call_rate` answers None and
    # the gate does not fire either way.
    rows = [_row(f"q{i}", 2, 2, trials=2) for i in range(20)]
    assert codec_call_rate(rows, "terse_ok") is None
    assert codec_verdict(rows)[0] == "SAFE"


def test_codec_call_rate_divides_by_the_arms_own_answered_trials():
    # `raw_trials` deliberately DIFFERS from the shared `trials` and from `terse_trials`:
    # dividing by `trials` instead would read 6/8 and 4/8 here and the test would still be
    # green, which is the fixture-cannot-fail mode.
    rows = [dict(_row("q0", 4, 4, trials=4, raw_calls=3, terse_calls=1), raw_trials=3),
            dict(_row("q1", 4, 4, trials=4, raw_calls=2, terse_calls=3), raw_trials=2)]
    assert codec_call_rate(rows, "raw_ok") == 5 / 5
    assert codec_call_rate(rows, "terse_ok") == 4 / 8


def test_codec_call_rate_prefers_the_answered_counter_over_the_trial_count():
    # An arm that lost half its calls to the backend but used the tool on every call it did
    # receive is fully compliant; reading `trials` would report it at 50% and the renderer
    # would blame the model for transport loss.
    rows = [dict(_row("q0", 2, 2, trials=4, raw_calls=4, terse_calls=2),
                 terse_answered=2)]
    assert codec_call_rate(rows, "terse_ok") == 1.0
    assert codec_call_rate(rows, "raw_ok") == 1.0  # no `raw_answered`: falls back to trials


def test_the_table_names_the_arm_and_the_rate_when_compliance_withholds_SAFE():
    # Falling through to the trial-count reason would be true and point at the wrong cause:
    # these rows are far past the trial floor.
    rows = [dict(_row(f"q{i}", 2, 2, trials=2, raw_calls=2, terse_calls=0),
                 tool="kb.read.list_nodes", shape="array-of-records") for i in range(20)]
    out = build_codec_verdict_report({"m1": rows})
    assert "**UNRESOLVED**" in out
    assert "terse arm delivered 0%" in out
    assert f"need {_CODEC_MIN_CALL_RATE:.0%}" in out
    assert "zero-failure trial(s), need" not in out


def test_the_table_names_BOTH_arms_worst_first_when_both_are_non_compliant():
    # Naming only the worse arm (the first cut) leaves the cell UNRESOLVED on the other one
    # after its fix — review of #411 found this.
    rows = [dict(_row(f"q{i}", 4, 4, trials=4, raw_calls=2, terse_calls=1),
                 tool="kb.read.list_nodes", shape="array-of-records") for i in range(10)]
    out = build_codec_verdict_report({"m1": rows})
    assert "terse arm delivered 25%" in out
    assert "raw arm delivered 50%" in out
    assert out.index("terse arm delivered") < out.index("raw arm delivered")


def test_an_arm_dependent_prose_habit_no_longer_manufactures_UNSAFE():
    """The #403 regression, driven through `codeceval` and into `codec_verdict`.

    Measured on `kb__kb.read.list_nodes` with `qwen3-coder-30b-fl` at `--trials 7`,
    temperature 0: the model answered `enumerate` CORRECTLY in prose on the raw arm and
    through the tool on the terse arm, then flipped channels for `deref`. Under the old
    scoring each prose turn was a miss, and `max(0, raw_ok - terse_ok)` converted the flip
    straight into `excess_terse_misses` — an UNSAFE verdict on trials where terse was read
    correctly every time.

    Hand-writing the post-fix row numbers here would pin nothing: deleting the prose
    fallback outright left such a test green. So this runs the real emitter with a scripted
    model that flips channel by arm, and feeds what it emits to the real verdict."""
    from terse import codeceval, fluency
    from terse.dropeval import ToolCall, Turn

    payload = {"result": [{"id": i, "meta": {"owner": o, "tags": ["x", "y"]}}
                          for i, o in enumerate(["a", "b", "c", "d", "e", "f"], 1)]}
    raw_text = json.dumps(payload)
    terse_text = fluency.compress(payload)
    questions = {q.qid: q for q in codeceval.gen_codec_questions(payload)}
    assert set(questions) == {"deref", "enumerate"}

    def flips_channel_by_arm(messages):
        content = messages[-1]["content"]
        raw_arm = raw_text in content
        assert raw_arm or terse_text in content
        q = next(q for q in questions.values() if q.prompt in content)
        # raw answers `enumerate` in prose and `deref` through the tool; terse the reverse.
        in_prose = raw_arm == (q.qtype == "enumerate")
        if in_prose:
            return Turn(text=json.dumps(q.expected), tool_calls=[])
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": q.expected})])

    rows = [dict(r, tool="kb__kb.read.list_nodes", shape="array-of-records")
            for r in codeceval.run_codec_payload(payload, raw_text, flips_channel_by_arm,
                                                 trials=7)]
    # The premise: the channel really did flip by arm, or the fixture proves nothing.
    by_qid = {r["qid"]: r for r in rows}
    assert (by_qid["enumerate"]["raw_calls"], by_qid["enumerate"]["terse_calls"]) == (0, 7)
    assert (by_qid["deref"]["raw_calls"], by_qid["deref"]["terse_calls"]) == (7, 0)
    # Every trial read its payload correctly, so no arm may show an excess of misses...
    assert all(r["raw_ok"] == 7 and r["terse_ok"] == 7 for r in rows)
    # ...and the cell is UNRESOLVED on compliance rather than UNSAFE. NOT SAFE either: at
    # 50% compliance the run never showed a value reaching a real tool argument.
    assert codec_verdict(rows)[0] == "UNRESOLVED"


# --------------------------------------------------------------------------- #
# Every reason an UNRESOLVED cell is unresolved — not just the first (#403)
# --------------------------------------------------------------------------- #
def _cell(report: str) -> str:
    return next(ln for ln in report.splitlines() if ln.startswith("| `t` |"))


def test_an_unresolved_cell_names_BOTH_compliance_and_the_trial_floor():
    """The 2026-09-14 re-run: `kb.read.list_principles` printed only "terse arm delivered 29%
    of its answers through the tool call" — true, and incomplete. It was also n=7 against a
    20-trial floor, so fixing compliance alone would still leave it UNRESOLVED. This row
    reproduces that cell's shape: one question, 7 trials, 2 of 7 through the tool."""
    rows = [dict(_row("q", 7, 7, trials=7, raw_calls=7, terse_calls=2),
                 tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "**UNRESOLVED**" in cell
    assert "delivered 29%" in cell
    assert "only 7 zero-failure trial(s), need 20" in cell


def test_an_unresolved_cell_names_the_trimmed_corpus_AND_the_trial_floor():
    from terse.codeceval import OversizedPayload

    rows = [dict(_row("q", 3, 3, trials=3, raw_calls=3, terse_calls=3),
                 tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    cell = _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))
    assert "trimmed corpus" in cell
    assert "only 3 zero-failure trial(s)" in cell


def test_all_three_reasons_at_once_are_all_named():
    from terse.codeceval import OversizedPayload

    rows = [dict(_row("q", 5, 5, trials=5, raw_calls=5, terse_calls=0),
                 tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    cell = _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))
    assert cell.count(";") == 2, cell
    for part in ("trimmed corpus", "delivered 0%", "only 5 zero-failure"):
        assert part in cell, part


def test_a_single_reason_is_still_rendered_alone():
    # Full compliance and nothing excluded — only the trial floor fails. The join must not
    # invent a separator around a lone reason.
    rows = [dict(_row("q", 5, 5, trials=5, raw_calls=5, terse_calls=5),
                 tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "only 5 zero-failure trial(s), need 20" in cell
    assert ";" not in cell.split("**UNRESOLVED**")[1]


def test_a_cell_unresolved_for_two_models_names_BOTH_models_reasons():
    """Review of #411: the worst-model tie-break kept the first UNRESOLVED model and printed
    only its reasons. Here `a` is short on trials and `b` is short on compliance — raising
    the trial count alone leaves the cell UNRESOLVED on `b`."""
    a = [dict(_row("q", 7, 7, trials=7, raw_calls=7, terse_calls=7),
              tool="t", shape="array-of-records")]
    b = [dict(_row("q", 20, 20, trials=20, raw_calls=20, terse_calls=6),
              tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"a": a, "b": b}))
    assert "**UNRESOLVED**" in cell
    assert "`a`: only 7 zero-failure trial(s), need 20" in cell
    assert "`b`: terse arm delivered 30%" in cell


def test_a_model_that_lost_EVERY_payload_still_blocks_SAFE_for_the_cell():
    """Review of #411, predating it: `b` lost the cell's only payload to its input limit, so
    it had no rows and never got a verdict — the cell printed SAFE on `a`'s 25 clean trials.
    A trimmed corpus must withhold SAFE whether the model lost some payloads or all of them."""
    from terse.codeceval import OversizedPayload

    a = [dict(_row("q", 25, 25, trials=25, raw_calls=25, terse_calls=25),
              tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="b", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    for results in ({"a": a}, {"a": a, "b": []}):
        cell = _cell(build_codec_verdict_report(results, excluded=excluded))
        assert "**UNRESOLVED**" in cell, cell
        assert "`b`" in cell and "every payload exceeded `b`'s input limit" in cell, cell


def test_a_SAFE_model_beside_an_unresolved_one_adds_no_reason():
    a = [dict(_row("q", 7, 7, trials=7, raw_calls=7, terse_calls=7),
              tool="t", shape="array-of-records")]
    b = [dict(_row("q", 20, 20, trials=20, raw_calls=20, terse_calls=20),
              tool="t", shape="array-of-records")]
    why = _cell(build_codec_verdict_report({"a": a, "b": b})).split("|")[-2]
    assert why.strip() == "only 7 zero-failure trial(s), need 20"


def test_the_verdict_is_UNRESOLVED_exactly_when_a_reason_is_named():
    """The invariant that keeps reasons and gates from drifting apart, over every gate input.

    `codec_verdict` now DERIVES its SAFE-blocking decision from `codec_unresolved_reasons`,
    so a gate cannot be added there without its sentence. This grid is what fails if someone
    re-adds one inline anyway: it varies both arms' compliance, the trial count and the
    exclusion count, so a new gate on any of them produces a cell where the verdict and the
    reason list disagree. (The first cut varied only terse compliance; a raw-arm gate with
    no reason survived it.)

    Scoped to cells neither withheld by `arm_gap` nor UNSAFE — the reasons function is not a
    verdict and says so; `test_reasons_are_not_a_verdict_on_their_own` pins that side."""
    from terse.report import codec_unresolved_reasons

    checked = 0
    for trials in (1, 5, 20, 25):
        for raw_share in (0.0, 0.79, 0.8, 1.0):
            for terse_share in (0.0, 0.79, 0.8, 1.0):
                for dropped in (0, 1):
                    rows = [_row("q", trials, trials, trials=trials,
                                 raw_calls=round(trials * raw_share),
                                 terse_calls=round(trials * terse_share))]
                    verdict, gap = codec_verdict(rows, excluded_from_group=dropped)
                    assert not gap.excluded and verdict != "UNSAFE"
                    reasons = codec_unresolved_reasons(rows, dropped, "m")
                    assert (verdict == "UNRESOLVED") == bool(reasons), (
                        trials, raw_share, terse_share, dropped, verdict, reasons)
                    checked += 1
    assert checked == 4 * 4 * 4 * 2


def test_reasons_are_not_a_verdict_on_their_own():
    # Review of #411: the docstring claimed "empty when not UNRESOLVED", which is false for an
    # UNSAFE cell. A caller must ask `codec_verdict` whether, and this function why.
    from terse.report import codec_unresolved_reasons

    rows = [_row("q", 5, 4, trials=5, raw_calls=5, terse_calls=0)]
    assert codec_verdict(rows)[0] == "UNSAFE"
    assert codec_unresolved_reasons(rows)
