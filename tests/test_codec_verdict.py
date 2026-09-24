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
def test_a_single_worse_question_is_not_UNSAFE_but_it_blocks_SAFE():
    # Sign test (replacing zero tolerance): one discordant question is p=0.5 — a
    # non-deterministic reader produces that from noise alone. It can never read SAFE either:
    # terse leaned worse, which `codec_unresolved_reasons` names.
    rows = [_row(f"q{i}", 1, 1) for i in range(50)]
    rows.append(_row("q-bad", 1, 0))
    verdict, gap = codec_verdict(rows)
    assert verdict == "UNRESOLVED"


def test_consistent_harm_across_questions_is_UNSAFE():
    # Five questions all worse, none better: p = 1/32 < 0.05.
    rows = [_row(f"q{i}", 1, 1) for i in range(20)]
    rows += [_row(f"bad{i}", 1, 0) for i in range(5)]
    assert codec_verdict(rows)[0] == "UNSAFE"


def test_noise_on_both_arms_is_not_UNSAFE():
    # The defect the sign test fixes: the same 50% reader on both arms, flipping per question.
    rows = [_row(f"w{i}", 2, 1, trials=3) for i in range(3)]
    rows += [_row(f"b{i}", 1, 2, trials=3) for i in range(3)]
    assert codec_verdict(rows)[0] != "UNSAFE"


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
    rows = [_row(f"q{i}", raw_ok=20, terse_ok=20, trials=25) for i in range(5)]
    verdict, gap = codec_verdict(rows)
    assert verdict == "SAFE"


def test_a_genuine_terse_specific_regression_is_UNSAFE_even_when_raw_is_imperfect():
    # raw succeeds MORE often than terse on the same row — a real excess, not just raw's
    # own baseline imperfection. Must still be UNSAFE.
    rows = [_row(f"q{i}", raw_ok=20, terse_ok=15, trials=25) for i in range(5)]
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


def test_significant_harm_beats_an_otherwise_unresolved_thin_sample():
    # Below the trial floor, but five questions all worse: significance is dispositive;
    # sample size only gates a clean run.
    rows = [_row(f"q{i}", 1, 0) for i in range(5)]
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


def test_a_SAFE_cell_scopes_its_compliance_to_the_run_too():
    """#412's silent direction: a run that happened to comply licenses SAFE, and the cell
    said nothing about the rate that let it through. The same (model, payload, question)
    read 29% and then 100%, so a passing rate is one run's as much as a failing one."""
    rows = [_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=1) for i in range(_CODEC_MIN_TRIALS)]
    cell = next(ln for ln in build_codec_verdict_report(
        {"m": _tagged(rows, "t", "array-of-records")}).splitlines() if ln.startswith("| `t` |"))
    assert "**SAFE**" in cell
    assert "tool-call compliance raw 100%, terse 100% in this run" in cell
    assert "not measured across runs" in cell


def test_a_SAFE_cell_without_compliance_counters_claims_nothing_about_them():
    """Pre-#403 rows never measured compliance; the caveat must not invent a rate."""
    rows = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    cell = next(ln for ln in build_codec_verdict_report(
        {"m": _tagged(rows, "t", "array-of-records")}).splitlines() if ln.startswith("| `t` |"))
    assert "**SAFE**" in cell and "compliance" not in cell


def _safe_cell(results: dict) -> str:
    cell = next(ln for ln in build_codec_verdict_report(results).splitlines()
                if ln.startswith("| `t` |"))
    assert "**SAFE**" in cell, cell
    return cell


def _tagged_t(rows):
    return _tagged(rows, "t", "array-of-records")


def test_a_multi_model_SAFE_cell_quotes_the_lowest_passing_rate_whatever_the_names():
    """Review of #432: the caveat read the tie-break model (first by NAME), so identical
    evidence printed terse 100% or 80%. The weakest model is what let the cell through."""
    full = [_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=1) for i in range(20)]
    floor = [_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=int(i >= 4)) for i in range(20)]
    for strong in ("a", "z"):                       # sorts before AND after `b`
        cell = _safe_cell({strong: _tagged_t(full), "b": _tagged_t(floor)})
        # raw ties at 100% across both models: nobody is singled out by sort order.
        assert "compliance raw 100%, terse 80% (`b`) in this run" in cell, cell


def test_a_partly_counted_model_withholds_the_cell_rate_rather_than_vanishing():
    """Review 2 of #432: `b` passed the gate on 8/10 counted rows plus 10 uncounted ones.
    Skipping `b` and quoting `a`'s 100% claimed more compliance than `b`'s evidence."""
    full = [_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=1) for i in range(20)]
    part = ([_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=int(i >= 2)) for i in range(10)]
            + [_row(f"p{i}", 1, 1) for i in range(10)])
    assert "compliance" not in _safe_cell({"a": _tagged_t(full), "b": _tagged_t(part)})


def test_a_near_full_SAFE_rate_never_prints_as_full():
    """2499/2500 is one prose answer, not full compliance; `:.0%` rounded it to 100%, and
    so would a one-decimal ROUND (100.0%) — only truncation keeps it under (99.9%)."""
    rows = [_row(f"q{i}", 125, 125, trials=125, raw_calls=125, terse_calls=125 - (i == 0))
            for i in range(20)]
    cell = _safe_cell({"m": _tagged_t(rows)})
    assert "terse 99.9%" in cell and "terse 100" not in cell, cell


def test_a_SAFE_cell_mixing_old_and_new_rows_claims_no_rate():
    """`codec_call_rate` counts only the rows carrying the counter; quoting that as the
    cell's rate would describe 10 of 20 trials as all of them."""
    rows = ([_row(f"q{i}", 1, 1, raw_calls=1, terse_calls=1) for i in range(10)]
            + [_row(f"p{i}", 1, 1) for i in range(10)])
    assert "compliance" not in _safe_cell({"m": _tagged_t(rows)})


def test_report_gates_on_the_worst_model_within_a_group():
    clean = [_row(f"q{i}", 1, 1) for i in range(_CODEC_MIN_TRIALS)]
    broken = [_row(f"q-bad{i}", 1, 0) for i in range(5)]
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
                 raw_calls=per_arm, terse_calls=at_floor) for i in range(5)]
    assert codec_verdict(rows)[0] == "SAFE"
    one_under = [_row(f"q{i}", per_arm, per_arm, trials=per_arm,
                      raw_calls=per_arm, terse_calls=at_floor - 1) for i in range(5)]
    assert codec_verdict(one_under)[0] == "UNRESOLVED"


def test_low_compliance_does_NOT_suppress_an_UNSAFE_verdict():
    # The direction that matters. Values are scored on whichever channel they arrive by, so
    # an observed excess is real regardless of compliance — withholding it would suppress
    # the finding this tier exists to make.
    rows = [_row(f"q{i}", 2, 2, trials=2, raw_calls=0, terse_calls=0) for i in range(20)]
    rows += [_row(f"q-bad{i}", 2, 0, trials=2, raw_calls=0, terse_calls=0) for i in range(5)]
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
    assert "trial(s), need" not in out


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
    # #412: one run's rate, scoped as one run's — the same cell later read 100% twice.
    assert "delivered 29% of its answers through the tool call in this run" in cell
    assert "only 7 trial(s), need 20" in cell


def test_an_unresolved_cell_names_the_trimmed_corpus_AND_the_trial_floor():
    from terse.codeceval import OversizedPayload

    rows = [dict(_row("q", 3, 3, trials=3, raw_calls=3, terse_calls=3),
                 tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    cell = _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))
    assert "trimmed corpus" in cell
    assert "only 3 trial(s)" in cell


def test_all_three_reasons_at_once_are_all_named():
    from terse.codeceval import OversizedPayload

    rows = [dict(_row("q", 5, 5, trials=5, raw_calls=5, terse_calls=0),
                 tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    cell = _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))
    assert cell.count(";") == 3, cell
    for part in ("trimmed corpus", "delivered 0%", "only 5 trial(s)",
                 "only 1 complete question(s)"):
        assert part in cell, part


def test_a_single_reason_is_still_rendered_alone():
    # Full compliance and nothing excluded — only the trial floor fails. The join must not
    # invent a separator around a lone reason.
    rows = [dict(_row(f"q{i}", 1, 1, trials=1, raw_calls=1, terse_calls=1),
                 tool="t", shape="array-of-records") for i in range(5)]
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "only 5 trial(s), need 20" in cell
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
    assert "`a`: only 7 trial(s), need 20" in cell
    assert "`b`: terse arm delivered 30%" in cell


def test_every_column_speaks_for_every_unresolved_model_once_why_does():
    """Review of #411: with the Why column naming several models, Questions and n still
    printed only the tie-break winner's rows — "n=0" beside "`b`: only 7 zero-failure
    trial(s)". Every column of the row now names the same models."""
    from terse.report import REASON_LABEL

    dead = [dict(_row("q", 0, 0, trials=20), raw_trials=0, terse_trials=0, fails=40,
                 attempts=40, tool="t", shape="array-of-records")]
    thin = [dict(_row("q1", 7, 7, trials=7, raw_calls=7, terse_calls=7), tool="t",
                 shape="array-of-records"),
            dict(_row("q2", 7, 7, trials=7, raw_calls=7, terse_calls=7), tool="t",
                 shape="array-of-records", qtype="enumerate")]
    safe = [dict(_row(f"q{i}", 25, 25, trials=25, raw_calls=25, terse_calls=25),
                 tool="t", shape="array-of-records") for i in range(5)]
    cols = [c.strip() for c in
            _cell(build_codec_verdict_report({"a": dead, "b": thin, "c": safe})).split("|")]
    _, _, _, questions, n, verdict, models, why, _ = cols
    label = REASON_LABEL.get(codec_verdict(dead)[1].excluded)
    assert verdict == "**UNRESOLVED**"
    assert questions == "`a`: — · `b`: deref 1, enumerate 1"
    assert n == "`a`: 0 · `b`: 14"
    assert models == "`a`, `b`"  # the SAFE model is in no column
    assert why == (f"`a`: {label} · `b`: only 14 trial(s), need 20; "
                   "only 2 complete question(s), need 5 — fewer could never show harm")


def test_a_single_unresolved_model_keeps_the_plain_columns():
    thin = [dict(_row("q", 7, 7, trials=7, raw_calls=7, terse_calls=7),
                 tool="t", shape="array-of-records")]
    safe = [dict(_row(f"q{i}", 25, 25, trials=25, raw_calls=25, terse_calls=25),
                 tool="t", shape="array-of-records") for i in range(5)]
    cols = [c.strip() for c in
            _cell(build_codec_verdict_report({"a": safe, "b": thin})).split("|")]
    assert cols[3:7] == ["deref 1", "7", "**UNRESOLVED**", "`b`"]


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


def test_a_cell_withheld_by_arm_gap_names_that_not_a_thin_sample():
    """Review of #411: this branch had no rendering test — replacing it with the trial-floor
    reasons survived the full suite, reporting a dead backend as "only 0 zero-failure
    trial(s)". Alone, and beside a model that is short on trials."""
    from terse.report import REASON_LABEL

    dead = [dict(_row("q", 0, 0, trials=20), raw_trials=0, terse_trials=0, fails=40,
                 attempts=40, tool="t", shape="array-of-records")]
    thin = [dict(_row("q", 7, 7, trials=7, raw_calls=7, terse_calls=7),
                 tool="t", shape="array-of-records")]
    gap = codec_verdict(dead)[1]
    assert gap.excluded
    label = REASON_LABEL.get(gap.excluded, gap.excluded)

    why = _cell(build_codec_verdict_report({"a": dead})).split("|")[-2].strip()
    assert why == label
    why = _cell(build_codec_verdict_report({"a": dead, "b": thin})).split("|")[-2].strip()
    assert why == (f"`a`: {label} · `b`: only 7 trial(s), need 20; "
                   "only 1 complete question(s), need 5 — fewer could never show harm")


def test_a_withheld_model_that_ALSO_lost_a_payload_names_both():
    # Review of #411: the withheld branch printed only its label, so fixing the backend alone
    # would still leave the cell UNRESOLVED on a trimmed corpus nobody was told about.
    from terse.codeceval import OversizedPayload
    from terse.report import REASON_LABEL

    dead = [dict(_row("q", 0, 0, trials=20), raw_trials=0, terse_trials=0, fails=40,
                 attempts=40, tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="a", tool="t", shape="array-of-records", sha="s2",
                                 arm="raw", tokens=99, limit=10)]
    label = REASON_LABEL.get(codec_verdict(dead)[1].excluded)
    why = _cell(build_codec_verdict_report({"a": dead}, excluded=excluded)).split("|")[-2]
    assert why.strip() == (f"{label}; 1 payload(s) not asked of `a` — over its input "
                           f"limit, so this cell was scored on a trimmed corpus")


def test_a_payload_over_the_limit_on_BOTH_arms_counts_as_one_payload():
    # Review of #411: one `OversizedPayload` is emitted per ARM, and the reason counted
    # records — a single payload with both arms over read "2 payload(s) not asked".
    from terse.codeceval import OversizedPayload

    rows = [dict(_row("q", 25, 25, trials=25, raw_calls=25, terse_calls=25),
                 tool="t", shape="array-of-records")]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s1",
                                 arm=arm, tokens=99, limit=10) for arm in ("raw", "terse")]
    excluded.append(OversizedPayload(model="m", tool="t", shape="array-of-records",
                                     sha="s2", arm="raw", tokens=99, limit=10))
    cell = _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))
    assert "2 payload(s) not asked of `m`" in cell, cell


def test_a_rate_just_below_the_floor_never_prints_as_the_floor():
    # Review of #411: 199/250 is 79.6%, and `.0%` printed "delivered 80% ... need 80%".
    rows = [dict(_row("q", 250, 250, trials=250, raw_calls=250, terse_calls=199),
                 tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "terse arm delivered 79.6%" in cell, cell
    rows[0]["terse_calls"] = 1999
    rows[0].update(raw_ok=2500, terse_ok=2500, trials=2500, raw_trials=2500,
                   terse_trials=2500, raw_calls=2500, attempts=5000)
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "terse arm delivered 79.9%" in cell, cell  # 79.96%: truncated, not rounded up


def test_a_SAFE_model_beside_an_unresolved_one_adds_no_reason():
    a = [dict(_row(f"q{i}", 1, 1, trials=1, raw_calls=1, terse_calls=1),
              tool="t", shape="array-of-records") for i in range(5)]
    b = [dict(_row(f"q{i}", 20, 20, trials=20, raw_calls=20, terse_calls=20),
              tool="t", shape="array-of-records") for i in range(5)]
    why = _cell(build_codec_verdict_report({"a": a, "b": b})).split("|")[-2]
    assert why.strip() == "only 5 trial(s), need 20"


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
    for trials in (1, 5, 19, 20, 25):
        # Call COUNTS, not shares: `round(t * 0.79)` lands ON the floor for every t tried
        # (review of #411), so the grid never probed just-below. These are the largest count
        # under the floor and the smallest at it, computed from the constant.
        below = max(c for c in range(trials + 1) if c / trials < _CODEC_MIN_CALL_RATE)
        at = min(c for c in range(trials + 1) if c / trials >= _CODEC_MIN_CALL_RATE)
        counts = sorted({0, below, at, trials})
        for raw_calls in counts:
            for terse_calls in counts:
                for dropped in (0, 1):
                    rows = [_row("q", trials, trials, trials=trials,
                                 raw_calls=raw_calls, terse_calls=terse_calls)]
                    verdict, gap = codec_verdict(rows, excluded_from_group=dropped)
                    assert not gap.excluded and verdict != "UNSAFE"
                    reasons = codec_unresolved_reasons(rows, dropped, "m")
                    assert (verdict == "UNRESOLVED") == bool(reasons), (
                        trials, raw_calls, terse_calls, dropped, verdict, reasons)
                    checked += 1
    assert checked >= 100, "the grid collapsed — the invariant was checked on too few cells"


def test_reasons_are_not_a_verdict_on_their_own():
    # Review of #411: the docstring claimed "empty when not UNRESOLVED", which is false for an
    # UNSAFE cell. A caller must ask `codec_verdict` whether, and this function why.
    from terse.report import codec_unresolved_reasons

    rows = [_row(f"q{i}", 5, 4, trials=5, raw_calls=5, terse_calls=0) for i in range(5)]
    assert codec_verdict(rows)[0] == "UNSAFE"
    assert codec_unresolved_reasons(rows)


# --------------------------------------------------------------------------- #
# Review of the sign test: lost calls, every model described, question floor
# --------------------------------------------------------------------------- #
def _lost_row(qid, raw_ok, terse_ok, trials, raw_answered, terse_answered):
    r = _row(qid, raw_ok, terse_ok, trials=trials)
    r.update(raw_answered=raw_answered, terse_answered=terse_answered,
             fails=(trials - raw_answered) + (trials - terse_answered))
    return r


def test_a_lost_raw_call_cannot_buy_a_SAFE_against_real_harm():
    # q-bad: terse wrong every time. q-lost: one raw call LOST (scored as a raw miss), which
    # made it look like terse did better. Counted, it cancelled q-bad and the cell read SAFE.
    rows = [_row(f"q{i}", 7, 7, trials=7) for i in range(5)]
    rows.append(_row("q-bad", 7, 0, trials=7))
    rows.append(_lost_row("q-lost", 6, 7, 7, raw_answered=6, terse_answered=7))
    verdict, _g = codec_verdict(rows)
    assert verdict != "SAFE"
    from terse.report import codec_sign
    # SAFE side: the lost raw call is charged as a raw success, so q-lost is a tie, not a
    # "better" that cancels q-bad (fix plan D2).
    assert codec_sign(rows, "safe")[:2] == (1, 0)


def test_every_verdict_names_lost_calls():
    safe = [dict(_row(f"q{i}", 20, 20, trials=20), tool="t", shape="array-of-records")
            for i in range(5)]
    safe.append(dict(_lost_row("q-lost", 19, 20, 20, 19, 20), tool="t",
                     shape="array-of-records"))
    assert "1 of " in _cell(build_codec_verdict_report({"m": safe}))
    thin = [dict(_lost_row("q", 3, 2, 3, 3, 2), tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"m": thin}))
    assert "**UNRESOLVED**" in cell and "1 of 6 calls lost" in cell


def test_a_SAFE_row_describes_every_model_not_the_first_name():
    a = [dict(_row(f"q{i}", 20, 20, trials=20), tool="t", shape="array-of-records")
         for i in range(5)]
    b = [dict(_row(f"q{i}", 20, 20, trials=20), tool="t", shape="array-of-records")
         for i in range(5)]
    b += [dict(_row("w1", 20, 19, trials=20), tool="t", shape="array-of-records"),
          dict(_row("b1", 19, 20, trials=20), tool="t", shape="array-of-records")]
    cell = _cell(build_codec_verdict_report({"a": a, "b": b}))
    assert "**SAFE**" in cell
    assert "`b`: worse on 1 question(s), better on 1" in cell


def test_significant_harm_survives_a_trimmed_corpus():
    # Restores what test_an_exclusion_never_suppresses_an_UNSAFE_that_survived pinned before
    # the sign test: evidence that survived the trim is still UNSAFE.
    from terse.codeceval import OversizedPayload
    rows = [dict(_row(f"q{i}", 7, 0, trials=7), tool="t", shape="array-of-records")
            for i in range(5)]
    excluded = [OversizedPayload(model="m", tool="t", shape="array-of-records", sha="s",
                                 arm="raw", tokens=99, limit=10)]
    assert "**UNSAFE**" in _cell(build_codec_verdict_report({"m": rows}, excluded=excluded))


# --------------------------------------------------------------------------- #
# Fix plan D1-D3: the dual review's reproductions, verbatim
# --------------------------------------------------------------------------- #
def _answered_row(qid, raw_ok, terse_ok, trials, raw_ans=None, terse_ans=None, **extra):
    raw_ans = trials if raw_ans is None else raw_ans
    terse_ans = trials if terse_ans is None else terse_ans
    r = _row(qid, raw_ok, terse_ok, trials=trials)
    r.update(raw_answered=raw_ans, terse_answered=terse_ans,
             fails=(trials - raw_ans) + (trials - terse_ans), **extra)
    return r


def test_one_lost_terse_call_on_a_harmed_question_cannot_buy_SAFE():
    rows = [_answered_row(f"c{i}", 4, 4, 4) for i in range(5)]
    rows.append(_answered_row("harm", 4, 0, 4, terse_ans=3))
    assert codec_verdict(rows)[0] != "SAFE"


def test_one_lost_call_per_question_cannot_hide_a_significant_UNSAFE():
    rows = [_answered_row(f"q{i}", 20, 0, 20, terse_ans=19) for i in range(6)]
    assert codec_verdict(rows)[0] == "UNSAFE"


def test_never_answered_trials_do_not_count_toward_the_trial_floor():
    rows = [_answered_row(f"c{i}", 3, 3, 3) for i in range(6)]
    assert codec_verdict(rows)[0] == "UNRESOLVED"                   # 18 < 20
    rows.append(_answered_row("dead", 0, 0, 3, raw_ans=0, terse_ans=0))
    assert codec_verdict(rows)[0] == "UNRESOLVED", "calls that never happened bought a SAFE"


def test_total_corruption_on_one_question_is_UNSAFE_at_any_cell_size():
    assert codec_verdict([_answered_row("q", 20, 0, 20)])[0] == "UNSAFE"


def test_noise_cannot_cancel_two_always_wrong_questions():
    rows = [_answered_row(f"bad{i}", 20, 0, 20) for i in range(2)]
    rows += [_answered_row(f"n{i}", 10, 11, 20) for i in range(3)]
    assert codec_verdict(rows)[0] == "UNSAFE"


def test_an_unparsed_text_reply_is_unanswered_not_corruption():
    # The primed terse arm explains instead of replying with bare JSON on every trial. That
    # is a format problem the primer caused, not a wrong value: UNRESOLVED, never UNSAFE.
    rows = [_answered_row(f"q{i}", 20, 0, 20, channel="text",
                          raw_parsed=20, terse_parsed=0) for i in range(6)]
    verdict, _g = codec_verdict(rows)
    assert verdict == "UNRESOLVED"
    from terse.report import codec_unresolved_reasons
    assert any("terse arm replied with a bare JSON value 0%" in r
               for r in codec_unresolved_reasons(rows))


def test_a_text_reader_that_rarely_replies_in_json_cannot_read_SAFE():
    rows = [_answered_row(f"q{i}", 1, 1, 20, channel="text",
                          raw_parsed=1, terse_parsed=1) for i in range(5)]
    assert codec_verdict(rows)[0] == "UNRESOLVED"


def test_an_UNSAFE_row_names_lost_calls_too():
    rows = [dict(_answered_row(f"q{i}", 20, 0, 20, terse_ans=19), tool="t",
                 shape="array-of-records") for i in range(6)]
    cell = _cell(build_codec_verdict_report({"m": rows}))
    assert "**UNSAFE**" in cell and "6 of 240 calls lost" in cell


def test_the_n_column_counts_answered_trials_only():
    # Pins the unit directly: a lost-call row leans worse on the SAFE side anyway, so the
    # verdict alone could not show whether the floor counted the never-answered trials.
    rows = [dict(_answered_row(f"c{i}", 3, 3, 3), tool="t", shape="array-of-records")
            for i in range(6)]
    rows.append(dict(_answered_row("dead", 0, 0, 3, raw_ans=0, terse_ans=0), tool="t",
                     shape="array-of-records"))
    cols = [c.strip() for c in _cell(build_codec_verdict_report({"m": rows})).split("|")]
    assert cols[4] == "18"


def test_a_lean_worse_reason_says_how_much_of_it_is_lost_calls():
    from terse.report import codec_unresolved_reasons
    rows = [_answered_row(f"c{i}", 5, 5, 5) for i in range(5)]
    rows.append(_answered_row("lost", 4, 4, 5, raw_ans=4))   # 4 v 4 seen; +1 lost raw -> "worse"
    assert any("1 of those only by counting lost calls against terse" in r
               for r in codec_unresolved_reasons(rows))
