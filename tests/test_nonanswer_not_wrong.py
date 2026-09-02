"""A non-answer is not a wrong answer (#279), and a responses file has a transport gate (#283).

Both issues are one root cause in the offline `score_pack` path. The live harness has always
treated a blank/None reply as a lost call — `answerers.openai_answerer` returns `None` for a
blank reply and `harnesses._ask_n` answers that with `fails += 1; continue`, keeping it out of
BOTH the numerator and the denominator. `score_pack` did neither: `forms.get("raw", "")` fed a
form the file never collected to the scorer as `""`, and `_score_form`'s `len(replies)`
denominator counted a stored `null` as a trial the model got wrong. It also emitted no
`fails`/`attempts`, so `report._unmeasured`'s `if not attempts: return False` short-circuited
and `terse fluency --responses` published a verdict with no transport gate at all.

The fixtures here assert their OWN loss shares before asserting a gate's verdict (per the
`mutate-the-fix-to-test-the-test` rule): a fixture that quietly stopped containing empty
replies would otherwise pass every one of these vacuously.
"""

from __future__ import annotations

import json

import pytest

from terse import fluency
from terse.fluency.scoring import MISSING, _score_form
from terse.report import (
    UNMEASURED_FAIL_SHARE,
    _unmeasured,
    build_fluency_report,
    paired_rows,
)

PAYLOAD = [
    {"id": i, "status": "active-long-status-string-value", "score": i * 5}
    for i in range(1, 7)
]
# 4 questions per payload; 6 payloads clears `_MIN_PAIRED_QUESTIONS` (20) so the report
# renders real percentages instead of withholding the model as underpowered.
_SHAS = [f"s{k}" for k in range(6)]


def _pack(trials: int = 1) -> dict:
    return fluency.build_pack(
        [{"tool": "demo", "sha": sha, "raw": json.dumps(PAYLOAD)} for sha in _SHAS],
        trials=trials)


def _gt(q: dict) -> str:
    return json.dumps(q["expected"]) if q["qtype"] == "enumerate" else str(q["expected"])


def _responses(pack: dict, form_value) -> dict:
    """{sha: {qid: forms}} where `form_value(q)` builds one question's forms dict."""
    return {p["sha"]: {q["qid"]: form_value(q) for q in p["questions"]}
            for p in pack["payloads"]}


def _cells(md: str) -> list[str]:
    """The model's row of the accuracy table, split into cells.

    `['', 'model', 'q', 'raw', 'terse', 'terse+primer', 'terse+inline', 'regressions',
    'primer recovers', '']` — so index 4 is terse and index 5 is terse+primer.
    """
    line = next(x for x in md.splitlines() if x.startswith("| `m`"))
    return [c.strip() for c in line.split("|")]


# --------------------------------------------------------------------------------------
# #279 hole 1+2 — the scorer itself.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("form_val, expected, why", [
    (MISSING, (0, 0, 0), "a form the file never collected is neither an answer nor a loss"),
    ("", (0, 0, 1), "a stored empty reply is one collected call that produced no answer"),
    ("   ", (0, 0, 1), "whitespace is blank"),
    (None, (0, 0, 1), "a stored null is a recorded non-answer, not a wrong answer"),
    (123, (0, 0, 1), "a non-str reply cannot be scored"),
    ([], (0, 0, 0), "an empty list collected nothing"),
    ([None, None, "x"], (1, 1, 3), "right on every reply it produced -> 100%, not 33%"),
    (["x", "y"], (1, 2, 2), "real replies are scored normally"),
])
def test_a_non_answer_leaves_both_numerator_and_denominator(form_val, expected, why):
    assert _score_form("lookup", "x", form_val) == expected, why


def test_the_regression_this_fixes_stated_as_arithmetic():
    """Executed on the pre-fix code, `[None, None, "x"]` returned `(1, 3)` — 33%."""
    successes, scored, collected = _score_form("lookup", "x", [None, None, "x"])
    assert collected == 3 and scored == 1, "two of three replies must be non-answers"
    assert successes / scored == 1.0
    assert successes / collected == pytest.approx(1 / 3), (
        "the old denominator is still the wrong one — pinned so this test names the bug")


# --------------------------------------------------------------------------------------
# #279 hole 1 + #283 — what `score_pack` emits.
# --------------------------------------------------------------------------------------

def test_a_form_absent_from_the_file_costs_no_attempts_and_no_fails():
    pack = _pack()
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), "terse": _gt(q)})})["m"]
    assert rows
    for r in rows:
        assert r["primer_attempts"] == 0 and r["primer_trials"] == 0
        assert r["attempts"] == r["raw_attempts"] + r["terse_attempts"] == 2
        assert r["fails"] == 0, "a form nobody collected is not a lost call"


@pytest.mark.parametrize("absent", ["raw", "terse", "primer"])
def test_every_form_gets_the_absent_sentinel_not_an_empty_string(absent):
    """Per FORM, because the `""` default was written out three times.

    Fixing one call site and leaving its neighbours is exactly the drift this repo keeps
    getting bitten by, and a mutation of the `raw` default alone survived a version of this
    file that only omitted `primer`.
    """
    present = [f for f in ("raw", "terse", "primer") if f != absent]
    pack = _pack()
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {f: _gt(q) for f in present})})["m"]
    assert rows
    for r in rows:
        assert r[f"{absent}_attempts"] == 0, f"{absent} was never collected"
        assert r[f"{absent}_trials"] == 0 and r[f"{absent}_ok"] == 0
        assert all(r[f"{f}_attempts"] == 1 for f in present)
        assert r["fails"] == 0, "an uncollected form is not a lost call"
        assert r["attempts"] == 2


def test_an_empty_stored_reply_is_counted_as_a_lost_call_not_a_miss():
    pack = _pack()
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), "terse": ""})})["m"]
    assert rows
    for r in rows:
        assert r["terse_attempts"] == 1 and r["terse_trials"] == 0
        assert r["fails"] == 1, "the empty terse reply is the only lost call"
        assert r["terse_ok"] == 0


def test_a_model_right_on_every_reply_it_produced_reports_full_accuracy():
    """End to end: two of three trials come back null, the third is correct."""
    pack = _pack(trials=3)
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": [_gt(q)] * 3, "terse": [None, None, _gt(q)]})})["m"]
    assert rows
    assert all(r["terse_ok"] == 1 and r["terse_trials"] == 1 for r in rows), (
        "the fixture must actually carry two nulls per terse form")
    accuracy = sum(r["terse_ok"] for r in rows) / sum(r["terse_trials"] for r in rows)
    assert accuracy == 1.0, "scoring a non-answer as a miss would report 33% here"


# --------------------------------------------------------------------------------------
# #283 — the gate `--responses` did not have.
# --------------------------------------------------------------------------------------

def test_a_responses_file_that_is_mostly_empty_cannot_publish_a_verdict():
    pack = _pack(trials=3)
    # terse loses 2 of every 3 calls; raw is clean.
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": [_gt(q)] * 3, "terse": ["", "", _gt(q)]})})["m"]
    lost = sum(r["terse_attempts"] - r["terse_trials"] for r in rows)
    share = lost / sum(r["terse_attempts"] for r in rows)
    assert share == pytest.approx(2 / 3), "the fixture must really be losing two calls in three"
    assert share > UNMEASURED_FAIL_SHARE

    assert _unmeasured(rows), (
        "before #283 `score_pack` emitted no `attempts`, so this returned False on the "
        "first line and a half-empty responses file published a comprehension number")
    md = build_fluency_report({"m": rows}, [])
    assert "n/a | n/a | n/a" in md


def test_a_totally_empty_arm_is_withheld_but_an_uncollected_one_is_not():
    """Trigger 1's two neighbours: zero completed trials means failure only if attempted."""
    pack = _pack()
    collected_but_empty = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), "terse": ""})})["m"]
    assert all(r["terse_attempts"] == 1 for r in collected_but_empty)
    assert _unmeasured(collected_but_empty), "an arm that answered nothing it was asked"

    never_collected = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), "terse": _gt(q)})})["m"]
    assert all(r["primer_attempts"] == 0 for r in never_collected)
    assert not _unmeasured(never_collected), (
        "a pack that simply never ran the primer arm must not withhold the whole model")


# --------------------------------------------------------------------------------------
# The trap: emitting `attempts` must not delete #91's uneven collection mode.
# --------------------------------------------------------------------------------------

def test_a_real_uneven_score_pack_with_counters_still_publishes():
    """3 raw replies and 2 terse ones for the same question is COLLECTION DESIGN, not loss.

    `test_an_uneven_score_pack_still_publishes` pins the same rule on hand-built rows that
    carry no `attempts` key. This one goes through `score_pack` itself, which now DOES emit
    `attempts` — so the escape that protects the hand-built rows cannot protect these, and
    only the per-arm `<arm>_attempts` counters keep the row paired.
    """
    pack = _pack(trials=3)
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": [_gt(q)] * 3, "terse": [_gt(q)] * 2})})["m"]
    assert rows
    assert all(r["raw_attempts"] == 3 and r["terse_attempts"] == 2 for r in rows), (
        "the fixture must really be uneven, or this test pins nothing")
    assert all(r["fails"] == 0 for r in rows), "an uneven form is not a lost call"
    assert all(r["trials"] == 3 for r in rows), "`trials` is still the across-form max"

    assert paired_rows(rows, "terse_ok", "raw_ok") == rows, (
        "reading `trials` as the terse arm's attempt count would void every row")
    assert not _unmeasured(rows)
    md = build_fluency_report({"m": rows}, [])
    assert "n/a | n/a | n/a" not in md, "an uneven pack was withheld as if calls were lost"
    # The TERSE cell specifically, not `"100%" in md` — the raw column reads 100% either
    # way, so that assertion passed even with `_form_stats`' per-form denominator deleted
    # (the whole of #91, which this test's docstring is about). Terse answered 2 of 2, so
    # 100%; dividing by the row `trials` of 3 would print 67%.
    assert _cells(md)[4].startswith("100%"), "terse must be scored over its own 2 replies"


# --------------------------------------------------------------------------------------
# The regression fixing #279 makes possible: an arm collected for only SOME questions.
# Found in adversarial review of this change, before it was committed.
# --------------------------------------------------------------------------------------

def test_an_arm_collected_for_only_some_questions_cannot_publish_against_a_full_control():
    """This is a FALSE PASS the #279 fix opens, and the most dangerous shape in the file.

    Scoring an uncollected form as a miss (the #279 defect) was accidentally covering it: the
    18 uncollected questions scored 0 and the arm read 25%, a loud FAIL. Once they stop being
    scored as misses they leave the denominator entirely, and the arm is re-based onto the
    questions it happens to have been collected on — here the six `count` questions, the
    easiest type — while `raw` keeps all 24. Measured before the `paired_rows` fix:
    `best terse-form 100% vs raw 100% ... **PASS**`, with `18 regressions` in the same row.

    A conservative wrong answer traded for a confident green is strictly worse than the bug.
    """
    pack = _pack()
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), **({"terse": _gt(q)} if q["qtype"] == "count" else {})},
    )})["m"]
    with_terse = [r for r in rows if r["terse_attempts"]]
    assert len(rows) == 24 and len(with_terse) == 6, (
        "the fixture must really collect terse for a MINORITY of the questions")
    assert all(r["raw_attempts"] == 1 for r in rows), "the control keeps every question"

    assert not _unmeasured(rows), (
        "no call was lost — every arm answered everything it was asked, so the transport "
        "gate cannot be what saves this and pairing has to")
    assert len(paired_rows(rows, "terse_ok", "primer_ok", "raw_ok")) == 6, (
        "the 18 questions terse never answered are not comparable and must be voided")

    md = build_fluency_report({"m": rows}, [])
    assert "**PASS**" not in md, "a terse arm scored on 6 of 24 questions must not publish"
    assert "Not concluded" in md


def test_a_score_pack_row_with_a_real_partial_loss_is_still_voided_by_pairing():
    """The other direction of the same predicate — and a mutation that survived the suite.

    Deleting pairing for every row that states `<arm>_attempts` (i.e. trusting the counters
    to have caught it) left all 1766 tests green, because nothing observed a `score_pack` row
    with a genuine sub-threshold loss. Here terse loses one of three calls on the `enumerate`
    questions only: 6 of 72 terse calls, 8.3% — comfortably under `UNMEASURED_FAIL_SHARE`, so
    `_unmeasured` stays quiet by design and pairing is the only thing left.
    """
    pack = _pack(trials=3)
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": [_gt(q)] * 3,
                         "terse": [_gt(q), _gt(q), None] if q["qtype"] == "enumerate"
                         else [_gt(q)] * 3})})["m"]
    lossy = [r for r in rows if r["terse_trials"] < r["terse_attempts"]]
    assert len(rows) == 24 and len(lossy) == 6, "the fixture must really lose calls"
    share = (sum(r["terse_attempts"] - r["terse_trials"] for r in rows)
             / sum(r["terse_attempts"] for r in rows))
    assert share == pytest.approx(6 / 72)
    assert share < UNMEASURED_FAIL_SHARE, (
        "if the loss tripped the transport gate this test would pass without pairing")
    assert not _unmeasured(rows)

    kept = paired_rows(rows, "terse_ok", "primer_ok", "raw_ok")
    assert len(kept) == 18 and not any(r in lossy for r in kept), (
        "a row where terse answered 2 of the 3 calls it was asked is not comparable")


# --------------------------------------------------------------------------------------
# Absence renders as absence, in every table.
# --------------------------------------------------------------------------------------

def test_an_arm_the_file_never_collected_renders_na_not_a_confident_zero():
    """`_form_stats` returns `(0.0, 0.0)` for an empty sample, which prints `0% ±0`.

    That is a confident zero off no evidence, and it reads as "primer comprehension
    collapsed" rather than "nobody ran the primer arm" — the exact mistake the inline arm's
    `n/a` already avoids. The by-transform table matters at least as much: its own comment
    says a reader uses it to "restrict the policy to the transforms that held".
    """
    pack = _pack()
    rows = fluency.score_pack(pack, {"m": _responses(
        pack, lambda q: {"raw": _gt(q), "terse": _gt(q)})})["m"]
    assert all(r["primer_attempts"] == 0 for r in rows), "primer must really be uncollected"

    md = build_fluency_report({"m": rows}, [])
    cells = _cells(md)
    assert cells[4].startswith("100%"), "terse WAS collected and must still publish"
    assert cells[5] == "n/a", "terse+primer was never collected"
    assert cells[8] == "n/a", "'primer recovers' cannot be 0 for an arm nobody asked"
    transform_rows = [x for x in md.splitlines() if x.startswith("| table")]
    assert transform_rows, "the by-transform table must still render"
    assert all(x.rstrip().endswith("| n/a |") for x in transform_rows), transform_rows


def test_a_live_row_set_without_the_counters_still_publishes_its_primer_arm():
    """The fallback the `n/a` rule rests on: no `<arm>_attempts` key means fully attempted.

    Every live harness row and every legacy result file is this shape, and reading the
    absent counter as "never collected" would blank the primer column of all of them.
    """
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 1, "raw_ok": 1, "terse_ok": 1, "primer_ok": 1} for i in range(24)]
    assert not any("primer_attempts" in r for r in rows)
    cells = _cells(build_fluency_report({"m": rows}, []))
    assert cells[5].startswith("100%") and cells[8] == "0"


# --------------------------------------------------------------------------------------
# Mutation checks (per memory `mutate-the-fix-to-test-the-test`). Each was applied by hand,
# the suite re-run, and the mutation reverted — recorded here so the next person does not
# have to rediscover which test catches which. Fifteen mutations, all KILLED; none survived.
#
# The scorer (fluency/scoring.py, fluency/pack.py):
#   1. `forms.get("<form>", MISSING)` -> `forms.get("<form>", "")`, once per form (3 runs)
#      -> reddens `test_every_form_gets_the_absent_sentinel_not_an_empty_string[<form>]`.
#      The per-form parametrisation is not decoration: mutating only the `raw` default
#      SURVIVED an earlier version of this file whose absent-form fixture omitted `primer`.
#   2. `_score_form` back to `len(replies)` as the denominator with non-str replies merely
#      skipped in the numerator -> reddens the parametrised non-answer test,
#      `test_the_regression_this_fixes_stated_as_arithmetic`, and the end-to-end
#      `test_a_model_right_on_every_reply_it_produced_reports_full_accuracy`.
#   3. drop `fails`/`attempts` from the emitted row -> reddens
#      `test_a_responses_file_that_is_mostly_empty_cannot_publish_a_verdict` (this is #283
#      exactly: `_unmeasured` short-circuits on `if not attempts` and publishes).
#   4. drop the per-arm `<arm>_attempts` keys -> reddens the two gate tests and
#      `test_a_real_uneven_score_pack_with_counters_still_publishes`.
#
# The gate (report.py):
#   5. `_paired_arm` compares each arm against the row `trials` instead of its own attempts
#      -> reddens `test_a_real_uneven_score_pack_with_counters_still_publishes` alone (the
#      trap: a design-uneven pack voided as if calls were lost).
#   6. `_unmeasured`'s per-arm trigger uses `trials` instead of `_arm_attempts`.
#   7. `_unmeasured` trigger 1 fires unconditionally on a zero-trial arm -> a pack that
#      never collected the primer form would withhold the whole model.
#   8. `_paired_arm` drops the run-level rule and keeps every zero-attempt arm
#      -> reddens `test_an_arm_collected_for_only_some_questions_cannot_publish_against_a_
#      full_control` ALONE. That is the false PASS this fix exists for, and #8 is the single
#      mutation that reaches it — the one test standing between the #279 fix and a green
#      verdict off six of twenty-four questions.
#   9. `_paired_arm` voids EVERY zero-attempt arm (absence read as loss) -> the opposite
#      direction, reddens the uncollected-primer and uneven-pack tests.
#  10. `_paired_arm` returns True for any row stating `<arm>_attempts` (i.e. trust the
#      counters, never pair) -> reddens
#      `test_a_score_pack_row_with_a_real_partial_loss_is_still_voided_by_pairing`. This
#      mutation SURVIVED the entire 1766-test suite before that test existed; adversarial
#      review found it, not the suite.
#  11. `arm_measured` returns True always -> a never-collected arm renders `0% ±0`.
#  12. `arm_measured` filters to rows carrying the key -> reddens
#      `test_a_live_row_set_without_the_counters_still_publishes_its_primer_arm` and
#      `test_fluency.py::test_fluency_report_renders_inline_column_when_rows_carry_it`.
#      This one was a real bug in the first cut, not a hypothetical: it blanked the primer
#      column of every legacy result file.
