"""#292: the inline arm is scored over the questions IT answered, and now says so.

`terse+inline` is a DISPLAY arm — it does not gate the pairing, deliberately, because
its prompt is the longest of the four (`run_payload` prefixes the whole primer to the
terse text) and pairing on it would void otherwise-complete runs over a column no verdict
consumes. `_gap`'s docstring makes that call and it stands.

The cost `_gap` did not account for: because inline does not gate, the paired subset keeps
questions inline never answered, and `_form_stats` divides each arm by its own
`<arm>_trials`. A question inline lost entirely therefore left inline's exam while staying
in every other arm's — and the cell printed the resulting accuracy beside columns over a
larger exam with nothing saying the exams differed.

Reproduced before the fix: 19 questions every arm aces plus one hard question every arm
fails, with inline's call lost on that hard one, published

    raw 95%   terse 95%   terse+primer 95%   terse+inline 100%

Note precisely what changed, because it is easy to oversell. Dropping a fully-lost row
changes the ACCURACY by nothing — `_form_stats` sums k and t, so a 0/0 row already
contributed zero to both, and the number was always over the questions inline answered.
What was missing was saying so. These tests pin the reported exam size, and pin that the
accuracy is deliberately untouched.
"""

from __future__ import annotations

from terse.report import (
    FLUENCY_CONTROL,
    FLUENCY_GATING,
    best_arm_gap,
    build_fluency_report,
)


def _row(qid: str, ok: int, inline_trials: int = 5) -> dict:
    """One question, five trials, every arm scoring `ok` — except that inline was given
    only `inline_trials` of them (a token-budget stop on the longest prompt)."""
    return {"qid": qid, "qtype": "deref", "transform": "identity", "trials": 5,
            "raw_ok": ok, "terse_ok": ok, "primer_ok": ok, "inline_ok": ok,
            "raw_trials": 5, "terse_trials": 5, "primer_trials": 5,
            "inline_trials": inline_trials,
            "fails": 5 - inline_trials, "attempts": 20}


def _fleet(inline_trials: int) -> list[dict]:
    """19 easy questions every arm aces, plus one HARD question every arm gets wrong —
    the one whose long inline prompt is the first to truncate."""
    return [_row(f"q{i}", 5) for i in range(19)] + [_row("hard", 0, inline_trials)]


def _gap(rows: list[dict]):
    return best_arm_gap(rows, list(FLUENCY_GATING), FLUENCY_CONTROL, ("inline_ok",))


def test_a_question_inline_never_answered_shrinks_the_exam_it_reports():
    """The defect. The hard question stays in the paired subset — the three gating arms
    completed every trial of it — so `raw`/`terse`/`primer` are scored over 20 questions
    while inline is scored over the 19 it answered."""
    g = _gap(_fleet(inline_trials=0))
    assert len(g.rows) == 20                       # inline does not gate: nothing voided
    assert g.display_n["inline_ok"] == 19          # but its own exam is smaller
    assert g.arms["raw_ok"][0] == 0.95
    # Deliberately unchanged: a 0/0 row never contributed to either sum.
    assert g.arms["inline_ok"][0] == 1.0


def test_the_cell_states_inlines_own_q_only_when_it_is_smaller():
    """The number is only honest if the reader can see which exam it is over. Rendered on
    attrition and NOT otherwise — a marker every row carries is one nobody reads."""
    degraded = build_fluency_report({"m": _fleet(inline_trials=0)}, [])
    healthy = build_fluency_report({"m": _fleet(inline_trials=5)}, [])
    assert "100% ±0 (q=19)" in degraded
    assert "(q=" not in healthy
    # The gating arms' q is untouched — this did not quietly become "pair on inline too",
    # the fix `_gap`'s docstring rejects.
    assert "| `m` | 20 |" in degraded and "| `m` | 20 |" in healthy


def test_a_PARTIAL_loss_keeps_the_question_and_only_costs_precision():
    """An arm that answered 3 trials of 5 still sat that question. Counting it as attrition
    would understate the exam and report a `q` the arm did not actually lose — the
    over-correction `_arm_full` would have produced here."""
    g = _gap(_fleet(inline_trials=3))
    assert g.display_n["inline_ok"] == 20
    assert "(q=" not in build_fluency_report({"m": _fleet(inline_trials=3)}, [])


def test_an_older_result_file_with_no_inline_arm_still_renders_n_a():
    """Absent is not zero. Rows predating the arm carry no `inline_ok`, and a `0%` there
    would read as "inline comprehension collapsed" rather than "not measured"."""
    rows = [{k: v for k, v in r.items() if not k.startswith("inline")}
            for r in _fleet(inline_trials=5)]
    out = build_fluency_report({"m": rows}, [])
    assert "n/a" in out and "(q=" not in out
