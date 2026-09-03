"""The `regressions` / `recovered` columns count each arm against its OWN trial count.

#353. Both columns compared `<arm>_ok` to the row's SHARED `trials`, which for a
`score_pack` row is `max(raw_t, terse_t, primer_t, 1)` across forms (`fluency/pack.py`)
— the documented #91 uneven-collection mode, not an attempt count for any one arm. An
arm that answered every reply it was actually given then read as `ok < trials` and was
counted as a regression, on the same line where its accuracy column — which divides by
`<arm>_trials` via `_form_stats` — printed 100%.

The table stated two denominators in one row, and `regressions` is the column a reader
scans to decide whether the compressed form costs comprehension.

MUTATION SWEEP: 10 mutants, 9 killed. The one SURVIVOR is recorded rather than implied
away — rewriting `_arm_trials(r, "terse_ok")` back to `r.get("terse_trials",
r.get("trials", 1))` at `report.py`'s two codec-verdict sums survives every test, and is
an EQUIVALENT MUTANT: a differential over 2000 generated rows (present/absent
`terse_trials`, present/absent `trials`, counts 0/1/2/3/10) found 0 differing results.
Those two call sites were routed through the helper for one spelling, not for behaviour,
so no test can distinguish them and none should pretend to.
"""

import ast
from pathlib import Path
from typing import Any

from terse import report


def _uneven(n: int = 24, **over: Any) -> list[dict[str, Any]]:
    """`n` questions where every arm answered ALL of its own replies, but the arms were
    collected unevenly: raw got 3 replies per question, terse and primer got 2.

    This is the shape `score_pack` emits for a hand-built pack, so `trials` is 3 (the max
    across forms) while `terse_trials` is 2. No arm lost anything.
    """
    row: dict[str, Any] = {
        "tool": "t", "sha": "s", "qtype": "lookup", "transform": "table",
        "trials": 3,
        "raw_ok": 3, "raw_trials": 3, "raw_attempts": 3,
        "terse_ok": 2, "terse_trials": 2, "terse_attempts": 2,
        "primer_ok": 2, "primer_trials": 2, "primer_attempts": 2,
        "fails": 0, "attempts": 8,
    }
    row.update(over)
    return [dict(row, qid=f"q{i}") for i in range(n)]


def _row_cells(md: str, model: str = "m") -> list[str]:
    line = next(ln for ln in md.splitlines() if ln.startswith(f"| `{model}` |"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_an_uneven_pack_with_no_losses_reports_zero_regressions():
    """The bug as filed: 24 questions, every arm complete, 100% across the board — and
    24 regressions out of 24. The row argued against itself."""
    md = report.build_fluency_report({"m": _uneven()}, [])
    cells = _row_cells(md)
    assert cells[2].startswith("100%"), cells      # raw
    assert cells[3].startswith("100%"), cells      # terse
    assert cells[-2] == "0", f"regressions, got {cells[-2]}: {cells}"
    assert cells[-1] == "0", f"recovered, got {cells[-1]}: {cells}"


def test_a_real_regression_in_an_uneven_pack_is_still_counted():
    """The fix must not silence the column it corrects. Same uneven shape, but terse
    genuinely dropped one of its two replies while raw kept all three."""
    md = report.build_fluency_report({"m": _uneven(terse_ok=1)}, [])
    cells = _row_cells(md)
    assert cells[-2] == "24", f"regressions, got {cells[-2]}: {cells}"


def test_a_real_primer_recovery_in_an_uneven_pack_is_still_counted():
    """`recovered` = terse lost the question, primer answered all of ITS trials."""
    md = report.build_fluency_report({"m": _uneven(terse_ok=1)}, [])
    assert _row_cells(md)[-1] == "24", _row_cells(md)


def test_the_even_live_path_is_unchanged():
    """Every live harness collects the same trial count for every arm, so `trials` and
    `<arm>_trials` agree and the counts must come out exactly as before."""
    even = _uneven(trials=2, raw_ok=2, raw_trials=2, raw_attempts=2)
    assert _row_cells(report.build_fluency_report({"m": even}, []))[-2] == "0"
    lost = _uneven(trials=2, raw_ok=2, raw_trials=2, raw_attempts=2, terse_ok=1)
    assert _row_cells(report.build_fluency_report({"m": lost}, []))[-2] == "24"


def test_rows_with_no_per_arm_counters_fall_back_to_the_shared_trials():
    """A result file predating the per-arm counters carries only `trials`. It must read
    exactly as it did before — absent is not zero."""
    legacy = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "lookup",
               "transform": "table", "trials": 2,
               "raw_ok": 2, "terse_ok": 1, "primer_ok": 2,
               "fails": 0, "attempts": 6} for i in range(24)]
    cells = _row_cells(report.build_fluency_report({"m": legacy}, []))
    assert cells[-2] == "24", cells
    assert cells[-1] == "24", cells


def test_the_diff_report_regressions_column_has_the_same_denominator():
    """`_build_diff_style_report` carried an independent copy of the same arithmetic
    (`report.py`, the `terse_ok`/`diff_ok` pair). The issue named only the fluency site;
    two hardcoded copies is the shape that produced #299."""
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "lookup",
             "transform": "table", "trials": 3,
             "terse_ok": 3, "terse_trials": 3, "terse_attempts": 3,
             "diff_ok": 2, "diff_trials": 2, "diff_attempts": 2,
             "fails": 0, "attempts": 5} for i in range(24)]
    md = report.build_diff_report({"m": rows})
    cells = _row_cells(md)
    assert cells[-1] == "0", f"regressions, got {cells[-1]}: {cells}"


def _suffix_swap_sites(root: ast.Module) -> set[str]:
    """Functions that build an `<arm>_trials` key from a form name themselves.

    Two spellings, because a raw substring count over the source both over- and
    under-fires: it accuses a docstring that merely QUOTES the derivation (this file's
    house style quotes code constantly), and it misses `f.replace("_ok", "_trials")`,
    which is the same duplicate written differently. AST is the idiom this repo already
    uses for structural guards — `test_gap_gate_boundary.py` and
    `test_attrition_visibility.py` both walk it — and #361 exists precisely because a
    hand-rolled textual guard was evadable by idioms the package already uses.
    """
    found: set[str] = set()
    for fn in ast.walk(root):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for n in ast.walk(fn):
            # `<expr> + "_trials"`
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
                    and isinstance(n.right, ast.Constant) and n.right.value == "_trials"):
                found.add(fn.name)
            # `<expr>.replace(..., "_trials")`
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "replace"
                    and any(isinstance(a, ast.Constant) and a.value == "_trials"
                            for a in n.args)):
                found.add(fn.name)
    return found


def test_only_one_place_derives_the_arm_to_trials_key():
    """`_trials_keys` claimed to be "the one place that mapping is written" while
    `_form_stats` and `arm_measured` both spelled the suffix swap inline. A second copy
    is how the two columns drifted apart from the accuracies beside them."""
    tree = ast.parse(Path(str(report.__file__)).read_text())
    assert _suffix_swap_sites(tree) == {"_trials_key"}, (
        "the arm->trials key derivation is spelled outside `_trials_key`; route it "
        "through `_trials_key` (or `_arm_trials`) instead")


def test_the_guard_ignores_a_docstring_that_merely_quotes_the_derivation():
    """The predecessor of the test above was a substring count over the source, so
    documenting the invariant tripped the test that protects it."""
    quoting = ast.parse('def f():\n    """Built as `form[:-3] + "_trials"` elsewhere."""\n')
    assert _suffix_swap_sites(quoting) == set()


def test_the_guard_catches_the_same_duplicate_written_differently():
    """`f.replace("_ok", "_trials")` is the identical duplicate; a substring count for
    the slice spelling passed straight over it."""
    evasion = ast.parse('def g(f):\n    return f.replace("_ok", "_trials")\n')
    assert _suffix_swap_sites(evasion) == {"g"}


def test_an_arm_with_zero_collected_trials_is_not_complete():
    """`0 == 0` reads as "answered every trial it was given" for an arm nobody asked
    anything. `regr` counts a question whenever the CONTROL looks complete and terse does
    not, so a responses file with no `raw` replies at all — `fluency/scoring.py` maps a
    missing form to `(0, 0, 0)` — would report EVERY question as a regression against a
    control that never ran. The old shared-`trials` comparison was accidentally immune:
    `score_pack` floors that count at `max(..., 1)`, so `0 == 3` was already False.

    `recovered` has its own guard (`arm_measured` renders `n/a`); the control arm feeding
    `regressions` has none, which is why this lives in the predicate."""
    assert not report._arm_full({"raw_ok": 0, "raw_trials": 0, "trials": 3}, "raw_ok")
    no_control = _uneven(raw_ok=0, raw_trials=0, raw_attempts=0)
    cells = _row_cells(report.build_fluency_report({"m": no_control}, []))
    assert cells[-2] == "0", f"regressions against an uncollected control: {cells}"


def test_a_non_arm_form_reads_the_shared_trials_not_its_own_success_count():
    """`_trials_key` returns a form that is not `<arm>_ok` UNCHANGED — that is what
    `arm_measured` needs, since it passes `<arm>_trials` keys straight through. But
    `_form_stats` is the caller that DIVIDES by what comes back, and its pre-#353 spelling
    fell back to the shared `trials` on that branch. Looking the form up instead would use
    the row's success count as its own denominator and publish a confident 100%.

    Unreachable today — every live caller passes an `_ok` form — and pinned because the
    failure is silent, which is the class this module spends its comments refusing."""
    rows = [{"trials": 4, "score": 1}, {"trials": 4, "score": 3}]
    assert report._arm_trials(rows[0], "score") == 4
    acc, _ = report._form_stats(rows, "score")
    assert acc == 0.5, f"expected 4/8, got {acc} (denominator is the success count)"
