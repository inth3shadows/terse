"""`_form_stats` may only be called from an explicit allowlist (#280).

The false-PASS in the diff-vs-control ship gate survived three fixes, and the reason was
structural rather than local. `_form_stats(rows, form)` computes ONE arm, so every gap site
called it twice and subtracted — and nothing in that shape can enforce that the two arms
answered the SAME questions, because pairing is a property of the pair and `_form_stats`
never sees a pair. Each pass wired pairing into the sites it happened to be looking at; the
next site was still writable by accident. The third attempt's own commit message claimed
"every diff-vs-control gap site", and reverting its pairing at two of the three sites left
the entire suite green.

So this test does not check that the six known sites are wired. It checks that a SEVENTH
cannot be written without saying so: `_form_stats` is reachable only from the functions
named below, and adding a new one is an edit to this file, visible in the diff a reviewer
reads. `test_paired_gap_gate.py` is the other half — it pins that each wired site actually
behaves, by mutation.

Modelled on `test_module_boundaries.py`, including its known-violation sweep: a detector
that has never been shown to fail is not evidence of anything.
"""
from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "terse"

GATED = "_form_stats"

# Every function permitted to call `_form_stats` directly, and why.
#
# The rule these share: a caller is allowed here only if it computes NO form-vs-control
# gap. `_gap` is the one exception, and it is the exception because it IS the gate — it
# pairs the rows first and hands back `ArmGap`.
ALLOWED = {
    # The chokepoint. Gates (`_unmeasured` -> `unpaired`), pairs, then computes every arm
    # over the paired subset. `arm_gap` / `best_arm_gap` are thin wrappers over it.
    "_gap",
    # Display-only, single-arm pooled columns. No gap, no verdict.
    "_per_transform_table",
    # dropeval scores against a fixed 100% ideal rather than a measured control arm, so
    # there is no second arm to pair with. That is itself a defect — see #269, which
    # proposes giving final-accuracy a real control arm. When it does, these two move
    # behind `arm_gap` and come OFF this list.
    "dropeval_gap_rows",
    "build_dropeval_report",
}


def _calls_by_enclosing_function(source: str, filename: str = "<test>") -> list[tuple[str, str]]:
    """[(enclosing function name, called name)] for every call to `GATED` in `source`.

    By AST, not grep: `report.py` names `_form_stats` in a dozen comments and docstrings
    explaining this very rule, so a text-based version of this test would fail on its own
    documentation and get deleted as noisy — the exact way a wrong test earns less than the
    convention it replaced (`test_module_boundaries.py` makes the same argument).

    A call at module scope, or inside a class body but not a method, reports an enclosing
    name of "<module>" and so is never allowed: the allowlist is a list of functions.
    """
    tree = ast.parse(source, filename=filename)
    # Walk with an explicit stack rather than ast.walk, because the enclosing FUNCTION is
    # the thing being asserted about and ast.walk discards that relationship. A nested
    # def reports its own name, not its parent's — a helper defined inside an allowed
    # function is a new function and has to be listed.
    found: list[tuple[str, str]] = []

    def visit(node: ast.AST, enclosing: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call):
                fn = child.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else "")
                if name == GATED:
                    found.append((enclosing, name))
            visit(child, enclosing)

    visit(tree, "<module>")
    return found


def test_form_stats_is_called_only_from_the_allowlist():
    """The boundary itself, over every shipped module."""
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        for enclosing, _ in _calls_by_enclosing_function(
                path.read_text(encoding="utf-8"), str(path)):
            if enclosing not in ALLOWED:
                violations.append(f"{path.relative_to(SRC)}::{enclosing}")
    assert not violations, (
        "these call `_form_stats` outside the allowlist:\n  " + "\n  ".join(violations)
        + "\n\nA form-vs-control gap must go through `arm_gap` / `best_arm_gap`, which pair "
          "the arms first (#280). If this really computes no gap — a single pooled arm for "
          "display — add it to ALLOWED in this file, with a line saying why.")


def test_the_detector_catches_a_known_violation():
    """The sweep that proves the assertion above can fail.

    Without this, a detector that silently matched nothing — a renamed target, a broken
    walk, an AST shape it does not handle — would pass the boundary test forever and read
    as compliance.
    """
    violating = (
        "def build_some_new_report(rows):\n"
        "    facc, fse = _form_stats(rows, 'diff_ok')\n"
        "    cacc, cse = _form_stats(rows, 'terse_ok')\n"
        "    return facc - cacc\n"
    )
    found = _calls_by_enclosing_function(violating)
    assert found == [("build_some_new_report", GATED)] * 2

    allowed = "def _gap(rows, gating, control):\n    return _form_stats(rows, control)\n"
    assert _calls_by_enclosing_function(allowed) == [("_gap", GATED)]


def test_a_nested_helper_does_not_inherit_its_parents_permission():
    """A def inside an allowed function is a different function, and is not allowed.

    Otherwise the allowlist could be defeated by moving the two `_form_stats` calls into a
    closure inside `_gap` — which is precisely the "write it somewhere the rule is not
    looking" move this test exists to prevent.
    """
    source = (
        "def _gap(rows, gating, control):\n"
        "    def sneaky(rs):\n"
        "        return _form_stats(rs, 'diff_ok')\n"
        "    return sneaky(rows)\n"
    )
    assert _calls_by_enclosing_function(source) == [("sneaky", GATED)]


def test_a_module_level_call_is_never_allowed():
    """The allowlist names functions; module scope is not one of them."""
    assert _calls_by_enclosing_function("X = _form_stats(rows, 'diff_ok')\n") == [
        ("<module>", GATED)]
