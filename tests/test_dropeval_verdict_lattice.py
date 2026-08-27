"""The dropeval verdict as a lattice with a closed reason set (#342).

Four review rounds on one ~200-line change to this path produced 7, 6, 5 then 9 findings
and did not converge; every round found a defect inside the previous round's fix. The two
dominant classes were "the fix was applied to one site and not the next" (10 findings) and
"branch precedence in a growing if/elif chain" (4 findings). Neither is a shortage of
care — both are the cost of an OPEN reason set consumed by hand at N sites, and of an
ORDERED chain over an unenumerated state space.

These tests pin the two structures that replaced them: `ExclusionReason` is closed and
every total consumer of it is checked against the whole set, and the directive is `max()`
over a lattice, so precedence is a property of the order rather than of the source line
number. The metamorphic property that guards the same thing end to end lives in
`tests/test_dropeval_monotonicity.py`.
"""

from __future__ import annotations

import itertools
import typing

from terse.report import (
    DROPEVAL_METRICS,
    REASON_HEADING,
    REASON_LABEL,
    Directive,
    ExclusionReason,
    _exclusion_remedy,
    _reason_directive,
    build_dropeval_report,
    dropeval_directive_line,
    dropeval_exclusion_bullets,
    dropeval_next_step_line,
    dropeval_verdict,
)
from terse.terminal_report import build_terminal_dropeval_report

_REASONS: tuple[ExclusionReason, ...] = typing.get_args(ExclusionReason)


# --------------------------------------------------------------------------- #
# Fixtures — shared with the monotonicity property's shape.
# --------------------------------------------------------------------------- #

_N = 24


def _rows(n, *, kind, answer, retrieve, control=None, errors=0, control_errors=0):
    """Rows in the shape `dropeval.run_drop_fluency` actually emits — see its `row = {...}`
    literal and the long note above it.

    The asymmetry is the whole point and an earlier version of this fixture erased it. The
    TREATMENT arm carries NO per-form denominator (`retrieve_trials`/`answer_trials`/
    `handle_trials`), so `_form_stats` falls back to the shared `trials` and an errored
    trial counts as a miss — deliberately, because scoring a treatment failure as a miss
    makes the drop look WORSE, which is the conservative direction. Only `control_trials`
    excludes its own failures, because a control failure scored as a miss would make the
    drop look BETTER. Adding the three treatment denominators was measured turning a 33%
    FAIL into a 100% "safe to enable" at an 11% error rate.

    It also mattered here mechanically: `_unmeasured` finds arms by `k.endswith("_trials")`,
    so the extra keys kept every arm non-zero and the `"unmeasured"` exclusion could not
    fire on ANY input this module generates. The property was sweeping a state space with
    one of its states unreachable."""
    out = []
    for i in range(n):
        r = {"kind": kind, "trials": 1,
             "retrieve_ok": retrieve, "answer_ok": answer, "handle_ok": 1,
             "errors": errors + control_errors,
             "treatment_errors": errors, "control_errors": control_errors,
             "attempts": 1 if control is None else 2, "qid": f"{kind}-q{i}"}
        if control is not None:
            r |= {"control_ok": control, "control_trials": 1 - control_errors}
        out.append(r)
    return out

def _model(*, answer, retrieve, control, n=_N):
    return (_rows(n, kind="recall", answer=answer, retrieve=retrieve, control=control)
            + _rows(n, kind="precision", answer=answer, retrieve=retrieve, control=control))


_SHAPES = list(itertools.product((0, 1), (0, 1), (None, 0, 1)))


def _fleets(models=2):
    for shapes in itertools.product(_SHAPES, repeat=models):
        yield {chr(ord("A") + i): _model(answer=s[0], retrieve=s[1], control=s[2])
               for i, s in enumerate(shapes)}


# --------------------------------------------------------------------------- #
# 1. The reason set is closed, and every total consumer covers all of it.
# --------------------------------------------------------------------------- #


def test_every_exclusion_reason_has_a_label_a_heading_a_directive_and_a_remedy():
    """The 10-finding class in one assertion.

    `excluded: str | None` said "some string", so each of the four maps below was a place a
    programmer had to remember the list — and four separate consumers, across four review
    rounds, did not. mypy now rejects a `_reason_directive` or `_exclusion_remedy` that
    misses a member (both end in `assert_never`); `REASON_LABEL` and `REASON_HEADING` are
    plain dicts and cannot be checked that way, so they are checked here.

    Both directions. A reason with no label renders as its raw internal string; a label
    with no reason is dead prose that outlived the code path it described, which is how a
    branch testing `why == "x"` sat unreachable across two releases (#284)."""
    assert set(REASON_LABEL) == set(_REASONS), "REASON_LABEL is not total over ExclusionReason"
    assert set(REASON_HEADING) == set(_REASONS), "REASON_HEADING is not total over ExclusionReason"
    for reason in _REASONS:
        assert isinstance(_reason_directive(reason), Directive)
        assert _exclusion_remedy(reason).strip(), f"{reason} has an empty remedy"


def test_no_exclusion_reason_authorizes_a_ship():
    """The lattice's bottom is unreachable from an exclusion, whatever the reason says.

    This is the invariant that makes `max()` safe to compute: a withheld model can only
    ever move the verdict UP. If any reason mapped to `SHIP`, adding it to a fleet would
    leave the verdict where it was, and "excluded" would once again mean "invisible"."""
    for reason in _REASONS:
        assert _reason_directive(reason) > Directive.SHIP, reason


def test_an_empty_fleet_is_not_concluded_rather_than_shipped():
    """The one `default=` left in `dropeval_verdict`, and the only place its value is a
    choice rather than a consequence.

    Both renderers stop before reading it — `build_dropeval_report` early-returns on an
    empty result set and the terminal returns "(no data)" — so mutating it to `SHIP` left
    the whole suite green. That is precisely why it needs a test: a `SHIP` sitting in an
    unread field is a `SHIP` the next consumer of `DropevalVerdict` reads."""
    v = dropeval_verdict({})
    assert v.directive is Directive.NOT_CONCLUDED
    assert all(mv.directive is Directive.NOT_CONCLUDED for mv in v.metrics.values())
    assert dropeval_verdict({"m": []}).directive is Directive.NOT_CONCLUDED


def test_the_lattice_order_is_strictness_not_declaration_order():
    """`max()` over the directive is only correct if the order means what the renderers
    assume: `BLOCK` must outrank every non-verdict, and `SHIP` must be the bottom.

    Pinned to the VALUES, in both directions, for the reason #337 gives: every other test
    here reads the enum and asserts a relative fact, so reordering the members would leave
    them all green while inverting the verdict."""
    assert Directive.SHIP < Directive.INSUFFICIENT < Directive.NOT_CONCLUDED < Directive.BLOCK
    assert max(Directive) is Directive.BLOCK
    assert min(Directive) is Directive.SHIP


# --------------------------------------------------------------------------- #
# 2. Which metrics can be withheld at all.
# --------------------------------------------------------------------------- #


def test_a_mechanism_metric_is_withheld_only_when_it_has_no_rows():
    """Recall and precision gate against a FIXED 100% ideal — no second arm, so no pairing,
    so none of `_gap`'s exclusions can fire on them. The ONE way they leave a gate is
    having no rows of that kind at all, which is `"empty"`.

    This test used to assert they were never withheld, full stop, and
    `dropeval_directive_line` leaned on that to license "so the mechanism works". Both were
    wrong in the same direction: `_form_stats([], f)` is `(0.0, 0.0)`, which against the
    fixed ideal published a `-100%` **FAIL** and `keep drop-to-retrieve off` from a metric
    with zero rows. Withholding it is right; the prose now reads the lattice instead of
    assuming this."""
    for fleet in _fleets(2):  # every shape here carries both kinds
        v = dropeval_verdict(fleet)
        assert not v.metrics["recall"].excluded
        assert not v.metrics["precision"].excluded
    precision_only = [r for r in _model(answer=1, retrieve=1, control=1)
                      if r["kind"] == "precision"]
    v = dropeval_verdict({"m": precision_only})
    assert v.metrics["recall"].excluded == {"m": "empty"}
    assert v.metrics["recall"].worst is None, "a metric with no rows must publish no gap"
    assert v.directive is Directive.NOT_CONCLUDED


def test_a_missing_mechanism_metric_does_not_claim_the_mechanism_works():
    """The prose half of the test above. With recall withheld, the report must not say
    "recall and no-overfetch clear tolerance for the worst model, so the mechanism works" —
    and must not publish the `-100%` FAIL that the fabricated 0% used to produce."""
    precision_only = [r for r in _model(answer=1, retrieve=1, control=1)
                      if r["kind"] == "precision"]
    md = build_dropeval_report({"m": precision_only})
    verdict = md.split("## Verdict", 1)[1]
    assert "so the mechanism works" not in verdict, verdict
    assert "MECHANISM itself was not gated" in verdict, verdict
    assert "keep drop-to-retrieve off" not in verdict, verdict
    assert "**FAIL**" not in verdict, verdict
    assert "retrieve-recall: not gated for `m`" in verdict, verdict


def test_inconclusive_for_enabling_is_only_reached_with_the_mechanism_passing():
    """The end-to-end form of the claim above: whenever the report prints "so the mechanism
    works", recall and no-overfetch really did pass."""
    for fleet in _fleets(2):
        v = dropeval_verdict(fleet)
        if v.directive in (Directive.INSUFFICIENT, Directive.NOT_CONCLUDED) and not v.inconclusive:
            for metric in ("recall", "precision"):
                worst = v.metrics[metric].worst
                assert worst is not None and worst.passed, (
                    f"{metric} did not pass, but the verdict says the mechanism works")


# --------------------------------------------------------------------------- #
# 3. One decision, two renderers.
# --------------------------------------------------------------------------- #


def test_the_chart_and_the_markdown_reach_the_same_directive():
    """`dropeval_gap_rows`' docstring has promised since #269 that the two "can never
    disagree". They disagreed three times across #335's four review rounds — on badge
    scope, on exclusion notes, and on a thin-sample caveat printed over a demonstrated
    FAIL — because each renderer decided for itself off shared numbers.

    Both now print `dropeval_directive_line(v)` off one `DropevalVerdict`; the terminal
    strips `**` and backticks mechanically rather than re-typing the sentence. That is the
    relation asserted here, over the whole 2-model cross product."""
    for fleet in _fleets(2):
        md = build_dropeval_report(fleet)
        chart = build_terminal_dropeval_report(fleet, color=False)
        verdict = md.split("## Verdict", 1)[1]
        # The directive is the LAST bullet, always — the worst-case lines and the
        # `not gated` bullets come first and the section ends there. Matching on content
        # instead is what the first draft of this test did, and it silently matched zero
        # lines for every fleet whose directive sentence happened to contain the words
        # "not gated": a test that cannot fail, of exactly the kind #342 counts five of.
        bullets = [line for line in verdict.splitlines() if line.startswith("- ")]
        assert bullets, f"no verdict bullet at all:\n{verdict}"
        plain = bullets[-1].removeprefix("- ").replace("**", "").replace("`", "")
        assert plain in chart, f"chart does not carry the markdown's directive:\n{chart}"


def test_the_unmeasured_claim_is_scoped_to_the_models_it_is_true_of():
    """"the OUTCOME impact of dropping is unmeasured" must never be said of a fleet whose
    accuracy WAS measured for some model.

    On the pre-#342 code that sentence sat inside the `accuracy_worst is None` branch, so
    it could only fire when no model was scored and the unscoped claim was true by
    construction. The lattice reaches the same directive whenever ANY model is withheld —
    so the unscoped sentence started printing two lines under a measured `**PASS**`, a
    false statement about the report's own contents. It is now scoped to the withheld
    models, and says explicitly that the rest of the fleet was gated."""
    scored = _model(answer=1, retrieve=1, control=1)
    fleet = {"A": scored, "B": _model(answer=1, retrieve=1, control=None)}
    v = dropeval_verdict(fleet)
    line = dropeval_directive_line(v)
    assert "unmeasured: final-accuracy for `B`" in line, line
    assert "It WAS gated for the rest of the fleet — worst-case `A` at +0%" in line, line
    # And with nothing scored, no such claim is made.
    only_withheld = dropeval_directive_line(dropeval_verdict({"B": fleet["B"]}))
    assert "unmeasured: final-accuracy for `B`" in only_withheld
    assert "WAS gated for the rest" not in only_withheld


def test_insufficient_and_not_concluded_do_not_render_identically():
    """`INSUFFICIENT` is a lattice member; if it renders exactly like `NOT_CONCLUDED` it is
    a distinction the code draws and the reader never sees — which is what it was until the
    headline split. The two mean different next actions: generate more questions, versus go
    find out why the comparison never happened at all."""
    thin = dropeval_verdict({"m": _model(answer=1, retrieve=1, control=1, n=1)})
    absent = dropeval_verdict({"m": _model(answer=1, retrieve=1, control=None)})
    assert thin.directive is Directive.INSUFFICIENT
    assert absent.directive is Directive.NOT_CONCLUDED
    assert "**INSUFFICIENT for enabling**" in dropeval_directive_line(thin)
    assert "**INCONCLUSIVE for enabling**" in dropeval_directive_line(absent)


def test_the_chart_names_every_model_the_shared_sentence_says_it_names():
    """`dropeval_directive_line` ends "Each withheld model is named above with the reason
    it was withheld". Both renderers print that sentence, so both must carry the referent.

    The markdown does, via `dropeval_exclusion_bullets`. The chart did not: its exclusion
    note sat AFTER a `continue` that fired when a metric had no bars to draw, so a metric
    from which every model was withheld lost the note and kept the sentence. Sharing one
    sentence removes disagreement about the CONCLUSION; it does nothing about whether each
    renderer carries what the sentence points at, and this is that assertion."""
    for fleet in _fleets(2):
        v = dropeval_verdict(fleet)
        if v.inconclusive:
            continue  # the chart refuses outright and draws no bars
        chart = build_terminal_dropeval_report(fleet, color=False)
        for metric, _, _ in DROPEVAL_METRICS:
            for model, reason in v.metrics[metric].excluded.items():
                assert model in chart, f"`{model}` withheld ({reason}) but absent:\n{chart}"
                assert REASON_LABEL[reason] in chart, (reason, chart)


def test_each_exclusion_reason_gets_its_own_bullet_naming_its_own_models():
    """A fleet whose models are withheld for DIFFERENT reasons.

    The verdict used to pick one remedy sentence by set-membership over the reasons
    present, and once both halves were handled it joined them — producing, in a single
    bullet, both "re-run with the no-drop control arm" and "the control arm is already on".
    Neither clause named a model, so the reader could not tell which applied to which.

    One bullet per reason, each naming its own models, makes that unconstructible."""
    fleet = {"nocontrol": _model(answer=1, retrieve=1, control=None),
             "thin": _model(answer=1, retrieve=1, control=1, n=1)}
    v = dropeval_verdict(fleet)
    assert v.metrics["accuracy"].excluded == {"nocontrol": "no control arm",
                                              "thin": "underpowered"}
    bullets = dropeval_exclusion_bullets(v)
    assert len(bullets) == 2, bullets
    by_model = {b.split("`")[1]: b for b in bullets}
    assert "Re-run without `--no-control`" in by_model["nocontrol"]
    assert "The control arm ran" in by_model["thin"]
    # The contradiction the joined version produced: no single bullet may both demand the
    # control arm be switched on and state that it already is.
    for bullet in bullets:
        assert not ("--no-control" in bullet and "The control arm ran" in bullet)


def test_every_withheld_model_is_named_in_the_verdict_prose():
    """The `not gated` bullets are the disclosure, and they must be in `## Verdict`.

    Mutation-found, and it is #342's colliding-needle class exactly: deleting
    `out += dropeval_exclusion_bullets(v)` left all 1727 tests green, because the one test
    that looked for "not gated" searched the WHOLE report — and the per-model table has a
    "not gated" cell of its own. The needle was already in the haystack, so the assertion
    could not fail. Scoped to the verdict section, and matched on the bullet's own shape."""
    for fleet in _fleets(2):
        v = dropeval_verdict(fleet)
        verdict = build_dropeval_report(fleet).split("## Verdict", 1)[1]
        for metric, label, _ in DROPEVAL_METRICS:
            for model, reason in v.metrics[metric].excluded.items():
                bullet = next((line for line in verdict.splitlines()
                               if line.startswith(f"- **{label}: not gated for")
                               and f"`{model}`" in line), None)
                assert bullet is not None, (
                    f"`{model}` was withheld ({reason}) and the verdict does not say so:"
                    f"\n{verdict}")
                assert REASON_LABEL[reason] in bullet, (reason, bullet)
                assert _exclusion_remedy(reason) in bullet, (reason, bullet)


def test_the_policy_instruction_never_authorizes_what_the_verdict_declined():
    """`terse tune --drop-eval` prints a "now edit your policy" line under the report.

    It used to say "If the worst-case model PASSES, enable the verified fields" — a rule
    the READER applies to the three worst-case lines, which report the worst SCORED model.
    A fleet with one model scored and one withheld therefore printed three `**PASS**`
    headlines under a verdict that authorized nothing. Swept over the cross product: the
    permissive sentence appears exactly when the directive is SHIP, and never otherwise."""
    for fleet in _fleets(2):
        v = dropeval_verdict(fleet)
        line = dropeval_next_step_line(v)
        authorizes = "The verdict authorizes it" in line
        assert authorizes is (v.directive is Directive.SHIP), (v.directive, line)
        if not authorizes:
            assert "does NOT authorize" in line, line


def test_the_table_reads_the_same_gates_the_verdict_does():
    """The per-model table used to hold a third copy of the gate math, under a comment
    promising it "cannot disagree" with the verdict about which control it used. Now it
    reads `v.gates`, so the accuracy cell is "not gated" exactly when the verdict withheld
    the model — checked over the cross product rather than on one fixture."""
    for fleet in _fleets(2):
        v = dropeval_verdict(fleet)
        md = build_dropeval_report(fleet)
        table = md.split("## Verdict", 1)[0]
        for model in fleet:
            row = next(line for line in table.splitlines() if line.startswith(f"| `{model}` |"))
            reason = v.metrics["accuracy"].excluded.get(model)
            assert ("not gated" in row) is (reason is not None), row
            if reason is not None:
                # The control cell names the ACTUAL reason. "not run" is reserved for the
                # one reason where the control really never ran — printing it for a control
                # that ran and failed is #300's finding 2, and printing the generic label
                # for a control that never ran drops the distinction the other way.
                # Mutation-found: collapsing the two into `REASON_LABEL[reason]` left the
                # whole suite green, because only the NEGATIVE ("not run" must be absent
                # for a broken control) had a test.
                expected = "not run" if reason == "no control arm" else REASON_LABEL[reason]
                assert f"| {expected} |" in row, (reason, row)


# --------------------------------------------------------------------------- #
# 4. The metric labels
# --------------------------------------------------------------------------- #


def test_a_dead_backend_gets_one_sentence_in_both_renderers():
    """The refusal an operator actually reaches on a broken backend was, until #342, the
    one verdict sentence each renderer wrote for itself — the markdown named the failure
    counts and the chart printed a shorter line of its own. Both now come from
    `dropeval_directive_line`, so the sentence is shared everywhere, not just on the paths
    that happened to have a test."""
    dead = [dict(r, errors=r["attempts"], treatment_errors=r["trials"],
                 control_errors=r["attempts"] - r["trials"])
            for r in _model(answer=1, retrieve=1, control=None)]
    v = dropeval_verdict({"m": dead})
    assert v.inconclusive, "fixture must actually trip the INCONCLUSIVE gate"
    line = dropeval_directive_line(v)
    assert "failed 48/48 model calls" in line, line
    assert line.replace("**", "").replace("`", "") in build_terminal_dropeval_report(
        {"m": dead}, color=False)
    assert "- " + line in build_dropeval_report({"m": dead})


def test_only_final_accuracy_claims_a_measured_control():
    """"vs ideal (100%)" and "vs no-drop control" are different claims, and #269 exists
    because a reader could not tell which one the verdict was making. The pairing lived in
    two places — a tuple here and a `key == "accuracy"` test in the terminal renderer —
    until #342; this pins the one that is left."""
    assert {m: c for m, _, c in DROPEVAL_METRICS} == {
        "recall": "ideal (100%)",
        "precision": "ideal (100%)",
        "accuracy": "no-drop control",
    }
