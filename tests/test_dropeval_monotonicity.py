"""The metamorphic invariant the dropeval verdict must obey (#342).

**Withholding evidence must never authorize a ship.**

Four review rounds on this verdict path produced 7, 6, 5 then 9 findings, and the worst
finding in *every* round was an instance of this one rule: a model that was going to be
gated got EXCLUDED instead — for want of a control arm, for a control that failed, for too
few paired questions — and because an excluded model leaves `_worst_case_gap` entirely, the
verdict computed over what remained came back cleaner than the verdict over everything.
#344's critical finding is the extreme case: stripping one model's control arm — strictly
LESS evidence, no other change — turns `keep drop-to-retrieve off` into `safe to enable`.

This is a metamorphic property, not an oracle. It never asserts which verdict is correct
for a given fleet; that is the hard question, and it is why 29 hand-written examples over a
~72-state cross product still missed the bug. It asserts only a RELATION between two
outputs, which is the standard move when the oracle is hard and the relation is easy: run
the same code twice, once on strictly less information, and compare.

Directions this deliberately does NOT assert:

- **`BLOCK` is not preserved under every perturbation.** Strip the control arm off the very
  model whose accuracy was failing and the honest answer really is "not concluded" — the
  measurement is gone. `NOT_CONCLUDED` does not authorize anything, so that transition is
  safe; only a transition *into* `SHIP` is not. Asserting `rank(after) >= rank(before)`
  would be a stronger property that is FALSE, and a false property gets weakened until it
  passes rather than fixing the code.
- **Deleting a model outright is not a perturbation here.** That is a different fleet, and a
  fleet without its worst model may legitimately ship. Every perturbation below leaves the
  model in `results`; it only takes away something that was measured about it.
"""

from __future__ import annotations

import itertools

from terse.report import Directive, build_dropeval_report, dropeval_verdict

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# Above `_MIN_PAIRED_QUESTIONS` (20), so an unperturbed model is scored rather than
# withheld as underpowered — the `shrink` perturbation below is what crosses that floor,
# and it can only be a perturbation if the baseline is on the other side of it.
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
    """A verdict needs recall AND precision rows — a missing kind scores that gate 0%."""
    return (_rows(n, kind="recall", answer=answer, retrieve=retrieve, control=control)
            + _rows(n, kind="precision", answer=answer, retrieve=retrieve, control=control))


# (answer, retrieve, control): every combination of "the model gets the answer right",
# "the model operates the retrieve protocol", and "a no-drop control ran / ran and died /
# never ran". 12 shapes per model.
_SHAPES = list(itertools.product((0, 1), (0, 1), (None, 0, 1)))


# --------------------------------------------------------------------------- #
# Perturbations: each returns rows carrying strictly LESS information than it got,
# about the same model, in the same fleet.
# --------------------------------------------------------------------------- #


def _drop_control(rows):
    """The control arm was never run — `_accuracy_gate` -> "no control arm"."""
    return [{k: v for k, v in r.items() if k not in ("control_ok", "control_trials")}
            for r in rows]


def _break_control(rows):
    """The control ran and scored 0% — `_gap` -> "broken control"."""
    return [dict(r, control_ok=0) if "control_ok" in r else dict(r) for r in rows]


def _partial_control(rows):
    """One row loses its control — `_accuracy_gate` -> "partial control coverage"."""
    out = [dict(r) for r in rows]
    for r in out:
        if "control_ok" in r:
            del r["control_ok"], r["control_trials"]
            break
    return out


def _shrink(rows):
    """One question per kind — below `_MIN_PAIRED_QUESTIONS`, so a non-failing arm is
    withheld as "underpowered".

    It CANNOT produce a violation, and that is a designed property rather than weak
    coverage: `_MIN_PAIRED_QUESTIONS`' floor is asymmetric, withholding only arms that are
    not behind their control, so what it removes from `_worst_case_gap` is always a
    would-be PASS. Measured over the 2-model cross product, its only effect on the
    directive is `SHIP -> INSUFFICIENT`, 2 cases — never toward SHIP.

    Left in the perturbation set anyway, with
    `test_shrinking_the_sample_only_ever_moves_the_verdict_away_from_SHIP` asserting that
    directly. A perturbation that cannot fail the property it is listed under is otherwise
    just a claim in a docstring."""
    kept, seen = [], set()
    for r in rows:
        if r["kind"] not in seen:
            seen.add(r["kind"])
            kept.append(dict(r))
    return kept


def _kill_calls(rows):
    """Every call to this model failed transport — `inconclusive_models` fires, and the
    run measures the harness rather than the model.

    Found by mutation: without it, `directive = Directive.NOT_CONCLUDED if inconclusive`
    could be changed to `Directive.SHIP` and the whole suite stayed green, because both
    renderers return early on `v.inconclusive` and never read `v.directive` on that path.
    The renderers were right; the VALUE was wrong, and a value nothing reads today is read
    by the next consumer. `attempts` is left alone so the failure RATE is 100%."""
    return [dict(r, errors=r["attempts"], treatment_errors=r["trials"],
                 control_errors=r["attempts"] - r["trials"],
                 **({"control_trials": 0} if "control_ok" in r else {}))
            for r in rows]


# Perturbations that WITHHOLD one metric of one model: the model stays in the fleet, its
# other metrics keep gating, only the accuracy gate loses it. These are the ones a
# demonstrated FAIL must survive.
def _empty_rows(rows):
    """The model answered nothing at all — it stays in `results` with an empty row list.

    NOT the same as deleting the model, which this module excludes from its perturbations
    on purpose (see the docstring). The model is still in the fleet, still named, still
    something the operator expects a verdict about; only its measurements are gone. That
    made it the sharpest form of the property: `dropeval_verdict` used to `continue` past
    it, so it entered neither `gates` nor any `excluded` dict and `max()` never saw it — 22
    of the 144 non-shipping two-model fleets started shipping under this perturbation."""
    return []


_WITHHOLDING = {
    "empty_rows": _empty_rows,
    "drop_control": _drop_control,
    "break_control": _break_control,
    "partial_control": _partial_control,
    "shrink": _shrink,
}

# Everything above, plus the fleet-level refusal. `kill_calls` is not a withholding: past
# the INCONCLUSIVE threshold the run measures the harness, so the WHOLE verdict is refused
# rather than one metric withheld — which is why it belongs in the "never authorizes a
# ship" property and not in the "a FAIL is never displaced" one.
_PERTURBATIONS = {**_WITHHOLDING, "kill_calls": _kill_calls}


# --------------------------------------------------------------------------- #
# Reading the directive back out of the report
# --------------------------------------------------------------------------- #

SHIP = "SHIP"
NOT_SHIP = "NOT_SHIP"


def directive_of(results, accept_degraded=False):
    """SHIP iff the report authorizes turning drop-to-retrieve on.

    Read from the rendered markdown rather than from an internal value on purpose: the
    markdown is what an operator acts on, so it is the surface the invariant has to hold
    at. `## Verdict` scoping matters — the preamble names the feature too, and matching
    the whole document would find the word "enable" in the guidance for an empty run."""
    md = build_dropeval_report(results, accept_degraded=accept_degraded)
    verdict = md.split("## Verdict", 1)[1] if "## Verdict" in md else ""
    return SHIP if "safe to enable drop-to-retrieve" in verdict else NOT_SHIP


# --------------------------------------------------------------------------- #
# The property
# --------------------------------------------------------------------------- #


def _violations():
    """Every (fleet, model, perturbation) where taking evidence away authorizes a ship."""
    bad = []
    for a_shape, b_shape in itertools.product(_SHAPES, repeat=2):
        fleet = {"A": _model(answer=a_shape[0], retrieve=a_shape[1], control=a_shape[2]),
                 "B": _model(answer=b_shape[0], retrieve=b_shape[1], control=b_shape[2])}
        before = directive_of(fleet)
        if before == SHIP:
            continue  # already shipping; nothing to authorize
        for victim in ("A", "B"):
            for name, perturb in _PERTURBATIONS.items():
                after_fleet = dict(fleet, **{victim: perturb(fleet[victim])})
                if after_fleet[victim] == fleet[victim]:
                    continue  # the perturbation was a no-op on this shape
                if directive_of(after_fleet) == SHIP:
                    bad.append((a_shape, b_shape, victim, name))
    return bad


def test_withholding_evidence_never_authorizes_a_ship():
    """The rule violated by the worst finding of all four review rounds on #335.

    A fleet that does not ship must not start shipping because one of its models lost a
    control arm, lost the control's score, lost control coverage on one question, or lost
    every question but one. All four are ways a model leaves the accuracy gate, and a model
    that leaves the gate stops being able to fail it."""
    bad = _violations()
    assert not bad, (
        f"{len(bad)} input pairs where removing evidence authorized a ship; first 5:\n"
        + "\n".join(
            f"  A={a} B={b}: perturbing {v} with {p} turned not-ship into SHIP"
            for a, b, v, p in bad[:5]))


# --------------------------------------------------------------------------- #
# The same property, straight over `decide()` — three models instead of two.
# --------------------------------------------------------------------------- #


def test_withholding_evidence_never_lowers_the_directive_to_SHIP_over_three_models():
    """`dropeval_verdict` is pure, so the cross product replays in a second rather than a
    minute — which is the argument for splitting it out of the renderer at all (#342).
    That budget buys a third model, and with it every fleet where two models are withheld
    for DIFFERENT reasons while a third is scored: the shape the old set-equality tests
    over the exclusion reasons could not see.

    Same relation as the markdown property above, and deliberately not stronger — see this
    module's docstring for why `rank(after) >= rank(before)` is FALSE and would have to be
    weakened until it passed."""
    bad = []
    for shapes in itertools.product(_SHAPES, repeat=3):
        fleet = {name: _model(answer=s[0], retrieve=s[1], control=s[2])
                 for name, s in zip("ABC", shapes, strict=True)}
        if dropeval_verdict(fleet).directive is Directive.SHIP:
            continue
        for victim in "ABC":
            for name, perturb in _PERTURBATIONS.items():
                after = dict(fleet, **{victim: perturb(fleet[victim])})
                if after[victim] == fleet[victim]:
                    continue
                if dropeval_verdict(after).directive is Directive.SHIP:
                    bad.append((shapes, victim, name))
    assert not bad, f"{len(bad)} violations; first 5: {bad[:5]}"


def test_a_demonstrated_regression_outranks_an_exclusion_on_another_model():
    """`BLOCK` is the top of the lattice, so a FAIL cannot be displaced by a model that
    left a gate — the `if/elif` version had an absent arm outranking a measured `-100%`,
    found in three separate branches across three review rounds.

    The perturbation is applied to a model OTHER than the failing one, which is what makes
    this a `BLOCK`-preservation claim rather than the (false) strict-monotonicity one: the
    failing evidence is untouched, so nothing licenses the verdict to soften.

    `_WITHHOLDING` only, not `_PERTURBATIONS`: killing a model's calls outright pushes the
    run past the INCONCLUSIVE threshold, and refusing a verdict for a dead backend is a
    fleet-level decision that legitimately outranks a per-model FAIL — a run that is mostly
    transport errors cannot support a behavioral claim in either direction."""
    failing = _model(answer=0, retrieve=1, control=1)  # -100% final accuracy
    for shape in _SHAPES:
        for name, perturb in _WITHHOLDING.items():
            fleet = {"bad": failing, "other": _model(answer=shape[0], retrieve=shape[1],
                                                     control=shape[2])}
            assert dropeval_verdict(fleet).directive is Directive.BLOCK
            fleet["other"] = perturb(fleet["other"])
            assert dropeval_verdict(fleet).directive is Directive.BLOCK, (
                f"perturbing `other` ({shape}) with {name} displaced a demonstrated FAIL")


def test_shrinking_the_sample_only_ever_moves_the_verdict_away_from_SHIP():
    """`_shrink`'s share of the property above, asserted rather than assumed.

    `_MIN_PAIRED_QUESTIONS` is deliberately ASYMMETRIC — it withholds a model only when its
    form arm is not behind its control, so an exclusion it causes is always the removal of a
    would-be PASS. That is the argument in `_MIN_PAIRED_QUESTIONS`' own comment, and it is
    what makes `_shrink` unable to contribute a violation to
    `test_withholding_evidence_never_authorizes_a_ship`. Asserted here in its own right,
    because a perturbation that can never fail is otherwise indistinguishable from one that
    is silently a no-op — and `_reason_directive("underpowered") -> SHIP` survives every
    test in this module without it."""
    moved = 0
    for shapes in itertools.product(_SHAPES, repeat=2):
        fleet = {name: _model(answer=s[0], retrieve=s[1], control=s[2])
                 for name, s in zip("AB", shapes, strict=True)}
        for victim in "AB":
            before = dropeval_verdict(fleet).directive
            after = dropeval_verdict(dict(fleet, **{victim: _shrink(fleet[victim])})).directive
            assert after >= before or before is Directive.BLOCK, (shapes, victim, before, after)
            moved += after is not before
    assert moved, "shrink never changed a directive — the fixture is above no floor at all"
