"""#375 — a drop suggestion attached to a `tiers: []` rule, and the eval that scored it
silently to zero.

`generate_policy` scores a tool under `--threshold` -> `tiers: []`, then attaches
`_suggested_fields` anyway. `activate_suggestions` promoted the fields and left the tiers,
`policy.apply` read `tiers: []` as an explicit hands-off passthrough and returned before
the drop step, and `_questions_and_staging` early-exited into the same anonymous
`return [], None, None` as nine unrelated conditions. The tool contributed nothing to the
eval and appeared nowhere in the report — a run that never tested it read exactly like a
run it passed.

Three separable claims, three tests each side of them: the eval now MEASURES the rule
(tiers restored on the copy), the generator SAYS a rename alone is not enough, and a
payload that still yields no question is DISCLOSED with a reason rather than dropped.
"""

import json

import pytest

from terse import dropeval
from terse.policy import Rule
from terse.policy_gen import TIERS_RESTORED_KEY, activate_suggestions
from terse.report import render_drop_coverage

# A record list whose `body` clears DEFAULT_DROP_MIN and is unique per record, with a
# scalar id column — the shape `_questions_and_staging` needs end to end.
PAYLOAD = {"result": [{"id": i, "name": f"n{i}", "body": f"{i} " + "lorem ipsum dolor " * 40}
                      for i in range(6)]}
DROP_FIELDS = {"result[].body": {"lossy": "drop-to-retrieve", "min": 200}}


def _doc(tiers):
    return {"version": 1,
            "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
            "policies": [{"match": {"tool": "kb.read.list_principles"},
                          "tiers": tiers,
                          "_suggested_fields": dict(DROP_FIELDS),
                          "_suggested_fields_note": "…"}]}


# --------------------------------------------------------------------------- #
# 1. The mechanism: `tiers: []` is what zeroes the eval, not the field set
# --------------------------------------------------------------------------- #
def test_a_passthrough_rule_carrying_a_drop_selector_poses_no_question():
    """The defect's root fact, stated directly. Identical rules but for `tiers`; only the
    compressing one is testable. This is the control for everything below — if it ever
    starts passing with `tiers=()`, the rest of this file is measuring nothing."""
    compressing = Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"),
                       fields=dict(DROP_FIELDS))
    passthrough = Rule(tool_glob="*", tiers=(), fields=dict(DROP_FIELDS))

    assert dropeval._questions_and_staging(PAYLOAD, compressing, "t.x").questions
    assert not dropeval._questions_and_staging(PAYLOAD, passthrough, "t.x").questions


def test_passthrough_is_its_own_reason_not_the_size_floor_bucket():
    """`policy.apply` returns `skipped=True` for a dozen unrelated reasons, and `tiers: []`
    used to be an indistinguishable member of that bucket. An operator debugging a silent
    eval needs "your rule says passthrough", not "the size floor ate it"."""
    probe = dropeval._questions_and_staging(
        PAYLOAD, Rule(tool_glob="*", tiers=(), fields=dict(DROP_FIELDS)), "t.x")
    assert probe.reason == "passthrough_tiers"

    # ...and a genuine size-floor miss still reports the floor, so the two are not merged
    # by making everything say "passthrough".
    tiny = {"result": [{"id": i, "body": "x"} for i in range(6)]}
    floor = dropeval._questions_and_staging(
        tiny, Rule(tool_glob="*", tiers=("minify", "tabularize"),
                   fields=dict(DROP_FIELDS)), "t.x")
    assert floor.reason == "size_floor"


def test_every_empty_probe_carries_a_reason_and_every_full_one_carries_none():
    """The invariant the whole disclosure rests on, rather than one test per exit: `reason`
    is set exactly when `questions` is empty. A future early-exit that forgets to tag
    itself fails here instead of silently reopening the hole."""
    cases = [
        (PAYLOAD, Rule(tool_glob="*", tiers=("minify",), fields={})),           # no spec
        (PAYLOAD, Rule(tool_glob="*", tiers=(), fields=dict(DROP_FIELDS))),      # passthrough
        ({"a": 1}, Rule(tool_glob="*", tiers=("minify",), fields=dict(DROP_FIELDS))),
        (PAYLOAD, Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"),
                       fields=dict(DROP_FIELDS))),                               # the full one
    ]
    for obj, rule in cases:
        probe = dropeval._questions_and_staging(obj, rule, "t.x")
        assert (probe.reason is None) == bool(probe.questions), probe.reason
        if probe.reason is not None:
            assert probe.reason in dropeval.DROP_SKIP_REASONS


# --------------------------------------------------------------------------- #
# 2. `activate_suggestions` restores tiers so the eval evaluates something
# --------------------------------------------------------------------------- #
def test_activation_restores_default_tiers_on_a_passthrough_entry():
    entry = activate_suggestions(_doc([]))["policies"][0]
    assert entry["tiers"] == ["minify", "tabularize", "dictionary"]
    assert entry[TIERS_RESTORED_KEY] is True
    assert entry["fields"] == DROP_FIELDS          # the suggestion was promoted too
    assert "_suggested_fields" not in entry


def test_activation_leaves_a_compressing_entry_and_its_marker_alone():
    """The lift is scoped to the broken case. A rule that already compresses keeps the
    tiers the corpus measured for it, and carries no marker — so the disclosure line in
    `_tune_drop_eval` names only the rules that actually needed lifting."""
    entry = activate_suggestions(_doc(["minify"]))["policies"][0]
    assert entry["tiers"] == ["minify"]
    assert TIERS_RESTORED_KEY not in entry


def test_activation_honors_a_docs_own_defaults_rather_than_the_builtin_triple():
    doc = _doc([])
    doc["defaults"]["tiers"] = ["minify"]
    assert activate_suggestions(doc)["policies"][0]["tiers"] == ["minify"]


def test_an_entry_with_no_suggestion_is_never_lifted():
    """The lift is a consequence of promoting a drop, not a blanket tiers rewrite: a
    deliberate `tiers: []` rule with nothing to promote must stay passthrough, or the eval
    copy would quietly compress tools the operator excluded."""
    doc = _doc([])
    doc["policies"][0].pop("_suggested_fields")
    entry = activate_suggestions(doc)["policies"][0]
    assert entry["tiers"] == []
    assert TIERS_RESTORED_KEY not in entry


def test_the_activated_policy_actually_produces_questions_end_to_end():
    """The point of the lift, proved through `load_policy` rather than by inspecting the
    dict — the eval's real path is doc -> JSON -> load_policy -> select -> probe, and a
    marker key the loader rejected would break it there and nowhere earlier."""
    import tempfile
    from pathlib import Path

    from terse.policy import load_policy

    for doc, expected in ((_doc([]), True), (_doc([]), True)):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(json.dumps(activate_suggestions(doc)))
            name = tf.name
        try:
            pol = load_policy(name)
        finally:
            Path(name).unlink(missing_ok=True)
        assert pol.has_drop()
        rule = pol.select("kb.read.list_principles", None)
        assert bool(dropeval._questions_and_staging(PAYLOAD, rule, "kb.read.list_principles")
                    .questions) is expected


# --------------------------------------------------------------------------- #
# 3. Disclosure: a skipped payload is reported, not omitted
# --------------------------------------------------------------------------- #
def test_coverage_reports_the_passthrough_tool_with_its_reason():
    envelopes = [{"tool": "kb.read.list_principles", "server": "kb", "sha": "abc",
                  "raw": json.dumps(PAYLOAD)}]
    rows = dropeval.drop_eval_coverage(
        envelopes, lambda t, s=None: Rule(tool_glob="*", tiers=(), fields=dict(DROP_FIELDS)))
    assert rows == [{"tool": "kb.read.list_principles", "server": "kb", "sha": "abc",
                     "reason": "passthrough_tiers"}]


def test_coverage_is_silent_when_every_payload_was_evaluated():
    """A fully-covered run must not print a scary empty section — the disclosure has to
    mean something when it appears."""
    envelopes = [{"tool": "t.x", "server": None, "sha": "abc", "raw": json.dumps(PAYLOAD)}]
    rule = Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"),
                fields=dict(DROP_FIELDS))
    assert dropeval.drop_eval_coverage(envelopes, lambda t, s=None: rule) == []
    assert render_drop_coverage([]) == ""


def test_the_rendered_section_names_the_tool_the_count_and_the_remedy():
    text = render_drop_coverage([
        {"tool": "kb.read.list_principles", "server": "kb", "sha": "a",
         "reason": "passthrough_tiers"},
        {"tool": "kb.read.list_principles", "server": "kb", "sha": "b",
         "reason": "passthrough_tiers"},
        {"tool": "gh.search", "server": "gh", "sha": "c", "reason": "size_floor"},
    ])
    assert "kb.read.list_principles" in text
    assert "2 payload(s)" in text
    assert "NOT a pass" in text                     # cannot be read as an endorsement
    assert "tiers" in text and "terse stats" in text  # the remedy, and #274's cross-check
    # Passthrough sorts first: it is the reason that means the RULE is wrong, not that the
    # payload was unsuitable.
    assert text.index("passthrough_tiers") < text.index("size_floor")


def test_a_tool_that_was_never_under_test_is_counted_not_itemized():
    """The section has to stay readable to stay useful. On the real session corpus 1,092 of
    1,515 skipped payloads are tools with no drop selector at all — itemizing those buries
    the handful an operator can act on under a wall of "this tool has no drop configured".
    They are still COUNTED, so the section accounts for every payload and the collapse
    cannot be used to hide something."""
    text = render_drop_coverage(
        [{"tool": "kb.read.list_principles", "server": "kb", "sha": "a",
          "reason": "passthrough_tiers"}]
        + [{"tool": f"boring{i}", "server": None, "sha": str(i), "reason": "no_drop_spec"}
           for i in range(9)])
    assert "kb.read.list_principles" in text
    assert "boring3" not in text                    # collapsed, not itemized
    assert "9 further payload(s)" in text           # but accounted for
    assert "1 payload(s) carry a drop selector" in text


def test_a_run_whose_every_skip_is_out_of_scope_still_renders_the_count():
    """The all-collapsed case must not produce a section that claims something was under
    test when nothing was — the "NOT a pass" sentence is a real claim and belongs only
    where it is true."""
    text = render_drop_coverage(
        [{"tool": "t", "server": None, "sha": "a", "reason": "no_drop_spec"}])
    assert "1 further payload(s)" in text
    assert "NOT a pass" not in text
    assert "carry a drop selector" not in text


def test_coverage_and_the_scored_run_never_disagree_about_a_payload():
    """The shared-probe invariant. Two loops deciding "was this evaluated?" independently
    is the defect class this issue is about, so the property is pinned directly: every
    envelope is either scored or disclosed, never both and never neither."""
    envelopes = [
        {"tool": "yes", "server": None, "sha": "1", "raw": json.dumps(PAYLOAD)},
        {"tool": "no", "server": None, "sha": "2", "raw": json.dumps(PAYLOAD)},
        {"tool": "no", "server": None, "sha": "3", "raw": "not json at all"},
    ]
    rules = {"yes": Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"),
                         fields=dict(DROP_FIELDS)),
             "no": Rule(tool_glob="*", tiers=(), fields=dict(DROP_FIELDS))}

    def rule_for(tool, server=None):
        return rules[tool]


    results = dropeval.run_drop_fluency(
        envelopes, rule_for,
        {"m": lambda messages: dropeval.Turn(text="I don't know", tool_calls=[])},
        trials=1, control=False)
    scored = {r["sha"] for r in results["m"]}
    disclosed = {r["sha"] for r in dropeval.drop_eval_coverage(envelopes, rule_for)}

    assert scored == {"1"}
    assert disclosed == {"2", "3"}
    assert scored.isdisjoint(disclosed)
    assert scored | disclosed == {"1", "2", "3"}    # nothing fell through unaccounted for


# --------------------------------------------------------------------------- #
# 4. The generator stops implying a rename is enough
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threshold,passthrough", [(5.0, True), (1.0, False)])
def test_the_suggestion_note_warns_only_when_the_rule_is_passthrough(threshold, passthrough):
    """Built through `generate_policy` from a real corpus rather than asserted on a
    hand-made dict: the note and the tiers are decided in the same loop, and the whole
    point is that they were allowed to contradict each other.

    ONE variable moves — the savings threshold, straddling this corpus's measured 4.9%.
    Same payloads, same drop suggestion both times, so the only difference between the two
    runs is whether the tool cleared the bar: exactly the branch that produced the inert
    suggestion in the first place."""
    from terse.policy_gen import generate_policy

    # Unique per record (the dictionary tier can't fold it -> a drop candidate) and well
    # over the mean-token floor, so `drop_suggestion` fires on both runs.
    raw = json.dumps({"result": [{"id": i, "name": f"n{i}",
                                  "body": f"record {i} " + "lorem ipsum dolor sit amet " * 40}
                                 for i in range(20)]})
    doc, _rows = generate_policy(
        [{"tool": "t.x", "server": None, "sha": str(i), "raw": raw} for i in range(3)],
        threshold=threshold)
    entry = doc["policies"][0]
    assert "_suggested_fields" in entry, "precondition: this corpus yields a drop suggestion"
    assert (not entry["tiers"]) is passthrough, "precondition: the threshold set the tiers"

    note = entry["_suggested_fields_note"]
    assert ("renaming ALONE enables nothing" in note) is passthrough
    if passthrough:
        assert "terse stats" in note      # #274's cross-check, reachable from the note


# --------------------------------------------------------------------------- #
# 5. The final directive cannot prescribe the rename the run knows is inert
# --------------------------------------------------------------------------- #
# Review finding on the first cut of this fix: `dropeval_next_step_line` on a SHIP verdict
# said "enable the verified fields by renaming '_suggested_fields' -> 'fields'". For a rule
# whose `tiers: []` the eval had just lifted, that rename alone leaves the passthrough in
# place and the drop never fires — #375 restated as an instruction the operator was told to
# follow. The warning printed before the eval is hundreds of report lines away by then, and
# the coverage section cannot re-flag those rules precisely BECAUSE they were lifted.
def _fleet(*, recall, precision, n=24):
    """Rows in the shape `run_drop_fluency` emits — one control-paired trial each. `recall`
    and `precision` are `(answer_ok, retrieve_ok)`; the control arm is always correct, so
    the treatment is the only thing under test."""
    out = []
    for kind, (answer, retrieve) in (("recall", recall), ("precision", precision)):
        out += [{"kind": kind, "trials": 1, "retrieve_ok": retrieve, "answer_ok": answer,
                 "handle_ok": 1, "errors": 0, "treatment_errors": 0, "control_errors": 0,
                 "attempts": 2, "qid": f"{kind}-q{i}", "control_ok": 1, "control_trials": 1}
                for i in range(n)]
    return {"m": out}


def test_the_ship_directive_names_the_lifted_rules_as_needing_tiers_too():
    from terse.report import Directive, dropeval_next_step_line, dropeval_verdict

    v = dropeval_verdict(_fleet(recall=(1, 1), precision=(1, 1)))
    assert v.directive is Directive.SHIP, "precondition: this fixture ships"

    plain = dropeval_next_step_line(v)
    lifted = dropeval_next_step_line(v, tiers_restored=["kb.read.list_nodes",
                                                       "kb.read.list_principles"])
    assert "renaming" in plain and "NOT sufficient" not in plain
    assert "NOT sufficient" in lifted
    assert "kb.read.list_principles" in lifted and "kb.read.list_nodes" in lifted
    assert "tiers" in lifted
    # Sorted, so two runs over the same policy print the same sentence.
    assert lifted.index("kb.read.list_nodes") < lifted.index("kb.read.list_principles")


def test_a_declined_verdict_says_nothing_about_tiers_either_way():
    """The caveat rides on the SHIP branch only. A verdict that already tells the operator
    to change nothing must not grow an "...and also set tiers" clause, which would read as a
    partial authorization the measurement does not support."""
    from terse.report import Directive, dropeval_next_step_line, dropeval_verdict

    v = dropeval_verdict(_fleet(recall=(0, 0), precision=(1, 1)))
    assert v.directive is not Directive.SHIP, "precondition: this fixture does not ship"
    line = dropeval_next_step_line(v, tiers_restored=["kb.read.list_principles"])
    assert "does NOT authorize" in line
    assert "NOT sufficient" not in line and "kb.read.list_principles" not in line


def test_tune_drop_eval_hands_the_lifted_rules_to_the_directive(monkeypatch, capsys):
    """The WIRING, not just the sentence. A mutation replacing `_tune_drop_eval`'s
    `tiers_restored=lifted` with `()` left all 1,982 other tests green — the renderer was
    pinned and the call site that feeds it was not, so the caveat could be dropped on the
    only path that ever produces it.

    Spies on the seam rather than driving a live SHIP verdict, deliberately: the sentence
    itself is pinned by `test_the_ship_directive_names_the_lifted_rules_as_needing_tiers_too`
    against both branches, so what is left to prove is that the CLI hands over the SAME
    lifted set it built the pre-eval note from. Reaching SHIP here would need five distinct
    corpus payloads (`_FIXED_IDEAL_MIN_QUESTIONS`) and a stub model per payload, all to
    re-assert a string this file already owns."""
    import argparse

    from terse import cli, dropeval, report

    seen: dict = {}

    def spy(v, tiers_restored=()):
        seen["tiers_restored"] = list(tiers_restored)
        return "…"

    monkeypatch.setattr(report, "dropeval_next_step_line", spy)
    monkeypatch.setattr(cli, "_build_answerers",
                        lambda args, make: {"m": lambda messages: dropeval.Turn(text="no")})

    doc = _doc([])          # one passthrough rule carrying a suggestion — the #375 shape
    env = {"tool": "kb.read.list_principles", "server": None, "sha": "a",
           "raw": json.dumps(PAYLOAD)}
    args = argparse.Namespace(trials=1, no_control=True, accept_degraded=False)
    assert cli._tune_drop_eval(args, doc, [env]) == 0

    assert seen["tiers_restored"] == ["kb.read.list_principles"]
    # ...and the same set reached the pre-eval note, so the two disclosures cannot diverge.
    assert "kb.read.list_principles" in capsys.readouterr().out
