"""A 200 carrying no content is a NON-ANSWER, not a wrong answer (#268).

#264 established that a call which never happened must not be scored, and gated every
report on it. That gate keys on TRANSPORT failures — exceptions `_safe_ask` turns into
`None`. A model that returns HTTP 200 with `content: null` raises nothing: it flowed
through as a real reply of `""`, was scored (wrong), and was invisible to `_unmeasured`.

When it happens across the board every arm reads 0%, the gap between them is exactly 0,
and the diff report prints **PASS** — a model that answered nothing green-lighting the
`proxy --diff` default flip. Same false-verdict class as #263, by a route #264's gate
cannot see. Observed live: `gemini-3.6-flash` returning null content when reasoning
consumed the token budget.

The fix normalises no-content to `None` in `_safe_ask` — the choke point every harness
funnels through, not just the one adapter — so it joins the existing failure channel and is
counted rather than scored.

Review of that fix found it opened a second hole, pinned here too: #268's live cause scales
with PROMPT LENGTH, and the diff arm's prompt is longer than its control's, so the arm under
test is systematically the arm that fails. Excluding those calls per-arm while gating on the
POOLED share turned a real -40% FAIL into a +0% PASS. `_unmeasured` grew a per-arm trigger;
`test_no_content_correlated_with_the_LONGER_arm_cannot_publish` holds it.
"""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

from terse.fluency.answerers import openai_answerer
from terse.fluency.harnesses import _ask_n
from terse.fluency.scoring import score
from terse.report import _unmeasured


def _gateway(monkeypatch, choice: dict):
    """Stub the HTTP layer so a real `openai_answerer` sees `choice` verbatim."""
    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    payload = json.dumps({"choices": [choice]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(payload))
    return openai_answerer("https://api.example/v1", "k", "empty-replying-model")


@pytest.mark.parametrize("choice, why", [
    ({"message": {"content": None}, "finish_reason": "length"}, "content: null"),
    ({"message": {"content": ""}, "finish_reason": "stop"}, "content: empty string"),
    ({"message": {"content": "   \n "}, "finish_reason": "stop"}, "content: whitespace"),
    ({"message": {}, "finish_reason": "content_filter"}, "content key absent"),
])
def test_no_content_is_reported_as_a_non_answer(monkeypatch, capsys, choice, why):
    """`content or ""` collapsed all four of these into a real empty reply.

    Blank is treated like null deliberately, and this is the explicit decision #268 asked
    for: no question has an empty expected answer (`questions.py` excludes them), so an
    empty reply can never be CORRECT. Scoring it wrong would charge terse for a backend
    quirk; counting it as a non-answer makes the report decline to publish. Refusing to
    answer is the safe direction — a false PASS is not."""
    assert _gateway(monkeypatch, choice)("", "q") is None, why
    err = capsys.readouterr().err
    assert "returned no content" in err and "empty-replying-model" in err, why
    # finish_reason is the actionable half: `length` -> raise max_tokens,
    # `content_filter` -> the payload tripped a safety filter. Neither is "unreachable".
    assert repr(choice.get("finish_reason")) in err, why


def test_real_content_still_comes_back_unchanged(monkeypatch):
    ask = _gateway(monkeypatch, {"message": {"content": "6"}, "finish_reason": "stop"})
    assert ask("", "q") == "6"


def test_a_no_content_model_cannot_green_light_a_verdict(monkeypatch):
    """The end-to-end defect, as reported: an answerer that produces nothing used to
    yield fails=0, so `_unmeasured` stayed False and the report published 0% vs 0% as a
    PASS. Now every trial lands in the failure count and the gate fires."""
    ask = _gateway(monkeypatch, {"message": {"content": None}, "finish_reason": "length"})
    ok, fails = _ask_n(ask, "", "q", "count", 6, 3)
    assert (ok, fails) == (0, 3), "no content must be counted, never scored"

    rows = [{"terse_ok": 0, "terse_trials": 0, "diff_ok": 0, "diff_trials": 0,
             "fails": 6, "attempts": 6} for _ in range(9)]
    assert _unmeasured(rows) is True, \
        "a panel where nothing was answered must not publish a verdict"


def test_no_content_is_never_scored_as_CORRECT(monkeypatch):
    """The quiet half. `score("lookup", "", "")` is True, so a no-content reply fed to
    `score` as `""` counts as a RIGHT answer — worse than counting it wrong, and the same
    secondary hazard #263 names. Only this assertion holds it on the no-content path."""
    assert score("lookup", "", "") is True, "the premise this test depends on"
    ask = _gateway(monkeypatch, {"message": {"content": ""}, "finish_reason": "stop"})
    assert _ask_n(ask, "", "q", "lookup", "", 3) == (0, 3)


# --- the same defect in dropeval, which is what a drop-rule verdict rests on ---

def test_dropeval_no_content_is_an_error_not_a_wrong_answer(monkeypatch, capsys):
    """`dropeval` had the identical `content or ""`, and it lands in the same place:
    `_run_question` feeds the text straight to `fluency.score`. A 200 with `content: null`
    scored as a wrong answer while `Turn.error` stayed False — invisible in the accuracy
    columns, which is precisely the confound #269 must see past before a drop-rule verdict
    can mean anything. Found while fixing #268, not reported separately.

    Note what `error=True` does NOT do here: the trial is still scored as a miss. That is
    deliberate and is the conservative direction on this side — a non-answer counted as a
    miss can only under-sell enabling a lossy drop tier, never over-sell it. See
    `report._unmeasured`, where excluding prompt-length-correlated failures turned a real
    FAIL into a PASS."""
    from terse.dropeval import openai_tool_answerer
    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def gw(choice):
        payload = json.dumps({"choices": [choice]}).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(payload))
        return openai_tool_answerer("https://api.example/v1", "k", "m", [])

    # All four no-content shapes, matching the fluency side rather than only `null` —
    # a blank-only mutation survived the whole suite when this tested `None` alone.
    for content, reason in ((None, "length"), ("", "stop"), ("   \n ", "stop"),
                            (..., "content_filter")):
        msg = {} if content is ... else {"content": content}
        turn = gw({"message": msg, "finish_reason": reason})([])
        assert turn.error is True, f"no content ({content!r}) must be recorded"
        err = capsys.readouterr().err
        assert "returned no content" in err and repr(reason) in err, content

    # ...but an empty text WITH a tool call is a NORMAL retrieve turn, not a failure.
    # Flagging it would mark every correct drop-to-retrieve run as an error.
    call = {"id": "1", "function": {"name": "terse_retrieve", "arguments": "{}"}}
    turn = gw({"message": {"content": "", "tool_calls": [call]}, "finish_reason": "tool_calls"})([])
    assert turn.error is False and len(turn.tool_calls) == 1
    assert "returned no content" not in capsys.readouterr().err


# --- the reported symptom, end to end, and the trap the FIX itself opened ---

def _diff_rows_ab(a, b, answerer, trials=4):
    from terse.fluency.harnesses import run_diff_payload
    return run_diff_payload(a, b, answerer, "t", trials)


def _diff_rows(answerer, trials=4):
    a = [{"id": i, "x": "a"} for i in range(4)]
    b = [{"id": i, "x": "b"} for i in range(4)]
    return _diff_rows_ab(a, b, answerer, trials)


def test_a_no_content_model_cannot_produce_a_PASS_on_the_diff_ship_gate():
    """The defect exactly as #268 reports it — through the real pipeline, not hand-built
    rows. `test_a_dead_backend_cannot_produce_a_PASS_on_the_diff_ship_gate` is the
    precedent; this is the same gate reached by a route that raises no exception."""
    from terse.report import build_diff_report
    rows = _diff_rows(lambda s, u: None)   # a 200 with no content, post-#268
    assert rows, "no rows generated — this would pin nothing"
    text = build_diff_report({"empty-replying-model": rows})
    verdict = text.split("## Verdict")[1]
    assert "safe to enable" not in verdict, \
        "a model that answered nothing must never green-light --diff"
    assert "NO VERDICT — nothing was measured" in verdict


def test_no_content_correlated_with_the_LONGER_arm_cannot_publish():
    """The trap the #268 fix itself opened, and the reason `_unmeasured` grew a per-arm
    trigger.

    #268's live cause is reasoning eating the token budget — a failure that scales with
    PROMPT LENGTH. The diff arm's prompt is strictly longer than its control's, so the arm
    under test is systematically the arm that returns nothing. Excluding those calls from
    that arm's own denominator, while gating only on the POOLED share, let a diff arm lose
    40% of its calls (20% pooled, and the comparison is strictly `>`) and still publish.

    Measured before the per-arm trigger: a real `-40% FAIL` rendered as `+0% PASS`, over a
    model that produced no content on 16 of 40 diff calls. Excluding a non-answer is only
    the safe direction while the exclusions are UNCORRELATED with the arm being measured."""
    from terse.report import build_diff_report

    seen = {"n": 0}
    def truncates_on_the_long_prompt(system, user):
        # Answer the control perfectly; produce nothing on 40% of the longer diff prompt.
        if "UPDATE" in user:
            seen["n"] += 1
            if seen["n"] % 5 in (1, 2):
                return None
        return "b"

    rows = _diff_rows(truncates_on_the_long_prompt, trials=10)
    assert rows, "no rows generated"
    lost = sum(r["trials"] - r["diff_trials"] for r in rows)
    attempted = sum(r["trials"] for r in rows)
    assert 0.2 < lost / attempted <= 0.5, \
        f"fixture must lose enough of the diff arm ALONE to matter: {lost}/{attempted}"
    assert sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows) <= 0.20, \
        "and must stay UNDER the pooled threshold, or it proves nothing about the per-arm one"

    verdict = build_diff_report({"gemini-truncating": rows}).split("## Verdict")[1]
    assert "safe to enable" not in verdict, (
        "a diff arm that silently lost 40% of its calls must not green-light --diff:\n"
        + verdict)


def test_an_unanswerable_question_is_never_generated():
    """`score("lookup", "", "")` is True, so a question whose expected answer is empty
    scores a model that answered NOTHING as CORRECT. `_flat_record_questions` had excluded
    empties since #263; the table and nested paths had not — `_pick_target_col`'s fallback
    returns a column whose every value is `""` (found reviewing #268).

    This is also the premise the blank-reply handling rests on. Rather than weaken that to
    fit the exception, the exception is removed: a question no wrong answer can fail was
    never measuring comprehension."""
    from terse.fluency.questions import gen_questions
    recs = [{"id": 1, "note": "", "n": 5}, {"id": 2, "note": "", "n": 7},
            {"id": 3, "note": "  ", "n": 9}]
    for payload in ({"items": recs}, {"g": {"a": {"rows": recs}}}):
        for q in gen_questions(payload):
            assert q.expected != "" and str(q.expected).strip() != "", \
                f"unfalsifiable question generated: {q.qtype} expects {q.expected!r}"


def test_every_renderer_describes_an_unanswered_call_the_same_way():
    """One condition, three renderers, one wording.

    The markdown verdicts were re-worded for #268 (a token-budget stop DID reach the
    backend, so "never reached … re-run once the backend is reachable" prescribed the wrong
    fix) — but the excluded-list line, the HTML card and the terminal forest plot were
    missed, so one run described one exclusion two different ways depending on output
    format. A reader comparing them concludes they are two different problems.

    Asserted over RENDERED OUTPUT, not source text: the first cut grepped the modules and
    failed on a docstring that quotes the old phrase while explaining why it changed —
    pinning prose rather than behaviour."""
    from terse.html_report import build_html_diff_report
    from terse.report import build_diff_report
    from terse.terminal_report import build_terminal_diff_report

    rows = _diff_rows(lambda s, u: None)          # every call unanswered -> excluded
    healthy = [{"qid": f"q{i}", "qtype": "count", "transform": "table", "trials": 2,
                "terse_ok": 2, "terse_trials": 2, "diff_ok": 2, "diff_trials": 2,
                "fails": 0, "attempts": 4} for i in range(4)]
    results = {"good": healthy, "dead-model": rows}

    rendered = {
        "markdown": build_diff_report(results),
        "html": build_html_diff_report(results),
        "terminal": build_terminal_diff_report(results, color=False),
    }
    for name, text in rendered.items():
        assert "never reached the backend" not in text, (
            f"the {name} renderer still tells the reader the call never reached the "
            f"backend; since #268 an exclusion can also mean the backend answered with no "
            f"content, for which that is the wrong remedy")
        # ...and it must still NAME the excluded model. A sweep that deleted the line
        # entirely would satisfy the assertion above and say less than before.
        assert "unanswered" in text and "dead-model" in text, name


# --- pairing: what a per-arm THRESHOLD could not close ---

def test_losing_one_question_TYPE_from_one_arm_cannot_publish():
    """The hole a per-arm loss threshold leaves open, and why the fix moved to pairing.

    Scoring is PAIRED — `harnesses`' module docstring calls that load-bearing: the same
    questions, same order, to every arm. `_form_stats` divides each arm by its OWN
    `<form>_trials`, so dropping an unanswered call re-bases that arm onto a DIFFERENT
    question set. Question difficulty varies far more than trial noise, so losing the HARD
    questions from one arm flatters it without bound — at ANY share.

    Measured with a per-arm threshold in place: a model answering the control perfectly and
    returning no content on the longest-prompt question type lost exactly 1 of 5 types —
    20.0%, on the boundary, and the comparison was strictly `>`, so the gate stayed quiet
    while a real -20% FAIL published as PASS / "safe to enable `proxy --diff`"."""
    from terse.report import build_diff_report

    def kills_the_hard_question(system, user):
        if "UPDATE" in user and ("value of" in user or "deref" in user.lower()):
            return None                      # no content, diff arm, hardest prompt only
        return "b"

    a = [{"id": i, "x": "a", "note": f"n{i}"} for i in range(4)]
    b = [{"id": i, "x": "b", "note": f"m{i}"} for i in range(4)]
    rows = _diff_rows_ab(a, b, kills_the_hard_question, trials=3)
    assert rows, "no rows generated"
    assert any(r["diff_trials"] < r["trials"] for r in rows), \
        "fixture must actually lose diff-arm calls, or it pins nothing"
    assert all(r["terse_trials"] == r["trials"] for r in rows), \
        "and must lose them from ONE arm only — that is the whole point"

    verdict = build_diff_report({"panel-model": rows}).split("## Verdict")[1]
    assert "safe to enable" not in verdict, (
        "an arm missing a whole question type must not green-light --diff:\n" + verdict)


def test_pairing_drops_a_row_from_BOTH_arms_not_just_the_one_that_failed():
    """The mechanism, directly. `paired_rows` is what makes the surviving gap compare like
    with like; `unpaired` is what declines when too little survives to be representative."""
    from terse.report import paired_rows, unpaired
    full = {"qid": "a", "trials": 2, "terse_ok": 2, "terse_trials": 2,
            "diff_ok": 2, "diff_trials": 2}
    half = {"qid": "b", "trials": 2, "terse_ok": 2, "terse_trials": 2,
            "diff_ok": 1, "diff_trials": 1}          # diff arm lost one trial
    assert paired_rows([full, half], "terse_ok", "diff_ok") == [full]
    # 1 of 5 rows unpaired is exactly 20.0%: the measured boundary case, so `unpaired`
    # uses `>=` where the pooled share uses `>`. On a ship gate the boundary refuses.
    assert unpaired([full] * 4 + [half], "terse_ok", "diff_ok") is True
    assert unpaired([full] * 9 + [half], "terse_ok", "diff_ok") is False


def test_an_ancillary_arm_does_not_void_a_measured_verdict():
    """The cost of the threshold this replaced: `_unmeasured` discovers arms from row keys,
    so `run_payload`'s `inline` arm gated the terse-vs-raw verdict — and `inline_u` carries
    the LONGEST prompt of the four, so a token-budget stop truncates it first. Fixing
    #268's detection must not start discarding multi-hour runs whose actual verdict was
    fully measured."""
    from terse.report import _unmeasured
    rows = [{"qid": f"q{i}", "trials": 4, "attempts": 16, "fails": 1,
             "raw_ok": 4, "raw_trials": 4, "terse_ok": 4, "terse_trials": 4,
             "primer_ok": 4, "primer_trials": 4,
             "inline_ok": 3, "inline_trials": 3} for i in range(4)]
    assert _unmeasured(rows) is False, \
        "the arm that gates nothing must not void the run"


def test_a_non_string_reply_does_not_abort_the_panel_run():
    """`_safe_ask`'s first promise is that one bad call never aborts a long multi-model
    run. The #268 normalisation was first written AFTER the `except`, so a user-supplied
    answerer — the module's documented extension point — returning a non-str made
    `.strip()` raise AttributeError straight past the handler."""
    from terse.fluency.harnesses import _safe_ask
    assert _safe_ask(lambda s, u: 42, "", "q") == 42        # coerced for the blank test only
    assert _safe_ask(lambda s, u: 0, "", "q") == 0          # falsy but a real reply
    assert _safe_ask(lambda s, u: object(), "", "q") is not None


def test_a_none_expected_answer_is_also_unanswerable():
    """`_answerable` guards `None` as well as blank, and that branch is reachable:
    `_pick_target_col` returns on `any(...)` with no all-scalar requirement, so a column
    whose probed value is None can be chosen as the lookup target."""
    from terse.fluency.questions import _answerable
    assert _answerable(None) is False
    assert _answerable("") is False and _answerable("  ") is False
    assert _answerable(0) is True and _answerable("x") is True and _answerable([]) is True


def test_an_arm_with_zero_completed_trials_still_voids_the_run():
    """`_unmeasured`'s first trigger, pinned independently. The per-arm share trigger that
    briefly existed covered this case incidentally; with it removed, only this holds it."""
    from terse.report import _unmeasured
    # FIVE arms, one of them dead. The first cut of this test used two arms, where the
    # dead one is 50% of all calls — so the POOLED trigger fired and the assertion held
    # with trigger 1 deleted, proving nothing. At five arms the loss is exactly 20.0%,
    # which the pooled rule's `>` does not catch, so only trigger 1 can.
    arms = ("raw", "terse", "primer", "inline", "diff")
    rows = []
    for i in range(9):
        r = {"qid": f"q{i}", "trials": 2, "attempts": 10, "fails": 2}
        for a in arms:
            dead = a == "diff"
            r[f"{a}_ok"] = 0 if dead else 2
            r[f"{a}_trials"] = 0 if dead else 2
        rows.append(r)
    assert sum(r["fails"] for r in rows) / sum(r["attempts"] for r in rows) == 0.20, \
        "fixture must sit ON the pooled boundary, or the pooled rule catches it instead"
    assert _unmeasured(rows) is True, "an arm that completed nothing cannot be scored"
