"""A transport failure is NOT a wrong answer (#263).

`_safe_ask` used to return `""` on any exception, which `score` counted as incorrect. An
unreachable model therefore reported ~0% accuracy, indistinguishable from one that could
not read terse's compressed form — and because the verdict gates on the WORST model, a
single rate-limited backend did not dilute a panel, it decided it.

The report already excluded a model whose raw control was exactly 0%. That guard catches a
TOTAL outage only, and only after scoring it as if the model had answered. It does not
catch a PARTIAL rate limit, which leaves `raw` non-zero while depressing every arm — the
case that reaches a plausible-looking verdict. These tests pin both.
"""

from __future__ import annotations

from terse.fluency.harnesses import _ask_n, _safe_ask, run_payload
from terse.report import build_fluency_report

PAYLOAD = [{"id": i, "state": "open", "repo": "acme/widgets"} for i in range(6)]


def _dead(system, user):
    raise ConnectionError("litellm.RateLimitError: geminiException")


def _perfect_but_flaky(fail_every):
    """Answers correctly, except every Nth call raises — a partial rate limit."""
    calls = {"n": 0}

    def ask(system, user):
        calls["n"] += 1
        if calls["n"] % fail_every == 0:
            raise ConnectionError("rate limited")
        return "6"
    return ask


# --- the primitive ---

def test_a_transport_failure_returns_none_not_an_empty_string():
    """None is distinct from every real reply, including an empty one. As `""` it could
    also MATCH a question whose expected answer was empty — scoring a total failure as
    CORRECT, which is why `questions.py` excludes such questions defensively."""
    assert _safe_ask(_dead, "", "q") is None
    assert _safe_ask(lambda s, u: "", "", "q") == ""      # a real empty reply is preserved


def test_ask_n_counts_failures_separately_and_never_scores_them():
    ok, fails = _ask_n(_dead, "", "q", "count", 6, 3)
    assert (ok, fails) == (0, 3), "a call that never happened must not be scored at all"
    ok, fails = _ask_n(lambda s, u: "6", "", "q", "count", 6, 3)
    assert (ok, fails) == (3, 0)


def test_run_payload_carries_the_failure_count_into_the_row():
    rows = run_payload(PAYLOAD, "[]", _dead, primer="P", trials=2)
    assert rows, "the corpus must generate questions or this pins nothing"
    for r in rows:
        # 4 arms x 2 trials, every one of them a failure.
        assert r["fails"] == 8
        assert r["raw_ok"] == 0 and r["terse_ok"] == 0


# --- the report ---

def _report(rows_by_model):
    return build_fluency_report(rows_by_model, [])


def _row(**kw):
    base = {"tool": "t", "sha": "s", "qid": "q", "qtype": "count", "transform": "table",
            "trials": 1, "raw_ok": 1, "terse_ok": 1, "primer_ok": 1, "inline_ok": 1,
            "fails": 0}
    return {**base, **kw}


def test_an_unreachable_model_publishes_no_accuracy_at_all():
    """Not a footnoted 0% — no number. A percentage beside a model name is read as
    comprehension however it is annotated."""
    text = _report({"dead-model": [_row(raw_ok=0, terse_ok=0, primer_ok=0,
                                        inline_ok=0, fails=4)]})
    assert "| `dead-model` | 1 | n/a | n/a | n/a | n/a | n/a | n/a |" in text
    assert "0%" not in text.split("## Verdict")[0].split("| `dead-model`")[1][:60]
    assert "Not measured" in text and "4 failed call(s)" in text


def test_a_partially_failing_model_cannot_force_a_regression_verdict():
    """THE case the pre-existing `raw == 0` guard misses, pinned by its EFFECT rather than
    by the presence of an exclusion line — the line is emitted from a separate list, so
    asserting on it alone passed even with the model back in the gate.

    A partial rate limit leaves `raw` non-zero (control never fires) while depressing the
    terse arms. Gated, that is a worst-model regression and the whole panel returns FAIL on
    a backend that was simply half down."""
    healthy = [_row() for _ in range(4)]                       # 100% everywhere
    flaky = [_row(raw_ok=1, terse_ok=0, primer_ok=0, inline_ok=0, fails=2) for _ in range(4)]
    text = _report({"good": healthy, "flaky": flaky})
    verdict = text.split("## Verdict")[1]
    assert "Excluded (calls never reached the backend — not measured)" in verdict
    assert "`flaky`" in verdict
    # The healthy model is measured and holds, so the run must PASS. With `flaky` gated,
    # its 0% terse against 100% raw is a -100pt worst-case gap and this reads FAIL.
    assert "regresses beyond tolerance" not in verdict
    assert "preserves comprehension" in verdict


def test_a_run_where_every_model_failed_states_a_NON_verdict_rather_than_staying_silent():
    """Silence is how a run that measured NOTHING gets read as a run that found nothing
    wrong. Absence of a regression is not evidence of comprehension."""
    text = _report({"dead": [_row(raw_ok=0, terse_ok=0, primer_ok=0, inline_ok=0, fails=4)]})
    verdict = text.split("## Verdict")[1]
    assert "NO VERDICT — nothing was measured" in verdict
    assert "preserves comprehension" not in verdict
    assert "regresses beyond tolerance" not in verdict


def test_a_healthy_panel_is_unaffected():
    """The guard must not swallow a real result: with no failures the report still
    publishes percentages and reaches a verdict."""
    text = _report({"good": [_row() for _ in range(4)]})
    assert "n/a | n/a" not in text
    assert "Not measured" not in text
    assert "NO VERDICT" not in text
    assert "100%" in text


def test_a_healthy_model_is_still_gated_alongside_an_unreachable_one():
    """A broken backend must not take a working model's verdict down with it."""
    text = _report({"good": [_row() for _ in range(4)],
                    "dead": [_row(raw_ok=0, terse_ok=0, primer_ok=0, inline_ok=0, fails=4)]})
    verdict = text.split("## Verdict")[1]
    assert "NO VERDICT" not in verdict, "one dead model must not void a measured one"
    assert "`dead`" in verdict and "not measured" in verdict
    assert "100%" in text
