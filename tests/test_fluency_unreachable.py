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
from terse.fluency.scoring import score
from terse.report import (
    UNMEASURED_FAIL_SHARE,
    _form_stats,
    _unmeasured,
    build_fluency_report,
)

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
    # The `count` case above pins the COUNTING half only: `""` scores 0 against an
    # expected `6` anyway, so it stays green even if a failure is scored instead of
    # skipped. This is the case that pins "never scores them" — `score("lookup", "", "")`
    # is True, so a failure fed to `score` as `""` would count as a CORRECT answer. That
    # is the secondary hazard #263 names, and only this assertion holds it.
    assert score("lookup", "", "") is True, "the premise this test depends on"
    assert _ask_n(_dead, "", "q", "lookup", "", 3) == (0, 3), \
        "a transport failure must not be scored, least of all as correct"


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
            "raw_trials": 1, "terse_trials": 1, "primer_trials": 1, "inline_trials": 1,
            "fails": 0, "attempts": 4}
    return {**base, **kw}


def test_an_unreachable_model_publishes_no_accuracy_at_all():
    """Not a footnoted 0% — no number. A percentage beside a model name is read as
    comprehension however it is annotated."""
    text = _report({"dead-model": [_row(raw_ok=0, terse_ok=0, primer_ok=0,
                                        inline_ok=0, fails=4)]})
    assert "| `dead-model` | 1 | n/a | n/a | n/a | n/a | n/a | n/a |" in text
    assert "0%" not in text.split("## Verdict")[0].split("| `dead-model`")[1][:60]
    assert "Not measured" in text and "4/4 calls lost" in text


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


# --- review findings against the first cut of this fix (#264) ---

def test_every_harness_returns_ints_not_the_raw_ask_n_tuple():
    """`_ask_n` returns `(ok, fails)`. The first cut updated two diff call sites by exact
    string match and MISSED `run_chain_payload`, whose second prompt variable is `chain_u`
    rather than `diff_u`. Rows then carried `terse_ok: (1, 0)` and every soak report died
    in `_form_stats` (`int()` on a tuple) AFTER paying for all the model calls.

    Asserted across all three harnesses by type, so a fourth harness added later cannot
    reintroduce it by using yet another variable name."""
    from terse.fluency.harnesses import (
        run_chain_payload,
        run_diff_payload,
        run_text_diff_payload,
    )
    ok = lambda s, u: "2"                                            # noqa: E731
    a = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    b = [{"id": 1, "x": "a"}, {"id": 2, "x": "c"}]
    batches = [run_diff_payload(a, b, ok, "t", 2),
               run_chain_payload([a, b, a], ok, "t", 2),
               run_text_diff_payload("l1\nl2\nl3\n", "l1\nl2x\nl3\n", ok, "t", 2)]
    assert any(batches), "no harness produced rows — this would pin nothing"
    for rows in batches:
        for r in rows:
            for key in ("terse_ok", "diff_ok"):
                assert isinstance(r[key], int), f"{key} is {type(r[key]).__name__}, not int"


def test_a_soak_report_renders_from_real_harness_rows():
    """The crash reproduced end-to-end: the soak report over rows the harness actually
    produced, not rows the test hand-built (which is why the existing soak tests missed
    it — they construct their own)."""
    from terse.fluency.harnesses import run_chain_payload
    from terse.report import build_diff_soak_report
    a = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    b = [{"id": 1, "x": "a"}, {"id": 2, "x": "c"}]
    rows = run_chain_payload([a, b, a], lambda s, u: "2", "t", 1)
    assert rows
    assert isinstance(build_diff_soak_report({"m": rows}), str)      # must not raise


def test_the_terminal_chart_excludes_the_same_models_the_markdown_does():
    """`cli` prints the markdown report and the terminal forest plot together. The first
    cut gated only the markdown, so a dead backend rendered `n/a` in the table and
    'not measured' in the verdict while the chart below plotted its gap as a red FAIL bar
    — the same false verdict, in the renderer a reader looks at first."""
    from terse.report import fluency_gap_rows
    # PARTIALLY failing: raw is non-zero, so the pre-existing `raw == 0` guard does NOT
    # fire and only the failure-share check can exclude it. A fully-dead model would pass
    # this test against either implementation and so would pin nothing.
    flaky = [_row(raw_ok=1, terse_ok=0, primer_ok=0, inline_ok=0, fails=3, attempts=4,
                  raw_trials=1, terse_trials=0, primer_trials=0, inline_trials=0)
             for _ in range(4)]
    gap_rows, excluded = fluency_gap_rows({"good": [_row() for _ in range(4)],
                                           "flaky": flaky})
    assert _form_stats(flaky, "raw_ok")[0] > 0, "must not be caught by the raw==0 guard"
    assert "flaky" not in gap_rows, "the chart would plot a FAIL bar for a dead backend"
    assert "flaky" in excluded
    assert "good" in gap_rows


def test_the_per_transform_table_drops_unmeasured_models():
    """The report refuses to publish a dead model's per-model numbers, then used to pool
    the same counts into the per-transform table unannotated — and that table is what a
    reader uses to decide 'restrict the policy to the transforms that held'."""
    dead = [_row(transform="table", raw_ok=0, terse_ok=0, primer_ok=0, inline_ok=0,
                 fails=4, attempts=4, raw_trials=0, terse_trials=0,
                 primer_trials=0, inline_trials=0) for _ in range(8)]
    good = [_row(transform="table") for _ in range(2)]
    text = _report({"good": good, "dead": dead})
    section = text.split("by stressed transform")[1].split("## Verdict")[0]
    # Pooling 8 zero rows with 2 perfect ones would read 20%; excluding them reads 100%.
    assert "| table | 2 | 100% | 100% |" in section


def test_a_few_transient_failures_do_not_void_an_otherwise_complete_run():
    """The first cut voided a model on ANY failure. With ~5 models x hundreds of calls,
    one transient 429 each would discard a multi-hour run — the same outcome, by a
    different route, as the bug being fixed.

    A failed call is now removed from its arm's DENOMINATOR instead, so the surviving
    sample stays honest and only a substantially-down backend is withheld."""
    # 20 rows, 4 arms x 1 trial = 80 attempts; 2 lost = 2.5%, well under the bar.
    rows = [_row(attempts=4, raw_trials=1, terse_trials=1, primer_trials=1, inline_trials=1)
            for _ in range(20)]
    rows[0].update(fails=2, terse_trials=0, terse_ok=0)   # one question's terse arm lost
    text = _report({"flaky": rows})
    assert "NO VERDICT" not in text
    assert "Partially degraded" in text and "2/80" in text
    # terse is still 100%: the 19 completed rows are the denominator, not 20.
    assert "| `flaky` | 20 | 100% ±0 | 100% ±0" in text


def test_a_mostly_dead_backend_is_still_withheld():
    """The threshold's other side — the case that must stay caught."""
    rows = [_row(raw_ok=0, terse_ok=0, primer_ok=0, inline_ok=0, fails=4, attempts=4,
                 raw_trials=0, terse_trials=0, primer_trials=0, inline_trials=0)
            for _ in range(20)]
    text = _report({"dead": rows})
    assert "80/80 calls lost" in text
    assert "NO VERDICT" in text


def test_a_results_set_with_no_rows_never_reaches_the_verdict_at_all():
    """Review raised that an empty `summary` would print "Every model was excluded above"
    with nothing in the exclusion lists. That state turns out to be UNREACHABLE: `summary`
    can only end up empty when every model's row list is empty, and `build_fluency_report`
    early-returns "No model answers provided" before the verdict in exactly that case.

    Pinned so the claim is checked rather than argued. The report's cause-specific wording
    is kept as a defensive guard, not because it fixes a live path — if a future refactor
    removes this early return, that guard becomes load-bearing and this test tells whoever
    does it what they just changed."""
    text = _report({"m": []})
    assert "## Verdict" not in text
    assert "No model answers provided" in text


def test_run_payload_excludes_failed_calls_from_the_arm_denominator():
    """Pins the DENOMINATOR at its source. The report-level test builds rows by hand, so it
    cannot tell whether `run_payload` still emits the per-arm `<form>_trials` keys that
    `_form_stats` divides by — remove them from the harness and that test stays green while
    every real run silently starts scoring failures as misses again."""
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] % 4 == 2:          # the terse arm of each question
            raise ConnectionError("429")
        return "6"

    rows = run_payload(PAYLOAD, "[]", flaky, primer="P", trials=1)
    assert rows
    for r in rows:
        assert r["raw_trials"] == 1
        assert r["terse_trials"] == 0, "a failed call must leave its arm's denominator"
        assert r["fails"] >= 1 and r["attempts"] == 4
    # And the consequence: terse is not scored 0% over a denominator of 1.
    acc, _ = _form_stats(rows, "terse_ok")
    assert acc == 0.0        # no completed trials -> 0/0 -> 0.0, flagged unmeasured
    assert _unmeasured(rows) is True


def test_a_dead_backend_cannot_produce_a_PASS_on_the_diff_ship_gate():
    """`_build_diff_style_report` had NO control of any kind — not even the `raw == 0` one
    the payload report already had. A backend that was entirely down scored 0% on BOTH
    arms, so the gap was exactly 0 and the verdict read "safe to enable `proxy --diff`".

    That is worse than the defect #263 was filed about. A false FAIL gets re-run because
    it blocks someone; a false PASS on a ship gate agrees with whoever ran it and is never
    checked again."""
    from terse.fluency.harnesses import run_diff_payload
    from terse.report import build_diff_report

    def dead(system, user):
        raise ConnectionError("503")

    a = [{"id": i, "x": "a"} for i in range(4)]
    b = [{"id": i, "x": "b"} for i in range(4)]
    rows = run_diff_payload(a, b, dead, "t", 2)
    assert rows, "no rows generated — this would pin nothing"
    text = build_diff_report({"gemini-dead": rows})
    verdict = text.split("## Verdict")[1]
    assert "safe to enable" not in verdict, "a dead backend must never green-light --diff"
    assert "NO VERDICT — nothing was measured" in verdict
    assert "16/16 calls lost" in text
    assert "| `gemini-dead` | 4 | n/a | n/a | n/a |" in text


def test_the_diff_harnesses_emit_the_attempts_counter_the_gate_divides_by():
    """`_unmeasured` returns False when `attempts` is absent — deliberately, so result
    files predating the counters still render. That default meant adding the gate to the
    diff report did nothing until the diff harnesses also emitted `attempts`: the report
    read every live row as pre-counter and published a verdict anyway."""
    from terse.fluency.harnesses import (
        run_chain_payload,
        run_diff_payload,
        run_text_diff_payload,
    )
    ok = lambda s, u: "2"                                            # noqa: E731
    a = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    b = [{"id": 1, "x": "a"}, {"id": 2, "x": "c"}]
    for rows in (run_diff_payload(a, b, ok, "t", 2),
                 run_chain_payload([a, b, a], ok, "t", 2),
                 run_text_diff_payload("l1\nl2\nl3\n", "l1\nl2x\nl3\n", ok, "t", 2)):
        assert rows
        for r in rows:
            assert r["attempts"] == 4, "2 arms x 2 trials"


def test_the_diff_forest_plot_excludes_the_same_models_the_diff_markdown_does():
    """`diff_gap_rows` promises in its own docstring that "a chart's gap can never read
    differently than build_diff_report's". Adding the transport-failure gate to the
    markdown alone BROKE that promise: the plot `cli` prints directly beneath it still
    drew a FAIL bar for a model the markdown had just refused to score.

    One function feeds the chart for all three diff paths (`--diff`, `--diff-soak`,
    `--text-diff-eval`), so this covers each of them."""
    from terse.report import diff_gap_rows
    from terse.terminal_report import build_terminal_diff_report
    dead = [{"qid": f"q{i}", "qtype": "count", "transform": "table", "trials": 2,
             "terse_ok": 0, "terse_trials": 0, "diff_ok": 0, "diff_trials": 0,
             "fails": 4, "attempts": 4} for i in range(4)]
    good = [{"qid": f"q{i}", "qtype": "count", "transform": "table", "trials": 2,
             "terse_ok": 2, "terse_trials": 2, "diff_ok": 2, "diff_trials": 2,
             "fails": 0, "attempts": 4} for i in range(4)]
    gap_rows, excluded = diff_gap_rows({"good": good, "dead": dead})
    assert "dead" not in gap_rows and "dead" in excluded
    assert "good" in gap_rows
    chart = build_terminal_diff_report({"good": good, "dead": dead}, color=False)
    assert "FAIL" not in chart, "a dead backend must not render a FAIL bar"
    assert "calls never reached the backend: dead" in chart


def test_the_soak_report_cannot_pass_off_a_dead_backend():
    """`build_diff_soak_report` had no control either. A backend that is down scores 0%
    on both arms at every depth — a gap of exactly 0, which reads PASS, under a by-depth
    table showing a flat reassuring no-drift line drawn from calls that never happened."""
    from terse.fluency.harnesses import run_chain_payload
    from terse.report import build_diff_soak_report

    def dead(system, user):
        raise ConnectionError("503")

    a = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    b = [{"id": 1, "x": "a"}, {"id": 2, "x": "c"}]
    rows = run_chain_payload([a, b, a], dead, "t", 2)
    assert rows, "no rows generated — this would pin nothing"
    text = build_diff_soak_report({"gemini-dead": rows})
    verdict = text.split("## Verdict")[1]
    assert "PASS" not in verdict, "a dead backend must never pass the soak gate"
    assert "NO VERDICT — nothing was measured" in verdict
    assert "| n/a | n/a | n/a |" in text
    assert "calls lost" in text


def test_the_html_report_cannot_render_a_green_PASS_banner_for_a_dead_backend():
    """`build_html_diff_report` builds its OWN gap rows rather than calling
    `diff_gap_rows`, so gating the markdown and the terminal chart left it untouched: a
    fully-down backend still rendered `<div class="banner good">✓ PASS`.

    This is the artifact people screenshot and paste into an issue. It is the last place
    a false pass should survive, not the first place to forget."""
    from terse.fluency.harnesses import run_diff_payload
    from terse.html_report import build_html_diff_report

    def dead(system, user):
        raise ConnectionError("503")

    a = [{"id": i, "x": "a"} for i in range(4)]
    b = [{"id": i, "x": "b"} for i in range(4)]
    rows = run_diff_payload(a, b, dead, "t", 2)
    assert rows, "no rows generated — this would pin nothing"
    html = build_html_diff_report({"gemini-dead": rows})
    assert "banner good" not in html, "a dead backend must never render a green PASS"
    assert "✓ PASS" not in html
    assert "NO VERDICT — nothing was measured" in html
    assert "gemini-dead" in html and "Not measured" in html


def test_unmeasured_discovers_arms_instead_of_hardcoding_the_payload_ones():
    """The arm list was `("raw", "terse", "primer", "inline")` — the payload harness's
    arms. The diff harnesses emit `terse`/`diff`, so a fully-lost `diff` arm was invisible
    to the zero-completed-trials trigger.

    The row shape here is SYNTHETIC (six arms), and deliberately so. Written against a
    real 2-arm diff row this test passes with the hardcoded list still in place — a fully
    lost arm out of two is 50% of the calls, so the 20% share trigger catches it and the
    arm trigger never has to fire. Verified by mutation: the obvious version of this test
    did not fail against the old code.

    So the arm trigger is genuinely redundant at today's arm counts, and this pins the
    only case where it is not: enough arms that one can vanish entirely while staying
    under the share bar. At six arms a lost arm is 16.7%, and only arm discovery catches
    it. That case is reachable the moment an arm is added or the threshold is raised,
    which is exactly when nobody will re-derive this."""
    wide = [{f"{a}_ok": 0 if a == "f" else 2, f"{a}_trials": 0 if a == "f" else 2}
            for a in ("a", "b", "c", "d", "e", "f")]
    row = {"trials": 2, "transform": "table", "fails": 2, "attempts": 12}
    for arm in wide:
        row.update(arm)
    assert row["fails"] / row["attempts"] < UNMEASURED_FAIL_SHARE, \
        "premise: the share trigger must NOT fire, or this pins the wrong mechanism"
    assert _unmeasured([row]) is True, "a fully-lost arm must be caught by name discovery"
