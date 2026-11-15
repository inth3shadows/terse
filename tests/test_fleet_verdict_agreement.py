"""Review of #589: every renderer reaches the same verdict from the same data.

The markdown, the HTML page and the terminal chart used to decide the fleet verdict
separately, and they disagreed: a soak whose deepest slice was withheld (or too wide)
read "Not concluded" in the markdown and a green PASS on the HTML banner; the terminal
diff and fluency charts printed per-model PASS badges with no fleet gate at all, so a
withheld model never blocked PASS there. These pin the agreement, plus the review's
dropeval, one-row-arm, never-reached-depth and clipped-badge findings."""
from __future__ import annotations

import re

from terse.html_report import build_html_diff_report, forest_plot
from terse.report import (
    build_diff_report,
    build_diff_soak_report,
    build_dropeval_report,
    build_fluency_report,
    dropeval_directive_line,
    dropeval_verdict,
    strip_markup,
)
from terse.terminal_report import (
    build_terminal_diff_report,
    build_terminal_fluency_report,
)


def _q(i, form, control, trials=1):
    return {"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
            "trials": trials, "diff_ok": form, "terse_ok": control}


def _soak_slice(depth, n=20, misses=0):
    return [{**_q(f"{depth}-{i}", 0 if i < misses else 1, 1), "depth": depth, "fails": 0,
             "attempts": 1, "diff_trials": 1, "terse_trials": 1} for i in range(n)]


def _banner(html: str) -> str:
    m = re.search(r'<div class="banner ([a-z]+)">(.*?)</div>', html, re.S)
    assert m, "no verdict banner"
    return m.group(1) + " " + m.group(2)


def _md_verdict(md: str) -> str:
    return md.split("## Verdict", 1)[1]


# --- B1: the soak's HTML banner and terminal chart agree with its markdown -------------

def _deepest_withheld_soak():
    # The PR's own fixture: `bad`'s depth-5 slice is dead, its pooled loss is small enough
    # to stay in the pooled gate, so only the deepest verdict is missing a model.
    dead = [{**_q("5-0", 0, 0), "depth": 5, "fails": 1, "attempts": 1,
             "diff_trials": 0, "terse_trials": 0}]
    bad = [r for d in range(1, 5) for r in _soak_slice(d)] + dead
    return {"good": [r for d in range(1, 6) for r in _soak_slice(d)], "bad": bad}


def _deepest_too_wide_soak():
    # Pooled 99/100 vs 100/100 clears -5% interval included; the depth-5 slice alone is
    # 19/20 vs 20/20, whose interval (about ±10 pts) reaches past it.
    return {"m": [r for d in range(1, 6) for r in _soak_slice(d, misses=1 if d == 5 else 0)]}


def test_soak_html_banner_is_not_pass_when_the_deepest_slice_is_withheld():
    results = _deepest_withheld_soak()
    md = _md_verdict(build_diff_soak_report(results))
    assert "Not concluded" in md and "No depth-correlated" not in md
    banner = _banner(build_html_diff_report(results, "chain-form"))
    assert "PASS" not in banner and "INCONCLUSIVE" in banner, banner


def test_soak_html_banner_is_not_pass_when_the_deepest_interval_is_too_wide():
    results = _deepest_too_wide_soak()
    md = _md_verdict(build_diff_soak_report(results))
    assert "**INCONCLUSIVE**" in md and "No depth-correlated" not in md
    banner = _banner(build_html_diff_report(results, "chain-form"))
    assert "PASS" not in banner and "INCONCLUSIVE" in banner, banner


def test_soak_terminal_chart_carries_the_same_verdict_as_the_markdown():
    for results in (_deepest_withheld_soak(), _deepest_too_wide_soak()):
        md = _md_verdict(build_diff_soak_report(results))
        text = build_terminal_diff_report(results, form_label="chain-form", color=False)
        # The markdown's verdict lines, markup stripped, appear verbatim in the chart.
        for line in (ln for ln in md.splitlines() if ln.startswith("- ")):
            assert strip_markup(line) in text, (line, text)


# --- B4: the terminal diff and fluency charts gate the fleet -------------------------

def _dead_diff_rows():
    return [{**_q(i, 0, 0), "diff_trials": 0, "terse_trials": 0, "fails": 1, "attempts": 1}
            for i in range(20)]


def _clean_diff_rows():
    return [{**_q(i, 1, 1), "diff_trials": 1, "terse_trials": 1, "fails": 0, "attempts": 1}
            for i in range(20)]


def test_terminal_diff_chart_states_the_fleet_verdict_a_withheld_model_blocks():
    results = {"good": _clean_diff_rows(), "dead": _dead_diff_rows()}
    assert "**INCONCLUSIVE**" in build_diff_report(results)
    text = build_terminal_diff_report(results, color=False)
    verdict = [ln for ln in text.splitlines() if "Worst-case model" in ln]
    assert verdict and "INCONCLUSIVE" in verdict[0], text
    assert "safe to enable" not in text


def _prow(**kw):
    base = {"tool": "t", "sha": "s", "qid": "q", "qtype": "count", "transform": "table",
            "trials": 1, "raw_ok": 1, "terse_ok": 1, "primer_ok": 1,
            "raw_trials": 1, "terse_trials": 1, "primer_trials": 1, "fails": 0, "attempts": 3}
    return {**base, **kw}


def test_terminal_fluency_chart_states_the_fleet_verdict_a_withheld_model_blocks():
    dead = [_prow(qid=f"q{i}", raw_ok=0, terse_ok=0, primer_ok=0, raw_trials=0,
                  terse_trials=0, primer_trials=0, fails=3) for i in range(20)]
    results = {"good": [_prow(qid=f"q{i}") for i in range(20)], "dead": dead}
    assert "**INCONCLUSIVE**" in build_fluency_report(results, [])
    text = build_terminal_fluency_report(results, color=False)
    verdict = [ln for ln in text.splitlines() if "Worst-case model" in ln]
    assert verdict and "INCONCLUSIVE" in verdict[0], text


# --- B5: an arm scored on one row cannot publish `±0 PASS` -------------------------

def _one_row_control():
    # Legacy rows (no `attempts`), diff perfect on all 20, the control scored on row 0.
    return {"m": [{**_q(i, 1, 1 if i == 0 else 0), "diff_trials": 1,
                   "terse_trials": 1 if i == 0 else 0} for i in range(20)]}


def test_a_control_scored_on_one_row_cannot_pass():
    results = _one_row_control()
    md = build_diff_report(results)
    assert "**PASS**" not in md and "safe to enable" not in md, _md_verdict(md)
    assert "✓ PASS" not in build_html_diff_report(results)
    assert "PASS" not in build_terminal_diff_report(results, color=False)


# --- B6: a model that never reached the deepest depth blocks the deepest verdict -----

def test_a_model_that_never_reached_the_deepest_depth_is_named_and_blocks_pass():
    results = {"deep": [r for d in range(1, 6) for r in _soak_slice(d)],
               "short": [r for d in range(1, 5) for r in _soak_slice(d)]}
    md = _md_verdict(build_diff_soak_report(results))
    assert "No depth-correlated comprehension drift" not in md
    deepest = md.split("deepest tested depth", 1)[1].split("\n", 1)[0]
    assert "**PASS**" not in deepest and "`short`" in deepest, deepest
    assert "PASS" not in _banner(build_html_diff_report(results, "chain-form"))


# --- B2/B3: dropeval's per-metric lines and directive prose -------------------------

def _drows(n, *, kind, answer=1, retrieve=1, control=1):
    return [{"kind": kind, "trials": 1, "retrieve_ok": retrieve, "answer_ok": answer,
             "handle_ok": 1, "errors": 0, "treatment_errors": 0, "control_errors": 0,
             "attempts": 2, "qid": f"{kind}-q{i}", "control_ok": control,
             "control_trials": 1} for i in range(n)]


def _metric_line(md: str, label: str) -> str:
    lines = [ln for ln in _md_verdict(md).splitlines()
             if ln.startswith("- Worst-case model") and f": {label} " in ln]
    assert len(lines) == 1, lines
    return lines[0]


def test_dropeval_metric_line_is_not_pass_when_a_model_was_withheld_from_it():
    # `b` has no precision rows, so no-overfetch is withheld for it; `a` passes it.
    results = {"a": _drows(40, kind="recall") + _drows(40, kind="precision"),
               "b": _drows(40, kind="recall")}
    v = dropeval_verdict(results)
    assert v.metrics["precision"].excluded == {"b": "empty"}
    line = _metric_line(build_dropeval_report(results), "no-overfetch")
    assert "**PASS**" not in line and "**INCONCLUSIVE**" in line, line
    assert "Measured models only" in line


def test_dropeval_metric_line_is_not_pass_when_a_model_is_thin_on_it():
    results = {"a": _drows(40, kind="recall") + _drows(3, kind="precision")}
    assert dropeval_verdict(results).metrics["precision"].thin == {"a": 3}
    line = _metric_line(build_dropeval_report(results), "no-overfetch")
    assert "**PASS**" not in line, line


def _full_fleet(**miss):
    rows = _drows(40, kind="recall") + _drows(40, kind="precision")
    for key, idx in miss.items():
        for i in idx:
            rows[i][key] = 0
    return {"m": rows}


def test_a_measured_recall_whose_interval_is_too_wide_is_not_called_ungated():
    v = dropeval_verdict(_full_fleet(retrieve_ok=[0]))   # recall 39/40
    assert v.metrics["recall"].worst.inconclusive and not v.metrics["recall"].excluded
    line = dropeval_directive_line(v)
    assert "INSUFFICIENT" in line
    assert "not gated" not in line and "was not gated" not in line, line
    assert "Each model above is named" not in line, line
    assert "interval" in line and "more questions" in line, line


def test_a_measured_accuracy_whose_interval_is_too_wide_is_not_called_unmeasured():
    v = dropeval_verdict(_full_fleet(answer_ok=[0, 1]))   # 78/80 vs 80/80
    assert v.metrics["accuracy"].worst.inconclusive and not v.metrics["accuracy"].excluded
    line = dropeval_directive_line(v)
    assert "unmeasured" not in line and "rest of the fleet" not in line, line
    assert "interval" in line and "more questions" in line, line


# --- B7: the HTML INCONCLUSIVE badge fits inside the forest plot --------------------

def test_the_forest_plot_badge_is_not_clipped():
    svg = forest_plot([{"model": "m", "form_acc": 0.95, "form_ci": 0.1, "control_acc": 1.0,
                        "control_ci": 0.0, "passed": False, "verdict": "INCONCLUSIVE"}],
                      "diff-form", "full-terse")
    width = float(re.search(r'viewBox="0 0 ([\d.]+) ', svg).group(1))
    x = float(re.search(r'<text x="([\d.]+)"[^>]*>INCONCLUSIVE</text>', svg).group(1))
    # 11px caps run about 0.72em a glyph in the page's sans stack; generous on purpose.
    assert x + len("INCONCLUSIVE") * 11 * 0.72 <= width, (x, width)
