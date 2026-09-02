"""Terminal bar-chart companion to report.py's markdown — same "is the win real and
stable" glance test as html_report.py's SVG charts, but zero new artifact: prints
straight to the terminal the moment `measure`/`verify` runs (issue #51 fast-follow;
the SVG half shipped in the PR that closed #51 as `--html`).

ANSI color is used only when the terminal supports it (isatty + NO_COLOR unset); the
bar glyphs themselves are plain unicode block characters, so piped/redirected output
(CI logs, `| tee`) still carries the shape of the win, just uncolored. Reuses
report.py's `_sum`/`_pct` so the numbers here can never diverge from the markdown.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from .report import (
    _FIXED_IDEAL_MIN_QUESTIONS,
    ATTRITION_NOTE,
    DIFF_ARMS,
    DROPEVAL_METRICS,
    FLUENCY_CONTROL,
    FLUENCY_GATING,
    _ci,
    _sum,
    attrition,
    attrition_block,
    diff_gap_rows,
    dropeval_attrition_note,
    dropeval_directive_line,
    dropeval_verdict,
    exclusion_note,
    fluency_gap_rows,
    is_diff_run,
    passes_tolerance,
    strip_markup,
)


def _plain(markdown: str) -> str:
    """`**bold**` and `` `code` `` stripped — the terminal's copy of a shared sentence.

    Delegates to `report.strip_markup` so two terminal renderers cannot strip different
    markup off the same shared string."""
    return strip_markup(markdown)


_BAR_WIDTH = 24
_BLOCK = "█"
_NEG_BLOCK = "▒"  # distinct glyph so a negative segment reads as an anomaly even without color
_TRACK_WIDTH = 32


def _color_enabled(stream: Any = None) -> bool:
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _c(sgr: str, text: str, color: bool) -> str:
    return f"\x1b[{sgr}m{text}\x1b[0m" if color else text


def diverging_bar_lines(items: list[tuple[str, float]], unit: str = "%",
                         color: bool | None = None) -> str:
    """One row per item: label, a fixed-width bar sized to |value| / max|value|,
    green for positive, red for negative. `items`: (label, signed value)."""
    if not items:
        return "  (no data)"
    enabled = _color_enabled() if color is None else color
    label_w = min(max((len(label) for label, _ in items), default=0), 28)
    vmax = max((abs(v) for _, v in items), default=0) or 1
    lines = []
    for label, value in items:
        n = round(min(abs(value), vmax) / vmax * _BAR_WIDTH)
        bar = _BLOCK * n + " " * (_BAR_WIDTH - n)  # pad BEFORE coloring — ANSI codes must
        sgr = "32" if value >= 0 else "31"          # never sit inside a width-formatted field
        lines.append(f"  {label[:label_w]:<{label_w}} {_c(sgr, bar, enabled)} {value:+.1f}{unit}")
    return "\n".join(lines)


def stacked_bar_lines(items: Sequence[tuple[str, Sequence[float]]], series_labels: tuple[str, ...],
                       series_sgr: tuple[str, ...] = ("34", "32", "33"),
                       color: bool | None = None) -> str:
    """One row per item: proportional multi-color bar across series_labels, sized by
    each series' share of the row's total ABSOLUTE magnitude (so a negative series
    still claims visible width instead of vanishing). Negative segments render with
    `_NEG_BLOCK` instead of `_BLOCK` so the anomaly reads even without color, and the
    exact signed total always follows the bar — the bar is a glance aid, the number
    is the truth (measure.py: "a tier can go negative at a small sample size")."""
    if not items:
        return "  (no data)"
    enabled = _color_enabled() if color is None else color
    label_w = min(max((len(label) for label, _ in items), default=0), 28)
    legend = "  " + "  ".join(
        f"{_c(sgr, _BLOCK, enabled)} {name}" for sgr, name in zip(series_sgr, series_labels, strict=True)
    )
    lines = [legend]
    for label, vals in items:
        denom = sum(abs(v) for v in vals) or 1
        segs, used = [], 0
        for sgr, v in zip(series_sgr, vals, strict=True):
            n = round(abs(v) / denom * _BAR_WIDTH)
            used += n
            glyph = _BLOCK if v >= 0 else _NEG_BLOCK
            segs.append(_c(sgr, glyph * n, enabled))
        bar = "".join(segs) + " " * max(_BAR_WIDTH - used, 0)
        total = sum(vals)
        lines.append(f"  {label[:label_w]:<{label_w}} {bar} {total:+.0f}")
    return "\n".join(lines)


def build_terminal_report(rows: list[dict[str, Any]], color: bool | None = None) -> str:
    """Terminal counterpart to html_report.build_html_report's two savings charts plus
    tier attribution — the three sections the markdown tables make hard to compare
    at a glance. Gate/coverage stay markdown-only (already glance-readable as text)."""
    shapes = sorted({r["shape"] for r in rows})
    # `or "?"` on BOTH lines, and the same expression on each. They used to differ
    # — `.get("tool", "?")` here, a bare `.get("tool")` in the filter below — which
    # agreed on every input except a row whose `tool` key is ABSENT: the set
    # substituted "?" and the filter compared None to it, so that row matched
    # nothing and its tokens left this table while still counting in every other
    # total on the page. A row with an explicit `"tool": None` was worse: the key
    # exists, so `None` entered the set and `sorted()` raised TypeError on the
    # whole report. `or` normalises absent, None and "" alike, once, in one place.
    tools = sorted({r.get("tool") or "?" for r in rows})

    shape_items = []
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        raw, cmp_ = _sum(sub, "cl100k", "raw"), _sum(sub, "cl100k", "compressed")
        shape_items.append((shape, ((raw - cmp_) / raw * 100) if raw else 0.0))

    tool_items = []
    for tool in tools:
        sub = [r for r in rows if (r.get("tool") or "?") == tool]
        raw, cmp_ = _sum(sub, "cl100k", "raw"), _sum(sub, "cl100k", "compressed")
        tool_items.append((tool, ((raw - cmp_) / raw * 100) if raw else 0.0))
    tool_items.sort(key=lambda kv: -kv[1])

    tier_items = []
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        m = _sum(sub, "saved_cl100k", "minify")
        t = _sum(sub, "saved_cl100k", "tabularize")
        d = _sum(sub, "saved_cl100k", "dictionary")
        tier_items.append((shape, [m, t, d]))

    out = [
        "Tier-0 savings by shape bucket",
        diverging_bar_lines(shape_items, color=color),
        "",
        "Tier-0 savings by tool",
        diverging_bar_lines(tool_items, color=color),
        "",
        "Tier attribution by shape (minify / tabularize / dictionary)",
        stacked_bar_lines(tier_items, ("minify", "tabularize", "dictionary"), color=color),
    ]
    return "\n".join(out)


_SPARK_LEVELS = "▁▂▃▄▅▆▇█"


def trend_sparkline_lines(runs: list[dict[str, Any]]) -> str:
    """One-line sparkline of `measure --history` saved_pct across runs, oldest to
    newest — the fastest possible glance at "is the win stable, improving, or
    regressing" without reading report.build_trend_report's full table. A flat
    reading (all bars level) with real historical data is itself a legitimate,
    useful signal (a stable win), not a sign something's broken."""
    pcts = [float(r["saved_pct"]) for r in runs if r.get("saved_pct") is not None]
    if len(pcts) < 2:
        return "  (need at least two --history runs to show a trend)"
    lo, hi = min(pcts), max(pcts)
    span = (hi - lo) or 1.0
    n_levels = len(_SPARK_LEVELS)
    spark = "".join(
        _SPARK_LEVELS[min(int((p - lo) / span * (n_levels - 1)), n_levels - 1)] for p in pcts
    )
    return f"  {spark}   {pcts[0]:+.1f}% -> {pcts[-1]:+.1f}%  (range {lo:+.1f}% .. {hi:+.1f}%)"


def _track(acc: float, ci: float, marker: str) -> str:
    """Fixed-width `_TRACK_WIDTH`+1 char track: '·' background, '─' whisker span over
    the 95% CI, `marker` at the point estimate. Built and clamped BEFORE any coloring
    is applied — see diverging_bar_lines for why that order matters."""
    lo = max(acc - ci, 0.0)
    hi = min(acc + ci, 1.0)
    lo_col = round(lo * _TRACK_WIDTH)
    hi_col = max(round(hi * _TRACK_WIDTH), lo_col)
    m_col = min(max(round(acc * _TRACK_WIDTH), 0), _TRACK_WIDTH)
    chars = ["·"] * (_TRACK_WIDTH + 1)
    for i in range(lo_col, hi_col + 1):
        if 0 <= i <= _TRACK_WIDTH:
            chars[i] = "─"
    chars[m_col] = marker
    return "".join(chars)


def forest_bar_lines(rows: list[dict[str, Any]], form_label: str, control_label: str,
                      color: bool | None = None) -> str:
    """Two-line-per-model forest plot: a 0%-100% track per series (point + 95% CI
    whisker), plus a pass/fail badge on the form-series line. `rows`: dicts with
    model/form_acc/form_ci/control_acc/control_ci/passed — same shape as
    html_report.forest_plot's input, so the two stay easy to keep in sync."""
    if not rows:
        return "  (no data)"
    enabled = _color_enabled() if color is None else color
    label_w = min(max((len(r["model"]) for r in rows), default=0), 24)
    scale = "0%" + "·" * (_TRACK_WIDTH + 1 - len("0%") - len("100%")) + "100%"
    lines = [f"  {'':<{label_w}}  ○ {control_label}   ● {form_label}   {scale}"]
    for r in rows:
        badge = "PASS" if r["passed"] else "FAIL"
        badge_sgr = "32" if r["passed"] else "31"
        c_track = _c("36", _track(r["control_acc"], r["control_ci"], "○"), enabled)
        f_track = _c("35", _track(r["form_acc"], r["form_ci"], "●"), enabled)
        lines.append(f"  {r['model'][:label_w]:<{label_w}}  {c_track}")
        lines.append(f"  {'':<{label_w}}  {f_track}  {_c(badge_sgr, badge, enabled)}")
    return "\n".join(lines)


def build_terminal_diff_report(results: dict, form_label: str = "diff-form",
                                control_label: str = "full-terse",
                                color: bool | None = None) -> str:
    """Terminal counterpart to report.build_diff_report's verdict section — a forest
    plot of per-model accuracy with 95% CI, gated on the worst model. Models whose calls
    went unanswered are excluded and named, same as the markdown: a FAIL bar for a model
    the markdown just declined to score is the same false verdict in a louder renderer
    (#264). "Unanswered" rather than "never reached the backend" because since #268 it
    also covers a backend that answered with no content at all."""
    gap_rows, excluded = diff_gap_rows(results)
    plot_rows = []
    for model, (facc, fse, cacc, cse) in gap_rows.items():
        gap = facc - cacc
        passed = passes_tolerance(gap)
        plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                           "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
    text = forest_bar_lines(plot_rows, form_label, control_label, color=color)
    # THIS chart is the one an operator sees without `--html`: `cli` prints the markdown
    # and this plot on every diff path and writes the HTML page only under `--html`, so
    # leaving it silent meant the disclosure existed exactly where it was least read
    # (#299). All three diff modes route here.
    attr = attrition_block({m: attrition(rows, *DIFF_ARMS) for m, rows in results.items()
                            if is_diff_run(rows)}, ATTRITION_NOTE)
    # `exclusion_note` rather than a hardcoded phrase: this line said "calls went
    # unanswered" for every exclusion, including a model whose calls were all answered and
    # whose arms simply could not be paired (#280).
    if excluded:
        text += f"\n  ({exclusion_note(excluded)})"
    return text + attr


def build_terminal_fluency_report(results: dict, color: bool | None = None) -> str:
    """Terminal counterpart to report.build_fluency_report's verdict section — a forest
    plot of best-terse-form vs raw accuracy per model, gated on the worst model. Models
    whose raw control failed (backend/config error) are excluded, same as the markdown."""
    gap_rows, broken = fluency_gap_rows(results)
    plot_rows = []
    for model, (facc, fse, cacc, cse) in gap_rows.items():
        gap = facc - cacc
        passed = passes_tolerance(gap)
        plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                           "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
    text = forest_bar_lines(plot_rows, "best terse-form", "raw", color=color)
    if broken:
        text += f"\n  ({exclusion_note(broken)})"
    # The bars are drawn over the PAIRED subset, and a chart that does not say what was
    # removed from it is the same silent exclusion the markdown stopped printing (#299).
    # Same arms as the gap: `fluency_gap_rows`, which feeds the bars above, now reads
    # `FLUENCY_GATING`/`FLUENCY_CONTROL` too, so the chart and its annotation cannot
    # describe different exams. It used to hardcode `["terse_ok", "primer_ok"], "raw_ok"`
    # and this comment was false when written.
    text += attrition_block(
        {m: attrition(rows, *FLUENCY_GATING, FLUENCY_CONTROL)
         for m, rows in results.items() if rows}, ATTRITION_NOTE)
    return text


# The metric names, their labels, and — the half that used to live only here — their
# CONTROL labels now come from `report.DROPEVAL_METRICS`. "vs ideal (100%)" and "vs
# no-drop control" are different claims (#269), and this file used to re-derive which one
# each metric got from a `key == "accuracy"` test written beside a two-column table.


def build_terminal_dropeval_report(results: dict, color: bool | None = None,
                                   accept_degraded: bool = False) -> str:
    """Terminal counterpart to report.build_dropeval_report's verdict section — three
    forest plots (retrieve-recall, no-overfetch, final-accuracy), then the same one-line
    directive the markdown prints.

    Both renderers consume ONE `DropevalVerdict` (#342), so "the chart and the markdown can
    never disagree" is a property of the data flow rather than a promise a test has to
    re-check. It was a promise, and they disagreed three times across #335's review rounds:
    on badge scope, on which models the exclusion note covered, and on whether a
    demonstrated FAIL deserved a thin-sample caveat. Each of those was two functions
    deciding the same thing separately.

    `accept_degraded` must mirror build_dropeval_report's own flag: without it, this used
    to refuse unconditionally on ANY transport-error rate past the INCONCLUSIVE threshold,
    while the markdown — given the same flag by `cli` — rendered a full verdict over the
    surviving questions right below it (review finding 4 on #300)."""
    v = dropeval_verdict(results, accept_degraded=accept_degraded)
    # Computed BEFORE the inconclusive early return. That return fires on the runs with
    # the MOST attrition — a 12/24 failure rate is exactly when a reader needs to know
    # which arm lost the calls — and it used to drop the disclosure on the floor while
    # the markdown printed it. The new test only passed because it set
    # `accept_degraded=True` and so never took this branch.
    attr = attrition_block(
        {m: attrition(rows, "answer_ok", "control_ok", kind_key="kind")
         for m, rows in results.items() if rows}, dropeval_attrition_note(results),
        extra={m: sum(r.get("treatment_errors", 0) for r in rows)
               for m, rows in results.items() if rows}).strip("\n")
    if v.inconclusive:
        # Never draw a forest plot from transport errors: the bars would be indistinguishable
        # from a model that answered and got it wrong. Same refusal build_dropeval_report
        # renders, from the same decision, so chart and markdown cannot disagree.
        return "  " + _plain(dropeval_directive_line(v)) + (f"\n\n{attr}" if attr else "")
    if not v.gates:
        # NOT `+ attr`: `dropeval_verdict` sets `gates[model] = {}` for every model with
        # rows, so `not v.gates` holds only when NO model has rows — and `attr` is built
        # from `... if rows`, so it is "" in exactly that case. The append was a branch no
        # run could reach, which is the thing `attrition_line`'s own docstring refuses to
        # ship. The `v.inconclusive` return above is different: that one is reachable and
        # is the whole point of moving the computation up.
        return "  (no data)"
    sections = []
    if v.degraded_accepted:
        sections.append("  (degraded run accepted --accept-degraded; see the markdown "
                        "report for the per-arm failure split before trusting this)")
    for key, label, control_label in DROPEVAL_METRICS:
        plot_rows = []
        for model, metrics in v.gates.items():
            # `.get`: the gates omit "accuracy" entirely for a withheld model (#269).
            # Drawing it against a 100% that was never measured is the thing that issue
            # exists to stop, and a KeyError here would be the renderer insisting on a
            # number the verdict declined to publish.
            metric = metrics.get(key)
            if metric is None:
                continue
            facc, fse, cacc, cse = metric
            gap = facc - cacc
            passed = passes_tolerance(gap)
            plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                               "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
        # The note comes FIRST, and is emitted even when there is nothing to plot. The
        # `continue` used to sit above it, so a metric from which EVERY model was withheld
        # lost its disclosure along with its empty chart — and `dropeval_directive_line`,
        # shared with the markdown, ends "Each withheld model is named above with the
        # reason it was withheld", which was then false in the terminal and true in the
        # markdown. One sentence consumed by two renderers only removes disagreement if
        # both renderers carry what it refers to.
        excluded = v.metrics[key].excluded
        parts = [exclusion_note(excluded)] if excluded else []
        # Thin samples get named here too, for the same reason the exclusions do: the
        # shared directive sentence ends "Each model above is named with the reason its
        # metric did not conclude", and a referent the chart does not carry makes that
        # sentence false in the chart and true in the markdown.
        thin = v.metrics[key].thin
        if thin:
            parts.append("measured, not concluded — fewer than "
                         f"{_FIXED_IDEAL_MIN_QUESTIONS} questions: "
                         + ", ".join(f"{m} (n={n})" for m, n in sorted(thin.items())))
        note = f"  ({'; '.join(parts)})" if parts else ""
        if not plot_rows:
            if note:
                sections.append(f"{label}:\n{note}")
            continue
        section = f"{label}:\n" + forest_bar_lines(plot_rows, label, control_label, color=color)
        if note:
            section += "\n" + note
        sections.append(section)
    # The directive, from the shared decision. Without it the chart showed bars and let the
    # reader infer the conclusion — which is how a per-model PASS badge came to sit under a
    # fleet verdict that said the opposite.
    #
    # De-emphasised MECHANICALLY rather than rewritten: `**bold**` and backticks are noise
    # in a terminal, but a second hand-written copy of the sentence is how the chart and the
    # markdown came to disagree three times over #335's review rounds. Stripping the markup
    # off the one string keeps them the same sentence by construction, and
    # `test_the_chart_and_the_markdown_reach_the_same_directive` pins the relation.
    sections.append("  " + _plain(dropeval_directive_line(v)))
    if attr:
        sections.append(attr)
    return "\n\n".join(sections)
