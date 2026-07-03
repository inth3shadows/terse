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
from typing import Any

from .report import _GAP_TOLERANCE, _ci, _sum, diff_gap_rows, dropeval_gap_rows, fluency_gap_rows

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


def stacked_bar_lines(items: list[tuple[str, list[float]]], series_labels: tuple[str, ...],
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
        f"{_c(sgr, _BLOCK, enabled)} {name}" for sgr, name in zip(series_sgr, series_labels)
    )
    lines = [legend]
    for label, vals in items:
        denom = sum(abs(v) for v in vals) or 1
        segs, used = [], 0
        for sgr, v in zip(series_sgr, vals):
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
    tools = sorted({r.get("tool", "?") for r in rows})

    shape_items = []
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        raw, cmp_ = _sum(sub, "cl100k", "raw"), _sum(sub, "cl100k", "compressed")
        shape_items.append((shape, ((raw - cmp_) / raw * 100) if raw else 0.0))

    tool_items = []
    for tool in tools:
        sub = [r for r in rows if r.get("tool") == tool]
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
    pcts = [r.get("saved_pct") for r in runs if r.get("saved_pct") is not None]
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
    plot of per-model accuracy with 95% CI, gated on the worst model."""
    gap_rows = diff_gap_rows(results)
    plot_rows = []
    for model, (facc, fse, cacc, cse) in gap_rows.items():
        gap = facc - cacc
        passed = gap >= -_GAP_TOLERANCE - 1e-9
        plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                           "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
    return forest_bar_lines(plot_rows, form_label, control_label, color=color)


def build_terminal_fluency_report(results: dict, color: bool | None = None) -> str:
    """Terminal counterpart to report.build_fluency_report's verdict section — a forest
    plot of best-terse-form vs raw accuracy per model, gated on the worst model. Models
    whose raw control failed (backend/config error) are excluded, same as the markdown."""
    gap_rows, broken = fluency_gap_rows(results)
    plot_rows = []
    for model, (facc, fse, cacc, cse) in gap_rows.items():
        gap = facc - cacc
        passed = gap >= -_GAP_TOLERANCE - 1e-9
        plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                           "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
    text = forest_bar_lines(plot_rows, "best terse-form", "raw", color=color)
    if broken:
        text += f"\n  (excluded — raw control failed: {', '.join(broken)})"
    return text


_DROPEVAL_METRICS = (("recall", "retrieve-recall"), ("precision", "no-overfetch"),
                     ("accuracy", "final-accuracy"))


def build_terminal_dropeval_report(results: dict, color: bool | None = None) -> str:
    """Terminal counterpart to report.build_dropeval_report's verdict section — three
    forest plots (retrieve-recall, no-overfetch, final-accuracy), each vs a fixed
    100%-ideal control, gated on the worst model per metric. Fed by report.py's
    dropeval_gap_rows so the gap a chart shows can never diverge from the markdown."""
    gaps = dropeval_gap_rows(results)
    if not gaps:
        return "  (no data)"
    sections = []
    for key, label in _DROPEVAL_METRICS:
        plot_rows = []
        for model, metrics in gaps.items():
            facc, fse, cacc, cse = metrics[key]
            gap = facc - cacc
            passed = gap >= -_GAP_TOLERANCE - 1e-9
            plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                               "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})
        sections.append(f"{label}:")
        sections.append(forest_bar_lines(plot_rows, label, "ideal (100%)", color=color))
    return "\n\n".join(sections)
