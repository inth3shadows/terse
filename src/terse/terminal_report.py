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

from .report import _sum

_BAR_WIDTH = 24
_BLOCK = "█"
_NEG_BLOCK = "▒"  # distinct glyph so a negative segment reads as an anomaly even without color


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
