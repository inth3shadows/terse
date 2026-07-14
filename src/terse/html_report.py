"""Static, self-contained HTML report — a charted companion to report.py's markdown.

Zero JS, zero CDN, zero new dependency: every chart is inline SVG built from plain
string templates (stdlib only), matching the no-egress / zero-setup guarantee the
markdown reports already carry (see build_verify_header). Hover tooltips are native
SVG <title> elements (no JS needed); the "table view" required for accessibility is a
plain <table> under each chart, collapsed behind a zero-JS <details> disclosure.

Palette, mark specs, and chart-form choices follow the project's data-viz method:
  - savings % is a polarity value (above/below zero) -> diverging blue/red bars
  - tier attribution is part-to-whole across 3 fixed series -> categorical stacked bars
  - fluesncy/diff comprehension is per-model magnitude -> point+whisker (forest) rows
Reuses report.py's stats math (_form_stats / _worst_case_gap) rather than
re-deriving it, so the verdict a reader sees here always matches the markdown report.
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from typing import Any

from .report import _ci, _form_stats, _pct, _sum, _worst_case_gap

# --- palette (dataviz skill reference instance) ---------------------------------

_CSS_VARS = """
:root {
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #1baf7a; --series-3: #eda100;
  --diverging-pos: #2a78d6; --diverging-neg: #e34948; --diverging-mid: #f0efec;
  --good: #0ca30c; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #199e70; --series-3: #c98500;
    --diverging-pos: #3987e5; --diverging-neg: #e66767; --diverging-mid: #383835;
    --good: #0ca30c; --critical: #d03b3b;
  }
}
"""

_PAGE_CSS = _CSS_VARS + """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--page); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 880px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; padding-bottom: .4rem;
     border-bottom: 1px solid var(--border); }
.sub { color: var(--text-secondary); margin: 0 0 2rem; font-size: .9rem; }
.card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.25rem 1.5rem;
}
.banner { display: flex; align-items: center; gap: .6rem; font-weight: 600; padding: .9rem 1.1rem;
          border-radius: 8px; }
.banner.good { color: var(--good); background: color-mix(in srgb, var(--good) 12%, transparent); }
.banner.critical { color: var(--critical); background: color-mix(in srgb, var(--critical) 12%, transparent); }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-size: .85em; }
details { margin-top: .75rem; }
summary { cursor: pointer; color: var(--text-secondary); font-size: .85rem; user-select: none; }
summary:hover { color: var(--text-primary); }
.legend { display: flex; gap: 1.1rem; margin-bottom: .75rem; font-size: .8rem; color: var(--text-secondary); }
.legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }
.axis-label { fill: var(--text-muted); font-size: 11px; }
.value-label { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
.row-label { fill: var(--text-primary); font-size: 12px; }
.gate-badge { display: inline-flex; align-items: center; gap: .3rem; font-size: .8rem; font-weight: 600;
              padding: .1rem .5rem; border-radius: 999px; }
.gate-badge.pass { color: var(--good); background: color-mix(in srgb, var(--good) 14%, transparent); }
.gate-badge.fail { color: var(--critical); background: color-mix(in srgb, var(--critical) 14%, transparent); }
footer { margin-top: 3rem; color: var(--text-muted); font-size: .8rem; }
"""

_ROW_H = 28  # band height per bar row: 24px bar + 4px air, per marks-and-anatomy.md
_BAR_H = 22
_GAP = 2  # surface-gap between touching/stacked marks


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _rounded_end_path(x_from: float, x_to: float, y: float, h: float, r: float = 4) -> str:
    """Bar path square at x_from (the baseline), rounded at the far cap toward x_to.
    Works for either direction (x_to greater or less than x_from)."""
    span = abs(x_to - x_from)
    r = min(r, span, h / 2) if span > 0 else 0
    if r <= 0:
        return f"M{x_from},{y} H{x_to} V{y + h} H{x_from} Z"
    sign = 1 if x_to > x_from else -1
    cap = x_to - sign * r
    return (f"M{x_from},{y} H{cap} "
            f"Q{x_to},{y} {x_to},{y + r} "
            f"V{y + h - r} "
            f"Q{x_to},{y + h} {cap},{y + h} "
            f"H{x_from} Z")


def _svg_open(width: float, height: float) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
            f'height="{height:.0f}" role="img" xmlns="http://www.w3.org/2000/svg">')


def diverging_bar_chart(items: list[tuple[str, float]], unit: str = "%") -> str:
    """Horizontal diverging bars for a signed magnitude (e.g. savings %). One hue for
    positive, one for negative, gray baseline gridline at zero. `items`: (label, value)."""
    if not items:
        return '<p class="sub">No data.</p>'
    label_w, right_pad = 190, 56
    plot_w = 560
    width = label_w + plot_w + right_pad
    height = _ROW_H * len(items) + 24
    vmax = max((abs(v) for _, v in items), default=1) or 1
    half = plot_w / 2 - 8
    cx = label_w + plot_w / 2
    out = [_svg_open(width, height)]
    out.append(f'<line x1="{cx}" y1="4" x2="{cx}" y2="{height - 20}" '
                f'stroke="var(--baseline)" stroke-width="1"/>')
    for i, (label, value) in enumerate(items):
        y = 12 + i * _ROW_H
        frac = (value / vmax) if vmax else 0
        x_to = cx + frac * half
        color = "var(--diverging-pos)" if value >= 0 else "var(--diverging-neg)"
        path = _rounded_end_path(cx, x_to, y, _BAR_H)
        out.append(f'<text x="{label_w - 10}" y="{y + _BAR_H / 2 + 4}" text-anchor="end" '
                    f'class="row-label">{_esc(label)}</text>')
        out.append(f'<path d="{path}" fill="{color}"><title>{_esc(label)}: '
                    f'{value:+.1f}{unit}</title></path>')
        anchor = "start" if value >= 0 else "end"
        lx = x_to + (6 if value >= 0 else -6)
        out.append(f'<text x="{lx:.1f}" y="{y + _BAR_H / 2 + 4}" text-anchor="{anchor}" '
                    f'class="value-label">{value:+.1f}{unit}</text>')
    out.append(f'<text x="{cx}" y="{height - 4}" text-anchor="middle" class="axis-label">0</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_bar_chart(items: Sequence[tuple[str, Sequence[float]]],
                       series_labels: tuple[str, ...],
                       series_colors: tuple[str, ...] = ("var(--series-1)", "var(--series-2)", "var(--series-3)")
                       ) -> str:
    """Horizontal part-to-whole bars, drawn as a waterfall of cumulative steps —
    one fixed hue per series (categorical), a 2px surface gap between segments,
    rounded cap on the true final total. A tier's value CAN be negative (measure.py:
    "at tiny N the table envelope can cost more than the keys it folds") — clamping
    it to zero-width would silently disagree with the printed total, so a negative
    step instead retreats left of its predecessor: reduced fill-opacity plus a
    critical-red outline flags it as the anomaly it is, and it always carries a
    direct label (never just a hover title) since it's the one point in the row
    the story is about."""
    if not items:
        return '<p class="sub">No data.</p>'
    label_w, right_pad = 190, 70
    plot_w = 500
    legend_h = 26
    row_h = _ROW_H + 8  # a little extra air for the rare negative-step label
    width = label_w + plot_w + right_pad
    height = legend_h + row_h * len(items) + 16

    cum_by_row: list[list[float]] = []
    lo = hi = 0.0
    for _, vals in items:
        c = 0.0
        seq = [0.0]
        for v in vals:
            c += v
            seq.append(c)
        cum_by_row.append(seq)
        lo, hi = min(lo, min(seq)), max(hi, max(seq))
    span = (hi - lo) or 1.0

    def sx(v: float) -> float:
        return label_w + (v - lo) / span * plot_w

    zero_x = sx(0.0)
    out = [_svg_open(width, height)]
    lx, ly = label_w, 14
    for name, color in zip(series_labels, series_colors, strict=True):
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{lx + 15}" y="{ly}" class="axis-label">{_esc(name)}</text>')
        lx += 18 + 9 * len(name)
    if lo < 0:
        out.append(f'<line x1="{zero_x:.1f}" y1="{legend_h}" x2="{zero_x:.1f}" y2="{height - 4}" '
                    f'stroke="var(--baseline)" stroke-width="1"/>')

    for i, (label, vals) in enumerate(items):
        y = legend_h + 8 + i * row_h
        seq = cum_by_row[i]
        out.append(f'<text x="{label_w - 10}" y="{y + _BAR_H / 2 + 4}" text-anchor="end" '
                    f'class="row-label">{_esc(label)}</text>')
        n = len(vals)
        prev_px = zero_x
        last_label_x: float | None = None  # gates negative-step labels below on collision
        for j, v in enumerate(vals):
            cur_px = sx(seq[j + 1])
            direction = 1 if cur_px >= prev_px else -1
            a = prev_px + (direction * _GAP / 2 if j > 0 else 0)
            b = cur_px - (direction * _GAP / 2 if j < n - 1 else 0)
            color = series_colors[j % len(series_colors)]
            title = f'<title>{_esc(series_labels[j])}: {v:+.0f}</title>'
            anomaly = ' fill-opacity="0.55" stroke="var(--diverging-neg)" stroke-width="2"' if v < 0 else ""
            seg_px = abs(b - a)
            if direction * (b - a) > 0.5:
                if j == n - 1:
                    path = _rounded_end_path(a, b, y, _BAR_H)
                    out.append(f'<path d="{path}" fill="{color}"{anomaly}>{title}</path>')
                else:
                    x0, x1 = (a, b) if b >= a else (b, a)
                    out.append(f'<rect x="{x0:.1f}" y="{y}" width="{(x1 - x0):.1f}" height="{_BAR_H}" '
                               f'fill="{color}"{anomaly}>{title}</rect>')
            # Direct label only where it fits and won't collide with a sibling label —
            # "measure first" (marks-and-anatomy.md): too cramped to draw falls back to
            # the hover title (always present, above) and the table view (always exact).
            if v < 0 and seg_px >= 14:
                mid = (a + b) / 2
                if last_label_x is None or abs(mid - last_label_x) >= 24:
                    out.append(f'<text x="{mid:.1f}" y="{y - 3}" text-anchor="middle" '
                                f'class="value-label" fill="var(--critical)">({v:+.0f})</text>')
                    last_label_x = mid
            prev_px = cur_px
        end_anchor = "start" if prev_px >= zero_x else "end"
        out.append(f'<text x="{prev_px + (8 if end_anchor == "start" else -8):.1f}" '
                    f'y="{y + _BAR_H / 2 + 4}" text-anchor="{end_anchor}" class="value-label">'
                    f'{sum(vals):+.0f}</text>')
    out.append("</svg>")
    return "".join(out)


def forest_plot(rows: list[dict[str, Any]], form_label: str, control_label: str) -> str:
    """Point + 95%-CI whisker per model, two series (form vs control) per row. Each row
    also carries a pass/fail status badge from the shared worst-case-gap verdict math."""
    if not rows:
        return '<p class="sub">No data.</p>'
    label_w, right_pad = 110, 70
    plot_w = 480
    width = label_w + plot_w + right_pad
    legend_h = 26
    row_h = _ROW_H + 10
    height = legend_h + row_h * len(rows) + 16

    def x(v: float) -> float:
        return label_w + v * plot_w

    out = [_svg_open(width, height)]
    lx, ly = label_w, 14
    for name, color in ((control_label, "var(--series-2)"), (form_label, "var(--series-1)")):
        out.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{color}"/>')
        out.append(f'<text x="{lx + 15}" y="{ly}" class="axis-label">{_esc(name)}</text>')
        lx += 20 + 8 * len(name)
    for t in (0, 25, 50, 75, 100):
        gx = x(t / 100)
        out.append(f'<line x1="{gx:.1f}" y1="{legend_h}" x2="{gx:.1f}" y2="{height - 16}" '
                    f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{gx:.1f}" y="{height - 4}" text-anchor="middle" '
                    f'class="axis-label">{t}%</text>')
    for i, r in enumerate(rows):
        y = legend_h + 8 + i * row_h
        out.append(f'<text x="{label_w - 10}" y="{y + 10}" text-anchor="end" '
                    f'class="row-label">{_esc(r["model"])}</text>')
        for key, color, dy in (("control", "var(--series-2)", -6), ("form", "var(--series-1)", 6)):
            acc, ci = r[f"{key}_acc"], r[f"{key}_ci"]
            cx_, cy_ = x(acc), y + dy + 8
            out.append(f'<line x1="{x(max(acc - ci, 0)):.1f}" y1="{cy_}" '
                        f'x2="{x(min(acc + ci, 1)):.1f}" y2="{cy_}" '
                        f'stroke="{color}" stroke-width="2"/>')
            out.append(f'<circle cx="{cx_:.1f}" cy="{cy_}" r="5" fill="{color}" '
                       f'stroke="var(--surface-1)" stroke-width="2">'
                       f'<title>{_esc(r["model"])} — {key}: {acc:.0%} ±{ci * 100:.0f}pt</title>'
                       f'</circle>')
        out.append(f'<text x="{label_w + plot_w + 10}" y="{y + 10}" '
                    f'class="value-label" fill="var(--{"good" if r["passed"] else "critical"})">'
                    f'{"PASS" if r["passed"] else "FAIL"}</text>')
    out.append("</svg>")
    return "".join(out)


def _gate_banner(rows: list[dict[str, Any]]) -> str:
    failures = [r for r in rows if not r.get("roundtrip_ok", False)]
    total, passed = len(rows), len(rows) - len(failures)
    if failures:
        items = "".join(f"<li><code>{_esc(r.get('tool'))}</code> / "
                        f"<code>{_esc(r.get('sha'))}</code> ({_esc(r.get('shape'))})</li>"
                        for r in failures)
        return (f'<div class="banner critical">✕ INVALID — {len(failures)}/{total} payloads '
                f'FAILED the round-trip gate</div><ul>{items}</ul>'
                f'<p class="sub">Savings below are meaningless until this is 0.</p>')
    return f'<div class="banner good">✓ All {passed}/{total} payloads round-trip losslessly</div>'


def _details(summary: str, table_html: str) -> str:
    return f'<details><summary>{_esc(summary)}</summary>{table_html}</details>'


def _attestation_card(corpus_label: str, n_payloads: int) -> str:
    """Compact HTML counterpart to report.build_verify_header's self-cert caveats —
    same claims (lossless gate, no egress, fail-open), condensed for a card."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("terse")
    except PackageNotFoundError:
        v = "(editable/dev)"
    return f"""<div class="card">
<p class="sub" style="margin:0 0 .5rem">
terse <code>{_esc(v)}</code> · python <code>{_esc(platform.python_version())}</code> ·
os <code>{_esc(platform.system())}</code> · corpus: {_esc(corpus_label)} —
{n_payloads} payloads</p>
<p class="sub" style="margin:0">This report proves lossless round-trip + measured
savings on this corpus. It does not replace: the <code>pytest</code> suite, a
no-egress check (<code>grep</code> for network calls), or reading
<code>proxy.py</code>'s fail-open path — verify those yourself.</p>
</div>"""


def build_html_report(rows: list[dict[str, Any]], coverage: dict[str, Any],
                       attestation: tuple[str, int] | None = None) -> str:
    """HTML counterpart to report.build_report: same sections (gate, coverage,
    savings by shape/tool, tier attribution), each chart backed by a plain <table>
    for the accessibility "table view" requirement. `attestation`, if given, is
    (corpus_label, n_payloads) — the `terse verify` self-cert card."""
    shapes = sorted({r["shape"] for r in rows})
    tools = sorted({r.get("tool", "?") for r in rows})

    # savings by shape
    shape_items, shape_rows_html = [], []
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        raw, cmp_ = _sum(sub, "cl100k", "raw"), _sum(sub, "cl100k", "compressed")
        saved = raw - cmp_
        pct = (saved / raw * 100) if raw else 0.0
        shape_items.append((shape, pct))
        shape_rows_html.append(f"<tr><td>{_esc(shape)}</td><td class='num'>{len(sub)}</td>"
                                f"<td class='num'>{raw}</td><td class='num'>{cmp_}</td>"
                                f"<td class='num'>{saved:+d}</td><td class='num'>{_pct(saved, raw)}</td></tr>")

    # savings by tool
    tool_items, tool_rows_html = [], []
    for tool in tools:
        sub = [r for r in rows if r.get("tool") == tool]
        raw, cmp_ = _sum(sub, "cl100k", "raw"), _sum(sub, "cl100k", "compressed")
        saved = raw - cmp_
        pct = (saved / raw * 100) if raw else 0.0
        tool_items.append((tool, pct))
        tool_rows_html.append(f"<tr><td><code>{_esc(tool)}</code></td><td class='num'>{raw}</td>"
                               f"<td class='num'>{cmp_}</td><td class='num'>{saved:+d}</td>"
                               f"<td class='num'>{_pct(saved, raw)}</td></tr>")
    tool_items.sort(key=lambda kv: -kv[1])

    # tier attribution
    tier_items, tier_rows_html = [], []
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        m = _sum(sub, "saved_cl100k", "minify")
        t = _sum(sub, "saved_cl100k", "tabularize")
        d = _sum(sub, "saved_cl100k", "dictionary")
        tier_items.append((shape, [m, t, d]))
        tier_rows_html.append(f"<tr><td>{_esc(shape)}</td><td class='num'>{m:+d}</td>"
                               f"<td class='num'>{t:+d}</td><td class='num'>{d:+d}</td>"
                               f"<td class='num'>{m + t + d:+d}</td></tr>")

    cov_tool_rows = "".join(f"<tr><td><code>{_esc(t)}</code></td><td class='num'>{n}</td></tr>"
                             for t, n in sorted(coverage.get("by_tool", {}).items(), key=lambda kv: -kv[1]))
    cov_shape_rows = "".join(f"<tr><td>{_esc(s)}</td><td class='num'>{n}</td></tr>"
                              for s, n in sorted(coverage.get("by_shape", {}).items(), key=lambda kv: -kv[1]))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>terse measurement report</title>
<style>{_PAGE_CSS}</style></head>
<body><main>
<h1>terse measurement report</h1>
<p class="sub">Token-savings measurement over the captured corpus. Charts are inline
SVG — no JS, no network. Hover a bar/point for its exact value.</p>
{_attestation_card(*attestation) if attestation else ""}

<h2>Lossless gate</h2>
<div class="card">{_gate_banner(rows)}</div>

<h2>Coverage</h2>
<div class="card">
<p class="sub">Total payloads captured: <strong>{coverage.get('total', 0)}</strong></p>
<table><thead><tr><th>Tool</th><th class="num">Payloads</th></tr></thead>
<tbody>{cov_tool_rows}</tbody></table>
<table style="margin-top:1rem"><thead><tr><th>Shape bucket</th><th class="num">Payloads</th></tr></thead>
<tbody>{cov_shape_rows}</tbody></table>
</div>

<h2>Tier-0 savings by shape bucket</h2>
<div class="card">
{diverging_bar_chart(shape_items)}
{_details("Table view", f"<table><thead><tr><th>Shape</th><th class='num'>n</th>"
          f"<th class='num'>raw tok</th><th class='num'>terse tok</th>"
          f"<th class='num'>saved</th><th class='num'>%</th></tr></thead>"
          f"<tbody>{''.join(shape_rows_html)}</tbody></table>")}
</div>

<h2>Tier-0 savings by tool</h2>
<div class="card">
{diverging_bar_chart(tool_items)}
{_details("Table view", f"<table><thead><tr><th>Tool</th><th class='num'>raw tok</th>"
          f"<th class='num'>terse tok</th><th class='num'>saved</th>"
          f"<th class='num'>%</th></tr></thead><tbody>{''.join(tool_rows_html)}</tbody></table>")}
</div>

<h2>Tier attribution by shape</h2>
<div class="card">
<p class="sub">minify = whitespace/escaping · tabularize = repeated keys folded ·
dictionary = repeated values folded, Tier 0.5. A tier can go negative at a
small sample size — the table envelope costs more than it folds — shown as a
red-outlined step back, never hidden.</p>
{stacked_bar_chart(tier_items, ("minify", "tabularize", "dictionary"))}
{_details("Table view", f"<table><thead><tr><th>Shape</th><th class='num'>minify</th>"
          f"<th class='num'>tabularize</th><th class='num'>dictionary</th>"
          f"<th class='num'>total</th></tr></thead><tbody>{''.join(tier_rows_html)}</tbody></table>")}
</div>

<footer>Generated by <code>terse</code> — offline, no telemetry, no egress.</footer>
</main></body></html>
"""


def build_html_diff_report(results: dict, form_label: str = "diff-form",
                            control_label: str = "full-terse") -> str:
    """HTML counterpart to report.build_diff_report / build_fluency_report's gap
    section: a forest plot of per-model accuracy with 95% CI, gated on the worst model."""
    plot_rows, gap_rows = [], {}
    for model, rows in results.items():
        n = len(rows)
        if not n:
            continue
        facc, fse = _form_stats(rows, "terse_ok" if "terse_ok" in rows[0] else "diff_ok")
        cacc, cse = facc, fse
        if "diff_ok" in rows[0] and "terse_ok" in rows[0]:
            facc, fse = _form_stats(rows, "diff_ok")
            cacc, cse = _form_stats(rows, "terse_ok")
        gap_rows[model] = (facc, fse, cacc, cse)

    worst = _worst_case_gap(gap_rows)
    for model, (facc, fse, cacc, cse) in gap_rows.items():
        gap = facc - cacc
        passed = gap >= -0.05 - 1e-9
        plot_rows.append({"model": model, "form_acc": facc, "form_ci": _ci(fse),
                           "control_acc": cacc, "control_ci": _ci(cse), "passed": passed})

    verdict_html = ""
    if worst:
        cls = "good" if worst.passed else "critical"
        verdict_html = (f'<div class="banner {cls}">{"✓ PASS" if worst.passed else "✕ FAIL"} — '
                        f'worst-case model <code>{_esc(worst.model)}</code>: {form_label} '
                        f'{worst.form_acc:.0%} vs {control_label} {worst.control_acc:.0%} '
                        f'(gap {worst.gap:+.0%} ±{worst.gap_ci * 100:.0f}pt)</div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>terse fluency — comprehension gap</title>
<style>{_PAGE_CSS}</style></head>
<body><main>
<h1>terse comprehension gap</h1>
<p class="sub">Per-model accuracy, {form_label} vs {control_label}, with 95% CI whiskers.
Gated on the worst model, never the mean.</p>
<h2>Verdict</h2>
<div class="card">{verdict_html}</div>
<h2>Accuracy by model</h2>
<div class="card">{forest_plot(plot_rows, form_label, control_label)}</div>
<footer>Generated by <code>terse</code> — offline, no telemetry, no egress.</footer>
</main></body></html>
"""
