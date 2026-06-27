"""Spike report: token delta per shape bucket with tier attribution.

Honesty requirements (plan Section 7, principle #24):
  - every shape bucket is shown, including near-zero / negative ones — never
    averaged away into a single headline number
  - coverage (which tools, how many payloads) is a first-class section, so a thin
    sample cannot read as "nothing to compress"
  - the lossless gate result gates the whole report: any round-trip failure prints
    an INVALID banner, because savings on top of lost data are meaningless
"""

from __future__ import annotations

from typing import Any


def _pct(saved: int, base: int) -> str:
    return f"{(saved / base * 100):+.1f}%" if base else "n/a"


def _sum(rows: list[dict[str, Any]], *path: str) -> int:
    total = 0
    for r in rows:
        v: Any = r
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            total += int(v)
    return total


def build_probe_report(
    vr_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]
) -> str:
    """Render the Tier-0.5 ceiling probes — value redundancy + cross-call overlap.

    These are UPPER BOUNDS on what a dictionary coder / diff encoder could save,
    measured ON TOP of what tabularize already achieves. They inform the go/no-go
    on building Tier 0.5; they do not compress anything.
    """
    out: list[str] = ["# terse ceiling probes (Tier 0.5)", ""]

    out += [
        "## Value redundancy — dictionary-coding headroom",
        "",
        "Repeated VALUE tokens across cells, beyond the repeated KEYS tabularize folds.",
        "`est dict saving` is a conservative upper bound (first occurrence kept as legend).",
        "",
        "| Tool | sha | cells | redundancy | redundant tok | est dict saving |",
        "|---|---|---|---|---|---|",
    ]
    for r in vr_rows:
        out.append(
            f"| `{r['tool']}` | `{r['sha']}` | {r['cells']} | {r['redundancy_ratio']:.1%} "
            f"| {r['redundant_value_tokens']} | {r['est_dict_saving_tokens']} |"
        )
    if vr_rows:
        ratios = sorted(r["redundancy_ratio"] for r in vr_rows)
        median = ratios[len(ratios) // 2]
        verdict = "worth a Tier 0.5 build" if median >= 0.15 else "thin — likely skip Tier 0.5"
        out += ["", f"Median value-redundancy: **{median:.1%}** → {verdict}.", ""]
    else:
        out += ["", "_No record-shaped payloads in corpus to probe._", ""]

    out += [
        "## Cross-call overlap — diffing headroom",
        "",
        "Token overlap between successive same-tool payloads. `est delta saving` is the",
        "fraction of the current payload already present in the prior one (upper bound).",
        "",
    ]
    if overlap_rows:
        out += [
            "| Tool | prev | curr | curr tok | shared | overlap |",
            "|---|---|---|---|---|---|",
        ]
        for r in overlap_rows:
            out.append(
                f"| `{r['tool']}` | `{r['prev_sha']}` | `{r['curr_sha']}` | {r['curr_tokens']} "
                f"| {r['shared_tokens']} | {r['overlap_ratio']:.1%} |"
            )
        out.append("")
    else:
        out += [
            "_No same-tool payload pairs in corpus — capture a tool 2+ times in an agent",
            "loop to measure diffing headroom._",
            "",
        ]
    return "\n".join(out)


def build_report(rows: list[dict[str, Any]], coverage: dict[str, Any]) -> str:
    out: list[str] = ["# terse spike report", ""]

    # --- Lossless gate (gates everything) ---
    failures = [r for r in rows if not r.get("roundtrip_ok", False)]
    total = len(rows)
    passed = total - len(failures)
    out += ["## Lossless gate", ""]
    if failures:
        out += [
            f"**INVALID — {len(failures)}/{total} payloads FAILED the round-trip gate.**",
            "Savings below are meaningless until this is 0. Failing shas:",
            "",
            *[f"- `{r.get('tool')}` / `{r.get('sha')}` ({r.get('shape')})" for r in failures],
            "",
        ]
    else:
        out += [f"All {passed}/{total} payloads round-trip losslessly. ✅", ""]

    # --- Coverage ---
    out += ["## Coverage", "", f"Total payloads captured: **{coverage.get('total', 0)}**", ""]
    out += ["| Tool | Payloads |", "|---|---|"]
    for tool, n in sorted(coverage.get("by_tool", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| `{tool}` | {n} |")
    out += ["", "| Shape bucket | Payloads |", "|---|---|"]
    for shape, n in sorted(coverage.get("by_shape", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| {shape} | {n} |")
    out.append("")

    # --- Savings per shape bucket (Tier-0 total, cl100k) ---
    shapes = sorted({r["shape"] for r in rows})
    out += [
        "## Tier-0 savings by shape bucket (cl100k)",
        "",
        "Headline = full Tier-0 (minify + tabularize) vs the raw bytes the model would see.",
        "",
        "| Shape | n | raw tok | terse tok | saved | % |",
        "|---|---|---|---|---|---|",
    ]
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        raw = _sum(sub, "cl100k", "raw")
        cmp_ = _sum(sub, "cl100k", "compressed")
        saved = raw - cmp_
        out.append(f"| {shape} | {len(sub)} | {raw} | {cmp_} | {saved:+d} | {_pct(saved, raw)} |")
    raw_all = _sum(rows, "cl100k", "raw")
    cmp_all = _sum(rows, "cl100k", "compressed")
    out.append(
        f"| **ALL** | {len(rows)} | {raw_all} | {cmp_all} | {raw_all - cmp_all:+d} | "
        f"{_pct(raw_all - cmp_all, raw_all)} |"
    )
    out.append("")

    # --- Per-tool savings (the proxy decision is per-tool, not per-shape) ---
    # Shape buckets can hide a deep-nested win next to a true no-op (e.g. runecho's
    # nested symbol lists vs a single compact object both land in 'compact-json').
    tools = sorted({r.get("tool", "?") for r in rows})
    out += [
        "## Tier-0 savings by tool (cl100k)",
        "",
        "Per-tool, because terse's value is shape-dependent and a blanket average hides it.",
        "",
        "| Tool | shape | raw tok | terse tok | saved | % |",
        "|---|---|---|---|---|---|",
    ]
    tool_rows = []
    for tool in tools:
        sub = [r for r in rows if r.get("tool") == tool]
        raw = _sum(sub, "cl100k", "raw")
        cmp_ = _sum(sub, "cl100k", "compressed")
        shape = sub[0]["shape"] if sub else "?"
        tool_rows.append((raw - cmp_, raw, cmp_, tool, shape))
    for saved, raw, cmp_, tool, shape in sorted(tool_rows, reverse=True):
        out.append(f"| `{tool}` | {shape} | {raw} | {cmp_} | {saved:+d} | {_pct(saved, raw)} |")
    out.append("")

    # --- Tier attribution (where the saving came from) ---
    out += [
        "## Tier attribution by shape (cl100k tokens saved)",
        "",
        "minify = whitespace + \\uXXXX unescaping · tabularize = repeated keys folded ·",
        "dictionary = repeated values folded into an inline legend (Tier 0.5).",
        "A ~0 minify column means the payload arrived already-compact (the headroom no-op).",
        "",
        "| Shape | minify | tabularize | dictionary | total |",
        "|---|---|---|---|---|",
    ]
    for shape in shapes:
        sub = [r for r in rows if r["shape"] == shape]
        m = _sum(sub, "saved_cl100k", "minify")
        t = _sum(sub, "saved_cl100k", "tabularize")
        d = _sum(sub, "saved_cl100k", "dictionary")
        out.append(f"| {shape} | {m:+d} | {t:+d} | {d:+d} | {m + t + d:+d} |")
    out.append("")

    # --- Anthropic ground truth (if measured) ---
    if any("anthropic" in r for r in rows):
        a_raw = _sum(rows, "anthropic", "raw")
        a_cmp = _sum(rows, "anthropic", "compressed")
        out += [
            "## Anthropic count_tokens (ground truth)",
            "",
            f"raw {a_raw} -> terse {a_cmp} = {a_raw - a_cmp:+d} ({_pct(a_raw - a_cmp, a_raw)})",
            "",
        ]

    return "\n".join(out)
