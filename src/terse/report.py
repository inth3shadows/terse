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

import math
from typing import Any, NamedTuple

_GAP_TOLERANCE = 0.05  # shared pass/fail tolerance for both worst-case verdict gates below


def _form_stats(rows: list[dict[str, Any]], form: str) -> tuple[float, float]:
    """(accuracy, standard_error) for one form over rows carrying success COUNTS.

    accuracy = Σsuccesses / Σtrials. SE is the pooled binomial SE of that estimator:
    each row is t trials of a Bernoulli with p̂=k/t, so Var(total successes)=Σ t·p̂(1-p̂)
    and SE(acc)=√Var / Σt. This is stable at the realistic small trial count (N=2–3),
    where an empirical std across N whole-eval runs would be pure noise. At trials=1
    every p̂∈{0,1} → SE=0, so single-trial runs report exactly as before.
    """
    tot_t = tot_k = 0
    var = 0.0
    for r in rows:
        t = r.get("trials", 1)
        k = int(r[form])
        tot_t += t
        tot_k += k
        if t > 0:
            p = k / t
            var += t * p * (1 - p)
    if tot_t == 0:
        return 0.0, 0.0
    return tot_k / tot_t, math.sqrt(var) / tot_t


def _ci(se: float) -> float:
    """95% half-width in accuracy units."""
    return 1.96 * se


class GapVerdict(NamedTuple):
    model: str
    gap: float
    form_acc: float
    control_acc: float
    gap_ci: float
    passed: bool


def _worst_case_gap(
    rows: dict[str, tuple[float, float, float, float]], tol: float = _GAP_TOLERANCE
) -> GapVerdict | None:
    """Shared verdict-gating math for both fluency-style reports — principle #24, gate on
    the worst model, never the mean. `rows` maps model to a 4-tuple of form_acc, form_se,
    control_acc, control_se. Returns the model with the lowest gap as a GapVerdict, or
    None if `rows` is empty. gap = form_acc minus control_acc; gap_ci is the 95%
    half-width of the pooled standard error; passed iff gap is at least -tol, inclusive
    of the boundary. Callers access fields by name, e.g. verdict.form_acc, never by
    position, so a future field reorder can't silently swap values."""
    worst = None  # (model, gap, facc, cacc, gap_ci) — cheapest to track positionally here;
    for model, (facc, fse, cacc, cse) in rows.items():  # this is a private local, not the
        gap = facc - cacc                               # public interface callers rely on.
        gap_ci = _ci(math.sqrt(fse ** 2 + cse ** 2))
        if worst is None or gap < worst[1]:
            worst = (model, gap, facc, cacc, gap_ci)
    if worst is None:
        return None
    model, gap, facc, cacc, gap_ci = worst
    passed = gap >= -tol - 1e-9
    return GapVerdict(model, gap, facc, cacc, gap_ci, passed)


def _format_worst_case_line(verdict: GapVerdict, tol: float, form_label: str, control_label: str) -> str:
    return (f"- Worst-case model `{verdict.model}`: {form_label} {verdict.form_acc:.0%} vs "
            f"{control_label} {verdict.control_acc:.0%} (gap {verdict.gap:+.0%} "
            f"±{verdict.gap_ci * 100:.0f} pts). **{'PASS' if verdict.passed else 'FAIL'}** "
            f"at {tol:.0%} tolerance.")


def diff_gap_rows(results: dict) -> dict[str, tuple[float, float, float, float]]:
    """(form=diff_ok, control=terse_ok) gap-row tuples per model — the same shape
    `_worst_case_gap` and the bar-chart renderers (html/terminal) consume, computed
    once here so a chart's gap can never read differently than build_diff_report's."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for model, rows in results.items():
        if not rows:
            continue
        facc, fse = _form_stats(rows, "diff_ok")
        cacc, cse = _form_stats(rows, "terse_ok")
        out[model] = (facc, fse, cacc, cse)
    return out


def fluency_gap_rows(results: dict) -> tuple[dict[str, tuple[float, float, float, float]], list[str]]:
    """(form=best of terse/primer, control=raw) gap-row tuples per model, for the bar-
    chart renderers. Excludes any model whose raw control failed (0% — a backend/config
    error, not a comprehension result), matching build_fluency_report's gate. Returns
    (gap_rows, excluded_model_names)."""
    out: dict[str, tuple[float, float, float, float]] = {}
    broken: list[str] = []
    for model, rows in results.items():
        if not rows:
            continue
        racc, rse = _form_stats(rows, "raw_ok")
        if racc == 0:
            broken.append(model)
            continue
        tacc, tse = _form_stats(rows, "terse_ok")
        pacc, pse = _form_stats(rows, "primer_ok")
        best, best_se = (tacc, tse) if tacc >= pacc else (pacc, pse)
        out[model] = (best, best_se, racc, rse)
    return out, broken


def dropeval_gap_rows(results: dict) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """Per-model (recall, precision, accuracy) gap-row tuples for build_dropeval_report
    and its terminal-bar companion. Control is always a fixed 100% ideal (se=0) — there's
    no raw/full-terse form to compare against here, only "did the model do the right
    thing." Same per-model math build_dropeval_report's own table loop uses, kept in one
    place so the two verdicts (markdown table, terminal chart) can never disagree."""
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        aacc, ase = _form_stats(rows, "answer_ok")
        out[model] = {
            "recall": (racc, rse, 1.0, 0.0),
            "precision": (pacc, pse, 1.0, 0.0),
            "accuracy": (aacc, ase, 1.0, 0.0),
        }
    return out


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


def build_tokenizer_report(rows: list[dict[str, Any]]) -> str:
    """Render cross-tokenizer invariance — cl100k vs o200k savings % per tool.

    True Anthropic ground truth needs the count_tokens API (a key we don't have, and
    Claude has no public local tokenizer). Invariance across two different vocabs is
    the keyless substitute: if the savings % barely moves, it's robust to Claude's.
    """
    from .tokenize import CL100K, O200K

    out: list[str] = [
        "# terse cross-tokenizer invariance",
        "",
        "No Anthropic key available (Claude Code uses OAuth; no public Claude tokenizer).",
        "Substitute: savings % under two different BPE vocabularies. Stability => robust",
        "to Claude's tokenizer, because structural folding removes tokens in any vocab.",
        "",
        "| Tool | cl100k % | o200k % | Δ (pts) |",
        "|---|---|---|---|",
    ]
    deltas = []
    for r in sorted(rows, key=lambda r: -(r[CL100K]["pct"] or 0)):
        a = r[CL100K]["pct"]
        b = r[O200K]["pct"]
        if a is None or b is None:
            out.append(f"| `{r['tool']}` | n/a | n/a | n/a |")
            continue
        d = abs(a - b)
        deltas.append(d)
        out.append(f"| `{r['tool']}` | {a:+.1f}% | {b:+.1f}% | {d:.1f} |")
    out.append("")
    if deltas:
        worst = max(deltas)
        mean = sum(deltas) / len(deltas)
        verdict = "savings are tokenizer-invariant" if worst <= 3.0 else "savings vary by tokenizer — investigate"
        out += [
            f"Max divergence: **{worst:.1f} pts**, mean **{mean:.1f} pts** → {verdict}.",
            "",
            "_To get a real Anthropic point-check: provide a key and run "
            "`terse measure --anthropic` (recommend gh-only — public data — to avoid "
            "sending private payloads externally)._",
            "",
        ]
    return "\n".join(out)


def build_verify_header(corpus_label: str, n_payloads: int) -> str:
    """Attestation header for `terse verify` — states what the report proves and what an
    adopter must still verify themselves (tests, no-egress, fail-open). Self-contained so
    the markdown stands alone as a shareable proof. No timestamp, so the artifact stays
    reproducible (principle #31)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("terse")
    except PackageNotFoundError:
        v = "(editable/dev)"
    return "\n".join([
        "# terse — verification report",
        "",
        f"- terse `{v}`  ·  python `{platform.python_version()}`  ·  os `{platform.system()}`",
        f"- corpus: {corpus_label} — {n_payloads} payloads",
        "",
        "## What the tables below prove",
        "",
        "- **Lossless** — every payload round-trips byte-faithfully through terse. The "
        "lossless gate INVALIDATES the whole report if any payload fails, because token "
        "savings on top of corrupted data are meaningless.",
        "- **Savings** — measured cl100k-token reduction per shape bucket and per tool on "
        "this corpus. terse's win is shape-dependent, so it is never averaged into one "
        "headline number.",
        "",
        "## What this does NOT replace — verify these yourself",
        "",
        "- **Correctness suite:** `pytest` — the full lossless / diff / proxy test set "
        "(runs in CI on Python 3.11–3.13).",
        '- **No UNEXPECTED egress:** `grep -rE "requests|urllib|socket" src/terse` finds '
        "real network code in three places — `fluency.py` and `dropeval.py` (each an "
        "explicit, opt-in model eval) and `transport.py` (the proxy's own downstream "
        "connection). A stdio-only downstream makes zero network calls; an HTTP/SSE "
        "downstream (opt-in via `--config`/a `url`-configured server) talks only to the "
        "target you configured — never a third party. The same grep also flags a few "
        "incidental, non-networking hits (the word \"requests\" in a comment or docstring) "
        "elsewhere in the tree; read the actual matches rather than trusting a count. The "
        "proxy persists nothing beyond what `--capture-dir`/`--debug-log` explicitly ask for.",
        "- **Fail-open:** read `src/terse/proxy.py` — any parse/compress error forwards the "
        "ORIGINAL tool result unchanged; terse never drops or blocks a tool call.",
        "",
        "---",
        "",
        "",
    ])


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


def build_trend_report(runs: list[dict[str, Any]]) -> str:
    """Render the `measure --history` trend (#51 fast-follow) — one row per past run,
    oldest first, so a reader sees whether the win is improving, flat, or regressing
    as the corpus grows/changes over time. `runs` is `history.load_history()`'s output
    WITH the current run already appended by the caller — this function only ever
    displays the persisted summary numbers, never re-derives them, so a rendered trend
    can never drift from what was actually written to the history file."""
    out: list[str] = ["## Trend across runs", ""]
    if len(runs) < 2:
        out += ["_Only one run recorded so far — trend needs at least two "
                "`--history` runs to show a delta._", ""]
        return "\n".join(out)
    out += [
        "| # | timestamp | label | payloads | lossless | raw tok | terse tok | saved % | Δ pts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    prev_pct: float | None = None
    for i, r in enumerate(runs, start=1):
        pct = r.get("saved_pct")
        pct_s = f"{pct:+.1f}%" if pct is not None else "n/a"
        delta = f"{pct - prev_pct:+.1f}" if pct is not None and prev_pct is not None else "—"
        gate = f"{r.get('lossless_pass', '?')}/{r.get('n_payloads', '?')}"
        out.append(
            f"| {i} | {r.get('ts', '?')} | {r.get('label') or '—'} | {r.get('n_payloads', '?')} "
            f"| {gate} | {r.get('raw_tok', '?')} | {r.get('compressed_tok', '?')} "
            f"| {pct_s} | {delta} |")
        prev_pct = pct
    out.append("")
    return "\n".join(out)


def _build_diff_style_report(results: dict, title: str, intro: list[str],
                             empty_hint: str, control_label: str = "full-terse") -> str:
    """Shared body for build_diff_report and build_text_diff_report — the row shape
    ({qid, qtype, transform, trials, terse_ok, diff_ok}) and verdict math are identical
    for both; only the title/intro/empty-hint copy and the control column's label
    differ."""
    out: list[str] = [title, ""]
    out += intro
    if not results or not any(results.values()):
        out += [empty_hint, ""]
        return "\n".join(out)

    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound.",
        "",
        f"| Model | q | {control_label} | diff | regressions |",
        "|---|---|---|---|---|",
    ]
    gap_rows: dict[str, tuple[float, float, float, float]] = {}
    for model, rows in results.items():
        n = len(rows)
        if not n:
            continue
        facc, fse = _form_stats(rows, "terse_ok")
        dacc, dse = _form_stats(rows, "diff_ok")
        regr = sum(1 for r in rows if int(r["terse_ok"]) == r.get("trials", 1)
                   and int(r["diff_ok"]) < r.get("trials", 1))
        gap_rows[model] = (dacc, dse, facc, fse)  # form=diff, control=control_label
        out.append(f"| `{model}` | {n} | {facc:.0%} ±{_ci(fse) * 100:.0f} "
                   f"| {dacc:.0%} ±{_ci(dse) * 100:.0f} | {regr} |")
    out.append("")

    out += ["## Verdict", ""]
    worst = _worst_case_gap(gap_rows)
    if worst:
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "diff-form", control_label))
        if worst.passed:
            out.append("- Reading the diff costs no comprehension beyond tolerance — safe to "
                       "enable `proxy --diff` for the tested models.")
        else:
            out.append("- The diff form regresses comprehension beyond tolerance — keep "
                       "`proxy --diff` off, or restrict it to tools whose diffs stay legible.")
    out.append("")
    return "\n".join(out)


def build_diff_report(results: dict) -> str:
    """Render the cross-call diff fluency eval: does a model read a diff against the
    prior result as accurately as the full current result?

    `results` is {model: [row,...]} from fluency.run_diff_fluency; each row carries
    full-terse (`terse_ok`) and diff-form (`diff_ok`) success counts over the same
    questions. The verdict gates on the worst model (principle #24): the proxy emits a
    diff only when smaller, so this bounds the comprehension cost of enabling it.
    """
    return _build_diff_style_report(
        results,
        "# terse cross-call diff fluency",
        ["Does a model read a diff against the prior same-tool result as accurately as the",
         "full current result? Same questions, paired per question; ground truth is",
         "deterministic. Risk-item check for `proxy --diff` before turning it on.", ""],
        "No model answers, or no same-tool payload PAIRS in the corpus. Capture a tool "
        "2+ times (an agent loop) and configure a backend, then re-run `terse fluency --diff`.",
    )


def build_text_diff_report(results: dict) -> str:
    """Render the text-diff fluency eval: does a model reconstruct the current TEXT as
    accurately from (previous text + text-diff) as from the full current text?

    `results` is {model: [row,...]} from fluency.run_text_diff_fluency. Unlike
    build_diff_report, the control form is raw text, not full-terse — Tier 0 doesn't
    compress non-JSON text at all — so the control column is labeled accordingly.
    """
    return _build_diff_style_report(
        results,
        "# terse text-diff fluency",
        ["Does a model reconstruct the current text as accurately from (previous text +",
         "text-diff) as from the full current text? Tier 0 doesn't compress non-JSON text",
         "at all, so the control form here is the raw text, not a compressed one. Risk-item",
         "check before enabling `proxy --diff` for text-heavy tools.", ""],
        "No model answers, or no same-tool TEXT payload PAIRS in the corpus (JSON pairs are "
        "`--diff`'s domain, not this one's). Capture a text-producing tool 2+ times, then "
        "re-run `terse fluency --text-diff-eval`.",
        control_label="raw text",
    )


def build_dropeval_report(results: dict) -> str:
    """Render the drop-to-retrieve behavioral eval: does a real tool-calling model call
    `terse.retrieve` when a dropped field is needed (recall), and leave it alone when it
    isn't (precision / no-overfetch)?

    `results` is {model: [row,...]} from dropeval.run_drop_fluency; each row carries
    `kind` ("recall"|"precision") plus retrieve_ok/answer_ok/handle_ok success counts
    over `trials`. The verdict gates on the WORST model across all three metrics
    (principle #24) — a policy that's unsafe for one model in the fleet is unsafe,
    full stop — reusing the same worst-case-gap machinery as build_diff_report/
    build_fluency_report, with a 100%-ideal control (a real tool call either happens
    correctly or it doesn't; there's no "raw form" to compare against here).
    """
    out: list[str] = ["# terse drop-to-retrieve behavioral eval", ""]
    out += [
        "Does a real tool-calling model call `terse.retrieve` when a `__terse_dropped__`",
        "marker matters (recall), and leave it alone when it doesn't (precision /",
        "no-overfetch)? Ground truth is deterministic; the loop mirrors the proxy's real",
        "2-turn retrieve protocol exactly (same primer, same tool, same miss string).",
        "",
    ]
    if not results or not any(results.values()):
        out += [
            "No tool-capable model answers, or no drop-marked record payloads in the",
            "corpus — set a policy with a `drop-to-retrieve` field and configure a model",
            "(TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS or --anthropic), then re-run",
            "`terse fluency --drop-eval --policy <file>`.",
            "",
        ]
        return "\n".join(out)

    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound.",
        "",
        "| Model | recall q | retrieve-recall | precision (no-overfetch) | final-accuracy "
        "| handle-accuracy |",
        "|---|---|---|---|---|---|",
    ]
    recall_gate: dict[str, tuple[float, float, float, float]] = {}
    precision_gate: dict[str, tuple[float, float, float, float]] = {}
    accuracy_gate: dict[str, tuple[float, float, float, float]] = {}
    for model, rows in results.items():
        if not rows:
            continue
        recall_rows = [r for r in rows if r["kind"] == "recall"]
        precision_rows = [r for r in rows if r["kind"] == "precision"]
        racc, rse = _form_stats(recall_rows, "retrieve_ok") if recall_rows else (0.0, 0.0)
        pacc, pse = _form_stats(precision_rows, "retrieve_ok") if precision_rows else (0.0, 0.0)
        aacc, ase = _form_stats(rows, "answer_ok")
        hacc, hse = _form_stats(recall_rows, "handle_ok") if recall_rows else (0.0, 0.0)
        # control is a fixed 100% ideal (se=0) — there's no raw/full-terse form to pair
        # against here, only "did the model do the right thing."
        recall_gate[model] = (racc, rse, 1.0, 0.0)
        precision_gate[model] = (pacc, pse, 1.0, 0.0)
        accuracy_gate[model] = (aacc, ase, 1.0, 0.0)
        out.append(f"| `{model}` | {len(recall_rows)} | {racc:.0%} ±{_ci(rse) * 100:.0f} "
                   f"| {pacc:.0%} ±{_ci(pse) * 100:.0f} | {aacc:.0%} ±{_ci(ase) * 100:.0f} "
                   f"| {hacc:.0%} ±{_ci(hse) * 100:.0f} |")
    out.append("")

    out += ["## Verdict", ""]
    recall_worst = _worst_case_gap(recall_gate)
    precision_worst = _worst_case_gap(precision_gate)
    accuracy_worst = _worst_case_gap(accuracy_gate)
    if recall_worst and precision_worst and accuracy_worst:
        out.append(_format_worst_case_line(recall_worst, _GAP_TOLERANCE, "retrieve-recall",
                                           "ideal (100%)"))
        out.append(_format_worst_case_line(precision_worst, _GAP_TOLERANCE, "no-overfetch",
                                           "ideal (100%)"))
        out.append(_format_worst_case_line(accuracy_worst, _GAP_TOLERANCE, "final-accuracy",
                                           "ideal (100%)"))
        if recall_worst.passed and precision_worst.passed and accuracy_worst.passed:
            out.append("- Recall, precision, and final accuracy all clear tolerance for the "
                       "worst model — safe to enable drop-to-retrieve.")
        else:
            out.append("- At least one metric misses tolerance for its worst model — keep "
                       "drop-to-retrieve off until this improves.")
    out.append("")
    return "\n".join(out)


def build_fluency_report(results: dict, token_rows: list[dict[str, Any]]) -> str:
    """Render the format-fluency eval: does the model read terse as well as raw JSON?

    `results` is {model: [scored_row,...]} from fluency.run_fluency / score_pack. Each
    row has raw_ok / terse_ok / primer_ok plus qtype/transform. Scoring is PAIRED, so a
    regression (raw right, terse wrong) is a first-class column. The verdict gates on
    the worst model, not the mean (principle #24): a format that helps one model but
    breaks the consumer is a regression, not a wash.
    """
    out: list[str] = ["# terse format-fluency eval", ""]
    out += [
        "Can a model read terse's compressed form as accurately as raw JSON?",
        "Ground truth is deterministic (no LLM judge); scoring is paired per question.",
        "",
    ]
    if not results or not any(results.values()):
        out += [
            "No model answers provided. Configure a backend and re-run:",
            "  - broker pool: set TERSE_FLUENCY_BASE_URL / TERSE_FLUENCY_API_KEY / TERSE_FLUENCY_MODELS",
            "  - real consumer: pass --anthropic (needs the anthropic extra + key)",
            "  - offline: `terse fluency` writes an eval pack you can drive by hand and score later.",
            "",
        ]
        return "\n".join(out)

    if token_rows:
        rt = sum(r["raw_tok"] for r in token_rows if r.get("raw_tok"))
        tt = sum(r["terse_tok"] for r in token_rows if r.get("terse_tok"))
        if rt:
            out += [
                f"Token cost over {len(token_rows)} record-shaped payloads: "
                f"raw {rt} -> terse {tt} ({_pct(tt - rt, rt)}). "
                "Comprehension is the price of that saving — measured below.",
                "",
            ]

    # --- per-model accuracy by form ---
    # Trial count is read from the rows (multi-trial via `--trials`); a `±` column shows
    # the 95% half-width so the verdict is a bound, not a single noisy point. A question
    # "regresses" when raw is fully right across its trials but terse is not.
    trials = max((r.get("trials", 1) for rows in results.values() for r in rows), default=1)
    out += [
        "## Accuracy by model and form",
        "",
        f"Trials per question: **{trials}**. `±` is the 95% half-width of a pooled "
        "binomial bound on the accuracy.",
        "",
        "| Model | q | raw | terse | terse+primer | regressions | primer recovers |",
        "|---|---|---|---|---|---|---|",
    ]
    summary: dict[str, dict[str, float]] = {}
    for model, rows in results.items():
        n = len(rows)
        if not n:
            continue
        racc, rse = _form_stats(rows, "raw_ok")
        tacc, tse = _form_stats(rows, "terse_ok")
        pacc, pse = _form_stats(rows, "primer_ok")
        regr = sum(1 for r in rows if int(r["raw_ok"]) == r.get("trials", 1)
                   and int(r["terse_ok"]) < r.get("trials", 1))
        rec = sum(1 for r in rows if int(r["terse_ok"]) < r.get("trials", 1)
                  and int(r["primer_ok"]) == r.get("trials", 1))
        summary[model] = {"n": n, "raw": racc, "raw_se": rse,
                          "terse": tacc, "terse_se": tse, "primer": pacc, "primer_se": pse}
        out.append(
            f"| `{model}` | {n} | {racc:.0%} ±{_ci(rse) * 100:.0f} | {tacc:.0%} ±{_ci(tse) * 100:.0f} "
            f"| {pacc:.0%} ±{_ci(pse) * 100:.0f} | {regr} | {rec} |"
        )
    out.append("")

    # --- per-transform breakdown (terse form, pooled across models) ---
    by_tf: dict[str, list[dict]] = {}
    for rows in results.values():
        for r in rows:
            by_tf.setdefault(r["transform"], []).append(r)
    if by_tf:
        out += [
            "## terse-form accuracy by stressed transform",
            "",
            "Which transform, if any, costs comprehension. `table+dict` rows resolve a "
            "`~N` alias; `table` rows map a column position to a value.",
            "",
            "| Transform | n | terse | terse+primer |",
            "|---|---|---|---|",
        ]
        for tf, rs in sorted(by_tf.items()):
            tacc, _ = _form_stats(rs, "terse_ok")
            pacc, _ = _form_stats(rs, "primer_ok")
            out.append(f"| {tf} | {len(rs)} | {tacc:.0%} | {pacc:.0%} |")
        out.append("")

    # --- verdict: gate on the worst model ---
    out += ["## Verdict", ""]
    # Raw JSON is the control: a model that can't read RAW (0%) is a backend/config
    # failure (bad model id, refusals), not a terse-comprehension result — exclude it
    # from the gate, but say so, so a broken run can't masquerade as a verdict.
    broken = [m for m, s in summary.items() if s["raw"] == 0]
    gated = {m: s for m, s in summary.items() if s["raw"] > 0}
    if broken:
        out.append(f"- Excluded (raw control failed — backend/config error, not comprehension): "
                   f"{', '.join(f'`{m}`' for m in broken)}.")
    # best terse-side form per model, carrying its own SE for the gap's confidence interval.
    # gap CI: raw and the best form are over the same questions (not independent), so
    # √(se_raw²+se_best²) is a conservative over-estimate of the gap's SE — the honest
    # direction for a bound that gates a ship decision.
    gap_rows = {}
    for model, s in gated.items():
        best, best_se = (s["terse"], s["terse_se"]) if s["terse"] >= s["primer"] \
            else (s["primer"], s["primer_se"])
        gap_rows[model] = (best, best_se, s["raw"], s["raw_se"])
    worst = _worst_case_gap(gap_rows)
    if worst:
        helps = sum(1 for s in gated.values() if s["primer"] > s["terse"] + 1e-9)
        out.append(_format_worst_case_line(worst, _GAP_TOLERANCE, "best terse-form", "raw"))
        if worst.gap_ci > 1e-9 and abs(worst.gap) < worst.gap_ci:
            out.append("- The gap is within its own confidence interval — terse and raw are "
                       "indistinguishable at this trial count (raise `--trials` to tighten).")
        out.append(f"- The primer improves terse-form accuracy for {helps}/{len(gated)} model(s).")
        if worst.passed:
            out.append("- terse's compressed form preserves comprehension within tolerance — "
                       "the proxy's in-place rewrite holds for the tested models.")
        else:
            out.append("- Comprehension regresses beyond tolerance — the proxy's in-place rewrite "
                       "is not safe to ship as-is for the worst model; prefer the primer or restrict "
                       "the policy to the transforms that held.")
    out.append("")
    return "\n".join(out)
