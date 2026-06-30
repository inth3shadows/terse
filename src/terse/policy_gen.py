"""Auto-author a conservative, lossless per-tool policy from a measured corpus (#24).

The maintenance/adoption blocker for terse is that `policy.json` is hand-written: add
a server, see no compression, then manually capture → measure → read the report → edit
JSON. `capture.py`/`measure.py` already produce per-tool, per-tier token savings — this
turns that measurement into a policy directly.

The decision is deliberately CONSERVATIVE and 100% lossless (it only ever enables the
round-trip-gated Tier-0/0.5 tiers, never a lossy mode):

  For each tool, aggregate its payloads' per-tier cl100k savings, then:
    1. any non-JSON payload OR any round-trip failure  -> passthrough  (never compress
       a tool we can't losslessly handle)
    2. total savings %  < threshold                    -> passthrough  (transform cost
       not worth a marginal gain)
    3. otherwise  ["minify","tabularize"] (+ "dictionary" iff its MARGINAL saving clears
       the threshold — mirrors the hand-authored example dropping dictionary on `kb.*`).

The output is a policy DOC (the same shape `policy.load_policy` parses) plus a row per
tool explaining the decision. Pure: no I/O, so the generator is unit-testable and the
CLI owns reading the corpus and writing the file.

It does NOT call a model. The #24 thesis — "prove the model still reads the compressed
form" — stays a separate, explicit `terse fluency` step: gating policy authoring on a
model call would couple a key/latency into what is otherwise a deterministic, lossless
transform. The CLI surfaces that pointer; it is not a hard gate here.
"""

from __future__ import annotations

from typing import Any

from .measure import measure_payload

# Tiers are cumulative in the codec (minify ⊂ tabularize ⊂ dictionary). minify is implied
# by re-serialization, so it never ships alone — tabularize always carries it.
_BASE_TIERS = ["minify", "tabularize"]


def _pct(saved: int, raw: int) -> float:
    """Savings as a percent of the tool's raw token volume (0.0 when raw is 0)."""
    return (saved / raw * 100.0) if raw else 0.0


def _tool_decision(tool: str, raws: list[str], threshold: float) -> dict[str, Any]:
    """Aggregate one tool's payloads and decide its tiers. Returns a summary row with
    the chosen `tiers`, the measured savings, and a human-readable `reason`."""
    rows = [measure_payload(r) for r in raws]
    n = len(rows)

    # A single non-JSON payload or round-trip failure disqualifies the whole tool: the
    # policy matches on tool name, so we can't enable a tier for "most" of its results.
    non_json = sum(1 for r in rows if not r["applicable"])
    gate_fail = sum(1 for r in rows if not r["roundtrip_ok"])

    raw_tok = sum(r["cl100k"]["raw"] or 0 for r in rows)
    minify = sum(r["saved_cl100k"]["minify"] or 0 for r in rows)
    tabularize = sum(r["saved_cl100k"]["tabularize"] or 0 for r in rows)
    dictionary = sum(r["saved_cl100k"]["dictionary"] or 0 for r in rows)
    total = sum(r["saved_cl100k"]["tier_total"] or 0 for r in rows)
    total_pct = _pct(total, raw_tok)
    dict_pct = _pct(dictionary, raw_tok)

    base = {
        "tool": tool, "n": n, "raw_tok": raw_tok,
        "saved_pct": round(total_pct, 1), "dict_pct": round(dict_pct, 1),
        "minify": minify, "tabularize": tabularize, "dictionary": dictionary,
    }

    if non_json or gate_fail:
        why = []
        if non_json:
            why.append(f"{non_json}/{n} non-JSON")
        if gate_fail:
            why.append(f"{gate_fail}/{n} failed round-trip")
        return {**base, "tiers": [], "reason": f"passthrough — {', '.join(why)}"}

    if total_pct < threshold:
        return {**base, "tiers": [],
                "reason": f"passthrough — {total_pct:.1f}% < {threshold:.1f}% threshold"}

    tiers = list(_BASE_TIERS)
    if dict_pct >= threshold:
        tiers.append("dictionary")
        reason = f"{total_pct:.1f}% saved (dictionary +{dict_pct:.1f}%)"
    else:
        reason = f"{total_pct:.1f}% saved (dictionary +{dict_pct:.1f}% below threshold — dropped)"
    return {**base, "tiers": tiers, "reason": reason}


def generate_policy(
    envelopes: list[dict[str, Any]], threshold: float = 5.0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Author a conservative lossless policy from captured payloads.

    `threshold` is the minimum total savings (percent of raw tokens) required to compress
    a tool at all, and the minimum marginal saving to add the dictionary tier on top of
    minify+tabularize. Returns `(policy_doc, rows)` — `policy_doc` is loadable by
    `policy.load_policy`; `rows` is one decision summary per tool (sorted by savings desc)
    for a report. Tools are emitted in the same order so the JSON is deterministic."""
    by_tool: dict[str, list[str]] = {}
    for env in envelopes:
        by_tool.setdefault(env.get("tool", "?"), []).append(env["raw"])

    rows = [_tool_decision(tool, raws, threshold) for tool, raws in by_tool.items()]
    # Highest-savings tools first: makes the policy and the report read top-down by value,
    # and keeps output stable regardless of corpus file ordering.
    rows.sort(key=lambda r: (-r["saved_pct"], r["tool"]))

    policies = []
    for r in rows:
        comment = f"{r['n']} payload(s), {r['reason']}"
        policies.append({"_comment": comment, "match": {"tool": r["tool"]}, "tiers": r["tiers"]})

    doc = {
        "version": 1,
        "_comment": (f"Auto-generated by `terse policy generate` (threshold {threshold:.1f}%). "
                     f"Conservative + lossless: a tier is enabled only where measured savings "
                     f"clear the threshold and every payload round-trips. Verify the model still "
                     f"reads the compressed form with `terse fluency --corpus <dir>` before relying "
                     f"on it."),
        "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
        "policies": policies,
    }
    return doc, rows
