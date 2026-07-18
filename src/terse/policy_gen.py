"""Auto-author a conservative, lossless per-tool policy from a measured corpus (#24).

The maintenance/adoption blocker for terse is that `policy.json` is hand-written: add
a server, see no compression, then manually capture → measure → read the report → edit
JSON. `capture.py`/`measure.py` already produce per-tool, per-tier token savings — this
turns that measurement into a policy directly.

The tier decision is deliberately CONSERVATIVE and 100% lossless (it only ever enables the
round-trip-gated Tier-0/0.5 tiers, never a lossy mode):

  For each tool, aggregate its payloads' per-tier cl100k savings, then:
    1. any non-JSON payload OR any round-trip failure  -> passthrough  (never compress
       a tool we can't losslessly handle)
    2. total savings %  < threshold                    -> passthrough  (transform cost
       not worth a marginal gain)
    3. otherwise  ["minify","tabularize"] (+ "dictionary" iff its MARGINAL saving clears
       the threshold — mirrors the hand-authored example dropping dictionary on `kb.*`).

Separately, it SUGGESTS drop-to-retrieve candidates (#47) — fields that are large AND
near-unique, where the lossless tiers are structurally powerless — nothing repeats to fold —
but a huge, rarely-needed value dominates the record. The measured example: a kb
`embedding` field, 77% of tokens, all unique — lossless gets +4%, dropping it gets +77%.
These are emitted as an INACTIVE `_suggested_fields` block the operator opts into by
renaming to `fields`: drop is lossy, so the generator never enables it automatically — it
only surfaces the opportunity the operator would otherwise miss.

The output is a policy DOC (the same shape `policy.load_policy` parses) plus a row per
tool explaining the decision. Pure: no I/O, so the generator is unit-testable and the
CLI owns reading the corpus and writing the file.

It does NOT call a model. The #24 thesis — "prove the model still reads the compressed
form" — stays a separate, explicit `terse fluency` step: gating policy authoring on a
model call would couple a key/latency into what is otherwise a deterministic, lossless
transform. The CLI surfaces that pointer; it is not a hard gate here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import probes
from .capture import find_record_list_with_path
from .measure import measure_payload

# Tiers are cumulative in the codec (minify ⊂ tabularize ⊂ dictionary). minify is implied
# by re-serialization, so it never ships alone — tabularize always carries it.
_BASE_TIERS = ["minify", "tabularize"]

# Field-role classification steers drop-to-retrieve suggestions away from load-bearing
# fields (the step-1 finding: the size+uniqueness heuristic alone confidently suggested
# dropping kb's `principle` — the very field the model reasons over — because it was the
# biggest unique blob). `identity` = a key/essence field the record needs IN-LINE (never a
# drop candidate); `prose` = supporting free text, the safe candidate (ranked first);
# `unknown` = the name doesn't reveal the role, so it MAY be load-bearing → still suggested
# but flagged for the dropeval gate. A name-only heuristic cannot catch a DOMAIN essence
# field (`principle`, `verdict`, `answer`) — those fall to `unknown` by design, which is
# why the behavioral dropeval gate, not this classifier, is the real safety net.
_IDENTITY_TOKENS = frozenset({
    "id", "ids", "key", "keys", "name", "title", "slug", "uuid", "guid", "hash", "type",
    "kind", "status", "state", "path", "url", "uri", "command", "cmd", "version", "ref",
    "sha", "label", "tag", "count", "size", "timestamp", "date", "time"})
_PROSE_TOKENS = frozenset({
    "evidence", "rationale", "note", "notes", "description", "desc", "summary", "detail",
    "details", "body", "text", "content", "comment", "comments", "explanation", "context",
    "snippet", "message", "msg", "reason", "readme", "abstract", "excerpt", "bio", "blurb"})


def _field_tokens(name: str) -> set[str]:
    """Lowercase word tokens of a field path's leaf key (drop the record path and `[]`,
    split camelCase + snake/space/hyphen). `result[].bodyText` -> {'body', 'text'}."""
    leaf = name.rsplit(".", 1)[-1].replace("[]", "")
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", leaf)
    return {t for t in re.split(r"[\s_\-]+", spaced.lower()) if t}


def classify_field_role(name: str) -> str:
    """Best-effort role of a record field from its NAME: 'identity' | 'prose' | 'unknown'.
    See _IDENTITY_TOKENS / _PROSE_TOKENS. Identity wins over prose on a tie (a field that is
    both a key and prose-shaped is treated as load-bearing — the safer call)."""
    toks = _field_tokens(name)
    if toks & _IDENTITY_TOKENS:
        return "identity"
    if toks & _PROSE_TOKENS:
        return "prose"
    return "unknown"


# drop-to-retrieve candidate thresholds (#47). A field is suggested only when all three hold:
_DROP_MIN_MEAN_TOK = 50.0    # large: ~200 serialized chars, matching lossy.DEFAULT_DROP_MIN
_DROP_MIN_UNIQ_RATIO = 0.9   # near-unique: the dictionary tier can't fold it, so lossless is out
_DROP_MIN_SHARE = 0.10       # worth a retrieve round-trip: >=10% of the record list's tokens


def activate_suggestions(doc: dict) -> dict:
    """Return a deep COPY of a generated policy doc with every entry's INACTIVE
    `_suggested_fields` promoted to active `fields` (merged over any existing fields), and
    the `_suggested_fields_note` dropped. Used by `terse tune --drop-eval` to verify the
    suggested drops AS IF enabled, without mutating the doc written to disk (which stays
    inactive until the operator opts in). Pure — no I/O."""
    import copy

    out = copy.deepcopy(doc)
    for entry in out.get("policies", []):
        sug = entry.pop("_suggested_fields", None)
        entry.pop("_suggested_fields_note", None)
        if sug:
            entry["fields"] = {**entry.get("fields", {}), **sug}
    return out


def _drop_candidates(
    raws: list[str],
    min_mean_tok: float = _DROP_MIN_MEAN_TOK,
    min_uniq_ratio: float = _DROP_MIN_UNIQ_RATIO,
    min_share: float = _DROP_MIN_SHARE,
) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    """Suggest drop-to-retrieve field paths for one tool from its payloads. Pools records
    across payloads that share the first payload's record path (per-tool shape is assumed
    consistent) and flags fields that are large + near-unique + a meaningful token share.
    Returns `(suggestion, rows)`: `suggestion` is an INACTIVE `{path: {"lossy": ...}}` block
    the operator opts into; `rows` are per-field stats for the report. Empty when a tool has
    no record list or no field clears every threshold.

    Each payload is profiled INDEPENDENTLY and the metrics are averaged: cardinality is a
    within-payload property (the drop runs per result at runtime), so pooling records across
    payloads would wrongly halve `uniq_ratio` when the same result is captured twice."""
    path: str | None = None
    per_payload: list[dict[str, dict[str, Any]]] = []
    for raw in raws:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        records, p = find_record_list_with_path(obj)
        if records is None or p is None:
            continue
        if path is None:
            path = p
        if p == path:
            per_payload.append(probes.field_profiles(records))
    if path is None or not per_payload:
        return {}, []

    def _avg(field: str, key: str) -> float:
        vals = [pp[field][key] for pp in per_payload if field in pp]
        return sum(vals) / len(vals) if vals else 0.0

    fields = {f for pp in per_payload for f in pp}
    agg = {f: {"n": sum(pp[f]["n"] for pp in per_payload if f in pp),
               "distinct": sum(pp[f]["distinct"] for pp in per_payload if f in pp),
               "uniq_ratio": round(_avg(f, "uniq_ratio"), 4),
               "mean_tok": round(_avg(f, "mean_tok"), 1),
               "max_tok": max(pp[f]["max_tok"] for pp in per_payload if f in pp),
               "tok_share": round(_avg(f, "tok_share"), 4)} for f in fields}

    suggestion: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    # Order safe-first, then by impact: prose (known-safe) fields lead, unknown-role fields
    # (may be load-bearing) follow, each by descending token-share. `identity` fields are
    # dropped from the candidate list entirely below — the record needs its keys in-line.
    _role_rank = {"prose": 0, "unknown": 1}
    ordered = sorted(
        agg.items(),
        key=lambda kv: (_role_rank.get(classify_field_role(f"{path}.{kv[0]}"), 1),
                        -kv[1]["tok_share"]),
    )
    for field, pr in ordered:
        role = classify_field_role(f"{path}.{field}")
        if role == "identity":
            continue  # a key/essence field is never a drop candidate — the record needs it
        if (pr["mean_tok"] >= min_mean_tok and pr["uniq_ratio"] >= min_uniq_ratio
                and pr["tok_share"] >= min_share):
            fpath = f"{path}.{field}"
            suggestion[fpath] = {"lossy": "drop-to-retrieve"}
            rows.append({"path": fpath, "role": role, **pr})
    return suggestion, rows


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

    # Drop-to-retrieve candidates are detected INDEPENDENTLY of the lossless-tier decision:
    # the highest-value case (kb `embedding`) is a tool whose lossless savings fall BELOW the
    # threshold — passthrough for tiers — yet is dominated by a huge unique field only drop
    # can shrink. So compute it here and carry it through every return path.
    drop_suggestion, drop_rows = _drop_candidates(raws)

    base = {
        "tool": tool, "n": n, "raw_tok": raw_tok,
        "saved_pct": round(total_pct, 1), "dict_pct": round(dict_pct, 1),
        "minify": minify, "tabularize": tabularize, "dictionary": dictionary,
        "drop_suggestion": drop_suggestion, "drop_rows": drop_rows,
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
        entry: dict[str, Any] = {"_comment": comment, "match": {"tool": r["tool"]},
                                 "tiers": r["tiers"]}
        # Drop-to-retrieve suggestions ride along INACTIVE: the loader reads `fields`, not
        # `_suggested_fields`, so this is a no-op until the operator renames it. Drop is
        # lossy — the human confirms; the generator never enables it.
        if r.get("drop_suggestion"):
            shares = ", ".join(
                f"{dr['path']} ~{dr['tok_share']*100:.0f}% [{dr.get('role', 'unknown')}]"
                for dr in r["drop_rows"])
            has_unknown = any(dr.get("role") == "unknown" for dr in r["drop_rows"])
            caution = (" A field tagged [unknown] may be LOAD-BEARING — the name doesn't "
                       "reveal its role, so dropping it forces the model to call retrieve for "
                       "it; treat [prose] as safer and verify any [unknown] before enabling."
                       if has_unknown else "")
            entry["_suggested_fields"] = r["drop_suggestion"]
            entry["_suggested_fields_note"] = (
                f"LOSSY drop-to-retrieve candidates (large + near-unique; safe-first, [role] "
                f"guessed from the field name): {shares}. Rename '_suggested_fields' -> "
                f"'fields' to enable, then confirm the model still answers with `terse fluency "
                f"--drop-eval`. Off until you do.{caution}")
        policies.append(entry)

    any_drops = any(r.get("drop_suggestion") for r in rows)
    doc = {
        "version": 1,
        "_comment": (f"Auto-generated by `terse policy generate` (threshold {threshold:.1f}%). "
                     f"Conservative + lossless: a tier is enabled only where measured savings "
                     f"clear the threshold and every payload round-trips. Verify the model still "
                     f"reads the compressed form with `terse fluency --corpus <dir>` before relying "
                     f"on it."
                     + (" Some tools carry INACTIVE `_suggested_fields` (lossy drop-to-retrieve "
                        "candidates) — opt in by renaming to `fields`." if any_drops else "")),
        "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
        "policies": policies,
    }
    return doc, rows
