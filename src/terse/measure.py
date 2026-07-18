"""Per-payload token measurement with a clean per-tier decomposition.

The decomposition is the point. For a JSON payload the model would otherwise see
as `raw`:

    minify_saved     = tokens(raw)        - tokens(minify(obj))     # whitespace + \\uXXXX
    tabularize_saved = tokens(minify(obj)) - tokens(compress(obj))  # repeated keys folded
    tier0_saved      = tokens(raw)        - tokens(compress(obj))   # = minify + tabularize

So a payload that arrives already-compact shows minify_saved ~ 0 (the headroom
no-op, made visible), and a payload with no record arrays shows tabularize_saved
~ 0. Negatives are real and reported, not clamped: at tiny N the table envelope
can cost more than the keys it folds.

Every measurement re-runs the lossless gate; a False here invalidates the row's
savings (you cannot bank tokens you lost data to).
"""

from __future__ import annotations

import json
from typing import Any

from . import transforms
from .capture import classify_shape
from .tokenize import CL100K, O200K, count, count_cl100k


def measure_payload(raw: str) -> dict[str, Any]:
    """Measure one raw payload: shape, gate, and per-tier cl100k token deltas."""
    shape = classify_shape(raw)
    applicable = True
    obj: Any = None
    try:
        obj = json.loads(raw)
        # Depth guard (#79): past the codec-wide cap the transforms would RecursionError,
        # and the proxy passes such a payload through anyway — report it the same way,
        # as not-applicable with zero savings, instead of crashing `terse measure`.
        if transforms.exceeds_depth(obj):
            applicable = False
    except (json.JSONDecodeError, TypeError, RecursionError):
        applicable = False
    if not applicable:
        # Non-JSON (long-text / other) or too-deep: Tier-0 is a pass-through. Lossless
        # trivially; any real saving here would come from the (unbuilt, opt-in) lossy tier.
        raw_tok = count_cl100k(raw)
        row: dict[str, Any] = {
            "shape": shape,
            "applicable": False,
            "roundtrip_ok": True,
            "cl100k": {"raw": raw_tok, "minified": raw_tok, "tabular": raw_tok, "compressed": raw_tok},
            "saved_cl100k": {"minify": 0, "tabularize": 0, "dictionary": 0, "tier_total": 0},
        }
        return row

    minified = transforms.minify(obj)
    tabular = transforms.compress_tabular(obj)  # Tier-0 only (minify + tabularize)
    compressed = transforms.compress(obj)       # + Tier-0.5 dictionary coding
    gate = transforms.roundtrip_ok(obj)

    raw_tok = count_cl100k(raw)
    min_tok = count_cl100k(minified)
    tab_tok = count_cl100k(tabular)
    cmp_tok = count_cl100k(compressed)

    def _saved(a: int | None, b: int | None) -> int | None:
        return None if a is None or b is None else a - b

    # "You cannot bank tokens you lost data to" (module docstring). If the lossless
    # gate FAILED, the compressed form is not a legal substitute for raw, so its token
    # deltas must not count toward any headline reduction — a downstream aggregator that
    # forgot to filter on roundtrip_ok would otherwise inflate the reported % with
    # savings from a payload the proxy would never actually emit. Zero the banked
    # savings here at the source; the raw cl100k counts stay for transparency.
    saved = {
        "minify": _saved(raw_tok, min_tok),
        "tabularize": _saved(min_tok, tab_tok),
        "dictionary": _saved(tab_tok, cmp_tok),
        "tier_total": _saved(raw_tok, cmp_tok),
    }
    if not gate:
        saved = dict.fromkeys(saved, 0)

    return {
        "shape": shape,
        "applicable": True,
        "roundtrip_ok": gate,
        "cl100k": {"raw": raw_tok, "minified": min_tok, "tabular": tab_tok, "compressed": cmp_tok},
        "saved_cl100k": saved,
    }


def cross_tokenizer_savings(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-tool Tier-0 savings under two different BPE vocabularies (cl100k, o200k).

    Stability of the savings % across two very different tokenizers is the keyless
    evidence that the structural savings hold for Claude's (unpublished) tokenizer:
    folding whitespace/keys/values removes tokens regardless of vocabulary.
    """
    out = []
    for env in envelopes:
        raw = env["raw"]
        try:
            comp = transforms.compress(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            comp = raw
        row: dict[str, Any] = {"tool": env.get("tool", "?")}
        for enc in (CL100K, O200K):
            r, c = count(raw, enc), count(comp, enc)
            pct = ((r - c) / r * 100) if (r and c is not None) else None
            row[enc] = {"raw": r, "compressed": c, "pct": pct}
        out.append(row)
    return out


def measure_corpus(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure every captured payload; attach tool/sha for traceability in the report."""
    rows = []
    for env in envelopes:
        row = measure_payload(env["raw"])
        row["tool"] = env.get("tool", "?")
        row["sha"] = env.get("sha", "?")
        row["bytes"] = env.get("bytes", len(env["raw"]))
        rows.append(row)
    return rows
