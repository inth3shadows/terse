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
from typing import Any, Optional

from . import transforms
from .capture import classify_shape
from .tokenize import CL100K, O200K, count, count_anthropic, count_cl100k


def measure_payload(raw: str, use_anthropic: bool = False) -> dict[str, Any]:
    """Measure one raw payload: shape, gate, and per-tier cl100k (+ optional Anthropic)."""
    shape = classify_shape(raw)
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Non-JSON (long-text / other): Tier-0 is a pass-through. Lossless trivially;
        # any real saving here would come from the (unbuilt, opt-in) lossy tier.
        raw_tok = count_cl100k(raw)
        row: dict[str, Any] = {
            "shape": shape,
            "applicable": False,
            "roundtrip_ok": True,
            "cl100k": {"raw": raw_tok, "minified": raw_tok, "tabular": raw_tok, "compressed": raw_tok},
            "saved_cl100k": {"minify": 0, "tabularize": 0, "dictionary": 0, "tier_total": 0},
        }
        if use_anthropic:
            a = count_anthropic(raw)
            row["anthropic"] = {"raw": a, "compressed": a, "saved": 0}
        return row

    minified = transforms.minify(obj)
    tabular = transforms.compress_tabular(obj)  # Tier-0 only (minify + tabularize)
    compressed = transforms.compress(obj)       # + Tier-0.5 dictionary coding
    gate = transforms.roundtrip_ok(obj)

    raw_tok = count_cl100k(raw)
    min_tok = count_cl100k(minified)
    tab_tok = count_cl100k(tabular)
    cmp_tok = count_cl100k(compressed)

    def _saved(a: Optional[int], b: Optional[int]) -> Optional[int]:
        return None if a is None or b is None else a - b

    row = {
        "shape": shape,
        "applicable": True,
        "roundtrip_ok": gate,
        "cl100k": {"raw": raw_tok, "minified": min_tok, "tabular": tab_tok, "compressed": cmp_tok},
        "saved_cl100k": {
            "minify": _saved(raw_tok, min_tok),
            "tabularize": _saved(min_tok, tab_tok),
            "dictionary": _saved(tab_tok, cmp_tok),
            "tier_total": _saved(raw_tok, cmp_tok),
        },
    }
    if use_anthropic:
        a_raw = count_anthropic(raw)
        a_cmp = count_anthropic(compressed)
        row["anthropic"] = {"raw": a_raw, "compressed": a_cmp, "saved": _saved(a_raw, a_cmp)}
    return row


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


def measure_corpus(
    envelopes: list[dict[str, Any]], use_anthropic: bool = False
) -> list[dict[str, Any]]:
    """Measure every captured payload; attach tool/sha for traceability in the report."""
    rows = []
    for env in envelopes:
        row = measure_payload(env["raw"], use_anthropic=use_anthropic)
        row["tool"] = env.get("tool", "?")
        row["sha"] = env.get("sha", "?")
        row["bytes"] = env.get("bytes", len(env["raw"]))
        rows.append(row)
    return rows


def measure_session_dict(
    envelopes: list[dict[str, Any]], keyframe: int = 5
) -> dict[str, Any]:
    """Replay the corpus through ONE shared session dictionary (#64) IN CAPTURE ORDER and
    report the token saving over the per-call baseline.

    The baseline is `compress()` — today's default, a fresh per-call legend every payload.
    The session run threads a single `SessionDict` across every payload (as the live
    multiproxy shares one across all peers), so a value first defined by an earlier payload
    is REFERENCED (definition elided) by later ones — the cross-call/cross-peer win a
    per-call legend structurally can't capture. Capture order matters: the saving only
    accrues to payloads that follow the one that first defined a shared value.

    Non-JSON payloads (no `compress` structure) contribute equally to both totals, so they
    neither help nor distort the delta. Every session payload is self-verified inside
    `sess_encode`; a payload that fails falls back to the full form, exactly as live.

    Returns totals + the realized percentage. A saving materially above what per-call
    compression already banks is the evidence Phase 1 set out to produce."""
    sess = transforms.SessionDict()
    base_tok = 0
    sess_tok = 0
    applicable = 0  # JSON payloads the session path could actually act on
    elided = 0      # payloads where the session form beat the per-call full form
    for env in envelopes:
        raw = env["raw"]
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Non-JSON: identical under both paths; count once into each so the delta is
            # driven purely by JSON payloads.
            tok = count_cl100k(raw) or 0
            base_tok += tok
            sess_tok += tok
            continue
        full = transforms.compress(obj)
        full_tok = count_cl100k(full) or 0
        base_tok += full_tok
        applicable += 1
        res = transforms.sess_compress(obj, sess, keyframe)
        if res is None:
            sess_tok += full_tok
            continue
        envelope, _ = res
        env_tok = count_cl100k(envelope) or 0
        sess_tok += env_tok
        if env_tok < full_tok:
            elided += 1
    saved = base_tok - sess_tok
    pct = (saved / base_tok * 100) if base_tok else 0.0
    return {
        "payloads": len(envelopes),
        "applicable": applicable,
        "elided_wins": elided,
        "baseline_cl100k": base_tok,
        "session_cl100k": sess_tok,
        "saved_cl100k": saved,
        "saved_pct": pct,
        "keyframe": keyframe,
    }
