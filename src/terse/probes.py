"""Lossless-ceiling probes (build order B).

These do NOT compress anything. They measure whether the higher-ceiling lossless
levers (Tier 0.5) are worth building, at near-zero cost on the corpus the spike
already captures:

  - value_redundancy: how many tokens are repeated VALUES across cells (beyond the
    repeated KEYS that tabularize already collapses). High -> a dictionary coder
    has real headroom above tabularize. Low -> skip Tier 0.5.
  - cross_call_overlap: token overlap between successive payloads of the same tool
    in an agent loop. High -> structural diffing (delta + reference) compounds in
    the loop. Low -> skip diffing.

Both report UPPER BOUNDS on what the corresponding coder could save (real coders
pay legend/framing overhead), and neither gates the lossless pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .transforms import minify
from .tokenize import count_cl100k, encode_cl100k


def _cell_str(value: Any) -> str:
    """Canonical string for a cell value: bare strings as-is, else minified JSON."""
    return value if isinstance(value, str) else minify(value)


def value_redundancy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Repeated-VALUE mass across all cells of a record list (global dictionary view).

    For each distinct cell value appearing n times with t tokens, (n-1)*t tokens are
    redundant. A dictionary coder replaces each redundant occurrence with a ~1-token
    alias (the first occurrence stays as the legend entry), so the conservative
    saving is (n-1)*(t-1) per repeated value. Reported as an upper bound above what
    tabularize already achieves.
    """
    cells = [_cell_str(v) for rec in records for v in rec.values()]
    counts = Counter(cells)
    tok = {c: (count_cl100k(c) or 0) for c in counts}

    total_value_tokens = sum(tok[c] * n for c, n in counts.items())
    redundant_value_tokens = sum(tok[c] * (n - 1) for c, n in counts.items() if n > 1)
    est_dict_saving = sum(max(0, (tok[c] - 1)) * (n - 1) for c, n in counts.items() if n > 1)

    ratio = (redundant_value_tokens / total_value_tokens) if total_value_tokens else 0.0
    return {
        "cells": len(cells),
        "distinct_values": len(counts),
        "total_value_tokens": total_value_tokens,
        "redundant_value_tokens": redundant_value_tokens,
        "redundancy_ratio": round(ratio, 4),
        "est_dict_saving_tokens": est_dict_saving,
    }


def field_profiles(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-field size + cardinality across a record list, for drop-to-retrieve candidate
    detection (#47). For each field: how often it's present, its distinct-value ratio, the
    mean/max cl100k tokens of its serialized value, and its share of the record list's total
    tokens. A large mean size + near-unique cardinality is the drop-to-retrieve signature:
    lossless value-folding can't help a near-unique field (nothing repeats), so evicting it
    to a retrievable handle is the only lever left. Pure measurement — no thresholds here."""
    present: Counter = Counter()
    toks: dict[str, list[int]] = {}
    uniq: dict[str, set] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            s = _cell_str(v)
            present[k] += 1
            toks.setdefault(k, []).append(count_cl100k(s) or 0)
            uniq.setdefault(k, set()).add(s)
    total = sum(sum(t) for t in toks.values()) or 1
    out: dict[str, dict[str, Any]] = {}
    for k, n in present.items():
        tk = toks[k]
        out[k] = {
            "n": n,
            "distinct": len(uniq[k]),
            "uniq_ratio": round(len(uniq[k]) / n, 4) if n else 0.0,
            "mean_tok": round(sum(tk) / len(tk), 1) if tk else 0.0,
            "max_tok": max(tk) if tk else 0,
            "tok_share": round(sum(tk) / total, 4),
        }
    return out


def cross_call_overlap(prev_raw: str, curr_raw: str) -> dict[str, Any]:
    """Token overlap of a payload with the previous same-tool payload (multiset).

    shared = sum of per-token-id minimums between the two token streams. overlap_ratio
    = shared / tokens(curr) is the fraction of the current payload already present in
    the prior one — an upper bound on what a delta-against-prev encoding could save.
    """
    prev_ids = encode_cl100k(prev_raw)
    curr_ids = encode_cl100k(curr_raw)
    if prev_ids is None or curr_ids is None:
        return {"available": False}
    prev_c, curr_c = Counter(prev_ids), Counter(curr_ids)
    shared = sum((prev_c & curr_c).values())
    curr_n = len(curr_ids)
    return {
        "available": True,
        "prev_tokens": len(prev_ids),
        "curr_tokens": curr_n,
        "shared_tokens": shared,
        "overlap_ratio": round(shared / curr_n, 4) if curr_n else 0.0,
        "est_delta_saving_tokens": shared,
    }
