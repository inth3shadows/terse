"""Lossless-ceiling probes (build order B).

These do NOT compress anything. They measure whether the higher-ceiling lossless
levers of Tier 0.5 are worth building, at near-zero cost on the corpus already
captured:

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

import math
import re
from collections import Counter
from itertools import combinations
from typing import Any

from .tokenize import count_cl100k, encode_cl100k
from .transforms import minify

# Reverse-map a captured tool name to its origin server. The capture envelope stores
# Legacy fallback only (#158): a pre-#156 corpus records no `server`, so identity has to
# be inferred from the tool name — the corpus was captured by separate per-server proxies
# sharing one --capture-dir. kb/codegraph carry a delimiter prefix; runecho's tools are
# bare verbs, so they need an explicit set. This list silently goes stale every time
# runecho gains a tool, which is exactly why an envelope that STATES its server no longer
# consults it — see `server_of_tool`.
_RUNECHO_TOOLS = {"locate", "structure", "status", "health", "hash", "diff"}


def server_of_tool(tool: str, server: str | None = None) -> str:
    """Origin server for a captured tool. Since #156 the envelope records `server`
    straight from the wrap, so pass it and it is returned verbatim — the truth, not a
    guess. The name-based heuristic below is the fallback for legacy envelopes that record
    no server (an empty string is treated as none, so "unknown" has one spelling)."""
    if isinstance(server, str) and server:
        return server
    if tool.startswith("kb."):
        return "kb"
    if tool.startswith("codegraph"):
        return "codegraph"
    if tool in _RUNECHO_TOOLS:
        return "runecho"
    head = re.split(r"[._]", tool, maxsplit=1)[0]
    return head or "other"


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


def token_idf(raws: list[str]) -> dict[int, float]:
    """Inverse document frequency per cl100k token id over a payload set.

    idf[t] = log(N / df_t) where df_t = payloads containing t. A token in EVERY payload
    (JSON framing: `{`, `"`, `:`, ubiquitous keys) → idf 0; a token in one payload → log N.
    Weighting a shared-token overlap by idf nets out the framing floor two disjoint JSON
    payloads share anyway, leaving the CONTENT overlap a dictionary coder could actually
    harvest across servers (#64 Phase 0 — Lever B, framing-normalized)."""
    df: Counter = Counter()
    n = 0
    for raw in raws:
        ids = encode_cl100k(raw)
        if ids is None:
            continue
        n += 1
        df.update(set(ids))
    return {t: math.log(n / c) for t, c in df.items()} if n else {}


def cross_call_overlap(
    prev_raw: str, curr_raw: str, idf: dict[int, float] | None = None
) -> dict[str, Any]:
    """Token overlap of a payload with the previous same-tool payload (multiset).

    shared = sum of per-token-id minimums between the two token streams. overlap_ratio
    = shared / tokens(curr) is the fraction of the current payload already present in
    the prior one — an upper bound on what a delta-against-prev encoding could save.

    When `idf` is supplied (from `token_idf`), also reports the framing-normalized
    `content_overlap_ratio`: the shared/current masses re-weighted by idf, so ubiquitous
    framing tokens drop out and only shared CONTENT counts. This is the trustworthy
    cross-server signal — raw overlap_ratio is inflated by shared JSON structure.
    """
    prev_ids = encode_cl100k(prev_raw)
    curr_ids = encode_cl100k(curr_raw)
    if prev_ids is None or curr_ids is None:
        return {"available": False}
    prev_c, curr_c = Counter(prev_ids), Counter(curr_ids)
    shared_ms = prev_c & curr_c
    shared = sum(shared_ms.values())
    curr_n = len(curr_ids)
    result = {
        "available": True,
        "prev_tokens": len(prev_ids),
        "curr_tokens": curr_n,
        "shared_tokens": shared,
        "overlap_ratio": round(shared / curr_n, 4) if curr_n else 0.0,
        "est_delta_saving_tokens": shared,
    }
    if idf is not None:
        shared_wt = sum(n * idf.get(t, 0.0) for t, n in shared_ms.items())
        curr_wt = sum(n * idf.get(t, 0.0) for t, n in curr_c.items())
        result["content_shared_weight"] = round(shared_wt, 1)
        result["content_overlap_ratio"] = round(shared_wt / curr_wt, 4) if curr_wt else 0.0
    return result


def cross_server_redundancy(records_by_server: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Does a shared cross-peer dictionary beat independent per-peer dictionaries? (#64 Phase 0)

    A gateway's one compression lever a native client multiplex can't match is a legend
    shared ACROSS servers. This isolates that increment:

      per_peer = Σ_server value_redundancy(that server's records).est_dict_saving  — what
                 today's per-peer compression already folds (each Interceptor in isolation).
      pooled   = value_redundancy(all servers' records concatenated).est_dict_saving  — what
                 one shared legend spanning every peer folds.
      increment = pooled − per_peer  — attributable to values repeated across ≥2 DISTINCT
                 servers (a within-peer repeat is already counted in that peer's per_peer term;
                 pooling a value present in k peers turns Σ(n_p−1) into (Σn_p−1), and the
                 difference is exactly the cross-peer occurrences).

    All figures are the same UPPER BOUND `value_redundancy` reports (first occurrence kept as
    legend). This is the primary #64 gate — it works on extracted record CELL VALUES, so it
    measures shared content, not the JSON framing that inflates raw-token overlap.
    """
    per_server: list[dict[str, Any]] = []
    for srv in sorted(records_by_server):
        recs = records_by_server[srv]
        vr = value_redundancy(recs)
        per_server.append({"server": srv, "record_lists_folded": len(recs), **vr})

    per_peer_saving = sum(r["est_dict_saving_tokens"] for r in per_server)
    all_records = [rec for srv in sorted(records_by_server) for rec in records_by_server[srv]]
    pooled = value_redundancy(all_records)
    increment = pooled["est_dict_saving_tokens"] - per_peer_saving
    corpus_tok = pooled["total_value_tokens"]

    return {
        "per_server": per_server,
        "per_peer_saving_tokens": per_peer_saving,
        "pooled_saving_tokens": pooled["est_dict_saving_tokens"],
        "pooled_total_value_tokens": corpus_tok,
        "cross_server_increment_tokens": increment,
        "increment_frac_of_corpus": round(increment / corpus_tok, 4) if corpus_tok else 0.0,
        "increment_frac_over_per_peer": round(increment / per_peer_saving, 4) if per_peer_saving else 0.0,
    }


def cross_server_overlap(
    raws_by_server: dict[str, list[tuple[str, str]]], cap_per_pair: int = 20
) -> dict[str, Any]:
    """Token overlap between payloads of DIFFERENT servers, framing-normalized (#64 Phase 0).

    For each unordered server pair, deterministically take the first `cap_per_pair` payloads
    of each (sorted by sha) and run `cross_call_overlap` positionally, weighted by a
    corpus-wide idf (`token_idf`). Reports two medians:
      - `median_overlap`         — RAW token overlap, inflated by shared JSON framing.
      - `median_content_overlap` — idf-weighted, framing netted out. THIS is the signal:
        it is the fraction of a payload's content mass that recurs in a different server —
        the headroom a shared cross-peer dictionary (but not per-peer coding) could harvest.
    Works on any payload shape, so unlike the record-value lever it spans text/source servers.
    """
    all_raws = [raw for lst in raws_by_server.values() for _, raw in lst]
    idf = token_idf(all_raws)

    rows: list[dict[str, Any]] = []
    capped = False
    for a, b in combinations(sorted(raws_by_server), 2):
        la, lb = raws_by_server[a], raws_by_server[b]
        capped = capped or len(la) > cap_per_pair or len(lb) > cap_per_pair
        sa = sorted(la, key=lambda t: t[0])[:cap_per_pair]
        sb = sorted(lb, key=lambda t: t[0])[:cap_per_pair]
        for (sha_a, raw_a), (sha_b, raw_b) in zip(sa, sb, strict=False):  # pair up to shorter
            res = cross_call_overlap(raw_a, raw_b, idf=idf)
            if res.get("available"):
                rows.append({"server_a": a, "server_b": b, "sha_a": sha_a, "sha_b": sha_b, **res})

    def _median(key: str) -> float:
        vals = sorted(r[key] for r in rows if key in r)
        return vals[len(vals) // 2] if vals else 0.0

    return {"rows": rows, "pairs": len(rows),
            "median_overlap": _median("overlap_ratio"),
            "median_content_overlap": _median("content_overlap_ratio"),
            "capped": capped, "cap_per_pair": cap_per_pair}
