#!/usr/bin/env python3
"""Supplementary, terse-ONLY measurement: the cross-call diff tier.

A stateless encoding (TOON, minify, terse's own single-shot codec) must resend the whole
payload every call. terse's diff tier, when the SAME tool is called repeatedly, emits a
lossless delta against the prior result instead — the agent-loop pattern (a list polled or
re-listed between calls, changing by a few records). No competitor here has an equivalent,
so this isn't a head-to-head; it's the extra axis, quantified on the same real corpus.

For every list-shaped payload we model ONE realistic repeated call: the base result, then a
`curr` with two records' fields changed and one new record appended (a poll-again delta).
We report the SECOND call's cost as a lossless terse diff vs a full terse re-send. Stateless
encoders have no second-call form — they pay the full column every call, forever.

Run: uv run scripts/bench/diff_demo.py   (add --json for machine-readable)
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from terse import transforms
from terse.tokenize import count_cl100k

CORPUS = Path(__file__).resolve().parent / "corpus"
_ID_PREFERENCE = ("id", "sha", "number", "node_id", "name")


def _id_col(records: list[dict]) -> str | None:
    """A column whose values are unique + hashable across records — the diff's stable key."""
    keys = set(records[0])
    for cand in _ID_PREFERENCE:
        if cand in keys and _unique_scalar(records, cand):
            return cand
    for cand in records[0]:
        if _unique_scalar(records, cand):
            return cand
    return None


def _unique_scalar(records: list[dict], col: str) -> bool:
    vals = [r.get(col) for r in records]
    if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in vals):
        return False
    return len(set(vals)) == len(vals)


def _churn(rec: dict, records: list[dict]) -> bool:
    """Change ONE value in `rec` to model a mutable-state edit, WITHOUT touching a column
    the row differ might use as its stable key. Preference: a top-level scalar whose values
    repeat across records (a real state field, never a key) → a scalar leaf inside a nested
    object (changes the record without touching any top-level key) → give up. Returns True
    if it changed something."""
    nonunique = [k for k, v in records[0].items()
                 if isinstance(v, (str, int)) and not isinstance(v, bool)
                 and len({r.get(k) for r in records}) < len(records)]
    if nonunique:
        _bump(rec, nonunique[0])
        return True
    for v in rec.values():  # nested-leaf fallback (e.g. a commit's nested `commit` dict)
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (str, int)) and not isinstance(vv, bool):
                    _bump(v, kk)
                    return True
    return False


def _bump(d: dict, col: str) -> None:
    v = d.get(col)
    if isinstance(v, str):
        d[col] = v + "~changed"
    elif isinstance(v, int) and not isinstance(v, bool):
        d[col] = v + 1


def _appended(base_rec: dict, idcol: str, records: list[dict]) -> dict:
    """A new record (copy of base_rec) with a fresh unique id — a row that just appeared."""
    new = copy.deepcopy(base_rec)
    cur = base_rec[idcol]
    if isinstance(cur, int) and not isinstance(cur, bool):
        new[idcol] = max(r[idcol] for r in records) + 1
    else:
        new[idcol] = f"{cur}-appended"
    return new


def measure_diff(name: str, base: list) -> dict | None:
    if not (isinstance(base, list) and len(base) >= 3
            and all(isinstance(r, dict) for r in base)):
        return None
    idcol = _id_col(base)
    if idcol is None:
        return None
    curr = copy.deepcopy(base)
    if not (_churn(curr[0], base) and _churn(curr[1], base)):
        return None
    curr.append(_appended(base[0], idcol, base))

    full = count_cl100k(transforms.compress(curr))
    wire = transforms.diff_wire(base, curr, tool=name)
    if wire is None or not transforms.diff_roundtrip_ok(base, curr):
        return {"name": name, "records": len(base), "full_terse": full, "diff": None,
                "smaller_pct": None}
    diff = count_cl100k(wire)
    return {"name": name, "records": len(base), "full_terse": full, "diff": diff,
            "smaller_pct": round(100 * (1 - diff / full), 1) if full else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for p in sorted(CORPUS.glob("*.json")):
        r = measure_diff(p.stem, json.loads(p.read_text()))
        if r is not None:
            rows.append(r)

    tot_full = sum(r["full_terse"] for r in rows if r["diff"] is not None)
    tot_diff = sum(r["diff"] for r in rows if r["diff"] is not None)
    tot_pct = round(100 * (1 - tot_diff / tot_full), 1) if tot_full else 0.0

    if args.json:
        print(json.dumps({"rows": rows, "total_smaller_pct": tot_pct}, indent=2))
        return 0

    print("\nterse cross-call diff — SECOND-call cost on real list payloads "
          "(2 records changed, 1 appended)\n")
    hdr = f"{'payload':<20}{'records':>8}{'full re-send':>13}{'diff':>8}{'diff smaller':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = f"{r['diff']:>8}" if r["diff"] is not None else f"{'n/a':>8}"
        s = f"{r['smaller_pct']:>13.1f}%" if r["smaller_pct"] is not None else f"{'—':>14}"
        print(f"{r['name']:<20}{r['records']:>8}{r['full_terse']:>13}{d}{s}")
    print("-" * len(hdr))
    print(f"{'WEIGHTED TOTAL':<20}{'':>8}{tot_full:>13}{tot_diff:>8}{tot_pct:>13.1f}%")
    print("\n'full re-send' = terse single-shot codec on the whole new result (what a "
          "stateless encoding pays EVERY call). 'diff' = the lossless delta terse emits on "
          "the repeated call instead. TOON / minify have no diff form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
