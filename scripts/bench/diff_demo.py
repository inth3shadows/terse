#!/usr/bin/env python3
"""Supplementary, terse-ONLY measurement: the cross-call diff tier.

A stateless encoding (TOON, minify, terse's own single-shot codec) must resend the whole
payload every call. terse's diff tier, when the SAME tool is called repeatedly, emits a
lossless delta against the prior result instead — the agent-loop pattern (a list that
grows/changes by a few records between calls). No competitor here has an equivalent, so
this isn't a head-to-head; it's the extra axis, quantified on real data.

We model one realistic repeated call on the real gh_pulls corpus: the base result, then a
`curr` that changes two records' `state`/`updated_at` and appends one new record — exactly
what a poll-again loop sees. We report the cost of the second call under each strategy.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from terse import transforms
from terse.tokenize import count_cl100k

CORPUS = Path(__file__).resolve().parent / "corpus"


def main() -> int:
    base = json.loads((CORPUS / "gh_pulls.json").read_text())
    if not isinstance(base, list) or len(base) < 3:
        print("gh_pulls corpus not a list of >=3 records; run fetch_corpus.sh")
        return 2

    # curr = the same list, two records mutated + one new record appended (the poll-again
    # delta). Built from real records so the diff operates on genuine structure.
    curr = copy.deepcopy(base)
    curr[0]["state"] = "closed"
    curr[0]["updated_at"] = "2026-07-17T00:00:00Z"
    curr[1]["state"] = "closed"
    new = copy.deepcopy(base[0])
    new["id"] = base[0]["id"] + 1
    new["number"] = base[0]["number"] + 1000
    curr.append(new)

    full_terse = count_cl100k(transforms.compress(curr))
    diff_wire = transforms.diff_wire(base, curr, tool="gh_pulls")
    diff_ok = diff_wire is not None and transforms.diff_roundtrip_ok(base, curr)
    diff_tok = count_cl100k(diff_wire) if diff_wire else None

    print("\nterse cross-call diff — second call cost on real gh_pulls "
          "(2 records changed, 1 appended)\n")
    print(f"  full re-send, terse single-shot codec : {full_terse:>7} tok")
    if diff_tok is not None:
        print(f"  terse cross-call diff (lossless={diff_ok})   : {diff_tok:>7} tok"
              f"   ({100 * (1 - diff_tok / full_terse):.1f}% smaller than a full re-send)")
    else:
        print("  terse cross-call diff                  : (no diff applied)")
    print("\n  TOON / minify / any stateless encoding has no cross-call form — it pays the "
          "full re-send column every call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
