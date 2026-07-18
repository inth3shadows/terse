#!/usr/bin/env python3
"""Column-width crossover: at what schema width does TOON overtake terse?

Holds ROW COUNT fixed and sweeps COLUMN COUNT, measuring terse vs TOON token reduction at
each width. This isolates the real dividing axis behind the terse/TOON headline result:
it is schema WIDTH (columns per record), not "flat vs nested". terse's win tracks rows +
value repetition (the dictionary tier folds repeated values), TOON's tracks column count
(it writes the header once per table instead of once per record, so its saving grows with
width).

Run:  cd scripts/bench && npm install     # once, for the pinned TOON encoder
      uv run scripts/bench/width_sweep.py

Deterministic (seeded). The exact crossover column is construction-specific — short key
names and low value-cardinality shift it — so read the printed OUTPUT, don't hardcode a
constant. Every row is verified lossless for both encoders before it is counted.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import _pct, toon_encode  # noqa: E402  (sibling script, not a package)

from terse import transforms  # noqa: E402
from terse.tokenize import count_cl100k  # noqa: E402

ROWS = 40
WIDTHS = range(2, 13)
# Low-cardinality categoricals (fold well for terse) alternated with high-cardinality
# integers (fold poorly) — a realistic mixed record, not stacked in either tool's favour.
CATEGORICAL = ["active", "closed", "pending", "merged", "draft", "queued"]


def _record(width: int, rnd: random.Random) -> dict:
    rec: dict = {}
    for c in range(width):
        rec[f"col{c}"] = rnd.choice(CATEGORICAL) if c % 2 == 0 else rnd.randint(0, 100_000)
    return rec


def main() -> int:
    rnd = random.Random(42)
    print(f"column-width sweep — {ROWS} rows, cl100k tokens, terse vs TOON (% fewer than raw)\n")
    hdr = f"{'cols':>4}{'raw_tok':>9}{'terse%':>8}{'TOON%':>8}{'winner':>9}"
    print(hdr)
    print("-" * len(hdr))
    crossover = None
    for w in WIDTHS:
        records = [_record(w, rnd) for _ in range(ROWS)]
        raw = json.dumps(records, separators=(",", ":"))
        terse_txt = transforms.compress(records)
        assert transforms.decompress(terse_txt) == records, "terse lost data"
        toon_txt, toon_ok = toon_encode(raw)
        assert toon_ok, "TOON lost data"
        raw_tok = count_cl100k(raw)
        t = _pct(raw_tok, count_cl100k(terse_txt))
        o = _pct(raw_tok, count_cl100k(toon_txt))
        winner = "terse" if t >= o else "TOON"
        if crossover is None and winner == "TOON":
            crossover = w
        print(f"{w:>4}{raw_tok:>9}{t:>7.1f}%{o:>7.1f}%{winner:>9}")
    print()
    if crossover:
        print(f"crossover: terse wins at <={crossover - 1} columns, TOON at >={crossover} "
              f"(this construction; short keys / low cardinality shift the exact column).")
    else:
        print("no crossover in the swept range — terse won at every width in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
