#!/usr/bin/env python3
"""Head-to-head token-reduction benchmark: terse vs TOON vs trivial baselines, on a
corpus of REAL GitHub API payloads (the record-shaped tool output terse targets).

Everything measured here is LOSSLESS and verified as such per payload — a row is dropped
from the aggregate if either encoder fails its round-trip, so no number is banked on a
payload that lost data. Token counts use cl100k_base (tiktoken), the same tokenizer terse
uses internally; the absolute % shifts under a different vocabulary but the ranking is
stable (that is terse's own cross-tokenizer-invariance claim, not re-litigated here).

Run:  uv run scripts/bench/benchmark.py            # table to stdout
      uv run scripts/bench/benchmark.py --json     # machine-readable

Requires the pinned TOON encoder: `cd scripts/bench && npm install` (see package.json).
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

from terse import transforms
from terse.tokenize import count_cl100k

BENCH_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCH_DIR / "corpus"
TOON_SCRIPT = BENCH_DIR / "toon_encode.mjs"


def toon_encode(raw: str) -> tuple[str, bool]:
    """Encode raw JSON to TOON via the official pinned encoder; returns (text, lossless)."""
    proc = subprocess.run(["node", str(TOON_SCRIPT)], input=raw, capture_output=True,
                          text=True, timeout=60, check=True)
    out = json.loads(proc.stdout)
    return out["toon"], bool(out["lossless"])


def measure_one(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    minified = transforms.minify(obj)
    terse_txt = transforms.compress(obj)
    terse_lossless = transforms.decompress(terse_txt) == obj
    toon_txt, toon_lossless = toon_encode(raw)

    raw_tok = count_cl100k(raw)
    return {
        "name": path.stem,
        "records": _record_count(obj),
        "raw_tok": raw_tok,
        "min_tok": count_cl100k(minified),
        "terse_tok": count_cl100k(terse_txt),
        "toon_tok": count_cl100k(toon_txt),
        # gzip is NOT a competitor (its output is not model-readable) — a reference ceiling
        # for "how much structural redundancy exists at all", in bytes not tokens.
        "gzip_bytes_pct": round(100 * (1 - len(gzip.compress(minified.encode())) / len(minified.encode())), 1),
        "terse_lossless": terse_lossless,
        "toon_lossless": toon_lossless,
    }


def _record_count(obj: object) -> int:
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        return max((len(v) for v in obj.values() if isinstance(v, list)), default=0)
    return 0


def _pct(raw: int, other: int) -> float:
    return round(100 * (raw - other) / raw, 1) if raw else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = sorted(CORPUS_DIR.glob("*.json"))
    if not paths:
        print(f"no corpus in {CORPUS_DIR} — run scripts/bench/fetch_corpus.sh first",
              file=sys.stderr)
        return 2

    rows = [measure_one(p) for p in paths]
    # Drop any row whose either encoder wasn't lossless — never bank a lossy win.
    good = [r for r in rows if r["terse_lossless"] and r["toon_lossless"]]

    for r in rows:
        r["terse_pct"] = _pct(r["raw_tok"], r["terse_tok"])
        r["toon_pct"] = _pct(r["raw_tok"], r["toon_tok"])
        r["min_pct"] = _pct(r["raw_tok"], r["min_tok"])
        # The structural win BEYOND free minification (both terse and TOON minify inherently).
        r["terse_vs_min_pct"] = _pct(r["min_tok"], r["terse_tok"])
        r["toon_vs_min_pct"] = _pct(r["min_tok"], r["toon_tok"])

    tot_raw = sum(r["raw_tok"] for r in good)
    tot = {
        "raw": tot_raw,
        "min_pct": _pct(tot_raw, sum(r["min_tok"] for r in good)),
        "terse_pct": _pct(tot_raw, sum(r["terse_tok"] for r in good)),
        "toon_pct": _pct(tot_raw, sum(r["toon_tok"] for r in good)),
        "terse_vs_min_pct": _pct(sum(r["min_tok"] for r in good), sum(r["terse_tok"] for r in good)),
        "toon_vs_min_pct": _pct(sum(r["min_tok"] for r in good), sum(r["toon_tok"] for r in good)),
        "n_payloads": len(good),
        "n_dropped_lossy": len(rows) - len(good),
    }

    if args.json:
        print(json.dumps({"rows": rows, "totals": tot}, indent=2))
        return 0

    print(f"\nterse vs TOON — cl100k token reduction vs raw JSON, {len(good)} real GitHub "
          f"API payloads (all lossless)\n")
    hdr = f"{'payload':<20}{'recs':>5}{'raw tok':>9}{'minify':>8}{'terse':>8}{'TOON':>8}   {'terse>min':>9}{'TOON>min':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "" if (r["terse_lossless"] and r["toon_lossless"]) else "  !LOSSY-SKIP"
        print(f"{r['name']:<20}{r['records']:>5}{r['raw_tok']:>9}"
              f"{r['min_pct']:>7.1f}%{r['terse_pct']:>7.1f}%{r['toon_pct']:>7.1f}%   "
              f"{r['terse_vs_min_pct']:>8.1f}%{r['toon_vs_min_pct']:>8.1f}%{flag}")
    print("-" * len(hdr))
    print(f"{'WEIGHTED TOTAL':<20}{'':>5}{tot['raw']:>9}"
          f"{tot['min_pct']:>7.1f}%{tot['terse_pct']:>7.1f}%{tot['toon_pct']:>7.1f}%   "
          f"{tot['terse_vs_min_pct']:>8.1f}%{tot['toon_vs_min_pct']:>8.1f}%")
    print("\n'minify/terse/TOON' columns = % fewer tokens than raw. 'terse>min'/'TOON>min' "
          "= the structural win BEYOND free minification.")
    if tot["n_dropped_lossy"]:
        print(f"note: {tot['n_dropped_lossy']} payload(s) dropped from the total for a "
              f"failed round-trip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
