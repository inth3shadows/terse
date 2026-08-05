#!/usr/bin/env python3
"""TOON's number on the SAME third-party-server payloads §6 measures terse on (#138).

§6 reports what terse does zero-config to real MCP servers. It had no competitor column,
so a reader could not tell whether the codec %s are good or merely non-zero. This script
adds TOON's, measured on the identical captured payloads — not on a re-run with different
arguments, which is the one way a head-to-head can quietly cheat.

    python toon_column.py <capture-dir> [<capture-dir> ...]

Input is the terse capture-dir the probe already writes (`mcp_probe.py`'s CORPUS argument):
one JSON envelope per payload carrying `{server, tool, shape, raw}`. `raw` is the tool
result's text block byte-exact, which is precisely TOON's input, so re-encoding a stored
capture is equivalent to re-running the server — and unlike a re-run it is guaranteed to be
the same bytes the published codec % came from.

terse's side comes from `measure_payload`, the same function `terse measure` calls; TOON's
from `../toon_encode.mjs`, the pinned official `@toon-format/toon` encoder §1 already uses.
Neither number is reimplemented here.

Two honesty rules, both inherited from `../benchmark.py`:

- A payload whose round-trip fails is DROPPED, never banked. Reported by name, not silently.
- A payload that is not JSON at all is `n/a`, never `0.0%`. TOON is a JSON serialization;
  "0%" would read as "tried, tied" when the truth is "cannot encode". terse scores 0% on
  those same rows, so the distinction is the whole point of the column.

Payloads are deduped by content sha across dirs: the size-axis runs and the multi-peer run
re-capture byte-identical payloads, and counting one three times would weight the total
toward whichever tool happened to be probed most.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from terse.capture import is_sidecar_filename
from terse.measure import measure_payload
from terse.tokenize import count_cl100k

TOON_SCRIPT = Path(__file__).resolve().parents[1] / "toon_encode.mjs"


def toon_encode(raw: str) -> tuple[str, bool] | None:
    """Encode `raw` to TOON via the pinned official encoder.

    Returns (text, lossless), or None when the payload is not JSON — the caller must
    render that as `n/a`, not as a 0% tie. A non-zero exit is also None: the encoder
    refusing a legal JSON value is still "TOON cannot represent this"."""
    try:
        json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    proc = subprocess.run(["node", str(TOON_SCRIPT)], input=raw, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        return None
    out = json.loads(proc.stdout)
    return out["toon"], bool(out["lossless"])


def load_payloads(dirs: list[Path]) -> list[dict]:
    """Every capture envelope under `dirs`, deduped by raw-content sha, sorted stably."""
    seen: dict[str, dict] = {}
    for d in dirs:
        # `_calls.json` (mcp_probe.py's argument sidecar, #138 Phase 2) is not a capture
        # envelope -- skip it by name rather than falling through to the noisy SKIP path
        # below, which would otherwise fire on every corpus dir from a self-describing run.
        for f in sorted(p for p in d.glob("*.json") if not is_sidecar_filename(p.name)):
            try:
                env = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  SKIP unreadable {f}: {exc}", file=sys.stderr)
                continue
            if not isinstance(env, dict) or "raw" not in env:
                print(f"  SKIP not a capture envelope: {f}", file=sys.stderr)
                continue
            # `sha` is the capture's own content hash; fall back to the raw text so a
            # future envelope without it still dedupes rather than double-counting.
            seen.setdefault(env.get("sha") or str(hash(env["raw"])), env)
    return sorted(seen.values(), key=lambda e: (e.get("server", ""), e.get("tool", "")))


def _pct(raw: int, other: int) -> float:
    return round(100 * (1 - other / raw), 1) if raw else 0.0


def measure_one(env: dict) -> dict:
    """One row: terse's codec % and TOON's, both against the same raw token count."""
    raw = env["raw"]
    m = measure_payload(raw)
    raw_tok = m["cl100k"]["raw"]
    # `saved_cl100k` is absolute tokens saved, and it is already zeroed when the lossless
    # gate failed — deriving the % from it rather than from `cl100k["embedded"]` is what
    # keeps a gate failure out of the published percentage. `tier_total` is the full
    # codec, which is what §6's "codec % (1-shot)" column reports.
    saved = m["saved_cl100k"]["tier_total"] or 0
    row = {
        "server": env.get("server", "?"),
        "tool": env.get("tool", "?"),
        "shape": m["shape"],
        "raw_tok": raw_tok,
        "terse_pct": round(100 * saved / raw_tok, 1) if raw_tok else 0.0,
        "terse_lossless": m["roundtrip_ok"],
        "toon_tok": None,
        "toon_pct": None,
        "toon_lossless": None,
    }
    enc = toon_encode(raw)
    if enc is not None:
        toon_txt, lossless = enc
        row["toon_tok"] = count_cl100k(toon_txt)
        row["toon_pct"] = _pct(raw_tok, row["toon_tok"])
        row["toon_lossless"] = lossless
    return row


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        print(f"not a directory: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2

    rows = [measure_one(env) for env in load_payloads(dirs)]
    if not rows:
        print("no capture payloads found — nothing to measure", file=sys.stderr)
        return 1

    # Never bank a number a lossy encoder produced. A False `toon_lossless` is a real
    # finding (TOON encoded it but lost data), distinct from `None` (could not encode).
    lossy = [r for r in rows if not r["terse_lossless"]
             or r["toon_lossless"] is False]
    good = [r for r in rows if r not in lossy]

    print(f"\nterse vs TOON on captured third-party MCP server payloads "
          f"({len(rows)} payloads, cl100k)\n")
    print(f"{'server':<22}{'tool':<26}{'shape':<20}{'raw tok':>9}"
          f"{'terse':>8}{'TOON':>9}")
    print("-" * 94)
    for r in rows:
        toon = "  n/a" if r["toon_pct"] is None else f"{r['toon_pct']:>6.1f}%"
        flag = "  !LOSSY-DROP" if r in lossy else ""
        print(f"{r['server']:<22}{r['tool']:<26}{r['shape']:<20}{r['raw_tok']:>9,}"
              f"{r['terse_pct']:>7.1f}%{toon:>9}{flag}")

    # The weighted total covers only rows TOON can actually encode: mixing in text-only
    # payloads would let a corpus of prose decide the winner, when neither tool claims
    # anything there (both score 0%). The all-rows total is printed separately below.
    encodable = [r for r in good if r["toon_pct"] is not None]
    if encodable:
        tot_raw = sum(r["raw_tok"] for r in encodable)
        t_out = sum(round(r["raw_tok"] * (1 - r["terse_pct"] / 100)) for r in encodable)
        print("-" * 94)
        print(f"{'TOTAL (JSON only)':<48}{len(encodable)} payloads{tot_raw:>13,}"
              f"{_pct(tot_raw, t_out):>7.1f}%"
              f"{_pct(tot_raw, sum(r['toon_tok'] for r in encodable)):>8.1f}%")
    n_text = sum(1 for r in rows if r["toon_pct"] is None)
    print(f"\n{n_text} of {len(rows)} payloads are not JSON — TOON cannot encode them "
          f"at all (terse scores 0% one-shot on those same rows).")
    if lossy:
        print("DROPPED as lossy (not banked): "
              + ", ".join(f"{r['server']}/{r['tool']}" for r in lossy))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
