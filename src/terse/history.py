"""Run-history persistence for `measure --history` (#51 fast-follow: "historical
trend across runs" — deferred from v1 specifically to not block the first visual
win on a persistence-format decision; jsonl chosen here for the same reasons the
corpus uses one-file-per-payload JSON, not sqlite: plain text, diffable, greppable,
no new dependency).

Unlike report.py's rendered reports (deliberately timestamp-free so the same corpus
always renders byte-identical, principle #31) or capture.py's corpus (content-addressed
so the same payload never gets two entries, same principle), a history file's entire
point is to record when distinct real runs happened — the trend IS that record. This
stays principle #31-compliant by injecting the clock as an explicit `ts` parameter
into the pure `summarize_run` rather than reading it implicitly inside; only the CLI
edge (`cli.py`) ever calls the actual clock.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._secure_io import append_restricted, mkdir_restricted
from .report import _sum

# One JSONL line per real `measure --history` run — tiny, but unbounded over a project's
# life. Rotate a single generation once the live file passes this so it can't grow forever
# (the archived `.1` isn't read into the trend; at this size that's tens of thousands of
# runs away and purely a safety bound, not routine behaviour).
MAX_HISTORY_BYTES = 5_000_000


def summarize_run(rows: list[dict[str, Any]], ts: str, *, label: str | None = None) -> dict[str, Any]:
    """One compact run-summary row for the history file — aggregate token counts
    only, never raw payload content (a history file may be committed/shared even
    when the corpus that produced it isn't)."""
    failures = [r for r in rows if not r.get("roundtrip_ok", False)]
    raw = _sum(rows, "cl100k", "raw")
    cmp_ = _sum(rows, "cl100k", "compressed")
    saved = raw - cmp_
    return {
        "ts": ts,
        "label": label,
        "n_payloads": len(rows),
        "lossless_pass": len(rows) - len(failures),
        "raw_tok": raw,
        "compressed_tok": cmp_,
        "saved_tok": saved,
        "saved_pct": (saved / raw * 100) if raw else None,
    }


def append_run(path: Path, run: dict[str, Any]) -> None:
    """Append one run summary as a single JSONL line. Creates the file (and its parent
    dir) on first use. Written via append_restricted (0600) for perms parity with the
    corpus/ledger — a run row can embed an operator-supplied `label`. Rotates one
    generation once the file passes MAX_HISTORY_BYTES so it can't grow without bound."""
    mkdir_restricted(path.parent)
    try:
        if path.stat().st_size >= MAX_HISTORY_BYTES:
            path.replace(path.with_name(path.name + ".1"))
    except FileNotFoundError:
        pass  # first run — nothing to rotate
    append_restricted(path, json.dumps(run, ensure_ascii=False) + "\n")


def load_history(path: Path) -> list[dict[str, Any]]:
    """Read every run summary from the history file, oldest first. A missing file
    is just "no history yet" ([]), not an error — every `--history` file starts
    this way. A malformed line is skipped rather than fatal, so one interrupted
    append can't take down every future trend render."""
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs
