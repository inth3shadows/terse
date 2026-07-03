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

from .report import _sum


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
    """Append one run summary as a single JSONL line. Creates the file (and its
    parent dir) on first use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")


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
