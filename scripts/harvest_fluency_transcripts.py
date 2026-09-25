"""One-time harvest of persisted `claude -p` fluency-probe transcripts, before #456 stops
writing new ones.

`terse fluency`'s `cli:<alias>` backend (`fluency/answerers.py`'s `cli_answerer`) shells out
to `claude -p` for every probe. Until #456, each of those calls was a real Claude Code
session — persisted, like any other, under `~/.claude/projects/<cwd-slug>/<session>.jsonl` —
because the harness never told `claude -p` not to. The harness itself writes no per-probe
record of its own (that gap is `fluency/ledger.py`'s job going forward), so these transcripts
are the ONLY raw record of every fluency run made before the harness stopped persisting them:
one user message (the harness's system+data prompt) and one assistant reply per file, sitting
under a tool-visible temp dir the harness names `-tmp-terse-fluency-cli-<random>`.

This script reads every one of those transcripts once, pulls out the (prompt, reply, usage)
triple each holds, and writes them to a single gzip JSONL file — before #456's own commit
(already on this branch) stops the leak going forward, and before an eventual archive+clear
of the source directories (a separate, later step; this script does not touch the source).

Structure notes, verified against the real corpus rather than assumed (2026-09-25, ~3,800
files, 6 terse-envelope markers seen: __terse_table__/__terse_dict__/__terse_diff__/
__terse_textdiff__/__terse_absent__/__terse_dropped__):
  - Every transcript here holds exactly one `type: "user"` line and at most one DISTINCT
    assistant `message.id` — but that one logical reply can be split across TWO `type:
    "assistant"` JSONL lines (one carrying a `thinking` content block, one a `text` block),
    both stamped with the same `usage` snapshot. Concatenating every `text` block across all
    assistant lines and taking `usage` from whichever carries it handles both the common
    one-line case and the split one.
  - A handful of transcripts (interrupted/failed probes) hold a user line and NO assistant
    line at all — skipped as "no assistant reply", not emitted as a null-reply row, since the
    output contract here is "one row per (user prompt, assistant reply) PAIR".
  - No `tool_use` blocks were observed (echoed by `answerers.cli_answerer` disabling every
    tool for the probe subprocess) and every assistant line carries `usage`, so those are not
    handled as separate skip reasons — a transcript missing either is treated as unparseable
    the same as one with no assistant line.

Read-only on the source tree: this only ever calls `Path.read_text`/`.glob` under `--root`,
never a write or delete.

Run:  uv run scripts/harvest_fluency_transcripts.py
      uv run scripts/harvest_fluency_transcripts.py --root <dir> --out <file.jsonl.gz>
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("~/.claude/projects").expanduser()
DEFAULT_OUT = Path("~/.local/share/terse/fluency-harvest-2026-09-25.jsonl.gz").expanduser()

# Every probe transcript sits under a session dir the harness names by its temp cwd,
# `mkdtemp(prefix="terse-fluency-cli-")` (see `fluency/answerers.py`), slugged by Claude
# Code's own `~/.claude/projects/<cwd-with-/-as-->` convention into this prefix.
DIR_PREFIX = "-tmp-terse-fluency-cli"

# Every terse wire-form marker `transforms.py`/`text_diff.py` can emit into a probe's DATA
# section. Seen in the real corpus: all six. `__terse_diff__`/`__terse_textdiff__` mark the
# diff-fluency/text-diff-fluency harnesses' diff arm; the rest mark `run_payload`'s terse/
# primer/inline arms (`__terse_dropped__` is drop-eval's own marker, included for the same
# reason the others are: ANY of them means this prompt carries terse's wire form, not raw
# JSON — that is the only distinction "arm" draws here).
TERSE_MARKERS = (
    "__terse_table__",
    "__terse_dict__",
    "__terse_diff__",
    "__terse_textdiff__",
    "__terse_absent__",
    "__terse_dropped__",
)


def classify_arm(user_prompt: str) -> str:
    """"terse" iff the prompt carries one of terse's wire-form markers, else "raw" — the
    only two forms a bare user-prompt string (no side channel recording which harness arm
    produced it) can be told apart by after the fact."""
    return "terse" if any(marker in user_prompt for marker in TERSE_MARKERS) else "raw"


def _reply_and_usage(assistant_lines: list[dict]) -> tuple[str | None, dict | None]:
    """Concatenate every `text` content block across all assistant lines for this
    transcript (there is at most one in the real corpus, but concatenating rather than
    taking-the-first is what stays correct if a future transcript ever splits the text
    itself), and take `usage` from whichever line carries it — they're repeats of the same
    snapshot when a reply spans a `thinking` line and a `text` line."""
    texts: list[str] = []
    usage: dict | None = None
    for line in assistant_lines:
        message = line.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        if message.get("usage") is not None:
            usage = message["usage"]
    return ("".join(texts) if texts else None), usage


def harvest_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """One transcript -> (row, None) or (None, skip_reason). Never raises on a malformed
    file — a torn/partial JSONL line (same tolerance `stats.load_stats` gives its own
    ledger) is skipped, not fatal to the rest of the file or the run."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"unreadable ({exc.__class__.__name__})"

    user: dict[str, Any] | None = None
    assistant_lines: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        line_type = obj.get("type")
        if line_type == "user" and user is None:
            user = obj
        elif line_type == "assistant":
            assistant_lines.append(obj)

    if user is None:
        return None, "no user message"
    content = (user.get("message") or {}).get("content")
    if not isinstance(content, str):
        return None, "user content is not plain text (unexpected transcript shape)"
    if not assistant_lines:
        return None, "no assistant reply"
    reply, usage = _reply_and_usage(assistant_lines)
    if reply is None:
        return None, "assistant line(s) carry no text block"

    model = None
    for line in assistant_lines:
        model = (line.get("message") or {}).get("model")
        if model is not None:
            break

    row = {
        "session_id": user.get("sessionId"),
        "ts": user.get("timestamp"),
        "cwd": user.get("cwd"),
        "model": model,
        "user_prompt": content,
        "reply": reply,
        "usage": usage,
        "arm": classify_arm(content),
        "expected": None,  # ground truth was never in the transcript (see module docstring)
        "source_path": str(path),
    }
    return row, None


def iter_transcript_files(root: Path):
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(DIR_PREFIX):
            yield from sorted(entry.glob("*.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Harvest persisted claude -p fluency-probe transcripts into one gzip "
                    "JSONL file.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"directory holding Claude Code's per-cwd session dirs "
                         f"(default: {DEFAULT_ROOT})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"gzip JSONL output path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    files = list(iter_transcript_files(args.root))
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for f in files:
        row, reason = harvest_file(f)
        if row is None:
            skipped[reason or "unknown"] += 1
        else:
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_model = Counter(r["model"] for r in rows)
    by_arm = Counter(r["arm"] for r in rows)
    by_month = Counter((r["ts"] or "")[:7] or "(no timestamp)" for r in rows)

    print(f"files read: {len(files)}")
    print(f"rows written: {len(rows)} -> {args.out}")
    print("\nrows per model:")
    for model, n in by_model.most_common():
        print(f"  {model or '(unknown)'}: {n}")
    print("\nrows per arm:")
    for arm, n in by_arm.most_common():
        print(f"  {arm}: {n}")
    print("\nrows per month:")
    for month, n in sorted(by_month.items()):
        print(f"  {month}: {n}")
    print(f"\nfiles skipped: {sum(skipped.values())}")
    for reason, n in skipped.most_common():
        print(f"  {n}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
