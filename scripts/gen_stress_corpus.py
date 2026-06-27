"""Generate a synthetic STRESS corpus for the fluency eval.

The real corpus is a thin, incidental sample (a handful of record-shaped payloads);
a verdict drawn from it would violate the project's own honesty bar (report.py warns
that thin samples must not read as "nothing to compress"). This generates payloads
that *maximally* stress the two transforms most likely to cost a model comprehension:

  - heavy `~N` dictionary-alias resolution (many repeated long values),
  - column->value mapping over WIDE tables and enumeration over LONG ones,
  - nested uniform-dict columns (the subcols form).

Deterministic (no randomness) so the eval is reproducible. Writes shape-tagged
envelopes via the same capture path the real corpus uses.

    python scripts/gen_stress_corpus.py [corpus_dir]   # default: corpus-stress
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terse.capture import capture_payload  # noqa: E402
from terse.transforms import minify  # noqa: E402

# A small pool of long, repeated strings — guarantees dictionary coding fires
# (a `~0` alias costs ~4 tokens, so aliasing only pays under real repetition).
STATUSES = ["awaiting-triage-from-maintainer", "in-progress-active-development",
            "blocked-on-upstream-dependency", "resolved-wont-fix-by-design"]
TEAMS = ["platform-infrastructure-team", "developer-experience-team",
         "security-and-compliance-team"]


def wide_table(n_rows: int = 8, n_cols: int = 12) -> list[dict]:
    """Many columns: a lookup on a far-right column stresses column->value mapping."""
    rows = []
    for i in range(n_rows):
        rec = {"id": i + 1}
        for c in range(1, n_cols):
            rec[f"col_{c:02d}"] = (i * n_cols + c) * 3  # distinct, addressable values
        rows.append(rec)
    return rows


def long_table(n_rows: int = 40) -> list[dict]:
    """Many rows, narrow: stresses enumeration / under-counting (the row-count hint)."""
    return [{"id": i + 1, "label": f"item-{i + 1:03d}", "weight": (i * 7) % 50}
            for i in range(n_rows)]


def heavy_alias(n_rows: int = 16) -> list[dict]:
    """Repeated long values across two columns: heavy `~N` alias resolution."""
    return [{
        "id": i + 1,
        "status": STATUSES[i % len(STATUSES)],
        "team": TEAMS[i % len(TEAMS)],
        "priority": (i % 5) + 1,
    } for i in range(n_rows)]


def nested_records(n_rows: int = 10) -> list[dict]:
    """A nested uniform-dict column -> the subcols (hoisted header) table form."""
    return [{
        "id": i + 1,
        "owner": {"name": f"user-{i + 1:02d}", "team": TEAMS[i % len(TEAMS)]},
        "status": STATUSES[i % len(STATUSES)],
        "count": (i * 11) % 30,
    } for i in range(n_rows)]


def mixed_realistic(n_rows: int = 12) -> list[dict]:
    """Width + repetition + numerics together — closest to real tool output."""
    return [{
        "id": 1000 + i,
        "name": f"resource-{i:02d}",
        "status": STATUSES[i % len(STATUSES)],
        "team": TEAMS[i % len(TEAMS)],
        "size_kb": (i * 13) % 100,
        "active": i % 2 == 0,
    } for i in range(n_rows)]


def object_alias(n: int = 12) -> list[dict]:
    """A column of repeated WHOLE objects with non-uniform key sets, so tabularize
    declines to hoist them to subcols and whole-subtree aliasing folds each into a
    `~N` that expands to an OBJECT — the comprehension case the deref question probes."""
    configs = [
        {"region": "us-east-1", "tier": "gold", "flags": ["a", "b"]},
        {"region": "eu-west-1", "tier": "silver", "extra": 1},          # different keys
        {"zone": "ap-south-1", "tier": "bronze", "flags": ["c", "d", "e"]},  # different keys
    ]
    return [{"id": i + 1, "name": f"node-{i + 1:02d}", "config": configs[i % len(configs)]}
            for i in range(n)]


PAYLOADS = {
    "stress.wide_table": wide_table(),
    "stress.long_table": long_table(),
    "stress.heavy_alias": heavy_alias(),
    "stress.nested_records": nested_records(),
    "stress.mixed_realistic": mixed_realistic(),
    "stress.object_alias": object_alias(),
}


def main(corpus_dir: str = "corpus-stress") -> int:
    for tool, obj in PAYLOADS.items():
        path = capture_payload(tool, minify(obj), corpus_dir)
        print(f"wrote {tool} ({len(obj)} records) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "corpus-stress"))
