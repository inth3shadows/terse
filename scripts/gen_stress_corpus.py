"""Generate a synthetic STRESS corpus for the fluency eval.

The real corpus is a thin, incidental sample (a handful of record-shaped payloads);
a verdict drawn from it would violate the project's own honesty bar (report.py warns
that thin samples must not read as "nothing to compress"). This generates payloads
that *maximally* stress the two transforms most likely to cost a model comprehension:

  - heavy `~N` dictionary-alias resolution (many repeated long values),
  - column->value mapping over WIDE tables and enumeration over LONG ones,
  - nested uniform-dict columns (the subcols form).

It also carries `synthetic.*` payloads for a different reason: a fleet SHAPE that no
capture corpus can hold, because the shipped policy forbids persisting that tool's output
(#403 Blocker 2). Those are structure-faithful and value-invented, and the prefix keeps
them distinguishable from real traffic wherever a report prints the tool name.

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


def secret_list_credentials(n_declared: int = 60, n_undeclared: int = 10) -> dict:
    """A SHAPE the capture corpus can never hold: `secret.list_credentials` (#403 Blocker 2).

    That tool is the single largest codec saver in the live ledger (73.4% of 30-day codec
    savings when measured, 2026-09-11) and has ZERO captured payloads, because the shipped
    policy sets `capture: false` for every secret-broker tool — "compress it, never persist
    it". Real values must not reach a corpus that feeds a published benchmark, so this
    reproduces the STRUCTURE from `secret_broker/ops/meta.py:46-150` and invents the values.

    What a verdict over this payload does and does not say: it says the codec reads THIS
    SHAPE back correctly — a records list with a union schema, one heavily repeated long
    string, and mostly-scalar columns. It says nothing about the real tool's values, and the
    `synthetic.` tool prefix keeps the two apart in the verdict table.

    Faithful in the ways the codec is sensitive to:
    - Declared rows carry `name, kind, source, env_var, has_value` + the three usage fields;
      UNDECLARED rows drop `kind`/`env_var` and add `note`, so the table has absent columns.
    - `note` is one long string repeated across every undeclared row — the dictionary tier's
      alias case.
    - `kind`/`source` repeat from small pools; `last_used`/`last_ok` are null on never-used
      credentials, which is the explicit-null-vs-absent distinction a deref failure destroys.
    """
    kinds = ["api_key", "oauth_token", "password", "ssh_key"]
    sources = ["config_toml", "connect_vault", "env"]
    note = ("In the 1Password vault, not declared in config.toml. Usable as-is — resolve or "
            "inject it by this name. It has no declared env_var, so name one at the call "
            "site (NAME=ENV_VAR). secret.add_credential is optional: it only adds a default "
            "env_var. A redaction schema, the vault item/field a declaration points at, the "
            "backend it resolves from, and the kind that labels redacted output are not "
            "settable from any tool — they are config.toml only.")
    creds: list[dict] = []
    for i in range(n_declared):
        used = i % 3 != 0                      # a third have never been called
        creds.append({
            "name": f"service-{i:02d}-credential",
            "kind": kinds[i % len(kinds)],
            "source": sources[i % len(sources)],
            "env_var": f"SERVICE_{i:02d}_CREDENTIAL",
            "has_value": i % 7 != 0,
            "call_count": (i * 13) % 90 if used else 0,
            "last_used": f"2026-09-{(i % 28) + 1:02d}T0{i % 10}:15:00Z" if used else None,
            "last_ok": f"2026-09-{(i % 28) + 1:02d}T0{i % 10}:15:02Z" if used else None,
        })
    for i in range(n_undeclared):
        creds.append({
            "name": f"Vault Item {i:02d}",
            "source": "connect_vault_undeclared",
            "has_value": True,
            "call_count": 0,
            "last_used": None,
            "last_ok": None,
            "note": note,
        })
    services = [{
        "name": f"upstream-{i:02d}",
        "host": f"api-{i:02d}.example.invalid",
        "credential": f"service-{i:02d}-credential",
        "non_mutating": ["GET", "HEAD"] if i % 2 else ["GET"],
        "redaction_scope": "full" if i % 3 else "reduced",
        "body_ceilings": {"/v1/items": 4096, "/v1/search": 2048},
        "body_required": ["query"] if i % 2 else [],
    } for i in range(6)]
    return {"credentials": creds, "services": services}


SYNTHETIC_TOOL = "synthetic.secret.list_credentials"

# Fleet SHAPES no capture corpus contains, kept apart from real traffic by a `synthetic.`
# prefix that survives into the codec verdict's tool column (#403 Blocker 2). THREE sizes
# under ONE tool name: they pool into a single (tool, shape) cell, and this payload yields
# exactly one codec question each (`credentials` has no container column, so `enumerate`
# only), so one payload could never clear `report._CODEC_MIN_TRIALS` — three at
# `--trials 7` reach 21. The sizes also straddle the tabularizer's fold threshold.
SYNTHETIC = [
    (SYNTHETIC_TOOL, secret_list_credentials()),
    (SYNTHETIC_TOOL, secret_list_credentials(n_declared=24, n_undeclared=4)),
    (SYNTHETIC_TOOL, secret_list_credentials(n_declared=8, n_undeclared=2)),
]

PAYLOADS = {
    "stress.wide_table": wide_table(),
    "stress.long_table": long_table(),
    "stress.heavy_alias": heavy_alias(),
    "stress.nested_records": nested_records(),
    "stress.mixed_realistic": mixed_realistic(),
    "stress.object_alias": object_alias(),
}


def main(corpus_dir: str = "corpus-stress") -> int:
    for tool, obj in list(PAYLOADS.items()) + SYNTHETIC:
        # `len(obj)` counts records for the list payloads and top-level keys for the dict
        # ones; the record count that matters for a dict payload is printed beside it.
        size = (f"{len(obj)} records" if isinstance(obj, list)
                else f"{len(obj['credentials'])} records")
        path = capture_payload(tool, minify(obj), corpus_dir)
        print(f"wrote {tool} ({size}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "corpus-stress"))
