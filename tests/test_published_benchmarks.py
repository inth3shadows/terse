"""The §1 benchmark tables in README.md and BENCHMARKS.md must match what the codec
actually does to the tracked corpus.

Both tables are hand-maintained prose, and they drifted: union-schema tabularize (#202)
moved `gh_issues` 32.7% -> 38.8% and the weighted total 58.3% -> 59.1%, and both files
went on publishing the old figures until someone happened to re-measure (#206). Nothing
failed, because nothing was checking — the same argument as KB principle #134: two things
that must stay in step need a test, not a note asking people to remember.

Scope is deliberately the **terse** column only. The TOON column comes from a pinned npm
encoder that CI has no node to run, and it cannot drift from a terse-side change anyway —
it moves only when `@toon-format/toon` is bumped, which is a visible dependency edit.
This checks the column that moves silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from terse import transforms
from terse.tokenize import count_cl100k

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "scripts" / "bench" / "corpus"
DOCS = ("README.md", "BENCHMARKS.md")

# "| gh_issues | 30 | 48,032 | **38.8%** | −8.0% |" — the terse cell may or may not be
# bold (bold marks the winner against TOON), so the emphasis is optional.
_ROW = re.compile(
    r"^\| (gh_\w+) \| [^|]* \| ([\d,]+) \| \*{0,2}(-?[\d.]+)%\*{0,2} \| [^|]*\|", re.M)
_TOTAL = re.compile(
    r"^\| \*\*weighted total\*\* \| \| \*{0,2}([\d,]+)\*{0,2} \| \*\*(-?[\d.]+)%\*\* \|", re.M)


def _measure() -> tuple[dict[str, tuple[int, float]], int, float]:
    """Per-payload (raw_tok, terse %) plus the weighted total, straight from the codec.

    Mirrors `scripts/bench/benchmark.py`'s terse column: full Tier-0 (`compress`) against
    the raw bytes, cl100k. If that script's definition changes, this drifts with it —
    which is why the assertion below is against the DOCS, not against a frozen constant.
    """
    per: dict[str, tuple[int, float]] = {}
    raw_sum = terse_sum = 0
    for path in sorted(CORPUS.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        raw_tok = count_cl100k(raw)
        terse_tok = count_cl100k(transforms.compress(json.loads(raw)))
        per[path.stem] = (raw_tok, 100.0 * (raw_tok - terse_tok) / raw_tok)
        raw_sum += raw_tok
        terse_sum += terse_tok
    return per, raw_sum, 100.0 * (raw_sum - terse_sum) / raw_sum


@pytest.mark.parametrize("doc", DOCS)
def test_published_per_payload_rows_match_the_codec(doc):
    per, _, _ = _measure()
    rows = _ROW.findall((REPO / doc).read_text(encoding="utf-8"))
    assert rows, f"{doc}: parsed no benchmark rows — did the table format change?"

    seen = set()
    for name, raw_txt, pct_txt in rows:
        assert name in per, f"{doc}: row {name!r} has no payload in scripts/bench/corpus"
        raw_tok, pct = per[name]
        seen.add(name)
        assert int(raw_txt.replace(",", "")) == raw_tok, (
            f"{doc}: {name} raw tokens published {raw_txt}, measured {raw_tok:,}")
        # One decimal place published, so a 0.05 window is exactly rounding tolerance.
        assert abs(float(pct_txt) - pct) < 0.05, (
            f"{doc}: {name} published {pct_txt}%, measured {pct:.1f}% — "
            f"re-run `uv run scripts/bench/benchmark.py` and update the table")

    # A payload silently dropped from the table is the same defect as a stale cell.
    assert seen == set(per), f"{doc}: table covers {sorted(seen)}, corpus has {sorted(per)}"


@pytest.mark.parametrize("doc", DOCS)
def test_published_weighted_total_matches_the_codec(doc):
    _, raw_sum, total_pct = _measure()
    m = _TOTAL.search((REPO / doc).read_text(encoding="utf-8"))
    assert m, f"{doc}: no weighted-total row found"
    assert int(m[1].replace(",", "")) == raw_sum, (
        f"{doc}: weighted raw published {m[1]}, measured {raw_sum:,}")
    assert abs(float(m[2]) - total_pct) < 0.05, (
        f"{doc}: weighted total published {m[2]}%, measured {total_pct:.1f}%")


def test_the_two_documents_publish_the_same_table():
    """README's §1 is a copy of BENCHMARKS' §1. They drifted apart once already, in the
    other direction — updating one and forgetting the other is the obvious next failure."""
    tables = {
        doc: {n: p for n, _, p in _ROW.findall((REPO / doc).read_text(encoding="utf-8"))}
        for doc in DOCS
    }
    assert tables["README.md"] == tables["BENCHMARKS.md"]
