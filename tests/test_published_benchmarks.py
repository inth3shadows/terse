"""The benchmark tables in README.md and BENCHMARKS.md must match what the code actually
does to the tracked corpus.

Both files are hand-maintained prose, and they drifted: union-schema tabularize (#202)
moved §1's `gh_issues` 32.7% -> 38.8% and the weighted total 58.3% -> 59.1%, and §3's
"full re-send" column with them — that column *is* the same single-shot codec. Both went
on publishing the old figures until someone happened to re-measure (#206). Nothing failed,
because nothing was checking; the first pass at #206 then missed §3 for exactly the same
reason. That is principle #134's argument: two things that must stay in step need a test,
not a note asking people to remember.

Covered here: §1 (terse column, both docs), §3 (the diff table, both docs), and §4's terse
column, which is §1's number reprinted in a differently-shaped table and so drifts
independently.

NOT covered, deliberately:
  * §1's TOON column — a pinned npm encoder CI has no node to run, and it cannot move
    without a visible `@toon-format/toon` bump.
  * §4's headroom columns — produced by standing up a live proxy.
  * §6 — live third-party servers behind pinned repo clones.
Those are dated in the documents instead.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from terse import transforms
from terse.tokenize import count_cl100k

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "scripts" / "bench"
CORPUS = BENCH / "corpus"
DOCS = ("README.md", "BENCHMARKS.md")

# §1: "| gh_issues | 30 | 48,032 | **38.8%** | −8.0% |" — records, raw tok, terse %, TOON.
# The terse cell's emphasis is optional: bold marks the winner against TOON.
_S1 = re.compile(
    r"^\| (gh_\w+) \| (\d+)(?: obj)? \| ([\d,]+) \| \*{0,2}(-?[\d.]+)%\*{0,2} \| [^|]*\|", re.M)
_S1_TOTAL = re.compile(
    r"^\| \*\*weighted total\*\* \| \| \*{0,2}([\d,]+)\*{0,2} \| \*\*(-?[\d.]+)%\*\* \|", re.M)

# §3: "| gh_issues | 30 | 29,611 | 4,448 | **85.0%** |" in README (records column) and
# "| gh_issues | 29,611 | 4,448 | **85.0%** |" in BENCHMARKS (no records column).
_S3 = re.compile(
    r"^\| (gh_\w+) \| (?:\d+ \| )?([\d,]+) \| ([\d,]+) \| \*\*([\d.]+)%\*\* \|", re.M)
_S3_TOTAL = re.compile(
    r"^\| \*\*weighted total\*\* \| (?:\| )?([\d,]+) \| ([\d,]+) \| \*\*([\d.]+)%\*\* \|", re.M)

# §4: "| gh_issues | 48,032 | **38.8%** | 33.1% | 0.0% |" — raw tok then the terse column.
_S4 = re.compile(
    r"^\| (gh_\w+) \| ([\d,]+) \| \*{0,2}([\d.]+)%\*{0,2} \| [^|]*\| [^|]*\|", re.M)


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def _table(doc: str, header_starts_with: str) -> str:
    """The rows of the one markdown table whose header line starts with the given text.

    Anchoring on the header rather than on a section heading, because the two documents
    organise their prose differently (BENCHMARKS has numbered sections, README does not)
    and prose mentions the section names too. The header line is the stable landmark, and
    a header edit that breaks the lookup fails loudly here rather than silently matching
    the wrong table.
    """
    lines = (REPO / doc).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_starts_with):
            body = []
            for row in lines[i + 1:]:
                if not row.startswith("|"):
                    break
                body.append(row)
            return "\n".join(body) + "\n"
    raise AssertionError(f"{doc}: no table whose header starts with {header_starts_with!r}")


# Header anchors, per document. §4 lives only in BENCHMARKS.
_S1_HEADER = {"README.md": "| payload (real GitHub API) |", "BENCHMARKS.md": "| payload |"}
_S3_HEADER = {"README.md": "| repeated call |", "BENCHMARKS.md": "| repeated call |"}
_S4_HEADER = "| file | raw tok |"


def _rounds_to(measured: float, published: str) -> bool:
    """The documents publish one decimal place, so the check is that the measured value
    ROUNDS to what is printed — not that it sits within some window. A tolerance band is
    subtly wrong at the edge: `gh_workflow_runs` measures 80.3477, which is 0.0477 from
    the published 80.3, so a `< 0.05` window leaves 0.002pp of headroom before a correct
    document starts failing."""
    return round(measured, 1) == float(published)


def _measure_s1() -> tuple[dict[str, tuple[int, float]], int, float]:
    """Per-payload (raw_tok, terse %) plus the weighted total, from the codec.

    Mirrors `scripts/bench/benchmark.py`'s terse column, INCLUDING its rule that a payload
    failing its round-trip is dropped from the total rather than banked (`good` there).
    Without that filter the two would diverge the moment any payload went lossy — the
    document would be right per the script and this test would fail it.
    """
    per: dict[str, tuple[int, float]] = {}
    raw_sum = terse_sum = 0
    for path in sorted(CORPUS.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        raw_tok = count_cl100k(raw)
        terse_tok = count_cl100k(transforms.compress(obj))
        per[path.stem] = (raw_tok, 100.0 * (raw_tok - terse_tok) / raw_tok)
        if transforms.roundtrip_ok(obj):      # benchmark.py's `good` filter
            raw_sum += raw_tok
            terse_sum += terse_tok
    return per, raw_sum, 100.0 * (raw_sum - terse_sum) / raw_sum


def _measure_s3() -> tuple[dict[str, tuple[int, int, float]], int, int, float]:
    """§3's diff table, by importing `diff_demo.py` rather than reimplementing it.

    Its model (churn two records, append one) is intricate enough that a second copy here
    would be its own drift risk — the very thing this file exists to prevent.
    """
    spec = importlib.util.spec_from_file_location("_diff_demo", BENCH / "diff_demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_diff_demo"] = mod
    spec.loader.exec_module(mod)

    per: dict[str, tuple[int, int, float]] = {}
    full_sum = diff_sum = 0
    for path in sorted(CORPUS.glob("*.json")):
        r = mod.measure_diff(path.stem, json.loads(path.read_text(encoding="utf-8")))
        if r is None or r["diff"] is None:
            continue
        per[r["name"]] = (r["records"], r["full_terse"], r["diff"])
        full_sum += r["full_terse"]
        diff_sum += r["diff"]
    return per, full_sum, diff_sum, 100.0 * (1 - diff_sum / full_sum)


@pytest.mark.parametrize("doc", DOCS)
def test_s1_rows_match_the_codec(doc):
    per, _, _ = _measure_s1()
    table = _table(doc, _S1_HEADER[doc])
    rows = _S1.findall(table)
    assert rows, f"{doc}: parsed no §1 rows — did the table format change?"

    seen = set()
    for name, records, raw_txt, pct_txt in rows:
        assert name in per, f"{doc}: §1 row {name!r} has no payload in scripts/bench/corpus"
        raw_tok, pct = per[name]
        seen.add(name)
        assert _num(raw_txt) == raw_tok, (
            f"{doc}: §1 {name} raw tokens published {raw_txt}, measured {raw_tok:,}")
        assert _rounds_to(pct, pct_txt), (
            f"{doc}: §1 {name} published {pct_txt}%, measured {pct:.4f}% — "
            f"re-run `uv run scripts/bench/benchmark.py` and update the table")
        # The record count is published too, so it can go stale on its own.
        recs = len(json.loads((CORPUS / f"{name}.json").read_text(encoding="utf-8")))
        expected = recs if isinstance(json.loads(
            (CORPUS / f"{name}.json").read_text(encoding="utf-8")), list) else 1
        assert int(records) == expected, (
            f"{doc}: §1 {name} publishes {records} records, corpus has {expected}")

    # A payload silently dropped from the table is the same defect as a stale cell.
    assert seen == set(per), f"{doc}: §1 covers {sorted(seen)}, corpus has {sorted(per)}"


@pytest.mark.parametrize("doc", DOCS)
def test_s1_weighted_total_matches_the_codec(doc):
    _, raw_sum, total_pct = _measure_s1()
    m = _S1_TOTAL.search(_table(doc, _S1_HEADER[doc]))
    assert m, f"{doc}: no §1 weighted-total row found"
    assert _num(m[1]) == raw_sum, (
        f"{doc}: §1 weighted raw published {m[1]}, measured {raw_sum:,}")
    assert _rounds_to(total_pct, m[2]), (
        f"{doc}: §1 weighted total published {m[2]}%, measured {total_pct:.4f}%")


@pytest.mark.parametrize("doc", DOCS)
def test_s3_diff_table_matches_diff_demo(doc):
    """§3's "full re-send" column is the single-shot codec, so it moves with every codec
    change exactly as §1 does — and it was missed on the first pass at #206."""
    per, full_sum, diff_sum, total_pct = _measure_s3()
    table = _table(doc, _S3_HEADER[doc])
    rows = _S3.findall(table)
    assert rows, f"{doc}: parsed no §3 rows"

    for name, full_txt, diff_txt, pct_txt in rows:
        assert name in per, f"{doc}: §3 row {name!r} has no diffable payload"
        _recs, full, diff = per[name]
        assert _num(full_txt) == full, (
            f"{doc}: §3 {name} full re-send published {full_txt}, measured {full:,} — "
            f"re-run `uv run scripts/bench/diff_demo.py`")
        assert _num(diff_txt) == diff, (
            f"{doc}: §3 {name} diff published {diff_txt}, measured {diff:,}")
        assert _rounds_to(100.0 * (1 - diff / full), pct_txt), (
            f"{doc}: §3 {name} published {pct_txt}%, measured {100 * (1 - diff / full):.4f}%")

    t = _S3_TOTAL.search(table)
    assert t, f"{doc}: no §3 weighted-total row"
    assert _num(t[1]) == full_sum, (
        f"{doc}: §3 total full re-send published {t[1]}, measured {full_sum:,}")
    assert _num(t[2]) == diff_sum, (
        f"{doc}: §3 total diff published {t[2]}, measured {diff_sum:,}")
    assert _rounds_to(total_pct, t[3]), (
        f"{doc}: §3 total published {t[3]}%, measured {total_pct:.4f}%")


def test_s4_terse_column_agrees_with_s1():
    """§4 reprints §1's terse numbers in a differently-shaped table, so it drifts on its
    own — it did, and the first pass at #206 had to hand-patch it."""
    per, _, _ = _measure_s1()
    rows = _S4.findall(_table("BENCHMARKS.md", _S4_HEADER))
    assert rows, "parsed no §4 rows"

    for name, raw_txt, pct_txt in rows:
        assert name in per, f"§4 row {name!r} has no payload in scripts/bench/corpus"
        raw_tok, pct = per[name]
        assert _num(raw_txt) == raw_tok, (
            f"§4 {name} raw tokens published {raw_txt}, measured {raw_tok:,}")
        assert _rounds_to(pct, pct_txt), (
            f"§4 {name} terse published {pct_txt}%, measured {pct:.4f}% — "
            f"§4's terse column must agree with §1's")


def test_the_two_documents_agree_where_they_overlap():
    """Updating one document and forgetting the other is the obvious next failure, and
    without this it is invisible — each file would still agree with the codec on the rows
    it happens to publish.

    §1 is a straight copy, so require full equality. §3 is not: README prints all six
    diffable payloads, BENCHMARKS prints a three-row excerpt of the same run (its totals
    still cover all six, which the §3 test above checks against `diff_demo`). So compare
    the intersection there, and assert the intersection is non-empty so an editorial change
    that leaves them disjoint can't turn this into a no-op.
    """
    s1 = {doc: {r[0]: r[1:] for r in _S1.findall(_table(doc, _S1_HEADER[doc]))}
          for doc in DOCS}
    assert s1["README.md"] == s1["BENCHMARKS.md"], "§1 differs between the documents"

    s3 = {doc: {r[0]: r[1:] for r in _S3.findall(_table(doc, _S3_HEADER[doc]))}
          for doc in DOCS}
    shared = set(s3["README.md"]) & set(s3["BENCHMARKS.md"])
    assert shared, "§3 tables share no rows — one of them is no longer being checked"
    for name in sorted(shared):
        assert s3["README.md"][name] == s3["BENCHMARKS.md"][name], f"§3 {name} differs"
