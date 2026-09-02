"""A stored envelope `shape` is a cache of `classify_shape(raw)`, never a fact (#355).

`7be9d41` (#208/#204) relaxed `_find_record_list` from an identical-keyset rule to the
codec's union-schema `is_tabularizable`. Every envelope captured before it and never
re-recorded kept the old bucket, so `terse measure` printed two shape tables 36 payloads
apart in ONE report — Coverage read the stored field, the savings table re-classified —
and the codec verdict filed groups under a shape the codec no longer assigns.

These pin the invariant the fix buys: a stored shape that disagrees with
`classify_shape(raw)` cannot reach a consumer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from terse import capture, codeceval, measure, report
from terse.dropeval import Turn

# The exact #355 shape: keysets are NON-uniform, so this list qualifies as records only
# under the post-#208 union-schema rule. Under the old identical-keyset rule it was
# `compact-json` — which is what a pre-#208 corpus still has stored on it.
DRIFTED = {"result": [{"id": 1, "name": "a"}, {"id": 2, "extra": True}]}
DRIFTED_RAW = json.dumps(DRIFTED)

# The same drift, but ALSO carrying a whole-object column on every record — so it yields a
# `deref` question and reaches `run_codec_fluency`'s per-`(tool, shape)` tagging. Non-uniform
# keysets do not block question generation: the generator selects out of the key INTERSECTION.
DRIFTED_DEREFABLE = [{"id": i, "meta": {"o": f"n{i}"}} for i in range(4)]
DRIFTED_DEREFABLE[3]["extra"] = True
STALE = "compact-json"
LIVE = "array-of-records"


def test_the_fixture_is_actually_drifted():
    """Guards the other tests: if `classify_shape` ever stops calling this payload
    records, every assertion below would pass for the wrong reason."""
    assert capture.classify_shape(DRIFTED_RAW) == LIVE != STALE


def test_envelope_shape_ignores_a_stale_stored_bucket():
    env = {"tool": "t", "shape": STALE, "raw": DRIFTED_RAW}
    assert capture.envelope_shape(env) == LIVE


def test_envelope_shape_falls_back_to_the_stored_value_with_no_raw_to_classify():
    """A hand-built envelope or a foreign corpus may carry no `raw`. There is nothing to
    re-derive from, so the stored label is the best available — not the default."""
    assert capture.envelope_shape({"tool": "t", "shape": STALE}) == STALE
    assert capture.envelope_shape({"tool": "t", "shape": STALE, "raw": {"not": "a str"}}) == STALE


def test_envelope_shape_uses_the_default_when_there_is_neither():
    assert capture.envelope_shape({"tool": "t"}) == "?"
    assert capture.envelope_shape({"tool": "t", "shape": ""}, "unknown") == "unknown"


def test_coverage_counts_the_live_bucket_not_the_stored_one():
    envs = [{"tool": "t", "shape": STALE, "raw": DRIFTED_RAW}]
    assert capture.coverage(envs)["by_shape"] == {LIVE: 1}


def test_load_corpus_leaves_the_stale_stored_field_alone(tmp_path):
    """Re-classification happens at the READ, and ONLY there. `load_corpus` deliberately
    does not rewrite the field: it would mutate a dict the caller owns, and it would erase
    the only signal that a corpus predates a classifier change — which a future "N
    envelopes carry a stale bucket" diagnostic has to be able to read.

    So the drift stays visible on the envelope while every consumer still reports live."""
    (tmp_path / "t__abc.json").write_text(json.dumps(
        {"tool": "t", "shape": STALE, "bytes": len(DRIFTED_RAW), "sha": "abc",
         "captured_at": 1, "raw": DRIFTED_RAW}))
    (loaded,) = capture.load_corpus(tmp_path)
    assert loaded["shape"] == STALE                        # evidence preserved
    assert capture.envelope_shape(loaded) == LIVE          # consumer sees live
    assert capture.coverage([loaded])["by_shape"] == {LIVE: 1}


def test_run_codec_fluency_stamps_the_live_shape():
    """The codec verdict is per `(tool, shape)`. A group headed by a shape the codec does
    not assign answers a question nobody asked.

    A drifted payload DOES reach this tagging, so the mis-filing was live, not theoretical.
    An earlier draft of this test claimed the opposite — that `fluency.gen_questions` needs
    uniform records, so no #355 payload could ever produce a `deref`. That is wrong twice
    over: the generator picks its `blobcol` out of `_intersection_cols` (the keys present on
    EVERY record), and a non-uniform record list can carry such a column perfectly well —
    `DRIFTED_DEREFABLE` below is one, and 2 of the live corpus's 36 drifted envelopes (both
    `kb.read.list_nodes`) are others. Pre-fix they filed under `compact-json` for a payload
    the codec tabularizes, which is precisely the wrong bucket for a per-`(tool, shape)`
    safety verdict."""
    raw = json.dumps(DRIFTED_DEREFABLE)
    assert capture.classify_shape(raw) == LIVE
    assert len({frozenset(r) for r in DRIFTED_DEREFABLE}) > 1   # non-uniform, i.e. drifted
    assert codeceval.gen_codec_questions(DRIFTED_DEREFABLE)     # ...and still derefable
    envs = [{"tool": "demo.get", "shape": STALE, "sha": "abc", "raw": raw}]
    rows = codeceval.run_codec_fluency(envs, {"m1": lambda m: Turn(text="", tool_calls=[])})
    assert rows["m1"]
    assert {r["shape"] for r in rows["m1"]} == {LIVE}


def _table_after(md: str, header: str) -> dict[str, int]:
    """Parse the `| label | count |` rows of the markdown table whose header row is
    `header`, stopping at the first blank line. Reads the RENDERED report, so a row the
    renderer drops is invisible here too — which is the point."""
    lines = md.splitlines()
    start = lines.index(header) + 2          # skip the header and its `|---|---|` rule
    out: dict[str, int] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().strip("`*") for c in line.strip("|").split("|")]
        if cells[0] == "ALL":                # the savings table's total row
            continue
        out[cells[0]] = int(cells[1])
    return out


def test_the_two_shape_tables_of_one_measure_report_agree(tmp_path):
    """The reported symptom, end to end, through the REAL renderer.

    `build_report` prints a Coverage → "Shape bucket" table fed by `capture.coverage`, and a
    "Tier-0 savings by shape bucket" table a few lines below fed by `measure.measure_corpus`'s
    own `classify_shape`. #355 is the two disagreeing. This asserts on the rendered markdown
    rather than on the two functions, because a test that re-implements the renderer's row
    loop cannot see the renderer drop a row — an earlier draft did exactly that, and deleting
    the `array-of-records` line from the Coverage table left it green."""
    for i, (raw, stored) in enumerate([(DRIFTED_RAW, STALE),
                                       (json.dumps({"a": 1}), LIVE),      # drifted the other way
                                       ("plain text", "pretty-json")]):   # not JSON at all
        (tmp_path / f"t__{i}.json").write_text(json.dumps(
            {"tool": "t", "shape": stored, "sha": str(i), "captured_at": i, "raw": raw}))

    envelopes = capture.load_corpus(tmp_path)
    md = report.build_report(measure.measure_corpus(envelopes), capture.coverage(envelopes))

    from_coverage = _table_after(md, "| Shape bucket | Payloads |")
    from_savings = _table_after(md, "| Shape | n | raw tok | terse tok | saved | % |")

    assert from_coverage        # both tables actually rendered rows
    assert from_savings
    assert from_coverage == from_savings
    # Exact buckets, so "they agree" cannot be satisfied by both being wrong the same way.
    # Pre-fix the Coverage side read the three stored labels verbatim and printed
    # `{compact-json: 1, array-of-records: 1, pretty-json: 1}` while the savings table below
    # it computed `{array-of-records: 1, compact-json: 1, other: 1}` — the same report,
    # disagreeing on two of three buckets. Verified by evaluating both against this fixture.
    assert from_coverage == {LIVE: 1, STALE: 1, "other": 1}


# --------------------------------------------------------------------------- #
# The structural guard — the only thing that watches `scripts/bench/`
# --------------------------------------------------------------------------- #
ENVELOPE_SHAPE_READ = re.compile(r'\b(?:env|envelope)(?:\["shape"\]|\.get\("shape")')


def test_nothing_outside_envelope_shape_reads_a_stored_bucket():
    """`envelope_shape` is only the single mechanism while it is the single READER.

    A new consumer that reaches for `env["shape"]` reintroduces #355 in one line, and the
    behavioural tests above cannot see it — they exercise the consumers that exist today.
    Worse, `scripts/bench/` has NO test coverage at all, so reverting
    `text_alias_ceiling.py:406` to `env.get("shape") or classify_shape(raw)` was invisible
    to the entire suite. This is what catches that.

    Scoped to the idiom rather than to the key: `"shape"` names three different values in
    this tree — an envelope's stored bucket, a measured row's live one, and a diff marker's
    `"rows"`/`"keys"` — and only a read off a variable named `env`/`envelope` is the one
    that can be stale. Read the module docstring of `capture.envelope_shape` before adding
    an exemption here."""
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted([*(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ENVELOPE_SHAPE_READ.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")

    # The one legitimate read: `envelope_shape`'s own fallback, which is the mechanism.
    assert offenders == ['src/terse/capture.py:197: stored = env.get("shape")'], (
        "read an envelope's stored `shape` outside `capture.envelope_shape` — "
        "call `envelope_shape(env)` instead (#355):\n" + "\n".join(offenders))
