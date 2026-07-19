"""Property-based fuzzing of the lossless guarantee (M3).

The hand-curated corpus in `test_roundtrip.py` proves losslessness on shapes we thought
of; this proves it on shapes we didn't. Two properties, over generated inputs:

1. Every value in terse's promised domain — valid JSON — survives `compress`/`decompress`
   byte-for-byte (`roundtrip_ok`).
2. Every diff terse *chooses* to emit reconstructs its target exactly **after the JSON
   serialization the wire performs** — a stronger check than `diff_encode`'s own in-memory
   self-verification, which compares before serialization and so can't catch a value that
   round-trips in memory but not through JSON (int keys, tuples, NaN, ...).

Domain note (honest scope): the strategies generate VALID JSON only — finite floats (JSON
has no NaN/Inf) and no lone surrogates (not representable in a JSON string). Those are
outside terse's contract, not fuzzed here; the proxy fails open on anything it can't encode.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from terse import text_diff, transforms

FUZZ = settings(max_examples=400, deadline=None,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

# --- strategies: the valid-JSON value domain ---------------------------------------

# No lone surrogates (category Cs) — not representable in a JSON string.
_text = st.text(st.characters(blacklist_categories=("Cs",)), max_size=40)
_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**18), max_value=10**18),
    st.floats(allow_nan=False, allow_infinity=False),
    _text,
)
_json = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(_text, children, max_size=6),
    ),
    max_leaves=50,
)

# Record-list bias: arbitrary JSON rarely yields a uniform record array, so hammer the
# tabularize / dictionary / subtree-aliasing tiers directly. Keys drawn from a small pool
# so records are often uniform (folds) but sometimes not (must decline, still lossless).
_cell = st.one_of(
    st.none(), st.booleans(),
    st.integers(-1000, 1000),
    st.text(st.characters(blacklist_categories=("Cs",)), max_size=12),
    st.lists(st.integers(-50, 50), max_size=4),
    st.dictionaries(st.sampled_from(["x", "y", "z"]), st.integers(-50, 50), max_size=3),
)
_keys = st.sampled_from(["id", "name", "status", "url", "score", "owner", "tags"])
_record = st.dictionaries(_keys, _cell, min_size=1, max_size=5)
_records = st.lists(_record, max_size=14)


# --- property 1: lossless round-trip -----------------------------------------------

@given(_json)
@FUZZ
def test_fuzz_lossless_roundtrip_arbitrary_json(obj):
    assert transforms.roundtrip_ok(obj), "compress/decompress altered or dropped data"


@given(_records)
@FUZZ
def test_fuzz_lossless_roundtrip_records(records):
    # Both the bare list (row tabularization) and the wrapped shape (real MCP payloads).
    assert transforms.roundtrip_ok(records)
    assert transforms.roundtrip_ok({"result": records, "total": len(records)})


# --- property 2: an emitted diff reconstructs, even through JSON serialization ------

def _mutate_rows(records, data):
    """Derive curr from prev with row-level edits (the poll-again pattern the row diff
    targets) so `diff_encode` actually produces a diff to exercise."""
    curr = [dict(r) for r in records]
    for i in range(len(curr)):
        if data.draw(st.booleans()):
            curr[i] = data.draw(_record)
    if curr and data.draw(st.booleans()):
        del curr[data.draw(st.integers(0, len(curr) - 1))]
    for _ in range(data.draw(st.integers(0, 3))):
        curr.append(data.draw(_record))
    return curr


@given(_records, st.data())
@FUZZ
def test_fuzz_row_diff_reconstructs_after_serialization(records, data):
    prev, curr = records, _mutate_rows(records, data)
    diff = transforms.diff_encode(prev, curr)
    if diff is None:
        return  # no diff applies -> proxy sends the full form; nothing to reconstruct
    wire = json.loads(json.dumps(diff))          # what the wire actually does
    assert transforms.diff_decode(prev, wire) == curr


@given(st.dictionaries(_keys, _cell, max_size=7), st.data())
@FUZZ
def test_fuzz_key_diff_reconstructs_after_serialization(prev, data):
    curr = dict(prev)
    for k in list(curr):
        roll = data.draw(st.integers(0, 2))
        if roll == 0:
            curr[k] = data.draw(_cell)
        elif roll == 1:
            del curr[k]
    for _ in range(data.draw(st.integers(0, 3))):
        curr[data.draw(_keys)] = data.draw(_cell)
    diff = transforms.diff_encode(prev, curr)
    if diff is None:
        return
    wire = json.loads(json.dumps(diff))
    assert transforms.diff_decode(prev, wire) == curr


@given(st.text(max_size=400), st.text(max_size=60), st.data())
@FUZZ
def test_fuzz_text_diff_reconstructs_after_serialization(prev, insert, data):
    # curr = prev with a span deleted and text inserted, so it SHARES chunks with prev and
    # the CDC differ produces a real delta (not just a full re-send) to reconstruct.
    if prev:
        i = data.draw(st.integers(0, len(prev)))
        j = data.draw(st.integers(i, len(prev)))
        curr = prev[:i] + insert + prev[j:]
    else:
        curr = insert
    diff = text_diff.text_diff_encode(prev, curr)
    if diff is None:
        return
    wire = json.loads(json.dumps(diff))
    assert text_diff.text_diff_decode(prev, wire) == curr


# --------------------------------------------------------------------------- #
# Text drop-to-retrieve (`$text.code_blocks`): the recoverability guarantee under
# adversarial *markdown*, where the hazard is the fence scanner rather than the codec —
# ragged indentation, tilde vs backtick fences, unterminated openers, fences nested
# inside longer fences, and text that already looks like a drop marker.
# --------------------------------------------------------------------------- #
_fence = st.sampled_from(["```", "````", "~~~", "```py", "   ```", "\t```", "~~~~"])
_line = st.one_of(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40),
    _fence,
    st.just('{"__terse_dropped__":"aaaaaaaaaaaaaaaa","bytes":9,"retrieve":"terse.retrieve"}'),
)


@given(st.lists(_line, max_size=40), st.integers(1, 600))
@FUZZ
def test_fuzz_text_drop_is_recoverable_or_untouched(lines, min_len):
    """The only two legal outcomes: the emitted text restores to the original byte for
    byte, or nothing was dropped at all. There is no third branch in which a payload is
    altered without being recoverable — that is the whole lossy-with-a-receipt claim."""
    from terse import lossy
    from terse.policy import Policy, Rule, apply

    raw = "\n".join(lines)
    rule = Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"),
                fields={lossy.TEXT_SELECTOR_CODE_BLOCKS:
                        {"lossy": "drop-to-retrieve", "min": min_len}})
    store: dict = {}
    out = apply(raw, "t", Policy(rules=[rule]), drop_sink=store.__setitem__).text
    if out == raw:
        return                       # nothing qualified / gate refused -> original intact
    assert lossy.restore_text_drops(out, store.__getitem__) == raw


@given(st.lists(_line, max_size=40))
@FUZZ
def test_fuzz_fenced_spans_are_disjoint_and_ordered(lines):
    """Spans must partition cleanly: non-overlapping, ascending, and always real slices —
    an overlap would make the back-to-front splice corrupt a neighbouring block."""
    from terse import lossy

    text = "\n".join(lines)
    spans = lossy.fenced_spans(text)
    assert all(0 <= s < e <= len(text) for s, e in spans)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
