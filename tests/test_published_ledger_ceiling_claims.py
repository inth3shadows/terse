"""Two absolute claims about `kb.read.list_principles` shipped in `BENCHMARKS.md` and
`docs/POSITIONING.md`, and both were false the moment they were written.

`BENCHMARKS.md` §5 published "3% -> 3% *(hard ceiling)* ... no tier combination changes
that", measured 2026-07-22 on a Tier-0-only, single-call corpus test. `docs/POSITIONING.md`
generalised it to "these servers ... nothing structural left to remove". Both are still
true of that ONE narrow measurement (Tier-0, isolated call) -- re-running it 2026-08-11
against 141 fresh captures reproduces ~3.5%, matching almost exactly. But the live
production ledger, the thing an operator actually experiences, read 15.1% blended for the
same tool the same day: a handful of large, multi-block calls routed through multiproxy
dominate the token-weighted average, which the narrow measurement never claimed to speak
for. "No tier combination changes that" and "nothing structural left to remove" asserted
more than the evidence supported, and nobody checked.

This is principle #134 again: a claim that must stay in step with a live, moving number
needs a test, not a note asking people to remember. This one is intentionally narrow --
it does not try to re-derive #202's percentages (`test_published_benchmarks.py` covers the
codec's *deterministic* corpus numbers; this file is about a *ledger-relative* absolute
claim that has no fixed corpus to recompute against). It only pins that the specific
retracted wording does not silently come back the next time someone edits these docs
around this topic.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COVERED = ("BENCHMARKS.md", "docs/POSITIONING.md")

# The exact phrases retracted in this PR. Matched case-sensitively and verbatim -- unlike
# the diff-default test, there is no family of rewrites to anticipate here, because these
# are direct quotes of specific sentences that were deleted, not a recurring claim shape
# that keeps getting re-expressed in new words.
_RETRACTED_PHRASES = (
    "no tier combination changes that",
    "nothing structural left to remove",
    "(hard ceiling)",
)


def test_retracted_ceiling_claims_do_not_reappear():
    for doc in COVERED:
        text = (REPO / doc).read_text(encoding="utf-8")
        for phrase in _RETRACTED_PHRASES:
            assert phrase not in text, (
                f"{doc}: retracted claim {phrase!r} has come back. The live production "
                f"ledger contradicts it (kb.read.list_principles read 15.1% blended on "
                f"2026-08-11 against the 3% this phrase was attached to) -- re-run "
                f"`terse stats` before republishing anything that asserts a hard ceiling.")


def test_the_narrow_tier0_measurement_this_claim_rested_on_is_still_documented_as_narrow():
    """The retraction removed an over-generalization, not the underlying measurement --
    BENCHMARKS.md §5 should still show the per-block -> joined comparison for
    `kb.read.list_principles`, just without asserting it bounds the tool's real number."""
    text = (REPO / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "kb.read.list_principles" in text
    assert "3% → 3%" in text, (
        "BENCHMARKS.md: the Tier-0-only per-block->joined figure for "
        "kb.read.list_principles is gone -- if it was removed rather than re-measured, "
        "the composition-vs-codec distinction this test protects no longer has evidence "
        "backing it in the doc.")
