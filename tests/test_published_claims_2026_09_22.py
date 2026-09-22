"""Guards for the claims the 2026-09-22 re-derivation retracted, and for the ones it
added that a future edit could quietly drop.

Three of the claims corrected that day were WRONG rather than merely stale, and each had
stood for weeks because nothing watched it:

  * "The only directly-comparable public tool is TOON" -- #298's sweep names ten projects
    in the niche, and `compressmcp` is a closer architectural match (an MCP-layer lossless
    JSON compressor). Measured in BENCHMARKS.md §4.
  * "TOON wins only on `gh_labels`" -- under TOON 4.1.1 it wins three of nine payloads.
  * "terse is new (pre-PyPI as of this date)" -- terse has shipped on PyPI since v0.3.1.

And the docs described `summarize` as "deferred" / "designed but not yet built" in five
places, which reads as a roadmap item. It is not: #261 closed it PERMANENTLY, because a
model in the proxy breaks terse's no-ML guarantee. That is a positioning claim, not a
scheduling one, and the wrong wording misrepresents what terse is.

This file follows `test_published_ledger_ceiling_claims.py`: it does not re-derive any
percentage (`test_published_benchmarks.py` owns the deterministic corpus numbers). It pins
that retracted WORDING does not come back, and that the honesty disclosures added that day
are still present.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_POSITIONING_DOCS = ("README.md", "BENCHMARKS.md", "docs/POSITIONING.md")
_LOSSY_TIER_DOCS = ("README.md", "TECHNICAL.md")


def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


# A retraction has to be able to QUOTE the sentence it retracts, so a bare substring ban
# would forbid the very edit that fixed this. The guard is on the claim being ASSERTED:
# every occurrence must sit next to language marking it as withdrawn.
_WITHDRAWAL_MARKERS = ("withdrawn", "retracted", "no longer", "is false", "superseded",
                       "previous edition", "earlier edition")
_WINDOW = 400


def _asserted_occurrences(text: str, phrase: str) -> list[int]:
    """Offsets where `phrase` appears WITHOUT a withdrawal marker nearby."""
    low = text.lower()
    bad, start = [], 0
    while (i := low.find(phrase, start)) != -1:
        window = low[max(0, i - _WINDOW): i + len(phrase) + _WINDOW]
        if not any(m in window for m in _WITHDRAWAL_MARKERS):
            bad.append(i)
        start = i + len(phrase)
    return bad


def test_the_only_comparable_tool_claim_is_never_asserted_again():
    """terse is not alone in this niche. The docs may quote the old claim to withdraw it;
    they may not state it."""
    for doc in _POSITIONING_DOCS:
        text = _read(doc)
        for phrase in ("only directly-comparable public tool",
                       "only directly comparable public tool"):
            assert not _asserted_occurrences(text, phrase), (
                f"{doc}: the retracted 'only directly-comparable tool' claim is asserted "
                f"again with no withdrawal marker within {_WINDOW} chars. #298 names ten "
                f"projects in this niche and BENCHMARKS.md §4 measures three of them -- "
                f"compressmcp is a closer architectural match than TOON.")


def test_the_withdrawal_guard_can_actually_fail():
    """A guard that cannot fail pins nothing. Feed it the claim with no withdrawal
    language and it must object; feed it the same claim next to 'withdrawn' and it must
    not."""
    asserted = "The only directly-comparable public tool is TOON, and nothing else is close."
    assert _asserted_occurrences(asserted, "only directly-comparable public tool")
    withdrawn = ("An earlier edition called TOON the only directly-comparable public "
                 "tool; that is withdrawn.")
    assert not _asserted_occurrences(withdrawn, "only directly-comparable public tool")


def test_star_counts_stay_out_of_the_docs():
    """Removed deliberately on 2026-09-22: they drifted constantly and never informed
    whether terse works. A star glyph is the cheap tell that one has been pasted back."""
    for doc in _POSITIONING_DOCS:
        text = _read(doc)
        # The removal note itself is allowed to say the word; the glyph is not.
        assert "★" not in text, (
            f"{doc}: a star count has come back. Popularity is not evidence about "
            f"compression -- publish what a project measurably DOES instead.")


def test_pre_pypi_adoption_claim_does_not_come_back():
    for doc in _POSITIONING_DOCS:
        assert "pre-PyPI" not in _read(doc), (
            f"{doc}: 'pre-PyPI' is false -- terse has shipped on PyPI since v0.3.1 "
            f"(2026-07-18).")


def test_summarize_is_documented_as_closed_not_deferred():
    """#261 closed `summarize` permanently. 'Deferred' / 'not yet built' reads as pending
    and misrepresents terse's deterministic-only positioning."""
    for doc in _LOSSY_TIER_DOCS:
        text = _read(doc)
        if "summarize" not in text:
            continue
        for phrase in ("`summarize` remains designed but",
                       "`summarize` is accepted by the schema but deferred",
                       "`summarize` deferred"):
            assert phrase not in text, (
                f"{doc}: {phrase!r} presents `summarize` as pending. It is closed "
                f"PERMANENTLY (#261) -- a model in the proxy breaks the no-ML guarantee.")
        assert "#261" in text, (
            f"{doc}: mentions `summarize` without citing #261, the issue that closed it "
            f"permanently. Without the citation a reader reads 'deferred'.")


def test_the_fleet_headline_never_appears_without_its_lossless_only_pair():
    """§5's 23.3% blended includes `codegraph_explore`'s drop-to-retrieve rule -- 22.2% of
    all savings from the fleet's ONE lossy row. terse's wedge is 'unconditionally
    lossless', so the lossless-only figure must travel with the blended one."""
    for doc in ("BENCHMARKS.md", "docs/POSITIONING.md"):
        text = _read(doc)
        if "23.3%" not in text:
            continue
        assert "19.3%" in text, (
            f"{doc}: publishes the 23.3% blended fleet figure without the 19.3% "
            f"lossless-only pair. One lossy tool supplies 22.2% of that saving.")
        assert "codegraph_explore" in text, (
            f"{doc}: publishes the blended fleet figure without naming "
            f"`codegraph_explore`, the lossy row responsible for the gap.")


def test_the_cost_basis_disclosure_is_still_published():
    """Derived 2026-09-22 over 1,093 sessions: 96.9% of raw input tokens are cache reads
    billed at 0.1x, so a raw-token basis overstates input cost by 7.47x. This is the
    caveat a skeptic checks first; it may not quietly disappear."""
    bench = _read("BENCHMARKS.md")
    assert "7.47x" in bench, (
        "BENCHMARKS.md: the cost-basis derivation is gone. A token reduction is not a "
        "proportional cost reduction, and this document says so on measured grounds.")
    assert "cache read" in bench.lower()
    positioning = _read("docs/POSITIONING.md")
    assert "7.47x" in positioning, (
        "docs/POSITIONING.md: the economic model must state the billed-cost basis, not "
        "only the token basis.")


def test_toon_is_measured_at_the_version_the_docs_name():
    """§1 was re-run on TOON 4.1.1; the pinned encoder must agree with the published
    column, or the table is measuring something the reader cannot reproduce."""
    import json

    pkg = json.loads(_read("scripts/bench/package.json"))
    pinned = pkg["dependencies"]["@toon-format/toon"]
    assert pinned == "4.1.1", (
        f"scripts/bench/package.json pins TOON {pinned}, but BENCHMARKS.md §1 publishes a "
        f"'TOON 4.1.1' column. Re-run scripts/bench/benchmark.py and update both together.")
    assert "TOON 4.1.1" in _read("BENCHMARKS.md")


def test_the_competitor_column_encoder_is_pinned_and_present():
    """§4/§6's compressmcp column must be reproducible by anyone, not a local one-off."""
    import json

    pkg = json.loads(_read("scripts/bench/package.json"))
    assert pkg["dependencies"].get("compressmcp") == "0.4.0", (
        "scripts/bench/package.json must pin the compressmcp version the docs publish.")
    assert (REPO / "scripts/bench/compressmcp_encode.mjs").is_file(), (
        "scripts/bench/compressmcp_encode.mjs is missing -- §4 and §6 publish a "
        "compressmcp column that nobody else could then reproduce.")


def test_the_router_per_turn_cadence_is_documented_beside_the_topology_advice():
    """§7 exists because the pre-#211 A/B table ('fold to escape the token tax') is now
    backwards for small fleets. The cadence distinction is the whole reason."""
    bench = _read("BENCHMARKS.md")
    assert "per-turn" in bench and "once/session" in bench, (
        "BENCHMARKS.md §7: the router-vs-standalone primer CADENCE distinction is the "
        "load-bearing fact. Without both cadences named, the old 'always fold' advice "
        "reads as current.")
    assert "pre-#211" in bench, (
        "BENCHMARKS.md §7: the +23.1%/+0.0% A/B table must stay labelled pre-#211, or it "
        "will be read as current guidance.")
