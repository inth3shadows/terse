"""Every primer token count published in prose must be a size the live primer actually has.

The primer's size is quoted in a dozen places across README, USAGE, POSITIONING and
`policy.py` — as the full-gate ceiling (555), as the diff paragraph's share of it
(190 of 555), as the dropped-field paragraph (64), and since #224 as the primer a real
install pays (248, or 312 where the dropped-field paragraph is reachable — which is NOT
the same as "this server has a drop rule", see `Policy.has_drop`). None of it was checked
by anything.

Two claims are pinned here, at different strengths, and the gap between them is #224:
membership (any published size must be A live size) is what the sweep can check
everywhere, but a published *decision threshold* gets the narrower claim that it must be
a size a real install actually pays. 555 satisfies the first and fails the second, which
is exactly how it survived as the published wrap/don't-wrap bar.

It has already moved once: #202 took `PRIMER_TABLE` from 55 to 155 cl100k tokens and the
whole primer from 402 to 555, and every one of those prose sites had to be found and
updated by hand. The next edit to a primer paragraph moves them all again, silently, and
the reader has no way to tell a current figure from a stale one — which is exactly how
`USAGE.md`'s multiproxy rationale went on citing the pre-#211 primer architecture and its
pre-#211 numbers long after #211 landed. That is principle #134: two things that must stay
in step need a test that parses both, not a note asking people to remember.

The claim pinned here is deliberately *membership*, not equality per site: a number quoted
as a primer size must be one of the live section sizes or the live full-gate total. That is
the strongest claim available without teaching a regex which of the eleven sites means
which section, and it fails the moment a paragraph is edited — every stale figure stops
being a live size at once. `test_positioning_publishes_the_live_section_breakdown` pins the
exact per-section mapping separately, because POSITIONING is the one document that
publishes the breakdown rather than a single number.

NOT covered, deliberately:
  * `src/terse/proxy.py`. Its primer comments record REJECTED alternatives on purpose — an
    87-token encoding that was measured and not adopted, a 68-token delta — and those are
    not live sizes. A sweep would demand they be "corrected", destroying the record of why
    the current encoding was chosen (#168).
  * `CHANGELOG.md`, which is historical by construction: "402 -> 555" must keep saying 402.
  * The A/B session percentages the same paragraphs quote (-3.8% / -3.5% at 1/3/6 servers,
    +23.1% pre-#211). Those come from a live multi-turn client session against real models
    and cannot be recomputed here; they are attributed to the PR that measured them instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from terse.policy import default_policy, load_policy
from terse.proxy import (
    PRIMER_DICT,
    PRIMER_DIFF,
    PRIMER_DROPPED,
    PRIMER_EMBEDDED,
    PRIMER_HEAD,
    PRIMER_TABLE,
    PRIMER_TAIL,
    _assemble_primer,
    build_primer,
)
from terse.tokenize import count_cl100k

REPO = Path(__file__).resolve().parent.parent

# The names are POSITIONING's row labels, so the table test can look each row up directly.
SECTIONS = {"head": PRIMER_HEAD, "table": PRIMER_TABLE, "dict": PRIMER_DICT,
            "embedded": PRIMER_EMBEDDED, "diff": PRIMER_DIFF, "dropped": PRIMER_DROPPED,
            "tail": PRIMER_TAIL}

# Every gate on, which is what "full-gate ceiling" means: the largest primer any policy can
# produce. Computed, not summed from SECTIONS — the assembly is what ships, and a future
# joiner or separator between sections would make the sum a different number from the truth.
FULL = count_cl100k(_assemble_primer(table=True, dictionary=True, diff=True, dropped=True,
                                     embedded=True))

# Sizes a real install actually PAYS, as against sizes the primer can be made to have.
# The distinction is #224: 555 is a live size, so the membership sweep below passed on it
# for months while it was published as the wrap/don't-wrap bar — and no standalone entry
# under a default policy ever pays 555, because `diff` is off (#170) and embedded/dropped
# are per-rule. A number an operator ACTS on has to clear the higher bar.
_EXAMPLE = load_policy(str(REPO / "policy.example.json"))
INSTALLED = {count_cl100k(build_primer(default_policy()))} | {
    count_cl100k(build_primer(_EXAMPLE, s))
    for s in ("runecho", "kb", "codegraph", "secret-broker", "gh", None)
} - {0}

COVERED = ("README.md", "USAGE.md", "docs/POSITIONING.md", "src/terse/policy.py")

# Three shapes the prose actually uses. The first two carry their own unit and are
# unambiguous at any magnitude; the third is bare prose and is therefore capped at three
# digits and required to sit next to the word "primer", because "23,962 tokens saved per
# call" is not a primer size and appears one line above one that is.
_EXPLICIT = re.compile(r"(?<![\d,.])(\d[\d,]*)\s*cl100k\s+tok(?:en)?s?\b", re.I)
# Up to two intervening words, so "64-token dropped-field paragraph" is found. Without it
# the sweep silently skipped that site, which `test_the_sweep_actually_finds_...` caught on
# its first run — the completeness guard earning its place immediately.
_ATTRIB = re.compile(
    r"(?<![\d,.])(\d[\d,]*)-tok(?:en)?\s+(?:[\w-]+\s+){0,2}(?:primer|paragraph)", re.I)
_BARE = re.compile(r"(?<![\d,.])(\d{1,3})\s+tok(?:en)?s?\b(?![\d,])", re.I)
_BARE_WINDOW = 60

# A mention that wraps across lines is still a mention. `190 of 555 cl100k\n    # tokens` in
# a Python comment matched nothing until this collapsed the break, and hard-wrapped markdown
# prose can split the same phrase at any point. The `#` strip is Python-only: in markdown a
# line-leading `#` is a heading, and folding it away would glue a heading onto the previous
# paragraph and invent adjacency the reader never sees.
# A BLANK line is a paragraph break, and folding it away would put the end of one paragraph
# within `_BARE`'s 60-character proximity window of the start of the next — synthesising an
# adjacency the reader never sees. Only single breaks (a hard wrap inside one paragraph) are
# collapsed; a blank line stays a newline, which no pattern here spans.
_JOIN_MD = re.compile(r"\n(?![ \t]*\n)[ \t]*")
_JOIN_PY = re.compile(r"\n(?![ \t]*(?:#[ \t]*)?\n)[ \t]*#?[ \t]*")

# "the diff paragraph is 190 of 555 cl100k tokens" — a section against the total, in one
# sentence. Checked as a PAIR as well as individually, because the individual check would
# pass if the two were swapped.
_PAIR = re.compile(r"(?<![\d,.])(\d[\d,]*) of (\d[\d,]*)\s*cl100k\s+tok", re.I)


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def _flat(doc: str) -> tuple[str, list[int]]:
    """One file with its line breaks (and Python comment prefixes) collapsed to spaces,
    plus a map from each flattened offset back to its offset in the original — so a match
    can still be reported at the line a human would look at."""
    text = (REPO / doc).read_text(encoding="utf-8")
    joiner = _JOIN_PY if doc.endswith(".py") else _JOIN_MD
    out: list[str] = []
    omap: list[int] = []
    i = 0
    for m in joiner.finditer(text):
        out.append(text[i:m.start()])
        omap.extend(range(i, m.start()))
        out.append(" ")
        omap.append(m.start())
        i = m.end()
    out.append(text[i:])
    omap.extend(range(i, len(text)))
    return "".join(out), omap


def _mentions(doc: str) -> list[tuple[int, int]]:
    """(line number, token count) for every primer size quoted in one file."""
    text = (REPO / doc).read_text(encoding="utf-8")
    flat, omap = _flat(doc)

    def line_of(pos: int) -> int:
        return text[:omap[pos]].count("\n") + 1

    found: list[tuple[int, int]] = []
    for pattern in (_EXPLICIT, _ATTRIB):
        for m in pattern.finditer(flat):
            found.append((line_of(m.start()), _num(m.group(1))))
    for m in _BARE.finditer(flat):
        window = flat[max(0, m.start() - _BARE_WINDOW):m.end() + _BARE_WINDOW]
        if "primer" in window.lower():
            found.append((line_of(m.start()), _num(m.group(1))))
    return found


def test_positioning_publishes_the_live_section_breakdown():
    """POSITIONING is the only document with the per-section table, so it is the only place
    the exact section->size mapping can be checked rather than mere membership. It is also
    the table an edit to any single paragraph invalidates one row at a time, which is the
    drift the membership sweep is weakest against."""
    text = (REPO / "docs" / "POSITIONING.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| (\w+) \| (\d+) \|$", text, re.M))
    assert set(rows) == set(SECTIONS), (
        f"POSITIONING's primer table covers {sorted(rows)}, the primer has "
        f"{sorted(SECTIONS)} — a section was added or renamed without the table following")
    for name, published in rows.items():
        live = count_cl100k(SECTIONS[name])
        assert int(published) == live, (
            f"POSITIONING publishes {published} cl100k tokens for the `{name}` primer "
            f"section; it is {live}")
    total = re.search(r"^\| \*\*full \(all sections gated on\)\*\* \| \*\*(\d+)\*\* \|$",
                      text, re.M)
    assert total, "POSITIONING's primer table lost its full-gate total row"
    assert int(total.group(1)) == FULL


def test_every_published_primer_size_is_a_size_the_primer_actually_has():
    live = {count_cl100k(s) for s in SECTIONS.values()} | {FULL} | INSTALLED
    for doc in COVERED:
        for line, published in _mentions(doc):
            assert published in live, (
                f"{doc}:{line} publishes {published:,} as a primer size, but the live "
                f"primer's sizes are {sorted(live)} (full-gate total {FULL}). Either a "
                f"primer paragraph was edited without the prose following it, or this is "
                f"a non-primer figure that happens to be measured in cl100k tokens — "
                f"BENCHMARKS.md already has one of those, which is why it is not COVERED. "
                f"If a covered file gains one, narrow the pattern rather than the claim.")


def test_the_published_wrap_threshold_is_a_primer_a_real_install_pays():
    """#224. The membership sweep above is satisfied by any live size, and 555 is one — so
    it passed for months while POSITIONING published 555 as the wrap/don't-wrap bar and a
    default install paid 248. A threshold is the one primer figure a reader ACTS on, in the
    direction that loses savings (unwrap a server that was in fact paying), so it gets the
    narrower claim: it must be a primer some real install actually pays, not merely one the
    assembly can be made to produce.

    Pinned against the rule's own sentence rather than a bare number sweep, because the
    defect was never a wrong arithmetic result — 555 was correct as a ceiling. It was a
    correct number doing a job it did not fit."""
    flat, _ = _flat("docs/POSITIONING.md")
    m = re.search(r"one-time primer\D{0,4}(\d[\d,]*) tokens, or (\d[\d,]*) where the "
                  r"dropped-field paragraph is reachable", flat)
    assert m, ("POSITIONING's wrap rule no longer states its threshold in the pinned form. "
               "If the wording changed, update this regex — do not delete the assertion; "
               "an unpinned threshold is what #224 was.")
    default_bar, drop_bar = _num(m.group(1)), _num(m.group(2))
    assert default_bar == count_cl100k(build_primer(default_policy())), (
        f"POSITIONING publishes {default_bar} as the default wrap threshold; a default "
        f"policy's primer is {count_cl100k(build_primer(default_policy()))}")
    # Structural, NOT `build_primer(_EXAMPLE, "kb")`. kb carries no drop rule of its own —
    # it pays 312 only because `codegraph.*` precedes `kb.*` in the example policy and
    # `has_drop`'s walk is deliberately conservative. Pinning the pair to kb would make a
    # benign reordering of policy.example.json fail this test with a message blaming
    # POSITIONING, and would re-publish the "server carries a drop rule" story that was
    # wrong in the first place. The claim that actually holds is the arithmetic one.
    # Against the real assembly, NOT `default_bar + count_cl100k(PRIMER_DROPPED)`. cl100k
    # counting is not additive across concatenation in general — that sum happens to hold
    # only because of where the current section boundaries fall. Asserting the sum would
    # fail, blaming POSITIONING, the day a section's leading token merges with a preceding
    # suffix. What ships is the assembly, so that is what the doc is checked against.
    with_dropped = count_cl100k(_assemble_primer(table=True, dictionary=True, diff=False,
                                                 dropped=True))
    assert drop_bar == with_dropped, (
        f"POSITIONING publishes {drop_bar} as the with-dropped-paragraph threshold; the "
        f"default primer plus the dropped-field paragraph assembles to {with_dropped}")
    assert {default_bar, drop_bar} <= INSTALLED, (
        f"a published threshold must be a primer some real install pays; INSTALLED is "
        f"{sorted(INSTALLED)}")
    assert FULL not in (default_bar, drop_bar), (
        "the wrap threshold is the all-gates ceiling again — that is #224 verbatim")


def test_positioning_publishes_the_default_primer_as_its_own_summands():
    """Membership cannot separate 248 from 312 — both are live installed sizes, so either
    can be swapped for the other anywhere in a covered file and every other check here
    still passes. POSITIONING happens to publish the addition itself ("head 41 + table 155
    + dict 44 + tail 8"), which makes the total checkable against its own stated parts
    rather than against a set. That closes the one substitution the sweep is blind to."""
    flat, _ = _flat("docs/POSITIONING.md")
    m = re.search(r"pays \*\*(\d[\d,]*) cl100k tokens\*\* — head (\d+) \+ table (\d+) "
                  r"\+ dict (\d+) \+ tail (\d+)", flat)
    assert m, "POSITIONING no longer publishes the default primer as a sum of its sections"
    total, parts = _num(m.group(1)), [int(g) for g in m.groups()[1:]]
    assert sum(parts) == total, (
        f"POSITIONING publishes {total} as the default primer but its own summands "
        f"{parts} add to {sum(parts)}")
    for name, published in zip(("head", "table", "dict", "tail"), parts, strict=True):
        assert published == count_cl100k(SECTIONS[name]), (
            f"POSITIONING publishes {published} for the `{name}` section; it is "
            f"{count_cl100k(SECTIONS[name])}")
    assert total == count_cl100k(build_primer(default_policy()))
    # Every "one-time N-token primer" phrase prices that same standalone primer — the table
    # header and the 464x sentence both do. Without this, swapping either one to 312 left
    # them disagreeing with each other AND with the sum above, and every check still passed,
    # because 248 and 312 are both live installed sizes.
    priced = [_num(x.group(1))
              for x in re.finditer(r"one-time (\d[\d,]*)-tok(?:en)? primer", flat)]
    assert priced, "POSITIONING no longer prices the one-time primer anywhere"
    assert all(p == total for p in priced), (
        f"POSITIONING prices the one-time primer at {sorted(set(priced))} in different "
        f"places; its own summands make it {total}")
    # The rule restates the bar in round numbers ("can't clear ~250") two lines from the
    # pinned sentence. Left unchecked, editing that alone back to ~550 re-publishes
    # 555-as-threshold — #224's exact failure mode — with every other assertion green.
    for rounded in re.finditer(r"clear ~(\d[\d,]*)", flat):
        assert _num(rounded.group(1)) == round(total, -1), (
            f"POSITIONING rounds the wrap bar to ~{rounded.group(1)}; the primer it must "
            f"round is {total} (~{round(total, -1)})")


def test_positionings_router_ceiling_is_the_full_gate_total():
    """The membership sweep cannot catch a ceiling swapped for another live size: with 248
    and 312 now legitimately published, rewriting "555 cl100k tokens is the hard ceiling"
    to say 312 passes membership while inverting the router's whole argument. The ceiling
    is a specific claim about a specific number, so it gets a specific assertion."""
    flat, _ = _flat("docs/POSITIONING.md")
    m = re.search(r"(\d[\d,]*) cl100k tokens is the hard ceiling", flat)
    assert m, "POSITIONING's router section lost its stated hard ceiling"
    assert _num(m.group(1)) == FULL, (
        f"POSITIONING calls {_num(m.group(1)):,} the router's hard ceiling; the full-gate "
        f"primer is {FULL}")


def test_a_section_quoted_against_the_total_is_the_smaller_of_the_two():
    """`190 of 555` — both halves are live sizes even if they are swapped, so membership
    alone would pass on `555 of 190`. A section is never the whole primer."""
    for doc in COVERED:
        text = (REPO / doc).read_text(encoding="utf-8")
        flat, omap = _flat(doc)
        for m in _PAIR.finditer(flat):
            part, whole = _num(m.group(1)), _num(m.group(2))
            line = text[:omap[m.start()]].count("\n") + 1
            assert whole == FULL, (
                f"{doc}:{line} quotes a section against {whole:,}, but the full-gate "
                f"primer is {FULL}")
            assert part in {count_cl100k(s) for s in SECTIONS.values()}, (
                f"{doc}:{line} quotes {part:,} as a primer section")
            assert part < whole


def test_the_sweep_actually_finds_the_prose_it_claims_to_cover():
    """A doc test whose regex matches nothing passes forever while pinning nothing — the
    failure mode that let these figures drift in the first place. Every covered file is
    asserted to contain at least one mention, so a reworded paragraph that slips out of all
    three patterns fails here instead of silently dropping out of coverage."""
    per_file = {doc: len(_mentions(doc)) for doc in COVERED}
    assert all(per_file.values()), f"no primer size found in: {[d for d, n in per_file.items() if not n]}"
    assert sum(per_file.values()) >= 10, per_file
    # Was >= 3, which #224's follow-up had to lower deliberately rather than route around.
    # The three `190 of 555` sites this counted were README, USAGE and `policy.py`, and all
    # three were describing a STANDALONE server's primer as 555 — the same defect #224 was
    # filed for. Fixing them emptied the pool, so a floor of 3 would have been a tripwire
    # that fires when you fix the bug. The one remaining pair is POSITIONING's ROUTER
    # section, where quoting a section against 555 is correct, and one live pair is all
    # `test_a_section_quoted_against_the_total...` needs to be non-vacuous.
    assert sum(1 for doc in COVERED for _ in _PAIR.finditer(_flat(doc)[0])) >= 1
