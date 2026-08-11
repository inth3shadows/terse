"""No shipped document may claim the cross-call diff tier is on by default while it isn't.

`Policy.diff` has been `False` since #170 flipped it: the tier is correct, but its primer
paragraph costs orders of magnitude more than the tier banks at the measured 0.38% hit
rate. #76 had flipped it the other way, and #170's revert updated the code, `USAGE.md` and
`BENCHMARKS.md` while leaving prose sites in `README.md` and `TECHNICAL.md` still
announcing "Default-on since its validation program completed". Two of them sat four lines
apart in the same README section — one saying OPT-IN, the next saying default-on.

That is the defect this file prevents, and it is not hypothetical rot: the contradiction
survived from #170 (2026-07-28) to 2026-08-11 unnoticed, because nothing checked. It is
principle #134 — two things that must stay in step need a test that parses both, not a
note asking people to remember. A GitHub issue would have rotted alongside the docs; a
test fails the pull request that reintroduces the claim.

The claim pinned here is deliberately ONE-DIRECTIONAL, and the direction matters:

  * While a default policy's `diff` is `False`, no covered document may assert the tier is on by
    default. That is the direction a reader is harmed in — they believe they are already
    getting diffs, never pass `--diff`, and the tier they are reading about never runs.
  * The mirror claim ("if the default ever flips back, no doc may say OFF") is NOT
    asserted. Flipping the default is a deliberate, reviewed act that edits `policy.py`;
    whoever does it rewrites the docs in the same breath. Silent rot only happens in the
    direction nobody is looking.

Matching is on ASSERTIONS, not on "diff" and "default" appearing near each other.
`USAGE.md` says "**OFF by default**", `BENCHMARKS.md` says "no longer the zero-config
default", `README.md` says "OPT-IN" and `TECHNICAL.md` says "Default-off since #170" —
every one of those would trip a proximity heuristic, and narrowing the docs to satisfy a
bad sweep would be fixing the test by breaking the documentation. So the patterns require
a default-ON predicate, and a match is exempted only when a *different* setting that
genuinely is on by default is the nearer subject (`_TRUE_DEFAULT_ON`, each entry grounded
against the live `Policy` field rather than asserted in a comment).

NOT covered, deliberately:
  * `CHANGELOG.md`. Historical by construction: #76's entry ("flip cross-call diffing to
    default-on") must keep saying exactly that, and #170's entry describes the flip back.
    Rewriting either would destroy the record of what happened when.
  * `src/terse/**`. `policy.py` IS the ground truth read here, and `proxy.py`'s primer
    text is pinned by `test_published_primer_sizes.py`. A prose sweep over source would
    also hit comments recording rejected alternatives, which are not claims about today.
"""

from __future__ import annotations

import re
from pathlib import Path

from terse.policy import default_policy

REPO = Path(__file__).resolve().parent.parent

# Every hand-maintained document that tells a user what the diff tier's default is. All of
# them are swept, including the ones that are correct today — "this file is fine" is
# exactly why README:75 was fine while README:48, thirty lines above it, was not.
COVERED = ("README.md", "TECHNICAL.md", "USAGE.md", "BENCHMARKS.md", "docs/POSITIONING.md")

# A hard-wrapped assertion is still an assertion: README:48-49 split "Default-on since
# its\nvalidation program completed" across a line break, so a line-oriented regex would
# miss the very site this test exists for.
#
# Two kinds of newline are KEPT, because both are real boundaries between things a reader
# understands as separate claims, and folding either invents an adjacency that isn't on
# the page:
#   * a blank line — the paragraph break;
#   * the start of a list item — TECHNICAL.md's File Descriptions are ~14 consecutive
#     bullets with no blank lines between them, so without this the `stats.py` bullet
#     ("ON by default in `cli.py`", about the ledger) shares a block with the `proxy.py`
#     bullet, borrows its `--diff` vocabulary, and is reported as this defect.
_JOIN = re.compile(r"\n(?![ \t]*\n)(?![ \t]*(?:[-*+]|\d+\.)[ \t])[ \t]*")

# The assertion shapes. Each needs a default-ON predicate; co-occurrence is not enough.
_CLAIMS = (
    # "Default-on since its validation program completed" and "flipped default-on when
    # that program completed" — the #76 remnant in both its README and TECHNICAL forms.
    re.compile(r"default[- ]on\b", re.I),
    # The phrasing a rewrite naturally reaches for. The optional verb also catches
    # "shipped on by default", which is how the claim survived in TECHNICAL's
    # Architecture section, two hundred lines from the tier's actual documentation.
    re.compile(r"\b(?:turned |switched |shipped |flipped )?on by default\b", re.I),
    re.compile(r"\benabled by default\b", re.I),
    # An opt-OUT instruction asserts the thing is on even with no "default" in the
    # sentence, and is the form that would survive a rewrite of the two above.
    re.compile(r"opt[- ]out\b[^.]{0,80}--no-diff", re.I),
)

# The claim must be ABOUT the diff tier; prose describing some other tier is not this
# defect. Scoped to the containing PARAGRAPH rather than a character window, because a
# window cannot be sized right: README:48's claim is ~1,200 characters below the only
# mention of "diff" in its paragraph (the paragraph opens "**2. The stateful cross-call
# diff — the defensible axis**" and never repeats the word before the claim lands), so any
# window loose enough to catch it reaches into neighbouring paragraphs about other tiers.
# `_JOIN` folds hard wraps to spaces and leaves exactly one newline per blank line, so a
# newline in the flattened text IS a paragraph boundary.
#
# The word "diff" alone is too weak even at paragraph scope: `stats.py`'s File-Descriptions
# bullet in TECHNICAL.md contains "diff/textdiff marker" (a decision LABEL) and ends "ON by
# default in `cli.py`" — about the ledger, not the tier — and a bare `diff` screen reports
# it as this defect. So the topic is the diff TIER's own vocabulary: its flags, its policy
# key, its name.
_TOPIC = re.compile(
    r"cross-call|diff tier|diff/lossy|diffing|--diff|--no-diff|[\"`]diff[\"`]", re.I)

# Settings that really ARE on by default and are correctly described that way inside the
# same diff-heavy sections, so `_TOPIC` cannot separate them. A match is exempt when one
# of these is the nearer subject. Each is keyed to the live `Policy` field it claims,
# and `test_the_exemptions_describe_settings_that_are_really_on` asserts the field is
# actually True — so the day one flips to opt-in, the exemption fails loudly instead of
# quietly hiding a second instance of this exact defect.
_TRUE_DEFAULT_ON = (
    (re.compile(r"cross-block join|join_blocks|--no-join-blocks", re.I), "join_blocks"),
)
# Wide enough to reach README:398's subject, which is separated from its predicate by a
# 100-character parenthetical ("Cross-block joining (N content blocks folded into one
# record array before compressing) is built and on by default"). The cost of the width is
# that a genuine diff claim landing within 160 characters after a join mention would be
# exempted; the self-check below pins the known stale wordings against exactly that.
_SUBJECT_WINDOW = 160


def _flat(text: str) -> str:
    return _JOIN.sub(" ", text)


def _paragraph(flat: str, pos: int) -> str:
    """The flattened block containing `pos` — everything between the surrounding newlines,
    which after `_JOIN` are only ever paragraph breaks or list-item starts."""
    return flat[flat.rfind("\n", 0, pos) + 1:(flat.find("\n", pos) + 1 or len(flat) + 1) - 1]


def _scan(flat: str) -> list[tuple[int, str]]:
    """(offset, matched text) for every live "the diff tier is on by default" assertion in
    one already-flattened document. Shared by the file sweep and the self-check below, so
    the self-check exercises the real screening rather than a re-implementation of it."""
    hits: list[tuple[int, str]] = []
    for pattern in _CLAIMS:
        for m in pattern.finditer(flat):
            if not _TOPIC.search(_paragraph(flat, m.start())):
                continue
            subject = flat[max(0, m.start() - _SUBJECT_WINDOW):m.start()]
            if any(p.search(subject) for p, _ in _TRUE_DEFAULT_ON):
                continue
            hits.append((m.start(), flat[max(0, m.start() - 80):m.end() + 80].strip()))
    return hits


def _claims_in(doc: str) -> list[tuple[int, str]]:
    """The same sweep against a file, with offsets translated back to line numbers so a
    failure names the line a human would open."""
    text = (REPO / doc).read_text(encoding="utf-8")
    flat = _flat(text)
    # Offsets survive flattening monotonically (every collapse replaces >=1 chars with 1),
    # so counting newlines in the original up to a flat offset undercounts at worst by the
    # wraps folded away before it. Rebuild the map instead of approximating.
    omap: list[int] = []
    i = 0
    for m in _JOIN.finditer(text):
        omap.extend(range(i, m.start()))
        omap.append(m.start())
        i = m.end()
    omap.extend(range(i, len(text)))
    return [(text[:omap[pos]].count("\n") + 1, frag) for pos, frag in _scan(flat)]


def test_no_shipped_doc_says_the_diff_tier_is_on_by_default():
    """The point of the file. `Policy().diff` is the ground truth and the docs are checked
    against IT, not against each other — so agreeing-but-wrong docs still fail, which
    matters because README:48 and README:79 disagreed with each other for two weeks and
    neither one was reading `policy.py`."""
    if default_policy().diff:
        # The default flipped back deliberately; this test's one-directional claim is
        # vacuous in that world by design — see the module docstring.
        return
    offenders = {doc: hits for doc in COVERED if (hits := _claims_in(doc))}
    assert not offenders, (
        "A default policy's `diff` is False (opt-in since #170), but these documents "
        "still tell a "
        "reader the cross-call diff tier is on by default:\n  "
        + "\n  ".join(f"{doc}:{line}: {frag}"
                      for doc, hits in sorted(offenders.items()) for line, frag in hits)
        + "\nEither the prose is stale — fix it; the tier is opt-in via `proxy --diff` / "
          "`install-mcp --diff` / a policy-file `\"diff\": true` — or the default really "
          "did flip, in which case `src/terse/policy.py` is what should have changed.")


def test_the_sweep_recognises_the_claim_it_screens_for():
    """A doc test whose regexes match nothing passes forever while pinning nothing — the
    exact failure mode that let the contradiction live for two weeks. It cannot be checked
    by counting matches in the live files, because once the docs are correct that count is
    ZERO and a broken sweep is indistinguishable from a working one. So the sweep is fed
    the verbatim sentences that were in `README.md` and `TECHNICAL.md` before this PR fixed
    them, and required to catch every one.

    The correct sentences from the same documents are fed in too, and required to produce
    silence: a sweep that fires on "OFF by default" has to be neutered to get the suite
    green, and a neutered sweep is how this rots the second time."""
    stale = (
        # README.md:41-49, verbatim, hard wraps included. Quoted as a whole PARAGRAPH
        # rather than as the offending sentence, because the paragraph is the unit the
        # topic screen works on — this is also the site that proves the screen has to be
        # paragraph-scoped: the claim is ~1,200 characters below the only "diffing" in it.
        "actually repeats a call with a similar-enough payload. That measured ~0.4% of "
        "results in\nterse's own 7-day traffic — but most of that was **structural, not "
        "workload**: results\narriving as N content blocks were excluded from diffing "
        "outright, which was 71% of tokens.\nThe cross-block join (below) removed that "
        "exclusion, and across every third-party server\nbenchmarked in BENCHMARKS §6 a "
        "repeated call now produces a delta. How often *your* loop\nrepeats a call is "
        "still yours to measure (`terse stats`).\nWhen your loop *does* re-fetch "
        "mostly-unchanged results it compounds hard; when it\ndoesn't, it costs nothing "
        "(lossless, and emitted only when smaller). Default-on since its\nvalidation "
        "program completed (see Status).",
        # USAGE.md:970-972, verbatim — a code-block comment, no "default" in it at all,
        # and the site the first run of this sweep found that no hand pass had.
        "# nothing to enable — a plain proxy diffs. Opt OUT per proxy:\n"
        "uv run terse proxy --no-diff -- uvx some-mcp-server\n"
        "# or per policy file: {\"diff\": false, ...}",
        # README.md:79-81, verbatim.
        "  Default-on since its validation program completed (fluency, nested-record "
        "coverage,\n  and the drift soak — see Status); opt out with `proxy --no-diff` / "
        "`install-mcp\n  --no-diff` or a policy-file `\"diff\": false`.",
        # TECHNICAL.md:455-464, verbatim — the whole bullet, since the bullet is the
        # block. Note it opens by correctly calling the tier OPT-IN and closes by calling
        # it default-on, which is why "the docs agree with each other" is not the check.
        "- **Cross-call diffing is built and OPT-IN (`proxy --diff` / policy\n  "
        "`\"diff\": true`).** Default-off since #170: the tier is correct, but its primer\n"
        "  paragraph is 34% of the primer (190 of 555) and the live ledger shows a 0.38% "
        "hit rate.\n The probe shows 91% overlap between successive\n  same-tool calls; "
        "the proxy emits a lossless delta against the prior result (keyed row\n  diff for "
        "record arrays, shallow key diff for objects) instead of the full payload. It\n  "
        "is stateful (per-tool last result), self-verifying (a diff is sent only when it\n"
        "  provably reconstructs the result), and fail-open (full form whenever a diff "
        "doesn't\n  apply or isn't smaller — the dangling-reference fallback). It shipped "
        "opt-in until its\n  two model-side risks were measured, and flipped default-on "
        "when that program completed:",
        # The opt-out-instruction form, carrying the claim with no "default" at all.
        "The cross-call diff tier runs unless you opt out with `proxy --no-diff`.",
        # A plausible future rewrite that uses none of the historical wording.
        "Cross-call diffing is enabled by default; pass `--no-diff` to turn it off.",
    )
    correct = (
        # README.md:75, right all along.
        "- **Tier 0.7 — cross-call diff (stateful, OPT-IN — `\"diff\": true`)**: when the "
        "same tool is called\n  repeatedly, the proxy emits a lossless delta",
        # USAGE.md:953.
        "It is stateful and **OFF by default** — opt in with `--diff` (or `\"diff\": "
        "true` in a policy).",
        # BENCHMARKS.md:311 — "no longer the zero-config default" is the phrase a
        # negation-blind sweep misreads as a default-on claim.
        "**† Since #170 (2026-07-28) cross-call diffing is no longer the zero-config "
        "default** —",
        # TECHNICAL.md:456.
        "- **Cross-call diffing is built and OPT-IN (`proxy --diff` / policy\n  "
        "`\"diff\": true`).** Default-off since #170: the tier is correct, but its primer",
        # README.md:82 — a TRUE default-on claim about a DIFFERENT setting, inside a
        # sentence that goes on to mention the diff tier. `_TRUE_DEFAULT_ON` is what
        # keeps it out; `_TOPIC` alone cannot.
        "- **Cross-block join (ON by default)**: some MCP servers return one record per "
        "content\n  block, so each block is a lone object the codec above can barely fold "
        "and the diff tier skips entirely",
    )
    for sample in stale:
        assert _scan(_flat(sample)), f"the sweep no longer catches a stale claim: {sample!r}"
    for sample in correct:
        assert not _scan(_flat(sample)), (
            f"the sweep fires on prose that is CORRECT: {sample!r} — narrowing the docs to "
            f"satisfy it would be fixing the test by breaking the documentation")


def test_the_exemptions_describe_settings_that_are_really_on():
    """`_TRUE_DEFAULT_ON` is a hole in the sweep, so it is grounded rather than trusted: a
    subject is exempt only while its `Policy` field really is on by default. If
    `join_blocks` is ever flipped to opt-in, its prose becomes the same defect this file
    exists for, and the exemption must stop covering it — this fails first and says so."""
    for pattern, field in _TRUE_DEFAULT_ON:
        live = getattr(default_policy(), field)
        assert live is True, (
            f"`_TRUE_DEFAULT_ON` exempts prose about `{field}` from the default-on sweep "
            f"on the grounds that it IS on by default, but a default policy's `{field}` "
            f"is {live!r}. Remove the exemption and check that setting's prose — pattern "
            f"{pattern.pattern!r}.")
