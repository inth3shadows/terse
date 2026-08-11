"""Every published release must have a CHANGELOG section, and `[Unreleased]` must be honest.

The file states its own process in its header: *"an entry moves from `[Unreleased]` to a
versioned section when its tag is pushed."* That had not happened for **37 tags**. The last
versioned section was `[0.4.1] - 2026-07-21` while `terse-mcp 0.22.2` was live on PyPI, so
1,133 lines described as "Unreleased" were in fact shipped — some of them weeks earlier.

The concrete harm is the one question a changelog exists to answer: a user on 0.22.2 asking
*"is the primer-cadence fix in my version?"* had no way to tell, because everything read as
unreleased.

These tests pin the process rather than the prose. They cannot check that an entry is
well-written; they can check that no release is silently undocumented and that
`[Unreleased]` does not accumulate shipped work again.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
# Repo-relative on purpose: it is pasted into a failure message for a human to run from the
# repo root, so an absolute path from whatever checkout CI used would be noise.
GRADUATE_SCRIPT = "scripts/release/graduate_changelog.py"
_SECTION = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.M)


def _ver(tag: str) -> tuple[int, ...]:
    """Sort key for a `vX.Y.Z` tag. NEVER order tags as strings: this repo is past v0.10,
    so lexicographic puts `v0.24.1` ahead of `v0.3.1` and `v0.9.0`. Named once here because
    getting it wrong is silent — the message still renders, it just names the wrong
    release, and every section is already backfilled so the script exits 0 with "already
    has a section" and the human is left following advice that does nothing."""
    return tuple(int(x) for x in tag.lstrip("v").split("."))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=False).stdout


def _tags() -> list[str]:
    return [t for t in _git("tag", "--sort=creatordate").split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]


@pytest.fixture(scope="module")
def text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def test_every_release_but_the_newest_has_a_changelog_section(text):
    """The rule the file's own header states. A tag is a published PyPI release (hatch-vcs
    cuts from tags), so a tag without a section is a shipped change nobody can look up.

    THE NEWEST TAG IS EXEMPT, and that is a consequence of how this repo releases, not a
    softening. `release.yml` fires on every push to `main`: it decides the version, runs the
    test gate, and only THEN pushes the tag. So the release a merge produces cannot exist
    while that merge's own CI runs — demanding a section for it would turn the NEXT PR red
    over a tag that did not exist when its code was written.

    Verified rather than assumed: v0.22.0, v0.22.1 and v0.22.2 are each tagged at one of the
    last three merges to `main`.

    One release of grace, then. It still catches the drift this file exists for — against
    the pre-backfill CHANGELOG it flags 25 releases — and the exemption expires as soon as
    another release lands, because the undocumented one is no longer newest."""
    tags = _tags()
    if not tags:
        pytest.skip("no release tags in this checkout (shallow clone or fresh fork)")
    documented = set(_SECTION.findall(text))
    newest = max(tags, key=lambda t: tuple(int(x) for x in t.lstrip("v").split(".")))
    missing = [t for t in tags if t != newest and t.lstrip("v") not in documented]
    assert not missing, (
        f"{len(missing)} release(s) have no CHANGELOG section: {missing}\n"
        f"(the newest, {newest}, is exempt — see this test's docstring.)\n"
        "Add one — the header promises an entry moves out of [Unreleased] when its tag is "
        "pushed, and a user on that version has no other way to find out what changed.")


def test_unreleased_does_not_describe_work_that_already_shipped():
    """The failure this file was written for, encoded as the invariant rather than a size
    heuristic.

    The first cut of this test compared `[Unreleased]`'s length against the biggest released
    section. That is too loose to be worth having: 200 lines of shipped work sat happily
    under a 280-line v0.5.0 section, and the mutation proving it caught nothing is why this
    was rewritten.

    The real rule is the file's own: an entry moves out of `[Unreleased]` when its tag is
    pushed. So blame each line still in `[Unreleased]` and ask git whether that commit is
    already contained in a tag. If it is, the work shipped and the entry is in the wrong
    place — which is exactly, and only, the drift that produced 1,133 stale lines. A
    genuinely pending entry blames to an untagged commit and is fine, so this never
    obstructs normal work.

    KNOWN LIMIT, measured rather than assumed: blame reports the last commit to touch a
    line, so physically MOVING a released entry back into `[Unreleased]` re-blames it to the
    (untagged) move commit and slips past. Verified — that mutation passes. This catches the
    drift that actually happened (entries written under `[Unreleased]` and never moved out,
    still blaming their original tagged commit) and not a deliberate relocation, which is
    not a failure mode anyone has. `test_every_release_tag_has_a_changelog_section` is the
    backstop there: the relocated entry's release still needs a section."""
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("## [Unreleased]"))
        end = next(i for i, ln in enumerate(lines) if i > start and ln.startswith("## ["))
    except StopIteration:
        pytest.skip("no [Unreleased] section")
    if end - start <= 1 or not _tags():
        return
    # Blame the WORKING TREE, not HEAD: the line numbers above come from the file on disk,
    # and blaming HEAD pairs them with a different revision's line numbering the moment
    # CHANGELOG.md has an uncommitted edit — which is every time someone is writing an
    # entry. Uncommitted lines come back as an all-zero sha and are skipped below, which is
    # correct: a line not yet committed cannot be in a tag.
    blame = _git("blame", "-L", f"{start + 2},{end}", "--line-porcelain",
                 "--", "CHANGELOG.md")
    shipped: dict[str, str] = {}
    for m in re.finditer(r"^([0-9a-f]{40}) \d+ (\d+)", blame, re.M):
        sha, lineno = m.group(1), int(m.group(2))
        if sha.startswith("0" * 20):        # uncommitted working-tree edit
            continue
        content = lines[lineno - 1]
        # Structure, not content: a blank separator or a `### Added` heading has sat in this
        # section since the file was created, so it blames to an ancient commit and would
        # flag on every run. Only an actual entry line can be "work that already shipped".
        if not content.strip() or content.startswith(("###", "_")):
            continue
        containing = _git("tag", "--contains", sha, "--sort=creatordate").split()
        if containing:
            shipped.setdefault(containing[0], lines[lineno - 1].strip()[:70])
    # The fix is one command, so print the command rather than a description of it. Whoever
    # trips this is usually not the person who cut the release — they opened the next PR and
    # inherited a red test about work that is not theirs — so making them go find the script
    # is the avoidable part of the friction. Graduation is deliberately manual (see
    # `.github/workflows/release.yml`'s header: every automated path was tried or ruled out),
    # which makes the message the only place this hint can live.
    # Guarded, not bare indexing: on the PASSING path `shipped` is empty, and an unguarded
    # index turns every green run into an IndexError. An assert's message is only evaluated
    # when it fires, but this line is not the message.
    oldest = min(shipped, key=_ver) if shipped else None
    # The single-release case is the only one the script can finish by itself, so it is the
    # only one that gets a command. With two or more pending, `graduate_changelog.py` would
    # move the ENTIRE [Unreleased] body — the newer release's entries, and any genuinely
    # unreleased ones — under the oldest tag, then exit 0 on a second run with "holds no
    # entries". This test would go green on the empty section and never flag the misfiling.
    # So say what actually has to happen instead of implying a loop that silently corrupts.
    if len(shipped) > 1:
        fix = ("\nSplit [Unreleased] by release BY HAND first — the script graduates the "
               f"whole body under ONE version per run, so pointing it at {oldest} would "
               "file the newer releases' entries there too.")
    else:
        fix = f"\nRun:\n    python3 {GRADUATE_SCRIPT} {oldest} CHANGELOG.md"
    assert not shipped, (
        "[Unreleased] describes work that is already released:\n  "
        + "\n  ".join(f"{tag}: {shipped[tag]}" for tag in sorted(shipped, key=_ver))
        + "\nMove these into their versioned sections — the header promises an entry "
          "leaves [Unreleased] when its tag is pushed." + fix)


def test_sections_are_ordered_newest_first_and_unique(text):
    """A duplicated or out-of-order version is how a reader ends up reading the wrong
    release's notes — and a duplicate silently hides one of the two."""
    found = _SECTION.findall(text)
    assert len(found) == len(set(found)), (
        f"duplicate version sections: "
        f"{sorted({v for v in found if found.count(v) > 1})}")
    keyed = [tuple(int(p) for p in v.split(".")) for v in found]
    assert keyed == sorted(keyed, reverse=True), (
        "version sections are not in descending order — "
        f"first break at {found[next(i for i in range(1, len(keyed)) if keyed[i] > keyed[i-1])]}")


def test_every_section_carries_the_release_date_git_records(text):
    """A hand-typed date drifts from the tag. Checked against `git log` on the tag itself so
    the two cannot disagree — the same read-both-and-compare rule the primer-size and
    benchmark tests apply to published numbers."""
    tags = {t.lstrip("v"): t for t in _tags()}
    if not tags:
        pytest.skip("no release tags in this checkout")
    wrong = []
    for ver, date in re.findall(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})", text, re.M):
        if ver not in tags:
            continue
        actual = _git("log", "-1", "--format=%cs", tags[ver]).strip()
        if actual and actual != date:
            wrong.append(f"{ver}: section says {date}, tag says {actual}")
    assert not wrong, "release dates disagree with their tags:\n  " + "\n  ".join(wrong)


def test_the_graduation_hint_points_at_a_script_that_exists():
    # The failure message above tells a human to run GRADUATE_SCRIPT. If that script is ever
    # moved or renamed, the hint keeps printing confidently and sends them to a path that is
    # not there — worse than the bare message it replaced, and invisible until someone
    # actually trips the (rare, release-gated) assertion. Pin it here instead, where it runs
    # every time.
    assert (REPO / GRADUATE_SCRIPT).is_file(), (
        f"{GRADUATE_SCRIPT} does not exist, but the graduation failure message tells "
        "people to run it")


def test_the_hint_names_the_OLDEST_pending_release_not_the_lexicographic_first():
    # The bug this pins shipped once and CI stayed green, because the guard above checked
    # that the script exists and nothing checked WHICH TAG it is handed. `sorted()` on tag
    # strings puts v0.24.1 ahead of v0.3.1 once a repo passes v0.10 — and since every
    # release here is already backfilled, the script would exit 0 with "already has a
    # section", so the hint reads authoritative and does nothing.
    assert min({"v0.24.1", "v0.3.1", "v0.9.0"}, key=_ver) == "v0.3.1"
    assert sorted(["v0.24.1", "v0.3.1", "v0.9.0"], key=_ver) == \
        ["v0.3.1", "v0.9.0", "v0.24.1"]
    # and the real tag namespace must parse — a tag shape _ver cannot read would raise
    # inside the failure path, replacing the message with a ValueError
    for tag in _tags():
        assert len(_ver(tag)) == 3, tag
