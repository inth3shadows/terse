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

import os
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


def _git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=False).stdout


def _tags(repo: Path = REPO) -> list[str]:
    return [t for t in _git("tag", "--sort=creatordate", repo=repo).split()
            if re.fullmatch(r"v\d+\.\d+\.\d+", t)]


@pytest.fixture(scope="module")
def text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


@pytest.mark.changelog_bookkeeping
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


def _unreleased_body(text: str) -> list[str] | None:
    """The lines strictly between `## [Unreleased]` and the next `## [` heading (or the end
    of the file), or None when there is no `[Unreleased]` section."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## [Unreleased]")), None)
    if start is None:
        return None
    end = next((i for i, ln in enumerate(lines) if i > start and ln.startswith("## [")),
               len(lines))
    return lines[start + 1:end]


def _entry_lines(lines: list[str]) -> set[str]:
    """Only actual entry text can be "work that already shipped". Structure recurs across
    releases and would match on every run: a blank separator, a `### Added` heading, the
    `_Nothing yet._` placeholder, a multi-line italic note (either end), a bare code fence.
    Measured on this CHANGELOG: once those are excluded, no entry line repeats."""
    out = set()
    for ln in lines:
        t = ln.strip()
        if (not t or t.startswith(("#", "_", "```")) or t.endswith("_")
                or not any(c.isalpha() for c in t)):
            continue
        out.add(t)
    return out


def _shipped_unreleased_entries(text: str, repo: Path = REPO) -> dict[str, str]:
    """Map each release tag to one `[Unreleased]` entry line (of `text`) it already shipped.

    Decided by CONTENT, not history (#436): an entry has shipped iff its line appears in
    `CHANGELOG.md` as of some release tag. At a tag, that snapshot's `[Unreleased]` holds
    exactly what the tag shipped (graduation happens in the NEXT pull request) and its
    versioned sections hold everything older, so any entry line in any tag's snapshot is
    released work. Each line is attributed to the OLDEST tag containing it — the release
    that actually shipped it, which is the one the graduation hint must name.

    The previous check blamed each line and asked `git tag --contains`. Blame depends on
    history, not content, so the same file could pass on a PR branch (a moved line blames
    to the untagged branch commit) and fail on its squash (the line blames back to its
    tagged original) — which is how #422's release was skipped. Content gives the same
    answer on both, and a relocated entry is still caught.

    No tags (a shallow clone or fresh fork) means nothing can be proven shipped: returns
    {}. A tag whose tree has no CHANGELOG.md contributes nothing."""
    body = _unreleased_body(text)
    if not body:
        return {}
    pending = _entry_lines(body)
    shipped: dict[str, str] = {}
    for tag in sorted(_tags(repo), key=_ver):
        if not pending:
            break
        snap = subprocess.run(["git", "show", f"{tag}:CHANGELOG.md"], cwd=repo,
                              capture_output=True, text=True, check=False).stdout
        hit = pending & {ln.strip() for ln in snap.splitlines()}
        if hit:
            shipped[tag] = sorted(hit)[0][:70]
            pending -= hit
    return shipped


@pytest.mark.changelog_bookkeeping
def test_unreleased_does_not_describe_work_that_already_shipped():
    """The failure this file was written for, encoded as the invariant rather than a size
    heuristic.

    The first cut of this test compared `[Unreleased]`'s length against the biggest released
    section. That is too loose to be worth having: 200 lines of shipped work sat happily
    under a 280-line v0.5.0 section, and the mutation proving it caught nothing is why this
    was rewritten.

    The real rule is the file's own: an entry moves out of `[Unreleased]` when its tag is
    pushed. So ask, for each entry still in `[Unreleased]`, whether that exact line is
    already in `CHANGELOG.md` at some release tag (`_shipped_unreleased_entries`). If it
    is, the work shipped and the entry is in the wrong place. A genuinely pending entry is
    new text that no tag has seen, so this never obstructs normal work.

    Reads the WORKING TREE file, so an uncommitted entry is checked too — correct, since
    new text cannot be in a tag, and a released line pasted back in is caught before it is
    committed. Content, not blame: a PR branch and its squash give the same answer (#436)."""
    text = CHANGELOG.read_text(encoding="utf-8")
    if _unreleased_body(text) is None:
        pytest.skip("no [Unreleased] section")
    shipped = _shipped_unreleased_entries(text)
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


def test_there_is_exactly_one_unreleased_section(text):
    """`test_sections_are_ordered_newest_first_and_unique` checks VERSIONED sections, and
    `_SECTION` does not match `[Unreleased]` — so the one heading every release passes
    through was the one heading nothing checked for duplicates.

    It happened: the graduation of `v0.28.7` re-inserted an empty `[Unreleased]` at the top,
    and the same pull request then added its own `[Unreleased]` below it. Both landed on
    `main`. Nothing went red, because `test_unreleased_does_not_describe_work_that_already_
    shipped` reads the FIRST section — which was the empty one — and so was satisfied while
    a second section sat under it holding unshipped notes.

    The consequence is not cosmetic. `scripts/release/graduate_changelog.py` also reads the
    first `[Unreleased]`, found it empty, and refused to graduate: the mechanism that exists
    to be a red reminder was silently a no-op, which is the one failure mode a
    reminder-shaped test cannot afford."""
    assert text.count("## [Unreleased]") == 1, (
        f"{text.count('## [Unreleased]')} `## [Unreleased]` headings — graduation reads the "
        "first one, so a second silently holds notes that never graduate")


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


# --- #436: the shipped-entry check must depend on content, not on history ---------------

_HEAD = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n"
_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1", "PATH": os.environ.get("PATH", "")}


class _Repo:
    """A throwaway git repo with a CHANGELOG.md, for reproducing history shapes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir()
        self("init", "-q", "-b", "main")

    def __call__(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.path, env=_GIT_ENV,
                              capture_output=True, text=True, check=True).stdout

    def commit(self, body: str, msg: str) -> None:
        (self.path / "CHANGELOG.md").write_text(body, encoding="utf-8")
        self("add", "CHANGELOG.md")
        self("commit", "-q", "-m", msg)

    def text(self) -> str:
        return (self.path / "CHANGELOG.md").read_text(encoding="utf-8")

    def blame_says_shipped(self, line: str) -> bool:
        """The OLD check (blame + `tag --contains`), kept only to prove the fixture below
        reproduces #436."""
        n = self.text().splitlines().index(line) + 1
        sha = self("blame", "-L", f"{n},{n}", "--porcelain", "--", "CHANGELOG.md").split()[0]
        return bool(self("tag", "--contains", sha).split())


def test_branch_head_and_its_squash_get_the_same_answer(tmp_path):
    """#422, reduced: v0.1.0 ships entry A from [Unreleased]. A PR branch moves A (down,
    then back) and adds B, never graduating A. Blame gives A to the branch's untagged
    commit on the branch head (green) but to the tagged original on the squash (red). The
    content check flags A on both, and never flags B."""
    r = _Repo(tmp_path / "repo")
    a, b = "- Entry A shipped in v0.1.0.", "- Entry B is still pending."
    r.commit(_HEAD + f"{a}\n- Filler line one.\n", "a")
    r("tag", "v0.1.0")
    r("switch", "-q", "-c", "pr")
    r.commit(_HEAD + f"- Filler line one.\n{a}\n", "move a down")
    r.commit(_HEAD + f"{a}\n{b}\n- Filler line one.\n", "move a back, add b")
    branch_text, branch_blame = r.text(), r.blame_says_shipped(a)
    branch_verdict = _shipped_unreleased_entries(branch_text, r.path)

    r("switch", "-q", "main")
    r("merge", "-q", "--squash", "pr")
    r("commit", "-q", "-m", "squash")
    assert r.text() == branch_text
    # the fixture really reproduces the bug: blame disagrees between the two
    assert (branch_blame, r.blame_says_shipped(a)) == (False, True)
    squash_verdict = _shipped_unreleased_entries(r.text(), r.path)
    assert branch_verdict == squash_verdict == {"v0.1.0": a}


def test_pending_entries_pass_and_untagged_history_proves_nothing(tmp_path):
    r = _Repo(tmp_path / "repo")
    r.commit(_HEAD + "- Old entry.\n", "a")
    # no tags (shallow clone / fresh fork): nothing can be proven shipped
    assert _shipped_unreleased_entries(_HEAD + "- Old entry.\n", r.path) == {}
    r("tag", "v0.1.0")
    graduated = _HEAD + "- Brand new entry.\n\n## [0.1.0] - 2026-01-01\n\n- Old entry.\n"
    assert _shipped_unreleased_entries(graduated, r.path) == {}
    # structure that recurs in every snapshot is never "shipped work"
    assert _shipped_unreleased_entries(_HEAD + "_Nothing yet._\n\n```\n", r.path) == {}


def test_each_entry_is_attributed_to_the_OLDEST_tag_that_shipped_it(tmp_path):
    r = _Repo(tmp_path / "repo")
    (r.path / "README").write_text("x")
    r("add", "README")
    r("commit", "-q", "-m", "pre-changelog")
    r("tag", "v0.0.9")                      # no CHANGELOG.md in this tree: contributes nothing
    r.commit(_HEAD + "- One.\n", "one")
    r("tag", "v0.1.0")
    r.commit(_HEAD + "- One.\n- Two.\n", "two")
    r("tag", "v0.10.0")                     # numeric, not lexicographic, order
    got = _shipped_unreleased_entries(_HEAD + "- Two.\n- One.\n- Three.\n", r.path)
    assert got == {"v0.1.0": "- One.", "v0.10.0": "- Two."}
