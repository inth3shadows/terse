"""`scripts/release/graduate_changelog.py` — the MANUAL step that moves `[Unreleased]`.

The backfill (#231) recovered 37 releases of stale `[Unreleased]` and added tests that
DETECT the drift. Graduating it was briefly automated inside `release.yml`; that bot push to
protected `main` failed 44/44 times and is now deleted (see the workflow header). A
maintainer runs this script in the first pull request after a release, and DETECTION is the
enforcement mechanism: `test_unreleased_does_not_describe_work_that_already_shipped` is
CI-run-gated, so the next PR after a release is red until the section moves.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release" / "graduate_changelog.py"
HEAD = "# Changelog\n\n## [Unreleased]\n\n"
REST = "## [0.9.0] - 2026-01-01\n\n### Added\n\n- old thing\n"


def _run(tmp: Path, body: str, tag: str = "v1.0.0"):
    f = tmp / "CHANGELOG.md"
    f.write_text(HEAD + body + REST, encoding="utf-8")
    p = subprocess.run([sys.executable, str(SCRIPT), tag, str(f)],
                       capture_output=True, text=True, check=False)
    return f.read_text(encoding="utf-8"), p


def test_a_pending_entry_moves_into_its_version_section(tmp_path):
    out, p = _run(tmp_path, "### Fixed\n\n- **A real entry.** Detail.\n\n")
    assert p.returncode == 0
    assert "## [1.0.0] - " in out
    assert "- **A real entry.** Detail." in out
    # ...and [Unreleased] is left empty for the next cycle, not deleted.
    unreleased = out.split("## [Unreleased]")[1].split("## [")[0]
    assert "A real entry" not in unreleased


def test_the_italic_placeholder_is_not_published_as_a_release_note(tmp_path):
    """Caught in a local dry run before this shipped: the `_Nothing yet._` line is section
    furniture, and carrying it into the release published "Nothing yet" under a version that
    plainly changed something."""
    out, _ = _run(tmp_path, "_Nothing yet._\n\n### Fixed\n\n- **Real.** Detail.\n\n")
    released = out.split("## [1.0.0]")[1]
    assert "Nothing yet" not in released.split("## [0.9.0]")[0]


def test_an_empty_unreleased_is_left_alone_rather_than_emitting_a_hollow_section(tmp_path):
    """An empty version section is worse than none: it asserts "this release changed
    nothing" about a release that exists because something changed. Leave it for a human and
    let the changelog test flag it on the next run."""
    out, p = _run(tmp_path, "_Nothing yet._\n\n")
    assert p.returncode == 0 and "## [1.0.0]" not in out
    assert "no entries" in p.stdout


def test_rerunning_is_a_no_op(tmp_path):
    """A maintainer may run it twice. A second run must not graduate an empty section on top
    of the one it already wrote."""
    f = tmp_path / "CHANGELOG.md"
    f.write_text(HEAD + "### Fixed\n\n- **Real.** Detail.\n\n" + REST, encoding="utf-8")
    for _ in range(2):
        subprocess.run([sys.executable, str(SCRIPT), "v1.0.0", str(f)], check=True,
                       capture_output=True)
    assert f.read_text(encoding="utf-8").count("## [1.0.0]") == 1


def test_release_yml_never_pushes_a_commit_to_a_branch():
    """`release.yml` may push a TAG and nothing else. Read the evidence before re-adding any
    commit-and-push automation here:

    * The deleted "Graduate the changelog" step tried to push a `chore(release):` commit to
      `main` after every release and failed 44 out of 44 times — silently, because the push
      was suffixed `|| echo "::warning::..."`. Zero such commits have ever landed
      (`git log --grep 'chore(release): graduate'` on `main` is empty).
    * It cannot be fixed with the built-in token. `inth3shadows/terse` is a USER-owned repo
      (`gh api repos/inth3shadows/terse --jq '.owner.type'` -> `User`) and Ruleset bypass
      actors are org-only, so no rule can exempt the workflow from branch protection.
    * Opening a PR instead does not work either: GITHUB_TOKEN-created PRs trigger no
      workflow runs, so with `required_status_checks.strict=true` the bot's PR has zero
      check runs on its head SHA and is permanently unmergeable, not merely slow.
    * That leaves a stored PAT / GitHub App with bypass rights — a standing credential and
      trust-boundary weakening on a solo-maintainer repo, to save a two-minute manual edit.

    Graduation is a manual `scripts/release/graduate_changelog.py <tag>` in the first PR
    after a release, enforced by
    `test_unreleased_does_not_describe_work_that_already_shipped`.
    """
    wf = (Path(__file__).resolve().parent.parent / ".github/workflows/release.yml").read_text()
    # Comment lines are excluded: the header EXPLAINS the deleted step (quoting its swallowed
    # `::warning::`), and a prose explanation is the point, not a violation.
    code = "\n".join(ln for ln in wf.splitlines() if not ln.lstrip().startswith("#"))
    assert "git commit" not in code, "release.yml must not create commits — see this docstring"
    assert not re.search(r"git push\s+\S+\s+[\"']?HEAD:", code), "must not push a branch"
    assert "::warning::" not in code, "a swallowed push failure is how the 44/44 silence happened"
    # ...and the one push that IS legitimate — the version tag — still survives.
    assert 'git push origin "${NEW_TAG}"' in code


def test_the_section_date_comes_from_the_tag_not_the_day_you_ran_it(tmp_path):
    """Graduation is manual now, so it runs days after the tag — and
    `test_every_section_carries_the_release_date_git_records` compares each section against
    `git log -1 --format=%cs <tag>`. Stamping `date.today()` therefore produces a red suite
    for anyone who graduates late (it did, the day this became a manual step)."""
    import datetime
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ,
           "GIT_AUTHOR_DATE": "2020-03-04T12:00:00+00:00",
           "GIT_COMMITTER_DATE": "2020-03-04T12:00:00+00:00",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    git("init", "-q")
    f = repo / "CHANGELOG.md"
    f.write_text(HEAD + "### Fixed\n\n- **Real.** Detail.\n\n" + REST, encoding="utf-8")
    git("add", "CHANGELOG.md")
    git("commit", "-qm", "seed")
    git("tag", "v1.0.0")

    p = subprocess.run([sys.executable, str(SCRIPT), "v1.0.0", str(f)],
                       capture_output=True, text=True, check=True)
    out = f.read_text(encoding="utf-8")
    assert "## [1.0.0] - 2020-03-04" in out, p.stdout + p.stderr
    assert datetime.date.today().isoformat() not in out.split("## [0.9.0]")[0]


def test_the_release_gate_excludes_exactly_the_bookkeeping_tests_and_PR_CI_runs_them():
    """The release job must not fail on PAST releases' CHANGELOG filing (it skipped #422's
    release), so it excludes the `changelog_bookkeeping` marker — and nothing else. Pinned
    three ways: the exact gate command (any extra -k/--ignore/--deselect fails it), the
    marked set (a rename or a new marked test fails it), and tests.yml running the suite
    unfiltered so the checks still run somewhere. Neither workflow may set PYTEST_ADDOPTS,
    which would filter the suite without touching the run line. (#436 tracks making the
    marked checks deterministic so they can rejoin the gate.)"""
    root = Path(__file__).resolve().parent.parent
    release = (root / ".github/workflows/release.yml").read_text()
    tests_yml = (root / ".github/workflows/tests.yml").read_text()
    assert "PYTEST_ADDOPTS" not in release + tests_yml
    gate = [ln.strip() for ln in release.splitlines() if "pytest" in ln and "uv run" in ln]
    assert gate == ['uv run pytest -q -m "not changelog_bookkeeping"'], gate
    pr = [ln.strip() for ln in tests_yml.splitlines() if "pytest" in ln and "run:" in ln]
    filters = (re.compile(r"pytest\b.*\s-m\s"), re.compile(r"\s-k\s"),
               re.compile(r"--deselect"), re.compile(r"--ignore"))
    assert pr and not any(f.search(ln) for ln in pr for f in filters), pr
    # The marked set, by a text scan of every test file in the repo — the gate runs bare
    # `pytest` from the root, so a marked test outside tests/ would be excluded too. A scan,
    # not a `pytest --co` subprocess: that re-collected the whole suite (~2.5s) per run and
    # broke under an inherited PYTEST_ADDOPTS.
    marked = []
    for f in sorted(root.rglob("test_*.py")):
        if ".venv" in f.parts or "node_modules" in f.parts:
            continue
        marked += [(f.relative_to(root).as_posix(), m) for m in re.findall(
            r"@pytest\.mark\.changelog_bookkeeping\s*\n\s*def (\w+)\(", f.read_text())]
        assert not re.search(r"^pytestmark\s*=.*changelog_bookkeeping", f.read_text(),
                             re.M), f"module-level mark in {f}"
    assert sorted(marked) == [
        ("tests/test_changelog_covers_every_release.py",
         "test_every_release_but_the_newest_has_a_changelog_section"),
        ("tests/test_changelog_covers_every_release.py",
         "test_unreleased_does_not_describe_work_that_already_shipped"),
    ], marked
