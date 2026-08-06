"""`scripts/release/graduate_changelog.py` — the step that makes the drift unrepresentable.

The backfill (#231) recovered 37 releases of stale `[Unreleased]` and added tests that
DETECT the drift. This graduates the section automatically at tag time, so it cannot recur:
`release.yml` runs it after pushing the tag, and the result lands on `main` as a follow-up
commit. Detection stays as the backstop.
"""
from __future__ import annotations

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
    """`release.yml` can retry a job. A second run must not graduate an empty section on top
    of the one it already wrote."""
    f = tmp_path / "CHANGELOG.md"
    f.write_text(HEAD + "### Fixed\n\n- **Real.** Detail.\n\n" + REST, encoding="utf-8")
    for _ in range(2):
        subprocess.run([sys.executable, str(SCRIPT), "v1.0.0", str(f)], check=True,
                       capture_output=True)
    assert f.read_text(encoding="utf-8").count("## [1.0.0]") == 1


def test_the_workflow_commit_prefix_cannot_trigger_another_release(tmp_path):
    """The loop guard, pinned against the workflow's OWN bump regexes rather than restated.
    Graduation pushes to `main`, which re-enters `release.yml`; only feat/fix/perf/breaking
    subjects bump, so the prefix must not be one of those."""
    import re
    wf = (Path(__file__).resolve().parent.parent / ".github/workflows/release.yml").read_text()
    subject = re.findall(r'git commit -m "([^"]+)"', wf)[0]
    subject = subject.replace("${NEW_TAG}", "v1.0.0")
    assert not re.match(r"^[a-z]+(\([^)]*\))?!:", subject), "a `!` subject bumps — infinite loop"
    assert not re.match(r"^(feat|fix|perf)(\([^)]*\))?:", subject), f"{subject!r} would re-release"
    assert "[skip ci]" in subject
