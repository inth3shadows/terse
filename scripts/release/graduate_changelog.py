#!/usr/bin/env python3
"""Move `## [Unreleased]` into a versioned section.

MANUAL helper — `release.yml` no longer calls it. Automated graduation pushed a bot commit
to protected `main` and failed 44/44 times; see the header of `.github/workflows/release.yml`
for why it cannot be fixed without a stored PAT. Run this by hand in the first pull request
opened after a release:

    python3 scripts/release/graduate_changelog.py <tag> [CHANGELOG.md] [YYYY-MM-DD]

Because it now runs days after the tag, the release date must come from the TAG, not from
today: `tests/test_changelog_covers_every_release.py::
test_every_section_carries_the_release_date_git_records` compares every section against
`git log -1 --format=%cs <tag>`.
"""
import datetime
import pathlib
import re
import subprocess
import sys

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

tag, path = sys.argv[1], pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "CHANGELOG.md")
ver = tag.lstrip("v")
text = path.read_text(encoding="utf-8")

if re.search(rf"^## \[{re.escape(ver)}\]", text, re.M):
    print(f"graduate: {ver} already has a section — nothing to do")
    sys.exit(0)

m = re.search(r"^## \[Unreleased\]\s*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
if not m:
    print("graduate: no [Unreleased] section", file=sys.stderr)
    sys.exit(1)

body = m.group(1).strip("\n")
# Drop the italic placeholder ("_Nothing yet._") before graduating: it is section
# furniture, not an entry, and carrying it into the release would publish "Nothing yet"
# under a version that plainly changed something. Caught in a local dry run.
body = "\n".join(ln for ln in body.splitlines()
                 if not (ln.startswith("_") and ln.rstrip().endswith("_"))).strip("\n")
# A placeholder-only body has nothing to graduate; leave it and let the changelog test
# flag the release on the next run rather than emitting an empty version section.
if not body or not re.search(r"^- ", body, re.M):
    print("graduate: [Unreleased] holds no entries — leaving it for a human")
    sys.exit(0)

# The date is the TAG's date, not the day this ran — graduation is a manual step now, so it
# is normally days late, and the changelog test compares each section against
# `git log -1 --format=%cs <tag>`. Precedence: explicit arg, then the tag as git records it
# (resolved in the CHANGELOG's own repo, not the caller's cwd), then today() with a warning.
override = sys.argv[3] if len(sys.argv) > 3 else None
if override:
    if not DATE_RE.match(override):
        print(f"graduate: {override!r} is not a YYYY-MM-DD date", file=sys.stderr)
        sys.exit(1)
    date, source = override, "explicit argument"
else:
    tagged = ""
    try:
        tagged = subprocess.run(
            ["git", "log", "-1", "--format=%cs", tag],
            cwd=path.resolve().parent, capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:  # no git on PATH
        tagged = ""
    if DATE_RE.match(tagged):
        date, source = tagged, f"git: {tag}"
    else:
        date, source = datetime.date.today().isoformat(), "today (FALLBACK)"
        print(f"graduate: tag {tag} not found locally — dating the section {date} instead. "
              f"Run `git fetch --tags` and re-run, or pass the date explicitly as a third "
              f"argument, or the changelog date test will fail.", file=sys.stderr)

new = (f"## [Unreleased]\n\n_Nothing yet._\n\n"
       f"## [{ver}] - {date}\n\n{body}\n\n")
path.write_text(text[:m.start()] + new + text[m.end():], encoding="utf-8")
print(f"graduate: [Unreleased] -> [{ver}] - {date} [from {source}] "
      f"({len(body.splitlines())} lines)")
