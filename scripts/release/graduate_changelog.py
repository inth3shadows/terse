#!/usr/bin/env python3
"""Move `## [Unreleased]` into a versioned section. Called by release.yml at tag time."""
import datetime
import pathlib
import re
import sys

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

date = datetime.date.today().isoformat()
new = (f"## [Unreleased]\n\n_Nothing yet._\n\n"
       f"## [{ver}] - {date}\n\n{body}\n\n")
path.write_text(text[:m.start()] + new + text[m.end():], encoding="utf-8")
print(f"graduate: [Unreleased] -> [{ver}] - {date} ({len(body.splitlines())} lines)")
