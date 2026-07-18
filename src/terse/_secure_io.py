"""Write/append text to files that may hold secrets (API keys in MCP `env` blocks, raw
tool-call payloads) without ever leaving them world/group-readable — including the brief
window a plain `write_text()` + `os.chmod()` sequence leaves at the process's default
umask between file creation and the chmod call. `os.fchmod` runs on the open descriptor
before any content is written, so the restrictive mode is in place first, not after."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Refuse to open a symlink as the final path component, so a pre-planted symlink in a
# terse-managed dir can't redirect a secret-bearing write (config, backup, captured
# payload) onto an attacker-chosen target. Guards only the LAST component (the standard
# O_NOFOLLOW limitation); `getattr` keeps this a no-op on a platform without the flag
# (e.g. Windows) rather than a crash.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def write_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    # Atomic full-file write: stage into a sibling temp file, fsync, then os.replace()
    # onto the target. A crash/SIGKILL mid-write can now only leave the temp behind —
    # the target is either the complete old file or the complete new one, never a
    # half-truncated ruin. This matters most for the real ~/.claude.json and policy.json
    # writes routed through here: the previous O_TRUNC-in-place write left a window in
    # which a crash corrupted the user's live config, recoverable only by hand from a
    # .bak. os.replace is atomic within a filesystem, so the temp sits in the same dir.
    path = Path(path)
    # Preserve the original O_NOFOLLOW contract: refuse to write onto a symlinked target.
    # os.replace() below would itself be safe (it replaces a destination symlink rather
    # than following it, so a secret can't be redirected onto an attacker's target), but
    # refusing keeps the loud, unchanged behavior — a planted symlink is surfaced, and a
    # legitimately symlinked config is never silently converted into a regular file.
    if _NOFOLLOW and path.is_symlink():
        raise OSError(f"terse: refusing to write through a symlink at {path}")
    # mkstemp gives a GUARANTEED-UNIQUE temp name in the target dir, created O_EXCL at
    # 0600. A fixed `.name.tmp-<pid>` name would collide: a prior process SIGKILL'd
    # mid-write leaves that temp, and since Linux reuses PIDs, a later write to the same
    # path would hit O_EXCL -> FileExistsError and stay permanently broken until the stale
    # temp is removed by hand. A random name also removes any symlink-preplant angle at the
    # temp (its name is unpredictable), so dropping _NOFOLLOW on it is safe.
    fd, tmpname = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmpname)
    try:
        os.fchmod(fd, mode)  # pin the mode before content, independent of umask
    except BaseException:
        os.close(fd)
        _silent_unlink(tmp)
        raise
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:  # takes ownership of fd
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        _silent_unlink(tmp)
        raise


def _silent_unlink(path: str | Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def append_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _NOFOLLOW, mode)
    try:
        os.fchmod(fd, mode)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(text)
