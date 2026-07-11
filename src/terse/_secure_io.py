"""Write/append text to files that may hold secrets (API keys in MCP `env` blocks, raw
tool-call payloads) without ever leaving them world/group-readable — including the brief
window a plain `write_text()` + `os.chmod()` sequence leaves at the process's default
umask between file creation and the chmod call. `os.fchmod` runs on the open descriptor
before any content is written, so the restrictive mode is in place first, not after."""
from __future__ import annotations

import os
from pathlib import Path

# Refuse to open a symlink as the final path component, so a pre-planted symlink in a
# terse-managed dir can't redirect a secret-bearing write (config, backup, captured
# payload) onto an attacker-chosen target. Guards only the LAST component (the standard
# O_NOFOLLOW limitation); `getattr` keeps this a no-op on a platform without the flag
# (e.g. Windows) rather than a crash.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def write_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW, mode)
    try:
        os.fchmod(fd, mode)  # also tightens a file that pre-existed at looser permissions
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def append_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _NOFOLLOW, mode)
    try:
        os.fchmod(fd, mode)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(text)
