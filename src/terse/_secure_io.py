"""Write/append text to files that may hold secrets (API keys in MCP `env` blocks, raw
tool-call payloads) without ever leaving them world/group-readable — including the brief
window a plain `write_text()` + `os.chmod()` sequence leaves at the process's default
umask between file creation and the chmod call. `os.fchmod` runs on the open descriptor
before any content is written, so the restrictive mode is in place first, not after."""
from __future__ import annotations

import os
from pathlib import Path


def write_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)  # also tightens a file that pre-existed at looser permissions
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def append_restricted(path: str | Path, text: str, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode)
    try:
        os.fchmod(fd, mode)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(text)
