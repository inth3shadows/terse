"""Tests for terse._secure_io (write_restricted / append_restricted)."""
from __future__ import annotations

import os
import stat

import pytest

from terse import _secure_io as sio


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


# --- write_restricted ---

def test_write_restricted_creates_file_with_mode_0600(tmp_path):
    path = tmp_path / "secret.txt"
    sio.write_restricted(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert _mode(path) == 0o600


def test_write_restricted_truncates_existing_content(tmp_path):
    path = tmp_path / "secret.txt"
    sio.write_restricted(path, "a long first write")
    sio.write_restricted(path, "short")
    assert path.read_text(encoding="utf-8") == "short"


def test_write_restricted_tightens_a_preexisting_looser_file(tmp_path):
    path = tmp_path / "secret.txt"
    path.touch()
    os.chmod(path, 0o644)
    sio.write_restricted(path, "hello")
    assert _mode(path) == 0o600


def test_write_restricted_bypasses_umask(tmp_path):
    # A naive open()+chmod() sequence would briefly expose the file at whatever
    # the permissive umask allows; fchmod-before-write must not inherit that.
    path = tmp_path / "secret.txt"
    original_umask = os.umask(0o000)
    try:
        sio.write_restricted(path, "hello")
    finally:
        os.umask(original_umask)
    assert _mode(path) == 0o600


def test_write_restricted_writes_utf8_content_correctly(tmp_path):
    path = tmp_path / "secret.txt"
    text = "em—dash and \U0001f512 emoji"
    sio.write_restricted(path, text)
    assert path.read_text(encoding="utf-8") == text


def test_write_restricted_custom_mode_param(tmp_path):
    path = tmp_path / "secret.txt"
    sio.write_restricted(path, "hello", mode=0o640)
    assert _mode(path) == 0o640


def test_write_restricted_reraises_on_fchmod_failure_without_masking(tmp_path, monkeypatch):
    path = tmp_path / "secret.txt"

    def _boom(*_a):
        raise OSError("boom")

    monkeypatch.setattr(os, "fchmod", _boom)
    with pytest.raises(OSError, match="boom"):
        sio.write_restricted(path, "hello")
    # Atomic write: on failure nothing is committed to the target (no orphan empty
    # file), and the staging temp is cleaned up — the directory is left as it was.
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_restricted_failed_write_preserves_existing_target(tmp_path, monkeypatch):
    # The whole point of the atomic write: a crash/error mid-write must leave the
    # PREVIOUS complete file intact, never a truncated ruin (the ~/.claude.json case).
    path = tmp_path / "config.json"
    path.write_text("original-and-valid", encoding="utf-8")

    def _boom(*_a):
        raise OSError("boom")

    monkeypatch.setattr(os, "fchmod", _boom)
    with pytest.raises(OSError, match="boom"):
        sio.write_restricted(path, "new-content-that-fails")
    assert path.read_text(encoding="utf-8") == "original-and-valid"
    assert list(tmp_path.iterdir()) == [path]  # no leftover temp


# --- append_restricted ---

def test_append_restricted_creates_file_with_mode_0600(tmp_path):
    path = tmp_path / "secret.txt"
    sio.append_restricted(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert _mode(path) == 0o600


def test_append_restricted_appends_without_truncating(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("original ", encoding="utf-8")
    sio.append_restricted(path, "first ")
    sio.append_restricted(path, "second")
    assert path.read_text(encoding="utf-8") == "original first second"


def test_append_restricted_tightens_a_preexisting_looser_file(tmp_path):
    path = tmp_path / "secret.txt"
    path.touch()
    os.chmod(path, 0o644)
    sio.append_restricted(path, "hello")
    assert _mode(path) == 0o600


def test_append_restricted_bypasses_umask(tmp_path):
    path = tmp_path / "secret.txt"
    original_umask = os.umask(0o000)
    try:
        sio.append_restricted(path, "hello")
    finally:
        os.umask(original_umask)
    assert _mode(path) == 0o600


def test_append_restricted_custom_mode_param(tmp_path):
    path = tmp_path / "secret.txt"
    sio.append_restricted(path, "hello", mode=0o640)
    assert _mode(path) == 0o640


def test_append_restricted_reraises_on_fchmod_failure_without_masking(tmp_path, monkeypatch):
    path = tmp_path / "secret.txt"

    def _boom(*_a):
        raise OSError("boom")

    monkeypatch.setattr(os, "fchmod", _boom)
    with pytest.raises(OSError, match="boom"):
        sio.append_restricted(path, "hello")
    assert path.read_text(encoding="utf-8") == ""


# --- symlink refusal (O_NOFOLLOW): a secret-bearing write must not be redirected
#     through a pre-planted symlink onto an attacker-chosen target ---

_needs_nofollow = pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"), reason="platform has no O_NOFOLLOW")


@_needs_nofollow
def test_write_restricted_refuses_to_follow_a_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(OSError):
        sio.write_restricted(link, "pwned")
    assert target.read_text(encoding="utf-8") == "original"  # target never truncated/written


@_needs_nofollow
def test_append_restricted_refuses_to_follow_a_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(OSError):
        sio.append_restricted(link, "pwned")
    assert target.read_text(encoding="utf-8") == "original"
