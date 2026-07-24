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


def test_write_restricted_survives_a_stale_temp_from_a_crashed_run(tmp_path):
    # A prior process SIGKILL'd mid-write can leave a staging temp behind. With a unique
    # (mkstemp) temp name, a later write to the same target must NOT collide with it —
    # a fixed pid-based name did (O_EXCL FileExistsError permanently broke the write).
    target = tmp_path / "config.json"
    target.write_text("original", encoding="utf-8")
    (tmp_path / f".{target.name}.leftover.tmp").write_text("junk from a crash", encoding="utf-8")
    sio.write_restricted(target, "new")  # must succeed, not raise FileExistsError
    assert target.read_text(encoding="utf-8") == "new"


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


# --- mkdir_restricted (security audit 2026-07-23) --------------------------------------
# Files terse writes were already 0600, but the directories holding them were created at
# the process umask (0755 in practice). Contents were never exposed — the NAMES were: a
# corpus entry is `{tool}__{sha8}.json`, so any local user could read off the tool
# inventory and payload content hashes.

def test_mkdir_restricted_creates_dir_owner_only(tmp_path):
    d = tmp_path / "corpus"
    sio.mkdir_restricted(d)
    assert d.is_dir()
    assert _mode(d) == 0o700


def test_mkdir_restricted_leaves_an_existing_dir_alone(tmp_path):
    # `--capture-dir` can point anywhere. Silently chmod'ing a directory the operator
    # created for their own reasons is a side effect a logging call has no business
    # having, so a pre-existing dir keeps whatever mode it had.
    d = tmp_path / "preexisting"
    d.mkdir(mode=0o755)
    sio.mkdir_restricted(d)
    assert _mode(d) == 0o755


def test_mkdir_restricted_is_idempotent_and_creates_parents(tmp_path):
    d = tmp_path / "a" / "b" / "corpus"
    sio.mkdir_restricted(d)
    sio.mkdir_restricted(d)          # second call must not raise
    assert _mode(d) == 0o700
    # Ancestors are ordinary path components (`~/.local/state`), not payload dirs —
    # deliberately left at the default umask rather than tightened.
    assert (tmp_path / "a").is_dir()


def test_capture_corpus_dir_is_created_owner_only(tmp_path):
    # End-to-end through the real caller: the corpus dir, not just the helper.
    from terse.capture import capture_payload

    corpus = tmp_path / "session-corpus"
    capture_payload("some.tool", '{"a": 1}', corpus)
    assert _mode(corpus) == 0o700
    assert all(_mode(f) == 0o600 for f in corpus.glob("*.json"))


def test_mkdir_restricted_still_raises_when_the_path_is_a_file(tmp_path):
    # `Path.mkdir(exist_ok=True)` deliberately re-raises for a NON-directory. Swallowing
    # every FileExistsError dropped that, turning a clear "File exists" into a baffling
    # NotADirectoryError thrown much later from inside tempfile.mkstemp on the first write.
    clash = tmp_path / "corpus"
    clash.write_text("i am a file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sio.mkdir_restricted(clash)
