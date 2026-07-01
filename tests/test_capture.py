"""Tests for corpus capture persistence, including the permission hardening on
`capture_payload`/`append_audit` — both can persist real MCP tool traffic to disk."""
from __future__ import annotations

import json
import stat

from terse.capture import append_audit, capture_payload


def test_capture_payload_writes_owner_only_file(tmp_path):
    raw = json.dumps({"id": 1})
    path = capture_payload("demo.tool", raw, tmp_path / "corpus")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["raw"] == raw


def test_capture_payload_is_idempotent_by_sha_and_stays_restricted(tmp_path):
    raw = json.dumps({"id": 1})
    corpus = tmp_path / "corpus"
    p1 = capture_payload("demo.tool", raw, corpus)
    p2 = capture_payload("demo.tool", raw, corpus)
    assert p1 == p2
    assert stat.S_IMODE(p1.stat().st_mode) == 0o600


def test_append_audit_writes_owner_only_and_appends(tmp_path):
    log = tmp_path / "debug.jsonl"
    append_audit({"id": 1}, log)
    append_audit({"id": 2}, log)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["id"] for ln in lines] == [1, 2]
