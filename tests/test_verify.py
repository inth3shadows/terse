"""Tests for `terse verify` — the self-contained verification report."""
from __future__ import annotations

import json

from terse import capture
from terse.cli import main


def test_verify_on_corpus_emits_gate_savings_and_attestation(tmp_path):
    payload = json.dumps([{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}])
    capture.capture_payload("demo.tool", payload, tmp_path / "corpus")
    out = tmp_path / "verify.md"

    rc = main(["verify", "--corpus", str(tmp_path / "corpus"), "--out", str(out)])
    assert rc == 0

    text = out.read_text(encoding="utf-8")
    assert "# terse — verification report" in text        # attestation header
    assert "round-trip losslessly" in text                # lossless gate ran
    assert "Fail-open" in text and "No UNEXPECTED egress" in text  # self-cert caveats present
    assert "your captured traffic" in text                 # labelled as real corpus


def test_verify_html_flag_writes_svg_report_alongside_markdown(tmp_path):
    payload = json.dumps([{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}])
    capture.capture_payload("demo.tool", payload, tmp_path / "corpus")
    out = tmp_path / "verify.md"

    rc = main(["verify", "--corpus", str(tmp_path / "corpus"), "--out", str(out), "--html"])
    assert rc == 0

    html_out = out.with_suffix(".html")
    assert html_out.exists()
    text = html_out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "your captured traffic" in text  # attestation card carries the corpus label
    assert "<script" not in text


def test_verify_bars_flag_prints_terminal_bars(tmp_path, capsys):
    payload = json.dumps([{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}])
    capture.capture_payload("demo.tool", payload, tmp_path / "corpus")
    out = tmp_path / "verify.md"

    rc = main(["verify", "--corpus", str(tmp_path / "corpus"), "--out", str(out), "--bars"])
    assert rc == 0

    text = capsys.readouterr().out
    assert "█" in text
    assert "minify" in text and "tabularize" in text and "dictionary" in text


def test_verify_empty_corpus_errors(tmp_path):
    (tmp_path / "corpus").mkdir()
    out = tmp_path / "o.md"
    rc = main(["verify", "--corpus", str(tmp_path / "corpus"), "--out", str(out)])
    assert rc == 1
    assert not out.exists()
