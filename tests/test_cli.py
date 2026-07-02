"""Direct end-to-end tests for `terse.cli.main` subcommand wiring.

Each of these exercises the real argparse setup (subparser + defaults + dest names)
through `main()`, not just the module the subcommand delegates to — a broken `dest`,
default, or subparser wiring would only surface at manual-run time otherwise.
"""
from __future__ import annotations

import io
import json
import stat
import sys

from terse.cli import _redact_args, main

PAYLOAD = json.dumps([{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}])


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_gate_cmd_pass(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    rc = main(["gate", str(f)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "round-trip lossless: PASS" in out


def test_gate_cmd_stdin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(PAYLOAD))
    rc = main(["gate", "-"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_capture_cmd_writes_corpus(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    rc = main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)])
    assert rc == 0
    assert list(corpus.rglob("*.json")) or list(corpus.iterdir())
    assert "captured demo" in capsys.readouterr().out


def test_measure_cmd_writes_report(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    rc = main(["measure", "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    assert "Lossless gate" in out.read_text(encoding="utf-8")


def test_measure_cmd_html_flag_writes_svg_report(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    rc = main(["measure", "--corpus", str(corpus), "--out", str(out), "--html"])
    assert rc == 0
    html_out = out.with_suffix(".html")
    assert html_out.exists()
    text = html_out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "<script" not in text


def test_measure_cmd_without_html_flag_writes_no_html(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    assert main(["measure", "--corpus", str(corpus), "--out", str(out)]) == 0
    assert not out.with_suffix(".html").exists()


def test_measure_cmd_bars_flag_prints_terminal_bars(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    rc = main(["measure", "--corpus", str(corpus), "--out", str(out), "--bars"])
    assert rc == 0
    text = capsys.readouterr().out
    # the markdown report itself already prints a "Tier-0 savings by shape bucket"
    # heading, so assert on content ONLY the bar renderer emits (block glyph + legend).
    assert "█" in text
    assert "minify" in text and "tabularize" in text and "dictionary" in text


def test_measure_cmd_without_bars_flag_prints_no_terminal_bars(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    assert main(["measure", "--corpus", str(corpus), "--out", str(out)]) == 0
    assert "█" not in capsys.readouterr().out


def test_measure_cmd_empty_corpus_errors(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rc = main(["measure", "--corpus", str(corpus), "--out", str(tmp_path / "o.md")])
    assert rc == 1


def test_probe_cmd_writes_report(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "probe.md"
    rc = main(["probe", "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    assert "ceiling probes" in out.read_text(encoding="utf-8")


def test_validate_cmd_writes_report(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "tok.md"
    rc = main(["validate", "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    assert "cross-tokenizer invariance" in out.read_text(encoding="utf-8")


def test_compress_cmd_default_policy(tmp_path, capsys):
    from terse import transforms

    f = _write(tmp_path, "payload.json", PAYLOAD)
    rc = main(["compress", str(f), "--tool", "demo"])
    assert rc == 0
    captured = capsys.readouterr()
    assert transforms.decompress(captured.out) == json.loads(PAYLOAD)  # lossless
    assert "[demo]" in captured.err


def test_policy_generate_cmd_writes_and_reloads(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "policy.json"
    rc = main(["policy", "generate", "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))


def test_install_mcp_print_is_dry_run(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({
        "mcpServers": {"demo": {"command": "uvx", "args": ["demo-mcp"]}}
    }), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    rc = main(["install-mcp", "demo", "--policy", str(policy), "--print"])
    assert rc == 0
    assert "[dry-run] would wrap demo" in capsys.readouterr().out
    # dry-run must not touch the config file on disk
    assert json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["demo"]["command"] == "uvx"


def test_uninstall_mcp_print_dry_run_nothing_managed(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))

    rc = main(["uninstall-mcp", "--all", "--print"])
    assert rc == 0
    assert "nothing to do" in capsys.readouterr().out


def test_install_then_uninstall_mcp_roundtrips_via_cli(tmp_path, monkeypatch):
    cfg = tmp_path / "claude.json"
    original_entry = {"command": "uvx", "args": ["demo-mcp"]}
    cfg.write_text(json.dumps({"mcpServers": {"demo": original_entry}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    assert main(["install-mcp", "demo", "--policy", str(policy)]) == 0
    wrapped = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["demo"]
    assert wrapped["command"] != "uvx"

    assert main(["uninstall-mcp", "demo"]) == 0
    restored = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["demo"]
    assert restored == original_entry


def test_redact_args_masks_two_arg_and_equals_form_secrets():
    # --flag VALUE form: value is the NEXT arg.
    assert _redact_args(["--api-key", "sk-live-abc123", "run"]) == \
        ["--api-key", "***", "run"]
    # --flag=VALUE form: value is embedded in the same arg.
    assert _redact_args(["--token=sk-live-abc123", "run"]) == ["--token=***", "run"]
    # Non-secret flags/values pass through untouched.
    assert _redact_args(["demo-mcp", "--verbose"]) == ["demo-mcp", "--verbose"]


def test_install_mcp_print_redacts_secret_in_args(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({
        "mcpServers": {"demo": {"command": "uvx",
                                "args": ["demo-mcp", "--api-key", "sk-live-SECRETVALUE"]}}
    }), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    rc = main(["install-mcp", "demo", "--policy", str(policy), "--print"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sk-live-SECRETVALUE" not in out
    assert "--api-key ***" in out


def test_fluency_cmd_keyless_pack_is_written_with_restricted_permissions(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    pack = tmp_path / "fluency-pack.json"

    rc = main(["fluency", "--corpus", str(corpus), "--pack", str(pack),
              "--out", str(tmp_path / "fluency-report.md")])
    assert rc == 0
    assert pack.exists()
    if sys.platform != "win32":
        mode = stat.S_IMODE(pack.stat().st_mode)
        assert mode == 0o600, f"pack file mode {oct(mode)} is not restricted to 0600"
