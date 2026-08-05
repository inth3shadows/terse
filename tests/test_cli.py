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

from terse.cli import _corpus_size, _redact_args, main

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


def test_measure_cmd_history_flag_records_and_prints_trend(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    history = tmp_path / "history.jsonl"

    rc1 = main(["measure", "--corpus", str(corpus), "--out", str(out),
               "--history", str(history)])
    assert rc1 == 0
    text1 = capsys.readouterr().out
    assert "1 run(s) recorded" in text1
    assert "at least two" in text1  # only one run so far -> no delta table yet
    assert history.exists()
    assert len(history.read_text(encoding="utf-8").splitlines()) == 1

    rc2 = main(["measure", "--corpus", str(corpus), "--out", str(out),
               "--history", str(history)])
    assert rc2 == 0
    text2 = capsys.readouterr().out
    assert "2 run(s) recorded" in text2
    assert "Trend across runs" in text2
    assert len(history.read_text(encoding="utf-8").splitlines()) == 2


def test_measure_cmd_without_history_flag_writes_no_history_file(tmp_path):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    out = tmp_path / "report.md"
    assert main(["measure", "--corpus", str(corpus), "--out", str(out)]) == 0
    assert not (tmp_path / "history.jsonl").exists()


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


def test_install_then_uninstall_mcp_scope_project_via_cli(tmp_path, monkeypatch, capsys):
    # #58: --scope project wraps/unwraps a .mcp.json directly, independent of
    # $CLAUDE_CONFIG (which stays untouched by a project-scope run).
    mcp_json = tmp_path / ".mcp.json"
    original_entry = {"command": "uvx", "args": ["demo-mcp"]}
    mcp_json.write_text(json.dumps({"mcpServers": {"demo": original_entry}}), encoding="utf-8")
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    rc = main(["install-mcp", "demo", "--policy", str(policy), "--scope", "project",
              "--file", str(mcp_json)])
    assert rc == 0
    assert "scope: project" in capsys.readouterr().out
    wrapped = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"]["demo"]
    assert wrapped["command"] != "uvx"

    assert main(["uninstall-mcp", "demo", "--scope", "project", "--file", str(mcp_json)]) == 0
    restored = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"]["demo"]
    assert restored == original_entry


def test_mcp_status_cmd_reports_wrapped_and_unwrapped(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "uvx", "args": ["plain-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    assert main(["install-mcp", "plain", "--policy", str(policy)]) == 0
    capsys.readouterr()  # drain install-mcp's own output

    rc = main(["mcp-status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[user]" in out
    assert "plain" in out and "wrapped" in out


def test_mcp_status_cmd_empty_prints_nothing_found(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)

    rc = main(["mcp-status"])
    assert rc == 0
    assert "no MCP servers found" in capsys.readouterr().out


def test_mcp_status_cmd_json_emits_empty_list_when_no_servers(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)

    assert main(["mcp-status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_mcp_status_cmd_shows_detail_line_and_json(tmp_path, monkeypatch, capsys):
    # The wrapped entry gets a second line surfacing what it fronts + the tiers baked
    # in — and --json exposes the same fields for scripts.
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "uvx", "args": ["plain-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))
    assert main(["install-mcp", "plain", "--policy", str(policy), "--no-diff"]) == 0
    capsys.readouterr()  # drain install-mcp output

    assert main(["mcp-status"]) == 0
    out = capsys.readouterr().out
    # wraps carries the full downstream command (orig command + its args)
    assert "wraps=uvx plain-mcp" in out and "diff=off" in out and "stats=on" in out

    assert main(["mcp-status", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    row = next(r for r in rows if r["server"] == "plain")
    assert row["wraps"] == "uvx plain-mcp" and row["diff"] == "off" and row["stats"] is True
    assert row["policy_missing"] is False


def test_mcp_status_cmd_flags_a_launcher_that_no_longer_resolves(tmp_path, monkeypatch, capsys):
    # The silent failure a wrapped entry can't report itself: if `command` stops
    # resolving (an upgrade moves a versioned uv-tool/pipx venv), the client cannot spawn
    # the proxy at all and the server just shows up with no tools. Status has to say so.
    from terse import install_mcp as im

    launcher = tmp_path / "python"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "uvx", "args": ["plain-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(im, "terse_invocation", lambda: [str(launcher), "-m", "terse"])
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))
    assert main(["install-mcp", "plain", "--policy", str(policy)]) == 0
    capsys.readouterr()

    assert main(["mcp-status"]) == 0
    assert "launcher=" not in capsys.readouterr().out  # resolves -> stays quiet

    launcher.unlink()
    assert main(["mcp-status"]) == 0
    out = capsys.readouterr().out
    assert f"launcher={launcher} (MISSING)" in out and "cannot start" in out

    assert main(["mcp-status", "--json"]) == 0
    row = next(r for r in json.loads(capsys.readouterr().out) if r["server"] == "plain")
    assert row["launcher_missing"] is True


def test_mcp_status_cmd_flags_a_missing_policy_file(tmp_path, monkeypatch, capsys):
    # A wrapped server whose policy file was deleted/moved after install would fail to
    # start (the proxy raises on load), but the flat status line showed it as normal.
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "uvx", "args": ["plain-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))
    assert main(["install-mcp", "plain", "--policy", str(policy)]) == 0
    capsys.readouterr()
    policy.unlink()  # the policy vanishes after install -> proxy would fail to launch

    assert main(["mcp-status"]) == 0
    assert "(MISSING)" in capsys.readouterr().out


def test_proxy_cmd_parses_and_forwards_headers(monkeypatch):
    # #5: --header NAME=VALUE (repeatable) must reach run_proxy as a dict, and the
    # positional REMAINDER cmd must still come through unchanged (a single URL here).
    captured = {}

    def fake_run_proxy(cmd, pol, debug=False, stdin=None, stdout=None,
                       capture_dir=None, debug_log=None, headers=None, stats_log=None,
                       server_name=None):
        captured["cmd"] = cmd
        captured["headers"] = headers
        captured["stats_log"] = stats_log
        return 0

    monkeypatch.setattr("terse.proxy.run_proxy", fake_run_proxy)
    rc = main(["proxy", "--header", "Authorization=Bearer xyz", "--header", "X-Id=42",
              "--", "https://example.com/mcp"])
    assert rc == 0
    assert captured["cmd"] == ["https://example.com/mcp"]
    assert captured["headers"] == {"Authorization": "Bearer xyz", "X-Id": "42"}
    # the savings ledger defaults ON, resolved to the XDG path (see stats.py)
    assert captured["stats_log"].endswith("terse/stats.jsonl")


def test_proxy_cmd_rejects_malformed_header_without_launching(monkeypatch, capsys):
    def fake_run_proxy(*_a, **_kw):
        raise AssertionError("run_proxy must not be called on a malformed --header")

    monkeypatch.setattr("terse.proxy.run_proxy", fake_run_proxy)
    rc = main(["proxy", "--header", "no-equals-sign", "--", "uvx", "some-mcp"])
    assert rc == 2
    assert "NAME=VALUE" in capsys.readouterr().err


def test_proxy_cmd_missing_command_with_bad_policy_still_shows_clean_error(tmp_path, capsys):
    # Regression: --policy used to be loaded BEFORE the missing-downstream-command
    # check, so a bad/missing --policy path crashed with an uncaught traceback (exit 1)
    # instead of the clean "provide the downstream server command" message (exit 2) —
    # regardless of whether --policy was even the thing the user got wrong.
    rc = main(["proxy", "--policy", str(tmp_path / "does-not-exist.json")])
    assert rc == 2
    assert "provide the downstream server command" in capsys.readouterr().err


def test_proxy_cmd_rejects_header_with_config(tmp_path, monkeypatch, capsys):
    # Regression: --header was silently discarded when combined with --config (only the
    # single-downstream branch ever read args.header) — no warning, no error.
    def fake_run_multi_proxy(*_a, **_kw):
        raise AssertionError("run_multi_proxy must not be called when --header is "
                             "combined with --config")

    monkeypatch.setattr("terse.multiproxy.run_multi_proxy", fake_run_multi_proxy)
    cfg = tmp_path / "peers.json"
    cfg.write_text("{}", encoding="utf-8")
    rc = main(["proxy", "--header", "Authorization=Bearer xyz", "--config", str(cfg)])
    assert rc == 2
    assert "--header" in capsys.readouterr().err


def test_proxy_cmd_forwards_server_name(monkeypatch):
    captured = {}

    def fake_run_proxy(cmd, pol, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr("terse.proxy.run_proxy", fake_run_proxy)
    assert main(["proxy", "--server-name", "runecho", "--", "uvx", "runecho-mcp"]) == 0
    assert captured["server_name"] == "runecho"
    # absent by default — no flag means pre-#83 matching, not a guessed name
    assert main(["proxy", "--", "uvx", "runecho-mcp"]) == 0
    assert captured["server_name"] is None


def test_proxy_cmd_rejects_server_name_with_config(tmp_path, monkeypatch, capsys):
    # A single flag can't name N peers; run_multi_proxy uses each peer's config "name"
    # instead. Reject loudly rather than silently ignoring it (as --header already does).
    def fake_run_multi_proxy(*_a, **_kw):
        raise AssertionError("run_multi_proxy must not be called with --server-name")

    monkeypatch.setattr("terse.multiproxy.run_multi_proxy", fake_run_multi_proxy)
    cfg = tmp_path / "peers.json"
    cfg.write_text("{}", encoding="utf-8")
    rc = main(["proxy", "--server-name", "gh", "--config", str(cfg)])
    assert rc == 2
    assert "--server-name" in capsys.readouterr().err


def test_proxy_cmd_no_stats_disables_ledger(monkeypatch):
    captured = {}

    def fake_run_proxy(cmd, pol, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr("terse.proxy.run_proxy", fake_run_proxy)
    rc = main(["proxy", "--no-stats", "--", "uvx", "some-mcp"])
    assert rc == 0
    assert captured["stats_log"] is None


def test_proxy_cmd_rejects_no_stats_with_stats_log(monkeypatch, capsys):
    def fake_run_proxy(*_a, **_kw):
        raise AssertionError("run_proxy must not be called on contradictory stats flags")

    monkeypatch.setattr("terse.proxy.run_proxy", fake_run_proxy)
    rc = main(["proxy", "--no-stats", "--stats-log", "/x/s.jsonl", "--", "uvx", "some-mcp"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_stats_cmd_reports_over_a_ledger(tmp_path, capsys):
    from terse.stats import append_stats
    log = tmp_path / "stats.jsonl"
    append_stats({"ts": 1, "server": "runecho", "tool": "structure",
                  "decision": "diff", "raw_chars": 400, "out_chars": 40,
                  "raw_tokens": 100, "out_tokens": 10}, log)
    rc = main(["stats", "--log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "saved 90" in out and "runecho" in out and "structure" in out


def test_stats_cmd_json_output(tmp_path, capsys):
    from terse.stats import append_stats
    log = tmp_path / "stats.jsonl"
    append_stats({"ts": 1, "server": "s", "tool": "t", "decision": "compressed",
                  "raw_chars": 400, "out_chars": 40,
                  "raw_tokens": 100, "out_tokens": 10}, log)
    rc = main(["stats", "--log", str(log), "--json"])
    assert rc == 0
    agg = json.loads(capsys.readouterr().out)
    assert agg["total"]["blocks"] == 1 and agg["total"]["raw_tokens"] == 100


def test_stats_cmd_missing_ledger_is_a_clean_error(tmp_path, capsys):
    rc = main(["stats", "--log", str(tmp_path / "absent.jsonl")])
    assert rc == 2
    assert "no ledger" in capsys.readouterr().err


def test_stats_cmd_rejects_bad_since_window(tmp_path, capsys):
    log = tmp_path / "stats.jsonl"
    log.write_text("", encoding="utf-8")
    rc = main(["stats", "--log", str(log), "--since", "fortnight"])
    assert rc == 2
    assert "bad --since window" in capsys.readouterr().err


def test_corpus_size_skips_underscore_prefixed_sidecars(tmp_path):
    # Real capture envelopes count...
    _write(tmp_path, "read_graph__a1b2c3d4.json", PAYLOAD)
    _write(tmp_path, "search_nodes__e5f6a7b8.json", PAYLOAD)
    # ...but `_calls.json` (mcp_probe.py's argument sidecar, #138) is never a capture
    # envelope and must not inflate the "already holds N payload(s)" install-time count.
    _write(tmp_path, "_calls.json", json.dumps({"server_name": "memory", "calls": []}))
    assert _corpus_size(str(tmp_path)) == 2


def test_corpus_size_absent_or_unreadable_is_zero(tmp_path):
    assert _corpus_size(None) == 0
    assert _corpus_size(str(tmp_path / "does-not-exist")) == 0


def test_redact_args_masks_two_arg_and_equals_form_secrets():
    # --flag VALUE form: value is the NEXT arg.
    assert _redact_args(["--api-key", "sk-live-abc123", "run"]) == \
        ["--api-key", "***", "run"]
    # --flag=VALUE form: value is embedded in the same arg.
    assert _redact_args(["--token=sk-live-abc123", "run"]) == ["--token=***", "run"]
    # Non-secret flags/values pass through untouched.
    assert _redact_args(["demo-mcp", "--verbose"]) == ["demo-mcp", "--verbose"]


def test_redact_args_masks_secret_shaped_header_values():
    # `--header NAME=VALUE` (#5) carries its secret in the VALUE half, not a flag NAME
    # `_SECRET_FLAG` would match — a bearer token must still be masked before printing.
    assert _redact_args(["--header", "Authorization=Bearer sk-live-abc123", "--", "url"]) == \
        ["--header", "Authorization=***", "--", "url"]
    # Inline `--header=NAME=VALUE` form.
    assert _redact_args(["--header=X-Api-Key=sk-live-abc123"]) == ["--header=X-Api-Key=***"]
    # A non-secret-shaped header name passes through untouched.
    assert _redact_args(["--header", "X-Request-Id=abc123"]) == \
        ["--header", "X-Request-Id=abc123"]


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


def test_install_mcp_print_shows_url_and_redacted_headers_for_url_server(tmp_path, monkeypatch, capsys):
    # Regression: _short_cmd only read entry["command"]/entry["args"], so a
    # url/headers-shaped entry (#5 HTTP downstream) printed just "before: ?" —
    # silently losing the url/headers info from the dry-run display.
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({
        "mcpServers": {"remote": {"url": "https://example.com/mcp",
                                  "headers": {"Authorization": "Bearer sk-live-SECRETVALUE"}}}
    }), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    policy = _write(tmp_path, "policy.json", json.dumps({"rules": []}))

    rc = main(["install-mcp", "remote", "--policy", str(policy), "--print"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "before: ?" not in out
    assert "https://example.com/mcp" in out
    assert "sk-live-SECRETVALUE" not in out          # secret-shaped header value masked
    assert "Authorization=***" in out


def test_fluency_cmd_bars_flag_prints_forest_plot(tmp_path, capsys):
    f = _write(tmp_path, "payload.json", PAYLOAD)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "demo", "--corpus", str(corpus)]) == 0
    pack_path = tmp_path / "fluency-pack.json"
    assert main(["fluency", "--corpus", str(corpus), "--pack", str(pack_path),
                "--out", str(tmp_path / "fluency-report.md")]) == 0
    capsys.readouterr()  # discard the pack-writing message

    pack = json.loads(pack_path.read_text(encoding="utf-8"))

    def gt(q):
        return json.dumps(q["expected"]) if q["qtype"] == "enumerate" else str(q["expected"])

    responses = {"perfect-model": {
        p["sha"]: {q["qid"]: {"raw": gt(q), "terse": gt(q), "primer": gt(q)}
                   for q in p["questions"]}
        for p in pack["payloads"]
    }}
    responses_path = _write(tmp_path, "responses.json", json.dumps(responses))

    rc = main(["fluency", "--corpus", str(corpus), "--pack", str(pack_path),
              "--responses", str(responses_path), "--out", str(tmp_path / "fluency-report.md"),
              "--bars"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "PASS" in text
    assert "perfect-model" in text


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


def test_fluency_diff_html_writes_forest_plot(tmp_path, monkeypatch, capsys):
    # `fluency --diff --html` writes the forest-plot companion next to --out. The diff
    # eval needs a live model, so stub the answerer factory with a canned one — this
    # exercises the real parser flag -> branch -> _maybe_write_diff_html wiring.
    from terse import cli

    monkeypatch.setattr(cli, "_build_answerers", lambda args, make: {"m": lambda s, u: "9"})
    corpus = tmp_path / "corpus"
    prev = json.dumps([{"id": i, "status": "active-long-status-value", "score": i}
                       for i in range(8)])
    curr = json.dumps([{"id": i, "status": "active-long-status-value", "score": i}
                       for i in range(8)] + [{"id": 8, "status": "active-long-status-value",
                                              "score": 99}])
    assert main(["capture", str(_write(tmp_path, "prev.json", prev)),
                 "--tool", "demo", "--corpus", str(corpus)]) == 0
    assert main(["capture", str(_write(tmp_path, "curr.json", curr)),
                 "--tool", "demo", "--corpus", str(corpus)]) == 0
    capsys.readouterr()

    out_md = tmp_path / "rep" / "diff.md"
    assert main(["fluency", "--diff", "--html", "--corpus", str(corpus),
                 "--out", str(out_md)]) == 0
    assert out_md.exists()
    html = out_md.with_suffix(".html")
    assert html.exists()
    text = html.read_text(encoding="utf-8")
    assert "<svg" in text and "diff-form" in text and "full-terse" in text


def test_fluency_html_flag_ignored_in_base_mode_prints_note(tmp_path, capsys):
    # --html only renders in the paired diff-family evals; elsewhere it must say so
    # rather than be a silent no-op.
    corpus = tmp_path / "corpus"
    assert main(["capture", str(_write(tmp_path, "p.json", PAYLOAD)),
                 "--tool", "demo", "--corpus", str(corpus)]) == 0
    capsys.readouterr()
    rc = main(["fluency", "--html", "--corpus", str(corpus),
               "--pack", str(tmp_path / "pack.json"), "--out", str(tmp_path / "rep.md")])
    assert rc == 0
    assert "applies only to --diff" in capsys.readouterr().out


def test_tune_cmd_surfaces_drop_candidate_and_writes_policy(tmp_path, capsys):
    payload = json.dumps({"result": [{"id": i, "description": "d" * 250 + str(i)}
                                     for i in range(20)]})
    f = _write(tmp_path, "p.json", payload)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "kb.x", "--corpus", str(corpus)]) == 0
    out_pol = tmp_path / "new_dir" / "policy.json"     # parent doesn't exist -> must be created
    assert main(["tune", "--corpus", str(corpus), "--out", str(out_pol)]) == 0
    out = capsys.readouterr().out
    assert "drop candidate" in out
    assert "result[].description" in out and "[prose]" in out
    doc = json.loads(out_pol.read_text())
    entry = next(p for p in doc["policies"] if p["match"]["tool"] == "kb.x")
    assert "result[].description" in entry["_suggested_fields"]      # written INACTIVE


def test_tune_cmd_shows_bucket_savings_rollup(tmp_path, capsys):
    # Beyond each candidate's individual token share, the SAFE bucket gets a rollup:
    # the estimated gross tokens dropping the whole bucket would evict + its corpus
    # share — the number an operator needs to decide "is enabling the SAFE set worth it?"
    import re

    payload = json.dumps({"result": [{"id": i, "description": "d" * 250 + str(i)}
                                     for i in range(20)]})
    f = _write(tmp_path, "p.json", payload)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "kb.x", "--corpus", str(corpus)]) == 0
    assert main(["tune", "--corpus", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "enabling all 1 here" in out and "of corpus" in out and "gross" in out
    m = re.search(r"≈([\d,]+) tok", out)             # a real, non-zero estimate rendered
    assert m and int(m.group(1).replace(",", "")) > 0


def test_tune_cmd_reports_no_candidates(tmp_path, capsys):
    payload = json.dumps({"result": [{"id": i, "status": "ok"} for i in range(5)]})
    f = _write(tmp_path, "p.json", payload)
    corpus = tmp_path / "corpus"
    assert main(["capture", str(f), "--tool", "x.y", "--corpus", str(corpus)]) == 0
    assert main(["tune", "--corpus", str(corpus)]) == 0
    assert "no drop-to-retrieve candidates" in capsys.readouterr().out


# --- #136: `policy autotune` writes nothing without --apply, and never a broken policy ---

def _autotune_setup(tmp_path):
    """A deployed policy carrying operator-only decisions, plus a corpus that would
    otherwise regenerate the rule from scratch."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(6):
        rec = {"tool": "gh.items", "captured_at": 1_000_000_000 + i,
               "raw": json.dumps({"result": [{"id": j, "status": "active",
                                              "url": "https://x.example/api/items"}
                                             for j in range(20)]})}
        (corpus / f"gh.items__{i}.json").write_text(json.dumps(rec), encoding="utf-8")
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({
        "version": 1,
        "never_lossy_servers": ["secret-broker"],
        "policies": [{"match": {"tool": "gh.items"}, "tiers": [], "capture": False,
                      "structured": "leave"}],
    }), encoding="utf-8")
    return pol, corpus


def test_policy_autotune_writes_nothing_without_apply(tmp_path, capsys):
    pol, corpus = _autotune_setup(tmp_path)
    before = pol.read_bytes()
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus)])
    out = capsys.readouterr().out
    assert rc == 0
    assert pol.read_bytes() == before                      # byte-for-byte untouched
    assert "nothing written" in out
    assert "gh.items" in out                               # the change WAS proposed


def test_policy_autotune_apply_preserves_operator_keys(tmp_path):
    pol, corpus = _autotune_setup(tmp_path)
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
               "--apply"])
    assert rc == 0
    doc = json.loads(pol.read_text(encoding="utf-8"))
    rule = doc["policies"][0]
    assert rule["capture"] is False                        # operator's, not the corpus's
    assert rule["structured"] == "leave"
    assert doc["never_lossy_servers"] == ["secret-broker"]
    assert rule["tiers"]                                   # the corpus DID re-decide this


def test_policy_autotune_refuses_to_write_a_policy_it_cannot_load(tmp_path, capsys, monkeypatch):
    # The live policy is deployed and working; a merged doc our own loader rejects must
    # never land on top of it.
    pol, corpus = _autotune_setup(tmp_path)
    before = pol.read_bytes()
    import terse.policy_gen as pg
    monkeypatch.setattr(pg, "merge_policy",
                        lambda existing, generated, identities=None: ({"version": 99}, []))
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
               "--apply"])
    assert rc == 1
    assert pol.read_bytes() == before                      # unchanged
    assert "refusing to write" in capsys.readouterr().err


# --- #136: a bare `terse policy autotune` resolves --policy/--corpus from the wiring ---

def _wire_one_terse_server(tmp_path, monkeypatch, pol, corpus):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"gh": {
        "command": "/usr/bin/python",
        "args": ["-m", "terse", "proxy", "--policy", str(pol),
                 "--capture-dir", str(corpus), "--", "gh-mcp"]}}}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    return cfg


def test_autotune_resolves_policy_and_corpus_from_install_wiring(tmp_path, monkeypatch, capsys):
    pol, corpus = _autotune_setup(tmp_path)
    _wire_one_terse_server(tmp_path, monkeypatch, pol, corpus)
    before = pol.read_bytes()
    rc = main(["policy", "autotune"])                       # bare — no flags at all
    out = capsys.readouterr().out
    assert rc == 0
    assert f"resolved from install-mcp wiring: --policy {pol}" in out
    assert str(corpus) in out                              # the corpus was resolved too
    assert "gh.items" in out and "nothing written" in out  # reached the real diff, wrote nothing
    assert pol.read_bytes() == before


def test_autotune_refuses_when_wiring_has_no_terse_servers(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"plain": {"command": "node", "args": ["s.js"]}}}),
                   encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    rc = main(["policy", "autotune"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "could not be resolved from the installed wiring" in err


def test_autotune_refuses_ambiguous_policies_and_lists_them(tmp_path, monkeypatch, capsys):
    pol_a = _write(tmp_path, "a.json", json.dumps({"version": 1, "policies": []}))
    pol_b = _write(tmp_path, "b.json", json.dumps({"version": 1, "policies": []}))
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "a": {"command": "/usr/bin/python",
              "args": ["-m", "terse", "proxy", "--policy", str(pol_a), "--", "a-mcp"]},
        "b": {"command": "/usr/bin/python",
              "args": ["-m", "terse", "proxy", "--policy", str(pol_b), "--", "b-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    rc = main(["policy", "autotune"])                       # ambiguous — must not guess
    err = capsys.readouterr().err
    assert rc == 2
    assert "2 distinct policies" in err
    assert str(pol_a) in err and str(pol_b) in err


def test_autotune_explicit_flags_do_not_announce_wiring(tmp_path, monkeypatch, capsys):
    # An unrelated wiring must not leak into an explicit invocation, and no resolution
    # line is printed when the operator named both paths.
    pol, corpus = _autotune_setup(tmp_path)
    _wire_one_terse_server(tmp_path, monkeypatch, _write(tmp_path, "other.json", "{}"), corpus)
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "resolved from install-mcp wiring" not in out    # explicit path is silent
    assert "nothing written" in out


def test_autotune_names_what_it_guessed_about_an_old_corpus(tmp_path, capsys):
    # A pre-#148 corpus can't be made exact retroactively, so the two inputs that were
    # inferred rather than read are stated. Silently presenting a guessed grouping as a
    # measurement is the failure this exists to prevent.
    from terse.cli import _print_corpus_identity_note

    _print_corpus_identity_note([{"tool": "structure", "raw": "{}", "captured_at": 1}])
    out = capsys.readouterr().out
    assert "1/1 payload(s) predate result ids" in out
    assert "1/1 payload(s) record no server" in out


def test_an_untimed_payload_is_not_reported_as_grouped_by_timing(capsys):
    # It became its own single-block group; the timing heuristic never ran on it. Saying
    # otherwise is a false alarm about the one number the note exists to qualify.
    from terse.cli import _print_corpus_identity_note

    _print_corpus_identity_note([{"tool": "structure", "raw": "{}", "server": "runecho"}])
    assert "predate result ids" not in capsys.readouterr().out


def test_no_note_on_a_fully_identified_corpus(capsys):
    from terse.cli import _print_corpus_identity_note

    _print_corpus_identity_note([{"tool": "structure", "raw": "{}",
                                  "server": "runecho", "result_id": "s:1"}])
    assert capsys.readouterr().out == ""


def test_autotune_refuses_ambiguous_corpora_even_when_policy_agrees(tmp_path, monkeypatch, capsys):
    # #136 review Finding 2: a single agreed policy but TWO distinct wired corpora must
    # refuse, not silently fall back to the relative DEFAULT_CORPUS.
    pol = _write(tmp_path, "p.json", json.dumps({"version": 1, "policies": []}))
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "a": {"command": "/usr/bin/python",
              "args": ["-m", "terse", "proxy", "--policy", str(pol),
                       "--capture-dir", str(tmp_path / "ca"), "--", "a-mcp"]},
        "b": {"command": "/usr/bin/python",
              "args": ["-m", "terse", "proxy", "--policy", str(pol),
                       "--capture-dir", str(tmp_path / "cb"), "--", "b-mcp"]},
    }}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    rc = main(["policy", "autotune"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "2 distinct capture dirs" in err
    assert "the installed wiring is ambiguous" in err


def test_autotune_explicit_policy_resolves_only_corpus_and_announces_only_it(tmp_path, monkeypatch, capsys):
    # An explicit --policy with --corpus omitted resolves the corpus from wiring, and the
    # announce line names ONLY the corpus — the operator's explicit policy is not
    # misattributed as "resolved from wiring" (#136 review minor obs).
    pol, corpus = _autotune_setup(tmp_path)
    _wire_one_terse_server(tmp_path, monkeypatch, pol, corpus)
    explicit = _write(tmp_path, "explicit.json", json.dumps({"version": 1, "policies": []}))
    rc = main(["policy", "autotune", "--policy", str(explicit)])
    out = capsys.readouterr().out
    assert rc == 0
    line = next(ln for ln in out.splitlines() if "resolved from install-mcp wiring" in ln)
    assert f"--corpus {corpus}" in line and "--policy" not in line


# --- #136: autotune reads the LEDGER too — the corpus cannot show a capture:false tool ---

def _ledger(tmp_path, rows):
    p = tmp_path / "stats.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _rec(server, tool, raw_tokens, out_tokens):
    return {"ts": 1, "server": server, "tool": tool, "decision": "passthrough",
            "raw_chars": raw_tokens * 4, "out_chars": out_tokens * 4,
            "raw_tokens": raw_tokens, "out_tokens": out_tokens}


def test_autotune_names_live_tools_the_corpus_never_saw(tmp_path, capsys):
    """#143's exact shape: a tool sitting at 0.0% on six figures of traffic, invisible to
    every corpus because `capture: false` keeps its payloads off disk. A diff that omits it
    reads as "the install is tuned now" while a fifth of the tokens were never an input."""
    pol, corpus = _autotune_setup(tmp_path)
    led = _ledger(tmp_path, [_rec("kb", "read.search", 122431, 122431)])
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
               "--ledger", str(led)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not tunable from this corpus" in out
    assert "kb.read.search" in out
    assert "122,431 tok" in out
    assert "cannot tune what it cannot see" in out


def test_autotune_does_not_flag_a_tool_the_corpus_does_cover(tmp_path, capsys):
    pol, corpus = _autotune_setup(tmp_path)
    led = _ledger(tmp_path, [_rec("gh", "gh.items", 900, 400)])
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    assert "not tunable from this corpus" not in capsys.readouterr().out


def test_autotune_blind_spot_survives_bare_vs_qualified_drift(tmp_path, capsys):
    """The corpus records `gh.items` with no server (pre-#152 capture); the ledger records
    server `gh` + tool `items`. Same tool, two spellings. Reporting it uncovered would be a
    false alarm that trains the operator to ignore the section."""
    pol, corpus = _autotune_setup(tmp_path)
    led = _ledger(tmp_path, [_rec("gh", "items", 900, 400)])
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    out = capsys.readouterr().out
    assert "not tunable from this corpus" not in out
    assert "! items" not in out


def test_autotune_without_any_ledger_says_nothing_about_one(tmp_path, capsys):
    """The ledger is an enrichment, never a precondition: a machine that has never written
    one must get byte-identical output. (conftest points XDG_STATE_HOME at a temp dir, so
    the default path genuinely does not exist here.)"""
    pol, corpus = _autotune_setup(tmp_path)
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus)])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "ledger" not in out and "ledger" not in err
    assert "gh.items" in out                                # the diff itself still ran


def test_autotune_refuses_a_named_ledger_that_does_not_exist(tmp_path, capsys):
    """Silently proceeding would misreport coverage on the one run where the operator
    explicitly asked for it to be checked."""
    pol, corpus = _autotune_setup(tmp_path)
    before = pol.read_bytes()
    rc = main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
               "--ledger", str(tmp_path / "absent.jsonl"), "--apply"])
    assert rc == 2
    assert pol.read_bytes() == before
    assert "no ledger at" in capsys.readouterr().err


def test_ledger_traffic_does_not_double_count_a_bare_named_tool():
    """A row whose qualified and bare names collide contributes once per name, and a
    lookup reads one — a row must never be summed into itself."""
    from terse.cli import _ledger_traffic

    idx = _ledger_traffic([{"server": "kb", "tool": "kb.search", "blocks": 3,
                            "raw_tokens": 100, "out_tokens": 40}])
    assert idx["kb.search"] == {"blocks": 3, "raw_tokens": 100, "out_tokens": 40}


def test_autotune_downgrade_warning_carries_live_block_counts(tmp_path, capsys, monkeypatch):
    """The frequency the corpus structurally lacks: it is idempotent by sha, so a tool
    called twice and one called 800 times weigh the same in the decision being second-
    guessed. The ledger has the denominator; print it rather than say "go look"."""
    import terse.policy_gen as pg

    pol, corpus = _autotune_setup(tmp_path)
    real = pg.merge_policy
    monkeypatch.setattr(pg, "merge_policy", lambda existing, generated, identities=None: (
        real(existing, generated, identities)[0],
        [{"kind": "tiers", "tool": "gh.items", "before": ["minify", "tabularize"],
          "after": ["minify"], "preserved": []}]))
    # Block counts come from the ledger RECORDS, one per emitted result block — so three
    # records is three live blocks, not a field the caller gets to assert into place.
    led = _ledger(tmp_path, [_rec("gh", "gh.items", 900, 400)] * 3)
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    out = capsys.readouterr().out
    assert "would LOSE a tier" in out
    assert "3 live block(s)" in out
    assert "1,500 tok saved today" in out          # 3 x (900 - 400)


# --- #136: install-mcp points at autotune once a corpus actually has payloads ---

def _install_with_corpus(tmp_path, monkeypatch, payloads: int):
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"demo": {"command": "uvx",
                                                       "args": ["demo-mcp"]}}}),
                   encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG", str(cfg))
    policy = _write(tmp_path, "policy.json", json.dumps({"version": 1, "policies": []}))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(payloads):
        (corpus / f"demo__{i}.json").write_text("{}", encoding="utf-8")
    rc = main(["install-mcp", "demo", "--policy", str(policy),
               "--capture-dir", str(corpus)])
    return rc, corpus


def test_install_mcp_suggests_autotune_when_the_corpus_already_has_payloads(
        tmp_path, monkeypatch, capsys):
    rc, corpus = _install_with_corpus(tmp_path, monkeypatch, payloads=4)
    out = capsys.readouterr().out
    assert rc == 0
    assert "holds 4 payload(s)" in out
    assert "terse policy autotune" in out


def test_install_mcp_stays_quiet_on_a_first_install(tmp_path, monkeypatch, capsys):
    """At first install the corpus is empty by definition, and shape decisions need
    payloads — suggesting autotune there would send the operator to a command that can
    only refuse."""
    rc, _ = _install_with_corpus(tmp_path, monkeypatch, payloads=0)
    out = capsys.readouterr().out
    assert rc == 0
    assert "autotune" not in out


def test_two_servers_sharing_a_bare_tool_name_are_not_one_tool(tmp_path, capsys):
    """The bare-name fallback must not collapse `docs.search` and `other.search` into one
    tool: capturing either would then mark the other covered, which is #143 with an extra
    step. Both are ATTRIBUTED here, so neither side is missing the information."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(4):
        (corpus / f"search__{i}.json").write_text(json.dumps({
            "tool": "search", "server": "docs", "captured_at": 1_000_000_000 + i,
            "raw": json.dumps({"result": [{"id": j, "hit": "x"} for j in range(20)]}),
        }), encoding="utf-8")
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"version": 1, "policies": []}), encoding="utf-8")
    led = _ledger(tmp_path, [_rec("other", "search", 100000, 100000)])
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    out = capsys.readouterr().out
    assert "not tunable from this corpus" in out
    assert "other.search" in out


def test_unattributed_ledger_row_matches_a_captured_bare_tool(tmp_path, capsys):
    """The ledger predates `server`, so it cannot say whose `gh.items` this is. Raising a
    blind spot the operator cannot act on trains them to ignore the section."""
    pol, corpus = _autotune_setup(tmp_path)
    led = _ledger(tmp_path, [_rec("unknown", "gh.items", 900, 400)])
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    assert "not tunable from this corpus" not in capsys.readouterr().out


def test_downgrade_tail_does_not_claim_counts_it_did_not_print(tmp_path, capsys, monkeypatch):
    """A ledger full of OTHER tools' traffic must not make the warning say "live block
    counts are shown above" beside a rule that carries no annotation."""
    import terse.policy_gen as pg

    pol, corpus = _autotune_setup(tmp_path)
    real = pg.merge_policy
    monkeypatch.setattr(pg, "merge_policy", lambda existing, generated, identities=None: (
        real(existing, generated, identities)[0],
        [{"kind": "tiers", "tool": "gh.items", "before": ["minify", "tabularize"],
          "after": ["minify"], "preserved": []}]))
    led = _ledger(tmp_path, [_rec("other", "unrelated", 900, 400)])
    main(["policy", "autotune", "--policy", str(pol), "--corpus", str(corpus),
          "--ledger", str(led)])
    out = capsys.readouterr().out
    assert "would LOSE a tier" in out
    assert "shown above" not in out
    assert "cross-check those tools in `terse stats`" in out
