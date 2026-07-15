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
    assert agg["total"]["results"] == 1 and agg["total"]["raw_tokens"] == 100


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
