"""terse CLI — Phase-0 spike entrypoint.

Subcommands:
  gate <file|->            run the lossless round-trip gate on a JSON payload
  capture --tool N <file|-> persist a tool output to corpus/ + bucket by shape
  measure [--anthropic]    token delta per tier per shape bucket over the corpus
  probe                    value-redundancy + cross-call-overlap ceiling probes
  validate                 cross-tokenizer invariance (cl100k vs o200k)
  compress --tool N        compress one tool output through a policy (the shell)
  proxy -- <cmd>           MCP stdio proxy: compress a downstream server's results
  fluency                  does a model read the compressed form as well as raw JSON?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import json as _json

from . import transforms
from .capture import capture_payload, classify_shape, coverage, extract_records, load_corpus
from .measure import cross_tokenizer_savings, measure_corpus
from .probes import cross_call_overlap, value_redundancy
from .report import build_probe_report, build_report, build_tokenizer_report
from .tokenize import count_cl100k

DEFAULT_CORPUS = "corpus"
DEFAULT_REPORT = "reports/spike-report.md"
DEFAULT_PROBE_REPORT = "reports/probe-report.md"
DEFAULT_FLUENCY_REPORT = "reports/fluency-report.md"
DEFAULT_FLUENCY_PACK = "reports/fluency-pack.json"


def _read(file: str) -> str:
    return sys.stdin.read() if file == "-" else Path(file).read_text(encoding="utf-8")


def _cmd_gate(args: argparse.Namespace) -> int:
    raw = _read(args.file)
    obj = json.loads(raw)
    ok = transforms.roundtrip_ok(obj)
    before, after = count_cl100k(raw), count_cl100k(transforms.compress(obj))
    print(f"round-trip lossless: {'PASS' if ok else 'FAIL'}")
    print(f"shape bucket:        {classify_shape(raw)}")
    if before is not None and after is not None:
        saved = before - after
        print(f"cl100k tokens:       {before} -> {after}  ({saved:+d}, {saved / before * 100:.1f}% saved)")
    else:
        print("cl100k tokens:       (tiktoken unavailable)")
    return 0 if ok else 1


def _cmd_capture(args: argparse.Namespace) -> int:
    raw = _read(args.file)
    path = capture_payload(args.tool, raw, args.corpus)
    print(f"captured {args.tool} ({classify_shape(raw)}, {len(raw)} bytes) -> {path}")
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1
    rows = measure_corpus(envelopes, use_anthropic=args.anthropic)
    report = build_report(rows, coverage(envelopes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .policy import default_policy, load_policy
    from .proxy import run_proxy

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("proxy: provide the downstream server command after `--`, e.g.\n"
              "  terse proxy --policy p.json -- uvx some-mcp-server", file=sys.stderr)
        return 2
    pol = load_policy(args.policy) if args.policy else default_policy()
    if args.diff:
        pol.diff = True  # CLI opt-in overrides the policy default (off)
    if args.diff_keyframe_interval is not None:
        pol.diff_keyframe_interval = args.diff_keyframe_interval
    return run_proxy(cmd, pol, debug=args.debug, capture_dir=args.capture_dir,
                     debug_log=args.debug_log)


def _cmd_compress(args: argparse.Namespace) -> int:
    from .policy import apply, default_policy, load_policy

    policy = load_policy(args.policy) if args.policy else default_policy()
    raw = _read(args.file)
    result = apply(raw, args.tool, policy)
    sys.stdout.write(result.text)
    summary = "skipped (passthrough)" if result.skipped else f"tiers={list(result.tiers)}"
    before, after = count_cl100k(raw), count_cl100k(result.text)
    if before and after is not None:
        summary += f"  cl100k {before}->{after} ({(before - after) / before * 100:+.1f}%)"
    print(f"[{args.tool}] {summary}", file=sys.stderr)
    for w in result.warnings:
        print(f"[warn] {w}", file=sys.stderr)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1
    report = build_tokenizer_report(cross_tokenizer_savings(envelopes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1

    vr_rows = []
    for env in envelopes:
        try:
            records = extract_records(_json.loads(env["raw"]))
        except (ValueError, TypeError):
            records = None
        if records:
            vr_rows.append({"tool": env["tool"], "sha": env.get("sha", "?"), **value_redundancy(records)})

    # Cross-call overlap: successive payloads sharing a tool (sorted by sha for determinism).
    overlap_rows = []
    by_tool: dict[str, list[dict]] = {}
    for env in envelopes:
        by_tool.setdefault(env["tool"], []).append(env)
    for tool, envs in by_tool.items():
        envs = sorted(envs, key=lambda e: e.get("sha", ""))
        for prev, curr in zip(envs, envs[1:]):
            res = cross_call_overlap(prev["raw"], curr["raw"])
            if res.get("available"):
                overlap_rows.append({"tool": tool, "prev_sha": prev.get("sha", "?"),
                                     "curr_sha": curr.get("sha", "?"), **res})

    report = build_probe_report(vr_rows, overlap_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")
    return 0


def _build_answerers(args: argparse.Namespace) -> dict:
    """Assemble named answerers from env + flags. Empty means keyless (pack) mode."""
    import os

    from . import fluency

    answerers: dict = {}
    # Flags win over env so a credential-injecting launcher (e.g. secret_inject_env,
    # which sets the key under its own env var) can drive the CLI without a shell.
    base = args.base_url or os.environ.get("TERSE_FLUENCY_BASE_URL")
    key = os.environ.get(args.api_key_env or "TERSE_FLUENCY_API_KEY")
    models = args.models or os.environ.get("TERSE_FLUENCY_MODELS", "")
    if base and key and models:
        for m in (x.strip() for x in models.split(",") if x.strip()):
            answerers[m] = fluency.openai_answerer(base, key, m)
    if args.anthropic:
        try:
            answerers[f"anthropic:{args.anthropic_model}"] = fluency.anthropic_answerer(
                args.anthropic_model
            )
        except Exception as e:  # missing extra/key — fall back, don't crash the run
            print(f"[warn] anthropic answerer unavailable: {e}", file=sys.stderr)
    return answerers


def _cmd_fluency(args: argparse.Namespace) -> int:
    from . import fluency
    from .report import build_diff_report, build_fluency_report

    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1

    # Diff mode: does a model read a cross-call DIFF as well as the full result? Needs a
    # live model (it measures comprehension of a form, not ground-truth math).
    if args.diff:
        answerers = _build_answerers(args)
        if not answerers:
            print("`fluency --diff` needs a configured model: set TERSE_FLUENCY_BASE_URL/"
                  "_API_KEY/_MODELS or pass --anthropic.")
            return 1
        results = fluency.run_diff_fluency(envelopes, answerers, trials=args.trials)
        _write_report(build_diff_report(results), args.out)
        return 0

    # Score mode: an externally-collected responses file against a previously-written pack.
    if args.responses:
        pack = _json.loads(Path(args.pack).read_text(encoding="utf-8"))
        responses = _json.loads(Path(args.responses).read_text(encoding="utf-8"))
        results = fluency.score_pack(pack, responses)
        report = build_fluency_report(results, fluency.token_summary(envelopes))
        _write_report(report, args.out)
        return 0

    answerers = _build_answerers(args)
    if not answerers:
        # Keyless default: write the eval pack and explain how to drive it.
        pack = fluency.build_pack(envelopes, trials=args.trials)
        out = Path(args.pack)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
        nq = sum(len(p["questions"]) for p in pack["payloads"])
        print(f"no model configured — wrote {nq} questions over {len(pack['payloads'])} "
              f"record-shaped payloads to {out}.")
        print("To run a model: set TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS (broker pool) "
              "or pass --anthropic, then re-run.")
        print(f"Or drive the pack by hand and score it: `terse fluency --responses <file> "
              f"--pack {out}`.")
        return 0

    results = fluency.run_fluency(envelopes, answerers, trials=args.trials)
    report = build_fluency_report(results, fluency.token_summary(envelopes))
    _write_report(report, args.out)
    return 0


def _short_cmd(entry) -> str:
    if not entry:
        return "(absent)"
    return " ".join([entry.get("command", "?"), *entry.get("args", [])])[:100]


def _cmd_install_mcp(args: argparse.Namespace) -> int:
    from .install_mcp import do_install

    try:
        res = do_install(args.servers, args.policy, dry_run=args.print,
                         capture_dir=args.capture_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"install-mcp: {e}", file=sys.stderr)
        return 2
    tag = "[dry-run] would wrap" if res["dry_run"] else "wrapped"
    for c in res["changes"]:
        print(f"{tag} {c['server']}:")
        print(f"    before: {_short_cmd(c['before'])}")
        print(f"    after:  {_short_cmd(c['after'])}")
    print(f"config: {res['config']}  policy: {res['policy']}")
    if res.get("capture_dir"):
        print(f"capture: raw tool results → {res['capture_dir']}")
    if res["backup"]:
        print(f"backup: {res['backup']}")
    if not res["dry_run"] and res["changes"]:
        print("→ restart Claude Code for the change to take effect.")
    return 0


def _cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    from .install_mcp import do_uninstall

    res = do_uninstall(args.servers, all_=args.all, dry_run=args.print)
    tag = "[dry-run] would restore" if res["dry_run"] else "restored"
    if not res["changes"]:
        print("nothing to do (no terse-managed servers).")
        return 0
    for c in res["changes"]:
        if c.get("restored"):
            print(f"{tag} {c['server']}")
        else:
            print(f"skip {c['server']}: {c.get('reason')}")
    if res["backup"]:
        print(f"backup: {res['backup']}")
    if not res["dry_run"] and any(c.get("restored") for c in res["changes"]):
        print("→ restart Claude Code for the change to take effect.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Self-contained verification report: lossless gate + token savings + an attestation
    header pointing at the checks terse can't self-certify (tests, no-egress, fail-open).
    Runs on captured traffic (--corpus) or, with none, a bundled deterministic sample so
    it works with zero setup from a checkout."""
    import subprocess
    import tempfile

    from .capture import coverage, load_corpus
    from .measure import measure_corpus
    from .report import build_report, build_verify_header

    if args.corpus:
        envelopes = load_corpus(args.corpus)
        if not envelopes:
            print(f"verify: no payloads in {args.corpus}/ — capture some first "
                  "(`terse capture --tool <name> <payload>`).", file=sys.stderr)
            return 1
        label = f"your captured traffic (`{args.corpus}`)"
    else:
        # scripts/ ships with the repo, not the installed wheel — so the zero-setup sample
        # path works from a checkout (where an adopter also runs pytest), not pip-only.
        script = Path(__file__).resolve().parents[2] / "scripts" / "gen_stress_corpus.py"
        if not script.exists():
            print("verify: no --corpus given and the bundled sample generator isn't "
                  "available here. Run from a repo checkout, or pass --corpus <dir> with "
                  "captured output (`terse capture`).", file=sys.stderr)
            return 2
        # TemporaryDirectory so the synthetic corpus doesn't accumulate in /tmp; envelopes
        # are read fully into memory by load_corpus, so the dir can go right after.
        with tempfile.TemporaryDirectory(prefix="terse-verify-") as sample_dir:
            try:
                subprocess.run([sys.executable, str(script), sample_dir], check=True,
                               stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as exc:
                print(f"verify: the bundled sample generator failed (exit {exc.returncode}). "
                      "Pass --corpus <dir> with captured output instead.", file=sys.stderr)
                return 2
            envelopes = load_corpus(sample_dir)
        label = ("bundled deterministic sample — synthetic; capture real traffic with "
                 "`terse capture` for your own numbers")

    rows = measure_corpus(envelopes, use_anthropic=False)
    report = build_verify_header(label, len(envelopes)) + build_report(rows, coverage(envelopes))
    _write_report(report, args.out)
    return 0


def _write_report(report: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terse", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="run the lossless round-trip gate on a JSON file")
    g.add_argument("file", help="path to a JSON payload, or - for stdin")
    g.set_defaults(func=_cmd_gate)

    c2 = sub.add_parser("compress", help="compress a tool output per policy (the shell)")
    c2.add_argument("file", help="path to the raw tool output, or - for stdin")
    c2.add_argument("--tool", required=True, help="tool name to match against the policy")
    c2.add_argument("--policy", help="path to a JSON policy file (default: lossless-everywhere)")
    c2.set_defaults(func=_cmd_compress)

    c = sub.add_parser("capture", help="persist a tool output to the corpus")
    c.add_argument("file", help="path to the raw tool output, or - for stdin")
    c.add_argument("--tool", required=True, help="source tool name (for coverage tracking)")
    c.add_argument("--corpus", default=DEFAULT_CORPUS)
    c.set_defaults(func=_cmd_capture)

    m = sub.add_parser("measure", help="token delta per tier per shape bucket over the corpus")
    m.add_argument("--corpus", default=DEFAULT_CORPUS)
    m.add_argument("--out", default=DEFAULT_REPORT)
    m.add_argument("--anthropic", action="store_true", help="also count with Anthropic (network)")
    m.set_defaults(func=_cmd_measure)

    p = sub.add_parser("probe", help="value-redundancy + cross-call-overlap ceiling probes")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--out", default=DEFAULT_PROBE_REPORT)
    p.set_defaults(func=_cmd_probe)

    v = sub.add_parser("validate", help="cross-tokenizer invariance (cl100k vs o200k)")
    v.add_argument("--corpus", default=DEFAULT_CORPUS)
    v.add_argument("--out", default="reports/tokenizer-report.md")
    v.set_defaults(func=_cmd_validate)

    px = sub.add_parser("proxy", help="MCP stdio proxy: compress a downstream server's "
                                      "tool results per policy")
    px.add_argument("--policy", help="path to a JSON policy file (default: lossless-everywhere)")
    px.add_argument("--diff-keyframe-interval", type=int, default=None, metavar="K",
                    help="with --diff, force a full result every K consecutive diffs per tool "
                         "to bound dangling-reference drift (default 5; 0 disables)")
    px.add_argument("--diff", action="store_true",
                    help="enable cross-call diffing (stateful; emits a lossless delta vs the "
                         "prior same-tool result when smaller). Opt-in: fluency unverified")
    px.add_argument("--debug", action="store_true", help="log compressions to stderr")
    px.add_argument("--capture-dir", metavar="DIR",
                    help="tee each raw tool-result payload into this corpus dir for later "
                         "`terse verify --corpus`/`measure` (opt-in; never affects forwarding)")
    px.add_argument("--debug-log", metavar="FILE",
                    help="append a structured raw->decision->emitted record per result to "
                         "this JSONL file for after-the-fact diagnosis/replay (opt-in; never "
                         "affects forwarding)")
    px.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- <downstream MCP server command and args>")
    px.set_defaults(func=_cmd_proxy)

    f = sub.add_parser("fluency", help="does a model read the compressed form as "
                                       "accurately as raw JSON? (proxy's open question)")
    f.add_argument("--corpus", default=DEFAULT_CORPUS)
    f.add_argument("--out", default=DEFAULT_FLUENCY_REPORT)
    f.add_argument("--pack", default=DEFAULT_FLUENCY_PACK,
                   help="path for the offline eval pack (written when no model is configured)")
    f.add_argument("--responses", help="score a collected responses JSON against --pack")
    f.add_argument("--trials", type=int, default=1,
                   help="repeat each question N times; report mean ± a binomial bound (default 1)")
    f.add_argument("--diff", action="store_true",
                   help="eval whether a model reads a cross-call DIFF as well as the full "
                        "result (needs same-tool corpus pairs + a configured model)")
    f.add_argument("--base-url", help="OpenAI-compatible base URL (else $TERSE_FLUENCY_BASE_URL)")
    f.add_argument("--models", help="comma-separated model ids (else $TERSE_FLUENCY_MODELS)")
    f.add_argument("--api-key-env", default="TERSE_FLUENCY_API_KEY",
                   help="env var holding the API key (default TERSE_FLUENCY_API_KEY)")
    f.add_argument("--anthropic", action="store_true",
                   help="also test the real consumer (needs the anthropic extra + key)")
    f.add_argument("--anthropic-model", default="claude-opus-4-8")
    f.set_defaults(func=_cmd_fluency)

    im = sub.add_parser("install-mcp", help="wrap Claude Code MCP server(s) with the "
                                            "terse proxy in ~/.claude.json")
    im.add_argument("servers", nargs="+", help="mcpServers name(s) to wrap (e.g. runecho)")
    im.add_argument("--policy", required=True, help="path to the JSON policy file")
    im.add_argument("--capture-dir", metavar="DIR",
                    help="also tee raw tool results into this corpus dir for later "
                         "`terse measure`/`verify` (opt-in; never affects forwarding)")
    im.add_argument("--print", action="store_true",
                    help="dry-run: show the before/after without writing")
    im.set_defaults(func=_cmd_install_mcp)

    um = sub.add_parser("uninstall-mcp", help="restore terse-wrapped MCP server(s) to "
                                              "their original command")
    um.add_argument("servers", nargs="*", help="server name(s) to restore (or use --all)")
    um.add_argument("--all", action="store_true", help="restore every terse-managed server")
    um.add_argument("--print", action="store_true",
                    help="dry-run: show what would be restored without writing")
    um.set_defaults(func=_cmd_uninstall_mcp)

    vf = sub.add_parser("verify", help="self-contained verification report: lossless gate "
                                       "+ token savings, and how to verify the rest")
    vf.add_argument("--corpus", help="captured-traffic corpus dir (default: a bundled "
                                     "deterministic sample, so it runs with zero setup)")
    vf.add_argument("--out", default="reports/verify-report.md")
    vf.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
