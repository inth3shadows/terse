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
    return run_proxy(cmd, pol, debug=args.debug)


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
