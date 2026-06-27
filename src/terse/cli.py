"""terse CLI — Phase-0 spike entrypoint.

Subcommands:
  gate <file|->            run the lossless round-trip gate on a JSON payload
  capture --tool N <file|-> persist a tool output to corpus/ + bucket by shape
  measure [--anthropic]    token delta per tier per shape bucket over the corpus
  probe                    value-redundancy + cross-call-overlap          (TODO)
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
