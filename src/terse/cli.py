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

from . import transforms
from .capture import capture_payload, classify_shape, coverage, load_corpus
from .measure import measure_corpus
from .report import build_report
from .tokenize import count_cl100k

DEFAULT_CORPUS = "corpus"
DEFAULT_REPORT = "reports/spike-report.md"


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


def _todo(name: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"`terse {name}` not implemented yet — see the plan / module TODOs.")
        return 2
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terse", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="run the lossless round-trip gate on a JSON file")
    g.add_argument("file", help="path to a JSON payload, or - for stdin")
    g.set_defaults(func=_cmd_gate)

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

    p = sub.add_parser("probe", help="value-redundancy + cross-call-overlap (TODO)")
    p.set_defaults(func=_todo("probe"))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
