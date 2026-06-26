"""terse CLI — Phase-0 spike entrypoint.

Subcommands:
  gate <file.json>   run the lossless round-trip gate on a JSON payload (works now)
  capture            persist tool outputs to corpus/ + bucket by shape   (TODO)
  measure            token delta per tier per shape bucket               (TODO)
  probe              value-redundancy + cross-call-overlap               (TODO)
"""

from __future__ import annotations

import argparse
import json
import sys

from . import transforms
from .capture import classify_shape
from .tokenize import count_cl100k


def _cmd_gate(args: argparse.Namespace) -> int:
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    obj = json.loads(raw)
    ok = transforms.roundtrip_ok(obj)
    before = count_cl100k(raw)
    after = count_cl100k(transforms.compress(obj))
    shape = classify_shape(raw)
    print(f"round-trip lossless: {'PASS' if ok else 'FAIL'}")
    print(f"shape bucket:        {shape}")
    if before is not None and after is not None:
        saved = before - after
        pct = (saved / before * 100) if before else 0.0
        print(f"cl100k tokens:       {before} -> {after}  ({saved:+d}, {pct:.1f}% saved)")
    else:
        print("cl100k tokens:       (tiktoken unavailable)")
    return 0 if ok else 1


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

    for name in ("capture", "measure", "probe"):
        s = sub.add_parser(name, help=f"{name} (TODO)")
        s.set_defaults(func=_todo(name))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
