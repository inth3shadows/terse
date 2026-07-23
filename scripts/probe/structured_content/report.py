#!/usr/bin/env python3
"""Turn the two capture arms into the verdict issue #128 needs.

Usage: report.py raw.jsonl terse.jsonl

Reports, per arm, what the client actually put in the model's context, then names which
of #128's four options the evidence selects. Deliberately refuses to guess when the
arms disagree in a way the harness didn't anticipate — a confidently-wrong verdict here
would be worse than no verdict, since the whole point is to stop arguing from priors.
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def describe(name: str, recs: list[dict]) -> dict:
    tokens = [r["tokens"] for r in recs if r.get("tokens") is not None]
    chars = [r["chars"] for r in recs]
    keys = sorted({k for r in recs for k in r.get("block_keys", [])})
    enveloped = sum(1 for r in recs if r.get("terse_envelope"))
    print(f"--- {name} ---")
    print(f"  tool_result blocks : {len(recs)}")
    print(f"  block keys sent    : {keys}")
    print(f"  terse envelope     : {enveloped}/{len(recs)}")
    print(f"  chars  (max)       : {max(chars) if chars else 0}")
    print(f"  tokens (max)       : {max(tokens) if tokens else 'n/a (no tiktoken)'}")
    return {"tokens": max(tokens) if tokens else None,
            "chars": max(chars) if chars else 0,
            "keys": keys, "enveloped": enveloped, "n": len(recs)}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    # Read each file EXACTLY once. Both the printed numbers and the verdict below must be
    # drawn from the same sample: a capture still being appended to would otherwise let
    # the two reads disagree, and the verdict would describe a record set the reader never
    # saw.
    raw_recs, terse_recs = load(sys.argv[1]), load(sys.argv[2])
    raw, terse = describe("raw", raw_recs), describe("terse", terse_recs)
    print()

    if not raw["n"] or not terse["n"]:
        print("VERDICT: inconclusive — an arm captured nothing. Re-run; an empty "
              "artifact is a failed measurement, not evidence.")
        return 1

    # `structuredContent` is not a valid key on an Anthropic tool_result block, so if the
    # client forwards it at all it must be serialized INTO the text. The token delta is
    # therefore the only reliable discriminator, not the key list.
    metric = "tokens" if raw["tokens"] and terse["tokens"] else "chars"
    before, after = raw[metric], terse[metric]
    saved = (before - after) / before * 100 if before else 0.0
    print(f"Context cost of the tool result ({metric}): {before} -> {after} "
          f"({saved:+.1f}% for terse)")
    print()

    # The sharpest discriminator needs no fixture knowledge at all: if the two arms put
    # BYTE-IDENTICAL text in the model's context, then interposing terse changed nothing
    # the model sees, and the client cannot be reading the text block terse compressed.
    raw_texts = sorted(r["verbatim"] for r in raw_recs if r["chars"])
    terse_texts = sorted(r["verbatim"] for r in terse_recs if r["chars"])
    if raw_texts and raw_texts == terse_texts:
        print("VERDICT: the client IGNORES the text block when `structuredContent` is")
        print("  present. Both arms put byte-identical text in the model's context, so")
        print("  terse compressing the text block bought exactly zero tokens — confirm")
        print("  against the proxy's --debug-log that it did compress, and if so the")
        print("  compressed block was discarded by the client, not by terse.")
        print("  -> This is WORSE than #128's framing (a halved saving): on such tools")
        print("     the saving the MODEL sees is ZERO. The ledger now counts the")
        print("     untouched duplicate, so it reports the wire truth rather than the")
        print("     text block's reduction — but even that is an upper bound for a")
        print("     client which reads the typed field.")
        print("     Options 2/3/4 all leave the real saving at zero; only option 1")
        print("     (compress `structuredContent` itself) can move it, at the cost of")
        print("     the typed field clients validate against `outputSchema`.")
        return 0

    # A terse envelope IN THE MODEL'S CONTEXT settles it without any threshold reasoning:
    # whatever field the client reads, terse reached it. The percentage above is then the
    # real saving, not a proxy for one. This is what `"structured": "compress"` (#128)
    # produces, and it must be checked BEFORE the threshold branches below — those were
    # calibrated for the default build, where the envelope never arrives, and would
    # misread a genuine 61% as the "duplicate halved it" signature.
    if terse["enveloped"]:
        print("VERDICT: terse's output REACHES the model — a terse envelope is in the")
        print(f"  context, so the {saved:.1f}% above is the real saving, not an estimate.")
        print("  If this run had `structured: compress` set, that is the #128 fix")
        print("  working: the codec is now landing on the field the client actually")
        print("  reads. Confirm the policy in use before quoting the number.")
        return 0

    print("VERDICT: terse did not compress this result at all (no envelope in the "
          "model's context), and the arms differ. Fix the wiring before reading "
          "anything into the numbers above.")
    return 1

    # The measured saving on the text block alone was 70.5%; with an untouched duplicate
    # riding along it was 56.2% (issue #128's table). Those are far enough apart to
    # discriminate, but the boundary is a judgement call, so print the reasoning too.
    if saved >= 65:
        print("VERDICT: the client forwards ONLY the text block. The untouched")
        print("  `structuredContent` duplicate never reaches the model, so it costs zero")
        print("  context and terse's real saving is the full text-block figure.")
        print("  -> #128 option 4 (document the shape divergence, change no behavior).")
    elif saved >= 40:
        print("VERDICT: the saving is roughly HALVED, which is the signature of the")
        print("  duplicate riding along uncompressed. The 396-token overhead in #128 is")
        print("  real and recurring.")
        print("  -> #128 option 2 (drop the redundant text mirror) is the only option")
        print("     that recovers it; weigh it against the backwards-compat risk.")
    else:
        print("VERDICT: terse's saving barely survives into the model's context. That")
        print("  points at the client reading `structuredContent` as the primary field")
        print("  and the compressed text block as near-dead weight — a different problem")
        print("  from the one #128 describes. Re-read the verbatim blocks before deciding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
