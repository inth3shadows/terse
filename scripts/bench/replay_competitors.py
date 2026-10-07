#!/usr/bin/env python3
"""Price other compressors on the same real tool results `replay_sessions.py` prices.

For every result the replay joins and prices, this rebuilds the raw payload from the
transcript, runs each encoder over it, and weights the tokens each one removes by the same
calls at the same rates. The output is each encoder's saving as a share of what those
results would have cost raw, beside terse's.

Rebuilding the raw payload
--------------------------
The ledger holds sizes, not payloads. The transcript holds what the model was shown:
  untouched by terse   the shown text IS the raw result.
  changed by terse     the shown text is terse's lossless form; `transforms.decompress`
                       returns the original value, and its minified JSON is the raw result
                       as the client would have shown it.
A primer that rode the result, as a typed wrapper or as leading text, is taken off first.
A rebuilt payload is used only if its cl100k size matches the raw size the ledger recorded
for that result (VERIFY_SHARE). Results that fail (a lossy drop, a diff, a raw form that
was not minified JSON) are left out of every encoder's row, terse's included. What terse
saved on them needs no payload, so it is reported per reason: the compared set must not
be read as all of terse.

Encoders
--------
  terse (as shipped)   the ledger's raw size minus what the model was actually shown,
                       primer included: the replay's own delta.
  terse (codec only)   today's `transforms.compress` on the payload, no primer.
  TOON                 the pinned `@toon-format/toon`; a payload it cannot round-trip, or
                       that is not JSON, passes through unchanged.
  headroom default     `headroom.compress` as shipped (base install: no `[ml]` extra, so
                       its learned text compressor never runs). It may drop rows or
                       strings behind a retrieve marker; the cost of fetching them back is
                       NOT charged, so this row is an upper bound. Its saving is also split
                       into results that carry a removal marker and results that do not.
  headroom agent       `mode="agent"`, headroom's own lossless regime.
  headroom verified    `verify_lossless=True`: whatever headroom cannot prove round-trips
                       is put back.
No format explainer is charged to TOON or headroom. Encoder output is counted in cl100k
like everything else, and no encoder's output is clamped to the raw size.

Payloads stay in memory and in the pipes to the two encoder processes; only sizes are
written. headroom runs under its own interpreter with the network unshared, stateless,
and with its workspace pointed at a temporary directory that is deleted afterwards.

Rows go to `<out>/competitors.jsonl` as each result is priced, then `<out>/summary.json`
and a text table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import replay_sessions as rs  # noqa: E402 -- after the sys.path setup

from terse import transforms  # noqa: E402
from terse.tokenize import count_cl100k  # noqa: E402

# A rebuilt payload must be within this share of the raw size the ledger recorded (and
# VERIFY_TOKENS covers a tiny result, where one token is a large share).
VERIFY_SHARE = 0.005
VERIFY_TOKENS = 3
PRIMER_KEY, PAYLOAD_KEY = "__terse_primer__", "__terse_payload__"
DIFF_KEYS = ("__terse_diff__", "__terse_textdiff__")
ENCODERS = ("terse_shipped", "terse_codec", "toon", "headroom_default", "headroom_agent",
            "headroom_verified")


def split_primer(text: str) -> tuple[object, bool]:
    """(parsed JSON value, found) for a shown text that is JSON, or a text primer followed
    by JSON to the end of the text."""
    try:
        return json.loads(text), True
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if not text[end:].strip():
            # Only a terse form can follow a primer; anything else is prose with a
            # JSON tail, which is the raw result itself.
            return (value, True) if transforms.has_terse_marker(value) else (None, False)
    return None, False


def rebuild(text: str) -> tuple[str | None, object]:
    """(raw text as the client would have shown it, parsed value or None) for one shown
    result, or (None, None) when it cannot be rebuilt. Decided by content, not by the
    ledger's `decision`, which describes the text block alone: a result can carry a
    compressed typed field beside an untouched text block."""
    parsed, found = split_primer(text)
    if not found:
        return text, None
    wrapped = isinstance(parsed, dict) and PRIMER_KEY in parsed and PAYLOAD_KEY in parsed
    if wrapped:
        parsed = parsed[PAYLOAD_KEY]
    if isinstance(parsed, dict) and any(key in parsed for key in DIFF_KEYS):
        return None, None
    if not wrapped and not transforms.has_terse_marker(parsed):
        return text, parsed
    try:
        obj = transforms.decompress(transforms.minify(parsed))
    except Exception:
        return None, None
    # A marker that survives decoding is a drop or a diff: not the original value.
    if transforms.has_terse_marker(obj):
        return None, None
    return transforms.minify(obj), obj


def verified(raw_tokens: int, ledger_raw: int) -> bool:
    return abs(raw_tokens - ledger_raw) <= max(VERIFY_TOKENS, VERIFY_SHARE * ledger_raw)


def collect_texts(paths: list[Path], wanted: set[str]) -> dict[str, str]:
    """The shown text of each wanted tool result, first sighting."""
    out: dict[str, str] = {}
    for path in paths:
        with path.open() as fh:
            for line in fh:
                if "tool_result" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("message")
                content = msg.get("content") if isinstance(msg, dict) else None
                if rec.get("type") != "user" or not isinstance(content, list):
                    continue
                for block in content:
                    if (isinstance(block, dict) and block.get("type") == "tool_result"
                            and block.get("tool_use_id") in wanted
                            and block["tool_use_id"] not in out):
                        out[block["tool_use_id"]] = rs._result_text(block.get("content"))
    return out


def weigh(scanned: list[tuple[Path, list[tuple]]], joined: dict,
          ratio: dict) -> dict[str, dict]:
    """Per joined, priced result: its weight (billed-tokenizer scale x carry weight,
    summed over every transcript that carried it) and the ledger's raw size. The same
    positions, stops and scale as `rs.replay`."""
    out: dict[str, dict] = {}
    for _, events in scanned:
        calls, marks, compacts = [], [], []
        for ev in events:
            if ev[0] == "call":
                calls.append(ev[1])
            elif ev[0] == "result":
                marks.append((ev[1], len(calls)))
            else:
                compacts.append(len(calls))
        for res, start in marks:
            led = joined.get(res.tool_use_id)
            if led is None or rs.norm_tool(res.name) == rs.RETRIEVE_SUFFIX:
                continue
            status, delta, _ = rs.classify(res, led)
            if status != "priced":
                continue
            later = [c for c in compacts if c >= start]
            per_w, _, _ = rs.carry(calls, start, later[0] if later else len(calls))
            model = calls[start].model if start < len(calls) else None
            slot = out.setdefault(res.tool_use_id, {
                "tool": rs.norm_tool(res.name), "weight": 0.0, "changed": led.changed,
                "ledger_raw": delta + res.tokens, "shown_tokens": res.tokens})
            slot["weight"] += ratio.get(model, ratio.get(None, 1.0)) * per_w
    return out


def run_toon(payloads: dict[str, str]) -> dict[str, dict]:
    """sha -> {"toon": text, "lossless": bool} for JSON payloads."""
    if not payloads:
        return {}
    feed = "".join(json.dumps({"id": k, "json": v}) + "\n" for k, v in payloads.items())
    proc = subprocess.run(["node", str(HERE / "toon_encode_batch.mjs")], input=feed,
                          capture_output=True, text=True, check=True, cwd=HERE)
    # split("\n"), not splitlines(): JSON.stringify leaves U+2028 and U+0085 unescaped.
    return {r["id"]: r for r in map(json.loads, filter(None, proc.stdout.split("\n")))}


def run_headroom(payloads: dict[str, str], python: Path) -> dict[str, dict]:
    """sha -> headroom_encode.py's record. No network and no state: the payloads are real
    tool results, and headroom's retrieve cache would otherwise keep their originals in a
    SQLite file."""
    if not payloads:
        return {}
    feed = "".join(json.dumps({"id": k, "text": v}) + "\n" for k, v in payloads.items())
    with tempfile.TemporaryDirectory() as work:
        env = {**os.environ, "HEADROOM_STATELESS": "1", "HEADROOM_WORKSPACE_DIR": work}
        proc = subprocess.run(
            ["unshare", "-rn", str(python), str(HERE / "headroom_encode.py")],
            input=feed, capture_output=True, text=True, check=True, env=env)
    out = {}
    for line in proc.stdout.split("\n"):
        if line.startswith("{"):
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def encoder_tokens(raw_tokens: int, obj, ledger_raw: int, shown_tokens: int,
                   toon: dict | None, hr: dict | None) -> tuple[dict[str, int], dict]:
    """Each encoder's output size for one payload, and flags about how it got there.

    Every size is read against `raw_tokens`, the rebuilt payload's own size, except
    terse's shipped one: that is the replay's delta (ledger raw minus shown), restated on
    the rebuilt basis, so a rebuild a few tokens off the ledger cannot erase a small
    saving."""
    flags = {"toon_passthrough": False, "headroom_error": False, "headroom_dropped": False}
    try:
        codec = count_cl100k(transforms.compress(obj)) if obj is not None else raw_tokens
    except Exception:  # one odd payload must not end the run after the encoders finished
        codec = raw_tokens
    sizes = {"terse_shipped": raw_tokens - (ledger_raw - shown_tokens), "terse_codec": codec}
    if toon and toon.get("lossless") and "toon" in toon:
        sizes["toon"] = count_cl100k(toon["toon"])
    else:
        sizes["toon"] = raw_tokens
        flags["toon_passthrough"] = True
    if hr and "error" not in hr:
        for mode in ("default", "agent", "verified"):
            sizes[f"headroom_{mode}"] = count_cl100k(hr[mode])
        flags["headroom_dropped"] = bool(hr["dropped"])
    else:
        for mode in ("default", "agent", "verified"):
            sizes[f"headroom_{mode}"] = raw_tokens
        flags["headroom_error"] = True
    return sizes, flags


def summarize(rows: list[dict], left_out: dict[str, dict], priced_weight: float) -> dict:
    def fold(group):
        base = sum(r["raw_tokens"] * r["weight"] for r in group)
        slot = {"results": len(group), "raw_weighted": round(base, 1)}
        for enc in ENCODERS:
            saved = sum((r["raw_tokens"] - r["sizes"][enc]) * r["weight"] for r in group)
            slot[enc] = {"saved_weighted": round(saved, 1),
                         "saved_pct": round(100 * saved / base, 2) if base else None}
        return slot

    by_tool = defaultdict(list)
    for r in rows:
        by_tool[r["tool"]].append(r)
    base = sum(r["raw_tokens"] * r["weight"] for r in rows)
    ledger_base = sum(r["ledger_raw"] * r["weight"] for r in rows)

    def default_split(dropped: bool):
        group = [r for r in rows if r["flags"]["headroom_dropped"] == dropped]
        saved = sum((r["raw_tokens"] - r["sizes"]["headroom_default"]) * r["weight"]
                    for r in group)
        return {"results": len(group), "saved_weighted": round(saved, 1),
                "saved_pct_of_all": round(100 * saved / base, 2) if base else None}

    return {
        "all": fold(rows),
        "by_tool": {t: fold(g) for t, g in by_tool.items()},
        "coverage_pct": round(100 * ledger_base / priced_weight, 2) if priced_weight else None,
        "priced_raw_weighted": round(priced_weight, 1),
        # How far the rebuilt sizes sit from the ledger's, by weight. Near zero or the
        # compared payloads are not the ones the ledger measured.
        "rebuild_residual_pct": round(100 * (base - ledger_base) / ledger_base, 3)
        if ledger_base else None,
        # Left out of every row. terse's saving on them is the replay's delta and needs
        # no payload; no other encoder was run on them.
        "left_out": {k: {"results": v["results"], "raw_weighted": round(v["raw"], 1),
                         "terse_saved_weighted": round(v["saved"], 1),
                         "terse_saved_pct": round(100 * v["saved"] / v["raw"], 2)
                         if v["raw"] else None,
                         "top_tools": dict(sorted(((t, round(c, 1)) for t, c in
                                                   v["tools"].items()),
                                                  key=lambda kv: -kv[1])[:4])}
                     for k, v in left_out.items()},
        "toon_passthrough": sum(r["flags"]["toon_passthrough"] for r in rows),
        "headroom_error": sum(r["flags"]["headroom_error"] for r in rows),
        "headroom_default_with_marker": default_split(True),
        "headroom_default_without_marker": default_split(False),
    }


def render(summary: dict) -> str:
    lines = [f"compared on {summary['all']['results']} results holding "
             f"{summary['coverage_pct']}% of the raw-path cost of all joined, priced results "
             f"({summary['priced_raw_weighted']:,.0f}); rebuilt sizes sit "
             f"{summary['rebuild_residual_pct']}% from the ledger's by weight"]
    for reason, slot in summary["left_out"].items():
        lines.append(f"left out, {reason}: {slot['results']} results, raw path "
                     f"{slot['raw_weighted']:,.0f}, terse saved {slot['terse_saved_pct']}% "
                     f"there; mostly {slot['top_tools']}")
    w, wo = summary["headroom_default_with_marker"], summary["headroom_default_without_marker"]
    lines += [f"TOON passed {summary['toon_passthrough']} through unchanged (not JSON, or no "
              f"round-trip); headroom errors: {summary['headroom_error']}",
              f"headroom default, {w['results']} results carrying a removal marker: "
              f"{w['saved_pct_of_all']} points of its saving; {wo['results']} without one: "
              f"{wo['saved_pct_of_all']} points",
              "",
              f"{'saved, % of raw-path cost':<28}{'results':>8}{'raw path':>13}"
              + "".join(f"{e:>18}" for e in ENCODERS)]

    def line(name, slot):
        lines.append(f"{name[:27]:<28}{slot['results']:>8}{slot['raw_weighted']:>13,.0f}"
                     + "".join(f"{str(slot[e]['saved_pct']) + '%':>18}" for e in ENCODERS))

    line("ALL COMPARED", summary["all"])
    for tool, slot in sorted(summary["by_tool"].items(),
                             key=lambda kv: -kv[1]["raw_weighted"]):
        line(tool, slot)
    lines.append("")
    lines.append("headroom default is an upper bound: fetch-backs of what it dropped are not "
                 "charged, and this is its base install (no learned text compressor). No "
                 "explainer is charged to TOON or headroom; terse (as shipped) carries its "
                 "primer. The ALL COMPARED row is not terse's overall figure: see left out.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", type=Path,
                    default=Path.home() / ".local/state/terse/stats.jsonl")
    ap.add_argument("--transcripts", type=Path, default=Path.home() / ".claude/projects")
    ap.add_argument("--headroom-python", type=Path, required=True,
                    help="interpreter of an environment with headroom-ai installed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    ledger = rs.load_ledger(args.ledger)
    paths = rs.find_transcripts(args.transcripts, ledger[0].ts - 86400)
    paths.sort(key=lambda p: p.stat().st_mtime)
    billed: set[tuple] = set()
    scanned = [(p, rs.scan_transcript(p, billed)) for p in paths]
    results = [ev[1] for _, events in scanned for ev in events
               if ev[0] == "result" and ev[1].ts >= ledger[0].ts - rs.JOIN_BEFORE]
    joined, _ = rs.join(ledger, results)
    calib: dict[str, list[float]] = defaultdict(list)
    for _, events in scanned:
        for model, sample in rs.calibration_samples(events):
            calib[model].append(sample)
    items = weigh(scanned, joined, rs.tokenizer_ratios(calib))
    holders = [p for p, events in scanned
               if any(ev[0] == "result" and ev[1].tool_use_id in items for ev in events)]
    texts = collect_texts(holders, set(items))
    print(f"{len(items)} priced results, {len(texts)} shown texts found", file=sys.stderr)

    priced_weight = sum(it["ledger_raw"] * it["weight"] for it in items.values())
    left_out: dict[str, dict] = defaultdict(
        lambda: {"results": 0, "raw": 0.0, "saved": 0.0, "tools": defaultdict(float)})

    def leave_out(reason: str, it: dict) -> None:
        slot = left_out[reason]
        slot["results"] += 1
        slot["raw"] += it["ledger_raw"] * it["weight"]
        slot["saved"] += (it["ledger_raw"] - it["shown_tokens"]) * it["weight"]
        slot["tools"][it["tool"]] += it["ledger_raw"] * it["weight"]

    ready: dict[str, dict] = {}
    payload_text: dict[str, str] = {}
    payload_json: dict[str, str] = {}
    for tid, it in items.items():
        if tid not in texts:
            leave_out("no_text", it)
            continue
        raw_text, obj = rebuild(texts[tid])
        if raw_text is None:
            leave_out("not_rebuilt", it)
            continue
        raw_tokens = count_cl100k(raw_text)
        if not verified(raw_tokens, it["ledger_raw"]):
            leave_out("size_differs", it)
            continue
        sha = hashlib.sha256(raw_text.encode()).hexdigest()
        payload_text[sha] = raw_text
        if obj is not None:
            payload_json[sha] = raw_text
        ready[tid] = dict(it, sha=sha, raw_tokens=raw_tokens, obj=obj)
    print(f"{len(ready)} rebuilt and verified, {len(payload_text)} distinct payloads",
          file=sys.stderr)

    toon = run_toon(payload_json)
    hr = run_headroom(payload_text, args.headroom_python)
    print(f"toon {len(toon)} of {len(payload_json)}, headroom {len(hr)} of "
          f"{len(payload_text)}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "competitors.jsonl").open("w") as fh:
        for it in ready.values():
            sizes, flags = encoder_tokens(it["raw_tokens"], it["obj"], it["ledger_raw"],
                                          it["shown_tokens"], toon.get(it["sha"]),
                                          hr.get(it["sha"]))
            row = {"tool": it["tool"], "weight": round(it["weight"], 4),
                   "raw_tokens": it["raw_tokens"], "ledger_raw": it["ledger_raw"],
                   "sizes": sizes, "flags": flags}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    summary = summarize(rows, left_out, priced_weight)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render(summary)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
