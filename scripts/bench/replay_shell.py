#!/usr/bin/env python3
"""What shell output and file reads cost in real sessions, and what RTK would do to it.

terse sits in front of MCP servers. Most tool output in a session is not MCP: it is Bash
and Read. This prices that output on the same calls at the same rates as
`replay_sessions.py`, then replays RTK (a shell-output filter) over the recorded Bash
results, without executing anything from a transcript:

  not_rewritten  `rtk rewrite <command>` printed nothing: RTK's hook never sees it.
  unchanged      `cat`/`head`/`tail`, which RTK turns into `rtk read`. At its default level
                 that returns the content byte for byte (checked on 0.51.0): no saving.
  measured       a plain `pytest` run (an optional leading `cd <dir> &&` and a trailing
                 `2>&1` aside). `rtk pipe -f pytest` over the recorded output is within
                 about 30% of what the live `rtk pytest` prints.
  unmeasured     every other rewritten command. Its saving CANNOT be replayed: for
                 grep, find, git log, git diff, git status and ruff, `rtk pipe -f` over
                 recorded output differs from the live `rtk <cmd>` by large amounts in
                 both directions (the live command changes flags, caps and formats; pipe
                 mode passes single-file grep through and empties `git diff --stat`).

So this reports RTK's REACH, not its saving: the share of Bash cost it would take, how
much of that it hands back unchanged, and a ceiling, the saving if it removed every token
of every other result it takes. A real saving needs a live trial.

Limits. What RTK's filters elide can be fetched back with `rtk recall`; that cost is not
charged anywhere here. The model might also have issued different commands. Sizes are
cl100k, scaled per model by the replay's billed-per-cl100k ratio.

Only sizes, command words and counts are written. Commands and output stay in memory and
in the pipe to `rtk_encode.py`, which runs with the network unshared and a temporary home.

Rows go to `<out>/shell.jsonl`, one per result, then `<out>/summary.json` and a table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import replay_sessions as rs  # noqa: E402 -- after the sys.path setup

TOOLS = ("Bash", "Read")
# RTK subcommand -> its stored-output filter (`rtk pipe -f`), only where pipe mode was
# checked against the live command on a fixture. "read" is not a filter: `rtk read` at the
# default level is byte-exact.
FILTERS = {("pytest",): "pytest"}
CD_PREFIX = re.compile(r"""^cd\s+(?:"[^"]*"|'[^']*'|[^\s;&|]+)\s*&&\s*""")
STDERR_SUFFIX = re.compile(r"\s+2>(?:&1|/dev/null)\s*$")
WORD = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]{0,19}$")
NOT_PLAIN = re.compile(r"[|;&<>`\n]|\$\(")
SIZE_BUCKETS = ((500, "under 500"), (2000, "500-2,000"), (8000, "2,000-8,000"),
                (10**9, "over 8,000"))


def plain_command(cmd: str) -> str | None:
    """The single command inside `cmd`, or None when it is a pipeline or compound: only
    then is the recorded output that one command's whole output."""
    cmd = STDERR_SUFFIX.sub("", CD_PREFIX.sub("", cmd.strip(), count=1))
    return None if not cmd or NOT_PLAIN.search(cmd) else cmd


def command_word(cmd: str) -> str:
    """A coarse label: the first word after any leading `cd <dir> &&`, plus the
    subcommand for git."""
    words = CD_PREFIX.sub("", cmd.strip(), count=1).split()
    if not words:
        return "(empty)"
    head = words[0].rsplit("/", 1)[-1]
    # The label is written to disk: a word that is not a bare program name (an
    # `ENV=value` prefix, a quoted path, a redirect) could carry content.
    if not WORD.match(head):
        return "(other)"
    if head == "git" and len(words) > 1 and words[1].isalpha():
        return f"git {words[1]}"
    return head


def method(rewritten: str) -> str | None:
    """How a rewritten plain command can be replayed: a filter name, "read", or None."""
    try:
        words = shlex.split(rewritten)
    except ValueError:
        return None
    while words and words[0] != "rtk":      # `uv run rtk pytest ...`
        words = words[1:]
    words = words[1:]
    if not words:
        return None
    if words[0] == "read":
        # With a gutter, a filter level or a preview cap, `rtk read` does change the text.
        changed = {"-n", "--line-numbers", "-l", "--level", "-m", "--max-lines"}
        return None if changed.intersection(w.split("=")[0] for w in words) else "read"
    return FILTERS.get(tuple(words[:2])) or FILTERS.get(tuple(words[:1]))


def size_bucket(tokens: int) -> str:
    for top, label in SIZE_BUCKETS:
        if tokens < top:
            return label
    return SIZE_BUCKETS[-1][1]


def run_rtk(jobs: list[dict], rtk: Path) -> dict[str, dict]:
    """id -> rtk_encode.py's record. No network, and RTK's state (its history, and what
    its filters elide) goes to a temporary home that is deleted."""
    if not jobs:
        return {}
    feed = "".join(json.dumps(j) + "\n" for j in jobs)
    with tempfile.TemporaryDirectory() as home:
        proc = subprocess.run(
            ["unshare", "-rn", sys.executable, str(HERE / "rtk_encode.py"), str(rtk), home],
            input=feed.encode("utf-8", "surrogatepass"), capture_output=True, check=True)
    lines = proc.stdout.decode("utf-8", "surrogatepass").split("\n")
    return {r["id"]: r for r in map(json.loads, filter(None, lines))}


def classify(rewrite: str | None, plain_rewrite: str | None) -> tuple[str, str | None]:
    """(status, method) for one Bash command. `rewrite` is RTK's form of the whole command,
    `plain_rewrite` its form of the single command inside it (None if not plain)."""
    if not rewrite:
        return "not_rewritten", None
    how = method(plain_rewrite) if plain_rewrite else None
    if how == "read":
        return "unchanged", how
    return ("measured", how) if how else ("unmeasured", None)


def summarize(rows: list[dict], sessions_weighted: float) -> dict:
    def total(group, key="cost"):
        return sum(r[key] for r in group)

    def share(part, whole):
        return round(100 * part / whole, 2) if whole else None

    bash = [r for r in rows if r["tool"] == "Bash"]
    read = [r for r in rows if r["tool"] == "Read"]
    bash_cost, read_cost = total(bash), total(read)
    by_status = _group(bash, "status")
    measured = by_status.get("measured", [])
    saved = total(measured, "saved")
    # Everything RTK takes and does not hand back unchanged, as if it removed all of it.
    ceiling = total(measured) + total(by_status.get("unmeasured", []))

    def fold(key, group):
        acc = defaultdict(lambda: {"results": 0, "cost": 0.0})
        for r in group:
            acc[r[key]]["results"] += 1
            acc[r[key]]["cost"] += r["cost"]
        return {k: {"results": v["results"], "cost": round(v["cost"], 1),
                    "share_of_bash_pct": share(v["cost"], bash_cost)}
                for k, v in sorted(acc.items(), key=lambda kv: -kv[1]["cost"])}

    def sizes(group):
        acc = defaultdict(float)
        for r in group:
            acc[size_bucket(r["tokens"])] += r["cost"]
        whole = total(group)
        return {label: share(acc[label], whole) for _, label in SIZE_BUCKETS}

    return {
        "sessions_weighted": round(sessions_weighted, 1),
        "bash": {"results": len(bash), "cost": round(bash_cost, 1),
                 "share_of_bill_pct": share(bash_cost, sessions_weighted)},
        "read": {"results": len(read), "cost": round(read_cost, 1),
                 "share_of_bill_pct": share(read_cost, sessions_weighted)},
        "rtk": {
            "by_status": {s: {"results": len(g), "share_of_bash_pct": share(total(g), bash_cost),
                              "share_of_bill_pct": share(total(g), sessions_weighted)}
                          for s, g in by_status.items()},
            "pytest_saved_pct": share(saved, total(measured)),
            "pytest_saved_share_of_bill_pct": share(saved, sessions_weighted),
            "ceiling_share_of_bash_pct": share(ceiling, bash_cost),
            "ceiling_share_of_bill_pct": share(ceiling, sessions_weighted),
            "filter_errors": sum(r["error"] for r in bash),
        },
        "bash_by_command": dict(list(fold("word", bash).items())[:20]),
        # Share of each command's COST that RTK takes, and that it hands back unchanged.
        "rtk_takes_by_command": {
            w: {"takes_pct": share(total([r for r in g if r["status"] != "not_rewritten"]),
                                   total(g)),
                "unchanged_pct": share(total([r for r in g if r["status"] == "unchanged"]),
                                       total(g))}
            for w, g in _group(bash, "word").items()},
        "size_shares": {"Bash": sizes(bash), "Read": sizes(read)},
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def _pct(value) -> str:
    return "-" if value is None else f"{value}%"


def render(summary: dict) -> str:
    b, r, k = summary["bash"], summary["read"], summary["rtk"]
    lines = [f"Bash: {b['results']} results, {b['cost']:,.0f} input-equivalent tokens = "
             f"{_pct(b['share_of_bill_pct'])} of all scanned sessions' bill",
             f"Read: {r['results']} results, {r['cost']:,.0f} = {_pct(r['share_of_bill_pct'])}",
             "", f"{'RTK on the Bash results':<26}{'results':>8}{'% of Bash':>11}"
                 f"{'% of bill':>11}"]
    for status in ("not_rewritten", "unchanged", "measured", "unmeasured"):
        slot = k["by_status"].get(status)
        if slot:
            lines.append(f"{status:<26}{slot['results']:>8}{_pct(slot['share_of_bash_pct']):>11}"
                         f"{_pct(slot['share_of_bill_pct']):>11}")
    lines += [f"measured (pytest only): RTK removes about {_pct(k['pytest_saved_pct'])} of "
              f"those results = {_pct(k['pytest_saved_share_of_bill_pct'])} of the bill; "
              f"{k['filter_errors']} filter errors",
              f"ceiling, if RTK removed every token of all it takes and changes: "
              f"{_pct(k['ceiling_share_of_bash_pct'])} of Bash cost = "
              f"{_pct(k['ceiling_share_of_bill_pct'])} of the bill. Its real saving is "
              f"below this and was not measured.",
              "", f"{'Bash cost by command':<26}{'results':>8}{'% of Bash':>11}"
                  f"{'RTK takes':>11}{'unchanged':>11}"]
    for word, slot in summary["bash_by_command"].items():
        takes = summary["rtk_takes_by_command"][word]
        lines.append(f"{word:<26}{slot['results']:>8}{_pct(slot['share_of_bash_pct']):>11}"
                     f"{_pct(takes['takes_pct']):>11}{_pct(takes['unchanged_pct']):>11}")
    lines.append("('RTK takes' and 'unchanged' are shares of that command's cost)")
    lines.append("")
    for tool, shares in summary["size_shares"].items():
        lines.append(f"{tool} cost by result size (cl100k tokens): {shares}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", type=Path,
                    default=Path.home() / ".local/state/terse/stats.jsonl",
                    help="only its first timestamp is used, to scan the replay's window")
    ap.add_argument("--transcripts", type=Path, default=Path.home() / ".claude/projects")
    ap.add_argument("--rtk", type=Path, required=True, help="the rtk binary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    args.rtk = args.rtk.resolve(strict=True)
    ledger = rs.load_ledger(args.ledger)
    if not ledger:
        print("no result rows in the ledger", file=sys.stderr)
        return 2
    # Before the long scan: is the sandbox there, and does this binary answer?
    probe = run_rtk([{"id": "probe", "cmd": "git status"}], args.rtk)
    if (probe.get("probe") or {}).get("rewrite") != "rtk git status":
        print("rtk did not rewrite `git status`: stopping", file=sys.stderr)
        return 2
    paths = rs.find_transcripts(args.transcripts, ledger[0].ts - 86400)
    paths.sort(key=lambda p: p.stat().st_mtime)
    billed: set[tuple] = set()

    def want(name: str) -> bool:
        return name in TOOLS or name.startswith("mcp__")

    calib: dict[str, list[float]] = defaultdict(list)
    scanned = []
    for n, path in enumerate(paths, 1):
        try:
            events = rs.scan_transcript(path, billed, want=want, keep_text=True)
        except OSError as exc:
            print(f"skipped {path}: {exc}", file=sys.stderr)
            continue
        # The ratio is the replay's: measured on MCP results only.
        mcp_only = [ev for ev in events
                    if ev[0] != "result" or ev[1].name.startswith("mcp__")]
        for model, sample in rs.calibration_samples(mcp_only):
            calib[model].append(sample)
        kept = [ev for ev in events if ev[0] != "result" or ev[1].name in TOOLS]
        for ev in kept:
            if ev[0] == "result" and ev[1].name != "Bash":
                ev[1].text = None       # only Bash output is replayed
        scanned.append((path, kept))
        if n % 1000 == 0:
            print(f"  scanned {n}/{len(paths)}", file=sys.stderr)
    ratio = rs.tokenizer_ratios(calib)

    # One item per tool result: its weight summed over every transcript that carried it.
    items: dict[str, dict] = {}
    sessions_weighted = 0.0
    for _, events in scanned:
        calls, marks, compacts = [], [], []
        for ev in events:
            if ev[0] == "call":
                calls.append(ev[1])
            elif ev[0] == "result":
                marks.append((ev[1], len(calls)))
            else:
                compacts.append(len(calls))
        sessions_weighted += sum(c.weighted() for c in calls if c.billed)
        for res, start in marks:
            later = [c for c in compacts if c >= start]
            per_w, _, _ = rs.carry(calls, start, later[0] if later else len(calls))
            model = calls[start].model if start < len(calls) else None
            slot = items.setdefault(res.tool_use_id, {
                "tool": res.name, "tokens": res.tokens or 0, "weight": 0.0,
                "text": res.text or "", "args": res.args, "offloaded": res.offloaded})
            slot["weight"] += ratio.get(model, ratio.get(None, 1.0)) * per_w
    del scanned
    print(f"{len(items)} Bash and Read results", file=sys.stderr)

    commands: dict[str, str] = {}
    for it in items.values():
        if it["tool"] != "Bash":
            continue
        try:
            cmd = json.loads(it["args"]).get("command") or ""
        except (ValueError, AttributeError):
            cmd = ""
        it["cmd"] = cmd if isinstance(cmd, str) else ""
        commands.setdefault(_sha(it["cmd"]), it["cmd"])
    jobs = [{"id": "w" + sha, "cmd": cmd} for sha, cmd in commands.items()]
    for sha, cmd in commands.items():
        plain = plain_command(cmd)
        if plain is not None and plain != cmd.strip():
            jobs.append({"id": "p" + sha, "cmd": plain})
    rewrites = run_rtk(jobs, args.rtk)
    failed = sum("error" in r for r in rewrites.values()) + len(jobs) - len(rewrites)
    print(f"{len(commands)} distinct commands, {len(jobs)} rewrite jobs, {failed} failed",
          file=sys.stderr)
    # A failed rewrite reads as "RTK never sees it": too many and the reach is understated.
    if failed > 0.01 * len(jobs):
        print("over 1% of rewrite jobs failed: stopping", file=sys.stderr)
        return 3

    filter_jobs: dict[str, dict] = {}
    for it in items.values():
        if it["tool"] != "Bash":
            it.update(status="", method="", word="")
            continue
        sha = _sha(it["cmd"])
        whole = (rewrites.get("w" + sha) or {}).get("rewrite")
        plain = plain_command(it["cmd"])
        if plain is None:
            plain_rw = None
        elif plain == it["cmd"].strip():
            plain_rw = whole
        else:
            plain_rw = (rewrites.get("p" + sha) or {}).get("rewrite")
        status, how = classify(whole, plain_rw)
        it.update(status=status, method=how or "", word=command_word(it["cmd"]))
        if status == "measured" and (it["offloaded"] or not it["text"].strip()):
            # A file preview, or nothing recorded: not that command's output.
            it.update(status="unmeasured", method="")
        if it["status"] == "measured":
            job_id = _sha(how + "\0" + it["text"])
            it["job"] = job_id
            filter_jobs.setdefault(job_id, {"id": job_id, "filter": how, "text": it["text"]})
    filtered = run_rtk(list(filter_jobs.values()), args.rtk)
    print(f"{len(filter_jobs)} filter jobs", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "shell.jsonl").open("w") as fh:
        for it in items.values():
            cost = it["tokens"] * it["weight"]
            saved, error = 0.0, False
            if it.get("job"):
                got = filtered.get(it["job"]) or {"error": "no record"}
                if "error" in got or not got.get("out", "").strip():
                    # Not measured after all: it joins the unmeasured rewritten results.
                    error, it["status"], it["method"] = True, "unmeasured", ""
                else:
                    saved = (it["tokens"] - rs.shown_tokens(got["out"])) * it["weight"]
            row = {"tool": it["tool"], "tokens": it["tokens"], "cost": round(cost, 2),
                   "saved": round(saved, 2), "status": it["status"], "method": it["method"],
                   "word": it["word"], "error": error}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    summary = summarize(rows, sessions_weighted)
    summary["rewrite_jobs_failed"] = failed
    summary["ratios_applied"] = {str(k): round(v, 3) for k, v in ratio.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render(summary)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
