"""terse CLI entrypoint.

Subcommands:
  gate <file|->            run the lossless round-trip gate on a JSON payload
  capture --tool N <file|-> persist a tool output to corpus/ + bucket by shape
  measure                  token delta per tier per shape bucket over the corpus
  probe                    value-redundancy + cross-call-overlap ceiling probes
  validate                 cross-tokenizer invariance (cl100k vs o200k)
  compress --tool N        compress one tool output through a policy (the shell)
  proxy -- <cmd>           MCP stdio proxy: compress a downstream server's results
  stats                    live savings report from the proxy's payload-free ledger
  fluency                  does a model read the compressed form as well as raw JSON?
  tune                     analyze a corpus -> safe-first drop candidates (+ optional verify)
"""

from __future__ import annotations

import argparse
import json
import json as _json
import re
import sys
import tempfile
import time
from datetime import UTC
from pathlib import Path

from . import transforms
from ._secure_io import write_restricted
from .capture import (
    capture_payload,
    classify_shape,
    coverage,
    extract_records,
    load_corpus,
)
from .html_report import build_html_diff_report, build_html_report
from .measure import cross_tokenizer_savings, measure_corpus
from .probes import (
    cross_call_overlap,
    cross_server_overlap,
    cross_server_redundancy,
    server_of_tool,
    value_redundancy,
)
from .report import (
    build_cross_server_probe_report,
    build_probe_report,
    build_report,
    build_tokenizer_report,
)
from .terminal_report import build_terminal_report
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


def _record_and_print_trend(history_path: str, rows: list, label: str) -> None:
    """Append this run's summary to the `--history` jsonl file and print the trend
    across every run recorded there so far (including this one). The only place
    in this command that reads the real clock (principle #31: inject nondeterminism
    at the edge, not inside the pure summarize_run/build_trend_report/
    trend_sparkline_lines core)."""
    from datetime import datetime

    from .history import append_run, load_history, summarize_run
    from .report import build_trend_report
    from .terminal_report import trend_sparkline_lines

    path = Path(history_path)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    run = summarize_run(rows, ts, label=label)
    all_runs = load_history(path) + [run]
    append_run(path, run)
    print("\n" + build_trend_report(all_runs))
    if len(all_runs) >= 2:
        print(trend_sparkline_lines(all_runs))
    print(f"[history: {len(all_runs)} run(s) recorded -> {path}]")


def _cmd_measure(args: argparse.Namespace) -> int:
    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1
    rows = measure_corpus(envelopes)
    cov = coverage(envelopes)
    report = build_report(rows, cov)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")
    if args.html:
        _write_html_report(build_html_report(rows, cov), out)
    if args.bars:
        print("\n" + build_terminal_report(rows))
    if args.history:
        _record_and_print_trend(args.history, rows, args.corpus)
    return 0


def _parse_headers(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated `--header NAME=VALUE` flags into a dict (#5, HTTP downstream
    auth). Raises ValueError with a clear message on a malformed entry (no `=`) rather
    than crashing with an unpacking error."""
    headers: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--header expects NAME=VALUE, got {p!r}")
        name, value = p.split("=", 1)
        headers[name] = value
    return headers


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .policy import default_policy, load_policy

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    # Validate the cmd/--config/--header combination BEFORE touching the policy file
    # (or anything else with a side effect): a missing downstream command must always
    # produce this clean, actionable message regardless of whether --policy also
    # happens to be bad — checking cmd only after loading the policy let an unrelated
    # bad --policy path crash with a raw traceback instead.
    if args.config:
        # --config (#5 Half B, multi-downstream fan-out) and a positional downstream
        # command are mutually exclusive: each names ITS downstream(s) a different way,
        # and silently picking one over the other would hide a likely typo/leftover flag.
        if cmd:
            print("proxy: --config and a downstream command (after `--`) are mutually "
                  "exclusive — use one or the other", file=sys.stderr)
            return 2
        if args.header:
            # --header has no --config equivalent: run_multi_proxy fronts N peers, each
            # with its own optional "headers" in the config file, so a single flag value
            # can't apply to all of them unambiguously. Reject loudly rather than
            # silently dropping a header the user thinks is taking effect.
            print("proxy: --header has no effect with --config — set a per-downstream "
                  '"headers" object in the config file instead', file=sys.stderr)
            return 2
    elif not cmd:
        print("proxy: provide the downstream server command after `--`, e.g.\n"
              "  terse proxy --policy p.json -- uvx some-mcp-server", file=sys.stderr)
        return 2

    try:
        pol = load_policy(args.policy) if args.policy else default_policy()
    except (OSError, ValueError) as e:
        print(f"proxy: {e}", file=sys.stderr)
        return 2
    # tri-state: --diff / --no-diff override the policy value; neither flag = keep it
    # (the Policy default is ON since the validation program completed — see #75).
    if args.diff and args.no_diff:
        print("proxy: --diff and --no-diff are mutually exclusive", file=sys.stderr)
        return 2
    diff_override = True if args.diff else (False if args.no_diff else None)
    if diff_override is not None:
        pol.diff = diff_override
    # --no-join-blocks overrides the policy value (join_blocks is ON by default, #116).
    join_blocks_override = False if args.no_join_blocks else None
    if join_blocks_override is not None:
        pol.join_blocks = join_blocks_override
    if args.diff_keyframe_interval is not None:
        pol.diff_keyframe_interval = args.diff_keyframe_interval
    if args.no_stats and args.stats_log:
        print("proxy: --no-stats and --stats-log are mutually exclusive", file=sys.stderr)
        return 2
    # The ledger is payload-free (sizes + decisions only — see stats.py), so unlike
    # capture/audit it defaults ON; --no-stats disables, --stats-log redirects.
    if args.no_stats:
        stats_log = None
    else:
        from .stats import default_stats_log

        stats_log = args.stats_log or str(default_stats_log())

    if args.config:
        from .multiproxy import run_multi_proxy
        try:
            # diff_override/diff_keyframe_override (not just the mutated `pol` above)
            # so --diff/--no-diff also applies to a peer with its OWN policy_path,
            # not just peers using this default policy.
            # No --server-name here: run_multi_proxy already knows each peer's own name
            # from the config file and passes it per-peer (a single flag couldn't name N
            # peers unambiguously — same reason --header is rejected with --config).
            if args.server_name:
                print("proxy: --server-name has no effect with --config — each peer's "
                      '"name" in the config file is used instead', file=sys.stderr)
                return 2
            return run_multi_proxy(args.config, pol, debug=args.debug,
                                   capture_dir=args.capture_dir, debug_log=args.debug_log,
                                   diff_override=diff_override,
                                   diff_keyframe_override=args.diff_keyframe_interval,
                                   join_blocks_override=join_blocks_override,
                                   stats_log=stats_log)
        except (OSError, ValueError) as e:
            print(f"proxy --config: {e}", file=sys.stderr)
            return 2

    from .proxy import run_proxy

    try:
        headers = _parse_headers(args.header)
    except ValueError as e:
        print(f"proxy: {e}", file=sys.stderr)
        return 2
    return run_proxy(cmd, pol, debug=args.debug, capture_dir=args.capture_dir,
                     debug_log=args.debug_log, headers=headers, stats_log=stats_log,
                     server_name=args.server_name)


def _cmd_stats(args: argparse.Namespace) -> int:
    from .stats import (
        aggregate,
        build_stats_report,
        default_stats_log,
        load_stats,
        parse_window,
    )

    log_path = args.log or str(default_stats_log())
    since_ts: int | None = None
    if args.since:
        try:
            since_ts = int(time.time()) - parse_window(args.since)
        except ValueError as e:
            print(f"stats: {e}", file=sys.stderr)
            return 2
    if not Path(log_path).exists() and not Path(log_path + ".1").exists():
        print(f"stats: no ledger at {log_path} — a terse-wrapped server writes one "
              f"automatically on its next tool call (unless run with --no-stats)",
              file=sys.stderr)
        return 2
    agg = aggregate(load_stats(log_path, since_ts))
    if args.json:
        print(json.dumps(agg, indent=2))
    else:
        print(build_stats_report(agg, log_path=log_path, window=args.since), end="")
    return 0


def _warn_if_dropping_capture_rules(out: Path) -> None:
    """`policy generate` emits a fresh document that never sets `capture: false`. If the
    target already holds hand-authored capture:false privacy rules (#85), overwriting them
    silently would re-enable payload persistence for those tools. Warn loudly and name
    them so the user re-adds any they still want — regeneration deliberately does NOT merge
    (a merge could resurrect a stale rule the operator removed on purpose)."""
    if not out.exists():
        return
    try:
        prior = _json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    guarded = [(p.get("match", {}) or {}).get("tool", "?")
               for p in (prior.get("policies") or [])
               if isinstance(p, dict) and p.get("capture") is False]
    if guarded:
        print(f"[warn] {out} already sets \"capture\": false for {len(guarded)} rule(s) "
              f"({', '.join(guarded)}); regeneration does NOT preserve them — re-add any "
              "you still want before relying on this file", file=sys.stderr)


def _cmd_policy_autotune(args: argparse.Namespace) -> int:
    """Re-tune an EXISTING policy against a corpus (#136).

    `policy generate` authors from nothing and overwrites; running it on a deployed policy
    silently drops every decision the corpus cannot see (it says so itself, for
    `capture: false` alone). This merges instead — the corpus owns `tiers`, the operator
    owns everything else — and writes NOTHING without `--apply`, so the diff is the default
    output rather than an after-the-fact warning."""
    from .policy import load_policy
    from .policy_gen import generate_policy, merge_policy

    existing_path = Path(args.policy)
    existing = _json.loads(existing_path.read_text(encoding="utf-8"))
    load_policy(existing_path)  # refuse to diff against a policy we can't even load

    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first "
              f"(`terse capture` or `proxy --capture-dir`).", file=sys.stderr)
        return 1

    generated, _rows = generate_policy(envelopes, threshold=args.threshold)
    merged, changes = merge_policy(existing, generated)

    kinds = {k: [c for c in changes if c["kind"] == k]
             for k in ("added", "tiers", "suggestions", "unchanged", "preserved")}
    print(f"# terse policy autotune — {len(envelopes)} payload(s), {existing_path}")
    for c in kinds["tiers"]:
        before = ",".join(c["before"]) or "(passthrough)"
        after = ",".join(c["after"]) or "(passthrough)"
        print(f"  ~ {c['tool']:<28} {before}  ->  {after}")
    for c in kinds["added"]:
        print(f"  + {c['tool']:<28} {','.join(c['after']) or '(passthrough)'}  (new rule)")
    for c in kinds["suggestions"]:
        print(f"  ~ {c['tool']:<28} drop-to-retrieve suggestions changed (still INACTIVE)")
    # Say what was deliberately NOT regenerated. An operator who can't see this can't tell
    # a preserved safety key from one the merge forgot.
    kept = sorted({k for c in changes for k in c.get("preserved", [])})
    if kept:
        print(f"  = preserved on existing rules, not regenerated: {', '.join(kept)}")
    if kinds["preserved"]:
        print(f"  = {len(kinds['preserved'])} rule(s) untouched "
              f"(not in this corpus): {', '.join(c['tool'] for c in kinds['preserved'])}")
    # `defaults` is listed explicitly rather than filtered out: it governs every tool with
    # no matching rule, so "was it regenerated?" is a real question about it.
    if top := sorted(k for k in existing if k not in ("version", "policies")):
        print(f"  = top-level preserved: {', '.join(top)}")
    if not (kinds["tiers"] or kinds["added"] or kinds["suggestions"]):
        print("  (no change — the deployed policy already matches this corpus)")

    # A proposed DOWNGRADE deserves a second look. It removes a tier from a rule that is
    # working today, on the evidence of a corpus that is a SAMPLE — idempotent by sha, so
    # it holds each payload's first sighting rather than every call, and it holds nothing at
    # all for a tool gated by `capture: false`. Adding a tier on thin evidence costs a
    # little transform time; removing one silently gives back measured savings.
    downgrades = [c for c in kinds["tiers"] if len(c["after"]) < len(c["before"])]
    if downgrades:
        print(f"\n[warn] {len(downgrades)} rule(s) would LOSE a tier "
              f"({', '.join(c['tool'] for c in downgrades)}). The corpus is a sample; cross-"
              "check those tools in `terse stats` — which counts every call, including ones "
              "whose payloads were never captured — before applying a downgrade.")
    if not args.apply:
        print("\nnothing written. Re-run with --apply to write the merged policy.")
        return 0

    text = _json.dumps(merged, ensure_ascii=False, indent=2)
    # Validate BEFORE writing, not after. `policy generate --out` can validate afterwards
    # because it is usually authoring a new file; autotune overwrites a policy that is
    # deployed and working, so a merged doc our own loader rejects must never reach it —
    # "fail loud" would otherwise mean "loudly, on top of the file you needed".
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as tf:
        tf.write(text)
        probe = tf.name
    try:
        load_policy(probe)
    except (OSError, ValueError) as exc:
        print(f"\nrefusing to write: the merged policy does not load ({exc}). "
              f"{existing_path} is unchanged.", file=sys.stderr)
        return 1
    finally:
        Path(probe).unlink(missing_ok=True)
    write_restricted(existing_path, text + "\n", mode=0o644)
    print(f"\n[policy written to {existing_path}]")
    return 0


def _cmd_policy_generate(args: argparse.Namespace) -> int:
    from .policy import load_policy
    from .policy_gen import generate_policy

    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first "
              f"(`terse capture` or `proxy --capture-dir`).", file=sys.stderr)
        return 1

    doc, rows = generate_policy(envelopes, threshold=args.threshold)
    text = _json.dumps(doc, ensure_ascii=False, indent=2)

    # Per-tool decision summary to stderr so stdout stays a clean policy when piped.
    print(f"# terse policy generate — {len(rows)} tool(s), threshold {args.threshold:.1f}%",
          file=sys.stderr)
    for r in rows:
        tiers = ",".join(r["tiers"]) or "(passthrough)"
        print(f"  {r['tool']:<28} {tiers:<28} {r['reason']}", file=sys.stderr)
        for dr in r.get("drop_rows", []):
            print(f"      ↳ drop-candidate {dr['path']} "
                  f"(~{dr['tok_share']*100:.0f}% of tokens, {dr['uniq_ratio']*100:.0f}% unique, "
                  f"~{dr['mean_tok']:.0f} tok/value) — suggested, off by default",
                  file=sys.stderr)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _warn_if_dropping_capture_rules(out)  # before we overwrite it
        # Atomic + O_EXCL temp write (see _secure_io): a crash mid-write can't leave a
        # half-truncated policy.json. mode 0o644 — a policy is config, not a secret.
        write_restricted(out, text + "\n", mode=0o644)
        # Fail loudly if we just wrote a policy our own loader rejects — a generated file
        # that can't be loaded is worse than none.
        load_policy(out)
        print(f"[policy written to {out} — verify comprehension with "
              f"`terse fluency --corpus {args.corpus}`]", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")
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

    if getattr(args, "cross_server", False):
        return _cmd_probe_cross_server(args, envelopes)

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
        for prev, curr in zip(envs, envs[1:], strict=False):  # sliding pairs
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


def _cmd_probe_cross_server(args: argparse.Namespace, envelopes: list[dict]) -> int:
    """#64 Phase 0: group the corpus by origin server, measure cross-peer redundancy."""
    records_by_server: dict[str, list[dict]] = {}
    raws_by_server: dict[str, list[tuple[str, str]]] = {}
    for env in envelopes:
        srv = server_of_tool(env["tool"])
        raws_by_server.setdefault(srv, []).append((env.get("sha", ""), env["raw"]))
        try:
            records = extract_records(_json.loads(env["raw"]))
        except (ValueError, TypeError):
            records = None
        if records:
            records_by_server.setdefault(srv, []).extend(records)

    # Gate on RAW payloads, not record-shaped ones: Lever B (framing-normalized overlap)
    # works on any payload shape and is the decisive signal when servers emit text/source
    # (codegraph) rather than record lists. Requiring 2 record-shaped servers would wrongly
    # block Lever B exactly when it matters most. Lever A degrades gracefully to empty and
    # the report's coverage guard flags it blind.
    if len(raws_by_server) < 2:
        print("need payloads from ≥2 servers to probe cross-server redundancy — found "
              f"{sorted(raws_by_server)}.")
        return 1

    redundancy = cross_server_redundancy(records_by_server)
    overlap = cross_server_overlap(raws_by_server, cap_per_pair=args.cap)
    report = build_cross_server_probe_report(
        redundancy, overlap, corpus_servers=sorted(raws_by_server)
    )
    out = Path(args.out if args.out != DEFAULT_PROBE_REPORT else "reports/cross-server-probe.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")
    return 0


def _build_answerers(args: argparse.Namespace, make_openai) -> dict:
    """Assemble named answerers from env + flags. Empty means keyless (pack) mode.

    Shared by plain `fluency` (`make_openai=fluency.openai_answerer`) and
    `fluency --drop-eval` (`make_openai` bound to `dropeval.openai_tool_answerer` +
    the retrieve tool) so the two eval modes stay configured identically — only the
    answerer FACTORY differs, never the env/flag precedence. Every model is reached over
    the OpenAI-compatible path (the broker pool or a loopback gateway) — there is no
    other model backend."""
    import os

    # Flags win over env so a credential-injecting launcher (e.g. secret_inject_env,
    # which sets the key under its own env var) can drive the CLI without a shell.
    answerers: dict = {}
    base = args.base_url or os.environ.get("TERSE_FLUENCY_BASE_URL")
    key = os.environ.get(args.api_key_env or "TERSE_FLUENCY_API_KEY")
    models = args.models or os.environ.get("TERSE_FLUENCY_MODELS", "")
    if base and key and models:
        for m in (x.strip() for x in models.split(",") if x.strip()):
            answerers[m] = make_openai(base, key, m)
    return answerers


def _tune_drop_eval(args: argparse.Namespace, doc: dict, envelopes: list) -> int:
    """Verify the generated drop suggestions with a live tool-calling model: promote the
    suggestions in-memory, run the real 2-turn retrieve eval, and print the verdict. Does
    NOT auto-enable — the operator enables after seeing the report (a model verdict must not
    silently edit a policy). Reuses the exact `fluency --drop-eval` machinery."""
    import tempfile

    from . import dropeval
    from .policy import load_policy
    from .policy_gen import activate_suggestions
    from .proxy import RETRIEVE_TOOL_DEF
    from .report import build_dropeval_report

    answerers = _build_answerers(
        args,
        lambda base, key, m: dropeval.openai_tool_answerer(base, key, m,
                                                            tools=[RETRIEVE_TOOL_DEF]),
    )
    if not answerers:
        print("--drop-eval needs a configured model: set TERSE_FLUENCY_BASE_URL/_API_KEY/"
              "_MODELS (or --base-url/--models).", file=sys.stderr)
        return 1
    active = activate_suggestions(doc)
    # Write inside the `with` (so the handle is CLOSED after it), then load by name: Windows
    # forbids reopening a temp file whose handle is still open, so a `delete=True` block that
    # loaded inside it would crash there. delete=False keeps the closed file for the reload.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        tf.write(_json.dumps(active))
        tmp_name = tf.name
    try:
        pol = load_policy(tmp_name)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    if not pol.has_drop():
        print("--drop-eval: no drop suggestions to verify.")
        return 0
    print("\nverifying the suggested drops with a live model (does it call terse.retrieve "
          "when the dropped field is needed, and skip it when not?)...")
    results = dropeval.run_drop_fluency(envelopes, pol.select, answerers, trials=args.trials)
    print("\n" + build_dropeval_report(results))
    print("If the worst-case model PASSES, enable the verified fields by renaming that tool's "
          "'_suggested_fields' -> 'fields' in the policy.")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    """One-command lossy tuning: analyze a captured corpus, surface safe-first drop-to-
    retrieve candidates (role-classified — prose is safe, unknown may be load-bearing), write
    the generated policy (suggestions INACTIVE), and optionally verify them with a live model.
    Chains `policy generate` + the drop-eval into the single flow an operator would otherwise
    assemble by hand."""
    from .policy_gen import generate_policy

    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture` or "
              f"`proxy --capture-dir`).", file=sys.stderr)
        return 1
    doc, rows = generate_policy(envelopes, threshold=args.threshold)
    cands = [{"tool": r["tool"], **dr} for r in rows for dr in r.get("drop_rows", [])]

    if args.out:
        from .policy import load_policy
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        load_policy(out)  # fail loud if we just wrote a policy our own loader rejects

    print(f"# terse tune — {len(envelopes)} payload(s), {len(rows)} tool(s), "
          f"{len(cands)} drop candidate(s)")
    if not cands:
        print("no drop-to-retrieve candidates — the lossless tiers already cover these tools, "
              "or every large field is a key/identity (never dropped).")
        if args.out:
            print(f"policy written to {args.out} (lossless).")
        return 0

    # Denominator for the rollup: every analyzed tool's raw token volume, so a bucket's
    # estimated drop can be stated as a share of the whole corpus, not just per field.
    corpus_raw = sum(r.get("raw_tok", 0) or 0 for r in rows)

    def _est_tokens(items: list) -> float:
        # Gross tokens a bucket's drops would evict: mean field tokens x record count,
        # summed. dimensionally clean across tools (unlike summing per-tool tok_share,
        # whose denominators differ). Net is slightly less — each drop leaves a small
        # retrieve handle in place of the field.
        return sum((c.get("mean_tok") or 0) * (c.get("n") or 0) for c in items)

    def _show(label: str, items: list) -> None:
        if not items:
            return
        print(f"\n{label}:")
        for c in items:
            print(f"  {c['tool']:<28} {c['path']:<26} "
                  f"~{c['tok_share'] * 100:.0f}% tok, {c['uniq_ratio'] * 100:.0f}% uniq  "
                  f"[{c.get('role', 'unknown')}]")
        est = _est_tokens(items)
        share = f", ~{est / corpus_raw * 100:.0f}% of corpus" if corpus_raw else ""
        print(f"  → enabling all {len(items)} here: ≈{est:,.0f} tok{share} "
              "(gross, before the per-record retrieve-handle cost)")

    _show("SAFE candidates — supporting prose, enable after a dropeval pass",
          [c for c in cands if c.get("role") == "prose"])
    _show("REVIEW candidates — role unknown, may be LOAD-BEARING; verify carefully",
          [c for c in cands if c.get("role") != "prose"])
    if args.out:
        print(f"\npolicy written to {args.out} — drop suggestions are INACTIVE "
              "(`_suggested_fields`) until you opt in.")

    if args.drop_eval:
        return _tune_drop_eval(args, doc, envelopes)
    print("\nNext — verify, then enable:")
    tgt = args.out or "<policy.json>"
    print(f"  terse tune --corpus {args.corpus} --out {tgt} --drop-eval --models ...  "
          "# verify with a live model")
    print(f"  then in {tgt}: rename a tool's '_suggested_fields' -> 'fields' (start with "
          "[prose]; leave any [unknown] that fails dropeval).")
    return 0


def _cmd_fluency(args: argparse.Namespace) -> int:
    from . import dropeval, fluency
    from .policy import default_policy, load_policy
    from .report import (
        build_diff_report,
        build_diff_soak_report,
        build_dropeval_report,
        build_fluency_report,
        build_text_diff_report,
    )
    from .terminal_report import (
        build_terminal_diff_report,
        build_terminal_dropeval_report,
        build_terminal_fluency_report,
    )

    envelopes = load_corpus(args.corpus)
    if not envelopes:
        print(f"no payloads in {args.corpus}/ — capture some first (`terse capture`).")
        return 1

    # --html renders the forest plot, which only the paired diff-family evals produce;
    # flag it as ignored elsewhere rather than let it be a silent no-op.
    if args.html and not (args.diff or args.diff_soak or args.text_diff_eval):
        print("note: fluency --html applies only to --diff / --diff-soak / --text-diff-eval "
              "(the forest-plot evals); ignoring it here.")

    # Drop-eval mode: does a real tool-calling model call terse.retrieve when a dropped
    # field is needed, and leave it alone when it isn't? Needs a policy with a
    # drop-to-retrieve field AND a live tool-capable model — like --diff, this is a
    # live-model-only behavioral measurement, no pack/offline mode.
    if args.drop_eval:
        pol = load_policy(args.policy) if args.policy else default_policy()
        if not pol.has_drop():
            print("`fluency --drop-eval` needs a policy with a drop-to-retrieve field "
                  "(pass --policy).")
            return 1
        from .proxy import RETRIEVE_TOOL_DEF

        answerers = _build_answerers(
            args,
            lambda base, key, m: dropeval.openai_tool_answerer(base, key, m,
                                                                tools=[RETRIEVE_TOOL_DEF]),
        )
        if not answerers:
            print("`fluency --drop-eval` needs a configured model: set TERSE_FLUENCY_BASE_URL/"
                  "_API_KEY/_MODELS.")
            return 1
        results = dropeval.run_drop_fluency(envelopes, pol.select, answerers, trials=args.trials)
        _write_report(build_dropeval_report(results), args.out)
        if args.bars:
            print("\n" + build_terminal_dropeval_report(results))
        return 0

    # Diff mode: does a model read a cross-call DIFF as well as the full result? Needs a
    # live model (it measures comprehension of a form, not ground-truth math).
    if args.diff:
        answerers = _build_answerers(args, fluency.openai_answerer)
        if not answerers:
            print("`fluency --diff` needs a configured model: set TERSE_FLUENCY_BASE_URL/"
                  "_API_KEY/_MODELS.")
            return 1
        results = fluency.run_diff_fluency(envelopes, answerers, trials=args.trials)
        _write_report(build_diff_report(results), args.out)
        _maybe_write_diff_html(args, results)
        if args.bars:
            print("\n" + build_terminal_diff_report(results))
        return 0

    # Diff-chain soak: the DEPTH dimension --diff can't see (#8/#20 follow-up) — does
    # comprehension drift as consecutive diffs chain off one full anchor? Depth 5 is
    # the production keyframe bound, so a PASS here covers the deployed worst case.
    if args.diff_soak:
        answerers = _build_answerers(args, fluency.openai_answerer)
        if not answerers:
            print("`fluency --diff-soak` needs a configured model: set TERSE_FLUENCY_BASE_URL/"
                  "_API_KEY/_MODELS.")
            return 1
        results = fluency.run_diff_soak(envelopes, answerers, trials=args.trials,
                                        max_depth=args.soak_depth,
                                        per_depth_cap=args.soak_windows)
        _write_report(build_diff_soak_report(results), args.out)
        _maybe_write_diff_html(args, results, form_label="chain-form")
        if args.bars:
            print("\n" + build_terminal_diff_report(results, form_label="chain-form"))
        return 0

    # Text-diff mode: does a model reconstruct the current TEXT as well from (previous
    # text + text-diff) as from the full current text? The text-payload analogue of
    # --diff above (text_diff.py, Tier 0.7 — non-JSON tool output).
    if args.text_diff_eval:
        answerers = _build_answerers(args, fluency.openai_answerer)
        if not answerers:
            print("`fluency --text-diff-eval` needs a configured model: set "
                  "TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS.")
            return 1
        results = fluency.run_text_diff_fluency(envelopes, answerers, trials=args.trials)
        _write_report(build_text_diff_report(results), args.out)
        _maybe_write_diff_html(args, results, control_label="raw text")
        if args.bars:
            print("\n" + build_terminal_diff_report(results, control_label="raw text"))
        return 0

    # Score mode: an externally-collected responses file against a previously-written pack.
    if args.responses:
        pack = _json.loads(Path(args.pack).read_text(encoding="utf-8"))
        responses = _json.loads(Path(args.responses).read_text(encoding="utf-8"))
        results = fluency.score_pack(pack, responses)
        report = build_fluency_report(results, fluency.token_summary(envelopes))
        _write_report(report, args.out)
        if args.bars:
            print("\n" + build_terminal_fluency_report(results))
        return 0

    answerers = _build_answerers(args, fluency.openai_answerer)
    if not answerers:
        # Keyless default: write the eval pack and explain how to drive it. The pack
        # embeds each payload's RAW captured text (fluency.build_pack) — the same
        # "may contain real data" class capture_payload protects at 0600 — so write it
        # the same way, not via plain write_text.
        pack = fluency.build_pack(envelopes, trials=args.trials)
        out = Path(args.pack)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_restricted(out, _json.dumps(pack, ensure_ascii=False, indent=2))
        nq = sum(len(p["questions"]) for p in pack["payloads"])
        print(f"no model configured — wrote {nq} questions over {len(pack['payloads'])} "
              f"record-shaped payloads to {out}.")
        print("To run a model: set TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS (broker pool), "
              "then re-run.")
        print(f"Or drive the pack by hand and score it: `terse fluency --responses <file> "
              f"--pack {out}`.")
        return 0

    results = fluency.run_fluency(envelopes, answerers, trials=args.trials)
    report = build_fluency_report(results, fluency.token_summary(envelopes))
    _write_report(report, args.out)
    if args.bars:
        print("\n" + build_terminal_fluency_report(results))
    return 0


# MCP servers commonly carry credentials directly in `args` (not just `env`); before/
# after command lines get printed unconditionally by install/uninstall-mcp, including
# to shared/logged terminals, so a secret-shaped flag's value is masked either way it
# appears: `--api-key VALUE` (two args) or `--api-key=VALUE` (one arg).
_SECRET_FLAG = re.compile(r"^--?(api[-_]?key|token|secret|password|passwd|auth|credential)s?$",
                          re.IGNORECASE)

# `--header NAME=VALUE` (#5, HTTP downstream auth) carries its secret in the VALUE half
# of a single arg, not in a flag NAME `_SECRET_FLAG` matches — so a bare `--header`
# match would print `Authorization=Bearer xyz` unredacted. Match against the header
# NAME instead, substring (not anchored like `_SECRET_FLAG`): a header name space is
# open-ended and caller-defined (`X-Api-Key`, `Proxy-Authorization`, `Cookie`, ...), so
# over-redacting a borderline name is the safe failure direction for a value that could
# be printed to a shared terminal.
_SECRET_HEADER = re.compile(r"(api[-_]?key|token|secret|password|passwd|auth|credential|cookie)",
                            re.IGNORECASE)


def _redact_header_value(entry: str) -> str:
    """Mask a `--header NAME=VALUE` entry's VALUE when NAME looks secret-shaped.
    Leaves the entry alone (name AND value) when it doesn't look secret-shaped, and
    when there's no `=` at all (a malformed entry `_parse_headers` would reject anyway —
    not this function's job to validate)."""
    if "=" not in entry:
        return entry
    name, _, _value = entry.partition("=")
    return f"{name}=***" if _SECRET_HEADER.search(name) else entry


def _redact_args(args: list) -> list:
    out = []
    pending: str | None = None  # None | "value" (generic secret flag) | "header"
    for a in args:
        if pending == "header":
            out.append(_redact_header_value(a))
            pending = None
            continue
        if pending == "value":
            out.append("***")
            pending = None
            continue
        flag = a.split("=", 1)[0]
        if flag == "--header":
            if "=" in a:
                # Inline form `--header=NAME=VALUE` (argparse also accepts this): the
                # first `=` separates the flag from the NAME=VALUE payload.
                _, rest = a.split("=", 1)
                out.append(f"--header={_redact_header_value(rest)}")
            else:
                out.append(a)
                pending = "header"  # the NEXT arg is the NAME=VALUE payload
            continue
        if _SECRET_FLAG.match(flag):
            out.append(f"{flag}=***" if "=" in a else a)
            pending = "value" if "=" not in a else None
            continue
        out.append(a)
    return out


def _short_cmd(entry) -> str:
    if not entry:
        return "(absent)"
    if "url" in entry:
        # A url/headers-shaped entry (#5 HTTP downstream) has no "command"/"args" at
        # all — falling through to entry.get("command", "?") used to print just "?",
        # silently losing the url/headers info from the before/after display.
        parts = [entry.get("url", "?")]
        for k, v in (entry.get("headers") or {}).items():
            parts += ["--header", _redact_header_value(f"{k}={v}")]
        return " ".join(parts)[:100]
    args = _redact_args(entry.get("args", []))
    return " ".join([entry.get("command", "?"), *args])[:100]


def _cmd_install_mcp(args: argparse.Namespace) -> int:
    from .install_mcp import classify_server_sensitivity, do_install

    if args.diff and args.no_diff:
        print("install-mcp: --diff and --no-diff are mutually exclusive", file=sys.stderr)
        return 2
    diff = True if args.diff else (False if args.no_diff else None)
    try:
        res = do_install(args.servers, args.policy, dry_run=args.print,
                         capture_dir=args.capture_dir, diff=diff,
                         diff_keyframe_interval=args.diff_keyframe_interval,
                         scope=args.scope, file=args.file, repo_path=args.repo_path,
                         no_stats=args.no_stats, no_join_blocks=args.no_join_blocks,
                         never_lossy=args.never_lossy)
    except (FileNotFoundError, ValueError) as e:
        print(f"install-mcp: {e}", file=sys.stderr)
        return 2
    tag = "[dry-run] would wrap" if res["dry_run"] else "wrapped"
    for c in res["changes"]:
        print(f"{tag} {c['server']}:")
        print(f"    before: {_short_cmd(c['before'])}")
        print(f"    after:  {_short_cmd(c['after'])}")
        if c.get("preserved"):
            print(f"    kept hand-edited key(s) from the live entry: "
                  f"{', '.join(c['preserved'])} (note: uninstall restores the "
                  f"pre-terse original, which does NOT carry them)")
    print(f"config: {res['config']}  scope: {res['scope']}  policy: {res['policy']}")
    if res.get("capture_dir"):
        print(f"capture: raw tool results → {res['capture_dir']}")
    if res.get("diff") is True:
        print("diff: explicit --diff baked in (overrides a policy-file opt-out)")
    elif res.get("diff") is False:
        print("diff: DISABLED for these server(s) (--no-diff baked in)")
    if res.get("no_stats"):
        print("stats: DISABLED for these server(s) (--no-stats baked in)")
    baked = res.get("never_lossy_added") or []
    if baked:
        verb = "would bake" if res["dry_run"] else "baked"
        print(f"never-lossy: {verb} {', '.join(baked)} into the policy's never_lossy_servers "
              f"— lossy transforms are now forbidden on them")
    elif not args.never_lossy:
        # Surface the classifier as a HINT for servers NOT explicitly marked: a
        # credential/personal-looking server is already floor-protected (PR #89), but
        # listing it makes the intent explicit and covers names the floor can't catch.
        for c in res["changes"]:
            before = c.get("before") or {}
            cmd_parts = [before.get("command", ""), *before.get("args", [])]
            if classify_server_sensitivity(c["server"], cmd_parts):
                print(f"hint: '{c['server']}' looks credential/personal — re-run with "
                      f"--never-lossy (or add it to never_lossy_servers) to forbid lossy "
                      f"on it explicitly.", file=sys.stderr)
    if res["backup"]:
        print(f"backup: {res['backup']}")
    if not res["dry_run"] and res["changes"]:
        print("→ restart Claude Code for the change to take effect.")
    return 0


def _cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    from .install_mcp import do_uninstall

    try:
        res = do_uninstall(args.servers, all_=args.all, dry_run=args.print,
                           scope=args.scope, file=args.file, repo_path=args.repo_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"uninstall-mcp: {e}", file=sys.stderr)
        return 2
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


def _cmd_mcp_status(args: argparse.Namespace) -> int:
    from .install_mcp import scan_scopes

    rows = scan_scopes(file=args.file, repo_path=args.repo_path)
    if args.json:
        # Scriptable/CI-checkable parity with `terse stats --json`; [] on no servers.
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no MCP servers found in any scope (user/project/local).")
        return 0
    by_scope: dict[str, list[dict]] = {}
    for r in rows:
        by_scope.setdefault(r["scope"], []).append(r)
    for scope in ("user", "project", "local"):
        scope_rows = by_scope.get(scope)
        if not scope_rows:
            continue
        print(f"[{scope}] {scope_rows[0]['config']}")
        for r in scope_rows:
            policy = ""
            if r["policy"]:
                miss = " (MISSING)" if r.get("policy_missing") else ""
                policy = f"  policy={r['policy']}{miss}"
            print(f"  {r['server']:<20} {r['state']}{policy}")
            # For a wrapped entry, a second indented line surfaces what it actually
            # fronts and the tiers baked into the entry — the diagnostic the flat
            # "wrapped policy=…" line couldn't answer when a server misbehaves.
            if r["state"] == "wrapped":
                stats = "on" if r.get("stats") else "off"
                detail = (f"wraps={r.get('wraps') or '?'}  "
                          f"diff={r.get('diff') or '?'}  stats={stats}")
                print(f"  {'':<20} {detail}")
                # A launcher that no longer resolves is fatal to the entry and invisible
                # everywhere else — the client just fails to spawn it. Print it last so
                # it reads as the verdict on the two lines above.
                if r.get("launcher_missing"):
                    print(f"  {'':<20} launcher={r.get('launcher')} (MISSING) "
                          f"— this entry cannot start; re-run install-mcp")
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
    from .report import build_report, build_verify_header, verify_summary

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

    rows = measure_corpus(envelopes)
    cov = coverage(envelopes)
    if args.json:
        # Machine mode: emit the aggregate and stop — no markdown/html/bars artifacts
        # (parity with `stats --json` / `mcp-status --json`).
        if args.html or args.bars:
            print("note: verify --json emits JSON only; --html/--bars are ignored.",
                  file=sys.stderr)
        print(_json.dumps(verify_summary(rows, cov, label), indent=2))
        return 0
    report = build_verify_header(label, len(envelopes)) + build_report(rows, cov)
    _write_report(report, args.out)
    if args.html:
        html = build_html_report(rows, cov, attestation=(label, len(envelopes)))
        _write_html_report(html, Path(args.out))
    if args.bars:
        print("\n" + build_terminal_report(rows))
    return 0


def _write_report(report: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {out}]")


def _write_html_report(html: str, md_out_path: Path) -> None:
    """Write the HTML chart companion alongside a markdown report's --out path,
    swapping its suffix for .html (e.g. reports/verify-report.md -> ...html)."""
    out = md_out_path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[html report written to {out}]")


def _maybe_write_diff_html(args: argparse.Namespace, results: dict,
                           form_label: str = "diff-form",
                           control_label: str = "full-terse") -> None:
    """With `fluency --html`, write the forest-plot HTML companion for a diff-family eval
    next to --out. Labels mirror the terminal report's so the two never disagree. No-op
    without --html."""
    if getattr(args, "html", False):
        _write_html_report(build_html_diff_report(results, form_label, control_label),
                           Path(args.out))


def _terse_version() -> str:
    """Installed distribution version, falling back to the package `__version__` when
    terse is run from a source tree that was never `pip install`ed (e.g. `python -m terse`
    in a checkout). importlib.metadata reflects the ACTUAL installed dist, so it stays
    correct once tag-derived versioning lands."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version
    try:
        return _dist_version("terse")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terse", description=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {_terse_version()}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="run the lossless round-trip gate on a JSON file")
    g.add_argument("file", help="path to a JSON payload, or - for stdin")
    g.set_defaults(func=_cmd_gate)

    pol = sub.add_parser("policy", help="author/inspect a per-tool policy")
    pol_sub = pol.add_subparsers(dest="policy_cmd", required=True)
    pg = pol_sub.add_parser("generate", help="auto-author a conservative lossless policy "
                                             "from a measured corpus")
    pg.add_argument("--corpus", default=DEFAULT_CORPUS)
    pg.add_argument("--out", help="write the policy here (default: stdout)")
    pg.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                    help="min total savings %% to compress a tool, and min marginal %% to add "
                         "the dictionary tier (default 5.0; conservative)")
    pg.set_defaults(func=_cmd_policy_generate)
    pa = pol_sub.add_parser("autotune", help="re-tune an EXISTING policy from a corpus — "
                                             "merge, don't overwrite; prints a diff")
    pa.add_argument("--policy", required=True, help="the existing policy to re-tune")
    pa.add_argument("--corpus", default=DEFAULT_CORPUS)
    pa.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                    help="as `policy generate` (default 5.0)")
    pa.add_argument("--apply", action="store_true",
                    help="write the merged policy. Without it, nothing is written.")
    pa.set_defaults(func=_cmd_policy_autotune)

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
    m.add_argument("--html", action="store_true",
                   help="also write a charted HTML report next to --out (inline SVG, no JS/CDN)")
    m.add_argument("--bars", action="store_true",
                   help="also print terminal bar charts for the savings sections (ANSI if a tty)")
    m.add_argument("--history", metavar="FILE",
                   help="append this run's summary to FILE (jsonl) and print the trend "
                        "across every run recorded there so far (#51 fast-follow)")
    m.set_defaults(func=_cmd_measure)

    p = sub.add_parser("probe", help="value-redundancy + cross-call-overlap ceiling probes")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--out", default=DEFAULT_PROBE_REPORT)
    p.add_argument("--cross-server", action="store_true",
                   help="#64 Phase 0: cross-peer dictionary headroom (writes "
                        "reports/cross-server-probe.md unless --out is set)")
    p.add_argument("--cap", type=int, default=20,
                   help="max payloads per server-pair for the raw-overlap lever (default 20)")
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
                    help="force cross-call diffing ON, overriding a policy file's "
                         '"diff": false (diffing is the DEFAULT since #75, so this is '
                         "only needed against such a policy)")
    px.add_argument("--no-diff", action="store_true",
                    help="disable cross-call diffing (emit the full compressed form "
                         "every call), overriding the default and any policy value")
    px.add_argument("--no-join-blocks", action="store_true",
                    help="disable joining a multi-block result into one record array before "
                         "compressing (#116; joining is the DEFAULT). Forwards each text "
                         "block compressed independently, preserving the block count.")
    px.add_argument("--debug", action="store_true", help="log compressions to stderr")
    px.add_argument("--capture-dir", metavar="DIR",
                    help="tee each raw tool-result payload into this corpus dir for later "
                         "`terse verify --corpus`/`measure` (opt-in; never affects forwarding)")
    px.add_argument("--debug-log", metavar="FILE",
                    help="append a structured raw->decision->emitted record per result to "
                         "this JSONL file for after-the-fact diagnosis/replay (opt-in; never "
                         "affects forwarding)")
    px.add_argument("--server-name", metavar="NAME",
                    help="this downstream's name in your MCP config (e.g. runecho). Makes "
                         "a server-scoped policy rule (\"runecho.*\") match a server whose "
                         "tools aren't self-prefixed, and labels `terse stats` with the "
                         "real server instead of the command basename. `install-mcp` bakes "
                         "this in automatically.")
    px.add_argument("--stats-log", metavar="FILE",
                    help="path for the payload-free savings ledger read by `terse stats` "
                         "(default: $XDG_STATE_HOME/terse/stats.jsonl; sizes + decisions "
                         "only, never payload content; never affects forwarding)")
    px.add_argument("--no-stats", action="store_true",
                    help="disable the savings ledger (it is ON by default — safe because "
                         "it stores no payload content)")
    px.add_argument("--header", action="append", metavar="NAME=VALUE",
                    help="HTTP header to send to an HTTP/SSE downstream (repeatable), e.g. "
                         "--header 'Authorization=Bearer xyz'. Ignored for a stdio downstream. "
                         "Not valid with --config — set headers per-downstream in that file.")
    px.add_argument("--config", metavar="FILE",
                    help="JSON file listing multiple downstream peers to front behind "
                         "one process (#5 Half B, fan-out): "
                         '{"downstreams":[{"name":"gh","command":[...]},'
                         '{"name":"kb","url":"https://...","headers":{...}}]}. '
                         "Mutually exclusive with a positional downstream command.")
    px.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- <downstream MCP server command and args, or a single URL>")
    px.set_defaults(func=_cmd_proxy)

    st = sub.add_parser("stats", help="live savings report from the proxy's always-on "
                                      "payload-free ledger")
    st.add_argument("--log", metavar="FILE",
                    help="ledger path (default: $XDG_STATE_HOME/terse/stats.jsonl — "
                         "where `terse proxy` writes unless --stats-log/--no-stats)")
    st.add_argument("--since", metavar="WINDOW",
                    help="only count records newer than this window, e.g. 30m, 24h, 7d "
                         "(default: all recorded history)")
    st.add_argument("--json", action="store_true",
                    help="emit the aggregate as JSON instead of the text report")
    st.set_defaults(func=_cmd_stats)

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
    f.add_argument("--text-diff-eval", action="store_true",
                   help="behavioral eval: does a model reconstruct the current TEXT as "
                        "accurately from (previous text + text-diff) as from the full "
                        "text? needs same-tool TEXT corpus pairs + a configured model")
    f.add_argument("--diff-soak", action="store_true",
                   help="drift soak: does comprehension degrade as a model reads chains "
                        "of consecutive diffs off one full anchor? scores depths "
                        "1..--soak-depth over real corpus runs (needs a configured model)")
    f.add_argument("--soak-depth", type=int, default=5, metavar="K",
                   help="--diff-soak: deepest chain to test (default 5 — the production "
                        "keyframe interval, i.e. the deployed worst case)")
    f.add_argument("--soak-windows", type=int, default=6, metavar="N",
                   help="--diff-soak: max chain windows per depth, round-robin across "
                        "tools (default 6)")
    f.add_argument("--drop-eval", action="store_true",
                   help="behavioral eval: does a real tool-calling model call terse.retrieve "
                        "when a dropped field is needed (recall), and leave it alone when "
                        "it isn't (precision)? needs --policy with a drop-to-retrieve field "
                        "+ a configured model")
    f.add_argument("--policy", help="policy file with a drop-to-retrieve field (used only "
                                    "by --drop-eval)")
    f.add_argument("--base-url", help="OpenAI-compatible base URL (else $TERSE_FLUENCY_BASE_URL)")
    f.add_argument("--models", help="comma-separated model ids (else $TERSE_FLUENCY_MODELS)")
    f.add_argument("--api-key-env", default="TERSE_FLUENCY_API_KEY",
                   help="env var holding the API key (default TERSE_FLUENCY_API_KEY)")
    f.add_argument("--bars", action="store_true",
                   help="also print a terminal forest plot (accuracy + 95%% CI per model, "
                        "ANSI if a tty)")
    f.add_argument("--html", action="store_true",
                   help="also write a charted HTML forest plot next to --out (inline SVG, "
                        "no JS/CDN); applies to --diff / --diff-soak / --text-diff-eval")
    f.set_defaults(func=_cmd_fluency)

    tn = sub.add_parser("tune", help="one-command lossy tuning: analyze a captured corpus, "
                                     "surface safe-first drop-to-retrieve candidates, write the "
                                     "policy, and optionally verify with a live model")
    tn.add_argument("--corpus", default=DEFAULT_CORPUS)
    tn.add_argument("--out", help="write the generated policy here (suggestions inactive)")
    tn.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                    help="min total savings %% to compress a tool at all (default 5.0)")
    tn.add_argument("--drop-eval", action="store_true",
                    help="also verify the suggested drops with a live tool-calling model "
                         "(needs TERSE_FLUENCY_BASE_URL/_API_KEY/_MODELS or --base-url/--models)")
    tn.add_argument("--trials", type=int, default=1, help="drop-eval trials per question")
    tn.add_argument("--base-url", help="OpenAI-compatible base URL (else $TERSE_FLUENCY_BASE_URL)")
    tn.add_argument("--models", help="comma-separated model ids (else $TERSE_FLUENCY_MODELS)")
    tn.add_argument("--api-key-env", default="TERSE_FLUENCY_API_KEY",
                    help="env var holding the API key (else TERSE_FLUENCY_API_KEY)")
    tn.set_defaults(func=_cmd_tune)

    im = sub.add_parser("install-mcp", help="wrap Claude Code MCP server(s) with the "
                                            "terse proxy")
    im.add_argument("servers", nargs="+", help="mcpServers name(s) to wrap (e.g. runecho)")
    im.add_argument("--policy", required=True, help="path to the JSON policy file")
    im.add_argument("--scope", choices=("user", "project", "local"), default="user",
                    help="MCP config scope (#58): user = ~/.claude.json top-level "
                         "(default), project = a .mcp.json file, local = this repo's "
                         "nested block in ~/.claude.json")
    im.add_argument("--file", help="--scope project: path to the .mcp.json to wrap "
                                   "(default: ./.mcp.json)")
    im.add_argument("--repo-path", help="--scope local: the projects.<repo-path> key "
                                        "to wrap inside ~/.claude.json (default: "
                                        "resolved via `git rev-parse --git-common-dir`, "
                                        "the bare-repo root for claudew/codexw worktrees)")
    im.add_argument("--capture-dir", metavar="DIR",
                    help="also tee raw tool results into this corpus dir for later "
                         "`terse measure`/`verify` (opt-in; never affects forwarding)")
    im.add_argument("--diff", action="store_true",
                    help="bake an explicit `--diff` into the wrapped entry (diffing is "
                         "already the proxy DEFAULT since #75 — only needed to override "
                         'a policy file\'s "diff": false)')
    im.add_argument("--no-diff", action="store_true",
                    help="bake `--no-diff` into the wrapped entry: this server gets "
                         "full results every call, no cross-call diffing")
    im.add_argument("--no-join-blocks", action="store_true",
                    help="bake `--no-join-blocks` into the wrapped entry: this server's "
                         "multi-block results are compressed per block, not joined into "
                         "one record array (joining is the proxy DEFAULT, #116)")
    im.add_argument("--diff-keyframe-interval", type=int, default=None, metavar="K",
                    help="force a full result every K consecutive diffs per tool "
                         "(default 5; 0 disables)")
    im.add_argument("--no-stats", action="store_true",
                    help="bake `--no-stats` into the wrapped entry: no savings-ledger "
                         "records for this server (the ledger is otherwise the proxy "
                         "default — payload-free, read by `terse stats`)")
    im.add_argument("--never-lossy", action="store_true",
                    help="mark the wrapped server(s) as never-lossy: bake them into the "
                         "policy's never_lossy_servers so lossy transforms are structurally "
                         "forbidden on them. Use for a credential/personal store whose name "
                         "the built-in secret-pattern floor can't catch (e.g. a personal KB)")
    im.add_argument("--print", action="store_true",
                    help="dry-run: show the before/after without writing")
    im.set_defaults(func=_cmd_install_mcp)

    um = sub.add_parser("uninstall-mcp", help="restore terse-wrapped MCP server(s) to "
                                              "their original command")
    um.add_argument("servers", nargs="*", help="server name(s) to restore (or use --all)")
    um.add_argument("--all", action="store_true", help="restore every terse-managed server")
    um.add_argument("--scope", choices=("user", "project", "local"), default="user",
                    help="MCP config scope to restore (#58) — see install-mcp --scope")
    um.add_argument("--file", help="--scope project: path to the .mcp.json to restore "
                                   "(default: ./.mcp.json)")
    um.add_argument("--repo-path", help="--scope local: the projects.<repo-path> key "
                                        "to restore (default: resolved via git)")
    um.add_argument("--print", action="store_true",
                    help="dry-run: show what would be restored without writing")
    um.set_defaults(func=_cmd_uninstall_mcp)

    ms = sub.add_parser("mcp-status", help="list terse-wrapped MCP servers across all "
                                           "three scopes (user/project/local) — "
                                           "read-only, writes nothing")
    ms.add_argument("--file", help="project scope: path to the .mcp.json to check "
                                   "(default: ./.mcp.json)")
    ms.add_argument("--repo-path", help="local scope: the projects.<repo-path> key to "
                                        "check (default: resolved via git; silently "
                                        "skipped if not in a git repo)")
    ms.add_argument("--json", action="store_true",
                    help="emit the rows as JSON instead of the text report "
                         "(scriptable/CI-checkable; [] when no servers)")
    ms.set_defaults(func=_cmd_mcp_status)

    vf = sub.add_parser("verify", help="self-contained verification report: lossless gate "
                                       "+ token savings, and how to verify the rest")
    vf.add_argument("--corpus", help="captured-traffic corpus dir (default: a bundled "
                                     "deterministic sample, so it runs with zero setup)")
    vf.add_argument("--out", default="reports/verify-report.md")
    vf.add_argument("--html", action="store_true",
                    help="also write a charted HTML report next to --out (inline SVG, no JS/CDN)")
    vf.add_argument("--bars", action="store_true",
                    help="also print terminal bar charts for the savings sections (ANSI if a tty)")
    vf.add_argument("--json", action="store_true",
                    help="emit the gate verdict + savings as JSON to stdout instead of the "
                         "report (scriptable / CI-checkable; no file/html/bars written)")
    vf.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
