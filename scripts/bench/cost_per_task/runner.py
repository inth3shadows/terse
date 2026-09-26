"""Task runner for the end-to-end cost-per-successful-task harness.

Runs task x arm x rep in randomized order, ONE `claude -p` session per
combination, and appends ONE JSONL row per run IMMEDIATELY -- never held for
the end, so a crash partway through a batch still leaves every completed run
usable. Reuses `ab_session.SessionStats` (loaded by file path -- scripts/bench
is not an installable package) for turns/tool-calls/usage accounting, so the
token math here is identical to the existing A/B harness rather than
reinvented.

See ~/.claude/plans/terse-cost-per-task-e2e.md for the protocol this
implements.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
AB_SESSION_PATH = HERE.parent / "ab_session.py"

sys.path.insert(0, str(HERE))
import arms as arms_mod  # noqa: E402
import checkers  # noqa: E402
import cost_model  # noqa: E402
import mcp_share as mcp_share_mod  # noqa: E402

# Infra-error categories (plan blocker #2). A row tagged with one of these is
# an infrastructure failure, not a task failure: it must be EXCLUDED from
# cost-per-success analysis (a rate limit says nothing about the task) and
# listed for re-run, never silently scored as a task loss.
INFRA_TIMEOUT = "timeout"
INFRA_RATE_LIMIT = "rate_limit"
INFRA_NONZERO_NO_TURN = "nonzero_exit_no_turn"
INFRA_UNPARSEABLE = "unparseable_output"
INFRA_TRANSCRIPT_MISSING = "transcript_missing"
# The CLI's own JSON result said is_error=true but it isn't recognizably a
# rate limit/API-error (see `classify_cli_error`) -- still infra, not a task
# result: `is_error` means the turn didn't complete normally, so scoring it
# as a task failure would blame the model for something that was never its
# answer (plan Partial #2: "detect API or rate-limit errors from the CLI
# JSON is_error/subtype fields as well as stderr").
INFRA_API_ERROR = "api_error"

# Per-category --tools value (plan F2): the exact set of BUILT-IN tools
# available to the session at all, identical across every arm for a given
# task. This is deliberately a SEPARATE flag from --allowedTools: arms B/C
# load the operator's real settings.json (setting_sources=None, "the user's
# setup" the plan calls for), whose ~191 permission rules include broad
# built-in allowances (Bash cat/rg/curl/ssh, bare Read) that --allowedTools
# alone does not override -- --allowedTools only ADDS permission rules, it
# never shrinks the built-in tool SET a settings file already granted.
# --tools does shrink it, identically for A/B/C. Reads by the tools it does grant are
# further fenced by `blockReadsOutsideWorkingDirectories` (build_settings_file).
CATEGORY_BUILTIN_TOOLS = {
    # kb tasks are answerable ONLY via the kb MCP tools. `Read` is present so the
    # session can open its OWN offloaded tool-result file (see offload_read_rule): Claude
    # Code writes a large MCP result to <projects>/<slug>/<session>/tool-results/ instead
    # of the context, and without Read an arm whose result was large (B, uncompressed)
    # fails where its compressed twin (C) does not -- the 2026-09-26 smoke-run bias.
    # Other reads are fenced by `blockReadsOutsideWorkingDirectories` (build_settings_file),
    # which still admits ~/.claude/CLAUDE.md and the skills/plugins/rules/agents/commands
    # folders; neither kb answer is in them (checked 2026-09-26). Grep too: an offloaded result can exceed Read's 25k-token cap (41,257 tokens for
    # list_nodes in the 2026-09-26 smoke), and the operator's real sessions would grep it
    # rather than page through it. Identical for every arm.
    "kb": "Read,Grep",
    "code": "Read,Grep,Glob",    # read-only repo exploration, no execution
    "control": "Read,Edit,Bash", # the one task that must actually edit + run pytest
}


CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def offload_read_rule(session_id: str, projects_dir: Path = CLAUDE_PROJECTS_DIR) -> str:
    """The one Read permission rule every arm gets on every task: this session's OWN
    offloaded tool results, and nothing else. Scoped by the session id (a fresh uuid per
    run), so a run can never read an EARLIER run's offloaded result -- which, for a kb
    task, would contain the answer. The project slug is globbed (`*`) rather than
    recomputed: the session id alone is unique, and Claude Code's slug rule is not ours
    to re-derive. `//` marks an absolute path in a permission rule."""
    return f"Read(/{projects_dir}/*/{session_id}/tool-results/**)"


def resolve_builtin_tools(task: dict) -> str:
    """This task's --tools value (plan F2), from its `category` -- the SAME
    value for every arm (A/B/C), never per-arm. Raises KeyError for an
    unrecognized category rather than silently defaulting to "" or
    "default": a category this harness doesn't know about must not run with
    an unreviewed built-in tool surface."""
    category = task.get("category")
    if category not in CATEGORY_BUILTIN_TOOLS:
        raise KeyError(
            f"task {task.get('id')!r} has category {category!r}, which has no "
            f"entry in CATEGORY_BUILTIN_TOOLS -- add one rather than guessing "
            f"a --tools value for it")
    return CATEGORY_BUILTIN_TOOLS[category]


def _load_ab_session():
    """Load scripts/bench/ab_session.py by path -- it is a standalone script,
    not part of an installable package, so a normal `import` can't reach it
    from here regardless of cwd."""
    spec = importlib.util.spec_from_file_location("ab_session", AB_SESSION_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ab_session = _load_ab_session()


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                          capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


# --------------------------------------------------------------------- claude -p

# Applied to EVERY arm, uniformly: no subagents (a Task-spawned subagent's
# own tool traffic and cost would otherwise be invisible to the per-task
# accounting this harness does -- cost_model.py belt-and-suspenders folds in
# a subagent transcript if one appears anyway) and no live network access
# (a benchmark run must be reproducible from the repo + MCP servers alone,
# not from whatever a web search returns that day). Plan blockers #3a, #5.
DEFAULT_DISALLOWED_TOOLS = ("Task", "WebFetch", "WebSearch")


def build_claude_command(*, prompt: str, model: str, mcp_config_path: Path,
                          session_id: str, setting_sources: str | None,
                          allowed_tools: list[str] | None = None,
                          disallowed_tools: tuple[str, ...] | None = DEFAULT_DISALLOWED_TOOLS,
                          effort: str | None = None,
                          settings_path: Path | None = None,
                          safe_mode: bool = False,
                          permission_mode: str = "dontAsk",
                          tools: str | None = None) -> list[str]:
    """The argv for one `claude -p` invocation. Pure -- no subprocess call --
    so it is directly unit-testable without ever launching Claude.

    `--permission-mode dontAsk` (never `bypassPermissions`) plus a per-task
    `--allowedTools` allowlist and a uniform `--disallowedTools` denylist is
    the harness's actual permission boundary now (plan blocker #5):
    `bypassPermissions` gave every arm, including a compromised or
    misbehaving model, unrestricted tool access for the whole run --
    `dontAsk` auto-denies anything not on the allowlist instead of prompting
    (there is no human to answer a prompt in an unattended `-p` run) while
    still enforcing one.

    `settings_path` (a generated `--settings <file>` JSON, e.g.
    `{"disableAllHooks": true}`) and `safe_mode` (`--safe-mode`, arm A only
    -- makes "no custom setup" explicit and checkable rather than inferred
    from an empty MCP config) are plan blockers #6 and #7.

    `tools` is this task's `--tools` value (plan F2, see
    `CATEGORY_BUILTIN_TOOLS`/`resolve_builtin_tools`) -- the built-in tool
    SET available at all, independent of and in addition to
    `--allowedTools`'s permission-rule allowlist. Passed whenever it is not
    None, INCLUDING the empty string ("" legitimately means "no built-in
    tools at all", not "flag omitted" -- `if tools is not None` rather than
    `if tools`, or a kb task's deliberate `--tools ""` would silently vanish.
    """
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--strict-mcp-config",
        "--mcp-config", str(mcp_config_path),
        "--session-id", session_id,
        "--permission-mode", permission_mode,
    ]
    if setting_sources is not None:
        cmd += ["--setting-sources", setting_sources]
    if allowed_tools:
        cmd += ["--allowedTools", ",".join(allowed_tools)]
    if disallowed_tools:
        cmd += ["--disallowedTools", ",".join(disallowed_tools)]
    if effort is not None:
        cmd += ["--effort", effort]
    if settings_path is not None:
        cmd += ["--settings", str(settings_path)]
    if safe_mode:
        cmd.append("--safe-mode")
    if tools is not None:
        cmd += ["--tools", tools]
    return cmd


def _run_claude_process(cmd: list[str], *, cwd: Path, env: dict, timeout: float
                         ) -> tuple[int | None, str, str, bool]:
    """Launch `cmd` (a `claude -p` invocation) in its OWN process group
    (`start_new_session=True`, i.e. setsid) so a timeout can kill the WHOLE
    tree -- claude itself, terse's router, and the MCP server children the
    router spawns -- via `os.killpg`, not just claude's own top process
    (plan F5). The previous `subprocess.run(cmd, timeout=...)` only ever
    kills the direct child on TimeoutExpired; everything it spawned (a
    multiproxy router process, which itself holds three more MCP server
    subprocesses open) is orphaned and keeps running for the rest of the
    batch, accumulating one leaked router+3-servers set per timeout.

    Returns (returncode, stdout, stderr, timed_out). `returncode` is None
    only if even the post-kill `communicate()` couldn't reap the process (an
    extremely stuck kernel-level hang) -- callers must treat that the same
    as a timeout, which `timed_out=True` already tells them."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone -- nothing to kill
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # the group kill didn't finish reaping in time -- one more direct
            # kill attempt, then take whatever communicate() gives us (empty
            # strings in the worst case) rather than block the batch forever.
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
        return proc.returncode, stdout, stderr, True


def looks_like_rate_limit(stderr: str) -> bool:
    """Heuristic rate-limit detector over a `claude -p` process's stderr.
    Pure string matching (no exception types available -- the CLI is a
    subprocess, not the SDK) is deliberately loose: a false positive just
    means a genuine task failure gets excluded from analysis and queued for
    re-run, which is the SAFE direction to be wrong in (a false negative
    would instead pollute cost-per-success with an infra hiccup)."""
    low = stderr.lower()
    return any(marker in low for marker in (
        "rate limit", "rate_limit", "429", "overloaded_error", "too many requests"))


def classify_cli_error(cli: dict, stderr: str) -> str | None:
    """Infra-error category for a `claude -p --output-format json` result
    that parsed fine but reports `is_error: true` -- checked from the CLI's
    OWN JSON (`is_error`/`subtype`/`result`) as well as stderr (plan Partial
    #2: "detect API or rate-limit errors from the CLI JSON is_error/subtype
    fields as well as stderr"), since a proxied/sandboxed run's stderr isn't
    guaranteed to carry the same text a rate limit prints directly. Returns
    None if `cli.get("is_error")` is falsy (a normal, non-error result --
    the caller scores it against the task's checker as usual).

    `subtype == "error_during_execution"` covers a mid-session API failure
    (rate limit, 5xx) the CLI still wrapped in a clean JSON result instead
    of surfacing a nonzero exit; a plain rate-limit text match against
    stderr/result covers the rest. Anything else with `is_error: true` is
    still infra (the turn didn't complete normally, which is not the same
    claim as a task failure) but isn't specifically a rate limit --
    INFRA_API_ERROR, not INFRA_RATE_LIMIT, so the two are distinguishable in
    the re-run list. `subtype == "error_max_turns"` is deliberately NOT
    classified as infra here -- the model genuinely ran out of turns on the
    task, which is a task result, not an infrastructure failure."""
    if not cli.get("is_error"):
        return None
    subtype = str(cli.get("subtype") or "")
    result_text = str(cli.get("result") or "")
    if subtype == "error_max_turns":
        return None
    if subtype == "error_during_execution" or looks_like_rate_limit(stderr) \
            or looks_like_rate_limit(result_text) or looks_like_rate_limit(subtype):
        return INFRA_RATE_LIMIT
    return INFRA_API_ERROR


def stderr_digest(stderr: str) -> dict:
    """Length + sha256 of `stderr`, never the raw text -- a benchmark result
    row is not the place to store a subprocess's raw stderr (plan Low
    priority item): it can be arbitrarily large and there is no reason to
    keep it once its length/hash establish whether two runs failed
    identically."""
    import hashlib
    return {"stderr_len": len(stderr), "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest()}


def parse_cli_json(raw_stdout: str) -> dict:
    """Parse `claude -p --output-format json`'s single JSON object. Raises
    ValueError (with the raw output attached, truncated) on malformed
    output, rather than letting a bare JSONDecodeError obscure what actually
    came back (a crash mid-stream, an auth prompt eating stdout, etc)."""
    try:
        return json.loads(raw_stdout)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"claude -p did not return valid JSON: {e}. First 500 chars: "
            f"{raw_stdout[:500]!r}") from e


def find_transcript(session_id: str,
                     search_root: Path | None = None) -> Path | None:
    """Locate `<session_id>.jsonl` under `search_root` (default
    ~/.claude/projects), without reimplementing Claude Code's cwd -> escaped
    directory-name mapping -- we control the session id, so a glob for it is
    exact and doesn't need to know the escaping rule at all. None if not
    found (e.g. the run never got far enough to write one)."""
    root = search_root or (Path.home() / ".claude" / "projects")
    matches = list(root.glob(f"*/{session_id}.jsonl"))
    return matches[0] if matches else None


# --------------------------------------------------------------------- workdirs

def make_workdir(task: dict, base_dir: Path, repo: Path, *,
                  pinned_commit: str | None = None) -> Path:
    """A fresh working directory for one run: a detached `git worktree` of
    `pinned_commit` for a `workdir: "repo"` task (so codegraph/runecho see
    the real repo tree, and arm A's built-in Read/Grep have the same shot at
    the answer), or a plain temp dir -- populated from the task's `fixture`
    files, if any -- for `workdir: "scratch"`.

    `pinned_commit` is REQUIRED for a 'repo' task -- always a fixed SHA from
    tasks.json (`pinned_worktree_commit`), never "whatever HEAD is right
    now". This harness's own tasks.json/checkers.py are untracked today, so
    a HEAD-relative worktree happens to exclude them; pinning to a SHA that
    predates the harness keeps that true even after the harness itself is
    committed to this branch, so the answer key can never end up inside a
    worktree a coding agent's Read tool can reach (plan blocker #5)."""
    if task.get("workdir") == "repo":
        if not pinned_commit:
            raise ValueError(
                "workdir='repo' task requires pinned_commit (tasks.json's "
                "'pinned_worktree_commit') -- refusing to create a worktree off "
                "a moving HEAD that could contain the harness's own answer key")
        wt = base_dir / f"wt-{uuid.uuid4().hex[:8]}"
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), pinned_commit],
                        cwd=repo, check=True, capture_output=True, text=True)
        return wt
    workdir = Path(tempfile.mkdtemp(dir=base_dir))
    for fname, fcontent in (task.get("fixture") or {}).items():
        (workdir / fname).write_text(fcontent)
    return workdir


def remove_workdir(task: dict, workdir: Path, repo: Path) -> None:
    if task.get("workdir") == "repo":
        subprocess.run(["git", "worktree", "remove", "--force", str(workdir)],
                        cwd=repo, check=False, capture_output=True, text=True)
    else:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------- shared codegraph worktree

# Plan F1: a fresh `git worktree` (what every 'repo' task previously got, one
# per rep) never carries `.codegraph/` -- it's gitignored, so a worktree
# checkout simply doesn't have it. `codegraph-gate` (personal_projects/
# tooling's bin/codegraph-gate, which the live peers file's 'codegraph'
# downstream actually launches instead of the real `codegraph` binary --
# see its own docstring) walks up from $PWD looking for a real, non-empty
# `.codegraph/` and, finding none, execs a zero-tool null stub instead of
# the real MCP server. B and C's code tasks were therefore silently getting
# ZERO codegraph tools every single rep. Fix: ONE shared worktree, indexed
# ONCE up front, reused by every code-task rep (code tasks only ask
# questions about the repo -- they never edit it, so sharing is safe); the
# control task, which DOES edit a file, keeps its own fresh temp dir via
# make_workdir/remove_workdir.

def make_shared_code_worktree(base_dir: Path, repo: Path, pinned_commit: str) -> Path:
    """ONE git worktree, pinned to `pinned_commit`, at a fixed path under
    `base_dir` -- created once per harness invocation and reused by every
    'repo' task rep, never removed/recreated per rep like `make_workdir`."""
    wt = base_dir / "shared-code-worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), pinned_commit],
                    cwd=repo, check=True, capture_output=True, text=True)
    return wt


def index_codegraph(worktree: Path, *, codegraph_bin: str = "codegraph") -> None:
    """Build `worktree`'s codegraph index ONCE, up front, using the REAL
    `codegraph` binary directly -- never `codegraph-gate` (which would just
    see the not-yet-indexed worktree and exec the null stub; indexing is
    exactly what makes the gate's LATER decision, made once claude actually
    launches the MCP server, come out real). Raises CalledProcessError if
    `codegraph init` fails -- refuses to silently run the pilot against an
    unindexed worktree."""
    subprocess.run([codegraph_bin, "init", str(worktree)], check=True,
                    capture_output=True, text=True)


def _sqlite_has_tables(db_path: Path) -> bool:
    """Mirrors codegraph-gate's own `has_tables()` bash function exactly: a
    schema-less codegraph.db (zero tables -- what a failed/interrupted
    `init` leaves behind) is rejected even though it exists, since upstream
    codegraph would otherwise treat its mere presence as 'indexed' and
    suppress a real sub-project scan underneath it (codegraph#1895, per the
    gate's own docstring). Unreadable -> True (fail safe, same as the gate's
    bare `except: sys.exit(0)`, which execs the REAL server rather than risk
    wrongly serving the null stub)."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            n = conn.execute(
                "select count(*) from sqlite_master where type='table'").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return True
    return n > 0


def codegraph_gate_would_serve_real(cwd: Path) -> bool:
    """Mirrors codegraph-gate's own upward-walk decision in Python, WITHOUT
    shelling out to the gate script itself (on its real-index branch it
    execs a long-running MCP server that blocks on stdin -- not something a
    preflight check can safely invoke and wait on). True only if an
    ancestor directory (never $HOME itself -- codegraph's OWN app-state dir,
    not a project) has a `.codegraph/` whose `codegraph.db` either doesn't
    exist yet (a fresh, not-yet-indexed project -- still 'real' to the gate)
    or has at least one table (see `_sqlite_has_tables`). False means the
    gate would serve the zero-tool null stub for this cwd.

    The climb stops once it reaches `home` (after performing home's own,
    always-skipped check) rather than continuing to the filesystem root like
    the bash gate literally does -- in every real invocation $HOME sits near
    the top of the tree with nothing project-shaped above it, so this is
    behaviorally identical for real use, and it keeps this function
    self-contained (it never inspects whatever happens to exist further up
    the REAL machine's filesystem above a caller-chosen cwd/home, which
    matters for testing this against a fake home nested under a real one)."""
    home = Path.home().resolve()
    p = cwd.resolve()
    while True:
        d = p / ".codegraph"
        if d.is_dir() and p != home:
            db = d / "codegraph.db"
            if not db.exists() or _sqlite_has_tables(db):
                return True
        if p == home or p.parent == p:
            return False
        p = p.parent


def cleanup_runecho_enrollment(worktree: Path) -> None:
    """Best-effort: runecho-ir auto-enrolls a repo under its cwd basename on
    first `structure`/`locate` call (a known gotcha, not something this
    harness asks for -- see the runecho-enrollment-gotcha memory note). The
    shared worktree fixes most of plan F3 (one enrollment per BATCH instead
    of one per rep), but that one enrollment still needs cleaning up so a
    benchmark worktree doesn't linger in the operator's real runecho index
    after the run. Checks `runecho-ir repo list` first and only removes an
    entry that actually matches this worktree's basename. Never raises --
    this is cleanup, not part of the harness's correctness contract, and a
    missing `runecho-ir` binary (or a locked/unavailable index) must not
    fail the whole run over a tidiness step."""
    name = worktree.name
    try:
        listed = subprocess.run(["runecho-ir", "repo", "list"], capture_output=True,
                                 text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return
    if listed.returncode != 0 or name not in listed.stdout:
        return
    try:
        subprocess.run(["runecho-ir", "repo", "rm", name], capture_output=True,
                        text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


# --------------------------------------------------------------------- one run

def run_one(*, task: dict, arm: str, rep: int, model: str, arm_config_path: Path,
            workdir: Path, setting_sources: str | None, out_fh,
            transcript_search_root: Path | None = None,
            allowed_tools: list[str] | None = None,
            effort: str | None = None,
            settings_path: Path | None = None,
            safe_mode: bool = False,
            builtin_tools: str | None = None) -> dict:
    """Run ONE `claude -p` session for (task, arm, rep), score it against the
    task's success check, and append its JSONL row to `out_fh` immediately.
    Returns the row dict too.

    The transcript is ALWAYS loaded when it exists on disk -- even after a
    timeout, a non-zero exit, or unparseable stdout -- because a cut-off run
    still spent billable tokens; throwing that away (the previous behavior)
    meant a crashed run's cost silently vanished instead of being counted
    and excluded. `infra_error` (see the INFRA_* constants) marks a row as an
    INFRASTRUCTURE failure -- timeout, rate limit, a non-zero exit with no
    recorded model turn, or unparseable CLI output -- distinct from a task
    the model genuinely failed; analysis.py excludes infra rows from
    cost-per-success and the runner CLI lists them for re-run.

    `allowed_tools` is this (task, arm)'s permission allowlist (tasks.json's
    `tools.<ARM>`); `effort`/`settings_path`/`safe_mode` are passed straight
    through to `build_claude_command` (plan blockers #5, #6, #7)."""
    session_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    allowed_tools = [*(allowed_tools or []), offload_read_rule(session_id)]
    cmd = build_claude_command(prompt=task["prompt"], model=model,
                                mcp_config_path=arm_config_path, session_id=session_id,
                                setting_sources=setting_sources, allowed_tools=allowed_tools,
                                effort=effort, settings_path=settings_path, safe_mode=safe_mode,
                                tools=builtin_tools)
    row: dict = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model, "arm": arm, "task_id": task["id"], "rep": rep,
        "session_id": session_id, "workdir": str(workdir),
        "arm_config_path": str(arm_config_path),
        "effort": effort, "allowed_tools": allowed_tools,
    }

    # CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 on every arm, uniformly, so a
    # benchmark run (repeated dozens of times) never writes real MEMORY.md
    # entries as a side effect -- the same "no persistent side effects" rule
    # this harness applies to kb writes and git pushes, not a per-arm
    # confound (it is applied identically everywhere).
    full_env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}

    def _finish(**fields) -> dict:
        row.update(fields)
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        return row

    infra_error: str | None = None
    error: str | None = None
    timeout_s = task.get("timeout", 300)
    returncode, stdout, stderr, timed_out = _run_claude_process(
        cmd, cwd=workdir, env=full_env, timeout=timeout_s)
    if timed_out:
        infra_error = INFRA_TIMEOUT
        error = f"timeout after {timeout_s}s"

    # ALWAYS attempt to load a transcript, regardless of what happened above.
    transcript = find_transcript(session_id, search_root=transcript_search_root)
    subagent_paths = cost_model.find_subagent_transcripts(transcript) if transcript else []
    all_transcripts = ([transcript] if transcript else []) + subagent_paths
    cost = cost_model.compute_ttl_weighted(all_transcripts, ab_session) if all_transcripts else None
    share = mcp_share_mod.mcp_share(transcript) if transcript else None
    # Kept for tool-call/mcp-call counting only (main transcript, not
    # subagents -- a subagent's own tool calls are not this task's turns).
    # `turns` is NOT read from `stats` (ab_session.SessionStats counts the
    # CLI's synthetic all-zero-usage error record as a real turn) -- use
    # cost_model.count_real_turns instead, which excludes it (plan
    # Partial #2: "don't count the fake assistant record as a turn").
    stats = ab_session.SessionStats(transcript) if transcript else None
    real_turns = cost_model.count_real_turns([transcript]) if transcript else 0

    cli: dict = {}
    if infra_error is None and returncode is not None:
        if returncode != 0:
            error = f"claude exited {returncode}"
            if looks_like_rate_limit(stderr):
                infra_error = INFRA_RATE_LIMIT
            elif real_turns == 0:
                infra_error = INFRA_NONZERO_NO_TURN
        else:
            try:
                cli = parse_cli_json(stdout)
            except ValueError as e:
                infra_error = INFRA_UNPARSEABLE
                error = str(e)
            else:
                cli_infra = classify_cli_error(cli, stderr)
                if cli_infra is not None:
                    infra_error = cli_infra
                    error = (f"CLI result is_error=True (subtype="
                             f"{cli.get('subtype')!r}): {cli.get('result')!r}")

    if infra_error is None and transcript is None:
        infra_error = INFRA_TRANSCRIPT_MISSING
        error = error or "transcript not found"

    cli_cost_usd = cli.get("total_cost_usd", cli.get("cost_usd")) if cli else None
    weighted_cost = cost["weighted_cost"] if cost else None
    computed_usd = cost_model.estimate_usd_cost(model, cost) if cost else None
    tool_calls_total = sum(stats.tool_calls.values()) if stats else None

    common_fields = dict(
        infra_error=infra_error,
        transcript_path=str(transcript) if transcript else None,
        subagent_transcripts=[str(p) for p in subagent_paths],
        has_subagent_cost=bool(subagent_paths),
        cli_cost_usd=cli_cost_usd,
        cli_duration_ms=cli.get("duration_ms"),
        turns=(real_turns if transcript else None),
        tool_calls_total=tool_calls_total,
        mcp_tool_calls=(stats.total_mcp if stats else None),
        terse_retrieve_calls=(stats.retrieve_calls if stats else None),
        input_tokens=(cost["input_tokens"] if cost else None),
        cache_write_5m_tokens=(cost["cache_write_5m_tokens"] if cost else None),
        cache_write_1h_tokens=(cost["cache_write_1h_tokens"] if cost else None),
        cache_read_tokens=(cost["cache_read_tokens"] if cost else None),
        output_tokens=(cost["output_tokens"] if cost else None),
        first_turn_cache_read_tokens=(cost["first_turn_cache_read_tokens"] if cost else None),
        first_turn_cache_write_tokens=(cost["first_turn_cache_write_tokens"] if cost else None),
        resolved_model=(cost["resolved_model"] if cost else None),
        cli_version=(cost["cli_version"] if cost else None),
        weighted_cost=weighted_cost,
        computed_cost_usd=computed_usd,
        cost_gap=cost_model.cost_gap_flag(computed_usd, cli_cost_usd),
        mcp_share_approx=share,
        **stderr_digest(stderr),
    )

    if infra_error is not None:
        return _finish(success=False, answer=None, error=error, **common_fields)

    answer = cli.get("result") or ""
    success, detail = checkers.run_check(task, answer, workdir)
    # Plan Partial #8: on a kb or code task, a PASS also requires at least
    # one real tool call. Both categories are answerable ONLY by using a
    # tool (kb data isn't in the model's training; the repo-internal
    # constant/function names these tasks ask for are chosen specifically
    # so they can't be guessed -- see tasks.json's per-task 'why') -- a
    # correct answer with zero tool calls means the checker got lucky on a
    # guess, not that the arm's tool surface earned the answer, and must not
    # count as a win for that arm.
    if success and task.get("category") in ("kb", "code") and not tool_calls_total:
        success = False
        detail = (f"{detail} -- REJECTED: tool_calls_total={tool_calls_total!r} on a "
                  f"{task.get('category')!r} task (plan Partial #8: a pass requires "
                  f"actual tool use, not a guessed/memorized answer)")
    return _finish(success=success, check_detail=detail, answer=answer, error=None,
                    **common_fields)


# --------------------------------------------------------------------- CLI

def load_tasks_doc(path: Path) -> dict:
    return json.loads(path.read_text())


def load_tasks(path: Path) -> list[dict]:
    return load_tasks_doc(path)["tasks"]


def resolve_allowed_tools(task: dict, arm: str) -> list[str]:
    """This (task, arm)'s --allowedTools list, from tasks.json's per-task
    `tools.<ARM>` field. [] (not None) when absent -- an empty allowlist
    plus the uniform --disallowedTools denylist is itself a meaningful,
    intentional permission state (e.g. arm A on a kb task, which has no
    tool that could answer it), not a missing-config bug."""
    return list((task.get("tools") or {}).get(arm, []))


def build_settings_file(out_dir: Path, name: str = "settings-no-hooks.json") -> Path:
    """The harness-wide `--settings` file, applied to EVERY arm uniformly:
    `{"disableAllHooks": true, "permissions": {"blockReadsOutsideWorkingDirectories": true}}`
    (plan blocker #6; the read fence per the Opus review of 2026-09-26, which neutralises the
    operator settings' blanket Read allow rule for B/C). Verified against the
    installed claude CLI's own settings schema strings (v2.1.283,
    2026-09-26): `disableAllHooks` is a real top-level settings key, honored
    from a user settings file OR a `--settings` file, and also gates
    plugin-supplied hooks (paired with `allowManagedHooksOnly` in the CLI's
    own effective-settings computation) -- so this closes kb session writes,
    handoff consumption, project-index-init, and budget/context injection
    hooks for every arm. It does NOT disable CLAUDE.md, output style, or MCP
    server loading -- B/C still get "the user's setup" the plan calls for."""
    doc: dict = {"disableAllHooks": True,
                 "permissions": {"blockReadsOutsideWorkingDirectories": True}}
    return arms_mod._write_restricted(out_dir / name, doc)


def _refuse_out_inside_repo(out_path: Path, repo: Path) -> None:
    """Refuse an --out path inside the repo (plan Low-priority item): a
    results file is run output, not something that belongs alongside the
    harness's own source in a commit or a stray `git add`."""
    try:
        out_path.resolve().relative_to(repo.resolve())
    except ValueError:
        return
    raise SystemExit(
        f"--out {out_path} is inside the repo ({repo}) -- pass a path outside "
        f"the repo (e.g. the scratchpad) so run results never land in a commit")


def warmup_missed_mcp(row: dict, allowed_tools: list[str]) -> bool:
    """True when a warm-up that was allowed MCP tools made zero MCP calls --
    in the pilot that meant the MCP servers had not finished loading (the C
    warm-up answered "I don't have access to kb.read.* tools"), so it cached a
    prefix WITHOUT their tool definitions and the first measured run of that
    combo still paid the full cold write. Arm C only, by luck of ordering.
    Over-inclusive by design: a model that simply chose Grep also makes zero
    MCP calls, and that costs one discarded rerun -- cheaper than a cold
    measured rep. An infra-error row is never a miss (the retry would not fix
    a rate limit or timeout)."""
    return (any(t.startswith("mcp__") for t in allowed_tools)
            and row.get("mcp_tool_calls") == 0
            and not row.get("infra_error"))


def plan_run_combos(tasks: list[dict], arm_list: list[str], reps: int, *,
                     seed: int | None = None
                     ) -> tuple[list[tuple[dict, str, int]], list[tuple[dict, str, int]]]:
    """(warmup_combos, real_combos) -- plan blocker #9 / Partial #9: ONE
    discarded warm-up run per (task, arm) (per model -- `main` runs one model
    per invocation) ALWAYS happens, not opt-in behind a flag a batch could be
    run without by omission. Warm-up combos use rep=0 and are never merged
    into the shuffled real combos: their only purpose is to run and be thrown
    away (to a separate output file the caller never hands to analysis.py)
    before the measured reps begin, to absorb the cold-start cache write.
    Per (task, arm), not per arm: each combo has its own cached prompt prefix
    (the per-task tool allowlist differs), so a warm-up on tasks[0] alone left
    6 of 45 pilot runs cold -- whichever arm hit a combo first paid the full
    first-turn write."""
    warmup_combos = [(t, arm, 0) for t in tasks for arm in arm_list]
    real_combos = [(t, arm, rep) for t in tasks for arm in arm_list for rep in range(1, reps + 1)]
    random.Random(seed).shuffle(real_combos)
    return warmup_combos, real_combos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=HERE / "tasks.json")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--c-policy", type=Path, default=None,
                     help="arm C only: pin THIS policy file's content instead of the live "
                          "policy (a C-prime variant; the live file is never written)")
    ap.add_argument("--c-terse", default=None,
                     help="arm C only: launch the router with this terse binary instead of "
                          "the live one (e.g. a worktree's .venv/bin/terse)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                     help="a FULL model ID, not an alias -- so the resolved model actually "
                          "run is unambiguous and comparable across sessions/reps")
    ap.add_argument("--effort", default="high",
                     help="--effort value passed IDENTICALLY to every arm (A, B, C) -- "
                          "the same effort setting is part of holding everything but the "
                          "treatment constant")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task-id", action="append",
                     help="restrict to this task id (repeatable); default: all")
    ap.add_argument("--config-dir", type=Path, required=True,
                     help="scratchpad dir to write arm MCP configs + per-run "
                          "workdirs into (never the repo)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--print-config", action="store_true",
                     help="print each arm's resolved argv and the generated --settings "
                          "content (no secrets), then exit without running anything -- "
                          "for the smoke run to verify permission-mode/tools/effort/hooks "
                          "before a real batch")
    args = ap.parse_args(argv)

    repo = repo_root()
    if not args.print_config:
        _refuse_out_inside_repo(args.out, repo)

    tasks_doc = load_tasks_doc(args.tasks)
    tasks = tasks_doc["tasks"]
    pinned_commit = tasks_doc.get("pinned_worktree_commit")
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            sys.exit(f"no such task id(s): {sorted(missing)}")
    if not tasks:
        sys.exit("no tasks selected")
    if any(t.get("workdir") == "repo" for t in tasks) and not pinned_commit:
        sys.exit("tasks.json has a workdir='repo' task but no top-level "
                  "'pinned_worktree_commit' -- refusing to guess a commit")

    arm_list = args.arms.split(",")

    args.config_dir.mkdir(parents=True, exist_ok=True)
    c_overrides = {"c_policy": args.c_policy, "c_terse": args.c_terse}
    arm_paths = {arm: arms_mod.write_arm_config(arm, args.config_dir,
                                                **(c_overrides if arm == "C" else {}))
                 for arm in arm_list}
    if args.c_policy or args.c_terse:
        # Rows carry arm "C" either way; keep a variant's --out/--config-dir separate.
        print(f"arm C is a VARIANT: policy={args.c_policy or 'live'} "
              f"terse={args.c_terse or 'live'}", file=sys.stderr)
    # Per-arm settings (Opus review 2026-09-26): B/C load the operator's real settings
    # (CLAUDE.md, rules, output style, skills, agents: "the user's setup"). Its blanket Read
    # allow rule is neutralised for EVERY arm by `blockReadsOutsideWorkingDirectories`, which
    # the CLI checks before allow rules and which "true in any settings source wins". The CLI
    # still lets a session read its own offloaded tool results under that setting.
    settings_paths = {"A": build_settings_file(args.config_dir),
                      "B": build_settings_file(args.config_dir, name="settings-setup.json"),
                      "C": build_settings_file(args.config_dir, name="settings-setup.json")}
    # Arm A loads no user/project/local settings (and therefore no CLAUDE.md)
    # and gets --safe-mode on top -- making "no custom setup" explicit and
    # checkable rather than inferred from an empty MCP config (plan blocker
    # #7). Arms B/C get the default (None = omit the flag => Claude Code's
    # own default, which loads the operator's real settings.json + CLAUDE.md
    # -- "the user's setup" the plan calls for).
    setting_sources = {"A": "", "B": None, "C": None}
    safe_mode = {"A": True, "B": False, "C": False}

    if args.print_config:
        for arm in arm_list:
            for task in tasks:
                cmd = build_claude_command(
                    prompt="<PROMPT REDACTED>", model=args.model,
                    mcp_config_path=arm_paths[arm], session_id="<SESSION-ID>",
                    setting_sources=setting_sources[arm],
                    allowed_tools=resolve_allowed_tools(task, arm),
                    effort=args.effort, settings_path=settings_paths[arm],
                    safe_mode=safe_mode[arm], tools=resolve_builtin_tools(task))
                print(f"[{arm}] {task['id']}:")
                print("  " + " ".join(cmd))
        for a in sorted(set(arm_list)):
            print(f"\nsettings file [{a}] ({settings_paths[a]}):")
            print("  " + settings_paths[a].read_text().strip())
        return 0

    # Plan F1: ONE shared, pinned worktree for every 'repo' task rep across
    # every arm -- indexed with codegraph ONCE, up front -- instead of a
    # fresh (never-indexed) worktree per rep, which made codegraph-gate
    # serve zero tools to B/C on every code-task rep. A preflight check
    # fails the whole run (before any claude -p session, real or warm-up)
    # if the gate would STILL serve the null server for this worktree even
    # after indexing.
    shared_code_wt: Path | None = None
    repo_tasks_present = any(t.get("workdir") == "repo" for t in tasks)
    if repo_tasks_present:
        shared_code_wt = make_shared_code_worktree(args.config_dir, repo, pinned_commit)
        try:
            index_codegraph(shared_code_wt)
        except subprocess.CalledProcessError as e:
            sys.exit(f"preflight FAILED: `codegraph init {shared_code_wt}` failed "
                      f"({e}) -- refusing to run 'repo' tasks against an unindexed "
                      f"shared worktree")
        if {"B", "C"} & set(arm_list) and not codegraph_gate_would_serve_real(shared_code_wt):
            sys.exit(f"preflight FAILED: codegraph-gate would still serve the null "
                      f"(zero-tool) server for {shared_code_wt} even after indexing -- "
                      f"refusing to run B/C code tasks that would silently get no "
                      f"codegraph_explore tool")

    combos_by_kind = plan_run_combos(tasks, arm_list, args.reps, seed=args.seed)
    warmup_combos, combos = combos_by_kind

    def _run_batch(batch: list[tuple[dict, str, int]], out_fh, *, is_warmup: bool) -> list:
        failures = []
        for task, arm, rep in batch:
            if task.get("workdir") == "repo" and shared_code_wt is not None:
                workdir, owns_workdir = shared_code_wt, False
            else:
                workdir = make_workdir(task, args.config_dir, repo, pinned_commit=pinned_commit)
                owns_workdir = True
            try:
                allowed = resolve_allowed_tools(task, arm)
                # A warm-up whose MCP tools never loaded warmed the wrong prefix:
                # retry it once (warm-up rows are discarded, so a retry costs
                # only tokens, never a measured rep).
                for _attempt in range(2 if is_warmup else 1):
                    row = run_one(task=task, arm=arm, rep=rep, model=args.model,
                                   arm_config_path=arm_paths[arm], workdir=workdir,
                                   setting_sources=setting_sources[arm], out_fh=out_fh,
                                   allowed_tools=allowed,
                                   effort=args.effort, settings_path=settings_paths[arm],
                                   safe_mode=safe_mode[arm], builtin_tools=resolve_builtin_tools(task))
                    if not (is_warmup and warmup_missed_mcp(row, allowed)):
                        break
                    print(f"[{arm}] WARMUP {task['id']}: 0 MCP calls (tools may not have loaded)"
                          f"{' -- retrying once' if _attempt == 0 else ' -- still 0 after retry'}",
                          file=sys.stderr)
                if row.get("infra_error"):
                    status = f"INFRA ({row['infra_error']}: {row.get('error')})"
                    failures.append((task["id"], arm, rep))
                else:
                    status = "PASS" if row["success"] else f"FAIL ({row.get('check_detail')})"
                tag = "WARMUP " if is_warmup else ""
                print(f"[{arm}] {tag}{task['id']} rep{rep}: {status}", file=sys.stderr)
            finally:
                if owns_workdir:
                    remove_workdir(task, workdir, repo)
        return failures

    try:
        # Warm-up ALWAYS runs now (plan Partial #9) -- there is no flag to
        # skip it.
        warmup_path = args.config_dir / "warmup.jsonl"
        with warmup_path.open("a") as warmup_fh:
            _run_batch(warmup_combos, warmup_fh, is_warmup=True)
        print(f"warm-up rows written to {warmup_path} -- NOT included in --out, "
              f"never pass this file to analysis.py", file=sys.stderr)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as out_fh:
            infra_failures = _run_batch(combos, out_fh, is_warmup=False)
    finally:
        # Plan F1/F3 cleanup: the shared worktree (and whatever runecho
        # enrollment it picked up) is only removed once, at the very end of
        # the whole batch -- never per rep, since every 'repo' task rep
        # reused the SAME worktree.
        if shared_code_wt is not None:
            cleanup_runecho_enrollment(shared_code_wt)
            remove_workdir({"workdir": "repo"}, shared_code_wt, repo)

    if infra_failures:
        print(f"\n{len(infra_failures)} infra failure(s) -- EXCLUDED from cost analysis, "
              f"re-run with:", file=sys.stderr)
        for task_id, arm, rep in infra_failures:
            print(f"  --task-id {task_id} --arms {arm}  (rep {rep})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
