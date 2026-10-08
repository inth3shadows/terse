#!/usr/bin/env python3
"""Shadow trial: what RTK would have printed for the read-only commands a session ran.

Two entry points:

    rtk_shadow.py hook <dir>      a Claude Code PostToolUse hook for Bash (JSON on stdin)
    rtk_shadow.py report <dir>    per-day summary of <dir>/shadow.jsonl

The hook never prints, always exits 0, and never changes what the session sees. It
detaches at once, so the Bash call is not held up, and appends one row per Bash call to
<dir>/shadow.jsonl. A row holds sizes and a label from a fixed list of program names,
never a command line, a path or any output.

For a command that is a single simple command (no redirects, expansions or globs; a
leading `cd <absolute dir> &&` and a trailing `2>&1` or `2>/dev/null` are the
exceptions), on the read-only list below, and for which `rtk rewrite` gives an RTK
form, the RTK form is run once in the same working directory and its output is sized
against what the session was shown. A chain of such commands (`&&`, `||`, `;`, a new
line, `|`) is taken the same way when every command in it is safe to run a second
time: the line `rtk rewrite` gives is checked to be the same chain with some commands
in RTK form, rebuilt from the checked words and run once under bash, which notes each
RTK command's exit code in a file beside the temporary home. Every other
command gets a row saying why it was skipped, so the share of Bash output the trial
covers is reported, not hidden. That share is what this harness dares to re-run, which
is less than what RTK itself would take.

The RTK run gets no network, its own process-id namespace (so nothing it starts
outlives it), a memory limit, and a home directory on tmpfs that is deleted when it
ends, since RTK keeps a history of command lines, and the content its filters elide,
under HOME.

<dir> holds `rtk` (the binary) and `config.json`: {"until": "YYYY-MM-DD"}, the last
day (UTC) the hook does anything. A file named OFF in <dir> turns the hook off.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

RUN_TIMEOUT = 15
REAP_TIMEOUT = 3
# A temporary home older than this belongs to a worker that died; three runs of
# RUN_TIMEOUT + REAP_TIMEOUT is the longest a live one can take.
STALE_HOME_SECONDS = 120
MAX_WORKERS = 3
MEMORY_LIMIT = 4 << 30
# What Claude Code keeps of a Bash result. A longer one is not measured: the session
# was shown something other than its first 30,000 characters.
SHOWN_CAP = 30_000
# A result Claude Code saved to a file reaches the session as a preview of about this
# many characters and a path (seen on a live call: "Preview (first 2KB)"), though the
# hook is handed its first 30,000.
PERSIST_PREVIEW = 2_000
# Bash output's share of the whole bill, from replay_shell.py's 2026-10-07 run.
BASH_SHARE_OF_BILL = 0.161
# What bash reports for a command killed by SIGPIPE: in a chain, `rtk ... | head` ends
# RTK this way once `head` has read enough, as it ended the original command.
PIPE_CLOSED = 141

READ_ONLY = {"grep", "rg", "cat", "head", "tail", "ls", "find", "wc"}
GIT_READ_ONLY = {"log", "diff", "status", "show"}
# Options that make a read-only command write a file, run another program or open a
# viewer. Matched whole or as `--opt=value`; none of these has a short form, and
# neither git, GNU find nor ripgrep accepts these abbreviated.
FORBIDDEN_ANY = ("--help",)
FORBIDDEN = {
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
             "-fprintf", "-fls"),
    "rg": ("--pre", "--pre-glob", "--hostname-bin"),
    "git": ("--output", "--ext-diff", "--textconv", "-h"),
}
# Taken inside a chain only: RTK has no form for these, but a chain often holds them.
# `cd` is not one: the hook is told one directory, and after a `cd` in the middle of a
# chain that may be where the chain ended, not where its first command ran.
CHAIN_ONLY = {"echo", "pwd", "true"}
# `sort` and `uniq` inside a chain, as filters: every argument must be an option of
# these shapes, so there is no file operand (`sort -o FILE` and `uniq IN OUT` write one).
FILTER_ARGS = {
    "sort": re.compile(r"-[bdfghnrRuV]+|-k\d[\d.,bdfghnrRV]*|-t[^\s]"),
    "uniq": re.compile(r"-[cdiu]+"),
}
# The only labels a row may carry. Anything else is "(other)", so a script's name or a
# mistyped secret at the start of a command line never reaches the log.
LABELS = READ_ONLY | {
    "git", "sed", "awk", "wc", "sort", "uniq", "diff", "echo", "cd", "pwd", "tree", "jq",
    "python", "python3", "uv", "pytest", "ruff", "mypy", "pip", "node", "npm", "npx",
    "gh", "curl", "wget", "make", "cargo", "go", "docker", "ssh", "scp", "rsync", "bash",
    "sh", "rm", "mv", "cp", "mkdir", "chmod", "touch", "tar", "sqlite3", "psql", "terse",
    "timeout", "env", "export", "source", "test", "for", "if", "while", "sleep", "kill",
    "which", "stat", "du", "df", "ps", "date", "xargs", "tee", "printf", "codegraph",
}
GIT_LABELS = GIT_READ_ONLY | {
    "add", "commit", "push", "pull", "fetch", "checkout", "switch", "branch", "merge",
    "rebase", "stash", "worktree", "remote", "config", "rev-parse", "ls-files", "blame",
    "grep", "reset", "restore", "tag", "cherry-pick", "clone", "init", "rm", "mv",
}
# Unquoted, any of these makes the shell do more than run one program with these words.
# The whitespace ones are here because bash splits words on space and tab only.
UNQUOTED_SPECIAL = set("|&;<>(){}*?[]~$`\\\n\r\v\f#!")
DQUOTED_SPECIAL = set("$`\\!")
# `cd <absolute dir> && <rest>`: the one compound form taken, since the directory is known.
LEAD_CD = re.compile(
    r"""[ \t]*cd[ \t]+(?:'([^']*)'|"([^"$`\\!]*)"|([^\s'"|&;<>(){}*?\[\]~$`\\#!]+))[ \t]*&&[ \t]*(.*)""",
    re.S)
TAIL_ERR = re.compile(r"(.*?)[ \t]+2>(&1|/dev/null)[ \t]*", re.S)
# The two stderr redirects a chain's command may end with, where a word would start.
CHAIN_ERR = re.compile(r"2>(?:&1|/dev/null)(?=[ \t\n|;&]|$)")
CHAIN_OPS = ("&&", "||", ";", "|", "\n")
EXPANSION = set("*?[]~$`\\!")


def peel_cd(cmd: str) -> tuple[str, str | None]:
    """(`cmd` without a leading `cd <absolute dir> &&`, that directory or None). A
    relative `cd` is left on, so the command is skipped: the hook cannot tell where it
    started from."""
    m = LEAD_CD.fullmatch(cmd)
    if m:
        target = next(g for g in m.group(1, 2, 3) if g is not None)
        if os.path.isabs(target):
            return m.group(4), target
    return cmd, None


def peel(cmd: str) -> tuple[str, str | None, bool]:
    """(`cmd` without a leading `cd <absolute dir> &&` and a trailing stderr redirect,
    that directory or None, whether stderr was sent to /dev/null)."""
    cmd, cwd = peel_cd(cmd)
    m = TAIL_ERR.fullmatch(cmd)
    if m:
        return m.group(1), cwd, m.group(2) == "/dev/null"
    return cmd, cwd, False


def simple_words(cmd: str) -> list[str] | None:
    """The words of `cmd` when it is one simple command the shell passes through as
    written, else None. Quoting is allowed; anything the shell would expand, chain or
    redirect is not, and neither is a leading `VAR=value`."""
    quote = None
    for ch in cmd:
        if quote == "'":
            if ch == "'":
                quote = None
        elif quote == '"':
            if ch == '"':
                quote = None
            elif ch in DQUOTED_SPECIAL:
                return None
        elif ch in "'\"":
            quote = ch
        elif ch in UNQUOTED_SPECIAL:
            return None
    if quote:
        return None
    try:
        words = shlex.split(cmd)
    except ValueError:
        return None
    if not words or "=" in words[0]:
        return None
    return words


def command_word(cmd: str) -> str:
    """A label from a fixed list: a program's name, or `git <subcommand>`."""
    parts = peel(cmd)[0].split()
    if not parts:
        return "(other)"
    word = parts[0].rsplit("/", 1)[-1]
    if word not in LABELS:
        return "(other)"
    if word == "git" and len(parts) > 1 and parts[1] in GIT_LABELS:
        return f"git {parts[1]}"
    return word


def read_only(words: list[str]) -> bool:
    """True when `words` is on the read-only list and carries no option that writes a
    file, runs a program or opens a viewer."""
    prog = words[0]
    if prog == "git":
        # The subcommand must come first: `git -c ...` and `git -C ...` are refused.
        if len(words) < 2 or words[1] not in GIT_READ_ONLY:
            return False
    elif prog not in READ_ONLY:
        return False
    banned = FORBIDDEN_ANY + FORBIDDEN.get(prog, ())
    return not any(w == b or w.startswith(b + "=") for w in words[1:] for b in banned)


Chain = list[tuple[str, list[str], str]]


def parse_chain(cmd: str) -> Chain | str:
    """`cmd` as [(operator before it, words, stderr redirect after it)] when it is
    simple commands joined by `&&`, `||`, `;`, a new line or `|`, else the skip status
    that says why not. The first operator is ""; a redirect is "", " 2>&1" or
    " 2>/dev/null"."""
    cmd = cmd.strip(" \t\n")
    texts, op, start, quote, i = [], "", 0, None, 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == quote:
                quote = None
            elif quote == '"' and ch in DQUOTED_SPECIAL:
                return "skip_expansion"
        elif ch in "'\"":
            quote = ch
        elif ch == "2" and (i == 0 or cmd[i - 1] in " \t") and CHAIN_ERR.match(cmd, i):
            i = CHAIN_ERR.match(cmd, i).end()
            continue
        else:
            sep = next((s for s in CHAIN_OPS if cmd.startswith(s, i)), None)
            if sep:
                texts.append((op, cmd[start:i]))
                op, start, i = sep, i + len(sep), i + len(sep)
                continue
            if ch in EXPANSION:
                return "skip_expansion"
            if ch in "<>":
                return "skip_redirect"
            if ch in UNQUOTED_SPECIAL:
                return "skip_compound"
        i += 1
    if quote:
        return "skip_compound"
    texts.append((op, cmd[start:]))
    chain: Chain = []
    for op, text in texts:
        redirect = ""
        m = TAIL_ERR.fullmatch(text)
        if m:
            text, redirect = m.group(1), f" 2>{m.group(2)}"
        words = simple_words(text)
        if words is None:
            return "skip_compound"
        chain.append((op, words, redirect))
    return chain


def rerunnable(words: list[str]) -> bool:
    """True when a chain's command is safe to run a second time."""
    prog = words[0]
    if prog in CHAIN_ONLY:
        return True
    if prog in FILTER_ARGS:
        return all(FILTER_ARGS[prog].fullmatch(w) for w in words[1:])
    return read_only(words)


def chain_script(chain: Chain, rewritten: str, rtk: str, exits: str) -> str | None:
    """A bash script for the line `rtk rewrite` gave for `chain`, built from checked
    words and never from RTK's text. None unless that line is the same chain, operator
    for operator, with each command either word for word the original or an `rtk ...`
    form, and at least one of them the latter. Each RTK command is run through a
    function that passes on its exit code and also appends it to the file `exits`: a
    chain's own exit code is its last command's, which says nothing of an RTK command
    in front of a pipe or a `;`."""
    form = parse_chain(rewritten)
    if isinstance(form, str) or len(form) != len(chain):
        return None
    parts, changed = [], False
    for (op, words, redirect), (form_op, form_words, form_redirect) in zip(chain, form,
                                                                           strict=True):
        if form_op != op or form_redirect != redirect:
            return None
        if form_words != words:
            if form_words[0] != "rtk" or len(form_words) < 2:
                return None
            words, changed = ["r", rtk, *form_words[1:]], True
        parts.append(f"{op} {shlex.join(words)}{redirect}")
    if not changed:
        return None
    return (f"exec 3>{shlex.quote(exits)}\n"
            'r() { "$@"; local s=$?; echo "$s" >&3; return "$s"; }\n'
            + " ".join(parts).strip())


def read_exits(path: str) -> list[int]:
    """The exit codes a chain's RTK commands left in `path`, in the order they ended."""
    try:
        return [int(line) for line in Path(path).read_text().split()]
    except (OSError, ValueError):
        return []


def shown(stdout: str, stderr: str) -> str:
    """A Bash result as the session is shown it: stdout, then stderr."""
    stdout, stderr = stdout.strip("\n"), stderr.strip("\n")
    return stdout if not stderr else (f"{stdout}\n{stderr}" if stdout else stderr)


_ENC = None


def tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text, disallowed_special=()))


def rtk_env(home: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home,
           "XDG_CONFIG_HOME": os.path.join(home, ".config"),
           "XDG_DATA_HOME": os.path.join(home, ".local", "share"),
           "XDG_CACHE_HOME": os.path.join(home, ".cache"),
           "RTK_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1",
           # A second `git status` must not take the index lock from the live session.
           "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C.UTF-8",
           # `sort` spills a large input to files here; they go when the home does.
           "TMPDIR": home}
    gitconfig = Path.home() / ".gitconfig"
    if gitconfig.is_file():
        env["GIT_CONFIG_GLOBAL"] = str(gitconfig)
    return env


def _limits() -> None:
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


def run_isolated(argv: list[str], cwd: str, env: dict[str, str]) -> tuple[int, str, str]:
    """Run `argv` with no network, in its own process-id namespace. At the timeout the
    wrapper is killed, which takes every process in that namespace with it, including
    one that left the process group; the wait for the pipes to close is bounded too."""
    proc = subprocess.Popen(
        ["unshare", "-rn", "--pid", "--fork", "--kill-child", "--", *argv], cwd=cwd,
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", start_new_session=True, preexec_fn=_limits)
    try:
        out, err = proc.communicate(timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            for pipe in (proc.stdout, proc.stderr):
                if pipe:
                    pipe.close()
        raise
    return proc.returncode, out, err


def home_root(base: Path) -> Path:
    """Where temporary homes go: tmpfs when there is one, so a reboot clears it."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    root = base / "tmp"
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def sweep_stale_homes(root: Path, now: float) -> None:
    """Delete the temporary home of any worker that died before it could."""
    for path in root.glob("rtk-shadow-*"):
        try:
            st = path.stat()
            if st.st_uid == os.getuid() and now - st.st_mtime > STALE_HOME_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def take_slot(base: Path) -> int | None:
    """A lock on one of MAX_WORKERS slot files, or None when all are held."""
    slots = base / "slots"
    slots.mkdir(mode=0o700, exist_ok=True)
    for i in range(MAX_WORKERS):
        fd = os.open(slots / str(i), os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            os.close(fd)
    return None


def measure(cmd: str, cwd: str, base: Path) -> dict:
    """The row fields for one Bash command: a skip reason, or RTK's output size."""
    chain_text = peel_cd(cmd)[0]
    cmd, cd_dir, drop_stderr = peel(cmd)
    cwd = cd_dir or cwd
    words, chain = simple_words(cmd), None
    if words is None:
        # Not one command: a chain keeps its redirects in the script it is run from.
        chain, cmd, drop_stderr = parse_chain(chain_text), chain_text, False
        if isinstance(chain, str):
            return {"status": chain}
        # One command with a new line after it is still one command: the same list.
        allowed = read_only if len(chain) == 1 else rerunnable
        if not all(allowed(w) for _, w, _ in chain):
            return {"status": "skip_not_listed"}
    elif not read_only(words):
        return {"status": "skip_not_listed"}
    if not os.path.isdir(cwd):
        return {"status": "skip_no_cwd"}
    slot = take_slot(base)
    if slot is None:
        return {"status": "skip_busy"}
    rtk = str(base / "rtk")
    try:
        root = home_root(base)
        sweep_stale_homes(root, time.time())
        with tempfile.TemporaryDirectory(prefix="rtk-shadow-", dir=root) as home:
            env = rtk_env(home)
            try:
                # Without this a missing binary, or a kernel that refuses the
                # namespaces, would read as "RTK has no form for this command".
                rc, _, _ = run_isolated([rtk, "--version"], home, env)
                if rc != 0:
                    return {"status": "rtk_error", "error": f"preflight exit {rc}"}
                rc, out, _ = run_isolated([rtk, "rewrite", cmd], home, env)
                if rc not in (0, 1, 3):
                    return {"status": "rtk_error", "error": f"rewrite exit {rc}"}
                if chain is None:
                    form = simple_words(out.strip()) if rc != 1 else None
                    if not form or form[0] != "rtk" or len(form) < 2:
                        return {"status": "skip_no_rtk_form"}
                    argv = [rtk, *form[1:]]
                else:
                    exits = os.path.join(home, "rtk-shadow-exits")
                    script = chain_script(chain, out, rtk, exits) if rc != 1 else None
                    if script is None:
                        return {"status": "skip_no_rtk_form"}
                    argv = ["/bin/bash", "--noprofile", "--norc", "-c", script]
                started = time.monotonic()
                rc, out, err = run_isolated(argv, cwd, env)
            except subprocess.TimeoutExpired:
                return {"status": "rtk_timeout"}
            except OSError as exc:
                return {"status": "rtk_error", "error": type(exc).__name__}
            text = shown(out, "" if drop_stderr else err)
            row = {"status": "ran", "rtk_exit": rc, "rtk_chars": len(text),
                   "rtk_tokens": tokens(text[:SHOWN_CAP]),
                   "rtk_ms": round((time.monotonic() - started) * 1000)}
            if chain is not None:
                row["segs"] = len(chain)
                row["seg_exits"] = read_exits(exits)[:len(chain)]
            return row
    finally:
        os.close(slot)


def active(base: Path, today: datetime.date) -> bool:
    if (base / "OFF").exists():
        return False
    try:
        until = json.loads((base / "config.json").read_text())["until"]
        return today <= datetime.date.fromisoformat(until)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def detach() -> bool:
    """True in a detached grandchild with no tie to the hook's pipes; the caller that
    gets False returns at once."""
    if os.fork():
        return False
    os.setsid()
    if os.fork():
        os._exit(0)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    return True


def append_row(base: Path, row: dict) -> None:
    line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    fd = os.open(base / "shadow.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def persisted(resp: dict) -> bool:
    return any("persist" in key.lower() and value for key, value in resp.items())


def not_comparable(tool_input: dict, resp: dict, raw: str) -> str | None:
    """Why a result's recorded text is not what a re-run should be sized against."""
    if resp.get("interrupted"):
        return "skip_interrupted"
    if tool_input.get("run_in_background") or any(
            "background" in key.lower() and value for key, value in resp.items()):
        return "skip_background"
    if len(raw) > SHOWN_CAP or persisted(resp):
        return "skip_over_cap"
    return None


def hook(base: Path) -> int:
    if not active(base, datetime.datetime.now(datetime.UTC).date()):
        return 0
    event = json.loads(sys.stdin.read())
    if not isinstance(event, dict) or event.get("tool_name") != "Bash":
        return 0
    tool_input, resp = event.get("tool_input"), event.get("tool_response")
    if not isinstance(tool_input, dict) or not isinstance(resp, dict):
        return 0
    cmd, stdout, stderr = tool_input.get("command"), resp.get("stdout"), resp.get("stderr")
    if not isinstance(cmd, str) or not isinstance(stdout, str):
        return 0
    if not detach():
        return 0
    try:
        row: dict = {"ts": int(time.time()), "word": command_word(cmd)}
        raw = shown(stdout, stderr if isinstance(stderr, str) else "")
        row["raw_chars"] = len(raw)
        row["raw_tokens"] = tokens(raw[:PERSIST_PREVIEW if persisted(resp) else SHOWN_CAP])
        cwd = event.get("cwd")
        skip = not_comparable(tool_input, resp, raw)
        row.update({"status": skip} if skip else
                   measure(cmd, cwd if isinstance(cwd, str) else "", base))
        append_row(base, row)
    except Exception as exc:  # a detached worker has nowhere to report; leave a row
        try:
            append_row(base, {"ts": int(time.time()), "status": "worker_error",
                              "error": type(exc).__name__})
        except OSError:
            pass
    os._exit(0)


def suspect(row: dict) -> bool:
    """RTK printed nothing for a result that was not empty: more likely a run that
    went wrong than a filter that removed everything."""
    return row["rtk_tokens"] == 0 and row["raw_tokens"] > 20


def rtk_exits(row: dict) -> list[int]:
    """The exit codes of the RTK commands a row ran: one for a single command; for a
    chain one per RTK command that started, with a reader closing the pipe read as 0."""
    if "segs" not in row:
        return [row["rtk_exit"]]
    return [0 if e == PIPE_CLOSED else e for e in row.get("seg_exits", [])]


def failed(row: dict) -> bool:
    """An RTK command exited non-zero, or (a chain) none of them got to start."""
    exits = rtk_exits(row)
    return not exits or any(e != 0 for e in exits)


def crashed(row: dict) -> bool:
    """An exit code no wrapped command passes on: killed by a signal, or not run."""
    return any(e < 0 or e >= 126 for e in rtk_exits(row))


def summarize(rows: list[dict]) -> dict:
    """Totals for a set of rows, with the saving counted two ways. RTK passes on the
    wrapped command's exit code (`grep` with no match exits 1), so a non-zero exit is
    not by itself a failure; but a run that failed with a one-line message would then
    read as a large saving. `saved` counts only exit-0 runs (for a chain: every RTK
    command in it started and exited 0); `saved_any` also counts the others. A suspect or crashed row is covered with nothing saved in both."""
    out = {"calls": 0, "bash_tokens": 0, "ran": 0, "ran_raw": 0, "ran_rtk": 0,
           "ran_rtk_any": 0, "suspect": 0, "nonzero": 0, "crashed": 0, "worker_errors": 0,
           "by_status": defaultdict(int)}
    for r in rows:
        if "raw_tokens" not in r or "status" not in r:
            out["worker_errors"] += 1
            continue
        out["calls"] += 1
        out["bash_tokens"] += r["raw_tokens"]
        out["by_status"][r["status"]] += r["raw_tokens"]
        if r["status"] == "ran":
            out["ran"] += 1
            out["ran_raw"] += r["raw_tokens"]
            out["nonzero"] += failed(r)
            if suspect(r) or crashed(r):
                out["suspect"] += suspect(r)
                out["crashed"] += crashed(r)
                out["ran_rtk"] += r["raw_tokens"]
                out["ran_rtk_any"] += r["raw_tokens"]
            else:
                out["ran_rtk_any"] += r["rtk_tokens"]
                out["ran_rtk"] += r["raw_tokens"] if failed(r) else r["rtk_tokens"]
    out["saved"] = out["ran_raw"] - out["ran_rtk"]
    out["saved_any"] = out["ran_raw"] - out["ran_rtk_any"]
    return out


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def render(rows: list[dict]) -> str:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        day = datetime.datetime.fromtimestamp(r["ts"], datetime.UTC).strftime("%Y-%m-%d")
        by_day[day].append(r)
    lines = ["day         calls  errors  bash tokens  covered  saved on covered"
             "  (any exit)  saved of logged Bash  est. of bill"]
    for label, group in [*sorted(by_day.items()), ("ALL", rows)]:
        s = summarize(group)
        lines.append(f"{label:10s} {s['calls']:6d} {s['worker_errors']:7d} "
                     f"{s['bash_tokens']:12,d} "
                     f"{pct(s['ran_raw'], s['bash_tokens']):>8s} "
                     f"{pct(s['saved'], s['ran_raw']):>17s} "
                     f"{pct(s['saved_any'], s['ran_raw']):>11s} "
                     f"{pct(s['saved'], s['bash_tokens']):>21s} "
                     f"{pct(s['saved'] * BASH_SHARE_OF_BILL, s['bash_tokens']):>13s}")
    s = summarize(rows)
    lines.append("")
    lines.append("share of Bash tokens by outcome: " + ", ".join(
        f"{k} {pct(v, s['bash_tokens'])}" for k, v in sorted(s["by_status"].items())))
    words: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r.get("status") == "ran" and not failed(r) and not suspect(r):
            # A chain is not its first command: `echo x && git log` is no `echo` row.
            w = words["(chain)" if "segs" in r else r["word"]]
            w[0] += 1
            w[1] += r["raw_tokens"]
            w[2] += r["rtk_tokens"]
    for word, (n, raw, rtk) in sorted(words.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {word:12s} n={n:5d} raw={raw:10,d} rtk={rtk:10,d} "
                     f"saved={pct(raw - rtk, raw)}")
    lines.append(
        f"Of {s['ran']} RTK runs, {s['suspect']} printed nothing for a non-empty result "
        f"and {s['crashed']} ended on a signal or could not start: no saving in either "
        f"column. {s['nonzero']} had an RTK command exit non-zero or not start; the rest "
        "of those count as no "
        "saving, except in '(any exit)', which takes their output at face value. "
        f"{s['worker_errors']} calls were lost to a worker error and are in no column.\n"
        "'covered' is what this harness re-runs (read-only commands and chains of "
        "them, with no expansion, glob or file redirect), which is less than what RTK "
        "would take, so the last two columns understate RTK on that side. A chain is "
        "run whole again, the commands RTK leaves alone included, and counts as saved "
        "only when every RTK command in it started and exited 0 (or was cut off by the "
        "command reading its output); a chain whose RTK command printed nothing is not "
        "caught when its other commands printed something. A skip reason names the first "
        "thing in a command the harness does not take, not the only one. On the other side, Claude Code sends no hook event for a Bash call that "
        "failed, so failed commands (a failing test run, a build error) are in no row "
        "and not in 'logged Bash': the last two columns are NOT a strict floor, and "
        "'est. of bill' applies a share of the bill that includes failed calls. A "
        "non-zero RTK exit on a covered row means the re-run went differently, not that "
        "the original failed.\nSizes are cl100k tokens, not priced by how long a result "
        "was carried. A result saved to a file is counted at its 2,000-character "
        "preview. The session's grep and find may be other programs than the ones RTK "
        "wraps. This prices what RTK removes, not whether the session then needed it.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["hook"]:
        # Whatever happens, the session must not see this hook fail.
        try:
            if len(args) == 2:
                hook(Path(args[1]))
        except Exception:
            pass
        return 0
    if len(args) != 2 or args[0] != "report":
        print("usage: rtk_shadow.py hook|report <dir>", file=sys.stderr)
        return 2
    path = Path(args[1]) / "shadow.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] \
        if path.exists() else []
    print(render([r for r in rows if "ts" in r]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
