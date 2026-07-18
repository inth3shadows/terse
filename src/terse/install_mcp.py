"""Install/remove terse in front of Claude Code MCP servers.

Rewrites the `mcpServers` entries of a Claude Code config so a named server's
command becomes, for a stdio server:

    <python> -m terse proxy --policy <policy> -- <original command + args>

or, for an HTTP/SSE server (`url` + optional `headers`, #5):

    <python> -m terse proxy --policy <policy> --header k=v ... -- <original url>

Claude Code has three MCP scopes (#58), each backed by a different location:
  - user    — top-level `mcpServers` in `~/.claude.json` (default; #27's original
              scope).
  - project — a `.mcp.json` file, normally checked into the repo and shared with
              every clone.
  - local   — nested `projects."<repo-path>".mcpServers` inside `~/.claude.json`,
              personal to one repo on one machine. `<repo-path>` resolves via
              `git rev-parse --git-common-dir` (see `default_repo_path`), not
              cwd, so every worktree of a claudew/codexw bare-worktree repo
              shares one entry instead of one per worktree.
`resolve_target` maps a scope (+ its scope-specific override flag) to the
physical file and the key path inside it that holds `mcpServers`.

The original entry is preserved verbatim in a sidecar stash so `uninstall` can
restore it byte-for-byte. The wrap is idempotent (re-running re-wraps from the
stashed original rather than double-wrapping). Cross-call diffing is the proxy
DEFAULT since #75; a plain wrap writes no diff flag and inherits it, while
`install-mcp --diff`/`--no-diff` bake an explicit override into the entry. The
stash is namespaced by scope (`Target.stash_prefix`) so the same server can be
independently managed in more than one scope — user and local both live in
`~/.claude.json` and would otherwise collide in one flat stash.

The core is pure functions over plain dicts (`wrap`/`unwrap`) so they are unit
testable without touching the filesystem; the `do_*` helpers add IO + backup.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ._secure_io import write_restricted

STASH_NAME = ".terse-mcp-stash.json"
VALID_SCOPES = ("user", "project", "local")

# How many timestamped config backups to retain per config file. Each backup is a full
# copy of the config, whose MCP `env` blocks can hold API keys — so an unbounded pile of
# them is long-lived secret sprawl (a rotated key lingers in old backups forever). Keep a
# short rollback window; prune the rest. 0 would disable pruning.
_MAX_BACKUPS = 5


# --------------------------------------------------------------------------- IO
def classify_server_sensitivity(name: str, command: object = "") -> bool:
    """Install-time best-effort guess: does this server carry credentials/personal data,
    so lossy transforms should be forbidden on it? Matches the server name and its launch
    command against `policy.SENSITIVE_SERVER_RE`. This is a SUGGESTION that should PROMPT
    the operator to confirm baking the server into `never_lossy_servers` — never an
    automatic decision: the operator knows sensitive servers whose names the pattern can't
    catch (a personal KB, a launcher alias), and this only surfaces the obvious ones. The
    runtime floor (PR #89) independently forbids lossy on pattern-matching names regardless."""
    from .policy import SENSITIVE_SERVER_RE
    parts = [name, *(command if isinstance(command, list) else [command])]
    return bool(SENSITIVE_SERVER_RE.search(" ".join(str(p) for p in parts)))


def add_never_lossy_server(policy_doc: dict, name: str) -> bool:
    """Add `name` to a policy doc's `never_lossy_servers` (deduped + sorted); return True if
    the doc changed. Pure — the caller owns reading/writing the file. `name` is the server's
    config key, which install-mcp also bakes as `--server-name`, so it matches the identity
    `policy.apply` sees at runtime — making lossy structurally forbidden on it (PR #89)."""
    existing = list(policy_doc.get("never_lossy_servers", []))
    if name in existing:
        return False
    policy_doc["never_lossy_servers"] = sorted([*existing, name])
    return True


def config_path() -> Path:
    """Claude Code config location. Honors $CLAUDE_CONFIG, else ~/.claude.json."""
    env = os.environ.get("CLAUDE_CONFIG")
    return Path(env).expanduser() if env else Path.home() / ".claude.json"


def stash_path(cfg: Path) -> Path:
    return cfg.parent / STASH_NAME


def default_repo_path() -> str:
    """Local scope's default `projects` key: `git rev-parse --git-common-dir`,
    absolute. For a plain repo this is `<repo>/.git`'s parent-equivalent identity;
    for a claudew/codexw bare-worktree layout it resolves to the bare root itself
    (e.g. `.../runecho/.bare`) regardless of which worktree you're standing in —
    matching how Claude Code itself keys local-scope entries for such repos (#58),
    confirmed against a live `~/.claude.json` local entry keyed at exactly that
    path. Raises ValueError (not a git repo, or git missing) so callers can tell
    the user to pass --repo-path explicitly instead of crashing on a subprocess
    error."""
    try:
        result = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        raise ValueError(
            "local scope resolves its default --repo-path from git, but this "
            "isn't a git repo (or git isn't installed) — pass --repo-path "
            "explicitly") from e
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = Path.cwd() / git_dir
    return str(git_dir.resolve())


@dataclass(frozen=True)
class Target:
    """A resolved scope: `cfg` is the physical file to read/write, `server_path`
    is the key path to walk from that file's root to the dict which itself holds
    `mcpServers` (empty for user/project — it sits at the top level; ("projects",
    "<repo>") for local), and `stash_prefix` namespaces this scope's slice of the
    sidecar stash."""
    cfg: Path
    server_path: tuple[str, ...]
    stash_prefix: str


def resolve_target(scope: str, *, cfg: Path | None = None, file: str | None = None,
                   repo_path: str | None = None) -> Target:
    """Map --scope (+ its scope-specific override) to a Target. `cfg` overrides the
    physical ~/.claude.json location for user/local scope (tests, $CLAUDE_CONFIG);
    `file` overrides the project-scope .mcp.json path; `repo_path` overrides local
    scope's `projects` key (else `default_repo_path()`)."""
    if scope == "user":
        return Target(cfg or config_path(), (), "user")
    if scope == "project":
        path = Path(file).expanduser().resolve() if file else Path(".mcp.json").resolve()
        return Target(path, (), "project")
    if scope == "local":
        repo = repo_path or default_repo_path()
        return Target(cfg or config_path(), ("projects", repo), f"local:{repo}")
    raise ValueError(f"unknown scope {scope!r}; must be one of {VALID_SCOPES}")


def _servers_root(config: dict, server_path: tuple[str, ...]) -> dict:
    """Walk `server_path` from `config`'s root, creating intermediate dicts as
    needed, and return the dict that itself should hold `mcpServers` — `config`
    itself when `server_path` is empty (user/project scope), else the nested
    per-repo block (local scope)."""
    node = config
    for key in server_path:
        node = node.setdefault(key, {})
    return node


def terse_invocation() -> list[str]:
    """How a wrapped entry should launch terse. Absolute interpreter + `-m terse`
    so it does not depend on `terse` being on the MCP launcher's PATH. Overridable
    via $TERSE_MCP_CMD (whitespace-split) for unusual installs."""
    override = os.environ.get("TERSE_MCP_CMD")
    if override:
        return override.split()
    return [sys.executable, "-m", "terse"]


# ------------------------------------------------------------------- pure core
def wrap(config: dict, stash: dict, server: str, policy: str,
         terse_cmd: list[str], capture_dir: str | None = None,
         diff: bool | None = None,
         diff_keyframe_interval: int | None = None,
         no_stats: bool = False) -> tuple[dict, dict]:
    """Wrap `server`'s entry with the terse proxy. Idempotent: if already managed
    (present in stash), re-wrap from the stashed original so policy/cmd updates
    apply cleanly without nesting proxies. Preserves all non-command/args (and, for a
    URL entry, non-url/headers) keys (env, cwd, type, …) of the original entry — and,
    on a re-wrap, hand-edits made to those keys on the LIVE wrapped entry win over the
    stashed original's values (the drift guard below). With
    `capture_dir`, the wrapped proxy tees raw tool results into that corpus for later
    measurement (#32). `diff` is tri-state: None writes no flag (the entry inherits
    the proxy default — ON since #75), True/False bake `--diff`/`--no-diff` into the
    entry; a re-wrap always reflects the latest invocation, flags never accumulate.

    Two shapes of original entry are wrappable: a stdio server (`command` + optional
    `args`) and an HTTP/SSE server (`url` + optional `headers`, #5) — the latter is
    proxied by pointing terse's HTTP downstream at that url, with any `headers`
    forwarded as repeated `--header k=v` (see `transport.HttpTransport`). Anything with
    neither key is not a valid MCP server entry and can't be wrapped."""
    servers = config.setdefault("mcpServers", {})
    live = servers.get(server)
    if server in stash:
        original = stash[server]
    elif live is not None:
        original = live
        stash[server] = original
    else:
        raise KeyError(server)

    # The config's own name for this server is the one identity terse can state rather
    # than guess (#83): it makes a server-scoped policy rule (`runecho.*`) match even
    # when the server's tools aren't self-prefixed, and labels the stats ledger with the
    # real server instead of the launch command's basename (kb behind `sb-run`).
    proxy_opts = ["--policy", policy, "--server-name", server]
    if capture_dir:
        proxy_opts += ["--capture-dir", capture_dir]
    if no_stats:
        # Only the opt-out is bakeable: the savings ledger is the proxy DEFAULT (it is
        # payload-free — see stats.py), so an entry needs a flag only to turn it off.
        proxy_opts += ["--no-stats"]
    if diff is not None:
        proxy_opts += ["--diff"] if diff else ["--no-diff"]
    if diff is not False and diff_keyframe_interval is not None:
        proxy_opts += ["--diff-keyframe-interval", str(diff_keyframe_interval)]

    orig_cmd = original.get("command")
    if orig_cmd:
        orig_args = list(original.get("args", []))
        new_entry = {k: v for k, v in original.items() if k not in ("command", "args")}
        new_entry["command"] = terse_cmd[0]
        new_entry["args"] = [*terse_cmd[1:], "proxy", *proxy_opts, "--", orig_cmd, *orig_args]
    else:
        orig_url = original.get("url")
        if not orig_url:
            # Neither 'command' nor 'url' — not a launchable stdio server NOR a
            # dispatchable HTTP one; nothing terse can wrap (#19).
            raise ValueError(
                f"server '{server}' has no 'command' or 'url' to wrap — it doesn't "
                f"look like a valid MCP server entry")
        orig_headers = original.get("headers") or {}
        header_opts: list[str] = []
        for k, v in orig_headers.items():
            header_opts += ["--header", f"{k}={v}"]
        new_entry = {k: v for k, v in original.items()
                    if k not in ("command", "args", "url", "headers")}
        new_entry["command"] = terse_cmd[0]
        new_entry["args"] = [*terse_cmd[1:], "proxy", *proxy_opts, *header_opts, "--", orig_url]

    if live is not None and live is not original:
        # Drift guard: a re-wrap used to rebuild the entry purely from the stashed
        # original, silently reverting any hand-edit made to the WRAPPED entry since
        # the last install (a scoped env.PATH pin, a cwd) — that reverted codegraph's
        # node pin in production on 2026-07-13. command/args are terse-owned and always
        # rebuilt (flags must reflect this invocation); url/headers never appear on a
        # wrapped entry (they're folded into args), so a drifted live copy of them must
        # not be resurrected either. Everything else on the live entry wins.
        for k, v in live.items():
            if k not in ("command", "args", "url", "headers"):
                new_entry[k] = v
    servers[server] = new_entry
    return config, stash


def unwrap(config: dict, stash: dict, server: str) -> tuple[dict, dict]:
    """Restore `server`'s original entry from the stash (byte-for-byte)."""
    if server not in stash:
        raise KeyError(server)
    config.setdefault("mcpServers", {})[server] = stash.pop(server)
    return config, stash


def is_managed(stash: dict, server: str) -> bool:
    return server in stash


# ------------------------------------------------------------------ IO helpers
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_stash(path: Path) -> dict:
    """Load the sidecar stash, transparently migrating the pre-#58 flat format
    ({server: original_entry}) to the scope-namespaced one ({stash_prefix: {server:
    original_entry}}) — before #58, "user" was the only scope, so every legacy entry
    is exactly that scope's stash. Detected by shape: a legacy entry's value is an
    MCP server entry itself (has 'command' or 'url'); a migrated file's top-level
    values are scope buckets (dicts of server entries), which don't. Migration is
    in-memory only here — do_install/do_uninstall persist the new shape on their
    next write, same as any other change."""
    raw = _load_json(path)
    if not raw:
        return {}
    looks_legacy = any(
        isinstance(v, dict) and ("command" in v or "url" in v) for v in raw.values()
    )
    return {"user": raw} if looks_legacy else raw


def _write_json(path: Path, obj: dict, *, trailing_newline: bool = True) -> None:
    # ensure_ascii=False keeps non-ASCII (em-dashes, emoji, …) literal, matching how
    # Claude Code itself serializes ~/.claude.json. With the default (True), the first
    # wrap rewrites the WHOLE file as \uXXXX escapes — huge spurious diff, and the
    # install→uninstall round-trip is no longer byte-identical to the backup (#27).
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    # MCP server entries can carry secrets in `env` blocks — write_restricted keeps this
    # file at 0600 from before any content lands on disk (see _secure_io).
    write_restricted(path, text + ("\n" if trailing_newline else ""))


def _prune_backups(cfg: Path, keep: int = _MAX_BACKUPS) -> None:
    """Keep only the `keep` most-recent `<cfg>.bak-*` files, deleting older ones — they
    hold copies of the config's secrets, so they must not accumulate without bound. No-op
    when `keep <= 0` (pruning disabled). Ordered by mtime so it's robust to the epoch-
    timestamp digit width changing; a same-second overwrite just leaves fewer to prune.
    Best-effort: a file that vanishes or can't be unlinked (race, permissions) is skipped,
    never fatal to the install/uninstall that triggered the backup."""
    if keep <= 0:
        return
    backups = sorted(cfg.parent.glob(f"{cfg.name}.bak-*"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def _backup(cfg: Path) -> Path:
    bak = cfg.with_name(f"{cfg.name}.bak-{int(time.time())}")
    write_restricted(bak, cfg.read_text(encoding="utf-8"))  # backup mirrors cfg's secrets
    _prune_backups(cfg)
    return bak


def do_install(servers: list[str], policy: str, *, dry_run: bool = False,
               cfg: Path | None = None, capture_dir: str | None = None,
               diff: bool | None = None, diff_keyframe_interval: int | None = None,
               scope: str = "user", file: str | None = None,
               repo_path: str | None = None, no_stats: bool = False,
               never_lossy: bool = False) -> dict:
    target = resolve_target(scope, cfg=cfg, file=file, repo_path=repo_path)
    if not target.cfg.exists():
        what = ".mcp.json" if scope == "project" else "Claude config"
        raise FileNotFoundError(f"{what} not found: {target.cfg}")
    raw = target.cfg.read_text(encoding="utf-8")
    config = json.loads(raw)
    had_nl = raw.endswith("\n")  # preserve trailing-newline state for byte-fidelity
    full_stash = _load_stash(stash_path(target.cfg))
    stash = full_stash.setdefault(target.stash_prefix, {})
    node = _servers_root(config, target.server_path)
    policy_abs = str(Path(policy).resolve())
    if not Path(policy_abs).exists():
        raise FileNotFoundError(f"policy not found: {policy_abs}")
    # Resolve to an absolute path so capture works regardless of the proxy's cwd; the
    # proxy/capture_payload creates the dir on first write, so no need to pre-create it.
    capture_abs = str(Path(capture_dir).resolve()) if capture_dir else None
    terse_cmd = terse_invocation()

    available = sorted((node.get("mcpServers") or {}).keys())
    managed = set(stash)
    missing = [s for s in servers if s not in set(available) and s not in managed]
    if missing:
        raise ValueError(
            f"unknown server(s): {', '.join(missing)}. "
            f"available: {', '.join(available) or '(none)'}")
    changes = []
    for s in servers:
        before = (node.get("mcpServers") or {}).get(s)
        # Hand-edits = non-terse-owned keys on the live WRAPPED entry that differ from
        # the stashed original — the drift guard in wrap() carries them forward; name
        # them in the result so the operator sees what survived (and what to move into
        # the original entry if it should also survive an uninstall).
        preserved = sorted(
            k for k in (before or {})
            if k not in ("command", "args", "url", "headers")
            and s in stash and (before or {}).get(k) != stash[s].get(k)
        )
        wrap(node, stash, s, policy_abs, terse_cmd, capture_dir=capture_abs,
             diff=diff, diff_keyframe_interval=diff_keyframe_interval,
             no_stats=no_stats)
        changes.append({"server": s, "before": before,
                        "after": node["mcpServers"][s], "preserved": preserved})

    result = {"config": str(target.cfg), "scope": scope, "policy": policy_abs,
              "available": available, "changes": changes, "dry_run": dry_run,
              "backup": None, "capture_dir": capture_abs, "diff": diff,
              "no_stats": no_stats, "never_lossy_added": []}
    if not dry_run and changes:
        result["backup"] = str(_backup(target.cfg))
        _write_json(target.cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(target.cfg), full_stash)

    # --never-lossy: bake the wrapped server(s) into the POLICY file's never_lossy_servers
    # (a separate file from the Claude config above), so lossy transforms are structurally
    # forbidden on them at runtime (PR #89). Computed even under dry-run for reporting, but
    # only written when not a dry-run and something actually changed.
    if never_lossy:
        pol_doc = json.loads(Path(policy_abs).read_text(encoding="utf-8"))
        added = [s for s in servers if add_never_lossy_server(pol_doc, s)]
        result["never_lossy_added"] = added
        if added and not dry_run:
            _write_json(Path(policy_abs), pol_doc)
    return result


# ------------------------------------------------------------------ read-only status
def _read_servers_root(config: dict, server_path: tuple[str, ...]) -> dict:
    """Non-mutating counterpart to `_servers_root` — a status scan must never create
    the intermediate dicts `setdefault` would, or every `mcp-status` run on a repo
    with no local-scope entry yet would spuriously fabricate one in memory (harmless
    since never written, but wrong to even construct)."""
    node: object = config
    for key in server_path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def _scan_target(target: Target, scope: str) -> list[dict]:
    if not target.cfg.exists():
        return []
    config = _load_json(target.cfg)
    node = _read_servers_root(config, target.server_path)
    servers = node.get("mcpServers") or {}
    full_stash = _load_stash(stash_path(target.cfg))
    stash = full_stash.get(target.stash_prefix, {})

    rows = []
    for name in sorted(set(servers) | set(stash)):
        managed = name in stash
        present = name in servers
        if managed and present:
            state = "wrapped"
        elif managed and not present:
            # A stash entry with no matching mcpServers entry — the entry was removed
            # or edited by hand after terse wrapped it. Surfacing this is the whole
            # point of #58: this exact kind of scope/state drift is what prompted it.
            state = "orphaned-stash"
        else:
            state = "unwrapped"
        policy = None
        policy_missing = False
        wraps = None
        diff = None
        stats_on = None
        if state == "wrapped":
            # A terse-wrapped entry's args are
            #   [-m terse] proxy <proxy-opts> -- <downstream cmd/url + args>
            # so the downstream it actually fronts, and the diff/stats flags baked in,
            # are all recoverable from here — none of which the old status line showed,
            # leaving no way to spot e.g. a --no-diff or a wrong downstream from status.
            args = servers[name].get("args") or []
            if "--policy" in args:
                i = args.index("--policy")
                if i + 1 < len(args):
                    policy = args[i + 1]
                    # Only an absolute policy path is unambiguously checkable: a relative
                    # one resolves against the MCP launcher's cwd, which a status scan
                    # can't know, so we never false-flag it (see #58's drift lineage — the
                    # point is to surface real drift, not manufacture noise).
                    if os.path.isabs(policy) and not os.path.exists(policy):
                        policy_missing = True
            if "--" in args:
                downstream = args[args.index("--") + 1:]
                if downstream:
                    wraps = " ".join(downstream)
            diff = "off" if "--no-diff" in args else ("on" if "--diff" in args
                                                      else "default")
            stats_on = "--no-stats" not in args
        rows.append({"scope": scope, "server": name, "state": state, "policy": policy,
                    "policy_missing": policy_missing, "wraps": wraps, "diff": diff,
                    "stats": stats_on, "config": str(target.cfg)})
    return rows


def scan_scopes(*, cfg: Path | None = None, file: str | None = None,
                repo_path: str | None = None) -> list[dict]:
    """Enumerate every terse-relevant mcpServers entry across all three scopes,
    read-only — no writes, no directory creation, never raises. One row per
    (scope, server): {scope, server, state, policy, policy_missing, wraps, diff,
    stats, config}, state one of "wrapped" (terse-managed and present),
    "orphaned-stash" (managed but the entry vanished — see `_scan_target`), or
    "unwrapped" (present, not terse's). The wrapped-only fields (policy_missing,
    wraps, diff, stats) are None/False for non-wrapped rows. Local scope is
    silently omitted, not an error, when it doesn't resolve (not in a git repo and
    no --repo-path given) — "no local scope here" is the common case, not a failure."""
    rows: list[dict] = []
    rows += _scan_target(resolve_target("user", cfg=cfg), "user")
    rows += _scan_target(resolve_target("project", file=file), "project")
    try:
        local_target = resolve_target("local", cfg=cfg, repo_path=repo_path)
    except ValueError:
        local_target = None
    if local_target is not None:
        rows += _scan_target(local_target, "local")
    return rows


def do_uninstall(servers: list[str] | None, *, all_: bool = False,
                 dry_run: bool = False, cfg: Path | None = None,
                 scope: str = "user", file: str | None = None,
                 repo_path: str | None = None) -> dict:
    target = resolve_target(scope, cfg=cfg, file=file, repo_path=repo_path)
    raw = target.cfg.read_text(encoding="utf-8") if target.cfg.exists() else ""
    config = json.loads(raw) if raw else {}
    had_nl = raw.endswith("\n") if raw else True  # preserve trailing-newline state
    full_stash = _load_stash(stash_path(target.cfg))
    stash = full_stash.setdefault(target.stash_prefix, {})
    node = _servers_root(config, target.server_path)
    targets = sorted(stash.keys()) if all_ else (servers or [])

    changes = []
    for s in targets:
        if not is_managed(stash, s):
            changes.append({"server": s, "restored": False, "reason": "not managed by terse"})
            continue
        unwrap(node, stash, s)
        changes.append({"server": s, "restored": True})

    result = {"config": str(target.cfg), "scope": scope, "changes": changes,
              "dry_run": dry_run, "backup": None}
    if not dry_run and any(c.get("restored") for c in changes):
        result["backup"] = str(_backup(target.cfg))
        _write_json(target.cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(target.cfg), full_stash)
    return result
