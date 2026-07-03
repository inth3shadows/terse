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
stashed original rather than double-wrapping) and never enables `--diff`
implicitly (diff fluency is unverified — see the diff-fluency reports). The
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


# --------------------------------------------------------------------------- IO
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
                                capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
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
         terse_cmd: list[str], capture_dir: str | None = None) -> tuple[dict, dict]:
    """Wrap `server`'s entry with the terse proxy. Idempotent: if already managed
    (present in stash), re-wrap from the stashed original so policy/cmd updates
    apply cleanly without nesting proxies. Preserves all non-command/args (and, for a
    URL entry, non-url/headers) keys (env, cwd, type, …) of the original entry. With
    `capture_dir`, the wrapped proxy tees raw tool results into that corpus for later
    measurement (#32).

    Two shapes of original entry are wrappable: a stdio server (`command` + optional
    `args`) and an HTTP/SSE server (`url` + optional `headers`, #5) — the latter is
    proxied by pointing terse's HTTP downstream at that url, with any `headers`
    forwarded as repeated `--header k=v` (see `transport.HttpTransport`). Anything with
    neither key is not a valid MCP server entry and can't be wrapped."""
    servers = config.setdefault("mcpServers", {})
    if server in stash:
        original = stash[server]
    elif server in servers:
        original = servers[server]
        stash[server] = original
    else:
        raise KeyError(server)

    proxy_opts = ["--policy", policy]
    if capture_dir:
        proxy_opts += ["--capture-dir", capture_dir]

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


def _backup(cfg: Path) -> Path:
    bak = cfg.with_name(f"{cfg.name}.bak-{int(time.time())}")
    write_restricted(bak, cfg.read_text(encoding="utf-8"))  # backup mirrors cfg's secrets
    return bak


def do_install(servers: list[str], policy: str, *, dry_run: bool = False,
               cfg: Path | None = None, capture_dir: str | None = None,
               scope: str = "user", file: str | None = None,
               repo_path: str | None = None) -> dict:
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
        wrap(node, stash, s, policy_abs, terse_cmd, capture_dir=capture_abs)
        changes.append({"server": s, "before": before,
                        "after": node["mcpServers"][s]})

    result = {"config": str(target.cfg), "scope": scope, "policy": policy_abs,
              "available": available, "changes": changes, "dry_run": dry_run,
              "backup": None, "capture_dir": capture_abs}
    if not dry_run and changes:
        result["backup"] = str(_backup(target.cfg))
        _write_json(target.cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(target.cfg), full_stash)
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
        if present:
            args = servers[name].get("args") or []
            if "--policy" in args:
                policy = args[args.index("--policy") + 1]
        rows.append({"scope": scope, "server": name, "state": state, "policy": policy,
                    "config": str(target.cfg)})
    return rows


def scan_scopes(*, cfg: Path | None = None, file: str | None = None,
                repo_path: str | None = None) -> list[dict]:
    """Enumerate every terse-relevant mcpServers entry across all three scopes,
    read-only — no writes, no directory creation, never raises. One row per
    (scope, server): {scope, server, state, policy, config}, state one of "wrapped"
    (terse-managed and present), "orphaned-stash" (managed but the entry vanished —
    see `_scan_target`), or "unwrapped" (present, not terse's). Local scope is
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
