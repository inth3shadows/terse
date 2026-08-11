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

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

from . import policy as policy_mod
from ._secure_io import write_restricted
from .stats import resolve_ledger_identity

STASH_NAME = ".terse-mcp-stash.json"
PEERS_STEM = ".terse-peers"
DEFAULT_ROUTER = "terse"
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
    via $TERSE_MCP_CMD (whitespace-split) for unusual installs — e.g. pointing at
    the `terse` console script, whose path survives upgrades that move a versioned
    `uv tool`/`pipx` venv out from under the baked interpreter.

    The override's argv[0] is `expanduser`ed: a wrapped entry is spawned from JSON
    via execve, with no shell to expand `~`, so a quoted `TERSE_MCP_CMD='~/bin/terse'`
    would otherwise write a literal tilde that can never resolve. Expanding here is
    what makes the documented override behave the same quoted or bare."""
    override = os.environ.get("TERSE_MCP_CMD")
    if not override:
        # No check needed on this branch: `sys.executable` is the interpreter currently
        # running, so it exists by construction. Only the override can name something
        # that doesn't.
        return [sys.executable, "-m", "terse"]
    parts = override.split()
    if parts:
        parts[0] = os.path.expanduser(parts[0])
    if (bad := launcher_missing(parts)) is not None:
        raise FileNotFoundError(
            f"$TERSE_MCP_CMD launcher not found: {bad}. Wrapped entries are spawned "
            f"directly (no shell), so this path must exist as written.")
    return parts


def launcher_missing(terse_cmd: list[str]) -> str | None:
    """The argv[0] of `terse_cmd` if it names a path that does not exist, else None.

    Only a path-like argv[0] (containing a separator) is checkable — a bare name like
    `terse` is resolved against the launcher's PATH, which we cannot know from here,
    so it is never flagged. Same restraint as the relative-policy-path rule in
    `_scan_target`: surface real drift, never manufacture noise."""
    if not terse_cmd:
        return None
    cmd = terse_cmd[0]
    if os.sep not in cmd and (os.altsep or os.sep) not in cmd:
        return None
    return None if os.path.exists(cmd) else cmd


# ------------------------------------------------------------------- pure core
def _runtime_opts(*, capture_dir: str | None, no_stats: bool, diff: bool | None,
                  diff_keyframe_interval: int | None, no_join_blocks: bool) -> list[str]:
    """The runtime flags baked onto a terse launcher, shared by a single-server `wrap`
    entry and a `--multiproxy` router entry (#179).

    Every one of these is accepted by `terse proxy` with OR without `--config` (see
    `cli._cmd_proxy`), so the router honors them for all its peers at once. Keeping one
    builder is what stops the router from silently ignoring a flag `wrap` respects —
    `--capture-dir` going nowhere means a measurement run records nothing."""
    opts: list[str] = []
    if capture_dir:
        opts += ["--capture-dir", capture_dir]
    if no_stats:
        # Only the opt-out is bakeable: the savings ledger is the proxy DEFAULT (it is
        # payload-free — see stats.py), so an entry needs a flag only to turn it off.
        opts += ["--no-stats"]
    if diff is not None:
        opts += ["--diff"] if diff else ["--no-diff"]
    if diff is not False and diff_keyframe_interval is not None:
        opts += ["--diff-keyframe-interval", str(diff_keyframe_interval)]
    if no_join_blocks:
        # Only the opt-out is bakeable: joining is the proxy DEFAULT (#116), so an entry
        # needs a flag only to turn it off.
        opts += ["--no-join-blocks"]
    return opts


def wrap(config: dict, stash: dict, server: str, policy: str,
         terse_cmd: list[str], capture_dir: str | None = None,
         diff: bool | None = None,
         diff_keyframe_interval: int | None = None,
         no_join_blocks: bool = False,
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
    proxy_opts = ["--policy", policy, "--server-name", server,
                  *_runtime_opts(capture_dir=capture_dir, no_stats=no_stats, diff=diff,
                                 diff_keyframe_interval=diff_keyframe_interval,
                                 no_join_blocks=no_join_blocks)]

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


def _default_diff_label(policy_path: str | None) -> str:
    """Describe the diff setting of an entry that passes neither `--diff` nor `--no-diff`.

    Printing a bare `default` caused a real misdiagnosis (#181): #170 flipped the default to
    OFF, so `default` read as "the feature's normal state, i.e. on", and a session that saw
    `diffs=0` concluded cross-call diffing was never implemented. It is implemented and
    deliberately off — a reporting bug, not a behavioural one, but the reader cannot tell.

    So resolve it rather than name it: against the entry's own policy file when that file
    states `diff`, and otherwise against the `Policy.diff` field default itself, read from
    the dataclass so this label can never drift from the value it claims to describe (the
    #144 failure — a hand-copied constant outliving the decision behind it)."""
    # Only an ABSOLUTE path may be read here, for the same reason the caller's
    # `policy_missing` check two lines away skips relative ones: a relative path resolves
    # against the MCP launcher's cwd, which a status scan cannot know. Reading it would
    # resolve against the *scanner's* cwd instead and report the diff setting of whatever
    # file happens to sit there — a confidently wrong label, which is precisely the
    # label-vs-reality divergence #181 exists to kill.
    #
    # Falling through to the dataclass default would NOT be honest either: the file does
    # state a value, the scanner simply cannot reach it, so printing `default (off)` while
    # that file says `"diff": true` is the same divergence pointing the other way. Say
    # unknown, and say why. (`do_install` always writes an absolute `--policy`, so only a
    # hand-edited entry reaches this branch.)
    if policy_path and not os.path.isabs(policy_path):
        return "policy (relative path — unknown)"
    if policy_path:
        try:
            doc = json.loads(Path(policy_path).read_text(encoding="utf-8"))
            # Truthiness is CORRECT here and must stay: `load_policy` builds the real policy
            # with `bool(doc.get("diff", False))`, so `"diff": "false"` (a non-empty string)
            # genuinely turns diffing ON at runtime. This label's job is to report EFFECTIVE
            # behaviour, not the author's apparent intent — a review flagged the truthiness as
            # a bug, but tightening it to `is True` would make the label print "off" while the
            # proxy diffs, which is precisely the label-vs-reality divergence #181 exists to
            # kill. If the coercion in `load_policy` ever changes, change it here too.
            if isinstance(doc, dict) and "diff" in doc:
                return f"policy ({'on' if doc['diff'] else 'off'})"
        except (OSError, ValueError, RecursionError):
            pass  # unreadable or malformed: the built-in default is still the truth below
    field_default = fields(policy_mod.Policy)
    on = next(f.default for f in field_default if f.name == "diff")
    return f"default ({'on' if on else 'off'})"


def peers_path(cfg: Path, stash_prefix: str = "user") -> Path:
    """Where a router's peers file lives, namespaced by SCOPE exactly as the stash is.

    User and local scope share one physical `~/.claude.json`, so a single
    `.terse-peers.json` beside it would let a local-scope install silently overwrite the
    user-scope fleet (and vice versa) — the same collision `Target.stash_prefix` exists
    to prevent. A local prefix carries a repo path, so it is slugified to a legal, bounded
    filename.

    The hash tail is UNCONDITIONAL, because the collision comes from slugification, not
    from truncation: `/home/e/a/b` and `/home/e/a-b` both slugify to `local-home-e-a-b`,
    well under any length limit. Two repos then shared one peers file, so repo 1's router
    launched repo 2's servers and exported their tools into repo 1's sessions — a
    cross-repo capability leak, not a bookkeeping mixup. `stash_prefix` is injective and
    the filename must not throw that away."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stash_prefix).strip("-") or "scope"
    digest = hashlib.sha256(stash_prefix.encode()).hexdigest()[:8]
    return cfg.parent / f"{PEERS_STEM}-{slug[:31]}-{digest}.json"


def _unnest(entry: dict) -> dict:
    """A terse-wrapped entry reduced to the downstream it wraps (#179).

    `wrap_multi` normally re-describes a peer from its STASHED original, which is already
    unwrapped. But a server can be wrapped with its stash under a DIFFERENT scope (or
    missing entirely — #172), and folding that entry in verbatim would put
    `terse proxy --policy ... -- kb-mcp` inside the router: a proxy nested in a proxy,
    charging the primer twice and defeating the entire point of consolidating. Recover
    the downstream from the launcher's own `--` separator, which `wrap` always emits.

    A URL downstream is a bare argument after `--`; a stdio one is command + args."""
    if not _looks_like_terse_launcher(entry):
        return entry
    args = list(entry.get("args") or [])
    if "--" not in args:
        return entry
    tail = args[args.index("--") + 1:]
    if not tail:
        return entry
    rest = {k: v for k, v in entry.items() if k not in ("command", "args")}
    if len(tail) == 1 and "://" in tail[0]:
        headers = {}
        for i, a in enumerate(args):
            if a == "--header" and i + 1 < len(args) and "=" in args[i + 1]:
                k, _, v = args[i + 1].partition("=")
                headers[k] = v
        return {**rest, "url": tail[0], **({"headers": headers} if headers else {})}
    return {**rest, "command": tail[0], "args": tail[1:]}


def _peer_spec(name: str, original: dict, policy: str) -> dict:
    """One `downstreams[]` entry from a server's ORIGINAL (unwrapped) config entry.

    Mirrors `wrap`'s two wrappable shapes: stdio (`command` + optional `args`) and
    HTTP/SSE (`url` + optional `headers`). `env`/`cwd` ride along for stdio because a
    peer is launched by the router exactly as the client would have launched it."""
    original = _unnest(original)
    spec: dict = {"name": name, "policy": policy}
    if original.get("command"):
        spec["command"] = [original["command"], *list(original.get("args", []))]
        env = original.get("env")
        if env:
            if not isinstance(env, dict):
                # A hand-edited `env` of the wrong shape used to reach `.items()` and
                # crash with a bare AttributeError — no server name, no file, no clue.
                raise ValueError(
                    f"server '{name}' has a non-object 'env' ({type(env).__name__}) — "
                    f"fix that entry in the config before folding it")
            nested = sorted(k for k, v in env.items()
                            if isinstance(v, dict | list | type(None)))
            if nested:
                # `str()` of a container yields a garbage variable like `K="['x']"`, which
                # the peer then reads as if it were meaningful. Scalars coerce; containers
                # are a mistake worth naming.
                raise ValueError(
                    f"server '{name}': env value(s) {', '.join(nested)} are not scalars — "
                    f"an MCP `env` maps names to strings")
            # COERCE to str->str, as the sibling `headers` handling below already does.
            # A client's own spawn coerces (`{"PORT": 3000}` is a working Claude Code
            # entry today, and a plain `wrap` preserves it untouched), but the router
            # reads this file with `load_multi_config`, which rejects a non-string value —
            # and that failure kills the WHOLE fleet at launch, not just this peer, on a
            # config the installer reported as a success.
            spec["env"] = {str(k): str(v) for k, v in env.items()}
        if original.get("cwd"):
            spec["cwd"] = original["cwd"]
    elif original.get("url"):
        spec["url"] = original["url"]
        if original.get("headers"):
            spec["headers"] = original["headers"]
    else:
        raise ValueError(
            f"server '{name}' has no 'command' or 'url' to wrap — it doesn't "
            f"look like a valid MCP server entry")
    return spec


def wrap_multi(config: dict, stash: dict, servers: list[str], policy: str,
               terse_cmd: list[str], *, router: str = DEFAULT_ROUTER,
               peers_file: str, proxy_opts: list[str] | None = None,
               existing_peers: dict | None = None) -> tuple[dict, dict, dict]:
    """Collapse `servers` into ONE router entry fronting them all (#179).

    The stash stays 1:1 — each server is stashed under its own name exactly as `wrap`
    does — but its live entry is DELETED rather than rewritten, and a single `router`
    entry running `proxy --config` is added. Keeping the stash shape means
    `uninstall-mcp --all` needs no special case: it still iterates the same keys.

    The PEERS FILE is the sole record of which servers a router fronts. That is
    deliberate — a second copy in the stash could disagree with it, and the peers file
    is the thing the proxy actually reads.

    Idempotent the same way `wrap` is: a server already managed is re-described from its
    stashed original, so re-running never nests a proxy inside a proxy.

    ADDITIVE across runs: `existing_peers` (the peers file already on disk) is merged
    rather than replaced, so `install-mcp c --multiproxy` after `install-mcp a b
    --multiproxy` leaves a fleet of three. Overwriting wholesale made "add one server to
    the fleet" — the single most likely second invocation — silently evict the other two,
    which stay stashed and therefore invisible to the client entirely: not a degraded
    fleet, a vanished one. A retained peer must still be in the STASH; one that was
    uninstalled (its live entry restored) is stale bookkeeping and is dropped rather than
    resurrected behind the router."""
    live_servers = config.setdefault("mcpServers", {})
    requested = set(servers)
    # Staleness is LIVENESS, not stash membership. A peer is stale exactly when its live
    # entry came back — which is what `do_uninstall` does when it detaches one. Testing
    # `in stash` instead evicted any peer whose stash entry had drifted (#172 says that
    # happens), and the peers file is precisely the LAST surviving record of how to launch
    # such a peer: it would vanish from the stash, the live config, the peers file and
    # status all at once, deleted by a run that never named it, with no `changes` row and
    # no backup (`_backup` covers the client config only).
    prior_router = _detect_router(config, Path(peers_file))
    downstreams: list[dict] = [
        d for d in ((existing_peers or {}).get("downstreams") or [])
        if isinstance(d, dict) and isinstance(d.get("name"), str)
        and d["name"] not in requested and d["name"] not in live_servers
    ]
    seen_peers: set[str] = set()
    for name in servers:
        if name in seen_peers:
            # `servers` comes from a `nargs="+"` positional, so `install-mcp a a` is one
            # typo away. Appending twice put the SAME peer in `downstreams` twice: the
            # downstream launched twice and every one of its tools exported twice — the
            # double-serve failure class already guarded for hand-edits, reachable here
            # from ordinary argv.
            continue
        seen_peers.add(name)
        if name == prior_router:
            # Folding the router into its own peers file writes a peer whose command is
            # `terse proxy --config <this very file>`: starting it spawns a router that
            # spawns a router, unbounded, on the next client restart. `_unnest` cannot
            # catch it (a router entry has no `--`), so it has to be refused by name.
            raise ValueError(
                f"'{name}' is the router fronting {peers_file} — it cannot also be one "
                f"of its own peers (that would launch a router inside itself)")
        if _is_router_entry(live_servers.get(name)):
            # ANY terse router, not just this peers file's. `_unnest` recovers a downstream
            # from the `--` a `wrap`ped entry carries; a router has `--config` and no `--`,
            # so it comes through VERBATIM and becomes a peer whose command is
            # `terse proxy --config <some other fleet>` — a proxy nested in a proxy, the
            # double-primer cost this whole feature exists to remove. The `name ==
            # prior_router` check above only covers the router for THIS file.
            raise ValueError(
                f"'{name}' is itself a terse multiproxy router (it runs `proxy --config`) "
                f"— folding it in would nest a proxy inside a proxy. Uninstall its fleet "
                f"first, or name its peers individually")
        if name in stash:
            original = stash[name]
        elif name in live_servers:
            # `_unnest` FIRST, then stash: a server can be live-wrapped with its stash
            # under another scope or missing (#172), and stashing the wrapper would make
            # `uninstall` restore `terse proxy --policy /old.json -- kb-mcp` while
            # reporting `restored: True` — the entry comes back still wrapped, pointing
            # at a policy that may no longer exist. The peers file already records the
            # unnested downstream; the stash has to agree with it.
            original = _unnest(live_servers[name])
            stash[name] = original
        else:
            raise KeyError(name)
        downstreams.append(_peer_spec(name, original, policy))
        live_servers.pop(name, None)

    # A rename (`--router-name` differing from the router already fronting THIS peers
    # file) must move the entry, not add a second one: two entries running the same
    # `proxy --config` launch every peer twice and export every tool twice, and
    # `_detect_router` then sees two matches, returns None, and neither status nor
    # `uninstall-mcp --all` can clean up the config at all.
    prior_entry = None
    if prior_router and prior_router != router:
        prior_entry = live_servers.pop(prior_router, None)
    # `prior_entry` is the fallback base so a RENAME carries the router's hand-edited keys
    # (an `env.PATH` pin is the router's base environment, inherited by every peer) instead
    # of dropping them — the same drift loss `wrap`'s guard exists to prevent, which the
    # rename path had reintroduced by looking the base up under the NEW name only.
    live_router = live_servers.get(router) or prior_entry
    new_entry = {k: v for k, v in (live_router or {}).items()
                 if k not in ("command", "args", "url", "headers")}
    new_entry["command"] = terse_cmd[0]
    new_entry["args"] = [*terse_cmd[1:], "proxy", *(proxy_opts or []),
                         "--config", peers_file]
    live_servers[router] = new_entry
    return config, stash, {"downstreams": downstreams}


def _parse_router_opts(entry: dict | None) -> dict:
    """The runtime flags baked onto an existing router entry, as `_runtime_opts` kwargs.
    All-defaults when there is no router yet — the inverse of `_runtime_opts`, so a flag
    survives a round-trip through the config instead of being reset by the next run."""
    args = list((entry or {}).get("args") or [])
    kf = None
    if "--diff-keyframe-interval" in args:
        i = args.index("--diff-keyframe-interval")
        if i + 1 < len(args) and args[i + 1].isdigit():
            kf = int(args[i + 1])
    capture = None
    if "--capture-dir" in args:
        i = args.index("--capture-dir")
        if i + 1 < len(args):
            capture = args[i + 1]
    return {"capture_dir": capture,
            "no_stats": "--no-stats" in args,
            "diff": True if "--diff" in args else (False if "--no-diff" in args else None),
            "diff_keyframe_interval": kf,
            "no_join_blocks": "--no-join-blocks" in args}


def _detect_router(node: dict, peers_p: Path) -> str | None:
    """The multiproxy router entry in `node`, or None (#179).

    Three conditions, all necessary. `_looks_like_terse_launcher` — an unrelated server
    whose own CLI happens to take a `--config` flag is NOT a terse router, and treating
    it as one would delete a third party's entry when its last "peer" left. The
    `--config` VALUE must be THIS scope's peers file — user and local scope share one
    `~/.claude.json`, so the other scope's router is visible here and must not be
    claimed. And exactly one match: two routers pointing at the same peers file is a
    hand-edit terse should not guess its way through, so it reports None and leaves the
    entries alone rather than pruning an arbitrary one."""
    routers = _detect_routers(node, peers_p)
    return routers[0] if len(routers) == 1 else None


def _entry_from_peer_spec(spec: dict) -> dict | None:
    """The inverse of `_peer_spec`: an mcpServers entry rebuilt from a peers-file record,
    or None when the record cannot launch anything.

    Best-effort by construction — the peers file records only what the ROUTER needs to
    launch the peer, so a `type` or any other key of the pre-terse entry is not in it.
    Used only to recover a folded peer whose stash entry drifted away; the caller reports
    the restore as partial so nobody mistakes it for the byte-for-byte stash path.

    Returning None matters: `peers_downstreams` only checks for a `name`, so a hand-edited
    record with a name and no `command`/`url` reaches here, and the old code wrote
    `{"url": None}` into the live config — an entry no client can launch, presented as a
    successful restore."""
    cmd = spec.get("command")
    if isinstance(cmd, list) and cmd and isinstance(cmd[0], str):
        entry: dict = {"command": cmd[0]}
        if len(cmd) > 1:
            entry["args"] = list(cmd[1:])
    elif isinstance(spec.get("url"), str) and spec["url"]:
        entry = {"url": spec["url"]}
        if spec.get("headers"):
            entry["headers"] = spec["headers"]
    else:
        return None
    for k in ("env", "cwd"):
        if spec.get(k):
            entry[k] = spec[k]
    return entry


def _is_router_entry(entry: object) -> bool:
    """True if `entry` is a terse MULTIPROXY router — a terse launcher carrying `--config`
    — regardless of which peers file it fronts. `_detect_routers` answers the narrower
    "is it THIS scope's router"; this one is what a fold has to refuse."""
    if not isinstance(entry, dict) or not _looks_like_terse_launcher(entry):
        return False
    return "--config" in list(entry.get("args") or [])


def _detect_routers(node: dict, peers_p: Path) -> list[str]:
    """EVERY entry fronting `peers_p`. More than one is a hand-edit terse must not guess
    its way through — but it must still be able to NAME them, because the ambiguous state
    is otherwise unreportable and unrepairable: status showed both entries as
    unrecoverable, uninstall deleted the peers file out from under them, and the error
    message's own advice (`--router-name`) added a third."""
    want = str(peers_p)
    routers = []
    for name, entry in (node.get("mcpServers") or {}).items():
        if not isinstance(entry, dict) or not _looks_like_terse_launcher(entry):
            continue
        args = list(entry.get("args") or [])
        if "--config" in args:
            i = args.index("--config")
            if i + 1 < len(args) and args[i + 1] == want:
                routers.append(name)
    return routers


def _prune_peer(peers_doc: dict, server: str) -> bool:
    """Drop `server` from a peers doc. Returns True if it was there. Also normalizes the
    list to its valid entries (see `peers_downstreams`) so a malformed leftover can never
    keep `downstreams` non-empty and strand the router entry forever."""
    downs = peers_doc.get("downstreams")
    if not isinstance(downs, list):
        return False
    kept = [d for d in peers_downstreams(peers_doc) if d.get("name") != server]
    was_there = any(isinstance(d, dict) and d.get("name") == server for d in downs)
    peers_doc["downstreams"] = kept
    return was_there


def unwrap(config: dict, stash: dict, server: str,
           peers_doc: dict | None = None, router: str | None = None) -> tuple[dict, dict]:
    """Restore `server`'s original entry from the stash (byte-for-byte).

    With a `peers_doc`, also detach the server from a multiproxy router (#179): prune it
    from `downstreams`, and remove the router entry entirely once its last peer leaves —
    otherwise an uninstall would leave a router process fronting nothing, which starts
    cleanly and serves zero tools, the most confusing possible end state."""
    if server not in stash:
        raise KeyError(server)
    servers = config.setdefault("mcpServers", {})
    servers[server] = stash.pop(server)
    if peers_doc is not None:
        _prune_peer(peers_doc, server)
        # Gated on "no peers REMAIN", not on "this server's prune fired". Requiring the
        # prune made router removal unreachable whenever the peers file had already
        # drifted — `downstreams: []`, a non-list, or one malformed entry — leaving an
        # entry that runs `terse proxy --config <nothing usable>` and exits 2 on every
        # client start, while status called it the healthy `router` state.
        if router and not peers_downstreams(peers_doc):
            servers.pop(router, None)
    return config, stash


def is_managed(stash: dict, server: str) -> bool:
    return server in stash


def _looks_like_terse_launcher(entry: dict) -> bool:
    """True if `entry` launches via terse. Covers every form `terse_invocation` /
    `$TERSE_MCP_CMD` can emit: the console script `terse` as `command`, `python -m terse`,
    and `uvx terse` / `uv tool run terse` (where `terse` is a bare token in `args`). A
    launcher this misses would drop that server's baked `--policy` from the ambiguity set —
    which could make a genuinely multi-policy wire look unambiguous — so it errs toward
    detection; a false positive is caught anyway by `parse_proxy_opts` requiring a `proxy`
    subcommand."""
    cmd = entry.get("command")
    if isinstance(cmd, str) and Path(cmd).name == "terse":
        return True
    args = entry.get("args")
    return isinstance(args, list) and "terse" in args


def parse_proxy_opts(entry: dict) -> dict[str, str] | None:
    """The terse proxy options baked into a wrapped MCP server `entry`'s args —
    `{'policy','capture_dir','server_name'}` for whichever keys are present — or None
    when `entry` is not a terse-wrapped entry.

    Only the segment BETWEEN the `proxy` subcommand and the first `--` (the downstream
    boundary `wrap` writes) is scanned, so a value on the DOWNSTREAM side that happens to
    equal `--policy` (e.g. the wrapped server's own flag) is never misread as terse's."""
    if not _looks_like_terse_launcher(entry):
        return None
    args = entry.get("args")
    if not isinstance(args, list):
        return None
    try:
        start = args.index("proxy")
    except ValueError:
        return None
    try:
        end = args.index("--", start + 1)
    except ValueError:
        end = len(args)   # no downstream separator (unusual) — scan to the end
    seg = args[start + 1:end]
    flag_map = {"--policy": "policy", "--capture-dir": "capture_dir",
                "--server-name": "server_name"}
    opts: dict[str, str] = {}
    i = 0
    while i < len(seg):
        key = flag_map.get(seg[i]) if isinstance(seg[i], str) else None
        if key is not None and i + 1 < len(seg) and isinstance(seg[i + 1], str):
            opts[key] = seg[i + 1]
            i += 2
        else:
            i += 1
    return opts


def discover_wrapped_opts(config: dict) -> list[dict[str, str]]:
    """Every terse-wrapped server in `config`'s top-level `mcpServers`, each as its baked
    proxy opts plus `{'server': name}`. Order-preserving; empty when none are wrapped.
    Read-only and scope-agnostic (user/project shape); the caller decides what to do with
    multiple distinct policies/corpora."""
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    out: list[dict[str, str]] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        opts = parse_proxy_opts(entry)
        if opts is not None:
            out.append({"server": name, **opts})
    return out


def discover_wrapped_opts_all_scopes(*, cfg: Path | None = None, file: str | None = None,
                                     repo_path: str | None = None) -> list[dict[str, str]]:
    """`discover_wrapped_opts`, across all three scopes — same target resolution as
    `scan_scopes` (user, project, local), so a bare `terse policy autotune` sees
    project- and local-scope wrapped servers, not just user-scope ones (#167 left this
    as a follow-up: the autotune wiring resolver only ever read `config_path()`, so an
    install wrapped entirely at project or local scope reported "no terse-wrapped
    servers found" and fell through to requiring explicit `--policy`/`--corpus`, even
    though the entries were right there in `.mcp.json` / the local `projects` block).
    Order-preserving (user, then project, then local); local scope is silently omitted
    when it doesn't resolve (not a git repo, no --repo-path), matching `scan_scopes`.
    Read-only, never raises: a corrupt/unreadable file in ONE scope is skipped (treated
    as no wrapped servers there), not allowed to blind resolution of the other two — user
    and local scope share the exact same physical file by default (`resolve_target`'s
    `cfg or config_path()`), so letting a broken `.mcp.json` (project) abort the whole
    call would cost a WORKING `~/.claude.json` (user/local) resolution too, and the
    reverse. That physical file is also read and parsed only once, not once per scope
    that happens to point at it."""
    targets = [resolve_target("user", cfg=cfg), resolve_target("project", file=file)]
    try:
        targets.append(resolve_target("local", cfg=cfg, repo_path=repo_path))
    except ValueError:
        pass  # no local scope here (not a git repo, no --repo-path) — same as scan_scopes
    out: list[dict[str, str]] = []
    loaded: dict[Path, dict] = {}
    for target in targets:
        if target.cfg not in loaded:
            try:
                loaded[target.cfg] = _load_json(target.cfg)
            except (OSError, ValueError):
                loaded[target.cfg] = {}
        # `_read_servers_root` returns the node THAT HOLDS `mcpServers` (itself, for
        # user/project scope's server_path=(); the per-repo block, for local) — the same
        # shape `discover_wrapped_opts` already unwraps via `config.get("mcpServers")`.
        # Re-wrapping it as `{"mcpServers": node}` here would nest it one level too deep
        # and silently discover nothing (caught by this function's own new tests).
        node = _read_servers_root(loaded[target.cfg], target.server_path)
        out += discover_wrapped_opts(node)
    return out


# ------------------------------------------------------------------ IO helpers
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def peers_downstreams(peers_doc: dict | None) -> list[dict]:
    """The VALID peer entries in a peers doc — dicts carrying a string `name`.

    One definition, used by every caller, because two callers disagreeing about what
    counts is how the router outlived its fleet: `_prune_peer` kept a nameless entry
    (so `downstreams` never emptied and the router was never removed) while `wrap_multi`
    dropped it. A hand-edited peers file is an operator-facing artifact — USAGE tells
    them where it lives — so malformed entries are expected input, not impossible."""
    return [d for d in ((peers_doc or {}).get("downstreams") or [])
            if isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"]]


def load_peers(path: Path) -> tuple[dict | None, str | None]:
    """Read a peers file, returning `(doc, error)` — never raising.

    A corrupt peers file used to traceback out of `mcp-status` (whose contract says it
    never raises) and to block `install-mcp`, `install-mcp --multiproxy`, `uninstall-mcp`
    and `uninstall-mcp --all` with a bare `json.decoder` message naming NO file: every
    route out of the state was closed, and nothing said which file to fix. `doc` is None
    when there is nothing usable; `error` is a message that names the path."""
    if not path.exists():
        return None, None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{path}: unreadable peers file ({e})"
    if not isinstance(doc, dict):
        return None, f"{path}: peers file must be a JSON object"
    return doc, None


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
               no_join_blocks: bool = False,
               never_lossy: bool = False,
               multiproxy: bool = False,
               router: str = DEFAULT_ROUTER) -> dict:
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
    # Validated inside terse_invocation when it comes from $TERSE_MCP_CMD — as strictly
    # as the policy path above. A bad policy path fails loudly the first time the proxy
    # starts; a bad launcher fails *silently*, because the MCP client cannot spawn the
    # entry at all, so the server just appears with no tools and nothing says why.
    terse_cmd = terse_invocation()

    available = sorted((node.get("mcpServers") or {}).keys())
    managed = set(stash)
    missing = [s for s in servers if s not in set(available) and s not in managed]
    if missing:
        raise ValueError(
            f"unknown server(s): {', '.join(missing)}. "
            f"available: {', '.join(available) or '(none)'}")
    # Re-running a plain `install-mcp` on a server that is currently FOLDED would restore
    # it as a standalone wrapped entry while the router keeps launching it too: the same
    # downstream running twice, every one of its tools exported twice, at double cost and
    # with nothing to say why. Forgetting the flag on a policy refresh is all it takes.
    if not multiproxy:
        peers_p = peers_path(target.cfg, target.stash_prefix)
        peers_now, peers_err = load_peers(peers_p)
        if peers_err:
            raise ValueError(peers_err)
        folded_now = {d["name"] for d in peers_downstreams(peers_now)}
        clash = [s for s in servers if s in folded_now]
        if clash:
            raise ValueError(
                f"{', '.join(clash)} already folded behind a --multiproxy router "
                f"({peers_p}) — re-run with --multiproxy to refresh the fleet, or "
                f"`uninstall-mcp {' '.join(clash)}` first to detach and wrap standalone")
        # ...and the router itself: `wrap` would nest `terse proxy --policy ... -- terse
        # proxy --config ...`, charging the primer twice — the exact cost --multiproxy
        # exists to remove — while `_detect_router` still matches it (the `--config` token
        # survives past the `--`), so status reports the nested entry as a healthy router.
        this_router = _detect_router(node, peers_p)
        if this_router and this_router in servers:
            raise ValueError(
                f"'{this_router}' is a --multiproxy router, not a wrappable server — "
                f"wrapping it would nest a proxy inside a proxy. Re-run with "
                f"--multiproxy to change the fleet, or `uninstall-mcp --all` first")
    if multiproxy:
        return _install_multiproxy(
            target, config, full_stash, stash, node, servers, policy_abs, terse_cmd,
            router=router, dry_run=dry_run, scope=scope, available=available,
            had_nl=had_nl, capture_dir=capture_abs, diff=diff,
            diff_keyframe_interval=diff_keyframe_interval, no_stats=no_stats,
            no_join_blocks=no_join_blocks, never_lossy=never_lossy)

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
             no_join_blocks=no_join_blocks, no_stats=no_stats)
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
        result["never_lossy_added"] = _apply_never_lossy(policy_abs, servers,
                                                         dry_run=dry_run)
    return result


def _apply_never_lossy(policy_abs: str, servers: list[str], *, dry_run: bool) -> list[str]:
    """Bake `servers` into the policy file's `never_lossy_servers` (PR #89); return the
    names actually added. Scope-independent and router-independent: the runtime matches
    on `--server-name`, which multiproxy sets per PEER from the peers file, so a folded
    server is protected by exactly the same name it had standalone."""
    pol_doc = json.loads(Path(policy_abs).read_text(encoding="utf-8"))
    added = [s for s in servers if add_never_lossy_server(pol_doc, s)]
    if added and not dry_run:
        _write_json(Path(policy_abs), pol_doc)
    return added


def allowlist_mapping(servers: list[str], router: str) -> list[dict]:
    """The permission-entry rewrite a multiproxy switch forces (#179).

    A Claude Code permission is either `mcp__<server>` (every tool of one server) or
    `mcp__<server>__<tool>` (one tool). Both forms are emitted per server, because both
    are forms real settings files actually hold — the earlier single `mcp__kb__*` row
    described a glob shape that is not one of them, so an operator matching on it found
    nothing to edit and concluded there was nothing to do.

    `widens` is the part that is a SECURITY change, not a rename: N per-server grants
    collapse onto one `mcp__terse` server segment, so a whole-server grant that used to
    reach one server now reaches every peer behind the router. Say it here rather than
    leave it to be discovered — but only when the fleet actually has more than one peer,
    or the warning fires on a one-server router where nothing widened and stops carrying
    signal by the time it matters.

    The TOOL segment additionally changes for any name two or more peers both export
    (`definition` -> `lsp-go__definition`), but install time has no tool names — that
    needs a live `tools/list` per peer — so it is flagged as a caveat rather than
    guessed. Guessing here would be worse than silence: a wrong mapping reads as
    authoritative and sends the operator to edit the wrong entries."""
    return [{"server": s,
             "from": f"mcp__{s}", "to": f"mcp__{router}",
             "from_tool": f"mcp__{s}__<tool>", "to_tool": f"mcp__{router}__<tool>",
             "widens": len(servers) > 1}
            for s in servers]


def _install_multiproxy(target, config: dict, full_stash: dict, stash: dict, node: dict,
                        servers: list[str], policy_abs: str, terse_cmd: list[str], *,
                        router: str, dry_run: bool, scope: str, available: list[str],
                        had_nl: bool, capture_dir: str | None = None,
                        diff: bool | None = None,
                        diff_keyframe_interval: int | None = None,
                        no_stats: bool = False, no_join_blocks: bool = False,
                        never_lossy: bool = False) -> dict:
    """`install-mcp --multiproxy` (#179): N entries -> one router entry + a peers file.

    This is the step that banks #168's measured win — six standalone proxies cost +23.1%
    raw input against an unwrapped control, the same six behind one router cost +0.0%,
    because each standalone proxy injects its own primer that the client re-reads every
    turn."""
    if router in servers:
        raise ValueError(
            f"router name '{router}' is also a server being wrapped — pick another with "
            f"--router-name, or the router entry would overwrite the peer it fronts")
    peers_p = peers_path(target.cfg, target.stash_prefix)
    peers_file = str(peers_p)
    existing_peers, peers_err = load_peers(peers_p)
    if peers_err:
        raise ValueError(peers_err)
    live_now = node.get("mcpServers") or {}
    all_routers = _detect_routers(node, peers_p)
    if len(all_routers) > 1:
        # Two entries already front this peers file (a hand-edit). Folding again would
        # write a third — which is exactly what the old "pick another name with
        # --router-name" advice produced, permanently poisoning detection. Name the
        # duplicates and stop.
        raise ValueError(
            f"{', '.join(sorted(all_routers))} all front {peers_file} — terse can't tell "
            f"which is the real router. Delete the duplicate entries from the config "
            f"(they are interchangeable; keep one), then re-run")
    # The router entry is WRITTEN OVER, not stashed — so a live entry that merely happens
    # to share the router's name is destroyed with no way back. `terse` is the DEFAULT
    # router name, so this needs no unusual flag to hit. A terse-owned router (this
    # scope's, or one being renamed) is fine to overwrite; anything else is refused.
    current_router = all_routers[0] if all_routers else None
    folded_now = {d["name"] for d in peers_downstreams(existing_peers)}
    if router != current_router and (router in live_now or router in stash
                                     or router in folded_now):
        # A FOLDED peer has no live entry by construction, so `router in live_now` alone
        # could not see it — and naming the router after one made `unwrap` later write
        # that peer's original OVER the router entry, silently stranding every other peer
        # (stashed, no live entry, no router left to serve them) while reporting success.
        raise ValueError(
            f"'{router}' is already a server terse manages or that exists in this config, "
            f"and the router entry is written over rather than stashed — folding here "
            f"would destroy it with nothing to restore from. Pick another name with "
            f"--router-name.")
    before = {s: (node.get("mcpServers") or {}).get(s) for s in servers}
    # The router honors every runtime flag `wrap` bakes onto a single-server entry —
    # `terse proxy` accepts them with `--config` too — so they must ride along here or
    # `--capture-dir`/`--no-stats`/`--diff` would be silently dropped by the very switch
    # that is supposed to be behavior-preserving.
    # ALL-OR-NOTHING inherit from the router already in place. An additive run ("fold one
    # more server in") names the new server, not the flags the fleet was installed with,
    # so rebuilding the args from that invocation alone silently cleared them for every
    # peer at once — a capture run that records nothing. But a PER-FLAG `or`-merge is
    # worse in the other direction: `--no-stats` and `--capture-dir` have no inverse flag,
    # so once set they could never be cleared except by hand-editing the config, and
    # `wrap`'s documented contract for a single-server entry is the opposite ("a re-wrap
    # always reflects the latest invocation, flags never accumulate"). So: an invocation
    # that names NO runtime flag inherits the whole set; one that names ANY defines the
    # whole set, which is also how you clear them.
    given = _runtime_opts(capture_dir=capture_dir, no_stats=no_stats, diff=diff,
                          diff_keyframe_interval=diff_keyframe_interval,
                          no_join_blocks=no_join_blocks)
    effective = ({"capture_dir": capture_dir, "no_stats": no_stats, "diff": diff,
                  "diff_keyframe_interval": diff_keyframe_interval,
                  "no_join_blocks": no_join_blocks} if given
                 else _parse_router_opts(live_now.get(current_router or "")))
    proxy_opts = _runtime_opts(**effective)
    _, _, peers_doc = wrap_multi(node, stash, servers, policy_abs, terse_cmd,
                                 router=router, peers_file=peers_file,
                                 proxy_opts=proxy_opts, existing_peers=existing_peers)
    changes: list[dict] = [{"server": s, "before": before[s], "after": None,
                            "preserved": [], "folded_into": router} for s in servers]
    fleet = [d.get("name") for d in peers_doc["downstreams"]]
    result = {"config": str(target.cfg), "scope": scope, "policy": policy_abs,
              "available": available, "changes": changes, "dry_run": dry_run,
              # The EFFECTIVE values (what actually got baked into the entry), not this
              # invocation's — an additive run that inherited `--capture-dir` otherwise
              # reported `capture_dir: None`, so `--print` printed no capture line and
              # cli's autotune follow-up hint never fired for any additive fleet run.
              "backup": None, "capture_dir": effective["capture_dir"],
              "diff": effective["diff"], "no_stats": effective["no_stats"],
              "never_lossy_added": [], "multiproxy": True, "router": router,
              "router_entry": node["mcpServers"][router], "peers_file": peers_file,
              "peers": peers_doc, "fleet": fleet,
              # The permission rewrite covers the WHOLE fleet, not just this run's
              # servers: a peer folded in by an earlier run keeps needing its
              # `mcp__<server>__*` grant remapped, and reporting only today's argument
              # list would read as "the others are fine".
              "allowlist": allowlist_mapping([f for f in fleet if f], router)}
    if not dry_run:
        result["backup"] = str(_backup(target.cfg))
        # RECOVERY DATA FIRST, then the destructive write. Each `_write_json` is atomic on
        # its own (mkstemp + os.replace) but the three together are not, and `wrap_multi`
        # DELETES a folded peer's live entry rather than rewriting it the way `wrap` does.
        # Config-first therefore had a window — one SIGKILL, OOM, or full disk wide — where
        # the live entry was already gone while the stash and peers file still described
        # the previous state: the original existed nowhere terse looks, so status reported
        # nothing missing and `uninstall --all` never mentioned the server at all. Only the
        # timestamped config backup held it, which no recovery path reads and
        # `_prune_backups` eventually deletes.
        #
        # In this order a crash before the config write leaves every original LIVE and
        # untouched; the stash and peers file merely describe a fold that didn't happen,
        # which the next run reconciles and which status already reports loudly
        # (`folded-and-live`). Recoverable beats silent.
        _write_json(stash_path(target.cfg), full_stash)
        _write_json(peers_p, peers_doc)
        _write_json(target.cfg, config, trailing_newline=had_nl)
    if never_lossy:
        result["never_lossy_added"] = _apply_never_lossy(policy_abs, servers,
                                                          dry_run=dry_run)
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


def _peers_policy(peers_doc: dict | None) -> str | None:
    """The one policy path every peer shares, or None when they differ (or there are no
    peers). Status shows a single `policy=` field, so reporting one peer's path when the
    fleet is mixed would be a confident lie; None reads as "look in the peers file"."""
    paths = {d.get("policy") for d in ((peers_doc or {}).get("downstreams") or [])
             if isinstance(d, dict)}
    return paths.pop() if len(paths) == 1 else None


def _peers_diff_label(peers_doc: dict | None) -> str | None:
    """The diff label a router should print: the one label every peer agrees on, or
    `peers (mixed)` when they genuinely disagree. None when there are no peers.

    `_peers_policy` returns None for a fleet whose peers carry DIFFERENT policy paths —
    correct for the `policy=` column, which has one slot and must not name one peer's file
    as the fleet's. But letting that None reach `_default_diff_label` printed the dataclass
    default `default (off)` while every peer diffed, which is #181's divergence surviving
    the first half of #191's fix. Two peers with different paths can still agree on `diff`,
    and that answer is knowable: resolve PER PEER and say "mixed" only when the answers
    differ, not when the paths do."""
    labels = {_default_diff_label(d.get("policy"))
              for d in ((peers_doc or {}).get("downstreams") or []) if isinstance(d, dict)}
    if not labels:
        return None
    return labels.pop() if len(labels) == 1 else "peers (mixed)"


def _scan_target(target: Target, scope: str) -> list[dict]:
    if not target.cfg.exists():
        return []
    config = _load_json(target.cfg)
    node = _read_servers_root(config, target.server_path)
    servers = node.get("mcpServers") or {}
    full_stash = _load_stash(stash_path(target.cfg))
    stash = full_stash.get(target.stash_prefix, {})
    # A multiproxy install (#179) is invisible to the two-state stash/live classification
    # below and reads as drift in BOTH directions: every folded peer is stashed-but-absent
    # (`orphaned-stash`), and the router itself launches via terse with no stash of its own
    # (`wrapped-unstashed`, "original command unrecoverable"). Both are healthy here — the
    # peers file is what says so, so it is read before classifying, not after.
    peers_p = peers_path(target.cfg, target.stash_prefix)
    # `load_peers` never raises: a corrupt peers file used to traceback straight out of
    # `mcp-status`, whose own contract says it never raises — and status is exactly where
    # you look when the config is broken.
    peers_doc, peers_err = load_peers(peers_p)
    all_routers = _detect_routers(node, peers_p)
    router_name = all_routers[0] if len(all_routers) == 1 else None
    folded = {d["name"] for d in peers_downstreams(peers_doc)}

    rows = []
    # Every folded name gets a row even with no stash entry and no live entry — otherwise
    # a peer whose stash drifted away appeared NOWHERE while the router kept launching it.
    for name in sorted(set(servers) | set(stash) | folded):
        stashed = name in stash
        present = name in servers
        # Classify from the CONFIG, with the stash as recovery data rather than as the
        # source of truth (#172). Keying purely on stash membership misfiled an entry
        # that plainly launches via terse as "unwrapped" whenever its stash entry was
        # missing — and since `do_uninstall` iterates the stash, that server could not be
        # unwrapped by terse at all: wrapped in the live config, absent from status,
        # skipped by `uninstall-mcp --all`. That is the exact inverse of the drift #58
        # surfaced as `orphaned-stash`; only one direction had been considered.
        launches_via_terse = present and _looks_like_terse_launcher(servers[name])
        if name in all_routers:
            # Two entries fronting one peers file is a hand-edit terse won't guess through
            # (`_detect_routers`), but it must NAME the state — reporting both as
            # `wrapped-unstashed` sent the operator to look for a `--` a router entry
            # doesn't have, and nothing said the duplication was the problem.
            state = "router" if router_name else "router-ambiguous"
        elif name in folded and present:
            # Listed as a peer AND live: the same downstream runs twice, every one of its
            # tools exported twice at double cost. `do_install` refuses to create this,
            # but a hand-edit or `claude mcp add <name>` still can, and the old
            # classification called it a plain `wrapped` entry.
            state = "folded-and-live"
        elif name in folded and not stashed:
            # Folded with its stash entry gone. Not healthy — but not lost either: the
            # peers file records enough to relaunch it, and `uninstall-mcp --all` now
            # rebuilds from exactly that.
            state = "folded-unstashed"
        elif stashed and not present and name in folded:
            # Folded behind the router: its live entry is GONE by design (that is the
            # whole point — one entry instead of N), and its original is safely stashed.
            state = "folded"
        elif present and (stashed or launches_via_terse):
            # `wrapped-unstashed` is deliberately its own state, not a flag: the original
            # command is UNRECOVERABLE, so `uninstall-mcp` cannot restore this entry and
            # `install-mcp`'s idempotence (re-wrap from the stash) does not hold either.
            # Reporting it as plain `wrapped` would hide a real, actionable loss.
            state = "wrapped" if stashed else "wrapped-unstashed"
        elif stashed and not present:
            # A stash entry with no matching mcpServers entry — the entry was removed
            # or edited by hand after terse wrapped it. Surfacing this is the whole
            # point of #58: this exact kind of scope/state drift is what prompted it.
            state = "orphaned-stash"
        else:
            state = "unwrapped"
        policy = None
        policy_missing = False
        launcher = None
        launcher_gone = False
        wraps = None
        diff = None
        stats_on = None
        if state in ("wrapped", "wrapped-unstashed", "router", "router-ambiguous"):
            # The launcher (`command`) is the entry's most silent failure mode: if it
            # no longer resolves, the client can't spawn the proxy at all and the server
            # just shows up with no tools. That is exactly what an upgrade moving a
            # versioned uv-tool/pipx venv does to every wrapped entry at once, so status
            # is where it has to be visible.
            launcher = servers[name].get("command")
            if isinstance(launcher, str):
                launcher_gone = launcher_missing([launcher]) is not None
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
            downstream: list[str] = []
            if "--" in args:
                downstream = args[args.index("--") + 1:]
                if downstream:
                    wraps = " ".join(downstream)
            # A router's policy lives in the peers file, not in `--policy` (it carries
            # `--config`), and it has to resolve BEFORE the label below reads it: resolving
            # it afterwards made every router row print the dataclass default `default
            # (off)` even when the shared peers policy set `"diff": true` — the proxy
            # diffs, the status line said it did not. That is #181's divergence again, in
            # the one branch #188/#190 didn't touch (#191).
            if state in ("router", "router-ambiguous"):
                policy = policy or _peers_policy(peers_doc)
                # Not `_default_diff_label(policy)`: a mixed-path fleet leaves `policy` None
                # by design, and that None would print `default (off)` over peers that diff.
                default_label = _peers_diff_label(peers_doc) or _default_diff_label(None)
            else:
                default_label = _default_diff_label(policy)
            # The router's own `--diff` / `--no-diff` still wins outright — `_build_peers`
            # applies the CLI flag over every peer's policy, so the label must too.
            diff = ("off" if "--no-diff" in args
                    else "on" if "--diff" in args else default_label)
            stats_on = "--no-stats" not in args
        ledger_identity = None
        ledger_identity_explicit = None
        if state in ("wrapped", "wrapped-unstashed") and downstream:
            # `resolve_ledger_identity` is the SAME rule `proxy.py`'s live write path
            # uses, imported rather than re-derived — a review round caught this and
            # `proxy.py`'s copy diverging in principle before this existed. A standalone
            # entry with no baked `--server-name` writes ledger records under a GUESSED
            # identity (the downstream command's basename), which splits from any other
            # install of the SAME logical server that guesses differently — e.g. a
            # hand-written entry launching `runecho-mcp` next to an `install-mcp`-managed
            # one launching the same server as `runecho`. `install-mcp` always bakes
            # `--server-name` (#152); a missing one here means a hand-edited entry, which
            # is exactly the case this can't catch any other way (#237's boundary keeps
            # this a detector, not a proxy behavior change — nothing here alters routing).
            opts = parse_proxy_opts(servers[name]) or {}
            explicit_name = opts.get("server_name")
            ledger_identity = resolve_ledger_identity(explicit_name, downstream)
            ledger_identity_explicit = explicit_name is not None
        if state in ("router", "router-ambiguous"):
            # A router has no `--` downstream and no single `--policy`: what it fronts is
            # the peers file's list, and each peer carries its own policy there.
            wraps = ", ".join(sorted(folded)) or "(no peers)"
            policy_missing = bool(policy and os.path.isabs(policy)
                                  and not os.path.exists(policy))
        rows.append({"scope": scope, "server": name, "state": state, "policy": policy,
                    "policy_missing": policy_missing, "launcher": launcher,
                    "launcher_missing": launcher_gone, "wraps": wraps, "diff": diff,
                    "stats": stats_on, "config": str(target.cfg),
                    # Which router a peer sits behind — the one fact a folded row can't
                    # otherwise state, and the first thing you need to un-fold it.
                    "router": (router_name if state in ("folded", "folded-unstashed",
                                                        "folded-and-live") else None),
                    # An unreadable peers file is reported, not raised: every route out of
                    # that state ran through code that used to traceback on it.
                    "peers_error": peers_err,
                    "ledger_identity": ledger_identity,
                    "ledger_identity_explicit": ledger_identity_explicit})
    return rows


def scan_scopes(*, cfg: Path | None = None, file: str | None = None,
                repo_path: str | None = None) -> list[dict]:
    """Enumerate every terse-relevant mcpServers entry across all three scopes,
    read-only — no writes, no directory creation, never raises. One row per
    (scope, server): {scope, server, state, policy, policy_missing, launcher,
    launcher_missing, wraps, diff, stats, config, router, peers_error}, state one of
    "wrapped"
    (stashed and present), "wrapped-unstashed" (the entry launches via terse but has no
    stash, so its original command cannot be restored — #172), "router" (a --multiproxy
    entry fronting the peers file; `wraps` lists its fleet), "folded" (stashed, no live
    entry, and named in that peers file — healthy, `router` says which one it sits
    behind), "folded-unstashed" (named in the peers file with no stash entry — recoverable
    from the peers file, which `uninstall-mcp --all` now does), "folded-and-live" (in the
    peers file AND live, so the downstream runs twice), "router-ambiguous" (two entries
    front one peers file — a hand-edit terse refuses to guess through),
    "orphaned-stash" (stashed
    but the entry vanished — see `_scan_target`), or "unwrapped" (present, not terse's). The wrapped-only
    fields (policy_missing, launcher, launcher_missing, wraps, diff, stats) are
    None/False for non-wrapped rows. Local scope is
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
    # A multiproxy install (#179) leaves the peers file as the only record of which
    # servers a router fronts, so an uninstall must prune it there as well as restoring
    # the entry — otherwise the router keeps launching a peer the client no longer knows.
    peers_p = peers_path(target.cfg, target.stash_prefix)
    peers_doc, peers_err = load_peers(peers_p)
    if peers_err:
        raise ValueError(peers_err)
    all_routers = _detect_routers(node, peers_p)
    router_name = all_routers[0] if len(all_routers) == 1 else None
    folded = {d["name"] for d in peers_downstreams(peers_doc)}
    # `--all` walks the stash UNION anything the config shows as terse-launched, so a
    # wrapped-but-unstashed entry is reported (and refused with a reason) rather than
    # silently omitted from the run — see #172.
    if all_:
        detected = {n for n, e in (node.get("mcpServers") or {}).items()
                    if isinstance(e, dict) and _looks_like_terse_launcher(e)}
        # ...but NOT the router(s) themselves. A router is terse-launched and has no stash
        # of its own, so it landed in the #172 "wrapped but unrecoverable — edit the config
        # by hand" branch on the documented happy path, telling the operator to hand-repair
        # an entry that `unwrap` deletes for them one line later. It is removed by the last
        # peer detaching, never as a target in its own right.
        #
        # `folded` IS unioned in: a folded peer has no live entry by construction, so it
        # can never be in `detected`, and if its stash entry has drifted away it was in
        # neither set — silently skipped by a run that reported success, and absent from
        # status too. The peers file still holds enough to launch it, so it is restorable.
        targets = sorted((set(stash) | detected | folded) - set(all_routers))
    else:
        targets = servers or []

    changes = []
    for s in targets:
        if not is_managed(stash, s):
            # Distinguish "not ours" from "ours but unrecoverable" (#172). An entry that
            # launches via terse with no stash CANNOT be restored — the original command
            # is gone — and silently reporting it as "not managed by terse" is how such a
            # server stays wrapped forever, invisible to both status and uninstall.
            entry = node.get("mcpServers", {}).get(s)
            spec = next((d for d in peers_downstreams(peers_doc) if d["name"] == s), None)
            rebuilt = _entry_from_peer_spec(spec) if spec is not None else None
            if entry is None and rebuilt is not None and peers_doc is not None:
                # Folded with its stash entry gone. Unlike the wrapped-unstashed case
                # below, the original IS recoverable: `_peer_spec` wrote this peer's
                # command/args (or url/headers) plus env/cwd into the peers file, which
                # is the record the router itself launches from. Restore from it and say
                # so — a peer that terse can relaunch but silently declines to restore is
                # the worst of both worlds.
                node.setdefault("mcpServers", {})[s] = rebuilt
                _prune_peer(peers_doc, s)
                changes.append({
                    "server": s, "restored": True, "restored_from": "peers-file",
                    "partial": True,
                    "reason": "stash entry missing — rebuilt from the peers file, which "
                              "records only the launch fields (command/args/env/cwd or "
                              "url/headers); any other key of the original entry is gone",
                })
                continue
            if isinstance(entry, dict) and _looks_like_terse_launcher(entry):
                changes.append({
                    "server": s, "restored": False,
                    "reason": "wrapped by terse but its stash entry is missing, so the "
                              "original command cannot be restored — edit the config by "
                              "hand, using the downstream shown after `--` in its args",
                })
                continue
            changes.append({"server": s, "restored": False, "reason": "not managed by terse"})
            continue
        detached = peers_doc is not None and any(
            isinstance(d, dict) and d.get("name") == s
            for d in (peers_doc.get("downstreams") or []))
        unwrap(node, stash, s, peers_doc=peers_doc, router=router_name)
        change = {"server": s, "restored": True}
        if detached and router_name:
            # Only present when it happened — the plain-unwrap result shape is a public
            # contract (`--json` consumers, existing tests) and must not gain a key that
            # is None for every non-multiproxy install.
            change["detached_from"] = router_name
        changes.append(change)

    # Router sweep, independent of any individual prune. `unwrap` can only remove the
    # router when it was given a `peers_doc`; with the peers file DELETED (the one bad
    # state reachable with no JSON hand-edit at all) `peers_doc` is None, so `--all`
    # restored every original, reported a clean uninstall, and left an entry running
    # `terse proxy --config <missing file>` that exits 2 on every client start forever.
    stranded = (all_ and len(all_routers) == 1 and not peers_downstreams(peers_doc)
                and node.get("mcpServers", {}).pop(all_routers[0], None) is not None)
    if stranded:
        changes.append({"server": all_routers[0], "restored": True, "router": True,
                        "reason": "router entry removed — no peers left to front"})

    result = {"config": str(target.cfg), "scope": scope, "changes": changes,
              "dry_run": dry_run, "backup": None,
              # Keyed on the ROUTER too, not just the doc: with the peers file missing a
              # `--json` consumer otherwise saw `peers_file: null` and all-restored, i.e.
              # "no multiproxy involved", while a router entry sat in the config.
              "peers_file": (str(peers_p) if peers_doc is not None or all_routers
                             else None),
              "routers": all_routers}
    if len(all_routers) > 1:
        # Ambiguous by hand-edit. Terse must not guess which entry to keep — but it also
        # must not destroy the peers file those entries depend on, which is what the old
        # unlink did while leaving both behind, unrepairable.
        result["router_ambiguous"] = all_routers
    if not dry_run and any(c.get("restored") for c in changes):
        result["backup"] = str(_backup(target.cfg))
        _write_json(target.cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(target.cfg), full_stash)
        if peers_doc is not None:
            if peers_downstreams(peers_doc) or len(all_routers) > 1:
                _write_json(peers_p, peers_doc)
            else:
                # Last peer detached: the router entry is gone (see `unwrap`), so an
                # empty peers file is a leftover that a later --multiproxy run would
                # read as state. Remove it rather than leave a zero-peer config behind.
                peers_p.unlink(missing_ok=True)
    return result
