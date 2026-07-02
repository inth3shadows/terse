"""Install/remove terse in front of Claude Code MCP servers.

Rewrites the `mcpServers` entries of the Claude Code config (`~/.claude.json` by
default) so a named server's command becomes, for a stdio server:

    <python> -m terse proxy --policy <policy> -- <original command + args>

or, for an HTTP/SSE server (`url` + optional `headers`, #5):

    <python> -m terse proxy --policy <policy> --header k=v ... -- <original url>

The original entry is preserved verbatim in a sidecar stash so `uninstall` can
restore it byte-for-byte. The wrap is idempotent (re-running re-wraps from the
stashed original rather than double-wrapping) and never enables `--diff`
implicitly (diff fluency is unverified — see the diff-fluency reports).

The core is pure functions over plain dicts (`wrap`/`unwrap`) so they are unit
testable without touching the filesystem; the `do_*` helpers add IO + backup.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ._secure_io import write_restricted

STASH_NAME = ".terse-mcp-stash.json"


# --------------------------------------------------------------------------- IO
def config_path() -> Path:
    """Claude Code config location. Honors $CLAUDE_CONFIG, else ~/.claude.json."""
    env = os.environ.get("CLAUDE_CONFIG")
    return Path(env).expanduser() if env else Path.home() / ".claude.json"


def stash_path(cfg: Path) -> Path:
    return cfg.parent / STASH_NAME


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
               cfg: Path | None = None, capture_dir: str | None = None) -> dict:
    cfg = cfg or config_path()
    if not cfg.exists():
        raise FileNotFoundError(f"Claude config not found: {cfg}")
    raw = cfg.read_text(encoding="utf-8")
    config = json.loads(raw)
    had_nl = raw.endswith("\n")  # preserve trailing-newline state for byte-fidelity
    stash = _load_json(stash_path(cfg))
    policy_abs = str(Path(policy).resolve())
    if not Path(policy_abs).exists():
        raise FileNotFoundError(f"policy not found: {policy_abs}")
    # Resolve to an absolute path so capture works regardless of the proxy's cwd; the
    # proxy/capture_payload creates the dir on first write, so no need to pre-create it.
    capture_abs = str(Path(capture_dir).resolve()) if capture_dir else None
    terse_cmd = terse_invocation()

    available = sorted((config.get("mcpServers") or {}).keys())
    managed = set(stash)
    missing = [s for s in servers if s not in set(available) and s not in managed]
    if missing:
        raise ValueError(
            f"unknown server(s): {', '.join(missing)}. "
            f"available: {', '.join(available) or '(none)'}")
    changes = []
    for s in servers:
        before = (config.get("mcpServers") or {}).get(s)
        wrap(config, stash, s, policy_abs, terse_cmd, capture_dir=capture_abs)
        changes.append({"server": s, "before": before,
                        "after": config["mcpServers"][s]})

    result = {"config": str(cfg), "policy": policy_abs, "available": available,
              "changes": changes, "dry_run": dry_run, "backup": None,
              "capture_dir": capture_abs}
    if not dry_run and changes:
        result["backup"] = str(_backup(cfg))
        _write_json(cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(cfg), stash)
    return result


def do_uninstall(servers: list[str] | None, *, all_: bool = False,
                 dry_run: bool = False, cfg: Path | None = None) -> dict:
    cfg = cfg or config_path()
    raw = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
    config = json.loads(raw) if raw else {}
    had_nl = raw.endswith("\n") if raw else True  # preserve trailing-newline state
    stash = _load_json(stash_path(cfg))
    targets = sorted(stash.keys()) if all_ else (servers or [])

    changes = []
    for s in targets:
        if not is_managed(stash, s):
            changes.append({"server": s, "restored": False, "reason": "not managed by terse"})
            continue
        unwrap(config, stash, s)
        changes.append({"server": s, "restored": True})

    result = {"config": str(cfg), "changes": changes, "dry_run": dry_run, "backup": None}
    if not dry_run and any(c.get("restored") for c in changes):
        result["backup"] = str(_backup(cfg))
        _write_json(cfg, config, trailing_newline=had_nl)
        _write_json(stash_path(cfg), stash)
    return result
