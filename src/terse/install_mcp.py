"""Install/remove terse in front of Claude Code MCP servers.

Rewrites the `mcpServers` entries of the Claude Code config (`~/.claude.json` by
default) so a named server's command becomes:

    <python> -m terse proxy --policy <policy> -- <original command + args>

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
         terse_cmd: list[str]) -> tuple[dict, dict]:
    """Wrap `server`'s entry with the terse proxy. Idempotent: if already managed
    (present in stash), re-wrap from the stashed original so policy/cmd updates
    apply cleanly without nesting proxies. Preserves all non-command/args keys
    (env, cwd, type, …) of the original entry."""
    servers = config.setdefault("mcpServers", {})
    if server in stash:
        original = stash[server]
    elif server in servers:
        original = servers[server]
        stash[server] = original
    else:
        raise KeyError(server)

    orig_cmd = original.get("command")
    if not orig_cmd:
        raise ValueError(f"server '{server}' has no 'command' to wrap")
    orig_args = list(original.get("args", []))

    new_entry = {k: v for k, v in original.items() if k not in ("command", "args")}
    new_entry["command"] = terse_cmd[0]
    new_entry["args"] = [*terse_cmd[1:], "proxy", "--policy", policy, "--",
                         orig_cmd, *orig_args]
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


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _backup(cfg: Path) -> Path:
    bak = cfg.with_name(f"{cfg.name}.bak-{int(time.time())}")
    bak.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


def do_install(servers: list[str], policy: str, *, dry_run: bool = False,
               cfg: Path | None = None) -> dict:
    cfg = cfg or config_path()
    if not cfg.exists():
        raise FileNotFoundError(f"Claude config not found: {cfg}")
    config = _load_json(cfg)
    stash = _load_json(stash_path(cfg))
    policy_abs = str(Path(policy).resolve())
    if not Path(policy_abs).exists():
        raise FileNotFoundError(f"policy not found: {policy_abs}")
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
        wrap(config, stash, s, policy_abs, terse_cmd)
        changes.append({"server": s, "before": before,
                        "after": config["mcpServers"][s]})

    result = {"config": str(cfg), "policy": policy_abs, "available": available,
              "changes": changes, "dry_run": dry_run, "backup": None}
    if not dry_run and changes:
        result["backup"] = str(_backup(cfg))
        _write_json(cfg, config)
        _write_json(stash_path(cfg), stash)
    return result


def do_uninstall(servers: list[str] | None, *, all_: bool = False,
                 dry_run: bool = False, cfg: Path | None = None) -> dict:
    cfg = cfg or config_path()
    config = _load_json(cfg)
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
        _write_json(cfg, config)
        _write_json(stash_path(cfg), stash)
    return result
