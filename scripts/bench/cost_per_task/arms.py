"""Arm config generator for the cost-per-task harness.

Writes an arm-specific Claude Code MCP config (for `--mcp-config <path>`) into
a caller-supplied output directory (the runner always points this at the
scratchpad, never the repo), mode 600. Three arms:

  A (vanilla)   -- empty mcpServers.
  B (no terse)  -- kb/codegraph/runecho's entries, built from the SAME
                   peers-file `downstreams` arm C's router launches today
                   (same command/args/env per server), so the router is the
                   ONLY difference between a B run and a C run.
  C (terse)     -- the LIVE multiproxy router entry that already fronts those
                   same three servers today, copied verbatim except for two
                   benchmark-only safety overrides (see
                   `_sanitize_router_args`) and a pinned COPY of its peers
                   file and policy.json (so a concurrent `install-mcp`
                   cannot change the fleet under an in-flight run): the
                   policy/diff/server-name flags actually under test are
                   untouched.

This module only reads terse's OWN `install_mcp` module (the single source of
truth for where the config/stash live and what a scope key means) -- it never
re-derives that resolution logic. Every value that could be a secret (an
`env` var, a `--header` value) is copied byte-for-byte into the generated
file and is NEVER logged, printed, or written anywhere else.
"""
from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

from terse import install_mcp

# The only servers the pilot task suite actually calls. Keeping the arm
# configs scoped to just these (rather than folding in every live server,
# e.g. secret-broker or lsp-py) keeps each arm's MCP surface identical to
# what the tasks can reach, so a call-count skew between arms can't be
# blamed on an unrelated server neither task touches.
SERVERS = ("kb", "codegraph", "runecho")
ROUTER_NAME = "terse"  # the live multiproxy entry fronting SERVERS today

VALID_ARMS = ("A", "B", "C")


def _write_restricted(path: Path, doc: dict) -> Path:
    """Write `doc` as JSON at `path`, mode 600 (owner read/write only) from
    the moment the file is CREATED -- no write-then-chmod window where a
    permissive umask (e.g. 022 is fine, but a hand-configured 000 is not)
    leaves it briefly group/world-readable. Parents created as needed.
    Mirrors terse's own `_secure_io.write_restricted` contract without
    importing a private module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    # os.open's mode is masked by umask; force the final bits explicitly so
    # a permissive umask can't leave this readable beyond the owner.
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def _load_live_config(cfg: Path | None = None) -> dict:
    cfg = cfg or install_mcp.config_path()
    return json.loads(cfg.read_text())


def _live_router_entry(cfg: Path | None) -> tuple[Path, dict]:
    """The resolved config path and the live '{ROUTER_NAME}' mcpServers
    entry, or raise KeyError -- shared by arm B (needs the router's peers
    file) and arm C (needs the router entry itself)."""
    resolved_cfg = cfg or install_mcp.config_path()
    live = _load_live_config(resolved_cfg)
    router = live.get("mcpServers", {}).get(ROUTER_NAME)
    if router is None:
        raise KeyError(
            f"no live '{ROUTER_NAME}' router entry in {resolved_cfg} -- arm B/C "
            f"both need the current terse-wrapped setup to exist")
    return resolved_cfg, router


def _downstream_to_mcp_entry(d: dict) -> dict:
    """Reverse of terse.install_mcp._peer_spec: turn one `downstreams[]`
    entry from the router's peers file back into a raw MCP client server
    entry (`{"command", "args", [env], [cwd]}` for stdio, or `{"url",
    [headers]}` for HTTP/SSE)."""
    if "command" in d:
        cmd_list = list(d["command"])
        entry: dict = {"command": cmd_list[0], "args": cmd_list[1:]}
        if d.get("env"):
            entry["env"] = dict(d["env"])
        if d.get("cwd"):
            entry["cwd"] = d["cwd"]
        return entry
    if "url" in d:
        entry = {"url": d["url"]}
        if d.get("headers"):
            entry["headers"] = dict(d["headers"])
        return entry
    raise ValueError(f"downstream entry {d.get('name')!r} has neither 'command' nor 'url'")


def _load_peers_downstreams(peers_path: Path) -> dict[str, dict]:
    """name -> downstream entry, from a peers file (the SAME file arm C's
    router reads via --config)."""
    doc = json.loads(peers_path.read_text())
    return {d["name"]: d for d in (doc.get("downstreams") or [])
            if isinstance(d, dict) and d.get("name")}


def arm_a_config() -> dict:
    """Vanilla: no MCP servers at all."""
    return {"mcpServers": {}}


def arm_b_config(cfg: Path | None = None) -> dict:
    """SERVERS' entries built from the SAME peers-file `downstreams` arm C's
    router launches today -- same command/args/env per server -- so the
    terse router is the ONLY difference between a B run and a C run.

    Previously built from terse's install-mcp STASH (the original,
    pre-wrap config captured once at install time). That stash can silently
    drift from the live peers file -- hand-edited or regenerated since
    install -- which would make a B-vs-C delta partly configuration noise
    instead of purely the router's effect. Reading the live peers file
    instead means B always reflects what the router ACTUALLY launches
    today (see terse-cost-per-task-e2e.md blocker #4)."""
    resolved_cfg, router = _live_router_entry(cfg)
    peers_path = _router_peers_path(router)
    if peers_path is None or not peers_path.exists():
        raise FileNotFoundError(
            f"router entry in {resolved_cfg} has no readable --config peers file -- "
            f"arm B needs it to recover {SERVERS}'s live command/args/env")
    by_name = _load_peers_downstreams(peers_path)
    missing = [s for s in SERVERS if s not in by_name]
    if missing:
        raise KeyError(
            f"peers file {peers_path} has no downstream entry for {missing} -- is "
            f"terse actually routing them?")
    servers = {name: _downstream_to_mcp_entry(by_name[name]) for name in SERVERS}
    return {"mcpServers": servers}


def _sanitize_router_args(args: list[str]) -> list[str]:
    """Strip/add the two flags that separate a BENCHMARK run of the router
    from a real one, so this harness cannot pollute production state:

      - drop `--capture-dir <path>`: without this, every benchmark tool
        result would be teed into the live capture corpus alongside real
        session data.
      - add `--no-stats` (if not already present): without this, every
        benchmark call would land in the live `terse stats` savings ledger,
        which is meant to answer "did terse save money in real use", not
        "did terse save money answering a synthetic KB lookup 45 times".

    Everything else -- `--policy`, `--server-name`, `--diff`/`--no-diff`,
    `--config` -- is the actual thing arm C is testing and is left alone.
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--capture-dir":
            i += 2  # drop the flag AND its value
            continue
        out.append(args[i])
        i += 1
    if "--no-stats" not in out:
        out.append("--no-stats")
    return out


def _router_flag_value(router_entry: dict, flag: str) -> Path | None:
    args = router_entry.get("args") or []
    if flag in args:
        j = args.index(flag)
        if j + 1 < len(args):
            return Path(args[j + 1])
    return None


def _router_peers_path(router_entry: dict) -> Path | None:
    return _router_flag_value(router_entry, "--config")


def _snapshot_peers_with_pinned_policy(peers_doc: dict, peers_dir: Path, out_dir: Path,
                                        arm: str) -> dict:
    """Deep-copy `peers_doc` (a router `--config` peers file already loaded
    from disk) and, for every downstream that carries its own `policy`
    field, copy the policy file it points at into `out_dir` and repoint that
    downstream's `policy` to the copy. `policy` is resolved the same way
    multiproxy.py's own loader resolves it -- relative to `peers_dir` (the
    peers file's OWN directory), not the process cwd (see
    `_build_downstreams`'s `config_path.parent / p`).

    Replaces the previous fake top-level `--policy` router-arg pinning
    (plan Partial #10): the LIVE router never actually carries a top-level
    `--policy` flag -- multiprocess/multiproxy mode reads a `policy` path
    PER DOWNSTREAM out of the peers file instead (`default_policy` -- the
    top-level flag -- is only a fallback for a downstream that has no
    `policy` of its own). Pinning a top-level `--policy` copy therefore
    pinned nothing real; this pins the actual mechanism, so a concurrent
    `install-mcp` (or hand edit) to policy.json can't change what an
    in-flight run is testing, matching the guarantee already given to the
    peers file itself.

    A downstream with no `policy` field is left untouched -- it uses
    whatever `default_policy` the router falls back to, which this
    benchmark harness doesn't pin (no top-level `--policy` flag exists on
    the live router to pin)."""
    doc = copy.deepcopy(peers_doc)
    copied: dict[str, Path] = {}  # resolved source path -> copy already written
    for d in doc.get("downstreams") or []:
        if not isinstance(d, dict):
            continue
        raw = d.get("policy")
        if not raw:
            continue
        src = Path(raw)
        if not src.is_absolute():
            src = peers_dir / src
        if not src.exists():
            continue
        key = str(src.resolve())
        dst = copied.get(key)
        if dst is None:
            dst = out_dir / f"policy-{arm}-{len(copied)}.json"
            _write_restricted(dst, json.loads(src.read_text()))
            copied[key] = dst
        d["policy"] = str(dst)
    return doc


def arm_c_config(cfg: Path | None = None, *, peers_copy_path: Path | None = None) -> dict:
    """The live multiproxy router fronting SERVERS today, copied verbatim
    apart from the benchmark-only overrides in `_sanitize_router_args`. If
    `peers_copy_path` is given, the router's `--config` is repointed at it
    instead of the live peers file -- decoupling a run in flight from a
    concurrent `install-mcp` (or hand edit) changing the fleet underneath it
    (plan blocker #10). The peers copy itself carries pinned policy.json
    copies too (see `_snapshot_peers_with_pinned_policy`), so there is no
    separate `--policy` flag to repoint here -- the live router doesn't
    carry one."""
    live = _load_live_config(cfg)
    servers = live.get("mcpServers", {})
    if ROUTER_NAME not in servers:
        raise KeyError(
            f"no live '{ROUTER_NAME}' router entry in "
            f"{cfg or install_mcp.config_path()} -- arm C needs the current "
            f"terse-wrapped setup to exist")
    entry = copy.deepcopy(servers[ROUTER_NAME])
    args = list(entry.get("args", []))
    if peers_copy_path is not None and "--config" in args:
        i = args.index("--config")
        if i + 1 < len(args):
            args[i + 1] = str(peers_copy_path)
    entry["args"] = _sanitize_router_args(args)
    return {"mcpServers": {ROUTER_NAME: entry}}


def write_arm_config(arm: str, out_dir: Path, cfg: Path | None = None) -> Path:
    """Build and write the MCP config for `arm` ('A'|'B'|'C') into `out_dir`.
    Returns the written path (mode 600). `out_dir` must be outside the repo
    (the runner always passes the scratchpad)."""
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm {arm!r}, must be one of {VALID_ARMS}")
    if arm == "A":
        doc = arm_a_config()
    elif arm == "B":
        doc = arm_b_config(cfg)
    else:  # "C"
        resolved_cfg = cfg or install_mcp.config_path()
        live = _load_live_config(resolved_cfg)
        router = live.get("mcpServers", {}).get(ROUTER_NAME, {})
        peers_src = _router_peers_path(router)
        peers_dst = None
        if peers_src is not None and peers_src.exists():
            peers_doc = json.loads(peers_src.read_text())
            snapshot = _snapshot_peers_with_pinned_policy(
                peers_doc, peers_src.parent, out_dir, arm)
            peers_dst = out_dir / f"peers-{arm}.json"
            _write_restricted(peers_dst, snapshot)
        doc = arm_c_config(resolved_cfg, peers_copy_path=peers_dst)
    return _write_restricted(out_dir / f"mcp-config-{arm}.json", doc)
