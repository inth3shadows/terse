#!/usr/bin/env python3
"""Ask RTK what it would do to recorded shell commands and their output. For
`replay_shell.py`.

Nothing from a transcript is executed. Two kinds of job, one JSON object per stdin line:

    {"id": ..., "cmd": "<shell command>"}            -> {"id": ..., "rewrite": "<text>"|null}
    {"id": ..., "filter": "<name>", "text": "..."}   -> {"id": ..., "out": "<text>"}
                                                     or {"id": ..., "error": "..."}

`rewrite` is `rtk rewrite`, which only prints the RTK form of a command; `out` is
`rtk pipe -f <filter>` over the recorded output. RTK keeps a history database, and the
content its filters elide, under HOME, so this refuses to start unless it was given a
home directory to use and loopback is the only network interface:

    unshare -rn python3 rtk_encode.py <rtk binary> <empty home dir> < jobs.jsonl
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

TIMEOUT = 20


def main() -> int:
    if len(sys.argv) != 3 or not os.path.isdir(sys.argv[2]):
        print("usage: rtk_encode.py <rtk binary> <home dir>", file=sys.stderr)
        return 2
    if [name for _, name in socket.if_nameindex()] != ["lo"]:
        print("refusing to run: a network interface other than lo exists (use unshare -rn)",
              file=sys.stderr)
        return 2
    rtk, home = os.path.abspath(sys.argv[1]), sys.argv[2]
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home,
           "XDG_CONFIG_HOME": os.path.join(home, ".config"),
           "XDG_DATA_HOME": os.path.join(home, ".local", "share"),
           "XDG_CACHE_HOME": os.path.join(home, ".cache"),
           "RTK_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1"}
    try:
        subprocess.run([rtk, "--version"], capture_output=True, env=env, cwd=home,
                       timeout=TIMEOUT, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"cannot run the rtk binary: {type(exc).__name__}", file=sys.stderr)
        return 2
    for line in sys.stdin:
        job = json.loads(line)
        try:
            if "cmd" in job:
                proc = subprocess.run([rtk, "rewrite", job["cmd"]], capture_output=True,
                                      text=True, env=env, cwd=home, timeout=TIMEOUT)
                # 3 is what 0.51.0 returns for a rewrite (its help says 0); 1 means no RTK
                # form. Anything else, or a command that is only an option, is not one.
                ok = proc.returncode in (0, 3) and not job["cmd"].lstrip().startswith("-")
                out = {"id": job["id"],
                       "rewrite": (proc.stdout.strip() or None) if ok else None}
            else:
                proc = subprocess.run([rtk, "pipe", "-f", job["filter"]], input=job["text"],
                                      capture_output=True, text=True, env=env, cwd=home,
                                      timeout=TIMEOUT)
                if proc.returncode != 0:
                    raise RuntimeError(f"rtk pipe exited {proc.returncode}")
                out = {"id": job["id"], "out": proc.stdout}
        except Exception as exc:  # one bad job must not end the run
            out = {"id": job["id"], "error": f"{type(exc).__name__}: {exc}"[:200]}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
