"""Directory/file permission guard for `scripts/bench/mcp_servers/mcp_probe.py`'s
`_calls.json` sidecar (#138 step 0).

A prior round shipped `os.makedirs(corpus, exist_ok=True)` + a plain `open(..., "w")`,
which raced `terse proxy`'s own `mkdir_restricted()` (that call no-ops on an already-
existing directory by design) and left the corpus dir at the process umask instead of
0700, with `_calls.json` itself at umask instead of 0600. Fixed in the same round these
tests pin, by reusing terse's own `mkdir_restricted`/`write_restricted`. Nothing else in
the suite imports this script (it isn't part of the `terse` package), so without this
file a regression back to `os.makedirs`/`open` would pass the full suite silently.
"""
from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "mcp_servers" / "mcp_probe.py"


def _load():
    # Registered under a test-file-qualified name, not the bare script name -- avoids
    # colliding in sys.modules with anything else that might load a same-named script
    # (the repo has a few of these importlib.util-loaded test modules for scripts/ files
    # outside the installed package; each uses its own ad hoc name today).
    spec = importlib.util.spec_from_file_location("tests.test_mcp_probe._mcp_probe", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_calls_sidecar_creates_corpus_dir_owner_only(tmp_path):
    corpus = tmp_path / "fresh-corpus"
    probe._write_calls_sidecar(str(corpus), "memory",
                                [{"name": "read_graph", "arguments": {}}])
    assert _mode(corpus) == 0o700


def test_write_calls_sidecar_file_is_owner_only(tmp_path):
    corpus = tmp_path / "fresh-corpus"
    probe._write_calls_sidecar(str(corpus), "memory",
                                [{"name": "read_graph", "arguments": {}}])
    sidecar = corpus / "_calls.json"
    assert _mode(sidecar) == 0o600


def test_write_calls_sidecar_content_round_trips(tmp_path):
    corpus = tmp_path / "fresh-corpus"
    calls = [{"name": "find_symbol", "arguments": {"name_path_pattern": "Foo"}}]
    probe._write_calls_sidecar(str(corpus), "serena", calls)
    saved = json.loads((corpus / "_calls.json").read_text(encoding="utf-8"))
    assert saved == {"server_name": "serena", "calls": calls}


def test_write_calls_sidecar_leaves_a_preexisting_dirs_mode_alone(tmp_path):
    # Documents the known, accepted residual: mkdir_restricted only pins the mode of a
    # directory IT creates. An operator-provided corpus dir (e.g. `mktemp -d`, already
    # 0700 per this script's own README) is left exactly as the operator set it -- this
    # test exists so that behavior stays a documented choice, not a silent surprise.
    corpus = tmp_path / "operator-dir"
    corpus.mkdir(mode=0o750)
    probe._write_calls_sidecar(str(corpus), "memory", [])
    assert _mode(corpus) == 0o750
