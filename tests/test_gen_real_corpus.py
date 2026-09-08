"""Guards for `scripts/gen_real_corpus.py`'s two load-bearing constraints (#291).

Both were live bugs in the session that wrote the script, and both are enforced there by
comments alone: `scripts/` is outside CI's mypy (`pyproject.toml` scopes it to
`src/terse`), nothing imported the generator, and `corpus-real/` is gitignored so a
corrupted corpus leaves no diff. A future edit lowering a prefix to 1, or restoring
`max_per_tool=None`, would reintroduce a MEASURED bias with nothing failing:

* a 1-record prefix yields ZERO questions (`gen_questions` needs >= 2 records to build a
  record-list question), so the payload silently leaves the exam. The first version
  shipped `gh_pulls: 1` and `gh_workflow_runs: 1` -- terse's two highest-compression
  shapes -- and scored 5/9 payloads instead of 7/9;
* `capture_payload` names envelopes by content sha, so without `max_per_tool=1` a
  re-run with a changed prefix ADDS a second envelope for the same tool and `load_corpus`
  scores that tool twice at two sizes. Reproduced: 10 envelopes on disk while the script
  printed `wrote 9 payloads`.

Pure and fast: no model calls, no network. The script is loaded by path, the same way
`test_text_alias_ceiling.py` loads its script -- `scripts/` is not a package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_real_corpus.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_real_corpus", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load()


def _payload(gen, tool):
    return json.loads((gen.BENCH_CORPUS / f"{tool}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Constraint 1: no record-list prefix may drop its payload out of the exam.
# --------------------------------------------------------------------------- #


def _prefixed_tools(gen):
    return sorted(t for t, n in gen.PREFIXES.items() if n is not None)


def test_every_prefixed_payload_is_a_record_list():
    """A prefix on a non-list is a no-op `prefix_of` hides -- the constraint below would
    then pass vacuously for that entry while the prefix stated a record count that is
    not being applied."""
    gen = _load()
    for tool in _prefixed_tools(gen):
        assert isinstance(_payload(gen, tool), list), tool


@pytest.mark.parametrize("tool", _prefixed_tools(_load()))
def test_every_prefix_keeps_its_payload_in_the_exam(gen, tool):
    """The threshold is not hard-coded: the assertion is `>= 1 question`, whatever
    `gen_questions` needs. Executed: every prefixed payload yields 0 at n=1 and its live
    count at n=2, so lowering any entry to 1 fails exactly that entry's case."""
    obj = gen.prefix_of(_payload(gen, tool), gen.PREFIXES[tool])
    assert len(gen.gen_questions(obj)) >= 1, (
        f"{tool}: a {gen.PREFIXES[tool]}-record prefix generates no questions, so the "
        f"payload is in the corpus but absent from the exam")


def test_the_prefix_that_drops_a_payload_is_the_one_the_guard_names():
    """The guard above is only a guard if the shape it was written against actually
    produces the failure. Pin that: a 1-record prefix of the two payloads the first
    version shipped at 1 yields zero questions."""
    gen = _load()
    for tool in ("gh_pulls", "gh_workflow_runs"):
        assert gen.gen_questions(gen.prefix_of(_payload(gen, tool), 1)) == []


# --------------------------------------------------------------------------- #
# Constraint 2: a re-run replaces a tool's envelope; it never adds a second one.
# --------------------------------------------------------------------------- #


def _run(gen, out_dir, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gen_real_corpus.py", str(out_dir)])
    rc = gen.main()
    out, err = capsys.readouterr()
    return rc, out, err


def _tools_on_disk(gen, out_dir):
    return [e["tool"] for e in gen.load_corpus(out_dir)]


def test_a_rerun_with_a_changed_prefix_replaces_the_envelope(tmp_path, monkeypatch, capsys):
    """The #291 reproduction: `gh_issues: 8` then `gh_issues: 3` into the same directory.
    Envelopes are named by content sha, so the second run's payload is a NEW file; only
    `max_per_tool=1` evicts the first. Without it the directory holds two `gh_issues`
    envelopes and the eval scores the tool twice."""
    gen = _load()
    out_dir = tmp_path / "corpus-real"
    rc, out, err = _run(gen, out_dir, monkeypatch, capsys)
    assert rc == 0 and f"wrote {len(gen.PREFIXES)} payloads" in out
    monkeypatch.setitem(gen.PREFIXES, "gh_issues", 3)
    rc, out, err = _run(gen, out_dir, monkeypatch, capsys)
    assert rc == 0
    tools = _tools_on_disk(gen, out_dir)
    assert len(tools) == len(gen.PREFIXES), (
        f"{len(tools)} envelopes for {len(gen.PREFIXES)} tools -- a stale prefix's "
        f"envelope survived the re-run")
    assert sorted(tools) == sorted(gen.PREFIXES)
    assert tools.count("gh_issues") == 1
    # And the surviving envelope is the NEW prefix, not the old one.
    (issues,) = [e for e in gen.load_corpus(out_dir) if e["tool"] == "gh_issues"]
    assert len(json.loads(issues["raw"])) == 3


def test_the_written_directory_holds_exactly_the_prefixed_tools(tmp_path, monkeypatch, capsys):
    """The set of tools the eval will load equals `PREFIXES` -- no more, no fewer -- and
    the script does not warn about its own output."""
    gen = _load()
    out_dir = tmp_path / "corpus-real"
    rc, out, err = _run(gen, out_dir, monkeypatch, capsys)
    assert rc == 0
    assert set(_tools_on_disk(gen, out_dir)) == set(gen.PREFIXES)
    assert "WARNING" not in err


def test_a_tool_dropped_from_prefixes_is_reported_as_scored_but_unwritten(
        tmp_path, monkeypatch, capsys):
    """`max_per_tool=1` cannot remove a tool that is no longer in `PREFIXES`; the eval
    would score it while the summary table said nothing. The #289 warning is the only
    signal, so it is pinned here through the real run rather than trusted."""
    gen = _load()
    out_dir = tmp_path / "corpus-real"
    _run(gen, out_dir, monkeypatch, capsys)
    monkeypatch.delitem(gen.PREFIXES, "gh_labels")
    rc, out, err = _run(gen, out_dir, monkeypatch, capsys)
    assert rc == 0
    assert "WARNING" in err and "gh_labels" in err
    assert "gh_labels" in _tools_on_disk(gen, out_dir), "the stale envelope is still scored"
