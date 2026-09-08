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


# Loaded ONCE, like `test_text_alias_ceiling.py`'s `tac`: the script prepends `src/` to
# `sys.path` on every load, and a per-test reload stacked seven copies of the working tree
# ahead of any installed `terse` for the rest of the pytest process (review finding).
gen = _load()

# The entries the prefix is a lever on. Named rather than derived-and-trusted: with the
# roster derived from `PREFIXES` alone, setting every prefix to `None` left the guard below
# with an EMPTY parameter set, which pytest reports as one `s` and exit 0 -- the file's
# central assertion gone, and gh_pulls at 151k tokens per prompt (review finding).
PREFIXED_TOOLS = ("gh_commits", "gh_commits_flat", "gh_issues", "gh_pulls", "gh_workflow_runs")

# The script's own second floor: "~30k is the ceiling that keeps a 4-arm trials=5 panel
# run tractable". gh_pulls sits at 29.7k cl100k at prefix 6 and 34.7k at 7, so 32k
# admits re-fetch drift on the source payloads and still fails on a one-record bump.
PROMPT_TOKEN_CEILING = 32_000


def _payload(tool):
    return json.loads((gen.BENCH_CORPUS / f"{tool}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Constraint 1: no record-list prefix may drop its payload out of the exam.
# --------------------------------------------------------------------------- #


def test_the_prefixed_roster_is_the_one_this_file_guards():
    """Both directions: every integer prefix is in the named roster (a new prefixed entry
    must be added here, or it is unguarded), and every roster entry still carries an
    integer prefix (setting one to `None` is the token-cost regression, not a cleanup)."""
    assert sorted(t for t, n in gen.PREFIXES.items() if n is not None) == sorted(PREFIXED_TOOLS)


def test_every_prefixed_payload_is_a_record_list():
    """A prefix on a non-list is a no-op `prefix_of` hides -- the constraint below would
    then pass vacuously for that entry while the prefix stated a record count that is
    not being applied."""
    for tool in PREFIXED_TOOLS:
        assert isinstance(_payload(tool), list), tool


@pytest.mark.parametrize("tool", PREFIXED_TOOLS)
def test_every_prefix_keeps_its_payload_in_the_exam(tool):
    """The threshold is not hard-coded: the assertion is `>= 1 question`, whatever
    `gen_questions` needs. Executed: every prefixed payload yields 0 at n=1 and its live
    count at n=2, so lowering any entry to 1 fails exactly that entry's case.

    Narrower than #291's wording ("for every `PREFIXES` entry") on purpose: `gh_rate_limit`
    is `None` and not a list, yields 0 questions at its live value, and the script's
    summary flags that as an accepted state -- the prefix is not a lever there."""
    obj = gen.prefix_of(_payload(tool), gen.PREFIXES[tool])
    assert len(gen.gen_questions(obj)) >= 1, (
        f"{tool}: a {gen.PREFIXES[tool]}-record prefix generates no questions, so the "
        f"payload is in the corpus but absent from the exam")


def test_the_prefix_that_drops_a_payload_is_the_one_the_guard_names():
    """The guard above is only a guard if the shape it was written against actually
    produces the failure. Pin that: a 1-record prefix of the two payloads the first
    version shipped at 1 yields zero questions."""
    for tool in ("gh_pulls", "gh_workflow_runs"):
        assert gen.gen_questions(gen.prefix_of(_payload(tool), 1)) == []


@pytest.mark.skipif(gen.count_cl100k("x") is None, reason="tiktoken not installed")
@pytest.mark.parametrize("tool", PREFIXED_TOOLS)
def test_no_prefix_breaches_the_prompt_token_ceiling(tool):
    """The script's OTHER hard floor, from the same comment block: raising a prefix costs
    run time and degraded calls superlinearly. Measured with the script's own count:
    gh_pulls is 29,729 tokens at 6 records and 151,165 at 30 -- and `gh_pulls: 30` passed
    every other test in this file (review finding)."""
    raw = gen.minify(gen.prefix_of(_payload(tool), gen.PREFIXES[tool]))
    tok = gen.count_cl100k(raw)
    assert tok is not None and tok <= PROMPT_TOKEN_CEILING, (
        f"{tool}: {tok:,} cl100k tokens at prefix {gen.PREFIXES[tool]}")


# --------------------------------------------------------------------------- #
# Constraint 2: a re-run replaces a tool's envelope; it never adds a second one.
# --------------------------------------------------------------------------- #


def _run(out_dir, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gen_real_corpus.py", str(out_dir)])
    rc = gen.main()
    out, err = capsys.readouterr()
    return rc, out, err


def _tools_on_disk(out_dir):
    return [e["tool"] for e in gen.load_corpus(out_dir)]


def test_a_rerun_with_a_changed_prefix_replaces_the_envelope(tmp_path, monkeypatch, capsys):
    """The #291 reproduction: `gh_issues: 8` then `gh_issues: 3` into the same directory.
    Envelopes are named by content sha, so the second run's payload is a NEW file; only
    `max_per_tool=1` evicts the first. Without it the directory holds two `gh_issues`
    envelopes and the eval scores the tool twice."""
    out_dir = tmp_path / "corpus-real"
    rc, out, err = _run(out_dir, monkeypatch, capsys)
    assert rc == 0 and f"wrote {len(gen.PREFIXES)} payloads" in out
    monkeypatch.setitem(gen.PREFIXES, "gh_issues", 3)
    rc, out, err = _run(out_dir, monkeypatch, capsys)
    assert rc == 0
    tools = _tools_on_disk(out_dir)
    assert len(tools) == len(gen.PREFIXES), (
        f"{len(tools)} envelopes for {len(gen.PREFIXES)} tools -- a stale prefix's "
        f"envelope survived the re-run")
    assert sorted(tools) == sorted(gen.PREFIXES)
    assert tools.count("gh_issues") == 1
    # And the surviving envelope is the NEW prefix, not the old one.
    (issues,) = [e for e in gen.load_corpus(out_dir) if e["tool"] == "gh_issues"]
    assert len(json.loads(issues["raw"])) == 3


def test_a_tool_dropped_from_prefixes_is_reported_as_scored_but_unwritten(
        tmp_path, monkeypatch, capsys):
    """`max_per_tool=1` cannot remove a tool that is no longer in `PREFIXES`; the eval
    would score it while the summary table said nothing. The #289 warning is the only
    signal, so it is pinned here through the real run rather than trusted -- and its
    absence on a clean run is asserted in the same test, where it is not vacuous: the
    directory it is checked against is the one the warning is about to fire on."""
    out_dir = tmp_path / "corpus-real"
    rc, out, err = _run(out_dir, monkeypatch, capsys)
    assert rc == 0 and "WARNING" not in err
    monkeypatch.delitem(gen.PREFIXES, "gh_labels")
    rc, out, err = _run(out_dir, monkeypatch, capsys)
    assert rc == 0
    assert "WARNING" in err and "gh_labels" in err
    assert "gh_labels" in _tools_on_disk(out_dir), "the stale envelope is still scored"
