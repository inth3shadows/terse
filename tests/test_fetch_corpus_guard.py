"""`fetch_corpus.sh` refuses to overwrite a dirty snapshot, and says what it changed.

#341. The script's own header states the invariant that makes every published number
reproducible — "the committed corpus/ snapshot is what the published numbers were measured
on" — and then overwrote that snapshot in place, with no cleanliness check and a closing
`wc -c` that reported byte sizes without saying whether anything had MOVED relative to
HEAD. Eight of the nine payloads come from live endpoints that change week to week.

The cost is not hypothetical: #293 was filed and investigated off figures that recorded 491
tokens for `gh_rate_limit.json`, a file byte-identical since `267af9e` that measures 357
today. Identical bytes cannot produce different counts, so those numbers describe content
that never entered git.

These tests never reach the network. The dirty check runs BEFORE the first `gh` call — by
design, so the refusal is testable without one — and the paths that do fetch get a stub
`gh` and `jq` ahead of the real ones on PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench/fetch_corpus.sh"

PAYLOADS = [
    "gh_pulls.json", "gh_issues.json", "gh_commits.json", "gh_workflow_runs.json",
    "gh_labels.json", "gh_dir_listing.json", "gh_repo_single.json", "gh_rate_limit.json",
]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git checkout holding the real script and a committed corpus."""
    corpus = tmp_path / "scripts/bench/corpus"
    corpus.mkdir(parents=True)
    shutil.copy(SCRIPT, tmp_path / "scripts/bench/fetch_corpus.sh")
    for name in PAYLOADS:
        (corpus / name).write_text('{"committed": true}\n')
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "corpus")
    return tmp_path


def _stub_tools(tmp_path: Path) -> Path:
    """A PATH entry whose `gh` and `jq` never touch the network."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text('#!/usr/bin/env bash\necho \'{"workflow_runs": [{"fetched": true}]}\'\n')
    gh.chmod(0o755)
    jq = bin_dir / "jq"
    jq.write_text('#!/usr/bin/env bash\ncat\n')
    jq.chmod(0o755)
    return bin_dir


def _run(repo: Path, *args: str, stub: Path | None = None):
    env = dict(os.environ)
    if stub is not None:
        env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    return subprocess.run(["bash", str(repo / "scripts/bench/fetch_corpus.sh"), *args],
                          cwd=repo, capture_output=True, text=True, env=env)


def test_a_dirty_corpus_refuses_the_fetch(repo: Path):
    """The defect as filed: a re-fetch on top of a modified snapshot is never what someone
    wants, and it used to be silent."""
    (repo / "scripts/bench/corpus/gh_pulls.json").write_text('{"edited": true}\n')
    r = _run(repo)
    assert r.returncode != 0, r.stdout
    assert "REFUSING" in r.stderr
    assert "gh_pulls.json" in r.stderr, "the refusal must name what is dirty"
    assert (repo / "scripts/bench/corpus/gh_pulls.json").read_text() == '{"edited": true}\n', \
        "a refused run must not have overwritten anything"


def test_an_untracked_payload_also_counts_as_dirty(repo: Path):
    """An un-committed `.json` in corpus/ is measured by everything that globs the
    directory — the same unreproducibility with a different cause."""
    (repo / "scripts/bench/corpus/gh_extra.json").write_text("{}\n")
    r = _run(repo)
    assert r.returncode != 0
    assert "gh_extra.json" in r.stderr


def test_the_check_runs_before_the_first_api_call(repo: Path):
    """A `gh` that records having been called at all. If the check ran after the fetches
    it would be useless — the overwrite would already have happened — and a refusal
    printed at the end would be a report, not a guard."""
    stub = _stub_tools(repo)
    sentinel = repo / "gh-was-called"
    gh = stub / "gh"
    gh.write_text(f'#!/usr/bin/env bash\ntouch "{sentinel}"\necho "{{}}"\n')
    gh.chmod(0o755)
    (repo / "scripts/bench/corpus/gh_pulls.json").write_text('{"edited": true}\n')
    r = _run(repo, stub=stub)
    assert "REFUSING" in r.stderr, r.stderr
    assert not sentinel.exists(), "the API was called before the cleanliness check"


def test_force_overrides_the_refusal_and_warns(repo: Path):
    """`--force` is the documented escape hatch, and it says so — an override that is
    silent is the original defect wearing a flag."""
    (repo / "scripts/bench/corpus/gh_pulls.json").write_text('{"edited": true}\n')
    r = _run(repo, "--force", stub=_stub_tools(repo))
    assert r.returncode == 0, r.stderr
    assert "--force given" in r.stderr
    assert "fetched" in (repo / "scripts/bench/corpus/gh_pulls.json").read_text(), \
        "--force must actually fetch"


def test_a_clean_run_that_changes_nothing_says_so(repo: Path):
    """`wc -c` reported byte sizes without ever saying whether anything MOVED. A run whose
    fetch reproduces HEAD must state that the published numbers still hold."""
    stub = _stub_tools(repo)
    gh = stub / "gh"
    gh.write_text('#!/usr/bin/env bash\nprintf \'{"committed": true}\\n\'\n')
    gh.chmod(0o755)
    r = _run(repo, stub=stub)
    assert r.returncode == 0, r.stderr
    assert "identical to HEAD" in r.stdout, r.stdout


def test_a_run_that_changes_the_snapshot_names_the_files(repo: Path):
    """The other half: the run itself announces that the tree now holds un-committed API
    content, instead of the reader discovering it after publishing."""
    r = _run(repo, stub=_stub_tools(repo))
    assert r.returncode == 0, r.stderr
    assert "DIFFERS from HEAD" in r.stdout
    assert "#341" in r.stdout, "the message should be traceable to its reason"

    # The two sources are asserted SEPARATELY. A bare `"gh_pulls.json" in stdout` cannot
    # tell them apart, because `git diff --stat` prints the filename too — so deleting the
    # porcelain line survived that assertion entirely (mutation sweep, #341). The porcelain
    # line is the one that reports an UNTRACKED payload, which `--stat` never shows.
    lines = r.stdout.splitlines()
    assert any(ln.startswith(" M ") and "gh_pulls.json" in ln for ln in lines), \
        f"no porcelain status line: {r.stdout}"
    assert any("|" in ln and "gh_pulls.json" in ln for ln in lines), \
        f"no diffstat line: {r.stdout}"


def test_the_changed_report_names_an_untracked_payload_too(repo: Path):
    """`git diff --stat` cannot see an untracked file, so the porcelain line is the only
    thing that reports one. This is the case that made the two assertions above worth
    separating."""
    (repo / "scripts/bench/corpus/gh_extra.json").write_text("{}\n")
    r = _run(repo, "--force", stub=_stub_tools(repo))
    assert r.returncode == 0, r.stderr
    assert any(ln.startswith("?? ") and "gh_extra.json" in ln
               for ln in r.stdout.splitlines()), r.stdout


def test_an_unknown_argument_is_rejected(repo: Path):
    """A typo'd flag must not silently fall through to a full overwrite."""
    r = _run(repo, "--forse")
    assert r.returncode == 2
    assert "unknown argument" in r.stderr


def test_help_does_not_fetch(repo: Path):
    r = _run(repo, "--help")
    assert r.returncode == 0
    assert "--force" in r.stdout
    assert (repo / "scripts/bench/corpus/gh_pulls.json").read_text() == '{"committed": true}\n'


def test_a_failed_fetch_still_reports_the_partially_rewritten_corpus(repo: Path):
    """The path that mattered most was the one path that said nothing.

    Under `set -e` a `gh` that dies partway — expired token, 403 rate limit — aborted
    before the closing report, leaving some payloads on today's API content, one truncated
    to zero bytes by its own redirect, and the rest on the committed content: a tree in a
    state that never existed at any point in time. The reader who just watched an error
    scroll past is the likeliest to re-run `benchmark.py` anyway, so reporting only on
    success withheld it from exactly the wrong person (#341 review)."""
    stub = _stub_tools(repo)
    counter = repo / "gh-calls"
    gh = stub / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\n"
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "{counter}"\n'
        'if [ "$n" -gt 3 ]; then echo "gh: HTTP 403 rate limit exceeded" >&2; exit 1; fi\n'
        'echo \'{"fetched": true}\'\n')
    gh.chmod(0o755)
    r = _run(repo, stub=stub)
    assert r.returncode != 0, "a failed fetch must not report success"
    assert "FETCH FAILED" in r.stderr, r.stderr
    assert "PARTIALLY rewritten" in r.stderr
    assert "DIFFERS from HEAD" in r.stdout, r.stdout
    assert any(ln.startswith(" M ") and "gh_pulls.json" in ln
               for ln in r.stdout.splitlines()), r.stdout


def test_the_refusal_prints_a_remedy_that_actually_works(repo: Path):
    """`git restore <path>` restores the worktree from the INDEX. It does not remove an
    untracked file and does not unstage — and it exits 0 either way, so the guard repeated
    itself verbatim with nothing to show for the attempt. The only way out was the
    `--force` this guard exists to discourage (#341 review).

    Runs the printed commands verbatim and requires the next invocation to proceed."""
    (repo / "scripts/bench/corpus/gh_extra.json").write_text("{}\n")       # untracked
    (repo / "scripts/bench/corpus/gh_pulls.json").write_text('{"x": 1}\n')  # staged
    _git(repo, "add", "scripts/bench/corpus/gh_pulls.json")
    (repo / "scripts/bench/corpus/gh_issues.json").write_text('{"y": 2}\n')  # unstaged

    r = _run(repo)
    assert r.returncode != 0
    remedies = [ln.strip() for ln in r.stderr.splitlines() if ln.strip().startswith("git -C")]
    assert len(remedies) == 2, f"expected restore + clean, got {remedies}"
    for cmd in remedies:
        out = subprocess.run(cmd.split("#")[0], shell=True, cwd="/",
                             capture_output=True, text=True)
        assert out.returncode == 0, f"{cmd!r} failed: {out.stderr}"

    assert _git(repo, "status", "--porcelain", "--", "scripts/bench/corpus") == "", \
        "the printed remedy did not actually clean the corpus"
    assert _run(repo, stub=_stub_tools(repo)).returncode == 0, \
        "after the remedy the guard must let the fetch proceed"


def test_the_remedy_is_runnable_from_outside_the_repo(repo: Path):
    """The earlier text embedded `$(git rev-parse --show-toplevel)`, which evaluates in
    the READER's shell. Invoked by absolute path from `/`, the guard refuses correctly and
    then printed a remedy that could not run at all."""
    (repo / "scripts/bench/corpus/gh_pulls.json").write_text('{"edited": true}\n')
    r = subprocess.run(["bash", str(repo / "scripts/bench/fetch_corpus.sh")],
                       cwd="/", capture_output=True, text=True)
    assert r.returncode != 0
    assert "$(git rev-parse" not in r.stderr, "remedy defers resolution to the reader's shell"
    assert str(repo) in r.stderr, "remedy must name an absolute path"


def test_help_is_delimited_by_a_sentinel_not_by_line_numbers(repo: Path):
    """`sed -n '2,19p'` was correct when written and rotted in both directions on the next
    header edit — truncating the help, or printing `set -euo pipefail` as documentation.
    Neither failed a test, because the usage line satisfied the old assertion by itself."""
    script = repo / "scripts/bench/fetch_corpus.sh"
    script.write_text(script.read_text().replace(
        "# WHY THIS REFUSES", "# An extra header line.\n# And another.\n# WHY THIS REFUSES"))
    out = _run(repo, "--help").stdout
    assert "An extra header line." in out, "inserted header text was truncated away"
    assert "Neither guard changes WHAT is fetched" in out, "the tail of the help was lost"
    assert "set -euo pipefail" not in out, "help is printing shell source"
    assert "end of --help" not in out, "the sentinel leaked into the output"


def test_outside_a_git_checkout_it_warns_and_still_fetches(repo: Path):
    """The non-git branch had no coverage at all, so deleting it was a surviving mutation
    and 'no survivors' was a wider claim than the sweep supported (#341 review)."""
    plain = repo.parent / "not-a-repo"
    shutil.copytree(repo / "scripts", plain / "scripts")
    r = subprocess.run(["bash", str(plain / "scripts/bench/fetch_corpus.sh")],
                       cwd=plain, capture_output=True, text=True,
                       env=dict(os.environ,
                                PATH=f"{_stub_tools(repo)}{os.pathsep}{os.environ['PATH']}"))
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stderr and "HEAD" in r.stderr, r.stderr
    # The load-bearing half: it must not claim a comparison it never made.
    assert "identical to HEAD" not in r.stdout, "claimed a comparison it never made"
    assert "DIFFERS from HEAD" not in r.stdout


def test_the_workflow_runs_payload_is_unwrapped_to_an_array(repo: Path):
    """`| jq '.workflow_runs'` is the one payload that is transformed rather than written
    straight through. With `jq` stubbed as `cat`, deleting that pipe changed the file from
    an array to an object without failing anything — while the test that claims to pin
    "WHAT is fetched" passed (#341 review)."""
    stub = _stub_tools(repo)
    jq = stub / "jq"
    jq.write_text('#!/usr/bin/env bash\nexec python3 -c "'
                  "import json,sys;o=json.load(sys.stdin);"
                  "k=sys.argv[1].lstrip('.');print(json.dumps(o[k] if k else o))"
                  '" "$1"\n')
    jq.chmod(0o755)
    r = _run(repo, stub=stub)
    assert r.returncode == 0, r.stderr
    import json
    payload = json.loads((repo / "scripts/bench/corpus/gh_workflow_runs.json").read_text())
    assert isinstance(payload, list), f"expected an array, got {type(payload).__name__}"


def test_the_script_still_fetches_every_payload(repo: Path):
    """The guard must not have changed WHAT is fetched. Pins the list against the
    committed corpus so a dropped endpoint fails here rather than as a quietly smaller
    benchmark."""
    r = _run(repo, stub=_stub_tools(repo))
    assert r.returncode == 0, r.stderr
    for name in PAYLOADS:
        assert "fetched" in (repo / "scripts/bench/corpus" / name).read_text(), name
