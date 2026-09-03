#!/usr/bin/env bash
# Rebuild the benchmark corpus from REAL, public GitHub API output — the record/symbol-
# shaped JSON terse targets. Reproducible: anyone with `gh` can regenerate it. The
# committed corpus/ snapshot is what the published numbers were measured on (the live API
# changes over time, so the snapshot is what makes the numbers reproducible).
#
# WHY THIS REFUSES TO RUN ON A DIRTY corpus/ (#341). The paragraph above states the
# invariant; the script used to break it silently. Eight of the nine payloads come from
# live endpoints that return different content week to week, and a run left the working
# tree holding un-committed API output that `terse measure --corpus`, `benchmark.py` and
# `tests/test_published_benchmarks.py` would all happily measure — the last of those
# compares the docs against a LIVE re-measurement, so on a dirty tree it measures content
# that never entered git.
#
# #293 is the cost already paid. It was investigated off figures — #249's — recording 491
# tokens for gh_rate_limit.json, a file byte-identical since 267af9e (2026-07-17) that
# measures 357 today. Identical bytes cannot produce different counts, so those numbers
# describe content nobody can reproduce. A dirty tree explains the EIGHT payloads this
# script writes; it does not explain gh_commits_flat.json, which moved in the same table
# and has no producer in the repo at all (gen_real_corpus.py says so outright). Something
# else is also in play there. This guard closes the half that is ours.
#
# Neither guard changes WHAT is fetched. Both make the moment the invariant breaks visible
# at the point of breakage instead of three days later in an issue.
# --- end of --help ---
set -euo pipefail

FORCE=0
# `while [ $# -gt 0 ]`, not `for arg in "$@"`: the latter is an unbound-variable error
# under `set -u` on bash < 4.4 (macOS ships 3.2) when there are no arguments at all.
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    -h|--help)
      # Delimited by the sentinel above, NOT by hardcoded line numbers. The line-number
      # form was correct when written and rotted in both directions the moment anyone
      # edited the header — silently truncating the help, or printing shell source as
      # documentation — and the help test could not see either, because the usage line
      # below satisfied its assertion on its own (#341 review).
      sed -n '2,/^# --- end of --help ---$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
      echo
      echo "usage: $(basename "$0") [--force]"
      echo "  --force   re-fetch even when corpus/ has uncommitted changes"
      exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/corpus"

IN_REPO=false
TOPLEVEL=
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null || true)" = "true" ]; then
  IN_REPO=true
  TOPLEVEL=$(git rev-parse --show-toplevel)
fi

# Untracked files count as dirty too: an un-committed .json sitting in corpus/ is measured
# by everything that globs this directory, which is the same failure with a different
# cause. A checkout that is not a git repo at all can still fetch — it just cannot make
# this promise.
if [ "$IN_REPO" = true ]; then
  dirty=$(git status --porcelain -- . 2>/dev/null || true)
  if [ -n "$dirty" ] && [ "$FORCE" -ne 1 ]; then
    {
      echo "REFUSING: corpus/ has uncommitted changes relative to HEAD."
      echo
      echo "$dirty"
      echo
      echo "Re-fetching on top of an already-modified snapshot means the numbers you"
      echo "measure next cannot be reproduced from git (#341). Restore it first:"
      echo
      # BOTH commands, and absolute paths resolved HERE. `git restore <path>` alone
      # restores the worktree from the INDEX, so it leaves a staged change staged and an
      # untracked file untouched — and exits 0 either way, so the guard just repeated
      # itself with nothing to show for the attempt, and the only way out was the --force
      # this guard exists to discourage. The earlier text also embedded a `git rev-parse`
      # that ran in the READER's shell, which fails outright when the script was invoked
      # by absolute path from outside the repo (#341 review).
      echo "    git -C '$TOPLEVEL' restore --source=HEAD --staged --worktree -- scripts/bench/corpus"
      echo "    git -C '$TOPLEVEL' clean -fd -- scripts/bench/corpus   # removes untracked payloads"
      echo
      echo "Or pass --force if overwriting is genuinely what you want."
    } >&2
    exit 1
  fi
  [ -n "$dirty" ] && echo "WARNING: --force given; overwriting an already-dirty corpus/." >&2
else
  echo "WARNING: corpus/ is not inside a git checkout, or git is unavailable —" >&2
  echo "         cannot check the snapshot against HEAD." >&2
fi

# The closing report runs from a TRAP, not as the last statement. Under `set -e` a `gh`
# that fails partway — an expired token, a 403 rate limit — aborts the script with some
# payloads rewritten from today's API, one truncated to zero bytes by its own redirect,
# and the rest still on the committed content: a tree in a state that never existed at any
# point in time. Reporting only on success made that the one path where the script said
# nothing, which is exactly the reader who most needs telling — they watched an error
# scroll past and are the likeliest to re-run `benchmark.py` anyway (#341 review).
FETCHING=0
report_state() {
  rc=$?
  [ "$FETCHING" -eq 1 ] || return "$rc"
  [ "$IN_REPO" = true ] || return "$rc"
  echo
  [ "$rc" -ne 0 ] && echo "FETCH FAILED (exit $rc) — corpus/ is PARTIALLY rewritten." >&2
  changed=$(git status --porcelain -- . 2>/dev/null || true)
  if [ -z "$changed" ]; then
    echo "corpus is identical to HEAD — nothing to commit, published numbers still hold."
  else
    echo "corpus now DIFFERS from HEAD. Any measurement taken before you commit this is"
    echo "unreproducible from git (#341):"
    echo "$changed"
    git diff --stat -- . || true
  fi
  return "$rc"
}
trap report_state EXIT

FETCHING=1
R=inth3shadows/terse
gh api "repos/$R/pulls?state=all&per_page=30"          > gh_pulls.json
gh api "repos/$R/issues?state=all&per_page=30"         > gh_issues.json
gh api "repos/$R/commits?per_page=30"                  > gh_commits.json
gh api "repos/$R/actions/runs?per_page=20"             | jq '.workflow_runs' > gh_workflow_runs.json
gh api "repos/$R/labels?per_page=30"                   > gh_labels.json
gh api "repos/$R/contents/src/terse"                   > gh_dir_listing.json
gh api "repos/$R"                                       > gh_repo_single.json   # single object: near-zero case
gh api "rate_limit"                                     > gh_rate_limit.json    # single object: near-zero case
echo "corpus rebuilt:"; wc -c -- *.json
