#!/usr/bin/env bash
# Re-measure Tier-0 savings on the pinned GitHub corpus at every release tag.
#
# Answers one question: has the codec's output on a fixed corpus moved between
# releases? Run it after any transforms.py change to see the delta in context.
#
# Usage:  ./version_sweep.sh [OUT_DIR]              # the recorded range, see TAGS
#         TAGS="v0.17.0 v0.19.0" ./version_sweep.sh /tmp/sweep
#
# Derives the repo root from its own location and rebuilds the measure corpus from
# the TRACKED payloads in ./corpus, so the whole input is reproducible from git.
# The 2026-07-31 original hardcoded /tmp paths and consumed a hand-built envelope
# set that no longer exists; only its numbers survive, in version_sweep.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Absolute, always. Half the paths below resolve against $REPO (`git -C`, the capture
# subshell) and half against the caller's cwd, so a relative OUT silently splits the
# run in two: envelopes land under the caller's cwd while `worktree add` puts the
# checkouts under $REPO, and every tag then fails to cd into a worktree that is not
# where it looked. Run from the repo root, it also scatters ~1.2 MB of envelopes into
# the working tree.
OUT="${1:-$(mktemp -d)/sweep}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# Pinned by default, not `tail -20`: this script exists to reproduce a recorded
# baseline, and a relative window silently redefines what it measures as new tags
# land — by 2026-08-04 `tail -20` had already slid past v0.5.1..v0.7.0, the very rows
# version_sweep.md records. Override TAGS to sweep a different range.
TAGS="${TAGS:-$(git -C "$REPO" tag --sort=creatordate |
                sed -n '/^v0\.5\.1$/,/^v0\.17\.0$/p')}"

CORPUS="$REPO/scripts/bench/corpus"
ENV_DIR="$OUT/envelopes"
RESULTS="$OUT/results.tsv"

# Envelopes are what `measure --corpus` reads. Rebuild from the tracked corpus unless
# a COMPLETE set is already present. Existence of the directory is not enough: `terse
# capture` writes one file at a time, so an interrupted run leaves a partial set that
# would be reused silently and produce a plausible wrong number reported as OK
# (measured: 3 of 9 envelopes gives +5.5% where the truth is +58.3%). A wrong number
# that looks right is the failure this repo can least afford.
count_json() {  # 0 for a missing dir. NOT `find … 2>/dev/null | wc -l`: find exits 1
  [ -d "$1" ] || { echo 0; return; }   # on a missing path and `pipefail` then kills
  find "$1" -maxdepth 1 -name '*.json' | wc -l   # the whole script under `set -e`.
}
want=$(count_json "$CORPUS")
have=$(count_json "$ENV_DIR")
if [ "$have" -ne "$want" ]; then
  [ "$have" -eq 0 ] || echo "envelopes: found $have of $want — rebuilding" >&2
  rm -rf "${ENV_DIR:?}"
  # No `mkdir -p "$ENV_DIR"` here: `terse capture`'s own `mkdir_restricted()` creates it
  # on the first iteration below, owner-only (0700). Pre-creating it at the shell's
  # umask would make `mkdir_restricted` no-op on an already-existing directory (by
  # design -- an operator's directory is left at the operator's mode), silently
  # stripping the restriction envelope filenames (tool + payload hash) are meant to have.
  for f in "$CORPUS"/*.json; do
    (cd "$REPO" && uv run --quiet python -m terse capture "$f" \
        --tool "$(basename "$f" .json)" --server bench --corpus "$ENV_DIR" >/dev/null)
  done
  have=$(count_json "$ENV_DIR")
  [ "$have" -eq "$want" ] || { echo "envelope build produced $have of $want" >&2; exit 1; }
fi
echo "corpus: $want payloads -> $ENV_DIR" >&2

# Every row carries all 6 fields, failure rows included — a ragged TSV is a parsing
# trap for whatever reads this next.
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "${3:--}" "${4:--}" "${5:--}" "${6:--}"; }
row tag status raw_tok terse_tok saved pct > "$RESULTS"

for tag in $TAGS; do
  wt="$OUT/wt-$tag"
  if ! git -C "$REPO" worktree add --detach "$wt" "$tag" >/dev/null 2>&1; then
    row "$tag" WORKTREE_FAIL >> "$RESULTS"
    continue
  fi

  status=OK
  line=""
  if res=$(cd "$wt" && timeout 420 uv run --quiet python -m terse measure \
             --corpus "$ENV_DIR" 2>&1); then
    # `|| true` is load-bearing. Under `set -euo pipefail` a grep that matches nothing
    # aborts the whole sweep mid-loop — after `worktree add`, before the cleanup below
    # — so one unexpected report shape both truncates the run and leaks a worktree into
    # the user's real (bare-layout, many-worktree) repo. Fail this tag, keep going.
    line=$(echo "$res" | grep -F '| **ALL** |' | tail -1 || true)
    [ -n "$line" ] || status=NO_ALL_ROW
  else
    status=RUN_FAIL
  fi

  # Unconditional, so no path through the loop leaves a worktree registered.
  git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true

  if [ "$status" = OK ]; then
    # The ALL row of the by-shape table is the headline. Grepping for "^total" instead
    # (what the original did) matches "Total payloads captured", so every version
    # recorded "9" and the savings series was lost.
    # shellcheck disable=SC2046  # deliberate split: awk emits the four fields
    row "$tag" OK $(echo "$line" | awk -F'|' \
      '{for (i = 4; i <= 7; i++) gsub(/ /, "", $i); print $4, $5, $6, $7}') >> "$RESULTS"
  else
    row "$tag" "$status" >> "$RESULTS"
  fi
done

echo "results: $RESULTS" >&2
cat "$RESULTS"
