#!/usr/bin/env bash
# Re-measure Tier-0 savings on the pinned GitHub corpus at every release tag.
#
# Answers one question: has the codec's output on a fixed corpus moved between
# releases? Run it after any transforms.py change to see the delta in context.
#
# Path-independent — derives the repo root from its own location, and rebuilds the
# measure corpus from the TRACKED payloads in ./corpus rather than a scratch copy.
# The 2026-07-31 original hardcoded /tmp paths and consumed a hand-built envelope
# set that no longer exists; only its numbers survive, in version_sweep.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$(mktemp -d)/sweep}"
TAGS="${TAGS:-$(git -C "$REPO" tag --sort=creatordate | tail -20)}"
mkdir -p "$OUT"

# Envelopes are what `measure --corpus` reads. Rebuild them from the tracked corpus
# so the input is reproducible from git alone.
ENV_DIR="$OUT/envelopes"
if [ ! -d "$ENV_DIR" ]; then
  mkdir -p "$ENV_DIR"
  for f in "$REPO"/scripts/bench/corpus/*.json; do
    (cd "$REPO" && uv run --quiet python -m terse capture "$f" \
        --tool "$(basename "$f" .json)" --server bench --corpus "$ENV_DIR")
  done
fi

printf 'tag\tstatus\traw_tok\tterse_tok\tsaved\tpct\n' > "$OUT/results.tsv"
for tag in $TAGS; do
  wt="$OUT/wt-$tag"
  [ -d "$wt" ] || git -C "$REPO" worktree add --detach "$wt" "$tag" >/dev/null 2>&1 || {
    printf '%s\tWORKTREE_FAIL\n' "$tag" >> "$OUT/results.tsv"; continue; }
  if ! res=$(cd "$wt" && timeout 420 uv run --quiet python -m terse measure \
              --corpus "$ENV_DIR" 2>&1); then
    printf '%s\tRUN_FAIL\t%s\n' "$tag" "$(echo "$res" | tail -1 | cut -c1-120)" \
      >> "$OUT/results.tsv"
    git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true
    continue
  fi
  # The ALL row of the by-shape table is the headline. Grepping for "^total" instead
  # (what the original did) matches "Total payloads captured", so every version
  # recorded "9" and the savings series was lost.
  line=$(echo "$res" | grep -F '| **ALL** |' | tail -1)
  printf '%s\tOK\t%s\n' "$tag" \
    "$(echo "$line" | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$5); gsub(/ /,"",$6); gsub(/ /,"",$7); print $4"\t"$5"\t"$6"\t"$7}')" \
    >> "$OUT/results.tsv"
  git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true
done
echo "results: $OUT/results.tsv"
cat "$OUT/results.tsv"
