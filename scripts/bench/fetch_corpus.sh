#!/usr/bin/env bash
# Rebuild the benchmark corpus from REAL, public GitHub API output — the record/symbol-
# shaped JSON terse targets. Reproducible: anyone with `gh` can regenerate it. The
# committed corpus/ snapshot is what the published numbers were measured on (the live API
# changes over time, so the snapshot is what makes the numbers reproducible).
set -euo pipefail
cd "$(dirname "$0")/corpus"
R=inth3shadows/terse
gh api "repos/$R/pulls?state=all&per_page=30"          > gh_pulls.json
gh api "repos/$R/issues?state=all&per_page=30"         > gh_issues.json
gh api "repos/$R/commits?per_page=30"                  > gh_commits.json
gh api "repos/$R/actions/runs?per_page=20"             | jq '.workflow_runs' > gh_workflow_runs.json
gh api "repos/$R/labels?per_page=30"                   > gh_labels.json
gh api "repos/$R/contents/src/terse"                   > gh_dir_listing.json
gh api "repos/$R"                                       > gh_repo_single.json   # single object: near-zero case
gh api "rate_limit"                                     > gh_rate_limit.json    # single object: near-zero case
echo "corpus rebuilt:"; wc -c *.json
