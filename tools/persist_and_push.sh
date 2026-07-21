#!/usr/bin/env bash
# Accumulate this run's per-city summaries onto the orphan `depacc-results`
# branch so separate workflow runs (single cities or batches) build one
# growing cross-city table. Idempotent; safe to call from any depacc workflow
# after the pipeline has produced data/derived/<city>/ outputs.
#
# Flow: check out (or create) the results branch in a git worktree, import
# every previously persisted city into data/derived, run `depacc cross` over
# the union, export the refreshed summaries + cross outputs back, then commit
# and push with a rebase-retry loop to survive concurrent runs.
set -euo pipefail

WORK="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
SUB="$WORK/deprivation-accessibility-eu"
[ -d "$SUB" ] || SUB="$WORK"   # standalone repo: subproject IS the root
DERIVED="$SUB/data/derived"
RESULTS="$WORK/results-branch"
BRANCH="depacc-results"

# Nothing to accumulate unless this run produced at least one per-city
# summary (written by the divergence stage). Ingest/access-only dispatches
# skip persistence entirely.
if ! ls "$DERIVED"/*/cityplane_row.csv >/dev/null 2>&1; then
  echo "persist: this run produced no city summary (ingest/access-only?) — skipping"
  exit 0
fi

git config --global user.name "github-actions[bot]" || true
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com" || true
git config --global --add safe.directory "$WORK" || true

cd "$WORK"
git worktree remove --force "$RESULTS" 2>/dev/null || true
git fetch origin "$BRANCH" || true
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git worktree add -B "$BRANCH" "$RESULTS" "origin/$BRANCH"
else
  echo "persist: results branch does not exist yet — creating it"
  git worktree add --detach "$RESULTS"
  git -C "$RESULTS" checkout --orphan "$BRANCH"
  git -C "$RESULTS" rm -rf . >/dev/null 2>&1 || true
fi

python "$SUB/tools/persist_results.py" import --derived "$DERIVED" --results "$RESULTS"
( cd "$SUB" && depacc cross )
python "$SUB/tools/persist_results.py" export --derived "$DERIVED" --results "$RESULTS"

cd "$RESULTS"
git add -A
if git diff --cached --quiet; then
  echo "persist: no changes to commit"
  exit 0
fi
git commit -q -m "Accumulate depacc results (run ${GITHUB_RUN_ID:-local})"
for i in 1 2 3 4 5; do
  git pull --rebase origin "$BRANCH" 2>/dev/null || true
  if git push origin "HEAD:$BRANCH"; then
    echo "persist: pushed to $BRANCH"
    exit 0
  fi
  echo "persist: push race, retry $i/5"
  sleep $(( (RANDOM % 5) + 2 ))
done
echo "persist: push failed after retries" >&2
exit 1
