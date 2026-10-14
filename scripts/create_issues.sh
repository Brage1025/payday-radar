#!/usr/bin/env bash
# Create all issues from docs/issues/*.md with the GitHub CLI.
# Usage: scripts/create_issues.sh [--assign-copilot]
# Needs: gh auth login. Assigning the Copilot coding agent from the CLI needs a recent gh;
# if --assign-copilot fails, assign @copilot to each issue in the GitHub UI instead.
set -euo pipefail
cd "$(dirname "$0")/.."
gh label create copilot-task --color 5319e7 --description "Scoped task for Copilot coding agent" 2>/dev/null || true
for f in docs/issues/*.md; do
  title=$(head -1 "$f" | sed 's/^# //')
  body=$(tail -n +3 "$f")
  args=(--title "$title" --body "$body" --label copilot-task)
  if [[ "${1:-}" == "--assign-copilot" ]]; then args+=(--assignee "@copilot"); fi
  gh issue create "${args[@]}"
done
