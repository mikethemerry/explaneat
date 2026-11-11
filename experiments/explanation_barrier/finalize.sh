#!/bin/bash
# Collect one finished agent run into the repo.
#   finalize.sh explain <label> <agent transcript id> <subagent_tokens>   -> verify_record + usage
#   finalize.sh verify  <label> <agent transcript id> <subagent_tokens>   -> verification/<label>.json/.md + usage
set -e
cd "$(dirname "$0")/../.."
S=/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/barrier
T=/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/tasks
role=$1; label=$2; aid=$3; tok=$4
B=experiments/explanation_barrier
if [ "$role" = explain ]; then
  DATABASE_URL=postgresql://localhost/explaneat_dev uv run python -W ignore $B/verify_record.py $S/$label/report.json 2>&1 | tail -1
else
  mkdir -p $B/verification
  cp $S/$label/verify/verdict.json $B/verification/$label.json
  cp $S/$label/verify/verdict.md $B/verification/$label.md
fi
python3 $B/agent_usage.py $label $role $T/$aid.output $tok | cut -c1-200
