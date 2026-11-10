#!/bin/bash
# Species-capture replays (exploratory, not pre-registered): Heart (40 gen) and
# churn100, seeds 0-9. At most 3 concurrent processes, torch 1 thread each.
# Logs in experiments/species_leads/logs/. Verification against the DB is a
# separate step: uv run python experiments/species_leads/build_species.py
#   bash experiments/species_leads/run_species_replays.sh
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p experiments/species_leads/logs
job() {
  set -- $1
  ds=$1; seed=$2; gens=$3
  label=$([ "$ds" = churn ] && echo "churn$gens" || echo "$ds")
  log=experiments/species_leads/logs/replay-$label-s$seed.log
  uv run python experiments/species_leads/replay_species.py "$ds" "$seed" --generations "$gens" > "$log" 2>&1
  echo "$label s$seed exit=$? $(tail -n 1 "$log")"
}
export -f job
{
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "heart $s 40"; done
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "churn $s 100"; done
} | xargs -P 3 -I{} bash -c 'job "{}"'
