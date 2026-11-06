#!/bin/bash
# Replay queue: at most 3 concurrent replays (torch 1 thread each), each followed
# by verification against the stored DB runs. Logs in experiments/lineage/logs/.
#   bash experiments/lineage/run_replays.sh
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
job() {
  set -- $1
  ds=$1; seed=$2; gens=$3; shift 3
  label=$([ "$ds" = churn ] && echo "churn$gens" || echo "$ds")
  log=experiments/lineage/logs/replay-$label-s$seed.log
  uv run python experiments/lineage/replay_lineage.py "$ds" "$seed" --generations "$gens" > "$log" 2>&1 \
    && uv run python experiments/lineage/verify_replay.py "replay_db/$label-s$seed.replay.csv.gz" \
         $(for e in "$@"; do echo "${e//SEED/$seed}"; done) >> "$log" 2>&1
  echo "$label s$seed exit=$? $(tail -n 3 "$log" | tr '\n' ' ')"
}
export -f job
{
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "churn $s 200 churn100-sSEED churn200-sSEED"; done
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "heart100 $s 100 10seed-heart100-sSEED 10seed-heart-sSEED"; done
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "monk2 $s 100 10seed-monk2-sSEED"; done
} | xargs -P 3 -I{} bash -c 'job "{}"'
