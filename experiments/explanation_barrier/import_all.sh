#!/bin/bash
# Import the selected ladder networks (task arch pairs) into explaneat_dev, one at a time.
cd "$(dirname "$0")/../.."
export DATABASE_URL=postgresql://localhost/explaneat_dev
while [ $# -ge 2 ]; do
  uv run python -W ignore experiments/explanation_barrier/import_mlp.py "$1" "$2" | grep -E '"label"|"genome_id"|"max_abs_diff"|"ok"|exists|Error'
  shift 2
done
