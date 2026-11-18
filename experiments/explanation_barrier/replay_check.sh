#!/bin/bash
# Replay every stored explanation with the new code and with the pre-change code (origin/main at
# de6d655, exported to $OLD), read only, and compare the model-state hashes.
#   replay_check.sh <database url> <tag> <old code dir> <out dir>
set -e
cd "$(dirname "$0")/../.."
URL=$1; TAG=$2; OLD=$3; OUT=$4
python3 -c "import pathlib,sys; p=pathlib.Path('experiments/explanation_barrier/replay_check.py'); pathlib.Path(sys.argv[1]).write_text(p.read_text())" "$OLD/replay_check.py"
DATABASE_URL=$URL uv run python -W ignore experiments/explanation_barrier/replay_check.py "$OUT/replay_new_$TAG.json"
(cd "$OLD" && DATABASE_URL=$URL PYTHONPATH="$OLD" uv run --project "$OLDPWD" python -W ignore replay_check.py "$OUT/replay_old_$TAG.json")
python3 - "$OUT/replay_new_$TAG.json" "$OUT/replay_old_$TAG.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [g for g in a if g in b and a[g].get("hash") and a[g].get("hash") == b[g].get("hash")]
diff = [g for g in a if g in b and a[g].get("hash") != b[g].get("hash")]
err = [g for g in a if "error" in a[g] or "error" in b.get(g, {})]
print(f"{len(a)} explanations: {len(same)} identical, {len(diff)} differ, {len(err)} with errors;",
      f"split/consolidate ops in identical ones: {sum(a[g]['n_split_ops'] for g in same)}")
for g in diff[:10]:
    print("DIFF", g, a[g], b[g])
EOF
