"""MONK-2 on its fixed split, read from the database in a READ ONLY transaction (layer 2).

    uv run python experiments/monk2_rule_comparison/extract_data.py

The same loader as the comparator suite (experiments/benchmarks/export_data.py): the split
CLASSIFICATION["monk2"] of experiments/benchmarks/common.py in explaneat_v4, read with
experiments/10seed/compute_runs.ReadOnlySession and load_split's tables. Written:
data/monk2_split.json, the dataset's unscaled 17 one-hot columns (0/1), the labels, the split's train
and test row indices and its stored StandardScaler (mean, scale). fit_models.py rebuilds the scaled
arrays exactly as compute_runs.load_split does and asserts their hash equals the suite's
results/data_manifest.json, so both studies fit on the same arrays. Nothing is written to any database.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "10seed"))
sys.path.insert(0, str(REPO / "experiments" / "benchmarks"))

OUT = HERE / "data" / "monk2_split.json"


def main():
    import compute_runs as cr
    from common import CLASSIFICATION
    from explaneat.db.models import Dataset, DatasetSplit

    dbname, sid, _ = CLASSIFICATION["monk2"]
    cr.db.init_db(f"postgresql://localhost/{dbname}")
    with cr.ReadOnlySession() as s:
        split = s.get(DatasetSplit, uuid.UUID(sid))
        ds = s.get(Dataset, split.dataset_id)
        X, y = ds.get_data()
        assert split.scaler_type == "StandardScaler", split.scaler_type
        out = {
            "source": {"database": dbname, "split_id": sid, "dataset_id": str(ds.id), "dataset_name": ds.name,
                       "loader": "experiments/10seed/compute_runs.ReadOnlySession (READ ONLY, rolled back)"},
            "feature_names": list(ds.feature_names),
            "X_onehot": np.asarray(X).astype(int).tolist(),
            "y": np.asarray(y).ravel().astype(int).tolist(),
            "train_indices": [int(i) for i in split.train_indices],
            "test_indices": [int(i) for i in split.test_indices],
            "scaler": {"mean": [float(v) for v in split.scaler_params["mean"]],
                       "scale": [float(v) for v in split.scaler_params["scale"]]},
        }
    Xa = np.asarray(X, float)
    assert np.array_equal(Xa, np.asarray(out["X_onehot"], float)), "one-hot columns are not 0/1"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    print("wrote", OUT.relative_to(REPO), "rows", len(out["y"]), "features", len(out["feature_names"]))


if __name__ == "__main__":
    main()
