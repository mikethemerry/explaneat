"""Export the MONK-2 split used by experiments/10seed (run_seed.py DATASETS['monk2'])
to experiments/modular/data/monk2.npz, once, in a READ ONLY transaction.

The stored StandardScaler (fit on the train rows) is applied, exactly as
experiments/10seed/compute_runs.py::load_split does.

    uv run python experiments/modular/export_monk2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "10seed"))

from compute_runs import ReadOnlySession, db, load_split  # noqa: E402

SPLIT_ID = "63257a45-49c9-42a6-8b28-8e346eab2bee"
DATASET_ID = "c2340904-9db2-4830-99be-e9ed637ed400"


def main():
    db.init_db()
    with ReadOnlySession() as s:
        d = load_split(s, SPLIT_ID)
    out = HERE / "data" / "monk2.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, X_train=d["X_train"], y_train=d["y_train"].astype(np.float64),
             X_test=d["X_test"], y_test=d["y_test"].astype(np.float64),
             feature_names=np.array(d["feature_names"]))
    meta = {"split_id": SPLIT_ID, "dataset_id": DATASET_ID,
            "n_train": int(len(d["y_train"])), "n_test": int(len(d["y_test"])),
            "n_features": int(d["X_train"].shape[1]),
            "train_positive": int(d["y_train"].sum()), "test_positive": int(d["y_test"].sum()),
            "feature_names": list(d["feature_names"]),
            "scaling": "stored StandardScaler of the split (fit on train rows)"}
    (HERE / "data" / "monk2_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_names"}))


if __name__ == "__main__":
    main()
