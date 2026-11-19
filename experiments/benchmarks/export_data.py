"""Export each classification set's train/test arrays from the database (read-only) to cache/.

    uv run python experiments/benchmarks/export_data.py

Every session is a READ ONLY transaction (experiments/10seed/compute_runs.ReadOnlySession) and is
rolled back; nothing is written to any database. The arrays are exactly what the PropNEAT runs
saw: the split's train and test indices, inputs scaled with the split's stored StandardScaler
(compute_runs.load_split). cache/<name>.npz holds Xtr, ytr, Xte, yte and the feature names; it is
not committed (it can be rebuilt) and the comparator script reads only these files, so it can run
in a separate environment (the EBM needs `interpret`).
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

from common import CACHE, CLASSIFICATION, REPO, jdump

sys.path.insert(0, str(REPO / "experiments" / "10seed"))


def main():
    import compute_runs as cr
    CACHE.mkdir(exist_ok=True)
    manifest = {}
    by_db = {}
    for name, (dbname, sid, _) in CLASSIFICATION.items():
        by_db.setdefault(dbname, []).append((name, sid))
    for dbname, items in by_db.items():
        cr.db.init_db(f"postgresql://localhost/{dbname}")
        for name, sid in items:
            with cr.ReadOnlySession() as s:
                d = cr.load_split(s, sid)
            ytr, yte = d["y_train"].astype(int), d["y_test"].astype(int)
            assert set(np.unique(ytr)) == {0, 1}, name
            np.savez_compressed(CACHE / f"{name}.npz", Xtr=d["X_train"], ytr=ytr, Xte=d["X_test"], yte=yte,
                                names=np.array(d["feature_names"]))
            h = hashlib.sha256(np.ascontiguousarray(d["X_train"]).tobytes() + ytr.tobytes()
                               + np.ascontiguousarray(d["X_test"]).tobytes() + yte.tobytes()).hexdigest()[:16]
            manifest[name] = {"database": dbname, "split_id": sid, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                              "n_features": int(d["X_train"].shape[1]), "positive_train": float(ytr.mean()),
                              "positive_test": float(yte.mean()), "sha256_16": h}
            print(name, manifest[name], flush=True)
        cr.db.engine.dispose() if getattr(cr.db, "engine", None) is not None else None
    jdump(manifest, CACHE.parent / "results" / "data_manifest.json")


if __name__ == "__main__":
    main()
