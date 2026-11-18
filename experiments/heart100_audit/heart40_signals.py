"""Stage A of the Heart-100 audit: the 40-generation Heart signal inventory, cached (read-only).

Rebuilds the 23 cross-seed Heart non-linear signals of the species-leads inventory
(experiments/species_leads/signal_inventory.py, ``heart:cross-seed``) from the 10-seed records
on explaneat_dev, and the signal basis of each 40-generation Heart seed's best genome. Checks the
rebuilt train restore gains against ``signals.csv``. Writes ``cache/heart40.npz`` (+ ``.json``),
which ``audit.py`` reads to match the Heart-100 signals against the 40-generation ones.

    DATABASE_URL=postgresql://localhost/explaneat_dev OMP_NUM_THREADS=1 \\
    uv run python experiments/heart100_audit/heart40_signals.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/10seed"), str(REPO / "experiments/churn"),
          str(REPO / "experiments/species_leads")):
    if p not in sys.path:
        sys.path.insert(0, p)

import signal_inventory as si  # noqa: E402
from compute_runs import ReadOnlySession, bootstrap_indices, load_split  # noqa: E402
from explaneat.db.base import db  # noqa: E402


def main():
    if "explaneat_dev" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_dev (read-only use)")
    db.init_db()
    ref = pd.read_csv(REPO / "experiments/species_leads/signals.csv")
    ref = ref[ref.set == "heart:cross-seed"]
    with ReadOnlySession() as s:
        data = load_split(s, si.SPLITS["heart"])
    boot = bootstrap_indices(len(data["y_test"]))
    arrays, meta = {}, {"signals": [], "seeds": {}}
    for sd in range(10):
        rec = json.loads((si.TEN / f"heart-s{sd}.json").read_text())
        g = si.Explained(f"heart-s{sd}", rec["genome_id"], data, boot)
        arrays[f"B_heart-s{sd}"] = g.B
        meta["seeds"][f"heart-s{sd}"] = {"genome_id": rec["genome_id"], "train_auc": g.train_auc}
        for name, sig in g.signals.items():
            r = ref[(ref.genome == f"heart-s{sd}") & (ref.signal == name)]
            assert len(r) == 1, (sd, name)
            diff = abs(float(r.train_restore_gain.iloc[0]) - sig["train_restore_gain"])
            assert diff < 1e-9, (sd, name, diff)
            key = f"heart-s{sd}:{name}"
            for j, x in enumerate(sig["resids"]):
                arrays[f"R_{key}:{j}"] = x
            meta["signals"].append({
                "key": key, "genome": f"heart-s{sd}", "signal": name, "n_exits": len(sig["resids"]),
                "nonlinear_share": sig["nonlinear_share"], "train_restore_gain": sig["train_restore_gain"],
                "test_restore_gain": sig["test_restore_gain"], "test_at_chance": sig["test_at_chance"],
                "cross_seed_found_resid": None if pd.isna(r.cross_seed_found_resid.iloc[0])
                else int(r.cross_seed_found_resid.iloc[0]),
                "hypothesis": sig["hypothesis"][:200]})
        print(f"heart-s{sd}", len(g.signals), flush=True)
    assert len(meta["signals"]) == len(ref) == 23, len(meta["signals"])
    meta["X_train_sha"] = __import__("hashlib").sha1(np.ascontiguousarray(data["X_train"]).tobytes()).hexdigest()
    (HERE / "cache").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "cache/heart40.npz", **arrays)
    (HERE / "cache/heart40.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("ok: 23 signals reproduce signals.csv train restore gains")


if __name__ == "__main__":
    main()
