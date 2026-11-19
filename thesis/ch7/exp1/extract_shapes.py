"""Synthetic shape study (Experiment 1): every plotted value of the sodium (U) and saddle figures (layer 2,
read-only on the database).

    PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_dev \\
    OMP_NUM_THREADS=1 uv run python thesis/ch7/exp1/extract_shapes.py

Writes thesis/ch7/exp1/data/shapes.json, read by thesis/ch7/exp1/make_figures.py. The study's own PNGs in
experiments/ushape_saddle/figures/ are raster and its -shape.json files keep only the summary of the 2-D grids,
so the sweeps are recomputed here with the study's own code, unchanged:

- shape-A, shape-B, shape-C (the explained genomes, experiments/ushape_saddle/explanations/<label>.json) and
  shape-C-s1-g300 (genome: g300_scores.csv, seed 1, generation 300; dataset "both"):
  ``experiments/ushape_saddle/shape_plots.compute`` (the function behind shape_plots.py). The recomputed summary
  is asserted equal to the committed figures/<label>-shape.json.
- shape-C-sig-best-g300 (genome: sig_gens.json, seed 0, cut-off 300; sigmoid hidden units):
  ``experiments/ushape_saddle/score_gens.surfaces`` (the function behind its figures/shape-C-sig-best-g300-*.png),
  with the shape scores asserted equal to that sig_gens.json row.

Every database session is read only (run with PGOPTIONS as above); nothing is written to any database.
About 3 minutes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
US = REPO / "experiments/ushape_saddle"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(US))
OUT = HERE / "data" / "shapes.json"


def _arr(x):
    return None if x is None else np.asarray(x, float).tolist()


def from_shape_plots(label, genome=None, task=None):
    import shape_plots as sp
    R = sp.compute(label, genome, task)
    committed = json.loads((US / "figures" / f"{label}-shape.json").read_text())
    assert json.loads(json.dumps(R["out"])) == committed, f"{label}: recomputed sweep differs from the committed -shape.json"
    pid = R["pid"]
    return dict(source="experiments/ushape_saddle/shape_plots.py", genome_id=R["out"]["genome_id"], task=R["task"],
                sodium=dict(x=_arr(R["xs"]), score=_arr(R["score"]), truth=_arr(R["truth"]),
                            typical=R["sodium_typical"],
                            parts={p: dict(name=R["pnames"][p], contribution=_arr(c)) for p, c in R["parts"].items()}),
                saddle=dict(heart_rate=_arr(R["hs"]), sbp=_arr(R["ps"]), score=_arr(R["S"]), truth=_arr(R["T"]),
                            part=None if pid is None else dict(id=pid, name=R["pnames"][pid], contribution=_arr(R["C"][pid])),
                            score_corr_with_truth=R["out"]["saddle"]["score_corr_with_truth"],
                            score_interaction_share=R["out"]["saddle"]["score_interaction_share"]),
                sodium_model_minimum_at=R["out"]["sodium"]["model_minimum_at"])


def from_score_gens(row):
    import score_gens as sg
    from explaneat.verify.model import load
    from explaneat.verify.probe import hold_row
    vm = load(row["genome_id"], operations=[], cache=False)
    sf = sg.surfaces(vm)
    u, ut, S, T = sf["u_score"], sf["u_truth"], sf["S"], sf["T"]
    u_r = float(np.corrcoef(u, ut)[0, 1])
    sd_r = float(np.corrcoef((S - S.mean()).ravel(), (T - T.mean()).ravel())[0, 1])
    got = dict(u_r=u_r, u_min_at=float(sg.SODIUM_GRID[int(np.argmin(u))]), saddle_r=sd_r,
               saddle_interaction_share=sg.interaction_share(S))
    for k, v in got.items():
        assert v == row[k], (k, v, row[k])
    col = {n: i for i, n in enumerate(vm.base.feature_names)}
    return dict(source="experiments/ushape_saddle/score_gens.py", genome_id=row["genome_id"], task="both",
                hidden_activation=row["hidden_activation"],
                sodium=dict(x=_arr(sg.SODIUM_GRID), score=_arr(u), truth=_arr(ut),
                            typical=float(hold_row(vm, "typical")[col["sodium"]]), parts={}),
                saddle=dict(heart_rate=_arr(sf["H"][0]), sbp=_arr(sf["P"][:, 0]), score=_arr(S), truth=_arr(T), part=None,
                            score_corr_with_truth=sd_r, score_interaction_share=got["saddle_interaction_share"]),
                sodium_model_minimum_at=got["u_min_at"], u_r=u_r)


def main():
    out = {}
    for label in ("shape-A", "shape-B", "shape-C"):
        out[label] = from_shape_plots(label)
        print(label, flush=True)
    with open(US / "g300_scores.csv") as f:
        g = next(r for r in csv.DictReader(f) if r["seed"] == "1" and r["generation"] == "300")
    out["shape-C-s1-g300"] = from_shape_plots("shape-C-s1-g300", g["genome_id"], "both")
    print("shape-C-s1-g300", flush=True)
    row = next(r for r in json.loads((US / "sig_gens.json").read_text()) if r["seed"] == 0 and r["gen_cut"] == 300)
    out["shape-C-sig-best-g300"] = from_score_gens(row)
    print("shape-C-sig-best-g300", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
