"""Adult, original network: the capital-gain part's response, for the main-text figure (read-only DB).

    uv run python thesis/ch7/exp1/extract_capgain.py        # -> data/capgain.json

The study's own computation, experiments/aaai27/run_d_capgain_figure.py (the AAAI figure of the capital-gain
pathway, recomputed by experiments/aaai27/recheck/paper_recompute.py into capgain-response_recomputed.png):
its pathway_forward (the part's seven units and four readouts, weights and biases from the stored genome),
its cross_check against the phenotype's StructureNetwork (asserts equal activations), its grid (N_GRID points
over the observed range of standardised capital gain), its reference (the five country-of-birth inputs of the
part at their raw 0) and its knot rule (second difference of the piecewise-linear curve). Here the numbers are
written to a file for thesis/ch7/exp1/make_figures.py (capgain_figure) instead of drawn.

Every transaction is READ ONLY (compute_runs.ReadOnlySession; PGOPTIONS default_transaction_read_only=on).
Run it twice: data/capgain.json is byte-identical.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E = REPO / "experiments"
for p in (str(REPO), str(E), str(E / "10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_dev")
os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()

OUT = HERE / "data" / "capgain.json"
LAYER = E / "10seed/plain_language/adult-paper.json"   # the part's title and the outcome's wording


def main():
    import torch
    torch.set_num_threads(1)
    import compute_runs as cr
    from aaai27 import run_d_capgain_figure as cg
    from aaai27.common import ADULT, load_model
    cr.db.init_db(os.environ["DATABASE_URL"])
    with cr.ReadOnlySession() as s:
        genome, config, X_scaled, y, split, _, ds = load_model(s, ADULT)
        X_raw, _ = ds.get_data()
        mean = np.array(split.scaler_params["mean"])
        scale = np.array(split.scaler_params["scale"])
    c = cg.CG_COL
    cg_raw = X_raw[:, c]
    z_lo, z_hi = X_scaled[:, c].min(), X_scaled[:, c].max()
    country_z = [(0.0 - mean[-k - 1]) / scale[-k - 1] for k in cg.COUNTRY_KEYS]
    grid = np.linspace(z_lo, z_hi, cg.N_GRID)
    _, contrib = cg.pathway_forward(genome, grid, country_z)
    sub = np.linspace(z_lo, z_hi, 25)
    ref = np.zeros((25, X_scaled.shape[1]))
    ref[:] = (0.0 - mean) / scale
    ref[:, c] = sub
    sub_acts, _ = cg.pathway_forward(genome, sub, country_z)
    cg.cross_check(genome, config, ref, sub, sub_acts)          # asserts (atol 1e-9)
    d2 = np.abs(np.diff(contrib, 2))
    knots = []
    for i in np.where(d2 > 1e-9)[0] + 1:
        if not knots or i - knots[-1] > 2:
            knots.append(int(i))
    layer = json.loads(LAYER.read_text())
    assert layer["genome_id"] == ADULT["genome_id"]
    part = next(p for p in layer["parts"] if p["name"] == "capital_gain_core")
    title = (layer.get("drafted") or {}).get("titles", {}).get(part["id"], part["name"])
    out = {"source": {"script": "experiments/aaai27/run_d_capgain_figure.py", "genome": ADULT["genome_id"],
                      "part": part["name"], "part_title": title, "outcome": layer["outcome"]["short"],
                      "n_rows": int(len(cg_raw)), "rows": "every row of the dataset (as the study script)"},
           "raw_grid": grid * scale[c] + mean[c], "z_grid": grid, "contribution": contrib,
           "knots_raw": [float(grid[i] * scale[c] + mean[c]) for i in knots],
           "knot_contribution": [float(contrib[i]) for i in knots],
           "share_zero_capital_gain": float((cg_raw == 0).mean()),
           "scaler": {"mean": float(mean[c]), "scale": float(scale[c])}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(out), indent=0, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO), "knots", out["knots_raw"])


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in list(x)]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 10)
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    main()
